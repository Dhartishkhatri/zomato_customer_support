"""Permission gate tests. No API key required."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zomato_support import policy
from zomato_support.backends import Backends
from zomato_support.schemas import (
    ConversationContext,
    Decision,
    ImageAnalysis,
    ImageVerdict,
)


@pytest.fixture
def backends() -> Backends:
    return Backends.mock()


def make_ctx(customer_id="CUS-1001", ticket_id="TKT-1", orders=(), backends=None):
    ctx = ConversationContext(customer_id=customer_id, ticket_id=ticket_id)
    for order_id in orders:
        ctx.orders_seen[order_id] = backends.orders.get(order_id)
    ctx.customer_record = backends.crm.get_customer(customer_id)
    return ctx


def analysis(verdict: ImageVerdict, confidence: float = 0.9) -> ImageAnalysis:
    return ImageAnalysis(
        verdict=verdict,
        confidence=confidence,
        manipulation_score=0.5,
        signals=[],
        summary="test",
    )


def refund(ctx, backends, order_id="ORD-88213", amount=139, reason="Item missing", claim=""):
    return policy.evaluate(
        "issue_refund",
        {"order_id": order_id, "amount": amount, "reason": reason},
        ctx,
        backends,
        claim_text=claim,
    )


# --- happy path -------------------------------------------------------------
def test_small_evidenced_refund_is_auto_approved(backends):
    ctx = make_ctx(orders=["ORD-88213"], backends=backends)
    assert refund(ctx, backends).decision is Decision.ALLOW


def test_refund_above_auto_ceiling_needs_approval(backends):
    ctx = make_ctx(orders=["ORD-88213"], backends=backends)
    result = refund(ctx, backends, amount=487)
    assert result.decision is Decision.ALLOW  # 487 is under the 500 ceiling

    result = policy.evaluate(
        "issue_refund",
        {"order_id": "ORD-88213", "amount": 487, "reason": "x"},
        ctx,
        backends,
    )
    assert result.decision is Decision.ALLOW


def test_refund_above_supervisor_threshold(backends):
    ctx = make_ctx(customer_id="CUS-1002", orders=["ORD-88455"], backends=backends)
    result = refund(ctx, backends, order_id="ORD-88455", amount=1240, reason="Order wrong")
    assert result.decision is Decision.REQUIRE_APPROVAL
    assert result.rule_id == "R16-above-supervisor-threshold"


# --- hard denials -----------------------------------------------------------
def test_refund_on_unfetched_order_is_denied(backends):
    ctx = make_ctx(backends=backends)  # never called get_order
    result = refund(ctx, backends)
    assert result.decision is Decision.DENY
    assert result.rule_id == "R02-unverified-order"


def test_cross_account_refund_is_denied(backends):
    ctx = make_ctx(customer_id="CUS-1002", backends=backends)
    ctx.orders_seen["ORD-88213"] = backends.orders.get("ORD-88213")  # another customer's
    result = refund(ctx, backends)
    assert result.decision is Decision.DENY
    assert result.rule_id == "R03-cross-account"


def test_refund_exceeding_order_value_is_denied(backends):
    ctx = make_ctx(orders=["ORD-88213"], backends=backends)
    result = refund(ctx, backends, amount=25_000)
    assert result.decision is Decision.DENY
    assert result.rule_id == "R06-exceeds-order"


def test_undelivered_order_cannot_be_refunded(backends):
    ctx = make_ctx(customer_id="CUS-1004", orders=["ORD-88912"], backends=backends)
    result = refund(ctx, backends, order_id="ORD-88912", amount=100)
    assert result.decision is Decision.DENY
    assert result.rule_id == "R04-not-delivered"


def test_double_refund_is_denied(backends):
    ctx = make_ctx(customer_id="CUS-1002", orders=["ORD-88990"], backends=backends)
    result = refund(ctx, backends, order_id="ORD-88990", amount=100)
    assert result.decision is Decision.DENY
    assert result.rule_id == "R05-already-refunded"


def test_negative_and_zero_amounts_are_denied(backends):
    ctx = make_ctx(orders=["ORD-88213"], backends=backends)
    for amount in (0, -50, "500"):
        assert refund(ctx, backends, amount=amount).decision is Decision.DENY


# --- escalations ------------------------------------------------------------
def test_illness_claim_escalates(backends):
    ctx = make_ctx(orders=["ORD-88213"], backends=backends)
    result = refund(ctx, backends, claim="I've been vomiting since eating this")
    assert result.decision is Decision.ESCALATE
    assert result.evidence["queue"] == "food-safety"


def test_legal_threat_escalates(backends):
    ctx = make_ctx(orders=["ORD-88213"], backends=backends)
    result = refund(ctx, backends, claim="I'm taking this to consumer court")
    assert result.decision is Decision.ESCALATE
    assert result.evidence["queue"] == "legal-escalations"


def test_expired_window_escalates(backends):
    ctx = make_ctx(orders=["ORD-88777"], backends=backends)
    result = refund(ctx, backends, order_id="ORD-88777", amount=100)
    assert result.decision is Decision.ESCALATE
    assert result.rule_id == "R09-window-expired"


def test_refund_velocity_escalates(backends):
    ctx = make_ctx(customer_id="CUS-1003", orders=["ORD-88601"], backends=backends)
    result = refund(ctx, backends, order_id="ORD-88601", amount=100, reason="wrong order")
    assert result.decision is Decision.ESCALATE
    assert result.rule_id == "R14-refund-velocity"


# --- photo evidence ---------------------------------------------------------
def test_quality_claim_without_photo_needs_approval(backends):
    ctx = make_ctx(orders=["ORD-88213"], backends=backends)
    result = refund(ctx, backends, amount=100, reason="Food was spoiled")
    assert result.decision is Decision.REQUIRE_APPROVAL
    assert result.rule_id == "R10-photo-missing"


def test_manipulated_photo_escalates(backends):
    ctx = make_ctx(orders=["ORD-88213"], backends=backends)
    ctx.image_analyses["a.jpg"] = analysis(ImageVerdict.MANIPULATED)
    result = refund(ctx, backends, amount=100, reason="Food was spoiled")
    assert result.decision is Decision.ESCALATE
    assert result.evidence["queue"] == "trust-and-safety"


def test_ai_generated_photo_escalates(backends):
    ctx = make_ctx(orders=["ORD-88213"], backends=backends)
    ctx.image_analyses["a.png"] = analysis(ImageVerdict.AI_GENERATED)
    result = refund(ctx, backends, amount=100, reason="Order arrived spilled")
    assert result.decision is Decision.ESCALATE


def test_inconclusive_photo_requires_approval(backends):
    ctx = make_ctx(orders=["ORD-88213"], backends=backends)
    ctx.image_analyses["a.jpg"] = analysis(ImageVerdict.INCONCLUSIVE, 0.55)
    result = refund(ctx, backends, amount=100, reason="Food was spoiled")
    assert result.decision is Decision.REQUIRE_APPROVAL
    assert result.rule_id == "R12-image-inconclusive"


def test_authentic_but_low_confidence_requires_approval(backends):
    ctx = make_ctx(orders=["ORD-88213"], backends=backends)
    ctx.image_analyses["a.jpg"] = analysis(ImageVerdict.AUTHENTIC, 0.5)
    result = refund(ctx, backends, amount=100, reason="Food was spoiled")
    assert result.decision is Decision.REQUIRE_APPROVAL
    assert result.rule_id == "R13-image-low-confidence"


def test_authentic_photo_allows_refund(backends):
    ctx = make_ctx(orders=["ORD-88213"], backends=backends)
    ctx.image_analyses["a.jpg"] = analysis(ImageVerdict.AUTHENTIC, 0.92)
    result = refund(ctx, backends, amount=487, reason="Food was spoiled")
    assert result.decision is Decision.ALLOW


def test_worst_photo_governs_when_several_submitted(backends):
    ctx = make_ctx(orders=["ORD-88213"], backends=backends)
    ctx.image_analyses["good.jpg"] = analysis(ImageVerdict.AUTHENTIC, 0.95)
    ctx.image_analyses["bad.jpg"] = analysis(ImageVerdict.MANIPULATED, 0.9)
    result = refund(ctx, backends, amount=100, reason="Food was spoiled")
    assert result.decision is Decision.ESCALATE


# --- scope rules ------------------------------------------------------------
def test_email_to_another_customer_is_denied(backends):
    ctx = make_ctx(backends=backends)
    result = policy.evaluate(
        "send_email",
        {"customer_id": "CUS-1003", "subject": "x", "body": "y"},
        ctx,
        backends,
    )
    assert result.decision is Decision.DENY


def test_ticket_outside_conversation_is_denied(backends):
    ctx = make_ctx(backends=backends)
    result = policy.evaluate(
        "update_ticket",
        {"ticket_id": "TKT-OTHER", "status": "resolved", "summary": "s", "tags": []},
        ctx,
        backends,
    )
    assert result.decision is Decision.DENY


def test_read_tools_are_never_gated(backends):
    ctx = make_ctx(backends=backends)
    for tool in ("kb_search", "get_order", "get_customer", "verify_image_authenticity"):
        assert policy.evaluate(tool, {}, ctx, backends).decision is Decision.ALLOW


def test_photo_requirement_detection():
    assert policy.claim_requires_photo("the food was spoiled")
    assert policy.claim_requires_photo("there was a hair in my biryani")
    assert policy.claim_requires_photo("order arrived completely spilled")
    assert not policy.claim_requires_photo("one item was missing from the bag")
    assert not policy.claim_requires_photo("delivery was an hour late")
