"""Pre-send guardrail tests. No API key required."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zomato_support import guardrails
from zomato_support.backends import Backends
from zomato_support.schemas import ConversationContext


@pytest.fixture
def backends() -> Backends:
    return Backends.mock()


@pytest.fixture
def ctx(backends) -> ConversationContext:
    context = ConversationContext(customer_id="CUS-1001", ticket_id="TKT-1")
    context.orders_seen["ORD-88213"] = backends.orders.get("ORD-88213")
    return context


GROUNDING = '{"order_id": "ORD-88213", "total": 487, "amount": 119, "refund_id": "RFD-1"}'


def check(reply, ctx, backends, grounding=GROUNDING):
    return guardrails.check_reply(reply, ctx, backends, grounding)


def codes(report):
    return {v.code for v in report.violations}


# --- clean replies ----------------------------------------------------------
def test_clean_reply_passes(ctx, backends):
    reply = (
        "Sorry about the missing cold coffee. I've sent 119 back to your UPI account, "
        "and it should land in 3-5 business days."
    )
    ctx.refunds_issued.append({"refund_id": "RFD-1", "amount": 119})
    assert not check(reply, ctx, backends).blocked


def test_neutral_escalation_reply_passes(ctx, backends):
    ctx.escalated = True
    reply = (
        "Thanks for flagging this. I've passed it to a specialist who will review "
        "your case and get back to you within 24 hours."
    )
    assert not check(reply, ctx, backends).blocked


# --- forensic vocabulary leaks ---------------------------------------------
@pytest.mark.parametrize(
    "reply",
    [
        "Our forensic check found your photo was edited.",
        "The image you sent appears to be AI generated.",
        "Your photo was flagged as manipulated by our system.",
        "This has been referred to Trust & Safety for fraud review.",
        "The EXIF metadata on your image is missing.",
    ],
)
def test_internal_vocabulary_is_blocked(reply, ctx, backends):
    report = check(reply, ctx, backends)
    assert report.blocked
    assert "internal_vocabulary" in codes(report)


# --- privacy ----------------------------------------------------------------
def test_naming_another_customer_is_blocked(ctx, backends):
    report = check("Rohit Malhotra had the same issue last week.", ctx, backends)
    assert report.blocked
    assert "third_party_pii" in codes(report)


def test_leaking_another_customers_email_is_blocked(ctx, backends):
    report = check("You can contact them at vikram.sethi@example.com.", ctx, backends)
    assert report.blocked


def test_delivery_partner_surname_is_blocked(ctx, backends):
    # Only in-flight orders carry live tracking, so use one: ORD-89044 belongs
    # to this customer and its partner is Sunita Menon.
    ctx.orders_seen["ORD-89044"] = backends.orders.get("ORD-89044")
    report = check("Sunita Menon is bringing your order.", ctx, backends)
    assert report.blocked
    assert "partner_pii" in codes(report)


def test_delivery_partner_first_name_is_allowed(ctx, backends):
    ctx.orders_seen["ORD-89044"] = backends.orders.get("ORD-89044")
    report = check("Sunita is on her way with your order.", ctx, backends)
    assert "partner_pii" not in codes(report)


# --- money grounding --------------------------------------------------------
def test_invented_amount_is_blocked(ctx, backends):
    report = check("I've arranged a refund of 2500 for you.", ctx, backends)
    assert report.blocked
    assert "ungrounded_amount" in codes(report)


def test_grounded_amount_passes(ctx, backends):
    ctx.refunds_issued.append({"refund_id": "RFD-1", "amount": 119})
    report = check("I've refunded 119 to your UPI account.", ctx, backends)
    assert not report.blocked


def test_bare_amount_without_currency_marker_is_caught(ctx, backends):
    ctx.refunds_issued.append({"refund_id": "RFD-1", "amount": 119})
    report = check("I've arranged a refund of 2500 for you.", ctx, backends)
    assert "ungrounded_amount" in codes(report)


def test_durations_are_not_mistaken_for_money(ctx, backends):
    ctx.refunds_issued.append({"refund_id": "RFD-1", "amount": 119})
    report = check(
        "Your refund of 119 will reach you within 24 hours, and card refunds take "
        "5-7 business days.",
        ctx,
        backends,
    )
    assert "ungrounded_amount" not in codes(report)


def test_refund_claimed_without_a_refund_is_blocked(ctx, backends):
    report = check("Your refund has been processed and will arrive soon.", ctx, backends)
    assert report.blocked
    assert "unbacked_refund_claim" in codes(report)


def test_pending_approval_cannot_be_presented_as_settled(ctx, backends):
    ctx.pending_approvals.append({"action": "issue_refund", "input": {}, "reason": "x"})
    report = check("Good news - your refund has been credited.", ctx, backends)
    assert report.blocked


# --- promises ---------------------------------------------------------------
def test_banned_promise_language_is_blocked(ctx, backends):
    report = check("We guaranteed this will never happen again.", ctx, backends)
    assert report.blocked
    assert "banned_phrase" in codes(report)


def test_agent_may_not_identify_itself_as_ai(ctx, backends):
    report = check("I am an AI assistant and cannot help with that.", ctx, backends)
    assert report.blocked


# --- advisory ---------------------------------------------------------------
def test_long_reply_is_advisory_not_blocking(ctx, backends):
    report = check("Sorry about that. " * 200, ctx, backends)
    assert "reply_too_long" in codes(report)
    assert not report.blocked


def test_silent_escalation_is_advisory(ctx, backends):
    ctx.escalated = True
    report = check("Okay, noted.", ctx, backends)
    assert "escalation_not_communicated" in codes(report)
    assert not report.blocked


def test_feedback_lists_every_violation(ctx, backends):
    report = check("Our forensic analysis shows a refund of 9999 was fraudulent.", ctx, backends)
    feedback = report.feedback()
    assert "forensic" in feedback or "internal_vocabulary" in feedback
    assert "ungrounded_amount" in feedback


def test_report_serialises(ctx, backends):
    payload = check("All sorted, thanks.", ctx, backends).to_dict()
    assert set(payload) == {"blocked", "checked_chars", "violations"}
