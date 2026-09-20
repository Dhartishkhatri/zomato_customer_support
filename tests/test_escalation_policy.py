"""Escalation policy layer tests - the containment stabiliser. No API key needed.

The property these protect: whether an escalation is upheld must depend on
ORDER DATA, not on how the model phrased its request. Same reason code,
different data, opposite outcome.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zomato_support.backends import Backends
from zomato_support.pipeline.escalation_policy import (
    ALWAYS_UPHELD,
    VERIFIED_CODES,
    verify_escalation,
)


@pytest.fixture
def backends() -> Backends:
    return Backends.mock()


def check(backends, code, order_id=None, customer_id=""):
    order = backends.orders.get(order_id) if order_id else None
    return verify_escalation(backends, code, order, customer_id)


# --- the core property ------------------------------------------------------
def test_same_code_opposite_outcomes_from_data_alone(backends):
    """ORD-89044's rider is stuck; ORD-89077's is moving normally."""
    stuck = check(backends, "DP_MOVEMENT_ISSUE", "ORD-89044", "CUS-1001")
    moving = check(backends, "DP_MOVEMENT_ISSUE", "ORD-89077", "CUS-1005")

    assert stuck.upheld and not moving.upheld
    assert "23 minutes" in stuck.evidence
    assert "normal" in moving.evidence


def test_rejected_escalation_keeps_the_chat_contained(backends):
    v = check(backends, "DP_MOVEMENT_ISSUE", "ORD-89077", "CUS-1005")
    assert v.requested and not v.upheld
    assert v.queue is None


# --- safety codes bypass verification --------------------------------------
@pytest.mark.parametrize("code", sorted(ALWAYS_UPHELD))
def test_safety_codes_are_always_upheld(code, backends):
    """These must never be suppressed to protect a containment metric."""
    v = check(backends, code)
    assert v.upheld
    assert v.auto_upheld
    assert v.queue == ALWAYS_UPHELD[code]


def test_illness_is_upheld_even_with_no_order(backends):
    assert check(backends, "FOOD_SAFETY_ILLNESS").upheld


# --- verified codes ---------------------------------------------------------
def test_lateness_needs_real_lateness(backends):
    # ORD-89044: ETA 26 vs 30 promised - not materially late.
    assert not check(backends, "ORDER_SIGNIFICANTLY_LATE", "ORD-89044", "CUS-1001").upheld


def test_outside_window_is_verified_against_delivery_time(backends):
    assert check(backends, "REFUND_OUTSIDE_WINDOW", "ORD-88777", "CUS-1001").upheld
    assert not check(backends, "REFUND_OUTSIDE_WINDOW", "ORD-88213", "CUS-1001").upheld


def test_repeat_refund_pattern_checks_the_customer(backends):
    assert check(backends, "REPEAT_REFUND_PATTERN", "ORD-88601", "CUS-1003").upheld
    assert not check(backends, "REPEAT_REFUND_PATTERN", "ORD-88213", "CUS-1001").upheld


def test_high_value_refund_checks_the_total(backends):
    assert check(backends, "HIGH_VALUE_REFUND", "ORD-88455", "CUS-1002").upheld
    assert not check(backends, "HIGH_VALUE_REFUND", "ORD-88213", "CUS-1001").upheld


def test_not_delivered_requires_delivered_status(backends):
    assert check(backends, "ORDER_NOT_DELIVERED", "ORD-88213", "CUS-1001").upheld
    assert not check(backends, "ORDER_NOT_DELIVERED", "ORD-89044", "CUS-1001").upheld


def test_payment_not_settled_requires_a_pending_refund(backends):
    assert not check(backends, "PAYMENT_NOT_SETTLED", "ORD-88213", "CUS-1001").upheld


# --- invalid input ----------------------------------------------------------
def test_invented_reason_code_is_rejected(backends):
    """A hallucinated code is not a reason, so the chat stays contained."""
    v = check(backends, "CUSTOMER_SEEMS_ANNOYED", "ORD-88213", "CUS-1001")
    assert v.requested and not v.upheld
    assert "not a recognised" in v.evidence


def test_no_code_means_no_escalation(backends):
    v = check(backends, None)
    assert not v.requested and not v.upheld


def test_codes_are_case_insensitive(backends):
    assert check(backends, "food_safety_illness").upheld


def test_verified_code_without_an_order_is_not_upheld(backends):
    assert not check(backends, "DP_MOVEMENT_ISSUE").upheld


def test_every_verified_code_has_a_queue():
    assert all(VERIFIED_CODES.values())


def test_verdict_serialises(backends):
    payload = check(backends, "DP_MOVEMENT_ISSUE", "ORD-89044", "CUS-1001").to_dict()
    assert payload["upheld"] is True
    assert payload["queue"] == "delivery-investigations"
