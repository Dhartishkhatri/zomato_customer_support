"""Action eligibility tests - step 2 of the action pipeline. No API key needed.

These cover the half of the action pipeline that does NOT use a model: given a
proposed action and real order state, is it actually permitted? This is where
an eager LLM's promises get checked against reality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zomato_support.backends import Backends
from zomato_support.pipeline.action_decisioning import verify_action


@pytest.fixture
def backends() -> Backends:
    return Backends.mock()


def verify(backends, action, order_id, customer_id, parameter=None):
    return verify_action(backends, action, order_id, parameter, customer_id)


# --- cancellation -----------------------------------------------------------
def test_recent_order_can_be_cancelled(backends):
    # ORD-89011 was placed 4 minutes ago, inside a 15-minute window.
    d = verify(backends, "CANCEL_ORDER", "ORD-89011", "CUS-1005")
    assert d.eligible
    assert d.should_offer
    assert "396" in d.prompt  # the refund amount is quoted in the popup


def test_out_for_delivery_order_cannot_be_cancelled(backends):
    """The failure an eager model makes: promising to cancel a live delivery."""
    d = verify(backends, "CANCEL_ORDER", "ORD-89044", "CUS-1001")
    assert not d.eligible
    assert "rider" in d.reject_reason.lower()


def test_delivered_order_cannot_be_cancelled(backends):
    d = verify(backends, "CANCEL_ORDER", "ORD-88213", "CUS-1001")
    assert not d.eligible
    assert "delivered" in d.reject_reason


def test_cancellation_window_expiry_is_enforced(backends):
    # ORD-88912 is out for delivery, 38 minutes after being placed.
    d = verify(backends, "CANCEL_ORDER", "ORD-88912", "CUS-1004")
    assert not d.eligible


# --- delivery instructions and address -------------------------------------
def test_instructions_can_be_added_to_a_live_order(backends):
    d = verify(backends, "ADD_DELIVERY_INSTRUCTIONS", "ORD-89011", "CUS-1005",
               parameter="Leave at the door")
    assert d.eligible
    assert "Leave at the door" in d.prompt


def test_instructions_need_actual_text(backends):
    d = verify(backends, "ADD_DELIVERY_INSTRUCTIONS", "ORD-89011", "CUS-1005")
    assert not d.eligible


def test_instructions_rejected_on_delivered_order(backends):
    d = verify(backends, "ADD_DELIVERY_INSTRUCTIONS", "ORD-88213", "CUS-1001",
               parameter="Ring the bell")
    assert not d.eligible


def test_address_cannot_change_once_the_rider_is_en_route(backends):
    d = verify(backends, "CHANGE_DELIVERY_ADDRESS", "ORD-89044", "CUS-1001",
               parameter="12 MG Road")
    assert not d.eligible
    assert "en route" in d.reject_reason


def test_address_can_change_while_preparing(backends):
    d = verify(backends, "CHANGE_DELIVERY_ADDRESS", "ORD-89011", "CUS-1005",
               parameter="12 MG Road")
    assert d.eligible


# --- contacting the rider ---------------------------------------------------
def test_rider_can_be_contacted_when_out_for_delivery(backends):
    d = verify(backends, "CONTACT_DELIVERY_PARTNER", "ORD-89044", "CUS-1001")
    assert d.eligible
    assert "Sunita" in d.prompt          # first name only
    assert "Menon" not in d.prompt       # surname withheld


def test_no_rider_to_contact_before_dispatch(backends):
    d = verify(backends, "CONTACT_DELIVERY_PARTNER", "ORD-89011", "CUS-1005")
    assert not d.eligible


# --- refunds ----------------------------------------------------------------
def test_refund_request_needs_a_delivered_order(backends):
    assert not verify(backends, "REQUEST_REFUND", "ORD-89011", "CUS-1005").eligible
    assert verify(backends, "REQUEST_REFUND", "ORD-88213", "CUS-1001").eligible


def test_already_refunded_order_is_rejected(backends):
    d = verify(backends, "REQUEST_REFUND", "ORD-88990", "CUS-1002")
    assert not d.eligible
    assert "already been refunded" in d.reject_reason


# --- ownership and unknown input -------------------------------------------
def test_action_on_another_customers_order_is_rejected(backends):
    d = verify(backends, "CANCEL_ORDER", "ORD-89011", "CUS-1001")  # belongs to 1005
    assert not d.eligible
    assert "different customer" in d.reject_reason


def test_unknown_order_is_rejected(backends):
    assert not verify(backends, "CANCEL_ORDER", "ORD-00000", "CUS-1001").eligible


def test_missing_order_id_is_rejected(backends):
    assert not verify(backends, "CANCEL_ORDER", None, "CUS-1001").eligible


def test_unknown_action_is_rejected(backends):
    d = verify(backends, "TELEPORT_ORDER", "ORD-89011", "CUS-1005")
    assert not d.eligible


def test_decision_serialises(backends):
    payload = verify(backends, "CANCEL_ORDER", "ORD-89011", "CUS-1005").to_dict()
    assert payload["eligible"] is True
    assert payload["action"] == "CANCEL_ORDER"
