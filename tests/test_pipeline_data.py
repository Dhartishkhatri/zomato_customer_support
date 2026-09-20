"""Tests for targeted fetching, JSON->prose narration, metrics and the web API.

All offline: none of these make a model call.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zomato_support import metrics
from zomato_support.backends import Backends
from zomato_support.llm import Usage, cost_usd
from zomato_support.pipeline.data_functions import fetch_and_narrate
from zomato_support.pipeline.function_classifier import FUNCTIONS


@pytest.fixture
def backends() -> Backends:
    return Backends.mock()


def narrate(backends, functions, order_id="ORD-89044", customer_id="CUS-1001", query=""):
    return fetch_and_narrate(backends, functions, order_id, customer_id, query)


# --- only the requested data is fetched ------------------------------------
def test_only_requested_lookups_appear(backends):
    """The whole point of classification: an ETA question gets no itemised bill."""
    text, raw = narrate(backends, ["DELIVERY_TRACKING"])
    assert "Sunita" in text
    assert "Chicken Kebab" not in text      # ORDER_ITEMS was not requested
    assert "tracking" in raw and "order" in raw


def test_items_lookup_returns_the_bill(backends):
    text, _ = narrate(backends, ["ORDER_ITEMS"])
    assert "Chicken Kebab" in text
    assert "Sunita" not in text


def test_empty_function_list_fetches_nothing(backends):
    text, raw = narrate(backends, [])
    assert text == ""
    assert "tracking" not in raw


def test_narration_is_prose_not_json(backends):
    text, _ = narrate(backends, ["ORDER_STATUS", "DELIVERY_TRACKING"])
    assert "{" not in text and "}" not in text
    assert text.endswith(".")


# --- the narration layer controls what the model can say -------------------
def test_courier_surname_and_phone_never_enter_the_prompt(backends):
    text, _ = narrate(backends, ["DELIVERY_TRACKING"])
    assert "Sunita" in text
    assert "Menon" not in text
    assert "99000" not in text


def test_lateness_is_precomputed_into_words(backends):
    """The model should never do arithmetic on timestamps."""
    text, _ = narrate(backends, ["DELIVERY_TRACKING"], order_id="ORD-89044")
    assert "not moved for 23 minutes" in text


def test_refund_window_is_precomputed(backends):
    late, _ = narrate(backends, ["ORDER_STATUS"], order_id="ORD-88777")
    assert "past the" in late and "support window" in late

    fresh, _ = narrate(backends, ["ORDER_STATUS"], order_id="ORD-88213")
    assert "still inside" in fresh


def test_already_refunded_order_says_so(backends):
    text, raw = narrate(backends, ["ORDER_PAYMENT"], order_id="ORD-88990",
                        customer_id="CUS-1002")
    assert "no further refund" in text
    assert raw["refunded_total"] == 410


def test_internal_risk_fields_never_appear(backends):
    text, _ = narrate(backends, ["CUSTOMER_PROFILE"], order_id="ORD-88601",
                      customer_id="CUS-1003")
    assert "trust" not in text.lower()
    assert "Trust & Safety" not in text
    assert "0.28" not in text


# --- ownership and missing data --------------------------------------------
def test_another_customers_order_is_not_narrated(backends):
    text, raw = narrate(backends, ["ORDER_STATUS"], order_id="ORD-89011",
                        customer_id="CUS-1001")
    assert "raw order" not in text
    assert "order" not in raw
    assert "not identified" in text


def test_unknown_order_is_reported(backends):
    text, _ = narrate(backends, ["ORDER_STATUS"], order_id="ORD-00000")
    assert "No order with id" in text


def test_missing_order_prompts_a_question(backends):
    text, _ = narrate(backends, ["ORDER_STATUS"], order_id=None)
    assert "Ask them which order" in text


def test_policy_lookup_carries_citations(backends):
    text, _ = narrate(backends, ["POLICY"], query="refund window")
    assert "KB-REFUND-001" in text


def test_no_live_tracking_on_a_delivered_order(backends):
    text, _ = narrate(backends, ["DELIVERY_TRACKING"], order_id="ORD-88213")
    assert "no live tracking" in text.lower()


def test_every_catalogue_entry_has_a_description():
    assert all(FUNCTIONS.values())


# --- cost model and dashboard ----------------------------------------------
def test_tiering_makes_the_chat_model_cheaper():
    """The reason for tiering, in one assertion."""
    assert cost_usd("claude-haiku-4-5", 1000, 500) < cost_usd("claude-opus-5", 1000, 500)


def test_usage_accumulates_across_calls():
    from zomato_support.llm import CallRecord

    usage = Usage()
    usage.add(CallRecord("classify", "claude-opus-5", 1000, 100, 900, 0.0075))
    usage.add(CallRecord("generate", "claude-haiku-4-5", 800, 200, 400, 0.0018))
    assert usage.input_tokens == 1800
    assert round(usage.total_cost, 4) == 0.0093
    assert len(usage.to_dict()["calls"]) == 2


def test_dashboard_reports_containment_and_rejections(backends):
    usage = Usage()
    metrics.record_turn(
        session_id=None, latency_ms=4200, classify_ms=900, generate_ms=2800,
        action_ms=0, usage=usage, escalated=False,
        escalation_reason="DP_MOVEMENT_ISSUE", escalation_upheld=False,
        functions=["DELIVERY_TRACKING"], action_proposed="NONE",
        guardrail_blocked=False,
    )
    data = metrics.dashboard()
    assert data["response_time"]["turns"] == 1
    assert data["response_time"]["within_target_pct"] == 100.0
    assert data["containment"]["escalations_rejected_by_policy"] == 1
    assert data["escalation_reasons"]["DP_MOVEMENT_ISSUE"] == 1


def test_dashboard_is_safe_on_an_empty_database(backends):
    data = metrics.dashboard()
    assert data["csat"]["average"] is None
    assert data["containment"]["pct"] == 0.0
    assert data["cost"]["total_usd"] == 0.0


# --- web API (no model calls) ----------------------------------------------
def test_web_api_endpoints(backends):
    from fastapi.testclient import TestClient

    import webapp

    client = TestClient(webapp.app)
    assert client.get("/").status_code == 200
    assert len(client.get("/api/customers").json()) == 5

    session = client.post("/api/session", json={"customer_id": "CUS-1005"}).json()
    assert session["session_id"].startswith("SES-")

    # Eligible action succeeds...
    ok = client.post("/api/action", json={
        "session_id": session["session_id"], "action": "ADD_DELIVERY_INSTRUCTIONS",
        "order_id": "ORD-89011", "parameter": "Leave at the door",
    }).json()
    assert ok["ok"]

    # ...and the server re-verifies rather than trusting the client.
    blocked = client.post("/api/action", json={
        "session_id": session["session_id"], "action": "CANCEL_ORDER",
        "order_id": "ORD-88213", "parameter": None,
    }).json()
    assert not blocked["ok"]


def test_web_api_rejects_unknown_customer(backends):
    from fastapi.testclient import TestClient

    import webapp

    client = TestClient(webapp.app)
    assert client.post("/api/session", json={"customer_id": "CUS-9999"}).status_code == 404


def test_web_api_validates_rating_range(backends):
    from fastapi.testclient import TestClient

    import webapp

    client = TestClient(webapp.app)
    session = client.post("/api/session", json={"customer_id": "CUS-1001"}).json()
    bad = client.post("/api/rating", json={"session_id": session["session_id"], "rating": 9})
    assert bad.status_code == 400
