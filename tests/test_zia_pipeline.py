"""End-to-end Zia pipeline tests, driven by a stub Anthropic client.

These exercise the real pipeline - classification, targeted fetch, narration,
parallel generation + action decisioning, escalation verification, guardrail,
persistence and metrics - without a network call. The stub returns whichever
structured type the caller asked for via `output_format`, which is exactly how
the SDK's messages.parse behaves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zomato_support.backends import Backends
from zomato_support.pipeline.action_decisioning import ActionProposal
from zomato_support.pipeline.function_classifier import FunctionPlan
from zomato_support.pipeline.zia import AgentReply, Zia


class _Usage:
    input_tokens = 500
    output_tokens = 120
    cache_read_input_tokens = 0


class _Parsed:
    def __init__(self, parsed):
        self.parsed_output = parsed
        self.usage = _Usage()
        self.stop_reason = "end_turn"
        self.content = []


class StubMessages:
    """Returns a canned object of whatever type the caller asked for."""

    def __init__(self, plan, reply, action):
        self.plan, self.reply, self.action = plan, reply, action
        self.calls = []

    def parse(self, **kwargs):
        fmt = kwargs["output_format"]
        self.calls.append((kwargs["model"], fmt.__name__))
        if fmt is FunctionPlan:
            return _Parsed(self.plan)
        if fmt is AgentReply:
            return _Parsed(self.reply)
        if fmt is ActionProposal:
            return _Parsed(self.action)
        raise AssertionError(f"unexpected output_format {fmt}")

    def create(self, **kwargs):
        raise AssertionError("pipeline should not use plain create()")


class StubClient:
    def __init__(self, plan, reply, action):
        self.messages = StubMessages(plan, reply, action)


def build(plan=None, reply=None, action=None):
    plan = plan or FunctionPlan(
        functions=["ORDER_STATUS", "DELIVERY_TRACKING"], order_id="ORD-89044",
        intent="track_order", reasoning="tracking question",
    )
    reply = reply or AgentReply(
        reply="Sunita is about 1.1 km away and should reach you in around 26 minutes.",
        escalate=False, escalation_reason_code=None, resolved=True,
    )
    action = action or ActionProposal(
        action="NONE", confidence=0.9, rationale="just a question"
    )
    backends = Backends.mock()
    return Zia(backends=backends, client=StubClient(plan, reply, action)), backends


@pytest.fixture
def zia_and_backends():
    return build()


# --- happy path -------------------------------------------------------------
def test_full_turn_runs_end_to_end(zia_and_backends):
    zia, _ = zia_and_backends
    session = zia.start_session("CUS-1001")
    result = zia.handle(session, "where is my order ORD-89044?")

    assert "Sunita" in result.reply
    assert result.intent == "track_order"
    assert result.functions_called == ["ORDER_STATUS", "DELIVERY_TRACKING"]
    assert result.contained
    assert result.timings["total_ms"] >= 0
    assert result.usage["input_tokens"] > 0


def test_model_tiering_is_applied(zia_and_backends):
    """Classification on the big model, the customer-facing reply on the fast one."""
    zia, _ = zia_and_backends
    session = zia.start_session("CUS-1001")
    zia.handle(session, "where is my order ORD-89044?")

    by_type = dict((fmt, model) for model, fmt in zia.client.messages.calls)
    assert by_type["FunctionPlan"] == "claude-opus-5"
    assert by_type["ActionProposal"] == "claude-opus-5"
    assert by_type["AgentReply"] == "claude-haiku-4-5"


def test_context_is_narrated_prose_not_json(zia_and_backends):
    zia, _ = zia_and_backends
    session = zia.start_session("CUS-1001")
    result = zia.handle(session, "where is my order ORD-89044?")
    assert "{" not in result.context_used
    assert "Sunita" in result.context_used
    assert "Menon" not in result.context_used   # surname withheld


def test_history_is_persisted(zia_and_backends):
    zia, backends = zia_and_backends
    session = zia.start_session("CUS-1001")
    zia.handle(session, "where is my order?")
    history = backends.crm.history(session)
    assert [m["role"] for m in history] == ["customer", "agent"]


def test_metrics_row_is_written(zia_and_backends):
    zia, _ = zia_and_backends
    from zomato_support import metrics

    session = zia.start_session("CUS-1001")
    zia.handle(session, "where is my order ORD-89044?")
    data = metrics.dashboard()
    assert data["response_time"]["turns"] == 1
    assert data["containment"]["pct"] == 100.0


# --- escalation verification in the live pipeline --------------------------
def test_unsupported_escalation_is_rejected_and_chat_stays_contained():
    """The model asks to escalate; the data does not back it up."""
    zia, _ = build(
        plan=FunctionPlan(functions=["DELIVERY_TRACKING"], order_id="ORD-89077",
                          intent="track_order", reasoning="late order"),
        reply=AgentReply(reply="Your order is running a little late, sorry about that.",
                         escalate=True, escalation_reason_code="DP_MOVEMENT_ISSUE",
                         resolved=False),
    )
    session = zia.start_session("CUS-1005")
    result = zia.handle(session, "my order is stuck, get me a human")

    assert result.escalation["requested"] is True
    assert result.escalation["upheld"] is False
    assert result.contained is True
    assert "normal" in result.escalation["evidence"]


def test_supported_escalation_is_upheld():
    zia, _ = build(
        plan=FunctionPlan(functions=["DELIVERY_TRACKING"], order_id="ORD-89044",
                          intent="delivery_partner_issue", reasoning="stuck"),
        reply=AgentReply(reply="Let me get someone to look at this.",
                         escalate=True, escalation_reason_code="DP_MOVEMENT_ISSUE",
                         resolved=False),
    )
    session = zia.start_session("CUS-1001")
    result = zia.handle(session, "my rider has not moved in ages")

    assert result.escalation["upheld"] is True
    assert result.escalation["queue"] == "delivery-investigations"
    assert result.contained is False
    assert "case_id" in result.escalation


def test_safety_escalation_bypasses_verification():
    zia, _ = build(
        plan=FunctionPlan(functions=[], intent="food_quality", reasoning="illness"),
        reply=AgentReply(reply="I'm sorry, let me get a specialist onto this.",
                         escalate=True, escalation_reason_code="FOOD_SAFETY_ILLNESS",
                         resolved=False),
    )
    session = zia.start_session("CUS-1001")
    result = zia.handle(session, "I've been vomiting since eating this")

    assert result.escalation["upheld"] is True
    assert result.escalation["auto_upheld"] is True
    assert result.contained is False


def test_invented_reason_code_does_not_escalate():
    zia, _ = build(
        reply=AgentReply(reply="Let me pass this on.", escalate=True,
                         escalation_reason_code="CUSTOMER_IS_CROSS", resolved=False),
    )
    session = zia.start_session("CUS-1001")
    result = zia.handle(session, "this is annoying")
    assert result.escalation["upheld"] is False
    assert result.contained is True


# --- action proposals -------------------------------------------------------
def test_eligible_action_is_offered():
    zia, _ = build(
        plan=FunctionPlan(functions=["ORDER_STATUS"], order_id="ORD-89011",
                          intent="change_delivery", reasoning="instructions"),
        reply=AgentReply(reply="I can add that for the rider.", escalate=False,
                         escalation_reason_code=None, resolved=True),
        action=ActionProposal(action="ADD_DELIVERY_INSTRUCTIONS", order_id="ORD-89011",
                              parameter="Leave at the door", confidence=0.95,
                              rationale="explicit request"),
    )
    session = zia.start_session("CUS-1005")
    result = zia.handle(session, "please leave it at the door")

    assert result.action is not None
    assert result.action["action"] == "ADD_DELIVERY_INSTRUCTIONS"
    assert result.action["eligible"] is True


def test_ineligible_action_is_not_offered():
    """Classifier says cancel; the order is already with the rider."""
    zia, _ = build(
        plan=FunctionPlan(functions=["ORDER_STATUS"], order_id="ORD-89044",
                          intent="cancel_order", reasoning="cancel"),
        reply=AgentReply(reply="Let me check that for you.", escalate=False,
                         escalation_reason_code=None, resolved=False),
        action=ActionProposal(action="CANCEL_ORDER", order_id="ORD-89044",
                              confidence=0.95, rationale="wants to cancel"),
    )
    session = zia.start_session("CUS-1001")
    result = zia.handle(session, "cancel it")

    assert result.action is None          # never offered to the customer
    assert result.reply                   # but the customer still gets a reply


def test_low_confidence_proposal_is_dropped():
    zia, _ = build(
        action=ActionProposal(action="CANCEL_ORDER", order_id="ORD-89011",
                              confidence=0.3, rationale="maybe"),
    )
    session = zia.start_session("CUS-1001")
    assert zia.handle(session, "hmm").action is None


# --- confirmed action execution --------------------------------------------
def test_confirmed_action_executes_and_persists():
    zia, backends = build()
    session = zia.start_session("CUS-1005")
    result = zia.execute_action(session, "ADD_DELIVERY_INSTRUCTIONS",
                                "ORD-89011", "Leave at the door")
    assert result["ok"]
    assert backends.orders.get("ORD-89011")["delivery_instructions"] == "Leave at the door"


def test_execute_action_reverifies_server_side():
    """The client could send anything; eligibility is checked again here."""
    zia, _ = build()
    session = zia.start_session("CUS-1001")
    result = zia.execute_action(session, "CANCEL_ORDER", "ORD-89044")
    assert not result["ok"]
    assert "rider" in result["message"].lower()


def test_cancel_refunds_the_order():
    zia, backends = build()
    session = zia.start_session("CUS-1005")
    result = zia.execute_action(session, "CANCEL_ORDER", "ORD-89011")
    assert result["ok"]
    assert backends.orders.get("ORD-89011")["status"] == "cancelled"
    assert backends.orders.refunded_total("ORD-89011") == 396


def test_guardrail_blocks_a_leaky_reply():
    zia, _ = build(
        reply=AgentReply(
            reply="Our forensic check flagged your photo as manipulated, so no refund.",
            escalate=False, escalation_reason_code=None, resolved=True),
    )
    session = zia.start_session("CUS-1001")
    result = zia.handle(session, "here is my photo")

    assert result.guardrail["blocked"] is True
    assert "forensic" not in result.reply.lower()


def test_unknown_session_raises():
    zia, _ = build()
    with pytest.raises(KeyError):
        zia.handle("SES-NOPE", "hello")
