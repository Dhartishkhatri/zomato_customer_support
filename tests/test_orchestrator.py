"""End-to-end agent loop tests driven by a scripted stub client.

These exercise the real orchestrator, permission gate, tool executor and
guardrail layer without touching the network. The stub mimics the shape the
Anthropic SDK returns (content blocks with `.type`, `.usage`, `.stop_reason`),
so the loop under test is the same code that runs against the live API.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zomato_support import SupportAgent
from zomato_support.approvals import AutoApprover, QueueApprover
from zomato_support.backends import Backends


# --- stub SDK objects -------------------------------------------------------
class Block:
    def __init__(self, type_: str, **fields):
        self.type = type_
        for key, value in fields.items():
            setattr(self, key, value)


class Usage:
    input_tokens = 100
    output_tokens = 50
    cache_read_input_tokens = 0


class Response:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason
        self.stop_details = None
        self.usage = Usage()


def text(body: str) -> Response:
    return Response([Block("text", text=body)])


def calls(*specs) -> Response:
    blocks = [
        Block("tool_use", id=f"tu_{i}", name=name, input=args)
        for i, (name, args) in enumerate(specs)
    ]
    return Response(blocks, stop_reason="tool_use")


class FakeMessages:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def create(self, **kwargs):
        # Snapshot the message list: the orchestrator appends to the same list
        # across turns, so storing the reference would show every request the
        # final state.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self.script:
            return text("No further action.")
        return self.script.pop(0)


class FakeClient:
    def __init__(self, script):
        self.messages = FakeMessages(script)


def agent_with(script, approver=None) -> SupportAgent:
    return SupportAgent(
        backends=Backends.mock(),
        client=FakeClient(script),
        approver=approver or QueueApprover(),
    )


# --- happy path -------------------------------------------------------------
def test_refund_flows_end_to_end():
    agent = agent_with([
        calls(("get_order", {"order_id": "ORD-88213"})),
        calls(("issue_refund", {"order_id": "ORD-88213", "amount": 119,
                                "reason": "Cold coffee missing from delivery"})),
        calls(("update_ticket", {"ticket_id": "TKT-1", "status": "resolved",
                                 "summary": "Refunded missing item.", "tags": ["refund"]})),
        text("Sorry about the missing cold coffee - I have sent 119 back to your UPI, "
             "which lands in 3-5 business days."),
    ])
    result = agent.handle("The cold coffee was missing from ORD-88213.",
                          "CUS-1001", "TKT-1")

    assert len(result.context.refunds_issued) == 1
    assert result.context.refunds_issued[0]["amount"] == 119
    assert not result.guardrail_report["blocked"]
    assert "cold coffee" in result.reply.lower()
    assert result.usage["input_tokens"] > 0


def test_gate_blocks_refund_on_unfetched_order():
    """The model skips get_order and goes straight for the money."""
    agent = agent_with([
        calls(("issue_refund", {"order_id": "ORD-88213", "amount": 400, "reason": "x"})),
        calls(("escalate_to_human", {"queue": "supervisor", "reason": "blocked",
                                     "customer_summary": "Needs review."})),
        text("I have passed this to a specialist who will get back to you within 24 hours."),
    ])
    result = agent.handle("Refund me for ORD-88213.", "CUS-1001", "TKT-1")

    assert not result.context.refunds_issued
    decisions = [e for e in result.audit if e["event"] == "policy_decision"]
    assert decisions[0]["decision"] == "deny"
    assert decisions[0]["rule_id"] == "R02-unverified-order"
    assert result.escalated


def test_cross_account_order_is_hidden_from_the_model():
    agent = agent_with([
        calls(("get_order", {"order_id": "ORD-88213"})),   # belongs to CUS-1001
        text("I could not find that order on your account."),
    ])
    result = agent.handle("Refund ORD-88213.", "CUS-1002", "TKT-1")

    lookup = next(e for e in result.audit if e["event"] == "tool_result")
    assert lookup["result"]["error"] == "not_found"
    assert "ORD-88213" not in result.context.orders_seen


def test_denied_refund_cannot_be_retried_smaller():
    agent = agent_with([
        calls(("get_order", {"order_id": "ORD-88912"})),   # out_for_delivery
        calls(("issue_refund", {"order_id": "ORD-88912", "amount": 815, "reason": "late"})),
        calls(("issue_refund", {"order_id": "ORD-88912", "amount": 100, "reason": "late"})),
        text("Your order is still on its way - Firoz has it now."),
    ])
    result = agent.handle("Cancel and refund ORD-88912.", "CUS-1004", "TKT-1")

    decisions = [e for e in result.audit if e["event"] == "policy_decision"]
    assert len(decisions) == 2
    assert all(d["decision"] == "deny" for d in decisions)
    assert not result.context.refunds_issued


# --- approvals --------------------------------------------------------------
def test_pending_approval_is_not_presented_as_settled():
    agent = agent_with([
        calls(("get_order", {"order_id": "ORD-88455"})),
        calls(("issue_refund", {"order_id": "ORD-88455", "amount": 1240,
                                "reason": "Order wrong"})),
        text("Your claim is with a supervisor and you will hear back within 24 hours."),
    ])
    result = agent.handle("ORD-88455 was completely wrong.", "CUS-1002", "TKT-1")

    assert not result.context.refunds_issued
    assert result.context.pending_approvals
    assert not result.guardrail_report["blocked"]


def test_supervisor_approval_lets_the_refund_through():
    agent = agent_with([
        calls(("get_order", {"order_id": "ORD-88455"})),
        calls(("issue_refund", {"order_id": "ORD-88455", "amount": 1240,
                                "reason": "Order wrong"})),
        text("That is sorted - 1240 is on its way back to your card, in 5-7 business days."),
    ], approver=AutoApprover())
    result = agent.handle("ORD-88455 was completely wrong.", "CUS-1002", "TKT-1")

    assert len(result.context.refunds_issued) == 1
    assert not result.guardrail_report["blocked"]


# --- guardrails -------------------------------------------------------------
def test_guardrail_blocks_forensic_leak_and_retries():
    agent = agent_with([
        text("Our forensic check shows your photo was manipulated, so no refund."),
        text("Thanks for sending that over. Your claim is being reviewed and we will "
             "get back to you within 24 hours."),
    ])
    result = agent.handle("Here is my photo.", "CUS-1001", "TKT-1")

    checks = [e for e in result.audit if e["event"] == "guardrail_check"]
    assert checks[0]["blocked"] is True
    assert checks[1]["blocked"] is False
    assert "forensic" not in result.reply.lower()
    assert "reviewed" in result.reply.lower()


def test_guardrail_retry_uses_a_mid_conversation_system_message():
    agent = agent_with([
        text("Our forensic analysis flagged this as fraud."),
        text("Your claim is under review; we will be back to you within 24 hours."),
    ])
    agent.handle("Here is my photo.", "CUS-1001", "TKT-1")

    retry_messages = agent.client.messages.requests[-1]["messages"]
    assert retry_messages[-1]["role"] == "system"
    assert "guardrail" in retry_messages[-1]["content"].lower()


def test_two_guardrail_failures_fall_back_and_escalate():
    agent = agent_with([
        text("Your photo was manipulated, this is fraud."),
        text("We detected your image was AI generated."),
    ])
    result = agent.handle("Here is my photo.", "CUS-1001", "TKT-1")

    assert result.reply.startswith("Thanks for your patience")
    assert result.escalated
    assert any(e["event"] == "guardrail_fallback_used" for e in result.audit)


def test_invented_refund_amount_is_caught():
    agent = agent_with([
        text("Good news, I have arranged a refund of 9999 for you."),
        text("Your claim is under review and we will update you within 24 hours."),
    ])
    result = agent.handle("Refund me.", "CUS-1001", "TKT-1")

    checks = [e for e in result.audit if e["event"] == "guardrail_check"]
    codes = {v["code"] for v in checks[0]["violations"]}
    assert "ungrounded_amount" in codes or "unbacked_refund_claim" in codes


# --- loop mechanics ---------------------------------------------------------
def test_turn_limit_falls_back_safely():
    # Never stops calling tools.
    script = [calls(("kb_search", {"query": "refund"})) for _ in range(40)]
    result = agent_with(script).handle("Hello?", "CUS-1001", "TKT-1")

    assert result.reply.startswith("Thanks for your patience")
    assert any(e["event"] == "turn_limit_reached" for e in result.audit)


def test_parallel_tool_calls_return_one_result_message():
    agent = agent_with([
        calls(("get_order", {"order_id": "ORD-88213"}),
              ("get_customer", {}),
              ("kb_search", {"query": "refund window"})),
        text("Thanks for waiting - everything checks out on that order."),
    ])
    result = agent.handle("Check my order ORD-88213.", "CUS-1001", "TKT-1")

    second_request = agent.client.messages.requests[1]["messages"]
    tool_results = second_request[-1]
    assert tool_results["role"] == "user"
    assert len(tool_results["content"]) == 3
    assert {b["type"] for b in tool_results["content"]} == {"tool_result"}
    assert result.turns == 2


def test_request_shape_is_cache_friendly():
    agent = agent_with([text("All sorted.")])
    agent.handle("Hi.", "CUS-1001", "TKT-1")

    request = agent.client.messages.requests[0]
    assert request["model"] == "claude-opus-5"
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    # No per-conversation values in the cached prefix.
    assert "TKT-1" not in request["system"][0]["text"]
    assert "CUS-1001" not in request["system"][0]["text"]
    # ...they belong in the first user turn instead.
    assert "TKT-1" in request["messages"][0]["content"]
    assert [t["name"] for t in request["tools"]][0] == "kb_search"


def test_model_refusal_falls_back_safely():
    refusal = Response([], stop_reason="refusal")
    result = agent_with([refusal]).handle("...", "CUS-1001", "TKT-1")

    assert result.reply.startswith("Thanks for your patience")
    assert any(e["event"] == "model_refusal" for e in result.audit)


def test_audit_trail_records_the_whole_conversation():
    agent = agent_with([
        calls(("get_order", {"order_id": "ORD-88213"})),
        text("That order looks fine on our side."),
    ])
    result = agent.handle("Check ORD-88213.", "CUS-1001", "TKT-1")

    events = [e["event"] for e in result.audit]
    assert events[0] == "conversation_start"
    assert events[-1] == "conversation_end"
    assert "tool_call" in events and "tool_result" in events
    assert all(e["seq"] == i + 1 for i, e in enumerate(result.audit))
