"""
===============================================================================
TOOL REGISTRY + GATED DISPATCHER                           [CORE - read first]
===============================================================================

TWO THINGS LIVE HERE

1. TOOL_SPECS - the JSON schemas sent to Claude.
   The list is fixed and ordered. Keep it that way: the prompt cache matches
   on a prefix of tools -> system -> messages, so reordering or rebuilding
   this list dynamically would invalidate the cache on every single request.

   Every tool uses "strict": True, which guarantees the arguments Claude sends
   validate against the schema exactly. That removes a whole class of "the
   model passed a string where an int was expected" bugs.

2. ToolExecutor - runs a tool call, but only after policy.evaluate() approves.
   This is where a gate DECISION becomes an enforced OUTCOME. Every branch
   also writes to the audit log.

WHAT THE MODEL SEES WHEN IT IS BLOCKED
   Note that denials and escalations come back as normal tool results carrying
   a "guidance" field, not as exceptions. The model needs to understand what
   happened well enough to explain it to the customer honestly - and to be
   told explicitly not to retry, because retrying a denied refund with a
   smaller number is the obvious failure mode.
===============================================================================
"""

from __future__ import annotations

from typing import Any

import anthropic

from .. import policy
from ..approvals import Approver, QueueApprover
from ..audit import AuditLog
from ..backends import Backends
from ..schemas import ConversationContext, Decision
from . import escalate, image_tools, read_tools, write_tools

# Mirrors policy.WRITE_TOOLS. Used here only to decide what gets an audit
# entry for its gate decision.
WRITE_TOOLS = ("issue_refund", "update_ticket", "send_email")


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "kb_search",
        "description": (
            "Search Zomato's support policy knowledge base. Use this before making any "
            "commitment about refunds, timelines or eligibility - policy claims must be "
            "grounded in a returned citation, never in memory."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language policy question, e.g. 'refund window for delivered orders'.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_order",
        "description": (
            "Fetch one order belonging to the customer in this conversation. A refund "
            "cannot be issued for an order that has not been fetched with this tool first."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "Order id, e.g. ORD-88213."}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_customer",
        "description": (
            "Fetch the profile of the customer in this conversation. Takes no arguments - "
            "you cannot look up any other customer."
        ),
        "strict": True,
        # Empty schema is intentional: no argument means no way to point it at
        # someone else's account.
        "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "name": "verify_image_authenticity",
        "description": (
            "Check whether a customer-submitted photo is a genuine camera capture, an "
            "edited image, or AI-generated. Run this on every image attached to a food "
            "quality, damage or spillage claim, before discussing a refund. Returns a "
            "verdict, a confidence, what the photo depicts, and how to handle it. The "
            "forensic details are internal: never repeat them to the customer."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path of the attachment, exactly as given in the conversation.",
                },
                "claim": {
                    "type": "string",
                    "description": "The customer's own description of what the photo shows.",
                },
            },
            "required": ["image_path", "claim"],
            "additionalProperties": False,
        },
    },
    {
        "name": "issue_refund",
        "description": (
            "Refund money to the customer for a delivered order. Subject to the "
            "permission gate: the call may be denied, held for supervisor approval, or "
            "escalated, and the result will tell you which. Never promise a refund to the "
            "customer before this tool has returned a completed or pending refund."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {
                    "type": "integer",
                    "description": "Whole rupees. Must not exceed the order's refundable balance.",
                },
                "reason": {
                    "type": "string",
                    "description": "Short factual reason, e.g. 'Two items missing from delivery'.",
                },
            },
            "required": ["order_id", "amount", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "update_ticket",
        "description": "Record the outcome on this conversation's support ticket.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["open", "pending_customer", "pending_supervisor", "resolved", "escalated"],
                },
                "summary": {"type": "string", "description": "One or two factual sentences."},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            # strict mode requires every property to be listed in `required`,
            # so `tags` is required - pass [] when there are none.
            "required": ["ticket_id", "status", "summary", "tags"],
            "additionalProperties": False,
        },
    },
    {
        "name": "send_email",
        "description": (
            "Email the customer in this conversation. Use for refund confirmations and "
            "case references the customer will want in writing."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["customer_id", "subject", "body"],
            "additionalProperties": False,
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Hand the case to a human queue and stop taking further action. Use for "
            "illness or injury, delivery partner misconduct, legal or press threats, "
            "anything you are unsure of, and whenever the customer asks for a human."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "queue": {"type": "string", "enum": list(escalate.QUEUES)},
                "reason": {"type": "string", "description": "Internal reason for the handover."},
                "customer_summary": {
                    "type": "string",
                    "description": "Neutral summary of the case for the human picking it up.",
                },
            },
            "required": ["queue", "reason", "customer_summary"],
            "additionalProperties": False,
        },
    },
]


class ToolExecutor:
    """Runs tool calls through the permission gate."""

    def __init__(
        self,
        backends: Backends,
        ctx: ConversationContext,
        audit: AuditLog,
        claim_text: str = "",
        approver: Approver | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.backends = backends
        self.ctx = ctx
        self.audit = audit
        # The customer's ORIGINAL message. The gate reads it for illness and
        # damage keywords, so the model cannot soften a claim by rewording the
        # `reason` argument it passes to issue_refund.
        self.claim_text = claim_text
        self.approver = approver or QueueApprover()
        self.client = client

    def execute(self, name: str, tool_input: dict[str, Any]) -> tuple[Any, bool]:
        """Run one tool call. Returns (result, is_error)."""
        self.audit.record("tool_call", tool=name, input=tool_input)

        verdict = policy.evaluate(name, tool_input, self.ctx, self.backends, claim_text=self.claim_text)
        if name in WRITE_TOOLS:
            self.audit.record("policy_decision", tool=name, **verdict.to_dict())

        # --- DENY: never allowed, and say so clearly -----------------------
        if verdict.decision is Decision.DENY:
            return (
                {
                    "blocked": True,
                    "rule": verdict.rule_id,
                    "reason": verdict.reason,
                    "guidance": (
                        "This action is not permitted. Do not retry it with different "
                        "arguments. Explain the position to the customer honestly, or "
                        "escalate if they are unhappy with it."
                    ),
                },
                True,
            )

        # --- ESCALATE: the gate hands off on the model's behalf ------------
        if verdict.decision is Decision.ESCALATE:
            return self._auto_escalate(name, tool_input, verdict), False

        # --- REQUIRE_APPROVAL: ask a human, honour the answer --------------
        if verdict.decision is Decision.REQUIRE_APPROVAL:
            approved = self.approver.request(name, dict(tool_input), verdict.reason)
            self.audit.record("approval", tool=name, rule=verdict.rule_id, approved=approved)
            if not approved:
                self.ctx.pending_approvals.append(
                    {"action": name, "input": tool_input, "reason": verdict.reason}
                )
                return (
                    {
                        "pending_approval": True,
                        "rule": verdict.rule_id,
                        "reason": verdict.reason,
                        "guidance": (
                            "A supervisor must sign this off and none is available in "
                            "this conversation. Tell the customer their claim is with a "
                            "supervisor and will be resolved within 24 hours. Do not "
                            "state an amount as though it were already approved. Then "
                            "record the ticket as pending_supervisor."
                        ),
                    },
                    False,
                )
            # Approved - fall through and run it.

        return self._run(name, tool_input)

    def _auto_escalate(self, name: str, tool_input: dict[str, Any], verdict: Any) -> dict[str, Any]:
        """Escalate on the model's behalf, to the queue the rule nominated."""
        queue = verdict.evidence.get("queue", "supervisor")
        result = escalate.escalate_to_human(
            self.backends,
            self.ctx,
            queue=queue,
            reason=f"[{verdict.rule_id}] {verdict.reason}",
            customer_summary=f"Automated gate escalated a {name} call. Original request: {tool_input}",
        )
        self.audit.record("auto_escalation", tool=name, rule=verdict.rule_id, **result)
        return {
            "blocked": True,
            "escalated": True,
            **result,
            "reason": verdict.reason,
            "guidance": (
                "The case is now with a human specialist. Tell the customer it is under "
                "review and give the case reference and timeline. Do not state or imply "
                "any suspicion about them, and do not attempt the action again."
            ),
        }

    def _run(self, name: str, tool_input: dict[str, Any]) -> tuple[Any, bool]:
        try:
            result = self._dispatch(name, tool_input)
        except (KeyError, ValueError) as exc:
            # Backend said no (unknown id, invariant violated). Report it back
            # as a tool error rather than crashing the conversation.
            self.audit.record("tool_error", tool=name, error=str(exc))
            return ({"error": type(exc).__name__, "message": str(exc)}, True)

        self.audit.record("tool_result", tool=name, result=result)
        return (result, False)

    def _dispatch(self, name: str, tool_input: dict[str, Any]) -> Any:
        """Plain name -> function mapping. No logic, deliberately."""
        b, ctx = self.backends, self.ctx

        if name == "kb_search":
            return read_tools.kb_search(b, ctx, tool_input["query"])
        if name == "get_order":
            return read_tools.get_order(b, ctx, tool_input["order_id"])
        if name == "get_customer":
            return read_tools.get_customer(b, ctx)
        if name == "verify_image_authenticity":
            return image_tools.verify_image_authenticity(
                b, ctx, tool_input["image_path"], tool_input.get("claim", ""), client=self.client
            )
        if name == "issue_refund":
            return write_tools.issue_refund(
                b, ctx, tool_input["order_id"], tool_input["amount"], tool_input["reason"]
            )
        if name == "update_ticket":
            return write_tools.update_ticket(
                b, ctx, tool_input["ticket_id"], tool_input["status"],
                tool_input["summary"], tool_input.get("tags") or [],
            )
        if name == "send_email":
            return write_tools.send_email(
                b, ctx, tool_input["customer_id"], tool_input["subject"], tool_input["body"]
            )
        if name == "escalate_to_human":
            return escalate.escalate_to_human(
                b, ctx, tool_input["queue"], tool_input["reason"], tool_input["customer_summary"]
            )
        raise KeyError(f"Unknown tool: {name}")


__all__ = ["TOOL_SPECS", "ToolExecutor", "WRITE_TOOLS"]
