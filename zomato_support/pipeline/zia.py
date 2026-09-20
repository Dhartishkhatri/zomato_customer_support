"""
===============================================================================
ZIA - the response generation pipeline                 [CORE - start here]
===============================================================================

This is the conversational agent. It replaces the tool-calling loop in
orchestrator.py for chat traffic, because a pipeline hits a latency target
that an agentic loop cannot: the loop needs several sequential round trips
before it can say anything, and each one is a full model call.

THE SHAPE OF ONE TURN

    customer message
          |
    [1] CLASSIFY FUNCTIONS ........ Opus 5. Which data does this need?
          |
    [2] FETCH + NARRATE ........... SQLite + templates. No model, ~1ms.
          |
          +---------------- run concurrently ----------------+
          |                                                  |
    [3a] GENERATE REPLY (Haiku)              [3b] DECIDE ACTION (Opus)
         reply text + escalation tag              classify, then verify
          |                                                  |
          +---------------- join ----------------------------+
          |
    [4] VERIFY ESCALATION ......... policy layer, checks the tag against data
          |
    [5] GUARDRAIL ................. deterministic pre-send checks
          |
    [6] PERSIST ................... message, metrics, proposed action
          |
    reply + optional confirmation popup

WHY 3a AND 3b RUN TOGETHER
    They need the same inputs and neither depends on the other's output, so
    running them in sequence would just add the slower one's latency to the
    turn. In parallel, the two-step action verification is effectively free.

WHAT EACH STAGE PROTECTS AGAINST
    [1] fetching everything -> bloated prompts, slower and less accurate
    [3b] an eager model promising impossible actions
    [4] escalation drifting with the model's mood (containment instability)
    [5] the reply saying something it should not
===============================================================================
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from .. import config, guardrails
from ..backends import Backends
from ..llm import LLM, Task
from ..metrics import record_turn
from ..schemas import ConversationContext
from . import action_decisioning, data_functions, escalation_policy
from .function_classifier import classify_functions

# The behaviour spec. Kept byte-stable so the prompt cache prefix survives;
# everything per-conversation arrives in the user turn instead.
SYSTEM_PROMPT = f"""You are Zia, Zomato's customer support agent, helping \
customers in India over chat. Amounts are in Indian rupees ({config.CURRENCY_SYMBOL}).

# How to reply
Warm, direct and SHORT - two or three sentences is usually right, and never more
than a short paragraph. Apologise once, concretely, then say what happens next
and by when. No filler openers, no corporate padding, no emoji, no bullet lists
unless you are listing order items.

Write as a person at Zomato. Never call yourself an AI, a model, a bot or an
assistant. Never mention tools, systems, checks, scores or internal processes.

# Grounding
You are given a CONTEXT block containing real data fetched for this turn. Every
factual claim you make - status, ETA, amounts, items, policy - must come from
that block. If the answer is not there, say you are checking rather than
guessing. Never invent an amount, a time or an order id.

If the context says the order could not be identified, ask which order they mean.

# What you must not promise
Do not promise a refund, a cancellation or any change. The system offers those
to the customer separately as a confirmation button. You may say an option is
available, but never say it is done.

# Escalation
If this needs a human, set escalate=true and tag it with exactly one reason
code from this list:

{escalation_policy.codes_for_prompt()}

The reason code is checked against real order data, so choose the one that
actually matches the situation. If the data does not support a code, do not
invent one - keep helping instead.

Always escalate illness, injury, legal or press threats, abuse, delivery
partner misconduct, and any explicit request to speak to a human."""


class AgentReply(BaseModel):
    """Structured reply, so the escalation tag is separate from the prose."""

    reply: str = Field(description="The message to send the customer. Short.")
    escalate: bool = Field(description="True if a human should take this over.")
    escalation_reason_code: str | None = Field(
        default=None, description="Exactly one code from the list, or null."
    )
    resolved: bool = Field(
        description="True if the customer's issue is now fully handled."
    )


@dataclass
class TurnResult:
    reply: str
    session_id: str
    intent: str = "other"
    functions_called: list[str] = field(default_factory=list)
    action: dict[str, Any] | None = None
    escalation: dict[str, Any] = field(default_factory=dict)
    contained: bool = True
    resolved: bool = False
    guardrail: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, int] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    context_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reply": self.reply,
            "session_id": self.session_id,
            "intent": self.intent,
            "functions_called": self.functions_called,
            "action": self.action,
            "escalation": self.escalation,
            "contained": self.contained,
            "resolved": self.resolved,
            "guardrail": self.guardrail,
            "timings": self.timings,
            "usage": self.usage,
        }


class Zia:
    def __init__(self, backends: Backends | None = None, client: Any = None) -> None:
        self.backends = backends or Backends.mock()
        self.client = client

    def start_session(self, customer_id: str, channel: str = "chat") -> str:
        session_id = f"SES-{uuid.uuid4().hex[:8].upper()}"
        ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"
        self.backends.crm.start_session(session_id, customer_id, ticket_id, channel)
        return session_id

    # -----------------------------------------------------------------------
    def handle(self, session_id: str, message: str) -> TurnResult:
        """Run one full turn. This is the function the UI calls."""
        started = time.perf_counter()
        llm = LLM(self.client)

        session = self.backends.crm.session(session_id)
        if session is None:
            raise KeyError(f"Unknown session {session_id}")
        customer_id = session["customer_id"]

        history = self.backends.crm.history(session_id)
        self.backends.crm.add_message(session_id, "customer", message)

        # --- [1] which data does this turn need? ---------------------------
        t0 = time.perf_counter()
        plan = classify_functions(llm, history, message)
        classify_ms = int((time.perf_counter() - t0) * 1000)

        order_id = plan.order_id
        if order_id is None and plan.refers_to_latest_order:
            order_id = self._latest_order_id(customer_id)

        # --- [2] fetch only that, and narrate it ---------------------------
        context_text, raw = data_functions.fetch_and_narrate(
            self.backends, plan.functions, order_id, customer_id, query=message
        )

        # --- [3] reply and action decisioning, concurrently ----------------
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            reply_future = pool.submit(
                self._generate, llm, history, message, context_text
            )
            action_future = pool.submit(
                action_decisioning.decide_action,
                llm, self.backends, history, message, order_id, customer_id,
            )
            agent_reply = reply_future.result()
            action = action_future.result()
        generate_ms = int((time.perf_counter() - t0) * 1000)

        # --- [4] does the escalation stand up to the data? -----------------
        verdict = escalation_policy.verify_escalation(
            self.backends,
            agent_reply.escalation_reason_code if agent_reply.escalate else None,
            raw.get("order"),
            customer_id,
        )

        reply = agent_reply.reply.strip()
        if verdict.upheld:
            escalation = self.backends.crm.escalate(
                session["ticket_id"],
                verdict.queue or "supervisor",
                verdict.evidence,
                reason_code=verdict.code,
                customer_id=customer_id,
            )
            verdict_dict = {**verdict.to_dict(), "case_id": escalation["case_id"]}
            # Only override the reply if the model failed to say a human is coming.
            if not any(w in reply.lower() for w in ("specialist", "team", "get back", "24 hours")):
                reply = (
                    "I'm passing this to a specialist on our team who will look into it "
                    "and get back to you within 24 hours."
                )
        else:
            verdict_dict = verdict.to_dict()

        # --- [5] guardrail check -------------------------------------------
        ctx = ConversationContext(customer_id=customer_id, ticket_id=session["ticket_id"])
        if raw.get("order"):
            ctx.orders_seen[raw["order"]["order_id"]] = raw["order"]
        ctx.escalated = verdict.upheld

        report = guardrails.check_reply(reply, ctx, self.backends, context_text)
        if report.blocked:
            # No retry here, unlike the agentic loop: a second model call would
            # blow the latency budget. A neutral holding reply is the safer
            # trade for a live chat.
            reply = guardrails.SAFE_FALLBACK

        # --- [6] persist ----------------------------------------------------
        self.backends.crm.add_message(session_id, "agent", reply)

        total_ms = int((time.perf_counter() - started) * 1000)
        contained = not verdict.upheld

        proposed = action.to_dict() if action.should_offer else None
        if action.action != "NONE":
            self._log_action(session_id, action)

        record_turn(
            session_id=session_id,
            latency_ms=total_ms,
            classify_ms=classify_ms,
            generate_ms=generate_ms,
            action_ms=0,
            usage=llm.usage,
            escalated=verdict.upheld,
            escalation_reason=verdict.code,
            escalation_upheld=verdict.upheld if verdict.requested else None,
            functions=plan.functions,
            action_proposed=action.action,
            guardrail_blocked=report.blocked,
        )
        self.backends.crm.close_session(session_id, contained)

        return TurnResult(
            reply=reply,
            session_id=session_id,
            intent=plan.intent,
            functions_called=plan.functions,
            action=proposed,
            escalation=verdict_dict,
            contained=contained,
            resolved=agent_reply.resolved,
            guardrail=report.to_dict(),
            timings={
                "total_ms": total_ms,
                "classify_ms": classify_ms,
                "generate_ms": generate_ms,
            },
            usage=llm.usage.to_dict(),
            context_used=context_text,
        )

    # -----------------------------------------------------------------------
    def _generate(
        self, llm: LLM, history: list[dict[str, Any]], message: str, context_text: str
    ) -> AgentReply:
        """[3a] Write the customer-facing reply. Fast model, tight token cap."""
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-8:])
        user = (
            f"CONTEXT (real data fetched for this turn):\n"
            f"{context_text or '(no data was needed for this message)'}\n\n"
            f"CONVERSATION SO FAR:\n{transcript or '(this is the first message)'}\n\n"
            f"CUSTOMER'S LATEST MESSAGE:\n{message}"
        )
        try:
            response = llm.parse(
                Task.GENERATE,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
                output_format=AgentReply,
                max_tokens=config.CHAT_MAX_TOKENS,
            )
            return response.parsed_output
        except Exception:
            # Escalate rather than guess if the generator is unavailable.
            return AgentReply(
                reply=guardrails.SAFE_FALLBACK,
                escalate=True,
                escalation_reason_code="CUSTOMER_REQUESTED_HUMAN",
                resolved=False,
            )

    def _latest_order_id(self, customer_id: str) -> str | None:
        orders = self.backends.orders.for_customer(customer_id)
        return orders[0]["order_id"] if orders else None

    def _log_action(self, session_id: str, action: action_decisioning.ActionDecision) -> None:
        import json

        from ..db import get_db

        with get_db() as conn:
            conn.execute(
                "INSERT INTO action_log (session_id, order_id, action_type, payload, "
                "proposed, verified, reject_reason) VALUES (?,?,?,?,1,?,?)",
                (
                    session_id, action.order_id, action.action,
                    json.dumps({"parameter": action.parameter, "details": action.details}),
                    1 if action.eligible else 0, action.reject_reason,
                ),
            )

    # --- action execution, after the customer confirms the popup -----------
    def execute_action(
        self, session_id: str, action: str, order_id: str, parameter: str | None = None
    ) -> dict[str, Any]:
        """Run a confirmed action. Re-verifies first - never trusts the client.

        The popup the customer clicked was rendered from a decision made
        moments ago. Between then and now the order may have moved on, and the
        request itself arrives over HTTP where anything could have been
        changed. So eligibility is checked again here, against live data.
        """
        session = self.backends.crm.session(session_id)
        if session is None:
            raise KeyError(f"Unknown session {session_id}")
        customer_id = session["customer_id"]

        decision = action_decisioning.verify_action(
            self.backends, action, order_id, parameter, customer_id
        )
        if not decision.eligible:
            return {"ok": False, "message": decision.reject_reason or "Action not permitted."}

        if action == "CANCEL_ORDER":
            self.backends.orders.cancel(order_id)
            order = self.backends.orders.get(order_id)
            refund = self.backends.payments.issue_refund(
                order_id, order["total"], "Customer cancelled within the free window"
            )
            result = {
                "ok": True,
                "message": (
                    f"Order {order_id} is cancelled. {config.CURRENCY_SYMBOL}"
                    f"{refund['amount']} goes back to your {refund['destination']} "
                    f"{refund['settles_in']}."
                ),
            }
        elif action == "ADD_DELIVERY_INSTRUCTIONS":
            self.backends.orders.set_delivery_instructions(order_id, parameter or "")
            result = {"ok": True, "message": f'Added "{parameter}" for the rider.'}
        elif action == "CHANGE_DELIVERY_ADDRESS":
            self.backends.orders.set_delivery_address(order_id, parameter or "")
            result = {"ok": True, "message": f"Delivery address updated to {parameter}."}
        elif action == "CONTACT_DELIVERY_PARTNER":
            tracking = self.backends.tracking.get(order_id)
            first = (tracking["partner_name"] or "Your rider").split()[0]
            result = {"ok": True, "message": f"{first} has been asked to call you shortly."}
        elif action == "REQUEST_REFUND":
            # Deliberately does NOT move money. It raises the request; the
            # permission gate in policy.py decides the rest.
            result = {
                "ok": True,
                "message": (
                    "Refund request raised. We'll confirm the amount within 24 hours."
                ),
            }
        else:
            return {"ok": False, "message": f"Unknown action {action}."}

        from ..db import get_db

        with get_db() as conn:
            conn.execute(
                "UPDATE action_log SET confirmed = 1, executed = 1 WHERE id = ("
                "SELECT id FROM action_log WHERE session_id = ? AND action_type = ? "
                "ORDER BY id DESC LIMIT 1)",
                (session_id, action),
            )
        self.backends.crm.add_message(session_id, "agent", result["message"])
        return result
