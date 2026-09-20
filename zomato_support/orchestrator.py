"""
===============================================================================
THE AGENT LOOP                                             [CORE - start here]
===============================================================================

WHAT THIS DOES
    Drives the conversation: send messages to Claude, run whatever tools it
    asks for, feed the results back, repeat until it produces a reply. Then
    guardrail-check that reply before anyone sees it.

THE SHAPE OF THE LOOP
    while under the turn limit:
        response = call model
        if no tool calls -> that is the reply, stop
        run every tool call through the gate
        append ALL results as ONE user message
    guardrail-check the reply; one retry if it fails; fallback if it fails twice

WHY A MANUAL LOOP RATHER THAN THE SDK'S TOOL RUNNER
    Two reasons. Every tool call has to cross a policy gate that can deny,
    hold or escalate it; and the audit trail needs each gate decision
    interleaved with the call it governed. The SDK runner is also still beta.
    If you drop the gate, the runner becomes the simpler choice.

PROMPT CACHING - why things are where they are
    The API matches the cache on a prefix: tools -> system -> messages. So the
    tool list is fixed and ordered, the system prompt contains NO
    per-conversation values, and the cache breakpoint sits on the system
    block. Ticket and customer ids go into the first user turn instead. Check
    result.usage["cache_read_input_tokens"] to confirm it is working - if it
    is always zero, something is varying in that prefix.
===============================================================================
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from . import config, guardrails, prompts
from .approvals import Approver, QueueApprover
from .audit import AuditLog
from .backends import Backends
from .schemas import AgentResult, ConversationContext
from .tools import TOOL_SPECS, ToolExecutor


class SupportAgent:
    def __init__(
        self,
        backends: Backends | None = None,
        client: anthropic.Anthropic | None = None,
        approver: Approver | None = None,
        model: str = config.ORCHESTRATOR_MODEL,
    ) -> None:
        self.backends = backends or Backends.mock()
        self.client = client or anthropic.Anthropic()
        self.approver = approver or QueueApprover()
        self.model = model

    def handle(
        self,
        customer_message: str,
        customer_id: str,
        ticket_id: str,
        channel: str = "chat",
        attachments: list[str] | None = None,
        verbose: bool = False,
    ) -> AgentResult:
        """Handle one customer message end to end."""
        ctx = ConversationContext(customer_id=customer_id, ticket_id=ticket_id, channel=channel)
        audit = AuditLog()
        audit.record(
            "conversation_start",
            customer_id=customer_id,
            ticket_id=ticket_id,
            channel=channel,
            attachments=attachments or [],
            message=customer_message,
        )

        executor = ToolExecutor(
            self.backends, ctx, audit,
            claim_text=customer_message,   # gate reads the ORIGINAL wording
            approver=self.approver,
            client=self.client,
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": prompts.opening_turn(
                    customer_message, ticket_id, customer_id, channel, attachments
                ),
            }
        ]

        # Everything a tool actually returned. The guardrail checks the final
        # reply's factual claims (amounts especially) against this corpus, so
        # the agent cannot quote a number no tool ever produced.
        grounding: list[str] = []
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
        reply = ""
        turns = 0

        while turns < config.MAX_AGENT_TURNS:
            turns += 1
            response = self._call_model(messages)

            usage["input_tokens"] += response.usage.input_tokens
            usage["output_tokens"] += response.usage.output_tokens
            usage["cache_read_input_tokens"] += (
                getattr(response.usage, "cache_read_input_tokens", 0) or 0
            )

            # Safety classifiers declined. Do not try to salvage it - hand off.
            if response.stop_reason == "refusal":
                audit.record("model_refusal", details=str(response.stop_details))
                reply = guardrails.SAFE_FALLBACK
                break

            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                # No tools requested -> this turn is the customer-facing reply.
                reply = "".join(b.text for b in response.content if b.type == "text").strip()
                break

            if verbose:
                for block in tool_uses:
                    print(f"    -> {block.name}({json.dumps(block.input)[:110]})")

            # Run every requested tool, then return ALL results in a SINGLE
            # user message. Splitting them across messages silently teaches the
            # model to stop making parallel calls.
            results = []
            for block in tool_uses:
                result, is_error = executor.execute(block.name, dict(block.input))
                payload = json.dumps(result, default=str)
                grounding.append(payload)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,     # must echo the call's id
                        "content": payload,
                        **({"is_error": True} if is_error else {}),
                    }
                )
            messages.append({"role": "user", "content": results})
        else:
            # Loop ran to the turn limit without settling. Cost guard fired.
            audit.record("turn_limit_reached", turns=turns)
            reply = guardrails.SAFE_FALLBACK

        reply, report = self._guardrail(reply, ctx, messages, "\n".join(grounding), audit, usage)

        audit.record(
            "conversation_end",
            turns=turns,
            escalated=ctx.escalated,
            refunds=len(ctx.refunds_issued),
            pending_approvals=len(ctx.pending_approvals),
        )

        return AgentResult(
            reply=reply,
            context=ctx,
            audit=audit.entries,
            turns=turns,
            escalated=ctx.escalated,
            guardrail_report=report.to_dict(),
            usage=usage,
        )

    # --- internals ----------------------------------------------------------
    def _call_model(self, messages: list[dict[str, Any]]):
        """One API call. Note there is no `thinking` parameter: Opus 5 runs
        adaptive thinking by default, and passing budget_tokens would 400."""
        return self.client.messages.create(
            model=self.model,
            max_tokens=config.MAX_TOKENS,
            output_config={"effort": config.EFFORT},
            system=[
                {
                    "type": "text",
                    "text": prompts.SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},  # cache breakpoint
                }
            ],
            tools=TOOL_SPECS,
            messages=messages,
        )

    def _guardrail(
        self,
        reply: str,
        ctx: ConversationContext,
        messages: list[dict[str, Any]],
        grounding_text: str,
        audit: AuditLog,
        usage: dict[str, int],
    ) -> tuple[str, guardrails.GuardrailReport]:
        """Check the draft reply; give the model exactly one chance to fix it."""
        report = guardrails.check_reply(reply, ctx, self.backends, grounding_text)
        audit.record("guardrail_check", attempt=1, **report.to_dict())

        if not report.blocked:
            return reply, report

        # The correction goes in as a MID-CONVERSATION SYSTEM MESSAGE - a
        # {"role": "system"} entry inside `messages`, supported on Opus 5. Two
        # advantages over editing the top-level system prompt: it carries
        # operator authority (not user authority, so it cannot be spoofed by
        # the customer's text), and it leaves the cached prefix intact.
        messages.append({"role": "system", "content": report.feedback()})

        try:
            retry = self._call_model(messages)
        except anthropic.APIError as exc:
            audit.record("guardrail_retry_failed", error=str(exc))
            return guardrails.SAFE_FALLBACK, report

        usage["input_tokens"] += retry.usage.input_tokens
        usage["output_tokens"] += retry.usage.output_tokens

        redraft = "".join(b.text for b in retry.content if b.type == "text").strip()
        second = guardrails.check_reply(redraft, ctx, self.backends, grounding_text)
        audit.record("guardrail_check", attempt=2, **second.to_dict())

        if second.blocked or not redraft:
            # Two failures means this loop cannot say it safely. Send a neutral
            # holding reply and put a human on it rather than trying again.
            audit.record("guardrail_fallback_used")
            if not ctx.escalated:
                from .tools import escalate

                escalate.escalate_to_human(
                    self.backends, ctx,
                    queue="supervisor",
                    reason="Drafted reply failed the guardrail check twice.",
                    customer_summary="Automated reply suppressed; needs a human response.",
                )
            return guardrails.SAFE_FALLBACK, second

        return redraft, second
