"""
===============================================================================
AUDIT TRAIL                                                          [CORE]
===============================================================================

WHAT THIS DOES
    An append-only list of everything that happened in one conversation: each
    tool call, each gate decision, each approval outcome, each guardrail
    verdict.

WHY IT EXISTS
    When a customer disputes a refund - or when someone asks why the agent
    refused one - this is the only record of what the system decided and which
    rule decided it. Every gate entry carries its rule id for exactly that
    reason.

IS IT CRUCIAL?
    Yes, if this ever handles real money. The class itself is tiny; the value
    is that the orchestrator calls it at every decision point.
===============================================================================
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class AuditLog:
    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        """Append one event. `seq` gives a stable order even at equal timestamps."""
        entry = {
            "seq": len(self._entries) + 1,
            "at": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> list[dict[str, Any]]:
        """A copy, so callers cannot mutate history after the fact."""
        return list(self._entries)
