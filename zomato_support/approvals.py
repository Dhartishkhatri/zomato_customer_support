"""
===============================================================================
HUMAN-APPROVAL TRANSPORTS                                            [CORE]
===============================================================================

WHAT THIS DOES
    The permission gate decides that a human is needed. These classes decide
    how that human is actually reached.

WHY THE SPLIT
    Keeping "does this need a human?" (policy.py) separate from "how do we ask
    one?" (here) means you can swap in a Slack round-trip or a supervisor
    console without touching a single policy rule.

TO PLUG IN A REAL SYSTEM
    Write a class with one `request(...) -> bool` method and pass it to
    SupportAgent(approver=...). Return True only if a human actually approved.
===============================================================================
"""

from __future__ import annotations

from typing import Any, Protocol


class Approver(Protocol):
    """The whole interface. Anything with this method works."""

    def request(self, action: str, details: dict[str, Any], reason: str) -> bool:
        """Return True if a human authorised the action."""
        ...


class QueueApprover:
    """DEFAULT, and the honest production behaviour.

    Files the request and returns False immediately rather than blocking a
    live chat waiting for a supervisor who may be asleep. Nothing is approved
    in-band, so the agent tells the customer their claim is with a supervisor -
    which is true.
    """

    def __init__(self) -> None:
        self.queued: list[dict[str, Any]] = []

    def request(self, action: str, details: dict[str, Any], reason: str) -> bool:
        self.queued.append({"action": action, "details": details, "reason": reason})
        return False


class AutoApprover:
    """Approves everything. Tests and `run_demo.py --approve-all` only.

    Never ship this. It exists so you can see the approved branch of the gate
    without wiring up a real approval channel.
    """

    def __init__(self) -> None:
        self.approved: list[dict[str, Any]] = []

    def request(self, action: str, details: dict[str, Any], reason: str) -> bool:
        self.approved.append({"action": action, "details": details, "reason": reason})
        return True
