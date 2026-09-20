"""
===============================================================================
ESCALATION POLICY LAYER - the containment stabiliser                 [CORE]
===============================================================================

THE PROBLEM THIS SOLVES
    When escalation is left entirely to the LLM, containment swings wildly -
    Zomato saw it move between 80% and 50% day to day, which makes staffing a
    support floor impossible. The cause is that "should I hand this to a
    human?" is a judgement call, and a model's judgement drifts with phrasing,
    mood and prompt changes.

THE FIX
    The model may still ASK to escalate, but it must tag the request with a
    REASON CODE from a fixed list. The system then checks that code against
    actual order data. No supporting evidence, no escalation.

      LLM says:  "escalate, DP_MOVEMENT_ISSUE"
      System asks: has the rider actually stopped moving?
      Database says: last moved 2 minutes ago
      Result: REJECTED - the turn stays contained, the bot keeps helping.

    Containment then becomes a property of the DATA, not of the model's mood,
    which is what makes it stable enough to staff against.

IMPORTANT: THIS IS NOT A COST-SAVING HACK
    Some codes are ALWAYS upheld and deliberately bypass verification -
    illness, injury, legal threats, abuse, an explicit request for a human.
    Those are exactly the cases where a bot must not argue. Suppressing a
    genuine escalation to protect a metric would be the worst possible
    outcome, so those codes are hard-coded as auto-upheld below.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..backends import Backends
from ..backends.tracking import STUCK_THRESHOLD_MIN

# Codes that always escalate. Never verified, never rejected - the cost of
# wrongly containing one of these is far higher than an unnecessary handover.
ALWAYS_UPHELD = {
    "FOOD_SAFETY_ILLNESS": "food-safety",
    "LEGAL_OR_PRESS_THREAT": "legal-escalations",
    "CUSTOMER_REQUESTED_HUMAN": "supervisor",
    "ABUSIVE_OR_DISTRESSED": "supervisor",
    "DELIVERY_PARTNER_MISCONDUCT": "trust-and-safety",
    "IMAGE_FRAUD_SUSPECTED": "trust-and-safety",
}

# Codes that must be corroborated by order data before they are allowed.
VERIFIED_CODES = {
    "DP_MOVEMENT_ISSUE": "delivery-investigations",
    "ORDER_NOT_DELIVERED": "delivery-investigations",
    "ORDER_SIGNIFICANTLY_LATE": "delivery-investigations",
    "REFUND_OUTSIDE_WINDOW": "grievance-officer",
    "REPEAT_REFUND_PATTERN": "trust-and-safety",
    "PAYMENT_NOT_SETTLED": "payments",
    "HIGH_VALUE_REFUND": "supervisor",
}

ALL_CODES = {**ALWAYS_UPHELD, **VERIFIED_CODES}


@dataclass
class EscalationVerdict:
    requested: bool
    code: str | None
    upheld: bool
    queue: str | None
    evidence: str
    auto_upheld: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "code": self.code,
            "upheld": self.upheld,
            "queue": self.queue,
            "evidence": self.evidence,
            "auto_upheld": self.auto_upheld,
        }


# --- verifiers: each returns (is_supported, evidence_sentence) -------------
def _verify_dp_movement(backends: Backends, order: dict[str, Any] | None) -> tuple[bool, str]:
    if not order:
        return False, "No order identified, so rider movement cannot be checked."
    tracking = backends.tracking.get(order["order_id"])
    if not tracking:
        return False, "Order has no live tracking; the rider is not en route."
    if tracking["is_stuck"]:
        return True, (
            f"Confirmed: rider has not moved for {tracking['last_moved_min_ago']} "
            f"minutes (threshold {STUCK_THRESHOLD_MIN})."
        )
    return False, (
        f"Rider moved {tracking['last_moved_min_ago']} minutes ago and is "
        f"{tracking['distance_km']} km away - movement is normal."
    )


def _verify_not_delivered(backends: Backends, order: dict[str, Any] | None) -> tuple[bool, str]:
    if not order:
        return False, "No order identified."
    if order["status"] == "delivered":
        return True, "Order is marked delivered but the customer disputes receiving it."
    return False, f"Order status is '{order['status']}', not a delivery dispute."


def _verify_late(backends: Backends, order: dict[str, Any] | None) -> tuple[bool, str]:
    if not order:
        return False, "No order identified."
    tracking = backends.tracking.get(order["order_id"])
    if not tracking:
        return False, "No live tracking; lateness cannot be confirmed."
    if tracking["is_late"]:
        return True, f"Confirmed: running {tracking['minutes_late']} minutes past the estimate."
    return False, (
        f"ETA {tracking['eta_minutes']} min vs {tracking['promised_eta_min']} min "
        "promised - the order is not materially late."
    )


def _verify_outside_window(backends: Backends, order: dict[str, Any] | None) -> tuple[bool, str]:
    from .. import config

    if not order:
        return False, "No order identified."
    hours = order.get("hours_since_delivery")
    if hours is not None and hours > config.REFUND_WINDOW_HOURS:
        return True, f"Confirmed: delivered {int(hours)}h ago, past the support window."
    return False, "Order is still inside the support window; handle it in chat."


def _verify_repeat_refunds(backends: Backends, order: dict[str, Any] | None,
                           customer_id: str = "") -> tuple[bool, str]:
    from .. import config

    if not customer_id:
        return False, "No customer identified."
    customer = backends.crm.get_customer(customer_id)
    count = customer.get("refunds_last_90d", 0)
    if count >= config.REFUND_VELOCITY_LIMIT:
        return True, f"Confirmed: {count} refunds in the last 90 days."
    return False, f"Only {count} refunds in 90 days - below the review threshold."


def _verify_payment(backends: Backends, order: dict[str, Any] | None) -> tuple[bool, str]:
    if not order:
        return False, "No order identified."
    pending = [r for r in order["refunds"] if r["status"] == "pending"]
    if pending:
        return True, f"Confirmed: {len(pending)} refund(s) still pending settlement."
    return False, "No unsettled payments on this order."


def _verify_high_value(backends: Backends, order: dict[str, Any] | None) -> tuple[bool, str]:
    from .. import config

    if not order:
        return False, "No order identified."
    if order["total"] > config.HUMAN_APPROVAL_THRESHOLD:
        return True, (
            f"Confirmed: order value {order['total']} exceeds the "
            f"{config.HUMAN_APPROVAL_THRESHOLD} supervisor threshold."
        )
    return False, f"Order value {order['total']} is within the automation ceiling."


_VERIFIERS: dict[str, Callable[..., tuple[bool, str]]] = {
    "DP_MOVEMENT_ISSUE": _verify_dp_movement,
    "ORDER_NOT_DELIVERED": _verify_not_delivered,
    "ORDER_SIGNIFICANTLY_LATE": _verify_late,
    "REFUND_OUTSIDE_WINDOW": _verify_outside_window,
    "REPEAT_REFUND_PATTERN": _verify_repeat_refunds,
    "PAYMENT_NOT_SETTLED": _verify_payment,
    "HIGH_VALUE_REFUND": _verify_high_value,
}


def verify_escalation(
    backends: Backends,
    code: str | None,
    order: dict[str, Any] | None,
    customer_id: str = "",
) -> EscalationVerdict:
    """Decide whether a requested escalation is actually warranted."""
    if not code:
        return EscalationVerdict(False, None, False, None, "No escalation requested.")

    code = code.strip().upper()

    if code in ALWAYS_UPHELD:
        return EscalationVerdict(
            requested=True, code=code, upheld=True, queue=ALWAYS_UPHELD[code],
            evidence="Safety, legal or explicit-request code - always upheld without verification.",
            auto_upheld=True,
        )

    if code not in VERIFIED_CODES:
        # An invented code is not a reason. Keep the turn contained.
        return EscalationVerdict(
            requested=True, code=code, upheld=False, queue=None,
            evidence=f"'{code}' is not a recognised escalation reason code.",
        )

    verifier = _VERIFIERS[code]
    supported, evidence = (
        verifier(backends, order, customer_id)
        if code == "REPEAT_REFUND_PATTERN"
        else verifier(backends, order)
    )
    return EscalationVerdict(
        requested=True,
        code=code,
        upheld=supported,
        queue=VERIFIED_CODES[code] if supported else None,
        evidence=evidence,
    )


def codes_for_prompt() -> str:
    """The code list, rendered for the system prompt."""
    lines = ["Always escalate (no verification needed):"]
    lines += [f"  {c}" for c in ALWAYS_UPHELD]
    lines.append("Escalate only if the order data supports it:")
    lines += [f"  {c}" for c in VERIFIED_CODES]
    return "\n".join(lines)
