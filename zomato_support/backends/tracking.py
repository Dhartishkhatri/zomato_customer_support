"""
===============================================================================
DELIVERY TRACKING BACKEND                                            [CORE]
===============================================================================

Live logistics state: where the delivery partner is, how late the order is,
and - critically - how long since the partner last moved.

WHY last_moved_min_ago MATTERS
    It is the system fact that the escalation policy layer checks a
    DP_MOVEMENT_ISSUE claim against. The LLM can believe an order is stuck;
    this number decides whether it actually is. That verification is what
    stopped containment swinging between 80% and 50% day to day.
===============================================================================
"""

from __future__ import annotations

from typing import Any

from ..db import get_db

# A partner stationary for longer than this is genuinely stuck, not at a light.
STUCK_THRESHOLD_MIN = 12

# Past the promised ETA by more than this counts as a real delay.
LATE_THRESHOLD_MIN = 10


class TrackingBackend:
    def get(self, order_id: str) -> dict[str, Any] | None:
        """Live tracking for an in-flight order, or None if there is none."""
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM delivery_tracking WHERE order_id = ?", (order_id,)
            ).fetchone()
        if row is None:
            return None

        tracking = dict(row)
        # Derived flags, computed here so every caller agrees on the definition.
        tracking["is_stuck"] = (tracking["last_moved_min_ago"] or 0) >= STUCK_THRESHOLD_MIN
        tracking["minutes_late"] = max(
            0, (tracking["eta_minutes"] or 0) - (tracking["promised_eta_min"] or 0)
        )
        tracking["is_late"] = tracking["minutes_late"] >= LATE_THRESHOLD_MIN
        return tracking
