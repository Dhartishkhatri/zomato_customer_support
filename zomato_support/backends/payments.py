"""
===============================================================================
PAYMENTS (mock)                                                      [CORE]
===============================================================================

The only code path in the system that moves money.

WHY IT VALIDATES ITS OWN INPUTS
    The policy gate already checks amounts before we get here - but a gate is
    a policy layer, not a substitute for a backend enforcing its own
    invariants. If someone later calls this directly, or adds a code path that
    skips the gate, these checks are the last line. Defence in depth.
===============================================================================
"""

from __future__ import annotations

import uuid
from typing import Any

from .orders import OrdersBackend

_SETTLEMENT_DAYS = {
    "UPI": "3-5 business days",
    "Credit Card": "5-7 business days",
    "Debit Card": "5-7 business days",
    "Zomato Wallet": "instantly",
}


class PaymentsBackend:
    def __init__(self, orders: OrdersBackend) -> None:
        self._orders = orders

    def issue_refund(self, order_id: str, amount: int, reason: str) -> dict[str, Any]:
        order = self._orders.get(order_id)
        already = self._orders.refunded_total(order_id)

        if amount <= 0:
            raise ValueError("Refund amount must be positive.")
        if already + amount > order["total"]:
            raise ValueError(
                f"Refund of {amount} exceeds refundable balance "
                f"({order['total'] - already}) on order {order_id}."
            )

        method = order["payment_method"]
        refund = {
            "refund_id": f"RFD-{uuid.uuid4().hex[:6].upper()}",
            "order_id": order_id,
            "amount": amount,
            "currency": order["currency"],
            "reason": reason,
            # Wallet refunds land instantly; card and UPI settle later, so the
            # agent must not tell the customer the money has already arrived.
            "status": "completed" if method == "Zomato Wallet" else "pending",
            "destination": method,
            "settles_in": _SETTLEMENT_DAYS.get(method, "5-7 business days"),
        }
        self._orders.record_refund(order_id, refund)
        return refund
