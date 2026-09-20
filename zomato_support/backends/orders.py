"""
===============================================================================
ORDERS BACKEND - reads the SQLite dummy DB                           [CORE]
===============================================================================

Replace this class with your real orders client. The methods the rest of the
system depends on are: get(), refunded_total(), record_refund(), and the
action helpers set_delivery_instructions() / cancel().

Delivery times are stored as "hours ago" offsets and resolved to real
timestamps on read, so fixtures never go stale and the 48-hour refund window
stays testable forever.
===============================================================================
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ..db import get_db


class OrdersBackend:
    def get(self, order_id: str) -> dict[str, Any]:
        """Full order with items and refunds. Raises KeyError if unknown."""
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM orders WHERE order_id = ?", (order_id,)
            ).fetchone()
            if row is None:
                raise KeyError(order_id)

            order = dict(row)
            order["items"] = [
                dict(r)
                for r in conn.execute(
                    "SELECT name, qty, price FROM order_items WHERE order_id = ?",
                    (order_id,),
                ).fetchall()
            ]
            order["refunds"] = [
                dict(r)
                for r in conn.execute(
                    "SELECT refund_id, amount, reason, status FROM refunds "
                    "WHERE order_id = ?",
                    (order_id,),
                ).fetchall()
            ]

        hours = order.pop("delivered_hours_ago", None)
        if hours is None:
            order["delivered_at"] = None
            order["hours_since_delivery"] = None
        else:
            order["delivered_at"] = (
                datetime.now(timezone.utc) - timedelta(hours=hours)
            ).isoformat()
            order["hours_since_delivery"] = hours
        return order

    def for_customer(self, customer_id: str) -> list[dict[str, Any]]:
        """Most recent orders first - used to resolve 'my last order'."""
        with get_db() as conn:
            ids = [
                r["order_id"]
                for r in conn.execute(
                    "SELECT order_id FROM orders WHERE customer_id = ? "
                    "ORDER BY COALESCE(placed_minutes_ago, 999999) ASC",
                    (customer_id,),
                ).fetchall()
            ]
        return [self.get(i) for i in ids]

    def refunded_total(self, order_id: str) -> int:
        """Money already committed against this order.

        Counts pending as well as completed, so two refunds cannot race through
        while the first is still settling.
        """
        with get_db() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(amount), 0) AS total FROM refunds "
                "WHERE order_id = ? AND status IN ('completed', 'pending')",
                (order_id,),
            ).fetchone()
        return int(row["total"])

    def record_refund(self, order_id: str, refund: dict[str, Any]) -> None:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO refunds (refund_id, order_id, amount, currency, reason, "
                "status, destination, settles_in) VALUES (?,?,?,?,?,?,?,?)",
                (
                    refund["refund_id"], order_id, refund["amount"], refund["currency"],
                    refund["reason"], refund["status"], refund["destination"],
                    refund["settles_in"],
                ),
            )

    # --- action helpers, used by the action pipeline -----------------------
    def set_delivery_instructions(self, order_id: str, instructions: str) -> dict[str, Any]:
        with get_db() as conn:
            conn.execute(
                "UPDATE orders SET delivery_instructions = ? WHERE order_id = ?",
                (instructions, order_id),
            )
        return {"order_id": order_id, "delivery_instructions": instructions}

    def set_delivery_address(self, order_id: str, address: str) -> dict[str, Any]:
        with get_db() as conn:
            conn.execute(
                "UPDATE orders SET delivery_address = ? WHERE order_id = ?",
                (address, order_id),
            )
        return {"order_id": order_id, "delivery_address": address}

    def cancel(self, order_id: str) -> dict[str, Any]:
        with get_db() as conn:
            conn.execute(
                "UPDATE orders SET status = 'cancelled' WHERE order_id = ?", (order_id,)
            )
        return {"order_id": order_id, "status": "cancelled"}
