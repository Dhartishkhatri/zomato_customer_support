"""
===============================================================================
BACKEND ADAPTERS - the seam where real systems plug in       [CORE structure]
===============================================================================

WHAT THIS DOES
    Bundles the four data sources the agent needs: orders, payments, CRM and
    the policy knowledge base.

WHY IT EXISTS
    Each adapter is a plain class with a narrow method surface, so swapping a
    mock for a real service - or for an MCP server exposing the same
    operations - is a one-class change. The orchestrator and the policy engine
    never learn that anything moved.

IS IT CRUCIAL?
    The *structure* is. The mock data inside is demo scaffolding: replace
    OrdersBackend and friends with real clients and delete data/*.json.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

from .crm import CRMBackend
from .kb import KnowledgeBase
from .orders import OrdersBackend
from .payments import PaymentsBackend
from .tracking import TrackingBackend


@dataclass
class Backends:
    orders: OrdersBackend
    payments: PaymentsBackend
    crm: CRMBackend
    kb: KnowledgeBase
    tracking: TrackingBackend

    @classmethod
    def mock(cls) -> "Backends":
        """Wire up the SQLite-backed adapters used by the demo, UI and tests.

        Creates and seeds the database on first use, so a fresh checkout works
        with no setup step.
        """
        from ..db import ensure_db

        ensure_db()
        orders = OrdersBackend()
        return cls(
            orders=orders,
            payments=PaymentsBackend(orders),
            crm=CRMBackend(),
            kb=KnowledgeBase(),
            tracking=TrackingBackend(),
        )


__all__ = [
    "Backends", "CRMBackend", "KnowledgeBase", "OrdersBackend",
    "PaymentsBackend", "TrackingBackend",
]
