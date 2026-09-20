"""
===============================================================================
CRM BACKEND - customers, tickets, outbound email, chat store         [CORE]
===============================================================================

Replace with your real CRM client. The idea worth carrying over is
INTERNAL_ONLY_FIELDS below.
===============================================================================
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from ..db import get_db

# Profile fields the POLICY ENGINE may read but the MODEL may never see.
#
# A deliberate split: the gate needs trust_score and refund history to decide
# whether to escalate; the model does not, and anything in the model's context
# can end up paraphrased into a customer-facing reply. Stripping these at the
# tool boundary removes the risk rather than relying on model discretion.
INTERNAL_ONLY_FIELDS = frozenset({"trust_score", "notes", "refunds_last_90d"})


class CRMBackend:
    # --- customers ----------------------------------------------------------
    def get_customer(self, customer_id: str) -> dict[str, Any]:
        """Raises KeyError for an unknown id - the tool executor catches it."""
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
            ).fetchone()
        if row is None:
            raise KeyError(customer_id)
        return dict(row)

    def all_customers(self) -> list[dict[str, Any]]:
        """Every customer's id, name and email.

        Used by the guardrail's third-party-PII check, which has to know whose
        names must NOT appear in a reply. Returns only the identity columns -
        never the internal risk fields.
        """
        with get_db() as conn:
            rows = conn.execute(
                "SELECT customer_id, name, email FROM customers"
            ).fetchall()
        return [dict(r) for r in rows]

    # --- tickets ------------------------------------------------------------
    def update_ticket(
        self,
        ticket_id: str,
        status: str,
        summary: str,
        tags: list[str] | None = None,
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(tags or [])
        with get_db() as conn:
            conn.execute(
                "INSERT INTO tickets (ticket_id, customer_id, status, summary, tags, "
                "updated_at) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(ticket_id) DO UPDATE SET status=excluded.status, "
                "summary=excluded.summary, tags=excluded.tags, updated_at=excluded.updated_at",
                (ticket_id, customer_id, status, summary, payload, now),
            )
        return {"ticket_id": ticket_id, "status": status, "tags": tags or [], "updated_at": now}

    def escalate(
        self,
        ticket_id: str,
        queue: str,
        reason: str,
        context: dict[str, Any] | None = None,
        reason_code: str | None = None,
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        case_id = f"ESC-{uuid.uuid4().hex[:6].upper()}"
        now = datetime.now(timezone.utc).isoformat()
        with get_db() as conn:
            conn.execute(
                "INSERT INTO tickets (ticket_id, customer_id, status, summary, "
                "escalation_queue, escalation_case, escalation_reason_code, updated_at) "
                "VALUES (?,?,'escalated',?,?,?,?,?) "
                "ON CONFLICT(ticket_id) DO UPDATE SET status='escalated', "
                "summary=excluded.summary, escalation_queue=excluded.escalation_queue, "
                "escalation_case=excluded.escalation_case, "
                "escalation_reason_code=excluded.escalation_reason_code, "
                "updated_at=excluded.updated_at",
                (ticket_id, customer_id, reason, queue, case_id, reason_code, now),
            )
        return {
            "case_id": case_id,
            "queue": queue,
            "reason": reason,
            "reason_code": reason_code,
            "context": context or {},
            "raised_at": now,
        }

    # --- email --------------------------------------------------------------
    def send_email(self, customer_id: str, subject: str, body: str) -> dict[str, Any]:
        customer = self.get_customer(customer_id)
        return {
            "message_id": f"MSG-{uuid.uuid4().hex[:6].upper()}",
            "to": customer["email"],
            "subject": subject,
            "body": body,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }

    # --- conversation store -------------------------------------------------
    def start_session(
        self, session_id: str, customer_id: str, ticket_id: str, channel: str = "chat"
    ) -> None:
        with get_db() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO chat_sessions (session_id, customer_id, "
                "ticket_id, channel) VALUES (?,?,?,?)",
                (session_id, customer_id, ticket_id, channel),
            )

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, content) VALUES (?,?,?)",
                (session_id, role, content),
            )

    def history(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Oldest-first message history for prompt construction."""
        with get_db() as conn:
            rows = conn.execute(
                "SELECT role, content FROM chat_messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def session(self, session_id: str) -> dict[str, Any] | None:
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return dict(row) if row else None

    def close_session(self, session_id: str, contained: bool) -> None:
        with get_db() as conn:
            conn.execute(
                "UPDATE chat_sessions SET ended_at = CURRENT_TIMESTAMP, contained = ? "
                "WHERE session_id = ?",
                (1 if contained else 0, session_id),
            )

    def rate_session(self, session_id: str, rating: int, comment: str = "") -> dict[str, Any]:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO chat_ratings (session_id, rating, comment) VALUES (?,?,?)",
                (session_id, int(rating), comment),
            )
        return {"session_id": session_id, "rating": int(rating)}
