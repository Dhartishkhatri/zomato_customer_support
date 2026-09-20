"""
===============================================================================
WEB APP - chat UI + ops dashboard                                    [CORE]
===============================================================================

    pip install -r requirements.txt
    python webapp.py            # then open http://127.0.0.1:8000

WHAT IT SERVES
    /                  the chat interface and dashboard (static/index.html)
    /api/session       start a chat as one of the seeded customers
    /api/chat          one turn through the Zia pipeline
    /api/action        execute an action the customer confirmed in the popup
    /api/rating        record a CSAT score for the chat
    /api/dashboard     containment, CSAT, latency, cost, quality
    /api/digest        the weekly-digest summary

WHY /api/chat IS A PLAIN POST AND NOT A STREAM
    The reply comes back as a structured object - prose plus an escalation
    tag - and the escalation tag has to be verified against order data BEFORE
    the customer sees the text, because verification can change what we say.
    Streaming tokens would mean showing the reply before it was cleared, so
    the UI shows a typing indicator instead and the turn lands atomically.

NOTE ON AUTH
    There is none. Session ownership is taken from the request. That is fine
    for a local demo and unacceptable in production - put real authentication
    in front of every /api route before this touches a real customer.
===============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zomato_support import evaluation, metrics
from zomato_support.backends import Backends
from zomato_support.db import ensure_db, get_db
from zomato_support.pipeline import Zia

STATIC = Path(__file__).resolve().parent / "static"

ensure_db()
backends = Backends.mock()
zia = Zia(backends=backends)

app = FastAPI(title="Zia - Zomato Support Agent")


# --- request bodies ---------------------------------------------------------
class SessionRequest(BaseModel):
    customer_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ActionRequest(BaseModel):
    session_id: str
    action: str
    order_id: str
    parameter: str | None = None


class RatingRequest(BaseModel):
    session_id: str
    rating: int
    comment: str = ""


# --- routes -----------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/customers")
def customers() -> list[dict[str, Any]]:
    """The seeded customers, for the demo's account picker."""
    out = []
    for customer in backends.crm.all_customers():
        orders = backends.orders.for_customer(customer["customer_id"])
        out.append({
            **customer,
            "order_count": len(orders),
            "live_orders": [
                {
                    "order_id": o["order_id"],
                    "restaurant": o["restaurant"],
                    "status": o["status"],
                    "total": o["total"],
                }
                for o in orders
                if o["status"] not in ("delivered", "cancelled")
            ],
        })
    return out


@app.get("/api/orders/{customer_id}")
def orders(customer_id: str) -> list[dict[str, Any]]:
    """Sidebar context: what this customer actually has on their account."""
    return [
        {
            "order_id": o["order_id"],
            "restaurant": o["restaurant"],
            "status": o["status"],
            "total": o["total"],
            "placed_minutes_ago": o.get("placed_minutes_ago"),
            "hours_since_delivery": o.get("hours_since_delivery"),
            "items": [i["name"] for i in o["items"]],
        }
        for o in backends.orders.for_customer(customer_id)
    ]


@app.post("/api/session")
def start_session(body: SessionRequest) -> dict[str, str]:
    try:
        backends.crm.get_customer(body.customer_id)
    except KeyError:
        raise HTTPException(404, f"Unknown customer {body.customer_id}")
    return {"session_id": zia.start_session(body.customer_id)}


@app.post("/api/chat")
def chat(body: ChatRequest) -> dict[str, Any]:
    """One full turn through the pipeline."""
    if not body.message.strip():
        raise HTTPException(400, "Message is empty.")
    try:
        result = zia.handle(body.session_id, body.message)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:  # noqa: BLE001 - surface the cause to the UI
        raise HTTPException(500, f"{type(exc).__name__}: {exc}")

    payload = result.to_dict()
    # The dashboard's "context used" panel - shows exactly what the model saw,
    # which is the fastest way to debug a wrong answer.
    payload["context_used"] = result.context_used
    return payload


@app.post("/api/action")
def action(body: ActionRequest) -> dict[str, Any]:
    """Execute an action the customer confirmed. Re-verified server-side."""
    try:
        return zia.execute_action(body.session_id, body.action, body.order_id, body.parameter)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/rating")
def rating(body: RatingRequest) -> dict[str, Any]:
    if not 1 <= body.rating <= 5:
        raise HTTPException(400, "Rating must be 1-5.")
    return backends.crm.rate_session(body.session_id, body.rating, body.comment)


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    return metrics.dashboard()


@app.get("/api/digest")
def digest() -> dict[str, Any]:
    return evaluation.weekly_digest()


@app.get("/api/transcript/{session_id}")
def transcript(session_id: str) -> dict[str, Any]:
    with get_db() as conn:
        turns = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM turn_metrics WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        ]
    return {"messages": backends.crm.history(session_id, limit=100), "turns": turns}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn

    print("\n  Zia is running at http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
