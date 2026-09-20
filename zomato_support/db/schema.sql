-- =============================================================================
-- ZOMATO SUPPORT AGENT - DUMMY DATABASE SCHEMA
-- =============================================================================
-- SQLite because it needs no server and ships in the stdlib. Every table here
-- maps to something a real deployment would read from a different system:
--
--   customers, orders, order_items   -> Orders service
--   delivery_tracking                -> Logistics / delivery-partner service
--   refunds                          -> Payments service
--   tickets                          -> CRM
--   chat_sessions, chat_messages     -> Conversation store
--   chat_ratings                     -> CSAT collection
--   action_log                       -> Audit of executed actions
--   turn_metrics                     -> Containment / latency / cost telemetry
--   eval_results                     -> LLM-as-judge + human grading
--
-- Times are stored as offsets ("delivered_hours_ago") rather than absolute
-- timestamps so the fixtures never go stale and the 48-hour refund window
-- stays testable forever.
-- =============================================================================

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS customers (
    customer_id      TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    email            TEXT NOT NULL,
    phone_last4      TEXT,
    city             TEXT,
    member_since     TEXT,
    gold_member      INTEGER DEFAULT 0,
    -- Internal risk fields. The policy gate reads these; the model never sees
    -- them (see backends/crm.py INTERNAL_ONLY_FIELDS).
    trust_score      REAL DEFAULT 1.0,
    lifetime_orders  INTEGER DEFAULT 0,
    refunds_last_90d INTEGER DEFAULT 0,
    notes            TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    order_id              TEXT PRIMARY KEY,
    customer_id           TEXT NOT NULL REFERENCES customers(customer_id),
    restaurant            TEXT NOT NULL,
    restaurant_phone      TEXT,
    -- placed | confirmed | preparing | picked_up | out_for_delivery
    -- | delivered | cancelled
    status                TEXT NOT NULL,
    delivered_hours_ago   REAL,
    placed_minutes_ago    INTEGER,
    total                 INTEGER NOT NULL,
    currency              TEXT DEFAULT 'INR',
    payment_method        TEXT,
    payment_ref           TEXT,
    delivery_address      TEXT,
    delivery_instructions TEXT,
    -- Drives the cancellation-eligibility rules in the action pipeline.
    cancellable_until_min INTEGER DEFAULT 60
);

CREATE TABLE IF NOT EXISTS order_items (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL REFERENCES orders(order_id),
    name     TEXT NOT NULL,
    qty      INTEGER NOT NULL DEFAULT 1,
    price    INTEGER NOT NULL
);

-- Live logistics state. This is what "where is my order?" actually reads.
CREATE TABLE IF NOT EXISTS delivery_tracking (
    order_id           TEXT PRIMARY KEY REFERENCES orders(order_id),
    partner_name       TEXT,
    partner_phone      TEXT,
    partner_rating     REAL,
    vehicle            TEXT,
    distance_km        REAL,
    eta_minutes        INTEGER,
    promised_eta_min   INTEGER,
    last_moved_min_ago INTEGER,   -- Verifies DP_MOVEMENT_ISSUE escalations.
    current_area       TEXT,
    tracking_url       TEXT
);

CREATE TABLE IF NOT EXISTS refunds (
    refund_id   TEXT PRIMARY KEY,
    order_id    TEXT NOT NULL REFERENCES orders(order_id),
    amount      INTEGER NOT NULL,
    currency    TEXT DEFAULT 'INR',
    reason      TEXT,
    status      TEXT NOT NULL,          -- pending | completed | failed
    destination TEXT,
    settles_in  TEXT,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id        TEXT PRIMARY KEY,
    customer_id      TEXT REFERENCES customers(customer_id),
    status           TEXT,
    summary          TEXT,
    tags             TEXT,
    escalation_queue TEXT,
    escalation_case  TEXT,
    escalation_reason_code TEXT,
    updated_at       TEXT DEFAULT CURRENT_TIMESTAMP
);

-- --- Conversation store -----------------------------------------------------
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id  TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    ticket_id   TEXT,
    channel     TEXT DEFAULT 'chat',
    started_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    ended_at    TEXT,
    contained   INTEGER              -- 1 = bot handled it, 0 = escalated
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES chat_sessions(session_id),
    role       TEXT NOT NULL,        -- customer | agent | system
    content    TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_ratings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES chat_sessions(session_id),
    rating     INTEGER NOT NULL,     -- 1..5 CSAT
    comment    TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- --- Actions the bot took, and whether a human confirmed them ---------------
CREATE TABLE IF NOT EXISTS action_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT REFERENCES chat_sessions(session_id),
    order_id     TEXT,
    action_type  TEXT NOT NULL,
    payload      TEXT,
    proposed     INTEGER DEFAULT 1,
    verified     INTEGER DEFAULT 0,   -- passed the eligibility check
    confirmed    INTEGER DEFAULT 0,   -- user accepted the popup
    executed     INTEGER DEFAULT 0,
    reject_reason TEXT,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);

-- --- Telemetry: one row per assistant turn ---------------------------------
CREATE TABLE IF NOT EXISTS turn_metrics (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         TEXT REFERENCES chat_sessions(session_id),
    latency_ms         INTEGER,
    classify_ms        INTEGER,
    generate_ms        INTEGER,
    action_ms          INTEGER,
    input_tokens       INTEGER DEFAULT 0,
    output_tokens      INTEGER DEFAULT 0,
    cost_usd           REAL DEFAULT 0,
    escalated          INTEGER DEFAULT 0,
    escalation_reason  TEXT,
    escalation_upheld  INTEGER,       -- did the policy layer agree?
    functions_called   TEXT,
    action_proposed    TEXT,
    guardrail_blocked  INTEGER DEFAULT 0,
    created_at         TEXT DEFAULT CURRENT_TIMESTAMP
);

-- --- Evaluation: LLM-as-judge and human grading ----------------------------
CREATE TABLE IF NOT EXISTS eval_results (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id           TEXT REFERENCES chat_sessions(session_id),
    turn_id              INTEGER,
    grader               TEXT NOT NULL,      -- llm_judge | human
    factual_consistency  REAL,
    information_relevance REAL,
    guideline_compliance REAL,
    pain_point_resolution REAL,
    notes                TEXT,
    created_at           TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_orders_customer  ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_items_order      ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_refunds_order    ON refunds(order_id);
CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_metrics_session  ON turn_metrics(session_id);
