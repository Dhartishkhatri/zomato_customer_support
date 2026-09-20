# Zia — Zomato Support Agent

A conversational support agent for a food delivery platform, built on Claude.
It understands a concern typed in plain language, fetches only the data that
concern needs, offers concrete actions (cancel, add delivery instructions,
change address) behind a confirmation step, and decides for itself when a
human should take over — with every one of those decisions checked against
real order data rather than taken on the model's word.

It also checks whether photos submitted against refund claims are genuine
camera captures, edited images, or AI-generated.

```
  customer message
        |
  [1] CLASSIFY FUNCTIONS ........ Opus 5 — which data does this turn need?
        |
  [2] FETCH + NARRATE ........... SQLite + templates. JSON → prose. No model.
        |
        +---------------- run concurrently ----------------+
        |                                                  |
  [3a] GENERATE REPLY (Haiku 4.5)          [3b] DECIDE ACTION (Opus 5)
       reply text + escalation tag              classify, then VERIFY
        |                                                  |
        +---------------- join ----------------------------+
        |
  [4] VERIFY ESCALATION ......... reason code checked against order data
        |
  [5] GUARDRAIL ................. deterministic pre-send checks
        |
  [6] PERSIST ................... message, metrics, proposed action
        |
  reply + optional confirmation popup
```

## Quick start

```bash
pip install -r requirements.txt
python -m zomato_support.db.seed          # build the dummy database
python -m pytest tests/ -q                # 154 tests, no API key needed

export ANTHROPIC_API_KEY=sk-ant-...       # needed for live chat only
python webapp.py                          # → http://127.0.0.1:8000
```

The UI has two tabs: **Chat** (pick a seeded customer, talk to Zia, confirm
actions in the popup, rate the chat) and **Dashboard** (containment, CSAT,
latency, cost, judge scores). A right-hand panel shows, per turn, which data
functions fired, what the escalation policy decided, and what the reply cost.

Also runnable standalone:

```bash
python mcp_server.py                                        # MCP server over stdio
python verify_image.py samples/morphed_curry.jpg --offline  # forensics, no key
```

## The four requirements, and where each one lives

| Requirement | Implementation |
|---|---|
| **Conversational, not transactional** | `pipeline/zia.py` — one free-text message in, one reply out. No menus, no button trees. |
| **Deep intent understanding** | `pipeline/function_classifier.py` extracts intent and which data is needed; `action_decisioning.py` reads "I'll be in a meeting, leave it at the door" as an add-instructions request. |
| **Action-oriented** | `action_decisioning.py` proposes, verifies, and the UI confirms. Cancel, add instructions, change address, contact rider, request refund. |
| **Sub-10s, contained, cheap** | Model tiering (`llm.py`), token reduction via function classification, capped output tokens, parallel generate + action. Measured per turn in `metrics.py`. |

## The two verification systems

These are the heart of it, and both exist for the same reason: **an LLM is
good at understanding intent and bad at knowing what it is allowed to do.**

### Action decisioning — two steps

```
STEP 1  CLASSIFY (Opus 5)   what action does this conversation call for?
STEP 2  VERIFY   (Python)   is that action actually permitted right now?
```

Step 1 without step 2 is a bot that cheerfully cancels orders that left the
restaurant twenty minutes ago. Step 2 is pure code against the database —
never a model — because "can this be cancelled?" is a fact, not a judgement.

Nothing there executes anything. A proposal becomes a confirmation popup; the
customer accepts; the server **re-verifies** (the popup was rendered from a
decision made seconds ago, and the request arrives over HTTP where anything
could have been changed); only then does it run.

### Escalation policy layer — the containment stabiliser

Left to the model alone, containment swings wildly — Zomato saw 80% to 50%
day to day, which makes staffing impossible. The fix:

> The model may ask to escalate, but must tag it with a **reason code** from a
> fixed list. The system checks that code against real order data.

```
LLM says:      "escalate — DP_MOVEMENT_ISSUE"
System asks:   has the rider actually stopped moving?
Database says: last moved 2 minutes ago, 3.8 km out
Result:        REJECTED — chat stays contained, bot keeps helping
```

Same code, different order, opposite outcome:

| Order | Rider state | `DP_MOVEMENT_ISSUE` |
|---|---|---|
| `ORD-89044` | stationary 23 min | **upheld** → delivery-investigations |
| `ORD-89077` | moved 2 min ago | **rejected** → stays contained |

Containment becomes a property of the *data*, not of the model's mood.

**This is not a cost-saving hack.** Six codes bypass verification entirely and
always escalate: illness or injury, legal and press threats, abuse, delivery
partner misconduct, suspected image fraud, and any explicit request for a
human. Suppressing one of those to protect a metric would be the worst
possible outcome, so they are hard-coded as auto-upheld.

## Token efficiency: fetch only what the question needs

"Where's my order?" fetches live tracking. "What was in it?" fetches the
itemised bill. Neither fetches both. Then the JSON becomes prose:

```
Sunita is delivering order ORD-89044 by scooter, currently around Sony World
Junction, about 1.1 km away. The estimated arrival is 26 minutes. The rider
has not moved for 23 minutes, which suggests they are held up.
```

**The narration layer is templated, not model-generated**, and that is a
control rather than an optimisation. It decides what the model is even *able*
to say:

- The courier's surname and phone never enter the prompt, so they cannot leak.
- Lateness is pre-computed into words, so the model never does arithmetic on
  timestamps (which it does badly).
- Internal risk fields are absent by construction.

It is also free and instant — an extra model call per lookup would be the
slowest thing in the pipeline.

## Model tiering

| Task | Model | Why |
|---|---|---|
| Function classification | Opus 5 | A wrong answer derails the whole turn |
| Action classification | Opus 5 | Decides what the customer is offered |
| **Chat completion** | **Haiku 4.5** | Sits in the user's latency budget |
| JSON → prose | Haiku 4.5 | Templated by default; this is the fallback path |
| LLM-as-judge | Opus 5 | It grades Haiku's work, so it must be stronger |
| Image forensics | Opus 5 | A wrong call can accuse a customer |

The same lesson Zomato learned with 70B/8B, mapped onto Claude. Per-call cost
and latency are recorded in `llm.py` and surfaced on the dashboard.

## Metrics

Four numbers, on the dashboard and in `metrics.py`:

- **CSAT** — 1–5 star rating per chat
- **Containment %** — share of chats handled without a human
- **Cost** — dollars per turn and per chat, from real token counts
- **Response time** — p50/p95 against the 10-second target

Containment sits next to CSAT deliberately: it is trivially gameable by
refusing to escalate, so it is only meaningful read alongside satisfaction.
The dashboard also reports **how often the policy layer rejected an escalation
the model asked for** — a little means the stabiliser is working, a lot means
the model has learned to escalate reflexively and needs prompt work.

## Evaluation

`evaluation.py` — LLM-as-judge (Opus 5) scoring four dimensions 0–1:

| Metric | Catches |
|---|---|
| Factual consistency | Invented ETAs, amounts, order ids |
| Information relevance | Answering something adjacent to the question |
| Guideline compliance | Tone, promises, internal vocabulary, "I am an AI" |
| Pain point resolution | Politely saying nothing |

`EVAL_SAMPLE_RATE` controls the share of live turns graded. Human grades go
into the same table via `record_human_grade`, and `judge_vs_human()` reports
the mean gap per metric — if the judge drifts from humans, trust the humans
and re-tune the judge prompt. `weekly_digest()` produces the Slack summary.

## MCP server

`mcp_server.py` exposes the backends as MCP tools over stdio:

`get_order` · `list_orders` · `get_delivery_tracking` · `get_customer` ·
`search_policy` · `check_action_eligibility` · `check_escalation_reason` ·
`list_escalation_codes`

```json
{
  "mcpServers": {
    "zomato-support": {
      "command": "python",
      "args": ["C:/Users/Test/Desktop/customer_support_agent/mcp_server.py"]
    }
  }
}
```

**Every tool is read-only or a dry-run verifier.** Nothing issues a refund,
cancels an order or sends an email — a security decision, not an oversight.
MCP tools are callable by whatever client connects, which puts them outside
this project's permission gate, approval transports and audit trail.
Money-moving operations stay behind the gate in `policy.py`. The verifier
tools let a client *ask* whether an action would be permitted, which is the
useful half, without being able to perform it. The same PII rules apply:
tracking returns `partner_first_name`, never the surname or phone.

Note the SDK API: this targets `mcp` 2.x, where `FastMCP` was renamed to
`MCPServer` (`from mcp.server.mcpserver import MCPServer`).

## The dummy database

SQLite, seeded by `zomato_support/db/seed.py`. Ten orders chosen so every
branch has something to fire on:

| Order | State | Exercises |
|---|---|---|
| `ORD-89011` | placed 4 min ago | cancellation + instructions happy path |
| `ORD-89044` | rider stuck 23 min | `DP_MOVEMENT_ISSUE` **upheld** |
| `ORD-89077` | late, rider moving | `DP_MOVEMENT_ISSUE` **rejected** |
| `ORD-88213` | delivered 3h ago | small refund, auto-approved |
| `ORD-88455` | ₹1,240 delivered | over the supervisor threshold |
| `ORD-88601` | serial refunder | fraud-review escalation |
| `ORD-88777` | delivered 126h ago | outside the refund window |
| `ORD-88990` | already refunded | double-refund denial |

Schema in `zomato_support/db/schema.sql`, with comments mapping each table to
the real system it would come from. Tests get an isolated, freshly seeded
database per test via `tests/conftest.py`, because the DB now holds mutable
state that would otherwise leak between them.

## The image authenticity check

This is the part worth reading closely, because it is the part that can hurt a
customer if it is wrong. The design principle throughout: **a false accusation
of fraud costs far more than a manual review**, so the pipeline is built to
answer "I don't know, send it to a human" rather than to accuse.

Two halves feed one verdict.

**Provenance** (`forensics/provenance.py`) — near-evidence rather than
inference. Generative-tool watermarks in PNG text chunks and XMP (Stable
Diffusion, ComfyUI, Midjourney, DALL·E, Firefly), C2PA / Content Credentials
manifests declaring `trainedAlgorithmicMedia`, and EXIF `Software` tags naming
an image editor. A C2PA AI declaration or a Stable Diffusion `parameters` chunk
short-circuits the whole pipeline to `ai_generated` at 97% confidence.

**Pixel statistics** (`forensics/signals.py`) — four measurements, each scored
0–1 and weighted:

| Signal | What it catches | Weight |
|---|---|---|
| `error_level_analysis` | Spliced regions with a different compression history | 2.5 |
| `copy_move` | A region duplicated elsewhere in the same frame | 2.2 |
| `noise_consistency` | Sensor-noise floor that changes across the frame | 1.8 |
| `generative_geometry` | Stock model output sizes with no camera metadata | 1.4 |

Signals combine by **noisy-OR**, not by weighted mean. A mean would let one
damning signal average away against the quiet ones, and would mean that
adding more benign checks silently weakened the detector. Two further
detectors were removed after measurement - see "What was removed" below.

**Claude Opus 5 vision** (`forensics/vision.py`) sees the image *and* the
forensic numbers, and returns a structured judgement via `messages.parse`. Its
job is the one statistics cannot do: deciding whether an anomaly has an
innocent explanation. A high-ELA cluster over an oily curry surface is a
specular highlight; the same cluster with a hard rectilinear boundary is a
paste. It also reports what the photo actually depicts and whether that matches
the customer's complaint — a pristine meal sent against a "food was spilled"
claim matters even when the image is perfectly genuine.

### Verdicts and what they do

| Verdict | Effect on a refund |
|---|---|
| `authentic` | Claim proceeds on its merits |
| `inconclusive` | Supervisor approval required; customer is told only that it is under review |
| `manipulated` / `ai_generated` | No automated refund; Trust & Safety owns the case |

Two safety properties are enforced in code and covered by tests:

1. **Local statistics alone can never reach an accusatory verdict.** Without
   the vision judge the pipeline caps at `inconclusive`. Only provenance
   evidence or Claude's visual confirmation of a specific, named defect can
   produce `manipulated` or `ai_generated`.
2. **Missing metadata is not evidence.** WhatsApp, Instagram and every other
   chat app strip EXIF from what they forward, so "no EXIF" is the *normal*
   case for a support photo. It is scored at 0.18 with weight 0.8 — barely
   enough to register.

### Measured behaviour on the sample images

| Image | Score | Verdict (local only) |
|---|---|---|
| `real_curry.jpg` | 0.07 | `authentic` |
| `morphed_curry.jpg` | 0.49 | `inconclusive` → human review |
| `ai_generated.png` | 1.00 | `ai_generated` (provenance) |

## The permission gate

`policy.py` evaluates every write call before it reaches a backend, returning
`ALLOW`, `REQUIRE_APPROVAL`, `ESCALATE` or `DENY`. Rules are ordered
most-restrictive-first so a `DENY` can never be masked by a later `ALLOW`.

The gate **never asks the model whether an action is allowed**. It re-derives
the facts from the backends. If the model claims an order is refundable and the
orders DB disagrees, the DB wins.

Selected rules:

| Rule | Decision | Condition |
|---|---|---|
| `R02-unverified-order` | DENY | Refunding an order never fetched with `get_order` |
| `R03-cross-account` | DENY | Order belongs to another customer |
| `R04-not-delivered` | DENY | Order is still out for delivery |
| `R05/R06` | DENY | Already refunded, or amount exceeds the order |
| `R07-illness` | ESCALATE | Illness, injury or foreign object → food-safety queue |
| `R08-legal` | ESCALATE | Legal, regulatory or press threat |
| `R09-window-expired` | ESCALATE | Past the 48-hour window → Grievance Officer |
| `R11-image-manipulated` | ESCALATE | Photo is edited or synthetic → Trust & Safety |
| `R12-image-inconclusive` | REQUIRE_APPROVAL | Photo cannot be verified either way |
| `R14-refund-velocity` | ESCALATE | 3+ refunds in 90 days |
| `R16/R17` | REQUIRE_APPROVAL | Over the ₹1,000 supervisor threshold / ₹500 auto ceiling |
| `R18-auto-approved` | ALLOW | Evidenced, in-window, in-band, customer in good standing |

Approval transports live in `approvals.py`: `QueueApprover` (default — files
the request and tells the customer it is with a supervisor) and `AutoApprover`
(tests and `--approve-all` only). To wire up a real channel, write a class
with one `request(...) -> bool` method and pass `SupportAgent(approver=...)`.

## The guardrail check

`guardrails.py` runs on the drafted reply before a character reaches the
customer. It is deliberately deterministic — a check that is itself a model
call can be talked out of its job by the same context that produced the bad
draft.

Blocking violations: internal forensic vocabulary ("forensic", "manipulated",
"AI generated", "trust score", "fraud", "EXIF"), another customer's name or
email, a delivery partner's surname, a money figure that appears in no tool
result, a refund asserted when no refund tool call succeeded, a pending
approval presented as settled, and prohibited promises.

On a block, the correction goes back as a **mid-conversation system message**
(`{"role": "system"}` inside `messages`), which carries operator authority and
leaves the cached prefix intact. The model gets exactly one rewrite. If the
second draft also fails, a neutral fallback is sent and a human is assigned.

## Layout

```
webapp.py            FastAPI app: chat + action + rating + dashboard APIs
static/index.html    the whole UI, one file, no build step
mcp_server.py        MCP server exposing the backends over stdio
verify_image.py      standalone image forensics CLI

zomato_support/
  llm.py             model tiering + per-call cost/latency tracking
  metrics.py         containment, CSAT, cost, response time
  evaluation.py      LLM-as-judge, human grading, weekly digest
  config.py          every tunable number in one place
  pipeline/
    zia.py                 THE AGENT. One turn, start to finish.
    function_classifier.py [1] which data does this turn need?
    data_functions.py      [2] targeted fetch + JSON -> prose
    action_decisioning.py  [3b] classify an action, then verify it
    escalation_policy.py   [4] check escalation tags against real data
  policy.py          permission gate for money-moving actions
  guardrails.py      deterministic pre-send checks
  orchestrator.py    the older tool-calling loop (refunds; see below)
  approvals.py       human-approval transports
  audit.py           append-only audit trail
  tools/             tool specs + gated dispatcher (used by orchestrator.py)
  backends/          orders, payments, CRM, tracking, BM25 knowledge base
  forensics/         provenance, signals, vision judge, aggregation
  db/                schema.sql + seed.py + connection helpers
tests/               154 tests, all offline
```

**Two agents, on purpose.** `pipeline/zia.py` handles chat: a fixed pipeline,
two model calls, one round trip, fast enough for a live conversation.
`orchestrator.py` is the older tool-calling loop, kept for refund workflows
where the permission gate, supervisor approvals and the audit trail matter
more than latency. The gate, guardrails and forensics are shared by both.

**Why the chat path is a pipeline, not an agentic loop:** a loop needs several
sequential model calls before it can say anything, and each one is a full
round trip. A pipeline classifies once, fetches once, and generates once —
which is what makes a 10-second budget achievable.

**Swapping in real backends:** each adapter in `backends/` is a plain class
with a narrow method surface over SQLite. Replacing one with a real service is
a one-class change — the pipeline and the policy engine do not move.

## Reading the code

Every module starts with a header block saying what it does, why it exists,
and whether it is `[CORE]`, `[DEMO only]` or replaceable. Read them in this
order:

| # | File | Why |
|---|---|---|
| 1 | `pipeline/zia.py` | The agent. One turn from message to reply, in one file. |
| 2 | `pipeline/escalation_policy.py` | The containment stabiliser. The most interesting idea here. |
| 3 | `pipeline/action_decisioning.py` | Two-step action verification: intent vs. permission. |
| 4 | `pipeline/data_functions.py` | Why the narration layer is templated and not model-generated. |
| 5 | `policy.py` | The permission gate for anything that moves money. |
| 6 | `guardrails.py` | What stops a bad sentence reaching a customer. |
| 7 | `forensics/pipeline.py` | How image signals become one verdict, and the safety stance. |

`schemas.py` is worth a look too — `ConversationContext` is the trust
boundary, and its docstring explains why.

### If you want to cut it down further

Safe to delete outright, in rough order:

| Delete | Cost |
|---|---|
| `run_demo.py`, `scripts/`, `samples/` | Nothing. Pure demo scaffolding. |
| `verify_image.py` | Nothing, but you lose the easiest way to debug an image verdict. |
| `mcp_server.py` | Nothing, unless you want external MCP clients. |
| `orchestrator.py` + `tools/` + `approvals.py` | Drops the tool-calling refund workflow. The chat pipeline is untouched. |
| `evaluation.py` | You stop measuring reply quality. Metrics and chat still work. |
| `backends/kb.py` + `data/kb/` | Agent stops citing policy; the grounding check gets much weaker. |
| `forensics/` (whole package) | Drops image authenticity entirely. Everything else works. |

Do **not** delete: `pipeline/escalation_policy.py`,
`pipeline/action_decisioning.py`, `policy.py`, `guardrails.py`,
`forensics/pipeline.py`, or the split between full and redacted image analysis
in `tools/image_tools.py`. Those are the parts that stop the system paying out
wrongly, promising the impossible, or accusing a customer.

## What was removed in the earlier trimming pass

Kept for the record. A trimming pass cut roughly 400 lines. The reasoning, so you can disagree with it:

**Two forensic detectors**, dropped on measured evidence rather than taste:

- `spectral_periodicity` (~55 lines of FFT) scored 0.000 / 0.000 / 0.081 on the
  three sample images. It never changed a verdict.
- `recompression_history` (~65 lines of JPEG-ghost analysis) scored
  0.057 / 0.036 / 0.000 after two rounds of false-positive fixes.

All three sample verdicts are **identical** after removal (0.065 authentic,
0.493 edited, 1.00 AI), and the test suite went from ~17.7s to ~10.4s.
`_LOCAL_SUSPICION` moved 0.55 -> 0.45 to keep the edited sample landing on
`inconclusive` - worth knowing, since that threshold is now doing real work.

This does leave a real conclusion worth stating: **the pixel statistics barely
move the needle. Provenance metadata and the Claude vision judge do nearly all
the real work.** If you want a better detector, invest there, not in more
signal functions.

**Dead code**, found by a reference scan: `config.CURRENCY`,
`FORENSICS_MANIPULATED_ABOVE`, `GUARDRAIL_MODEL`, `AuditLog.to_json`,
`AuditLog.__len__`, `PolicyResult.allowed`, `ConversationContext.kb_citations`
(written but never read), `READ_TOOLS` / `ESCALATION_TOOLS`, `__version__`, and
the `OrderNotFound` / `CustomerNotFound` subclasses (plain `KeyError` is what
the executor actually catches).

**`ConsoleApprover`** and its `--ask` flag. `QueueApprover` is the honest
production behaviour and `AutoApprover` covers the approved branch in tests.

## Limitations

Worth being straight about:

- **The live pipeline has not been run against the real API in this
  environment** — no credentials were available. Classification, generation,
  action decisioning, escalation verification, guardrails, persistence and
  metrics are covered end to end by `tests/test_zia_pipeline.py` with a stub
  client that mimics the SDK's `messages.parse` shape, and every request field
  follows the current Anthropic Python SDK (1.x) docs. But the first live run
  deserves a careful read of a few transcripts, especially the
  classifier's function selection.
- **The latency target is unverified.** `TARGET_RESPONSE_MS` is 10s and the
  dashboard measures against it, but the measured numbers so far come from
  stubbed calls. Real p95 depends on Opus classification latency, which is the
  longest call in the turn. If it misses, the first lever is dropping the
  action classifier to Sonnet, not shrinking the chat model further.
- **Prompt caching is not yet wired into the Zia path.** The tool-calling
  orchestrator uses it; the pipeline's system prompt is byte-stable and ready
  for a cache breakpoint but does not set one. Worth adding — the system
  prompt is resent on every turn.
- **No authentication on the web app.** Session ownership is taken from the
  request body. Fine for a local demo, unacceptable in production.
- **CSAT is self-reported on a demo UI**, so the dashboard's satisfaction
  number means very little until real users rate real chats.
- **The detector thresholds are calibrated against three synthetic sample
  images, not a real corpus.** The design is conservative, but the numbers in
  `config.py` need re-tuning against real customer photos, with the
  false-positive rate on genuine images as the metric that matters.
- **Statistical image detectors degrade on heavily compressed images.** A photo
  that has been through two chat apps has had its noise floor and JPEG grid
  largely destroyed. The pipeline answers `inconclusive`, which is correct but
  means chat-forwarded photos rely mostly on the vision judge.
- **Offline forensics mode is materially weaker.** With
  `ZOMATO_AGENT_OFFLINE_FORENSICS=1` an AI image whose metadata has been
  stripped will likely pass as authentic.
- **The knowledge base is BM25 over four policy files.** Fine for grounding a
  demo; a real deployment wants a vector store and far more policy coverage.
- **The action catalogue is five actions.** Adding one means a new entry in
  `action_decisioning.PROMPTS`, an eligibility branch in `verify_action`, and
  an execution branch in `Zia.execute_action` — deliberately three places, so
  no action can ship without an eligibility rule.
