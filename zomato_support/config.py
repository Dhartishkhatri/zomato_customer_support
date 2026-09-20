"""
===============================================================================
CONFIGURATION - every tunable number in one place                    [CORE]
===============================================================================

WHY ONE FILE
    The policy engine and the forensics pipeline both make decisions that can
    cost money or wrongly accuse a customer. Keeping every threshold here means
    changing one is a config review, not a code review - and means you can see
    the whole risk posture on one screen.

WHAT TO TUNE FIRST
    The refund thresholds are business policy - set them from your real refund
    rules. The forensics thresholds are calibrated on three synthetic sample
    images and genuinely need re-tuning against real customer photos.
===============================================================================
"""

from __future__ import annotations

import os

# --- Models: the tiering policy ---------------------------------------------
# Big model where a wrong answer derails the whole turn, small model where it
# only costs a slightly clumsier sentence. See llm.py for the reasoning.
#
#   CLASSIFIER -> picks which data to fetch and which action to offer.
#                 Getting this wrong sends the turn down the wrong path.
#   CHAT       -> writes the customer-facing reply and narrates JSON. Sits in
#                 the user's latency budget, so speed wins.
#   JUDGE      -> offline evaluation; accuracy matters, latency does not.
#   VISION     -> image authenticity; a wrong call here can accuse a customer.
#
# Opus 5 runs adaptive thinking by default, so we deliberately never pass a
# `thinking` parameter anywhere in this codebase.
ORCHESTRATOR_MODEL = "claude-opus-5"
CLASSIFIER_MODEL = "claude-opus-5"
CHAT_MODEL = "claude-haiku-4-5"
JUDGE_MODEL = "claude-opus-5"
VISION_MODEL = "claude-opus-5"

MAX_TOKENS = 16_000
EFFORT = "high"

# Customer-facing replies are capped hard. Response time scales roughly
# linearly with output tokens, and this is the number that protects the
# sub-10-second target.
CHAT_MAX_TOKENS = 600

# Target for a full turn, used by the dashboard to flag slow responses.
TARGET_RESPONSE_MS = 10_000

# Hard ceiling on orchestrator turns. Without this, a confused loop can call
# tools until it runs out of money. This is a cost guard, not a policy.
MAX_AGENT_TURNS = 12

CURRENCY_SYMBOL = "₹"  # Indian rupee

# --- Refund policy: business rules, set these from your real policy ---------
# At or below this, a well-evidenced refund needs no human at all.
AUTO_REFUND_CEILING = 500

# Above this, a human always signs off, no matter how good the evidence is.
HUMAN_APPROVAL_THRESHOLD = 1_000

# Refund requests later than this after delivery are not auto-processable.
REFUND_WINDOW_HOURS = 48

# More refunds than this in the trailing window routes to fraud review.
REFUND_VELOCITY_LIMIT = 3
REFUND_VELOCITY_WINDOW_DAYS = 90

# An account whose trust score is below this never gets an automated refund.
MIN_TRUST_SCORE_FOR_AUTO_REFUND = 0.45

# --- Image forensics: re-tune these against real photos ---------------------
# Aggregate manipulation score below this reads as a normal camera photo.
# Anything above it (but without vision confirmation) becomes `inconclusive`,
# which means a human looks - never an accusation.
FORENSICS_AUTHENTIC_BELOW = 0.35

# A refund resting on a photo needs a verdict at least this confident before
# it can settle automatically.
MIN_IMAGE_CONFIDENCE_FOR_AUTO_REFUND = 0.70

# Set ZOMATO_AGENT_OFFLINE_FORENSICS=1 to skip the Claude vision judge and run
# local signals only. Useful for tests and for running without an API key, but
# materially weaker: a metadata-stripped AI image will likely pass.
OFFLINE_FORENSICS = os.environ.get("ZOMATO_AGENT_OFFLINE_FORENSICS") == "1"

# --- Guardrails -------------------------------------------------------------
MAX_REPLY_CHARS = 1_800

# Phrases the agent must never send: commitments the policy cannot honour, and
# anything that breaks the "write as a person at Zomato" rule.
BANNED_PHRASES = (
    "guaranteed",
    "we promise",
    "lifetime free",
    "unlimited refunds",
    "100% refund every time",
    "i am an ai",
    "as an ai language model",
)


# --- Evaluation -------------------------------------------------------------
# Fraction of turns graded automatically by the LLM judge. Zomato grade every
# system change plus a sample of live traffic; this is the live-traffic rate.
EVAL_SAMPLE_RATE = 0.25

# A turn scoring below this on any judge metric is surfaced in the weekly
# digest as needing human review.
EVAL_ALERT_THRESHOLD = 0.6
