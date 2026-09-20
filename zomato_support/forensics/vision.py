"""
===============================================================================
CLAUDE OPUS 5 AS THE SECOND OPINION ON AUTHENTICITY                  [CORE]
===============================================================================

WHAT THIS DOES
    Sends the image AND the local forensic measurements to Claude, and gets
    back a structured judgement (via messages.parse, so the shape is
    guaranteed by the VisionJudgement model below).

WHY BOTH, RATHER THAN JUST THE IMAGE
    Because the model can then do the one thing statistics cannot: decide
    whether an anomaly has an innocent explanation. A high-ELA cluster over an
    oily curry surface is a specular highlight. The same cluster with a hard
    rectilinear boundary is a paste. The numbers alone cannot tell those
    apart; the numbers plus the picture can.

WHY THE SYSTEM PROMPT IS SO CALIBRATION-HEAVY
    An unguided model asked "is this fake?" is far too willing to say yes.
    The numbered rules below exist to push back: missing metadata is normal,
    compression artefacts are normal, oily food reflects light, and
    `inconclusive` is an acceptable answer. A false accusation costs more than
    a manual review.

NOTE ON FAILURE
    Every failure path returns None rather than raising - no API key, bad
    media type, network error, refusal. The pipeline then falls back to
    local-only mode, where it can never reach an accusatory verdict.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Literal

import anthropic
from pydantic import BaseModel, Field

from .. import config

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_SYSTEM = """You are an image forensics analyst for a food delivery platform's \
refund-fraud team. You judge whether a customer-submitted food photo is a genuine \
camera capture, an edited photo, or AI-generated.

You are given the image and the output of an automated forensics pipeline. Weigh both.

Calibration rules, in order of importance:

1. Missing EXIF metadata is NOT evidence of tampering. Every major messaging app \
strips it. Screenshots and forwards are normal customer behaviour.
2. Statistical anomalies with an innocent physical explanation are not evidence. \
Specular highlights on oily or wet food, plastic packaging reflections, steam, \
motion blur, and low-light phone noise all trip these detectors.
3. Real evidence of editing looks like: object boundaries that do not follow the \
underlying geometry, lighting or shadow directions that disagree between regions, \
resolution or focus that changes abruptly across a seam, duplicated texture patches, \
text or logos that are warped or illegible.
4. Real evidence of AI generation looks like: anatomically impossible cutlery or \
hands, text that is almost-letters, food geometry that does not obey physics, \
impossibly clean or uniform surfaces, background objects that melt into each other.
5. When the image is merely low quality, compressed, or ambiguous, answer \
`inconclusive`. Inconclusive is a correct and useful answer. A false accusation of \
fraud costs far more than a manual review.

Also state plainly what the photo actually depicts and whether the depicted damage \
matches the customer's stated complaint - a pristine meal submitted against a \
"food was spilled" claim matters even when the image is perfectly authentic."""


class VisionJudgement(BaseModel):
    verdict: Literal["authentic", "inconclusive", "manipulated", "ai_generated"]
    confidence: float = Field(ge=0.0, le=1.0)
    depicts: str = Field(description="What the photo actually shows, one sentence.")
    supports_claim: Literal["yes", "no", "unclear"]
    observations: list[str] = Field(
        description="Specific visual observations that drove the verdict.",
    )
    innocent_explanations: list[str] = Field(
        description="Benign explanations for any anomalies the pipeline flagged.",
    )


def judge_image(
    path: Path,
    local_signals: list[dict[str, Any]],
    local_score: float,
    claim_text: str,
    client: anthropic.Anthropic | None = None,
) -> VisionJudgement | None:
    """Return Claude's judgement, or None if the call could not be made."""
    media_type = _MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        return None

    try:
        client = client or anthropic.Anthropic()
    except Exception:
        return None  # No credentials configured - caller falls back to local-only.

    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    evidence = json.dumps(
        {"aggregate_manipulation_score": round(local_score, 3), "signals": local_signals},
        indent=2,
        sort_keys=True,
    )

    try:
        response = client.messages.parse(
            model=config.VISION_MODEL,
            max_tokens=config.MAX_TOKENS,
            system=_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Customer's claim:\n{claim_text}\n\n"
                                f"Automated forensics output:\n{evidence}\n\n"
                                "Judge this image."
                            ),
                        },
                    ],
                }
            ],
            output_format=VisionJudgement,
        )
    except anthropic.APIError:
        return None

    if response.stop_reason == "refusal":
        return None
    return response.parsed_output
