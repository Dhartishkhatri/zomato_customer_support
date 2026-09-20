"""
===============================================================================
FORENSICS AGGREGATION - turns many signals into one verdict          [CORE]
===============================================================================

WHAT THIS DOES
    Runs provenance checks + pixel signals + the Claude vision judge, then
    combines them into a single verdict:
        authentic | inconclusive | manipulated | ai_generated

WHY IT EXISTS
    This is where the project's most important design decision lives, so read
    this file before changing any threshold.

THE DESIGN STANCE - the thing to understand
    A false accusation of fraud against a real customer costs far more than a
    manual review does. So the pipeline is deliberately built to be hard to
    weaponise:

      * Statistics ALONE can never reach an accusatory verdict. With no vision
        judge available, the worst this can output is `inconclusive`, which
        routes to a human. See the `judgement is None` branch below.
      * Only hard provenance evidence (a generator watermark, a C2PA manifest
        declaring AI origin) or Claude visually confirming a SPECIFIC named
        defect can produce `manipulated` / `ai_generated`.
      * Missing metadata is never treated as evidence. Chat apps strip it from
        everything they forward.

IS IT CRUCIAL?  Yes. This file is the safety logic. Do not simplify it away.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anthropic
from PIL import Image

from .. import config
from ..schemas import ForensicSignal, ImageAnalysis, ImageVerdict
from . import signals as sig
from .provenance import analyse_provenance
from .vision import VisionJudgement, judge_image

# Score at or above which local statistics say "a human should look at this".
# Note it can only ever produce `inconclusive`, never an accusation.
#
# TUNING NOTE: the edited sample lands at ~0.49, the authentic one at ~0.08.
# This threshold sits between them but the gap is not wide, and it is
# calibrated on three synthetic images. Re-tune against real photos, watching
# the false-positive rate on genuine ones as the metric that matters.
_LOCAL_SUSPICION = 0.45

# The weight of the strongest possible signal, used to normalise the rest.
_MAX_WEIGHT = 6.0


def _aggregate(items: list[ForensicSignal]) -> float:
    """Combine signals as ACCUMULATING EVIDENCE, not as an average.

    WHY NOT A WEIGHTED MEAN (this matters):
    With a mean, one damning signal among eight quiet ones averages away to
    nothing - and worse, adding more benign checks would silently *weaken* the
    detector. That is a trap.

    Noisy-OR instead treats each signal as an independent chance of having
    caught something real. Strong evidence survives however many quiet checks
    sit beside it, and adding checks can only ever raise the score.

        combined = 1 - (chance signal A missed it)
                     * (chance signal B missed it) * ...
    """
    surviving = 1.0
    for signal in items:
        probability = signal.score * min(1.0, signal.weight / _MAX_WEIGHT)
        surviving *= 1.0 - probability
    return 1.0 - surviving


def analyse_image(
    path: str | Path,
    claim_text: str = "",
    client: anthropic.Anthropic | None = None,
    use_vision: bool | None = None,
) -> ImageAnalysis:
    """Full authenticity check on one image. The only entry point you need."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw = path.read_bytes()
    with Image.open(path) as opened:
        opened.load()
        original_size = opened.size
        is_jpeg = (opened.format or "").upper() in ("JPEG", "MPO")
        provenance = analyse_provenance(path, opened, raw)
        had_exif = bool(opened.getexif())

        # Two working copies, and the distinction is load-bearing:
        #   work = resampled  -> fine for noise and texture signals
        #   grid = cropped    -> required by ELA, which reads the JPEG 8x8 grid
        work = sig.load_for_analysis(opened)
        grid = sig.load_grid_preserving(opened)

    pixel_signals = [
        sig.error_level_analysis(grid, is_jpeg),
        sig.noise_consistency(work),
        sig.copy_move(work),
        sig.generative_geometry(original_size, had_exif),
    ]

    all_signals = provenance + pixel_signals
    local_score = _aggregate(all_signals)

    # --- Path A: hard provenance evidence, short-circuit -------------------
    # A Stable Diffusion parameters chunk or a C2PA manifest declaring AI
    # origin is near-proof. No statistics needed.
    hard_ai = next(
        (
            s
            for s in provenance
            if s.name in ("generator_metadata", "c2pa_manifest") and s.score >= 0.99
        ),
        None,
    )

    should_use_vision = (not config.OFFLINE_FORENSICS) if use_vision is None else use_vision
    judgement: VisionJudgement | None = None
    if should_use_vision:
        judgement = judge_image(
            path,
            [s.to_dict() for s in all_signals],
            local_score,
            claim_text or "(no claim text supplied)",
            client=client,
        )

    if hard_ai is not None:
        return ImageAnalysis(
            verdict=ImageVerdict.AI_GENERATED,
            confidence=0.97,
            manipulation_score=max(local_score, 0.9),
            signals=all_signals,
            summary=hard_ai.detail,
            vision_used=judgement is not None,
            vision_verdict=judgement.verdict if judgement else None,
            vision_notes=_notes(judgement),
        )

    # --- Path B: no vision judge available ---------------------------------
    # THE SAFETY CAP. Statistics do not accuse people, so the worst outcome
    # here is `inconclusive` and a handover to a human.
    if judgement is None:
        verdict = (
            ImageVerdict.AUTHENTIC
            if local_score < config.FORENSICS_AUTHENTIC_BELOW
            else ImageVerdict.INCONCLUSIVE
        )
        return ImageAnalysis(
            verdict=verdict,
            confidence=0.66 if verdict is ImageVerdict.AUTHENTIC else 0.55,
            manipulation_score=local_score,
            signals=all_signals,
            summary=(
                "Local forensic signals are all within normal range for a camera photo."
                if verdict is ImageVerdict.AUTHENTIC
                else "No vision review available; local signals alone cannot settle this."
            ),
            vision_used=False,
        )

    # --- Path C: reconcile statistics with Claude's judgement --------------
    return _reconcile(all_signals, local_score, judgement)


def _reconcile(
    all_signals: list[ForensicSignal],
    local_score: float,
    judgement: VisionJudgement,
) -> ImageAnalysis:
    """Merge the numbers with the vision judgement into a final verdict."""
    vision_verdict = ImageVerdict(judgement.verdict)
    accusatory = vision_verdict in (ImageVerdict.MANIPULATED, ImageVerdict.AI_GENERATED)

    if accusatory:
        # GATE ON ACCUSATIONS: Claude must be confident AND have named at least
        # one specific observation. "It looks off" is not grounds to refuse
        # someone's refund.
        if judgement.confidence >= 0.75 and len(judgement.observations) >= 1:
            verdict = vision_verdict
            # Agreement between the two halves raises confidence; disagreement
            # (clean statistics but a damning picture) lowers it.
            confidence = min(0.96, judgement.confidence * (1.12 if local_score > 0.45 else 0.9))
        else:
            verdict = ImageVerdict.INCONCLUSIVE
            confidence = 0.6

    elif vision_verdict is ImageVerdict.AUTHENTIC:
        if local_score >= _LOCAL_SUSPICION:
            # Claude found innocent explanations but the numbers still object.
            # Split the difference: a human decides.
            verdict = ImageVerdict.INCONCLUSIVE
            confidence = 0.58
        else:
            verdict = ImageVerdict.AUTHENTIC
            confidence = min(0.95, judgement.confidence * (1.05 if local_score < 0.3 else 0.92))
    else:
        verdict = ImageVerdict.INCONCLUSIVE
        confidence = max(0.5, judgement.confidence)

    summary_bits = [judgement.depicts.strip().rstrip(".") + "."]
    if judgement.observations:
        summary_bits.append("Observed: " + "; ".join(judgement.observations[:3]) + ".")
    if judgement.innocent_explanations and verdict is not ImageVerdict.MANIPULATED:
        summary_bits.append(
            "Benign explanations: " + "; ".join(judgement.innocent_explanations[:2]) + "."
        )

    return ImageAnalysis(
        verdict=verdict,
        confidence=round(confidence, 3),
        manipulation_score=local_score,
        signals=all_signals,
        summary=" ".join(summary_bits),
        vision_used=True,
        vision_verdict=judgement.verdict,
        vision_notes=_notes(judgement),
    )


def _notes(judgement: VisionJudgement | None) -> str:
    """Flatten the vision judgement into one auditable string."""
    if judgement is None:
        return ""
    return (
        f"supports_claim={judgement.supports_claim}; "
        f"confidence={judgement.confidence:.2f}; "
        f"observations={judgement.observations}"
    )


def analysis_for_agent(analysis: ImageAnalysis) -> dict[str, Any]:
    """The REDACTED view the support agent is allowed to see.

    WHY REDACT: policy forbids forensic detail reaching a customer. The surest
    way to stop the model paraphrasing a manipulation score into a chat reply
    is to never put that score in its context at all. The model gets a verdict
    and an instruction; the numbers stay in the audit log.
    """
    return {
        "verdict": analysis.verdict.value,
        "confidence": round(analysis.confidence, 2),
        "what_the_photo_shows": analysis.summary,
        "supports_the_claim": (
            analysis.vision_notes.split("supports_claim=")[1].split(";")[0]
            if "supports_claim=" in analysis.vision_notes
            else "unclear"
        ),
        "handling": {
            ImageVerdict.AUTHENTIC.value: "Photo is usable as evidence. Proceed under KB-REFUND-001.",
            ImageVerdict.INCONCLUSIVE.value: (
                "Photo cannot be verified either way. A supervisor reviews it. "
                "Do NOT tell the customer their photo was rejected or doubted."
            ),
            ImageVerdict.MANIPULATED.value: (
                "Photo shows editing. No automated refund. Route to Trust & Safety. "
                "Do NOT accuse the customer."
            ),
            ImageVerdict.AI_GENERATED.value: (
                "Photo is synthetic. No automated refund. Route to Trust & Safety. "
                "Do NOT accuse the customer."
            ),
        }[analysis.verdict.value],
    }
