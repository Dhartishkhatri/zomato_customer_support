"""
===============================================================================
PROVENANCE CHECKS - metadata, watermarks, C2PA               [CORE, high value]
===============================================================================

WHAT THIS DOES
    Reads what the file says about itself: EXIF tags, PNG text chunks, XMP
    packets, and C2PA / Content Credentials manifests.

WHY IT IS THE MOST VALUABLE PART OF THE PIPELINE
    These are close to EVIDENCE rather than inference. A C2PA manifest saying
    "trainedAlgorithmicMedia", or a Stable Diffusion `parameters` chunk full
    of prompt text, is a near-certain answer. The statistical signals in
    signals.py only ever produce suspicion; this file produces proof.

    That asymmetry is why generator_metadata carries weight 6.0 - the highest
    in the system - and why pipeline.py lets it short-circuit everything else.

THE ASYMMETRY THAT MATTERS MOST
    Presence of a marker is strong evidence. ABSENCE OF METADATA IS NOT
    EVIDENCE OF ANYTHING. WhatsApp, Instagram, Messenger and most chat apps
    strip EXIF from every image they forward, so "no EXIF" is the normal case
    for a customer-support photo, not a red flag. That is why the no-metadata
    branch below scores 0.18 at weight 0.8 - almost nothing. Do not raise it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image, ExifTags

from ..schemas import ForensicSignal

# Editors that rewrite pixel data. Presence is not proof of fraud - people crop
# and rotate photos - but it raises the prior when paired with pixel signals.
_EDITOR_PATTERNS = (
    "adobe photoshop",
    "gimp",
    "paint.net",
    "snapseed",
    "picsart",
    "facetune",
    "lightroom",
    "pixlr",
    "canva",
    "affinity photo",
    "krita",
)

# Strings that identify synthetic imagery outright.
_GENERATOR_PATTERNS = (
    "stable diffusion",
    "stablediffusionwebui",
    "automatic1111",
    "comfyui",
    "midjourney",
    "dall-e",
    "dalle",
    "openai",
    "firefly",
    "adobe firefly",
    "imagen",
    "flux.1",
    "black forest labs",
    "leonardo.ai",
    "nano banana",
    "grok-imagine",
    "seedream",
    "trainedalgorithmicmedia",
    "compositewithtrainedalgorithmicmedia",
    "ai generated",
    "ai-generated",
)

_C2PA_MARKERS = (b"c2pa", b"jumbf", b"urn:uuid:", b"contentauth", b"c2pa.assertions")


def _exif_dict(img: Image.Image) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        exif = img.getexif()
    except Exception:
        return out
    for tag_id, value in exif.items():
        tag = ExifTags.TAGS.get(tag_id, str(tag_id))
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        out[tag] = value
    return out


def _text_blobs(img: Image.Image, raw: bytes) -> str:
    """Every scrap of embedded text, lowercased, for pattern matching."""
    parts: list[str] = []

    # PNG tEXt / iTXt chunks - where SD WebUI and ComfyUI write prompts.
    for key, value in (getattr(img, "text", None) or {}).items():
        parts.append(f"{key}={value}")
    for key, value in (getattr(img, "info", None) or {}).items():
        if isinstance(value, (str, bytes)):
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            parts.append(f"{key}={value}")

    # XMP packets live in the raw bytes for both JPEG and PNG.
    for match in re.finditer(rb"<x:xmpmeta.*?</x:xmpmeta>", raw, re.DOTALL):
        parts.append(match.group(0).decode("utf-8", errors="replace"))

    return " ".join(parts).lower()


def analyse_provenance(path: Path, img: Image.Image, raw: bytes) -> list[ForensicSignal]:
    exif = _exif_dict(img)
    blobs = _text_blobs(img, raw)
    haystack = f"{blobs} {str(exif).lower()}"
    signals: list[ForensicSignal] = []

    # --- 1. Outright generator watermarks ----------------------------------
    hits = sorted({p for p in _GENERATOR_PATTERNS if p in haystack})
    if hits:
        signals.append(
            ForensicSignal(
                name="generator_metadata",
                score=1.0,
                weight=6.0,
                detail=f"Embedded metadata names a generative tool: {', '.join(hits)}.",
                raw={"matches": hits},
            )
        )
    else:
        signals.append(
            ForensicSignal(
                name="generator_metadata",
                score=0.0,
                weight=1.0,
                detail="No generative-tool markers found in metadata.",
                raw={},
            )
        )

    # --- 2. C2PA / Content Credentials -------------------------------------
    c2pa_hits = [m.decode() for m in _C2PA_MARKERS if m in raw.lower()]
    if c2pa_hits:
        ai_claim = any(p in haystack for p in ("trainedalgorithmicmedia", "ai generated"))
        signals.append(
            ForensicSignal(
                name="c2pa_manifest",
                score=1.0 if ai_claim else 0.15,
                weight=5.0 if ai_claim else 1.5,
                detail=(
                    "C2PA manifest declares AI-generated origin."
                    if ai_claim
                    else "C2PA manifest present; parse it server-side to verify the signing chain."
                ),
                raw={"markers": c2pa_hits, "declares_ai": ai_claim},
            )
        )

    # --- 3. Editing software tag -------------------------------------------
    software = str(exif.get("Software", "")).lower()
    editor = next((p for p in _EDITOR_PATTERNS if p in software), None)
    if editor:
        signals.append(
            ForensicSignal(
                name="editor_software_tag",
                score=0.72,
                weight=3.0,
                detail=f"EXIF Software tag names an image editor: {exif.get('Software')}.",
                raw={"software": exif.get("Software")},
            )
        )

    # --- 4. Camera capture metadata ----------------------------------------
    has_camera = bool(exif.get("Make") or exif.get("Model"))
    has_capture_time = bool(exif.get("DateTimeOriginal") or exif.get("DateTime"))
    if has_camera and has_capture_time:
        # A coherent capture record is mild evidence *for* authenticity.
        device = f"{exif.get('Make', '')} {exif.get('Model', '')}".strip()
        signals.append(
            ForensicSignal(
                name="camera_metadata",
                score=0.0,
                weight=2.5,
                detail=f"Intact camera capture metadata ({device}).",
                raw={
                    "make": exif.get("Make"),
                    "model": exif.get("Model"),
                    "captured_at": exif.get("DateTimeOriginal") or exif.get("DateTime"),
                },
            )
        )
    elif has_camera or has_capture_time:
        signals.append(
            ForensicSignal(
                name="camera_metadata",
                score=0.30,
                weight=1.0,
                detail="Camera metadata is partially present - consistent with a crop or an edit-and-resave.",
                raw={"make": exif.get("Make"), "model": exif.get("Model")},
            )
        )
    else:
        # Deliberately near-zero: messaging apps strip EXIF from everything.
        signals.append(
            ForensicSignal(
                name="camera_metadata",
                score=0.18,
                weight=0.8,
                detail=(
                    "No camera metadata. Normal for photos forwarded through a messaging "
                    "app, so this is weak on its own."
                ),
                raw={},
            )
        )

    return signals
