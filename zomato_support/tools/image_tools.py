"""
===============================================================================
THE IMAGE AUTHENTICITY TOOL - forensics' entry point into the agent   [CORE]
===============================================================================

This is the bridge between the forensics pipeline and the agent loop, and the
important thing it does is SPLIT the result in two:

    full analysis  -> ctx.image_analyses  (policy gate + audit log read this)
    redacted view  -> returned to model   (verdict + handling instruction only)

WHY SPLIT IT
    Policy forbids forensic detail reaching a customer. You can instruct a
    model not to repeat something, or you can simply never tell it. The second
    is reliable. Scores, signal names and confidence numbers stay out of the
    model's context entirely, so there is nothing to paraphrase into a reply.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anthropic

from ..backends import Backends
from ..forensics import analyse_image
from ..forensics.pipeline import analysis_for_agent
from ..schemas import ConversationContext


def verify_image_authenticity(
    backends: Backends,
    ctx: ConversationContext,
    image_path: str,
    claim: str = "",
    client: anthropic.Anthropic | None = None,
) -> dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        return {"error": "image_not_found", "message": f"No attachment at {image_path}."}

    try:
        analysis = analyse_image(path, claim_text=claim, client=client)
    except OSError as exc:
        # A corrupt or non-image attachment is a customer-support problem, not
        # a crash. Report it and let the agent ask for the photo again.
        return {"error": "unreadable_image", "message": f"Attachment could not be decoded: {exc}"}

    ctx.image_analyses[path.name] = analysis        # full detail, internal
    return {"attachment": path.name, **analysis_for_agent(analysis)}  # redacted
