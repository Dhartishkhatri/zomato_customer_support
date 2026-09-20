"""
===============================================================================
STANDALONE IMAGE AUTHENTICITY CLI                          [useful, not core]
===============================================================================

Runs the forensics pipeline on one file and prints every signal with its
score, so you can see WHY a verdict came out the way it did. The fastest way
to build intuition for the detector, and to re-tune thresholds against real
photos.

    python verify_image.py samples/morphed_curry.jpg --claim "curry was burnt"
    python verify_image.py samples/real_curry.jpg --offline --json

--offline skips the Claude vision judge and runs local signals only, so it
needs no API key. Remember that offline mode can never return an accusatory
verdict - the worst it will say is `inconclusive`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode the rupee sign
# or other non-ASCII characters in policy text.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from zomato_support.forensics import analyse_image

_COLOURS = {
    "authentic": "\033[32m",
    "inconclusive": "\033[33m",
    "manipulated": "\033[31m",
    "ai_generated": "\033[31m",
}
_RESET = "\033[0m"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether an image is real, edited or AI-generated.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--claim", default="", help="What the customer says the photo shows.")
    parser.add_argument("--offline", action="store_true", help="Local signals only; no API call.")
    parser.add_argument("--json", action="store_true", help="Emit the full analysis as JSON.")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"No such file: {args.image}", file=sys.stderr)
        return 2

    analysis = analyse_image(
        args.image, claim_text=args.claim, use_vision=not args.offline
    )

    if args.json:
        print(json.dumps(analysis.to_dict(), indent=2))
        return 0

    colour = _COLOURS.get(analysis.verdict.value, "")
    print(f"\n  file       : {args.image.name}")
    print(f"  verdict    : {colour}{analysis.verdict.value.upper()}{_RESET}")
    print(f"  confidence : {analysis.confidence:.0%}")
    print(f"  evidence   : {analysis.manipulation_score:.3f} aggregate manipulation score")
    print(f"  vision     : {'Claude Opus 5' if analysis.vision_used else 'not used (local only)'}")
    print(f"\n  {analysis.summary}\n")

    print("  signals:")
    for signal in sorted(analysis.signals, key=lambda s: -s.score * s.weight):
        bar = "#" * int(signal.score * 22)
        print(f"    {signal.name:<24} {signal.score:5.3f} w{signal.weight:<4} {bar}")
        print(f"      {signal.detail}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
