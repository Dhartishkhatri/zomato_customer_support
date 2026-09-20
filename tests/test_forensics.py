"""Forensics pipeline tests, run entirely on local signals (no API key).

These assert the *stance* of the detector as much as its accuracy: statistics
alone must never produce an accusatory verdict, and ordinary camera photos must
not be flagged.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zomato_support.forensics import analyse_image
from zomato_support.forensics import signals as sig
from zomato_support.schemas import ImageVerdict

SAMPLES = ROOT / "samples"


@pytest.fixture(scope="session", autouse=True)
def ensure_samples():
    if not (SAMPLES / "real_curry.jpg").exists():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "make_sample_images.py")],
            check=True,
            cwd=ROOT,
        )


def analyse(name: str):
    return analyse_image(SAMPLES / name, use_vision=False)


# --- verdicts ---------------------------------------------------------------
def test_authentic_photo_is_not_flagged():
    result = analyse("real_curry.jpg")
    assert result.verdict is ImageVerdict.AUTHENTIC
    assert result.manipulation_score < 0.35


def test_ai_generated_is_caught_by_provenance():
    result = analyse("ai_generated.png")
    assert result.verdict is ImageVerdict.AI_GENERATED
    assert result.confidence > 0.9


def test_edited_photo_scores_higher_than_the_original():
    assert analyse("morphed_curry.jpg").manipulation_score > analyse(
        "real_curry.jpg"
    ).manipulation_score


def test_local_signals_alone_never_accuse():
    """The core safety property: no vision judge means no accusatory verdict."""
    for name in ("real_curry.jpg", "morphed_curry.jpg"):
        verdict = analyse(name).verdict
        assert verdict in (ImageVerdict.AUTHENTIC, ImageVerdict.INCONCLUSIVE)


def test_edited_photo_defers_to_a_human():
    assert analyse("morphed_curry.jpg").verdict is ImageVerdict.INCONCLUSIVE


# --- individual signals -----------------------------------------------------
def test_editor_software_tag_is_detected():
    names = {s.name for s in analyse("morphed_curry.jpg").signals}
    assert "editor_software_tag" in names


def test_camera_metadata_recognised_on_real_photo():
    signal = next(
        s for s in analyse("real_curry.jpg").signals if s.name == "camera_metadata"
    )
    assert signal.score == 0.0
    assert "samsung" in signal.detail.lower()


def test_generator_metadata_names_the_tool():
    signal = next(
        s for s in analyse("ai_generated.png").signals if s.name == "generator_metadata"
    )
    assert signal.score == 1.0
    assert "stable diffusion" in signal.detail.lower()


def test_copy_move_ignores_periodic_texture():
    """A striped background repeats at a lattice; that is not a forgery."""
    img = Image.new("RGB", (512, 512), (40, 40, 40))
    pixels = img.load()
    for y in range(512):
        for x in range(512):
            # Strongly periodic both ways, plus variation so it is not flat.
            value = 60 + (x % 16) * 9 + (y % 16) * 4
            pixels[x, y] = (value, value // 2, value // 3)

    result = sig.copy_move(img)
    assert result.score < 0.3, result.detail


def test_tiny_images_degrade_gracefully():
    tiny = Image.new("RGB", (24, 24), (120, 90, 60))
    for signal in (
        sig.error_level_analysis(tiny, True),
        sig.noise_consistency(tiny),
        sig.copy_move(tiny),
    ):
        assert 0.0 <= signal.score <= 1.0


def test_grid_preserving_load_does_not_resample():
    """ELA and ghost analysis depend on the JPEG grid surviving."""
    img = Image.new("RGB", (2000, 1500), (100, 100, 100))
    cropped = sig.load_grid_preserving(img)
    assert max(cropped.size) <= 1024
    assert cropped.width % 8 == 0 and cropped.height % 8 == 0


def test_all_signal_scores_are_normalised():
    for name in ("real_curry.jpg", "morphed_curry.jpg", "ai_generated.png"):
        for signal in analyse(name).signals:
            assert 0.0 <= signal.score <= 1.0, f"{name}/{signal.name}"
            assert signal.weight > 0


def test_analysis_serialises():
    payload = analyse("real_curry.jpg").to_dict()
    assert payload["verdict"] == "authentic"
    assert isinstance(payload["signals"], list)
    assert payload["vision_used"] is False


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        analyse_image(SAMPLES / "nope.jpg", use_vision=False)
