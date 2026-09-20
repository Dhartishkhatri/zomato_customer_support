"""
===============================================================================
PIXEL-LEVEL FORENSIC SIGNALS                                          [CORE]
===============================================================================

WHAT THIS DOES
    Measures an image for statistical traces of editing. Each function returns
    one ForensicSignal scored 0..1 (1 = "looks manipulated") plus a weight
    saying how much to trust it. pipeline.py combines them.

WHY IT EXISTS
    Provenance (provenance.py) catches images that *admit* what they are, via
    metadata. Anything that has been stripped of metadata - which is most
    things - needs to be judged on its pixels. These are those judgements.

HOW MUCH TO TRUST IT
    Not much on its own, by design. Read the aggregation comment in
    pipeline.py: local statistics can never produce an accusatory verdict.
    On the three sample images these signals move the score by ~0.1-0.4; the
    metadata checks and the Claude vision judge do the heavy lifting.

IS IT CRUCIAL?
    Yes, but only just. If you wanted to cut this file entirely you would keep
    a working system (provenance + vision judge), with a blind spot for
    metadata-stripped edits. Two weaker detectors were already removed - see
    "REMOVED DETECTORS" at the bottom of this file.
===============================================================================
"""

from __future__ import annotations

import io
from collections import Counter, deque

import numpy as np
from PIL import Image, ImageFilter

from ..schemas import ForensicSignal

# Working resolution for analysis. Big enough for block statistics to mean
# something, small enough that the whole pipeline stays well under a second.
_MAX_DIM = 1024

# Image dimensions that generative models emit verbatim. A photo is almost
# never exactly 1024x1024; a diffusion model output very often is.
_GENERATIVE_SIZES = {
    (512, 512), (768, 768), (1024, 1024), (1536, 1536), (2048, 2048),
    (512, 768), (768, 512), (1024, 768), (768, 1024),
    (1024, 1536), (1536, 1024), (896, 1152), (1152, 896),
    (832, 1216), (1216, 832), (1344, 768), (768, 1344),
}


def _clamp(x: float) -> float:
    """Squash any raw measurement into the 0..1 range every signal must use."""
    return float(min(1.0, max(0.0, x)))


# ---------------------------------------------------------------------------
# Two ways to load an image, and the difference between them matters a lot
# ---------------------------------------------------------------------------
def load_for_analysis(img: Image.Image) -> Image.Image:
    """Shrink the image to a manageable size by RESAMPLING.

    Safe for signals that look at noise and texture, because those survive a
    resize well enough. NOT safe for anything reading JPEG structure - use
    load_grid_preserving for that instead.
    """
    work = img.convert("RGB")
    if max(work.size) > _MAX_DIM:
        scale = _MAX_DIM / max(work.size)
        work = work.resize(
            (max(1, int(work.width * scale)), max(1, int(work.height * scale))),
            Image.LANCZOS,
        )
    return work


def load_grid_preserving(img: Image.Image) -> Image.Image:
    """Shrink the image by CROPPING, leaving original pixels untouched.

    WHY THIS EXISTS (this was a real bug, worth understanding):
    JPEG compresses in 8x8 pixel blocks. Error-level analysis works by
    detecting inconsistencies in those blocks. If you resize the image first,
    every pixel is recomputed and the 8x8 grid is destroyed - so ELA then
    measures your own resizing artefacts instead of the image's compression
    history, and reports confident nonsense. Cropping on an 8-pixel boundary
    keeps the grid aligned and the pixels original.
    """
    work = img.convert("RGB")
    w, h = work.size
    if max(w, h) <= _MAX_DIM:
        return work

    cw, ch = min(w, _MAX_DIM) // 8 * 8, min(h, _MAX_DIM) // 8 * 8
    left = ((w - cw) // 2) // 8 * 8
    top = ((h - ch) // 2) // 8 * 8
    return work.crop((left, top, left + cw, top + ch))


def _gray(img: Image.Image) -> np.ndarray:
    """Greyscale pixel array as floats, so arithmetic does not wrap at 255."""
    return np.asarray(img.convert("L"), dtype=np.float32)


def _blocks(arr: np.ndarray, size: int) -> np.ndarray:
    """Chop an image into a grid of square tiles.

    Returns shape (tiles_down, tiles_across, size, size). Nearly every signal
    here works per-tile rather than per-pixel, because a forgery is a *region*
    that differs from its surroundings, not a scattering of odd pixels.
    """
    h, w = arr.shape[:2]
    bh, bw = h // size, w // size
    if bh == 0 or bw == 0:
        return np.empty((0, 0, size, size), dtype=arr.dtype)
    trimmed = arr[: bh * size, : bw * size]
    return trimmed.reshape(bh, size, bw, size).swapaxes(1, 2)


def _largest_component(mask: np.ndarray) -> int:
    """Size of the biggest connected blob of True tiles (flood fill).

    WHY: scattered suspicious tiles are just noise in the measurement. A real
    paste is one contiguous patch. Measuring the largest *connected* region
    instead of the total count is what separates the two.
    """
    if not mask.any():
        return 0
    seen = np.zeros_like(mask, dtype=bool)
    best = 0
    h, w = mask.shape
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            size = 0
            queue = deque([(sy, sx)])
            seen[sy, sx] = True
            while queue:
                y, x = queue.popleft()
                size += 1
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            best = max(best, size)
    return best


# ---------------------------------------------------------------------------
# SIGNAL 1: Error Level Analysis - the strongest pixel signal here
# ---------------------------------------------------------------------------
def error_level_analysis(img: Image.Image, is_jpeg: bool) -> ForensicSignal:
    """Re-save the image as JPEG and see which regions change the most.

    THE IDEA: saving a JPEG loses a little detail. An area that has already
    been saved several times has little left to lose, so it changes little on
    the next save. An area pasted in from elsewhere has a different history,
    so it sheds a different amount - and shows up as a patch with an odd
    "error level" compared to the rest of the frame.

    MEASURED: ~8x ratio on the authentic sample, ~20x on the edited one. This
    is the signal that actually separates them.
    """
    rgb = np.asarray(img, dtype=np.float32)

    # Re-encode at a fixed quality and diff against the original.
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    buf.seek(0)
    recompressed = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)
    diff = np.abs(rgb - recompressed).max(axis=2)

    block_means = _blocks(diff, 16).mean(axis=(2, 3))
    if block_means.size < 16:
        return ForensicSignal(
            "error_level_analysis", 0.0, 0.5, "Image too small for reliable ELA.", {}
        )

    # How extreme is the worst 1% of tiles compared to a typical tile?
    median = float(np.median(block_means))
    p99 = float(np.percentile(block_means, 99))
    ratio = p99 / (median + 1e-3)

    # And is the high-error area one contiguous patch (a paste) or scattered
    # (just a detailed photo)?
    threshold = median + 3.0 * float(block_means.std() + 1e-6)
    hot = block_means > threshold
    cluster_frac = _largest_component(hot) / block_means.size

    # Calibrated so an ordinary single-save phone photo, whose specular
    # highlights and sharp edges naturally give ~8x, scores near zero.
    spread_score = _clamp((ratio - 7.0) / 20.0)
    cluster_score = _clamp((cluster_frac - 0.004) / 0.06)
    score = _clamp(0.55 * spread_score + 0.45 * cluster_score)

    return ForensicSignal(
        name="error_level_analysis",
        score=score,
        # A file that was already JPEG gives a cleaner read than a PNG we had
        # to encode ourselves just to run this test.
        weight=2.5 if is_jpeg else 1.2,
        detail=(
            f"Error-level ratio {ratio:.1f}x median; largest contiguous high-error "
            f"region covers {cluster_frac * 100:.1f}% of the frame."
        ),
        raw={
            "p99_over_median": round(ratio, 2),
            "largest_cluster_fraction": round(cluster_frac, 4),
        },
    )


# ---------------------------------------------------------------------------
# SIGNAL 2: Noise consistency - catches rendered/synthetic images
# ---------------------------------------------------------------------------
def noise_consistency(img: Image.Image) -> ForensicSignal:
    """Compare the sensor-noise floor across flat areas of the picture.

    THE IDEA: a camera sensor lays down a consistent speckle of noise across
    the whole frame. A pasted-in region carries a different noise floor, and a
    rendered or AI image often has almost none at all.

    MEASURED: this is what flags the AI sample (104% variation) while leaving
    both real photographs at zero.
    """
    gray = _gray(img)
    # A median filter removes the speckle; subtracting gives just the speckle.
    denoised = np.asarray(img.convert("L").filter(ImageFilter.MedianFilter(3)), dtype=np.float32)
    residual = gray - denoised

    res_blocks = _blocks(residual, 32)
    gray_blocks = _blocks(gray, 32)
    if res_blocks.size == 0 or res_blocks.shape[0] * res_blocks.shape[1] < 12:
        return ForensicSignal(
            "noise_consistency", 0.0, 0.5, "Image too small for noise analysis.", {}
        )

    noise_std = res_blocks.std(axis=(2, 3)).ravel()

    # Only look at FLAT tiles. A tile full of edges has a big residual for
    # perfectly innocent reasons, and would swamp the measurement.
    detail_std = gray_blocks.std(axis=(2, 3)).ravel()
    flat = detail_std < np.percentile(detail_std, 60)
    sample = noise_std[flat] if noise_std[flat].size >= 8 else noise_std

    mean = float(sample.mean())
    if mean < 0.12:
        # Essentially no noise floor anywhere: heavy denoising, or synthesis.
        return ForensicSignal(
            name="noise_consistency",
            score=0.62,
            weight=1.8,
            detail=(
                f"Flat regions carry almost no sensor noise (mean residual {mean:.3f}), "
                "which suits a rendered or heavily denoised image."
            ),
            raw={"mean_noise": round(mean, 4)},
        )

    # Otherwise: how much does the noise floor VARY across the frame?
    cv = float(sample.std() / mean)
    return ForensicSignal(
        name="noise_consistency",
        score=_clamp((cv - 0.45) / 0.85),
        weight=1.8,
        detail=f"Noise floor varies by {cv * 100:.0f}% across flat regions (mean {mean:.2f}).",
        raw={"coefficient_of_variation": round(cv, 3), "mean_noise": round(mean, 3)},
    )


# ---------------------------------------------------------------------------
# SIGNAL 3: Copy-move - a region duplicated within the same image
# ---------------------------------------------------------------------------
def copy_move(img: Image.Image) -> ForensicSignal:
    """Find a patch that has been cloned somewhere else in the same frame.

    A common, low-effort forgery: copy some spilled sauce and paste it twice
    to make the damage look worse.

    TWO STAGES, and the second one is the important one:
      1. Propose  - coarse tile fingerprints suggest candidate displacements.
      2. Verify   - actually shift the image by each candidate and check where
                    the pixels really agree.
    Voting alone is not enough: repetitive-but-distinct texture (rice grains,
    a tiled floor, the curry blobs in our sample) proposes plenty of
    candidates. Only a true clone survives verification.
    """
    # Work small - this is the most expensive signal in the file.
    small = img.convert("L")
    if max(small.size) > 512:
        scale = 512 / max(small.size)
        small = small.resize(
            (max(1, int(small.width * scale)), max(1, int(small.height * scale))),
            Image.LANCZOS,
        )
    gray = np.asarray(small, dtype=np.float32)

    # --- Stage 1: propose candidate offsets --------------------------------
    block, stride = 16, 8
    h, w = gray.shape
    buckets: dict[tuple, list[tuple[int, int]]] = {}

    for y in range(0, h - block + 1, stride):
        for x in range(0, w - block + 1, stride):
            patch = gray[y : y + block, x : x + block]
            if patch.std() < 6.0:
                continue  # Flat tiles (plate, tablecloth) match each other harmlessly.
            half = block // 2
            # Crude fingerprint: brightness of each quadrant plus contrast.
            feature = (
                round(float(patch[:half, :half].mean()) / 2),
                round(float(patch[:half, half:].mean()) / 2),
                round(float(patch[half:, :half].mean()) / 2),
                round(float(patch[half:, half:].mean()) / 2),
                round(float(patch.std()) / 2),
            )
            buckets.setdefault(feature, []).append((y, x))

    # Tiles sharing a fingerprint vote for the displacement between them.
    offsets: Counter[tuple[int, int]] = Counter()
    for positions in buckets.values():
        if len(positions) < 2 or len(positions) > 24:
            continue
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dy = positions[j][0] - positions[i][0]
                dx = positions[j][1] - positions[i][1]
                if abs(dy) + abs(dx) < block * 2:
                    continue  # Overlapping neighbours, not a real copy.
                offsets[(dy, dx)] += 1

    # --- Stage 2: verify each candidate ------------------------------------
    best_frac, best_offset, rejected_periodic = 0.0, None, 0
    for (dy, dx), _votes in offsets.most_common(12):
        frac = _match_fraction(gray, dy, dx)
        if frac <= 0.006:
            continue

        # PERIODICITY GUARD (this fixed a real false positive):
        # A repeating texture - striped tablecloth, tiled floor, the ribs of a
        # takeaway container - matches itself at some offset AND at twice that
        # offset, because it is a lattice. A genuine paste matches at exactly
        # one offset and nowhere else. Without this check every patterned
        # background reads as fraud.
        if _match_fraction(gray, dy * 2, dx * 2) > 0.6 * frac:
            rejected_periodic += 1
            continue

        if frac > best_frac:
            best_frac, best_offset = frac, (dy, dx)

    return ForensicSignal(
        name="copy_move",
        score=_clamp((best_frac - 0.006) / 0.05),
        weight=2.2,
        detail=(
            f"A contiguous region covering {best_frac * 100:.1f}% of the frame is "
            f"pixel-identical to another region offset by {best_offset}."
            if best_frac > 0.006
            else "No verified duplicated regions."
        ),
        raw={
            "duplicated_area_fraction": round(best_frac, 4),
            "offset": list(best_offset) if best_offset else None,
            "rejected_as_periodic": rejected_periodic,
        },
    )


def _match_fraction(gray: np.ndarray, dy: int, dx: int) -> float:
    """How much of the frame repeats itself at displacement (dy, dx)?

    Shift the image by the offset, overlay it on itself, and measure the
    largest contiguous region where (a) the pixels agree to within a few grey
    levels and (b) there is real texture there to agree about.
    """
    h, w = gray.shape
    ys1, ys2 = max(0, -dy), h - max(0, dy)
    xs1, xs2 = max(0, -dx), w - max(0, dx)
    if ys2 - ys1 < 32 or xs2 - xs1 < 32:
        return 0.0

    a = gray[ys1:ys2, xs1:xs2]
    b = gray[ys1 + dy : ys2 + dy, xs1 + dx : xs2 + dx]

    cell = 8
    diff_blocks = _blocks(np.abs(a - b), cell)
    texture_blocks = _blocks(a, cell)
    if diff_blocks.size == 0:
        return 0.0

    matched = (diff_blocks.mean(axis=(2, 3)) < 4.0) & (texture_blocks.std(axis=(2, 3)) > 6.0)
    return _largest_component(matched) * (cell * cell) / float(h * w)


# ---------------------------------------------------------------------------
# SIGNAL 4: Generative geometry - almost free, so worth keeping
# ---------------------------------------------------------------------------
def generative_geometry(original_size: tuple[int, int], had_exif: bool) -> ForensicSignal:
    """Exact model output dimensions with no camera metadata is a soft tell.

    Ten lines of code, no image processing at all, and it contributes 0.55 on
    the AI sample. Cheap signals that only fire on a narrow pattern are good
    value; that is the whole reason this one survives.
    """
    if original_size in _GENERATIVE_SIZES and not had_exif:
        return ForensicSignal(
            name="generative_geometry",
            score=0.55,
            weight=1.4,
            detail=(
                f"{original_size[0]}x{original_size[1]} is a stock generative-model "
                "output size and the file carries no camera metadata."
            ),
            raw={"size": list(original_size)},
        )
    if original_size in _GENERATIVE_SIZES:
        return ForensicSignal(
            name="generative_geometry",
            score=0.2,
            weight=0.8,
            detail=f"{original_size[0]}x{original_size[1]} is a common generative size, but camera metadata is present.",
            raw={"size": list(original_size)},
        )
    return ForensicSignal(
        name="generative_geometry",
        score=0.0,
        weight=0.8,
        detail=f"{original_size[0]}x{original_size[1]} is not a stock generative output size.",
        raw={"size": list(original_size)},
    )


# ===========================================================================
# REMOVED DETECTORS - deliberately cut, recorded here so the decision is
# visible rather than mysterious.
#
# spectral_periodicity (~55 lines of FFT)
#     Looked for the regular frequency spikes that learned upsamplers leave
#     behind. Scored 0.000 / 0.000 / 0.081 on the three sample images - it
#     never once changed a verdict. Good against older GANs, weak against
#     modern diffusion decoders. Worst complexity-to-value ratio in the file.
#
# recompression_history (~65 lines, localised JPEG-ghost)
#     Re-encoded across a quality ladder to find a region saved at a different
#     quality than the rest. Correct in principle and it took two rewrites to
#     stop it false-positiving, but after the fixes it scored 0.057 / 0.036 /
#     0.000 - almost nothing.
#
# Both are in the pre-trim backup if you ever want them back. Neither changed
# any verdict on the sample set; cutting them left behaviour identical.
# ===========================================================================
