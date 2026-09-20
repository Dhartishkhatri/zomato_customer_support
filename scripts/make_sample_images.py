"""Generate the sample attachments the demo scenarios reference.

Three files, each exercising a different path through the forensics pipeline:

  real_curry.jpg      - camera EXIF, a coherent sensor-noise floor, one JPEG save
  morphed_curry.jpg   - the same scene with a spliced and duplicated region,
                        an editor Software tag, and a second JPEG save
  ai_generated.png    - 1024x1024, no sensor noise, Stable Diffusion PNG chunks

These are synthetic stand-ins, not photographs. They are built so the pipeline
has something real to measure; they are not a substitute for evaluating the
detector on real customer images.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, PngImagePlugin

OUT = Path(__file__).resolve().parent.parent / "samples"
RNG = np.random.default_rng(20260920)


def _plate(width: int = 1280, height: int = 960) -> Image.Image:
    """A curry-on-a-plate scene: warm background, white plate, sauce, garnish."""
    img = Image.new("RGB", (width, height), (108, 84, 62))
    draw = ImageDraw.Draw(img)

    # Table grain.
    for i in range(0, height, 7):
        shade = 100 + int(14 * np.sin(i / 23.0))
        draw.line([(0, i), (width, i)], fill=(shade, shade - 22, shade - 40), width=4)

    cx, cy, r = width // 2, height // 2, int(min(width, height) * 0.40)
    draw.ellipse([cx - r - 14, cy - r + 10, cx + r + 14, cy + r + 30], fill=(58, 44, 34))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(238, 236, 231))
    draw.ellipse(
        [cx - int(r * 0.82), cy - int(r * 0.82), cx + int(r * 0.82), cy + int(r * 0.82)],
        fill=(226, 223, 216),
    )

    # Curry.
    inner = int(r * 0.66)
    draw.ellipse([cx - inner, cy - inner, cx + inner, cy + inner], fill=(176, 84, 30))
    for _ in range(190):
        angle = RNG.uniform(0, 2 * np.pi)
        dist = RNG.uniform(0, inner * 0.93)
        px, py = cx + dist * np.cos(angle), cy + dist * np.sin(angle)
        size = RNG.uniform(7, 26)
        tone = RNG.integers(0, 3)
        colour = [(206, 118, 44), (146, 62, 24), (232, 176, 88)][tone]
        draw.ellipse([px - size, py - size, px + size, py + size], fill=colour)

    # Coriander.
    for _ in range(34):
        angle = RNG.uniform(0, 2 * np.pi)
        dist = RNG.uniform(0, inner * 0.8)
        px, py = cx + dist * np.cos(angle), cy + dist * np.sin(angle)
        draw.ellipse([px - 5, py - 3, px + 5, py + 3], fill=(74, 128, 52))

    return img.filter(ImageFilter.GaussianBlur(0.6))


def _add_sensor_noise(img: Image.Image, sigma: float = 3.1) -> Image.Image:
    """One coherent noise floor across the frame, as a single sensor produces."""
    arr = np.asarray(img, dtype=np.float32)
    # Mild luminance dependence, like real photon shot noise.
    luma = arr.mean(axis=2, keepdims=True) / 255.0
    noise = RNG.normal(0, sigma, arr.shape) * (0.55 + 0.75 * luma)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def _camera_exif(software: str | None = None) -> Image.Exif:
    exif = Image.Exif()
    exif[271] = "samsung"                      # Make
    exif[272] = "SM-S928B"                     # Model
    exif[306] = "2026:09:20 13:42:11"          # DateTime
    exif[274] = 1                              # Orientation
    if software:
        exif[305] = software                   # Software
    return exif


def make_real(path: Path) -> None:
    img = _add_sensor_noise(_plate())
    img.save(path, "JPEG", quality=88, exif=_camera_exif(), subsampling=2)


def make_morphed(path: Path, base: Path) -> None:
    """Edit the authentic photo: duplicate a patch, splice in a cleaner region."""
    with Image.open(base) as opened:
        img = opened.convert("RGB").copy()

    w, h = img.size
    cx, cy = w // 2, h // 2

    # 1. Copy-move: duplicate a chunk of curry to fake a larger spill.
    patch = img.crop((cx - 210, cy - 190, cx - 30, cy - 10))
    img.paste(patch, (cx + 40, cy + 30))

    # 2. Splice: paste a region that has been denoised and re-toned, so its
    #    noise floor and compression history differ from the rest of the frame.
    region = img.crop((cx - 120, cy + 60, cx + 160, cy + 260))
    region = region.filter(ImageFilter.GaussianBlur(1.4))
    arr = np.asarray(region, dtype=np.float32)
    arr[..., 0] = np.clip(arr[..., 0] * 0.55, 0, 255)   # darken to fake burning
    arr[..., 1] = np.clip(arr[..., 1] * 0.48, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] * 0.42, 0, 255)
    img.paste(Image.fromarray(arr.astype(np.uint8)), (cx - 120, cy + 60))

    # 3. Second save at a different quality, with an editor's Software tag.
    img.save(
        path,
        "JPEG",
        quality=72,
        exif=_camera_exif(software="Adobe Photoshop 26.2 (Windows)"),
        subsampling=2,
    )


def make_ai_generated(path: Path) -> None:
    """Synthetic-looking render at a stock model size, with generator metadata."""
    size = 1024
    img = _plate(size, size).filter(ImageFilter.GaussianBlur(1.1))

    # Flatten local texture the way a decoder does - no sensor noise at all.
    arr = np.asarray(img, dtype=np.float32)
    arr = arr * 0.92 + arr.mean(axis=(0, 1), keepdims=True) * 0.08
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    meta = PngImagePlugin.PngInfo()
    meta.add_text(
        "parameters",
        "photo of spilled curry in a delivery box, damaged food, realistic, 8k\n"
        "Negative prompt: text, watermark\n"
        "Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 1884422031, "
        "Size: 1024x1024, Model: sd_xl_base_1.0",
    )
    meta.add_text("Software", "Stable Diffusion WebUI (AUTOMATIC1111)")
    img.save(path, "PNG", pnginfo=meta)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    real = OUT / "real_curry.jpg"
    make_real(real)
    make_morphed(OUT / "morphed_curry.jpg", real)
    make_ai_generated(OUT / "ai_generated.png")
    for path in sorted(OUT.iterdir()):
        print(f"  {path.name:<22} {path.stat().st_size / 1024:6.1f} KB")


if __name__ == "__main__":
    main()
