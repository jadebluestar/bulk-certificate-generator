import os
import uuid
from datetime import datetime
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from app.config import DEFAULT_TEMPLATE, OUTPUT_DIR
from app.placement import find_all_placements


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/timesbd.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _fit_font(draw: ImageDraw.ImageDraw, text: str, box_w: int, box_h: int,
              max_size: int = 120) -> ImageFont.FreeTypeFont:
    """Shrink font until text fits inside the box (with padding)."""
    for size in range(max_size, 8, -2):
        font = _load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= box_w * 0.95 and th <= box_h * 1.4:
            return font
    return _load_font(8)


def _draw_centered(draw: ImageDraw.ImageDraw, text: str, box: tuple,
                   fill=(20, 20, 20), max_size: int = 120):
    x, y, w, h = box
    font = _fit_font(draw, text, w, h, max_size=max_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x + (w - tw) / 2 - bbox[0]
    ty = y + (h - th) / 2 - bbox[1]
    draw.text((tx, ty), text, font=font, fill=fill)


def generate_certificate(
    recipient_name: str,
    course_name: Optional[str] = None,
    issue_date: Optional[str] = None,
    template_path: Optional[str] = None,
) -> str:
    """
    Render a certificate PNG for one recipient.
    Returns the file path. Raises on failure (caller catches per-recipient).
    """
    import cv2
    import numpy as np

    template_path = template_path or str(DEFAULT_TEMPLATE)
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")

    img = Image.open(template_path).convert("RGB")
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Detect where each field lives on the template.
    # This uses the {{...}} placeholders if present, else ML/heuristics.
    placements = find_all_placements(img_bgr)

    draw = ImageDraw.Draw(img)

    # --- Step 1: erase placeholder text so it doesn't show through ---
    # For each field we plan to render (or that has a placeholder we want gone),
    # paint a filled rectangle over the original placeholder area.
    # We use a slightly padded box and sample the border color to match the
    # background so this works on non-white templates too.
    for field in ("name", "course", "date"):
        if field in placements:
            _erase_box(img, placements[field])

    # --- Step 2: draw the actual values on top of the erased areas ---
    # Name is required.
    if "name" in placements:
        _draw_centered(draw, recipient_name, placements["name"], max_size=140)

    # Course and date are optional — only draw if provided.
    # If not provided, the placeholder was already erased in Step 1,
    # so we simply leave the area blank.
    if course_name and "course" in placements:
        _draw_centered(draw, course_name, placements["course"],
                       fill=(60, 60, 60), max_size=60)

    if issue_date and "date" in placements:
        _draw_centered(draw, issue_date, placements["date"],
                       fill=(60, 60, 60), max_size=44)

    # Last-resort fallback: if the template had no detectable placements at all
    if "name" not in placements:
        w, h = img.size
        _draw_centered(draw, recipient_name,
                       (int(w * 0.15), int(h * 0.42), int(w * 0.7), int(h * 0.1)),
                       max_size=140)

    filename = f"{uuid.uuid4().hex}.png"
    out_path = os.path.join(OUTPUT_DIR, filename)
    img.save(out_path, "PNG")
    return out_path

def _sample_bg_color(img: Image.Image, box: tuple, pad: int = 6) -> tuple:
    """
    Guess the background color of the region around `box` by sampling pixels
    just outside its border. This makes the erase step work on templates whose
    background isn't pure white.
    Returns an (R, G, B) tuple.
    """
    import numpy as np

    x, y, w, h = box
    W, H = img.size
    arr = np.array(img)

    # Sample a thin ring just outside the box
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)

    ring = []
    if y0 < y:
        ring.append(arr[y0:y, x0:x1].reshape(-1, 3))
    if y1 > y + h:
        ring.append(arr[y + h:y1, x0:x1].reshape(-1, 3))
    if x0 < x:
        ring.append(arr[y:y + h, x0:x].reshape(-1, 3))
    if x1 > x + w:
        ring.append(arr[y:y + h, x + w:x1].reshape(-1, 3))

    if not ring:
        return (255, 255, 255)

    pixels = np.concatenate(ring, axis=0)
    # Median is robust to a few stray dark pixels (e.g. the border)
    median = np.median(pixels, axis=0).astype(int)
    return tuple(int(c) for c in median)


def _erase_box(img, box, pad_x=6, pad_y=18):   # <- pad_y much bigger now
    x, y, w, h = box
    bg = _sample_bg_color(img, box, pad=8)
    erase = (
        max(0, x - pad_x),
        max(0, y - pad_y),
        min(img.size[0], x + w + pad_x),
        min(img.size[1], y + h + pad_y),
    )
    ImageDraw.Draw(img).rectangle(erase, fill=bg)