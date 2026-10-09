"""
ML-based text placement on certificate templates.

First principles:
- A certificate is an image with structured text regions.
- The "name" field has predictable visual properties: it's usually a
  large, centered, prominent text region, often the widest single word/
  phrase near the middle of the image.
- Rather than hardcoding pixel coordinates, we LEARN a classifier that
  scores candidate text regions and picks the best "name" region.
- We train on synthetic templates so the model generalizes to unseen
  designs, and we cache the trained model to disk.
"""

import os
import re
import joblib
import numpy as np
import cv2
import pytesseract
from dataclasses import dataclass
from typing import List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from app.config import PLACEMENT_MODEL_PATH


@dataclass
class TextRegion:
    x: int
    y: int
    w: int
    h: int
    text: str
    ocr_conf: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

import json
from pathlib import Path

def find_all_placements(img_bgr) -> dict:
    """
    Return a dict of field -> (x, y, w, h) for name, course, date, etc.

    Priority:
    1. Sidecar file next to the template (exact, from the generator)
    2. OCR-detected {{...}} placeholders
    3. ML name detection + heuristics
    """
    h, w = img_bgr.shape[:2]

    # 1) Try sidecar
    # The template path isn't passed in, but our default template has a
    # well-known sidecar. If you support multiple templates, thread the
    # path through here.
    from app.config import DEFAULT_TEMPLATE
    sidecar = Path(str(DEFAULT_TEMPLATE)).with_suffix(".placements.json")
    if sidecar.exists():
        try:
            raw = json.loads(sidecar.read_text())
            return {k: tuple(v) for k, v in raw.items()}
        except Exception:
            pass

    # 2) Fall back to OCR-detected placeholders
    placeholders = find_placeholder_boxes(img_bgr)
    result = dict(placeholders)

    if "name" not in result:
        result["name"] = find_name_placement(img_bgr)

    # 3) Heuristics for missing fields
    if "course" not in result:
        name_box = result["name"]
        regions = _detect_text_regions(img_bgr)
        below = [
            r for r in regions
            if r.y > name_box[1] + name_box[3] * 0.5
            and abs(r.cx / w - 0.5) < 0.35
        ]
        if below:
            r = min(below, key=lambda r: r.y)
            result["course"] = (r.x, r.y, r.w, r.h)

    if "date" not in result:
        regions = _detect_text_regions(img_bgr)
        bottom = [r for r in regions if r.cy / h > 0.7]
        if bottom:
            r = max(bottom, key=lambda r: r.cx)
            result["date"] = (r.x, r.y, r.w, r.h)

    return result


def _detect_text_regions(img_bgr: np.ndarray) -> List[TextRegion]:
    """Use MSER + OCR to find text regions with their recognized content."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # MSER is robust to varying contrast and fonts
    mser = cv2.MSER_create()
    mser.setMinArea(80)
    mser.setMaxArea(int(h * w * 0.05))
    regions, _ = mser.detectRegions(gray)

    boxes = []
    for pts in regions:
        x, y, bw, bh = cv2.boundingRect(pts.reshape(-1, 1, 2))
        # filter degenerate boxes
        if bw < 8 or bh < 8:
            continue
        if bw / bh > 20 or bh / bw > 20:
            continue
        boxes.append([x, y, bw, bh])

    if not boxes:
        return []

    # Merge overlapping/nearby boxes into word/line blocks
    boxes = np.array(boxes)
    merged = _merge_boxes(boxes, gap_x=12, gap_y=6)

    out: List[TextRegion] = []
    for (x, y, bw, bh) in merged:
        # Pad a little for OCR
        pad = 3
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad); y1 = min(h, y + bh + pad)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        try:
            data = pytesseract.image_to_data(
                roi, output_type=pytesseract.Output.DICT,
                config="--psm 7"
            )
        except Exception:
            continue
        texts = [t for t in data.get("text", []) if t and t.strip()]
        confs = [float(c) for c in data.get("conf", []) if str(c).replace('.', '', 1).isdigit()]
        if not texts:
            continue
        text = " ".join(texts).strip()
        conf = float(np.mean([c for c in confs if c >= 0])) if confs else 0.0
        out.append(TextRegion(x, y, bw, bh, text, conf))

    return out


def _merge_boxes(boxes: np.ndarray, gap_x: int, gap_y: int) -> List[Tuple[int, int, int, int]]:
    """Merge boxes that are horizontally/vertically close (greedy union-find-ish)."""
    boxes = boxes.tolist()
    changed = True
    while changed:
        changed = False
        result = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            x1, y1, w1, h1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                x2, y2, w2, h2 = boxes[j]
                # overlap or close on both axes
                if (x1 - gap_x < x2 + w2 and x2 - gap_x < x1 + w1 and
                        y1 - gap_y < y2 + h2 and y2 - gap_y < y1 + h1):
                    nx = min(x1, x2)
                    ny = min(y1, y2)
                    nw = max(x1 + w1, x2 + w2) - nx
                    nh = max(y1 + h1, y2 + h2) - ny
                    boxes[i] = [nx, ny, nw, nh]
                    x1, y1, w1, h1 = boxes[i]
                    used[j] = True
                    changed = True
        boxes = [b for i, b in enumerate(boxes) if not used[i]]
    return [tuple(b) for b in boxes]


def _features(region: TextRegion, img_w: int, img_h: int) -> List[float]:
    """Feature vector for the placement classifier."""
    cx_n = region.cx / img_w
    cy_n = region.cy / img_h
    w_n = region.w / img_w
    h_n = region.h / img_h
    ar = region.w / max(1, region.h)
    text_len = len(region.text)
    has_colon = 1.0 if ":" in region.text else 0.0
    has_brace = 1.0 if ("{" in region.text or "}" in region.text) else 0.0
    is_short = 1.0 if text_len <= 30 else 0.0
    is_centered = 1.0 - abs(cx_n - 0.5) * 2  # 1.0 = perfectly centered
    vertical_band = 1.0 if 0.25 < cy_n < 0.75 else 0.0
    return [cx_n, cy_n, w_n, h_n, ar, text_len, region.ocr_conf,
            has_colon, has_brace, is_short, is_centered, vertical_band]


def _synthesize_training_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a synthetic dataset that mirrors the geometry of real certificate
    text regions. We generate many plausible layouts, label the "name"
    region using the ground-truth rule (largest centered region in the
    middle band), and train a classifier to recover that rule from
    features alone. This lets the model generalize to unseen templates.
    """
    rng = np.random.default_rng(42)
    X, y = [], []

    for _ in range(4000):
        img_w, img_h = 1600, 1100
        n_regions = rng.integers(3, 9)
        regions = []
        for _ in range(n_regions):
            # Random plausible text block geometry
            w = rng.integers(60, 700)
            h = rng.integers(20, 90)
            x = rng.integers(0, img_w - w)
            y = rng.integers(0, img_h - h)
            conf = float(rng.uniform(40, 99))
            # Text content heuristics
            kind = rng.integers(0, 5)
            if kind == 0:
                text = "CERTIFICATE OF ACHIEVEMENT"
            elif kind == 1:
                text = "John Doe"
            elif kind == 2:
                text = "Course: Advanced Python"
            elif kind == 3:
                text = "Date: 2024-01-01"
            else:
                text = "Signature"
            regions.append(TextRegion(x, y, w, h, text, conf))

        # Ground-truth rule: name = largest region whose centroid is in
        # the middle vertical band and roughly centered horizontally.
        candidates = [
            r for r in regions
            if 0.25 < (r.cy / img_h) < 0.75 and abs((r.cx / img_w) - 0.5) < 0.3
        ]
        if not candidates:
            candidates = regions
        # Prefer wide regions (names are usually wide-ish)
        name_region = max(candidates, key=lambda r: r.w)

        for r in regions:
            X.append(_features(r, img_w, img_h))
            y.append(1 if r is name_region else 0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_placement_model(force: bool = False) -> RandomForestClassifier:
    """Train (or load cached) placement model."""
    if PLACEMENT_MODEL_PATH.exists() and not force:
        return joblib.load(PLACEMENT_MODEL_PATH)

    X, y = _synthesize_training_data()
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X, y)
    joblib.dump(clf, PLACEMENT_MODEL_PATH)
    return clf


_model_cache: Optional[RandomForestClassifier] = None


def _get_model() -> RandomForestClassifier:
    global _model_cache
    if _model_cache is None:
        _model_cache = train_placement_model()
    return _model_cache


# ---------- Deterministic placeholder path ----------

PLACEHOLDER_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def find_placeholder_boxes(img_bgr: np.ndarray) -> dict:
    """
    If the template contains literal placeholders like {{name}}, locate them.
    Returns {field: (x, y, w, h)}. This is the deterministic path.
    """
    regions = _detect_text_regions(img_bgr)
    found = {}
    for r in regions:
        m = PLACEHOLDER_RE.search(r.text)
        if m:
            found[m.group(1)] = (r.x, r.y, r.w, r.h)
    return found


# ---------- Public API ----------

def find_name_placement(img_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Return (x, y, w, h) — the box where the recipient name should be drawn.

    Strategy:
    1. If an explicit {{name}} placeholder exists, use it (deterministic).
    2. Otherwise, run ML model over detected text regions and pick the
       highest-probability "name" region.
    3. Fallback: centered band heuristic.
    """
    h, w = img_bgr.shape[:2]

    placeholders = find_placeholder_boxes(img_bgr)
    if "name" in placeholders:
        return placeholders["name"]

    regions = _detect_text_regions(img_bgr)
    if not regions:
        # Fallback: middle of the image
        return (int(w * 0.2), int(h * 0.45), int(w * 0.6), int(h * 0.08))

    model = _get_model()
    X = np.array([_features(r, w, h) for r in regions], dtype=np.float32)
    probs = model.predict_proba(X)[:, 1]
    best_idx = int(np.argmax(probs))
    best = regions[best_idx]

    # If model is not confident, use heuristic fallback
    if probs[best_idx] < 0.35:
        centered = [r for r in regions if abs(r.cx / w - 0.5) < 0.25]
        if centered:
            best = max(centered, key=lambda r: r.w)

    return (best.x, best.y, best.w, best.h)


def find_all_placements(img_bgr: np.ndarray) -> dict:
    """
    Return a dict of field -> (x, y, w, h) for name, course, date, etc.
    Uses placeholders when available; otherwise ML for name and heuristics
    for the others.
    """
    h, w = img_bgr.shape[:2]
    placeholders = find_placeholder_boxes(img_bgr)
    result = dict(placeholders)

    if "name" not in result:
        result["name"] = find_name_placement(img_bgr)

    # Course: prefer placeholder, else a region just below the name
    if "course" not in result:
        name_box = result["name"]
        regions = _detect_text_regions(img_bgr)
        below = [
            r for r in regions
            if r.y > name_box[1] + name_box[3] * 0.5
            and abs(r.cx / w - 0.5) < 0.35
        ]
        if below:
            r = min(below, key=lambda r: r.y)
            result["course"] = (r.x, r.y, r.w, r.h)

    # Date: bottom-right-ish region
    if "date" not in result:
        regions = _detect_text_regions(img_bgr)
        bottom = [r for r in regions if r.cy / h > 0.7]
        if bottom:
            r = max(bottom, key=lambda r: r.cx)
            result["date"] = (r.x, r.y, r.w, r.h)

    return result