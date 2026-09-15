"""
clip_bbox.py — Grounding DINO open-vocabulary bounding-box detection for the
Kinova Gen 3 simulation.

Uses **Grounding DINO** (DINO with Grounded Pre-Training) to detect scene
objects in camera images by text query and annotate them with coloured
bounding boxes.

First run will download the Grounding DINO weights (~700 MB).

Dependencies (install once):
    pip install torch transformers Pillow
"""

import os
import numpy as np
from PIL import Image

# ── Lazy-loaded model singletons ─────────────────────────────────────────────
_model = None
_processor = None
_device = None
GDINO_MODEL = "IDEA-Research/grounding-dino-tiny"

# Local cache directory (next to this script) so weights persist across runs
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")

# Detection confidence thresholds
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.15


def _ensure_model(console=None):
    """Load Grounding DINO model + processor once (lazy singleton)."""
    global _model, _processor, _device
    if _model is not None:
        return

    cached = os.path.isdir(os.path.join(_CACHE_DIR, "models--IDEA-Research--grounding-dino-tiny"))
    if console:
        if cached:
            console.write(f"[GDINO] Loading Grounding DINO from local cache...")
        else:
            console.write(f"[GDINO] Downloading Grounding DINO model ({GDINO_MODEL})...")
            console.write("[GDINO] (One-time download, ~700 MB. Cached locally after this.)")

    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _processor = AutoProcessor.from_pretrained(GDINO_MODEL, cache_dir=_CACHE_DIR)
    _model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL, cache_dir=_CACHE_DIR).to(_device)
    _model.eval()

    if console:
        console.write(f"[GDINO] Grounding DINO loaded on {_device.upper()}.")


# ── Text query mapping — gives descriptive prompts per object ────────────────
_GDINO_QUERIES = {
    "block":         "a small colored cube block",
    "pyramid":       "a pyramid",
    "cylinder":      "a cylinder",
    "bottle":        "a bottle",
    "remote":        "a black TV remote control with buttons",
    "mug":           "a mug with a handle",
    "box":           "an open top box tray",
    "pill_bottle":   "a small orange cylindrical prescription pill bottle with white cap",
    "phone":         "a flat black smartphone",
    "book":          "a blue rectangular hardcover book",
    "cup":           "a light blue translucent drinking cup",
    "water_cup":     "a large red mug with a handle filled with water",
    "water_bottle":  "a blue cylindrical water bottle with grey cap",
    "toothbrush":    "a white toothbrush with blue bristle head",
    "soup_can":      "a large green soup can with gray label band",
    "person":        "a stick figure person",
}

# Reverse map: words that appear in free-text → canonical object name.
# Used to detect when a user phrase like "red block" or "small cube"
# refers to a known object, so we can use the richer GDINO query.
_KEYWORD_TO_CANONICAL = {}
for _canon, _query in _GDINO_QUERIES.items():
    _KEYWORD_TO_CANONICAL[_canon] = _canon
    for _w in _query.lower().split():
        if _w not in ("a", "an", "the"):
            _KEYWORD_TO_CANONICAL[_w] = _canon
# Extra keyword aliases for common user phrasing
_KEYWORD_TO_CANONICAL.update({
    "cube":        "block",
    "brick":       "block",
    "cyl":         "cylinder",
    "tube":        "cylinder",
    "triangle":    "pyramid",
    "cone":        "pyramid",
    "tray":        "box",
    "crate":       "box",
    "bin":         "box",
    "control":     "remote",
    "controller":  "remote",
    "pill":        "pill_bottle",
    "medication":  "pill_bottle",
    "medicine":    "pill_bottle",
    "prescription": "pill_bottle",
    "pills":       "pill_bottle",
    "smartphone":  "phone",
    "cellphone":   "phone",
    "mobile":      "phone",
    "novel":       "book",
    "textbook":    "book",
    "cup":         "cup",
    "glass":       "cup",
    "tumbler":     "cup",
    "water_cup":   "water_cup",
    "watercup":    "water_cup",
    "water":       "water_bottle",
    "waterbottle": "water_bottle",
    "hydration":   "water_bottle",
    "brush":       "toothbrush",
    "teeth":       "toothbrush",
    "dental":      "toothbrush",
    "soup":        "soup_can",
    "can":         "soup_can",
    "canned":      "soup_can",
    "food":        "soup_can",
    "person":      "person",
    "human":       "person",
    "patient":     "person",
    "user":        "person",
    "man":         "person",
    "woman":       "person",
})


def _query_for(name: str) -> str:
    """Return a descriptive Grounding DINO text query for *name*.

    If *name* is a known canonical key (e.g. 'block') we return the
    curated prompt.  If it's a free-text phrase (e.g. 'red block',
    'small cube') we check whether any word maps to a known object
    and return that object's curated query, so Grounding DINO gets
    the most distinctive description possible.

    Falls back to ``'a <name>'`` for truly novel objects.
    """
    # Direct lookup
    if name in _GDINO_QUERIES:
        return _GDINO_QUERIES[name]

    # Check individual words against keyword map
    for word in name.lower().split():
        canon = _KEYWORD_TO_CANONICAL.get(word)
        if canon and canon in _GDINO_QUERIES:
            return _GDINO_QUERIES[canon]

    # Fallback: prepend article if not already present
    lower = name.lower().strip()
    if lower.startswith(("a ", "an ", "the ")):
        return name
    return f"a {name}"


# ── Colour palette (R, G, B) — bright and easily distinguishable ────────────
_PALETTE = [
    (255,  50,  50),   # 0  Red
    ( 50, 150, 255),   # 1  Blue
    ( 50, 255,  50),   # 2  Green
    (255, 200,  50),   # 3  Yellow
    (200,  50, 255),   # 4  Purple
    (255, 128,   0),   # 5  Orange
    (  0, 255, 200),   # 6  Cyan
    (255, 100, 200),   # 7  Pink
]

_COLOR_NAMES = [
    "RED", "BLUE", "GREEN", "YELLOW", "PURPLE", "ORANGE", "CYAN", "PINK",
]


def _bbox_color(index: int) -> tuple:
    return _PALETTE[index % len(_PALETTE)]


def _bbox_color_name(index: int) -> str:
    return _COLOR_NAMES[index % len(_COLOR_NAMES)]


# ── Grounding DINO detection ─────────────────────────────────────────────────

def _match_label_to_query(label: str, text_queries: list) -> int:
    """Return index of the text_query that best matches a detection label.

    Grounding DINO returns labels that are text spans from the input prompt.
    We use a multi-signal approach:
      1. Exact substring containment (highest priority)
      2. Normalized word overlap (intersection / label words)
      3. Tie-break by query length (prefer more specific queries)
    """
    label_clean = label.lower().strip()
    label_words = set(label_clean.split())
    if not label_words:
        return -1

    # Pass 1: exact substring — label appears inside query or vice versa
    substring_hits = []
    for i, q in enumerate(text_queries):
        q_lower = q.lower()
        if label_clean in q_lower or q_lower in label_clean:
            substring_hits.append(i)
    if len(substring_hits) == 1:
        return substring_hits[0]
    if len(substring_hits) > 1:
        # Multiple substring matches — prefer the one with highest
        # normalized word overlap (more specific match)
        best_i, best_score = substring_hits[0], -1.0
        for i in substring_hits:
            q_words = set(text_queries[i].lower().split())
            score = len(label_words & q_words) / len(label_words)
            if score > best_score or (score == best_score and len(q_words) > len(set(text_queries[best_i].lower().split()))):
                best_score = score
                best_i = i
        return best_i

    # Pass 2: normalized word overlap
    best_idx, best_score = -1, 0.0
    for i, q in enumerate(text_queries):
        q_words = set(q.lower().split())
        overlap = len(label_words & q_words)
        # Normalize by label word count so "block" matching
        # "a small cube block" scores 1.0 not 0.25
        score = overlap / len(label_words) if label_words else 0.0
        if score > best_score or (score == best_score and overlap > 0
                                   and len(q_words) > len(set(text_queries[best_idx].lower().split()))):
            best_score = score
            best_idx = i
    # Require at least one word overlap
    return best_idx if best_score > 0 else -1


def _detect_objects(rgb: np.ndarray, object_names: list,
                    threshold: float = BOX_THRESHOLD,
                    text_threshold: float = TEXT_THRESHOLD) -> dict:
    """Run Grounding DINO on an HxWx3 uint8 image and return per-object bounding boxes.

    Returns {name: {"bbox": (x1,y1,x2,y2)|None,
                    "visible": bool,
                    "score": float,
                    "all_boxes": [(x1,y1,x2,y2,score), ...]}}
    """
    import torch

    pil_img = Image.fromarray(rgb)
    h, w = rgb.shape[:2]
    text_queries = [_query_for(n) for n in object_names]

    # Grounding DINO expects period-separated phrases as a single string
    text_prompt = " . ".join(text_queries) + " ."

    inputs = _processor(images=pil_img, text=text_prompt, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)

    results = _processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[(h, w)],
    )[0]

    det_scores = results["scores"].cpu().numpy()
    det_labels = results["text_labels"]    # list of decoded text spans
    det_boxes = results["boxes"].cpu().numpy()   # (N, 4) x1, y1, x2, y2 pixel

    # Group detections by matched object name
    pad = 4
    per_object = {i: [] for i in range(len(object_names))}
    for score, label, box in zip(det_scores, det_labels, det_boxes):
        idx = _match_label_to_query(label, text_queries)
        if idx < 0:
            print(f"  [GDINO] Unmatched label '{label}' (score={score:.3f}) — skipped")
            continue
        x1, y1, x2, y2 = box
        per_object[idx].append((
            max(0, int(x1) - pad),
            max(0, int(y1) - pad),
            min(w - 1, int(x2) + pad),
            min(h - 1, int(y2) + pad),
            float(score),
        ))
        print(f"  [GDINO] '{label}' → '{object_names[idx]}' (score={score:.3f})")

    detections = {}
    for i, name in enumerate(object_names):
        obj_boxes = per_object[i]
        if not obj_boxes:
            detections[name] = {"bbox": None, "visible": False, "score": 0.0,
                                "all_boxes": []}
            continue
        # Sort by score descending
        obj_boxes.sort(key=lambda b: b[4], reverse=True)
        best = obj_boxes[0]
        detections[name] = {
            "bbox": best[:4],
            "visible": True,
            "score": best[4],
            "all_boxes": obj_boxes,
        }
    return detections


# ── Drawing helpers ──────────────────────────────────────────────────────────

def _draw_rect(img: np.ndarray, x1, y1, x2, y2, color, thickness=3):
    """Draw a rectangle outline on an HxWx3 uint8 array (in-place)."""
    h, w = img.shape[:2]
    c = np.array(color, dtype=np.uint8)
    img[max(0, y1):min(h, y1 + thickness), max(0, x1):min(w, x2 + 1)] = c
    img[max(0, y2 - thickness + 1):min(h, y2 + 1), max(0, x1):min(w, x2 + 1)] = c
    img[max(0, y1):min(h, y2 + 1), max(0, x1):min(w, x1 + thickness)] = c
    img[max(0, y1):min(h, y2 + 1), max(0, x2 - thickness + 1):min(w, x2 + 1)] = c


def _draw_label(img: np.ndarray, x1, y1, color, text: str, font_size=16):
    """Draw a text label with coloured background above the bounding box using PIL."""
    from PIL import Image as _PILImage, ImageDraw, ImageFont

    pil = _PILImage.fromarray(img)
    draw = ImageDraw.Draw(pil)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 3
    lx1 = x1
    ly1 = max(0, y1 - th - 2 * pad)
    lx2 = lx1 + tw + 2 * pad
    ly2 = y1

    draw.rectangle([lx1, ly1, lx2, ly2], fill=color)

    # White or black text depending on background brightness
    brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    txt_color = (0, 0, 0) if brightness > 150 else (255, 255, 255)
    draw.text((lx1 + pad, ly1 + pad), text, fill=txt_color, font=font)

    img[:] = np.array(pil)[:, :, :3]


def _draw_corner_marks(img: np.ndarray, x1, y1, x2, y2, color,
                       length=12, thickness=3):
    """Draw L-shaped corner marks for a more distinctive bounding box."""
    h, w = img.shape[:2]
    c = np.array(color, dtype=np.uint8)
    L = length
    img[max(0, y1):min(h, y1 + thickness), max(0, x1):min(w, x1 + L)] = c
    img[max(0, y1):min(h, y1 + L), max(0, x1):min(w, x1 + thickness)] = c
    img[max(0, y1):min(h, y1 + thickness), max(0, x2 - L + 1):min(w, x2 + 1)] = c
    img[max(0, y1):min(h, y1 + L), max(0, x2 - thickness + 1):min(w, x2 + 1)] = c
    img[max(0, y2 - thickness + 1):min(h, y2 + 1), max(0, x1):min(w, x1 + L)] = c
    img[max(0, y2 - L + 1):min(h, y2 + 1), max(0, x1):min(w, x1 + thickness)] = c
    img[max(0, y2 - thickness + 1):min(h, y2 + 1), max(0, x2 - L + 1):min(w, x2 + 1)] = c
    img[max(0, y2 - L + 1):min(h, y2 + 1), max(0, x2 - thickness + 1):min(w, x2 + 1)] = c


def annotate_image(rgb: np.ndarray, detections: dict, color_map: dict) -> np.ndarray:
    """Return a copy of *rgb* with coloured bounding boxes drawn on it."""
    out = rgb.copy()
    for name, info in detections.items():
        if not info["visible"]:
            continue
        x1, y1, x2, y2 = info["bbox"]
        color = color_map.get(name, (255, 255, 255))
        _draw_rect(out, x1, y1, x2, y2, color, thickness=3)
        _draw_corner_marks(out, x1, y1, x2, y2, color, length=14, thickness=4)
        label_text = f"{name} {info['score']:.2f}"
        _draw_label(out, x1, y1, color, label_text)
    return out


# ── Public API ───────────────────────────────────────────────────────────────

def clip_bbox(loaded_objects: dict,
              robot_id: int,
              ee_link_index: int,
              bird_img: np.ndarray,
              ee_img: np.ndarray,
              iso_img: np.ndarray,
              width: int = 640,
              height: int = 480,
              console=None):
    """Run Grounding DINO object detection on three camera views,
    annotate the images with coloured bounding boxes, and return a text
    legend for the LLM prompt.

    Parameters
    ----------
    loaded_objects : dict
        {name: pybullet_body_id} of scene objects.
        (body IDs are unused — detection is purely vision-based.)
    robot_id, ee_link_index : int
        Accepted for call-signature compatibility (unused).
    bird_img, ee_img, iso_img : np.ndarray
        HxWx3 uint8 camera snapshots (raw, un-annotated).
    width, height : int
        Reserved (resolution is inferred from the arrays).
    console : CommandConsole or None
        Optional console for status messages.

    Returns
    -------
    ann_bird, ann_ee, ann_iso : np.ndarray
        Annotated copies of the three camera images.
    legend_text : str
        Textual description of the colour assignments for the LLM prompt.
    """
    _ensure_model(console)

    # Deterministic colour assignment (alphabetical)
    names_sorted = sorted(loaded_objects.keys())
    color_map    = {n: _bbox_color(i)      for i, n in enumerate(names_sorted)}
    color_labels = {n: _bbox_color_name(i) for i, n in enumerate(names_sorted)}

    if console:
        queries = {n: _query_for(n) for n in names_sorted}
        console.write("[GDINO] Running Grounding DINO detection...")
        for n, q in queries.items():
            console.write(f"  Query: \"{q}\" → {color_labels[n]} box")

    # Detect objects in each view
    bird_dets = _detect_objects(bird_img, names_sorted)
    ee_dets   = _detect_objects(ee_img,   names_sorted)
    iso_dets  = _detect_objects(iso_img,  names_sorted)

    # Annotate images
    ann_bird = annotate_image(bird_img, bird_dets, color_map)
    ann_ee   = annotate_image(ee_img,   ee_dets,   color_map)
    ann_iso  = annotate_image(iso_img,  iso_dets,   color_map)

    if console:
        console.write("[GDINO] Detection results:")
        for name in names_sorted:
            vis = []
            if bird_dets[name]["visible"]:
                vis.append(f"bird({bird_dets[name]['score']:.2f})")
            if ee_dets[name]["visible"]:
                vis.append(f"EE({ee_dets[name]['score']:.2f})")
            if iso_dets[name]["visible"]:
                vis.append(f"iso({iso_dets[name]['score']:.2f})")
            console.write(
                f"  {name}: {color_labels[name]} box — "
                f"detected in: {', '.join(vis) or 'none'}"
            )

    # Build legend text for the LLM prompt
    legend_lines = [
        "BOUNDING-BOX COLOUR LEGEND (drawn on the attached images):"
    ]
    for name in names_sorted:
        legend_lines.append(f'  \u2022 "{name}" \u2014 {color_labels[name]} bounding box')
    legend_lines.append(
        "\nEach object is highlighted with a uniquely coloured bounding box "
        "in every camera view where it is visible.  Use the boxes to identify "
        "which object is which and plan interactions accordingly."
    )
    legend_text = "\n".join(legend_lines)

    return ann_bird, ann_ee, ann_iso, legend_text
