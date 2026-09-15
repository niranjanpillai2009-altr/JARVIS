"""
GR-ConvNet v2 grasp detection wrapper.

Runs inference on an RGB-D image pair using a pre-trained GR-ConvNet model
(https://github.com/skumra/robotic-grasping) and returns grasp candidates
in a format compatible with the simulation pipeline.

Setup:
  git clone https://github.com/skumra/robotic-grasping.git
  pip install torch torchvision scikit-image

Then either:
  - Set GRCONVNET_REPO env var to the cloned repo path, OR
  - Place the repo folder next to this file (sibling directory).

Pre-trained models live under <repo>/trained-models/.
"""
from __future__ import annotations

import os
import sys
import math
import numpy as np
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Locate the robotic-grasping repo so we can import its modules
# ---------------------------------------------------------------------------
_REPO_DIR: Optional[str] = None


def _find_repo() -> str:
    global _REPO_DIR
    if _REPO_DIR is not None:
        return _REPO_DIR

    candidates = [
        os.environ.get("GRCONVNET_REPO", ""),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "robotic-grasping"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robotic-grasping"),
    ]
    for c in candidates:
        if c and os.path.isdir(c) and os.path.isfile(os.path.join(c, "inference", "post_process.py")):
            _REPO_DIR = os.path.abspath(c)
            return _REPO_DIR

    raise FileNotFoundError(
        "Cannot find the robotic-grasping repository.\n"
        "Please clone it and either:\n"
        "  - Set the GRCONVNET_REPO environment variable to the repo path, or\n"
        "  - Place it as a sibling folder named 'robotic-grasping' next to this file.\n"
        "  git clone https://github.com/skumra/robotic-grasping.git"
    )


def _ensure_on_path():
    repo = _find_repo()
    if repo not in sys.path:
        sys.path.insert(0, repo)


# ---------------------------------------------------------------------------
# Model cache (load once, reuse)
# ---------------------------------------------------------------------------
_LOADED_MODEL = None
_DEVICE = None


def _load_model(model_path: Optional[str] = None):
    """Load a pre-trained GR-ConvNet model. Cached after first call."""
    global _LOADED_MODEL, _DEVICE
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL, _DEVICE

    import torch

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    repo = _find_repo()
    # The saved checkpoint pickles the full model object, so Python must be
    # able to import the `inference` package that lives inside the repo.
    if repo not in sys.path:
        sys.path.insert(0, repo)

    if model_path is None:
        # Auto-discover a trained model inside the repo
        trained_dir = os.path.join(repo, "trained-models")
        # Prefer cornell RGB-D model
        preferred = [
            "cornell-randsplit-rgbd-grconvnet3-drop1-ch32",
            "cornell-randsplit-rgbd-grconvnet3-drop1-ch16",
            "jacquard-rgbd-grconvnet3-drop0-ch32",
            "jacquard-d-grconvnet3-drop0-ch32",
        ]
        found = None
        for name in preferred:
            p = os.path.join(trained_dir, name)
            if os.path.isdir(p):
                # Look for the epoch_* model file (or *.pt / *.pth)
                for f in os.listdir(p):
                    if f.startswith("epoch_") or f.endswith((".pt", ".pth")):
                        found = os.path.join(p, f)
                        break
            if found:
                break
        if found is None:
            raise FileNotFoundError(
                f"No pre-trained GR-ConvNet model found in {trained_dir}.\n"
                "Expected folders: " + ", ".join(preferred)
            )
        model_path = found

    _LOADED_MODEL = torch.load(model_path, map_location=_DEVICE, weights_only=False)
    _LOADED_MODEL.eval()
    return _LOADED_MODEL, _DEVICE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_grconvnet_grasps(
    rgb: np.ndarray,
    depth: np.ndarray,
    top_k: int = 5,
    model_path: Optional[str] = None,
) -> List[Dict]:
    """
    Run GR-ConvNet inference on an RGB + depth image pair.

    Parameters
    ----------
    rgb : np.ndarray
        HxWx3 uint8 RGB image.
    depth : np.ndarray
        HxW float32 depth image (metres).
    top_k : int
        Maximum number of grasps to return.
    model_path : str, optional
        Path to saved model file.  Auto-discovered if None.

    Returns
    -------
    List of dicts, each with:
        pixel : (u, v)   — grasp centre in the original image coords
        angle : float    — rotation angle in radians
        width : float    — gripper opening width in pixels (original scale)
        quality : float  — confidence score
    """
    _ensure_on_path()

    import torch
    from inference.post_process import post_process_output

    model, device = _load_model(model_path)

    orig_h, orig_w = depth.shape[:2]

    # --- Prepare input (replicating CameraData logic) ---
    # Crop to centre square
    crop = min(orig_h, orig_w)
    top = (orig_h - crop) // 2
    left = (orig_w - crop) // 2
    depth_crop = depth[top:top + crop, left:left + crop].copy()
    rgb_crop = rgb[top:top + crop, left:left + crop].copy()

    # Resize to model input size (224x224 for Cornell models)
    target_size = 224
    from skimage.transform import resize as sk_resize
    depth_resized = sk_resize(depth_crop, (target_size, target_size),
                              preserve_range=True).astype(np.float32)
    rgb_resized = sk_resize(rgb_crop, (target_size, target_size),
                            preserve_range=True).astype(np.float32)

    # Normalise depth (mean-subtract, clip)
    depth_resized = np.clip(depth_resized - depth_resized.mean(), -1.0, 1.0)

    # Normalise RGB to [0, 1]
    rgb_resized = rgb_resized / 255.0

    # Build input tensor: (1, 4, H, W) — depth channel + 3 RGB channels
    depth_t = torch.tensor(depth_resized, dtype=torch.float32).unsqueeze(0)   # (1, H, W)
    rgb_t = torch.tensor(rgb_resized, dtype=torch.float32).permute(2, 0, 1)  # (3, H, W)
    x = torch.cat([depth_t, rgb_t], dim=0).unsqueeze(0).to(device)            # (1, 4, H, W)

    # --- Inference ---
    with torch.no_grad():
        pred = model.predict(x)

    q_img, ang_img, width_img = post_process_output(
        pred["pos"], pred["cos"], pred["sin"], pred["width"],
    )

    # --- Extract grasps ---
    # Use the repo's detect_grasps if available, otherwise manual peak finding
    try:
        from utils.dataset_processing.grasp import detect_grasps
        grasps_raw = detect_grasps(q_img, ang_img, width_img, no_grasps=top_k)
    except ImportError:
        grasps_raw = _fallback_detect(q_img, ang_img, width_img, top_k)

    # Scale pixel coords back to original image space
    scale = crop / target_size
    results = []
    for g in grasps_raw:
        # g.center is (row, col) in the 224x224 space
        row, col = g.center
        orig_row = row * scale + top
        orig_col = col * scale + left
        # Convert row,col to (u,v) where u=col, v=row
        results.append({
            "pixel": (float(orig_col), float(orig_row)),
            "angle": float(g.angle),
            "width": float(g.length) * scale,   # scale width to original
            "quality": float(q_img[int(row), int(col)]),
        })

    results.sort(key=lambda g: g["quality"], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Fallback grasp extraction (when detect_grasps import fails)
# ---------------------------------------------------------------------------

class _SimpleGrasp:
    """Minimal grasp container matching the interface of the repo's Grasp class."""
    def __init__(self, center, angle, length):
        self.center = center
        self.angle = angle
        self.length = length


def _fallback_detect(q_img, ang_img, width_img, top_k):
    """Peak-finding fallback without skimage.feature.peak_local_max."""
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(q_img, size=20) == q_img
    threshold = max(0.2, q_img.max() * 0.5)
    local_max &= q_img > threshold

    rows, cols = np.where(local_max)
    if len(rows) == 0:
        return []

    # Sort by quality descending
    qualities = q_img[rows, cols]
    order = np.argsort(-qualities)
    grasps = []
    for idx in order[:top_k]:
        r, c = int(rows[idx]), int(cols[idx])
        grasps.append(_SimpleGrasp(
            center=(r, c),
            angle=float(ang_img[r, c]),
            length=float(width_img[r, c]),
        ))
    return grasps
