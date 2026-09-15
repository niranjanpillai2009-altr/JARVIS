"""
Lightweight GQ-CNN integration wrapper.

This module provides a small, well-documented scaffold to call a
pre-trained GQ-CNN model (parallel-jaw) on an RGB-D snapshot. It tries
to use an installed `gqcnn` package if available and otherwise raises an
informative ImportError guiding the user to the tutorial and download
scripts.

The entrypoint `detect_gqcnn_grasps(...)` is intentionally conservative:
it returns a list of candidate grasps in the camera frame as dictionaries
with keys: `pixel`=(u,v), `angle`=radians, `width`=meters, `quality`=score.

Note: Running the actual GQ-CNN requires installing the Berkeley
`gqcnn` package, downloading the pre-trained models (e.g. `GQCNN-4.0-PJ`),
and ensuring the examples scripts are available on PATH or by running
from the gqcnn repo root. See the tutorial:
https://berkeleyautomation.github.io/gqcnn/tutorials/tutorial.html#grasp-planning
"""
from __future__ import annotations

import os
import tempfile
import numpy as np
import subprocess
import json
from typing import List, Dict, Optional


def detect_gqcnn_grasps(depth_npy_path: str,
                        segmask_path: Optional[str],
                        camera_intr_path: Optional[str],
                        model_name: str = "GQCNN-4.0-PJ",
                        top_k: int = 5) -> List[Dict]:
    """
    Run a GQ-CNN policy on the supplied depth/segmask files and return
    a list of candidate grasps.

    Parameters
    - depth_npy_path: path to a .npy file containing a HxW float32 depth array (meters).
    - segmask_path: path to a binary PNG segmentation mask (optional; some policies require it).
    - camera_intr_path: path to a .intr camera intrinsics file (optional).
    - model_name: name of the pretrained model to use (default: GQCNN-4.0-PJ).
    - top_k: maximum number of grasps to return.

    Returns
    - List of dicts: {"pixel":(u,v), "angle":theta_radians, "width":meters, "quality":score}

    Implementation note: This function intentionally does not assume a
    single importable API because different installs expose the
    examples/policy script differently. If the `gqcnn` package is
    importable and exposes a programmatic policy API, this function
    *should* be updated to call it directly. For now it raises an
    informative ImportError with instructions when the package is
    missing.
    """
    try:
        import gqcnn  # type: ignore
    except Exception as e:
        raise ImportError(
            "GQ-CNN integration requires the 'gqcnn' package.\n"
            "Please install and download models as described in:\n"
            "https://berkeleyautomation.github.io/gqcnn/tutorials/tutorial.html#grasp-planning\n"
            "Quick install notes:\n"
            "  pip install gqcnn\n"
            "  # From the gqcnn repo root:\n"
            "  ./scripts/downloads/models/download_models.sh\n"
            "Then you can run the example policy script to test on saved images:\n"
            f"  python examples/policy.py {model_name} --depth_image {depth_npy_path} \\\n"
            f"    --segmask {segmask_path or 'None'} --camera_intr {camera_intr_path or 'None'}\n"
            "Alternatively run the ROS grasp planning service and query it.\n"
            "Once 'gqcnn' is installed, update this wrapper to call the\n"
            "programmatic API for better performance."
        )

    # If we reach here, try to locate the examples/policy.py script inside
    # the installed package location and run it as a subprocess to obtain
    # candidate grasps. This is a pragmatic fallback and may require the
    # package to include the examples folder (standard gqcnn installs do).
    try:
        import importlib.util
        spec = importlib.util.find_spec('gqcnn')
        if spec is None or not spec.origin:
            raise RuntimeError('Could not locate gqcnn package files')
        pkg_dir = os.path.dirname(spec.origin)
        # common examples path
        example_script = os.path.join(pkg_dir, '..', 'examples', 'policy.py')
        example_script = os.path.abspath(example_script)
        if not os.path.exists(example_script):
            # Try alternate location: package root + '/examples/policy.py'
            example_script = os.path.join(pkg_dir, 'examples', 'policy.py')
        if not os.path.exists(example_script):
            raise RuntimeError('examples/policy.py not found in gqcnn install')

        # Call the example script with the minimal args and capture stdout.
        cmd = [
            sys_executable(), example_script,
            model_name,
            '--depth_image', depth_npy_path,
        ]
        if segmask_path:
            cmd += ['--segmask', segmask_path]
        if camera_intr_path:
            cmd += ['--camera_intr', camera_intr_path]

        # Note: the example script prints results but does not provide
        # a machine-readable JSON by default. Users can modify the
        # example script to emit JSON; here we run it for manual testing.
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        out = proc.stdout.decode('utf-8', errors='ignore')
        err = proc.stderr.decode('utf-8', errors='ignore')

        # Heuristic: if the script printed candidate grasps, attempt to
        # parse lines that look like 'BEST GRASP: u v angle width score'
        candidates = []
        for line in out.splitlines():
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0].lower().startswith('grasp'):
                try:
                    u = float(parts[-5])
                    v = float(parts[-4])
                    ang = float(parts[-3])
                    wid = float(parts[-2])
                    score = float(parts[-1])
                    candidates.append({"pixel": (u, v), "angle": ang, "width": wid, "quality": score})
                except Exception:
                    continue

        # If no candidates found, return empty list but include stderr for debugging
        if not candidates:
            raise RuntimeError(f'GQ-CNN ran but no candidate grasps parsed.\nSTDOUT:\n{out}\nSTDERR:\n{err}')

        # Return top_k
        return sorted(candidates, key=lambda c: c.get('quality', 0), reverse=True)[:top_k]

    except Exception as e:
        raise RuntimeError(
            "GQ-CNN invocation failed. Ensure the package is installed and\n"
            "the models were downloaded. See the tutorial for setup steps.\n"
            f"Error detail: {e}"
        )


def sys_executable() -> str:
    """Return the Python executable path in a cross-platform way."""
    import sys
    return getattr(sys, 'executable', 'python')
