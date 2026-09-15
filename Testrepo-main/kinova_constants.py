"""
Shared constants for the Kinova Gen 3 simulation.

Imported by simulate_kinova.py, jarvis_primitives.py, and jarvis_prompts.py
so that configuration lives in one place.
"""
import os

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("JARVIS_API_KEY", "YOUR_API_KEY_HERE")
API_URL = "https://api.openai.com/v1/chat/completions"
API_MODEL = "gpt-5"

# ---------------------------------------------------------------------------
# Joint descriptions — what each joint does in the real world
# ---------------------------------------------------------------------------
JOINT_DESCRIPTIONS = {
    "joint_1": (
        "Base rotation (continuous). Rotates the entire arm around the "
        "vertical axis, like turning on a lazy-susan. "
        "Positive = counter-clockwise when viewed from above."
    ),
    "joint_2": (
        "Shoulder pitch (revolute, -2.41 to +2.41 rad). Tilts the upper arm "
        "forward/backward relative to the base. "
        "0 = arm pointing straight up; positive = leaning forward."
    ),
    "joint_3": (
        "Arm rotation (continuous). Rolls/twists the upper arm around its "
        "own long axis, like rotating your bicep."
    ),
    "joint_4": (
        "Elbow pitch (revolute, -2.66 to +2.66 rad). Bends the forearm "
        "up/down relative to the upper arm, like a human elbow."
    ),
    "joint_5": (
        "Forearm rotation (continuous). Rolls/twists the forearm around its "
        "own long axis, like rotating your wrist while keeping your elbow still."
    ),
    "joint_6": (
        "Wrist pitch (revolute, -2.23 to +2.23 rad). Tilts the end-effector "
        "up or down, like nodding your hand."
    ),
    "joint_7": (
        "Wrist rotation (continuous). Spins the end-effector (tool/gripper) "
        "around the approach axis — the final orientation adjustment."
    ),
    "gripper": (
        "Gripper open/close (0.0 = fully open, 1.2 = fully closed). "
        "Controls both fingers symmetrically. Use to grasp or release objects."
    ),
}

# ---------------------------------------------------------------------------
# Real joint limits — practical bounds for all joints (used in LLM prompts)
# ---------------------------------------------------------------------------
REAL_JOINT_LIMITS = {
    "joint_1": (-6.2832, 6.2832),
    "joint_2": (-2.41, 2.41),
    "joint_3": (-6.2832, 6.2832),
    "joint_4": (-2.66, 2.66),
    "joint_5": (-6.2832, 6.2832),
    "joint_6": (-2.23, 2.23),
    "joint_7": (-6.2832, 6.2832),
    "gripper": (0.0, 1.2),
}

# ---------------------------------------------------------------------------
# Example successful movement sequences (few-shot references for the LLM)
# ---------------------------------------------------------------------------
EXAMPLE_MOVEMENTS = """
Example 1 — "Look left":
  step 1: joint_1 -> 1.57  (rotate base 90° counter-clockwise)

Example 2 — "Reach forward and down":
  step 1: joint_2 -> 1.0   (tilt shoulder forward)
  step 2: joint_4 -> -1.2  (bend elbow downward)

Example 3 — "Point the end-effector straight down at the table":
  step 1: joint_2 -> 1.57  (shoulder forward 90°)
  step 2: joint_4 -> 0.0   (elbow straight)
  step 3: joint_6 -> 0.0   (wrist level)

Example 4 — "Rotate the gripper 180°":
  step 1: joint_7 -> 3.14  (spin wrist rotation half-turn)

Example 5 — "Close the gripper":
  step 1: gripper -> 1.2   (fully close fingers)

Example 6 — "Open the gripper":
  step 1: gripper -> 0.0   (fully open fingers)

Example 7 — "Pick up an object to the front-left":
  STRATEGY: First rotate the base (joint_1) so the whole arm faces the
  object, then extend/lower the arm to create an approach vector, and
  finally grasp. Estimate the object's angle from the camera images.
  step 1: joint_1 -> <angle> (rotate base toward the object — estimate from images)
  step 2: gripper -> 0.0    (open fingers before approach)
  step 3: joint_2 -> 1.3    (tilt shoulder forward toward table)
  step 4: joint_4 -> -1.0   (bend elbow to lower arm)
  step 5: joint_6 -> 0.6    (angle wrist down for top-down approach)
  step 6: joint_5 -> 0.0    (level the forearm twist for clean grip)
  step 7: gripper -> 0.55   (close fingers around object)
  step 8: joint_2 -> 0.5    (lift arm back up with object)

Example 8 — "Grab an object off to the right":
  STRATEGY: Align base joint with the object's direction first,
  then approach from above.
  step 1: joint_1 -> <angle> (rotate base to face the object — estimate from images)
  step 2: gripper -> 0.0    (open gripper)
  step 3: joint_2 -> 1.4    (lean shoulder far forward)
  step 4: joint_4 -> -1.2   (lower elbow toward table height)
  step 5: joint_6 -> 0.7    (tilt wrist to approach from above)
  step 6: gripper -> 0.6    (close around the object)
  step 7: joint_2 -> 0.3    (retract shoulder to lift object)
  step 8: joint_4 -> -0.3   (raise elbow back up)

Example 9 — "Reach toward an object straight ahead":
  STRATEGY: The object appears directly in front in the images, so
  minimal joint_1 rotation needed. Extend arm directly forward.
  step 1: joint_1 -> 0.0    (ensure base faces forward)
  step 2: joint_2 -> 1.2    (shoulder forward)
  step 3: joint_4 -> -0.9   (elbow down toward table)
  step 4: joint_6 -> 0.5    (wrist angled for approach)

IMPORTANT OBJECT INTERACTION STRATEGY:
  When approaching an object, ALWAYS do this in order:
  1. Open the gripper first (gripper -> 0.0)
  2. Rotate the BASE (joint_1) so the arm faces the object's direction
  3. Extend shoulder (joint_2) and elbow (joint_4) to create an approach path
  4. Adjust wrist pitch (joint_6) for the angle of approach (top-down = ~0.6-0.8)
  5. Fine-tune forearm orientation (joint_5) and wrist rotation (joint_7) if needed
  6. Close gripper to grasp
  7. Retract shoulder/elbow to lift
  Never try to reach an object without first aligning joint_1 with it.
"""

# ---------------------------------------------------------------------------
# Motion Primitives — predefined named macros the LLM can reference
# ---------------------------------------------------------------------------
MOTION_PRIMITIVES = {
    "home": [
        {"joint": "joint_1", "target": 0.0},
        {"joint": "joint_2", "target": 0.0},
        {"joint": "joint_3", "target": 0.0},
        {"joint": "joint_4", "target": 0.0},
        {"joint": "joint_5", "target": 0.0},
        {"joint": "joint_6", "target": 0.0},
        {"joint": "joint_7", "target": 0.0},
    ],
    "reach_forward": [
        {"joint": "joint_2", "target": 1.0},
        {"joint": "joint_4", "target": -0.5},
        {"joint": "joint_6", "target": 0.3},
    ],
    "reach_down": [
        {"joint": "joint_2", "target": 1.4},
        {"joint": "joint_4", "target": -1.0},
        {"joint": "joint_6", "target": 0.5},
    ],
    "reach_far_down": [
        {"joint": "joint_2", "target": 1.8},
        {"joint": "joint_4", "target": -1.5},
        {"joint": "joint_6", "target": 0.8},
    ],
    "look_left": [
        {"joint": "joint_1", "target": 1.57},
    ],
    "look_right": [
        {"joint": "joint_1", "target": -1.57},
    ],
    "look_behind": [
        {"joint": "joint_1", "target": 3.14},
    ],
    "open_gripper": [
        {"joint": "gripper", "target": 0.0},
    ],
    "close_gripper": [
        {"joint": "gripper", "target": 0.7},
    ],
    "wrist_down": [
        {"joint": "joint_6", "target": 1.0},
    ],
    "wrist_level": [
        {"joint": "joint_6", "target": 0.0},
    ],
}


def get_primitives_description() -> str:
    """Return a text block describing available motion primitives for prompts."""
    lines = []
    for name, steps in MOTION_PRIMITIVES.items():
        step_strs = ", ".join(f"{s['joint']}->{s['target']}" for s in steps)
        lines.append(f"  {name}: [{step_strs}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Object descriptions (sent to LLM for scene understanding)
# ---------------------------------------------------------------------------
OBJECT_DESCRIPTIONS = {
    "block":    "red block (5 cm cube)",
    "cylinder": "blue cylinder (r=2 cm, h=8 cm)",
    "bottle":   "green bottle (r=2.5 cm, h=8 cm)",
    "remote":   "dark-grey TV remote",
    "mug":      "yellow mug (r=3 cm, h=7 cm)",
    "box":      "black open-top box / tray (25 cm × 25 cm, 5 cm walls)",
}
