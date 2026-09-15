"""
Jarvis Prompts & API calls — prompt building and LLM interaction for all
Jarvis modes (jarvis1, jarvis2, jarvis3).

Extracted from simulate_kinova.py to keep that file focused on simulation.
"""

import json
import os
import time
import urllib.request
import urllib.error

import pybullet as p

from kinova_constants import (
    API_KEY, API_URL, API_MODEL,
    JOINT_DESCRIPTIONS, REAL_JOINT_LIMITS,
    EXAMPLE_MOVEMENTS, MOTION_PRIMITIVES,
    OBJECT_DESCRIPTIONS,
    get_primitives_description,
)
from jarvis_primitives import expand_plan
from clip_bbox import clip_bbox as compute_clip_bboxes, _GDINO_QUERIES


# ---------------------------------------------------------------------------
# Jarvis 1 — text-only LLM control (sends joint positions, no images)
# ---------------------------------------------------------------------------

def get_current_joint_positions(robot_id, movable_joints, gripper_info=None):
    """Return a dict of joint_name -> current angle (rad) for movable joints."""
    positions = {}
    for j in movable_joints:
        state = p.getJointState(robot_id, j["index"])
        positions[j["name"]] = round(state[0], 4)
    if gripper_info:
        left_state = p.getJointState(gripper_info["id"], gripper_info["left_idx"])
        positions["gripper"] = round(left_state[0], 4)
    return positions


def build_llm_prompt(user_request: str, joint_positions: dict, movable_joints: list) -> str:
    """Assemble the system + user prompt sent to the LLM."""

    # 1 — Current position
    pos_lines = "\n".join(f"  {name}: {angle:.4f} rad" for name, angle in joint_positions.items())

    # 2 — Joint descriptions (use REAL_JOINT_LIMITS for accurate bounds)
    desc_lines = []
    for j in movable_joints:
        desc = JOINT_DESCRIPTIONS.get(j["name"], "No description.")
        lo, hi = REAL_JOINT_LIMITS.get(j["name"], (j["lower"], j["upper"]))
        desc_lines.append(f"  {j['name']}  limits=[{lo}, {hi}]\n    {desc}")
    # Include gripper in the prompt
    g_desc = JOINT_DESCRIPTIONS.get("gripper", "Gripper control.")
    g_lo, g_hi = REAL_JOINT_LIMITS.get("gripper", (0.0, 1.2))
    desc_lines.append(f"  gripper  limits=[{g_lo}, {g_hi}]\n    {g_desc}")
    desc_block = "\n".join(desc_lines)

    # 3 — Strict output requirements
    format_rules = (
        "You MUST respond with ONLY valid JSON \u2014 no prose, no markdown.\n"
        "The JSON must be an object with a single key 'steps' containing an array.\n"
        "Each element: {\"joint\": \"<joint_name>\", \"target\": <value>}\n"
        "Example: {\"steps\": [{\"joint\": \"joint_2\", \"target\": 1.0}, {\"joint\": \"gripper\", \"target\": 0.6}]}\n"
        "Rules:\n"
        "  - One joint per step. Never move two joints in the same step.\n"
        "  - target must be a number, respecting the joint limits listed above.\n"
        "  - Steps execute sequentially; each completes before the next begins.\n"
        "  - Use realistic, safe values. Avoid limit extremes unless necessary.\n"
        "  - For arm joints (joint_1 through joint_7): target is in radians.\n"
        "  - For the gripper: use joint name \"gripper\". target is a float\n"
        "    from 0.0 (fully open) to 1.2 (fully closed). Both fingers move\n"
        "    symmetrically \u2014 you only specify one value.\n"
        "  - When a task involves grasping, always open the gripper before\n"
        "    positioning the arm, then close it after the arm is in place.\n"
        "  - When a task involves releasing, move the arm first, then open.\n"
    )

    prompt = (
        f"=== KINOVA GEN 3  7-DOF ARM \u2014 MOVEMENT PLANNER ===\n\n"
        f"CURRENT JOINT POSITIONS (radians):\n{pos_lines}\n\n"
        f"JOINT DESCRIPTIONS & LIMITS:\n{desc_block}\n\n"
        f"OUTPUT FORMAT REQUIREMENTS:\n{format_rules}\n\n"
        f"EXAMPLES OF SUCCESSFUL MOVEMENTS:\n{EXAMPLE_MOVEMENTS}\n\n"
        f"USER REQUEST:\n  \"{user_request}\"\n"
    )
    return prompt


def call_jarvis(user_request: str, robot_id, movable_joints: list, console, gripper_info=None) -> list | None:
    """Send the prompt to the LLM API and return parsed movement steps."""

    if API_KEY == "YOUR_API_KEY_HERE":
        console.write("Error: No API key configured.")
        console.write("  Set JARVIS_API_KEY env var or edit API_KEY in the script.")
        return None

    positions = get_current_joint_positions(robot_id, movable_joints, gripper_info)
    prompt = build_llm_prompt(user_request, positions, movable_joints)

    print("[JARVIS] Sending request to LLM...")

    body = json.dumps({
        "model": API_MODEL,
        "temperature": 1.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": (
                "You are a robotic-arm motion planner for a Kinova Gen 3 7-DOF arm "
                "equipped with a Robotiq 2F-85 two-finger gripper. "
                "You can command 7 arm joints (joint_1 through joint_7) AND the gripper. "
                "The gripper is controlled via the joint name 'gripper' with a target "
                "value from 0.0 (fully open) to 1.2 (fully closed). "
                "You MUST respond with ONLY valid JSON. "
                "The JSON must be an object with a single key 'steps' whose value is an array "
                "of step objects. Each step: {\"joint\": \"<name>\", \"target\": <number>}. "
                "Example: {\"steps\": [{\"joint\": \"joint_2\", \"target\": 1.0}]}. "
                "No extra keys, no prose, no markdown \u2014 only the JSON object."
            )},
            {"role": "user", "content": prompt},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        console.write(f"Error: HTTP {e.code}")
        print(f"[JARVIS] HTTP {e.code}: {err_body[:300]}")
        return None
    except Exception as e:
        console.write(f"Error: Request failed")
        print(f"[JARVIS] Request failed: {e}")
        return None

    # Parse the response
    try:
        content = data["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]
        steps = json.loads(content)
    except (KeyError, json.JSONDecodeError) as e:
        console.write(f"[JARVIS] Failed to parse LLM response: {e}")
        console.write(f"[JARVIS] Raw content: {data.get('choices', [{}])[0].get('message', {}).get('content', '<empty>')[:400]}")
        return None

    # Validate steps
    valid_names = {j["name"] for j in movable_joints}
    valid_names.add("gripper")
    validated = []
    for i, step in enumerate(steps):
        jname = step.get("joint")
        target = step.get("target")
        if jname not in valid_names:
            console.write(f"[JARVIS] WARNING: Step {i+1} references unknown joint '{jname}' \u2014 skipped.")
            continue
        if not isinstance(target, (int, float)):
            console.write(f"[JARVIS] WARNING: Step {i+1} has non-numeric target \u2014 skipped.")
            continue
        validated.append({"joint": jname, "target": float(target)})

    return validated


def _parse_llm_steps(data: dict, movable_joints: list, console, tag: str = "JARVIS") -> list | None:
    """Extract and validate movement steps from an LLM API response.

    Supports three step formats:
      1. {"joint": "<name>", "target": <value>}  -- direct joint command
      2. {"action": "<primitive_name>"}           -- expands to predefined steps
      3. {"action": "gripper", "target": <value>} -- gripper shorthand
    """
    try:
        content = data["choices"][0]["message"]["content"]
        if content is None or content.strip() == "":
            refusal = data["choices"][0]["message"].get("refusal", "")
            finish = data["choices"][0].get("finish_reason", "unknown")
            print(f"[{tag}] LLM returned empty content.")
            print(f"[{tag}] Finish reason: {finish}")
            print(f"[{tag}] Refusal: {refusal or 'none'}")
            safe_dump = json.dumps(data, indent=2, default=str)[:800]
            print(f"[{tag}] Full API response:\n{safe_dump}")
            return None
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "steps" in parsed:
            steps = parsed["steps"]
        elif isinstance(parsed, list):
            steps = parsed
        else:
            print(f"[{tag}] Unexpected JSON structure: {str(parsed)[:300]}")
            return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"[{tag}] Failed to parse LLM response: {e}")
        try:
            raw = data.get("choices", [{}])[0]
            raw_content = raw.get("message", {}).get("content", "<empty>")
            print(f"[{tag}] Raw content: {str(raw_content)[:500]}")
            safe_dump = json.dumps(data, indent=2, default=str)[:800]
            print(f"[{tag}] Full API response:\n{safe_dump}")
        except Exception:
            print(f"[{tag}] Could not extract response. Data: {str(data)[:500]}")
        return None

    valid_names = {j["name"] for j in movable_joints}
    valid_names.add("gripper")
    validated = []
    for i, step in enumerate(steps):
        action = step.get("action")
        if action:
            if action == "gripper":
                target = step.get("target")
                if isinstance(target, (int, float)):
                    validated.append({"joint": "gripper", "target": float(target)})
                else:
                    console.write(f"[{tag}] WARNING: Step {i+1} gripper action missing target \u2014 skipped.")
                continue
            if action in MOTION_PRIMITIVES:
                console.write(f"[{tag}] Expanding primitive '{action}' ({len(MOTION_PRIMITIVES[action])} sub-steps)")
                validated.extend(MOTION_PRIMITIVES[action])
                continue
            console.write(f"[{tag}] WARNING: Step {i+1} unknown action '{action}' \u2014 skipped.")
            continue
        jname = step.get("joint")
        target = step.get("target")
        if jname not in valid_names:
            console.write(f"[{tag}] WARNING: Step {i+1} references unknown joint '{jname}' \u2014 skipped.")
            continue
        if not isinstance(target, (int, float)):
            console.write(f"[{tag}] WARNING: Step {i+1} has non-numeric target \u2014 skipped.")
            continue
        validated.append({"joint": jname, "target": float(target)})
    return validated


# ---------------------------------------------------------------------------
# Jarvis 2 — Vision-based LLM control (sends camera images, no positions)
# ---------------------------------------------------------------------------

def build_vision_prompt(user_request: str, movable_joints: list,
                        joint_positions: dict | None = None) -> str:
    """Build the text portion of the vision prompt."""

    desc_lines = []
    for j in movable_joints:
        desc = JOINT_DESCRIPTIONS.get(j["name"], "No description.")
        lo, hi = REAL_JOINT_LIMITS.get(j["name"], (j["lower"], j["upper"]))
        desc_lines.append(f"  {j['name']}  limits=[{lo}, {hi}]\n    {desc}")
    g_desc = JOINT_DESCRIPTIONS.get("gripper", "Gripper control.")
    g_lo, g_hi = REAL_JOINT_LIMITS.get("gripper", (0.0, 1.2))
    desc_lines.append(f"  gripper  limits=[{g_lo}, {g_hi}]\n    {g_desc}")
    desc_block = "\n".join(desc_lines)

    prims_block = get_primitives_description()

    format_rules = (
        "You MUST respond with ONLY valid JSON \u2014 no prose, no markdown.\n"
        "The JSON must be an object with a single key 'steps' containing an array.\n"
        "Each element is ONE of:\n"
        '  1. Direct joint: {"joint": "<joint_name>", "target": <value>}\n'
        '  2. Motion primitive: {"action": "<primitive_name>"}\n'
        '  3. Gripper shorthand: {"action": "gripper", "target": <value>}\n'
        'Example: {"steps": [{"action": "open_gripper"}, {"joint": "joint_1", "target": 0.87}, '
        '{"action": "reach_forward"}, {"action": "gripper", "target": 0.6}]}\n'
        "Rules:\n"
        "  - Steps execute sequentially; each completes before the next begins.\n"
        "  - Use primitives when they match the intent (faster, proven safe).\n"
        "  - Use direct joint commands for fine adjustments.\n"
        "  - Mix primitives and direct commands freely in the same plan.\n"
        "  - target must be a number, respecting the joint limits listed above.\n"
        "  - For arm joints (joint_1 through joint_7): target is in radians.\n"
        '  - For the gripper: use joint name "gripper" or action "gripper".\n'
        "    target is a float from 0.0 (fully open) to 1.2 (fully closed).\n"
        "\n"
        "CRITICAL STRATEGY for object interaction:\n"
        "  1. ALWAYS set joint_1 first to rotate the base so the arm faces the\n"
        "     object's direction. USE THE CAMERA IMAGES to determine where\n"
        "     objects are \u2014 estimate direction and distance from what you see.\n"
        "  2. Open the gripper BEFORE approaching (gripper -> 0.0).\n"
        "  3. Extend shoulder (joint_2) and elbow (joint_4) to reach toward object.\n"
        "  4. Adjust wrist pitch (joint_6) for approach angle (top-down \u2248 0.6-0.8).\n"
        "  5. Close gripper to grasp, then retract to lift.\n"
        "  Never try to reach an object without first aligning joint_1 with it.\n"
        "  Use the images to judge distances, heights, and object positions.\n"
    )

    pos_block = ""
    if joint_positions:
        pos_lines = "\n".join(f"  {name}: {angle:.4f} rad" for name, angle in joint_positions.items())
        pos_block = (
            f"CURRENT JOINT POSITIONS (radians):\n{pos_lines}\n\n"
            "Use these alongside the images to plan accurate movements.\n\n"
        )

    prompt = (
        "=== KINOVA GEN 3  7-DOF ARM \u2014 VISION-BASED MOVEMENT PLANNER ===\n\n"
        "You are provided with THREE camera images of the current scene:\n"
        "  IMAGE 1 \u2014 Birds-eye view: A top-down overhead camera looking straight\n"
        "            down at the robot and workspace. Use this to judge the\n"
        "            horizontal position of the arm, gripper, and any objects.\n"
        "  IMAGE 2 \u2014 End-effector camera: A camera mounted on the robot's wrist,\n"
        "            showing what the gripper sees. Use this to judge proximity\n"
        "            to objects, alignment, and gripper orientation.\n"
        "  IMAGE 3 \u2014 Isometric view: A wide-angle 3/4 perspective from above and\n"
        "            to the side (zoomed out). Gives overall spatial context \u2014\n"
        "            the full robot, pedestal, and surrounding workspace are\n"
        "            visible. Use this to judge heights, depth, and the overall\n"
        "            scene layout.\n\n"
        f"{pos_block}"
        "Plan your movements based on what you SEE in the images"
        f"{' combined with the joint position data' if joint_positions else ''}"
        " and the user's natural-language request.\n\n"
        f"JOINT DESCRIPTIONS & LIMITS:\n{desc_block}\n\n"
        f"AVAILABLE MOTION PRIMITIVES:\n{prims_block}\n\n"
        f"OUTPUT FORMAT REQUIREMENTS:\n{format_rules}\n\n"
        f"EXAMPLES OF SUCCESSFUL MOVEMENTS:\n{EXAMPLE_MOVEMENTS}\n\n"
        f"USER REQUEST:\n  \"{user_request}\"\n"
    )
    return prompt


def call_jarvis2(user_request: str, robot_id, ee_link_index: int,
                 movable_joints: list, console, gripper_info=None,
                 # Camera helpers injected from simulate_kinova at call time
                 capture_ee_image=None, capture_birdseye_image=None,
                 capture_isometric_image=None, images_to_base64=None,
                 _write_png=None, SNAP_DIR=None) -> list | None:
    """Vision-based Jarvis: sends EE + birds-eye screenshots instead of positions."""

    if API_KEY == "YOUR_API_KEY_HERE":
        console.write("[JARVIS2] ERROR: No API key configured.")
        return None

    # 1 — Capture fresh screenshots
    console.write("[JARVIS2] Capturing camera views...")
    ee_img = capture_ee_image(robot_id, ee_link_index)
    bird_img = capture_birdseye_image()
    iso_img = capture_isometric_image()

    # Also save them for reference
    os.makedirs(SNAP_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ee_path = os.path.join(SNAP_DIR, f"jarvis2_ee_{timestamp}.png")
    bird_path = os.path.join(SNAP_DIR, f"jarvis2_bird_{timestamp}.png")
    iso_path = os.path.join(SNAP_DIR, f"jarvis2_iso_{timestamp}.png")
    _write_png(ee_path, ee_img)
    _write_png(bird_path, bird_img)
    _write_png(iso_path, iso_img)
    console.write(f"[JARVIS2] Saved: {os.path.basename(ee_path)}, {os.path.basename(bird_path)}, {os.path.basename(iso_path)}")

    # 2 — Encode as base64 for the vision API
    ee_b64 = images_to_base64(ee_img)
    bird_b64 = images_to_base64(bird_img)
    iso_b64 = images_to_base64(iso_img)

    # 3 — Build prompt (include joint positions for hybrid mode)
    positions = get_current_joint_positions(robot_id, movable_joints, gripper_info)
    text_prompt = build_vision_prompt(user_request, movable_joints, joint_positions=positions)

    console.write("[JARVIS2] Sending images + request to LLM...")

    # 4 — Assemble multi-modal message (OpenAI vision format)
    body = json.dumps({
        "model": API_MODEL,
        "temperature": 1.0,
        "max_completion_tokens": 16384,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": (
                "You are a vision-based robotic-arm motion planner for a Kinova Gen 3 "
                "7-DOF arm equipped with a Robotiq 2F-85 two-finger gripper. "
                "You receive three camera images of the current scene and must plan "
                "movements based on what you SEE \u2014 the robot's pose, object positions, "
                "and spatial relationships visible in the images. "
                "You can command 7 arm joints (joint_1 through joint_7) AND the gripper. "
                "The gripper is controlled via the joint name 'gripper' with a target "
                "value from 0.0 (fully open) to 1.2 (fully closed). "
                "You MUST respond with ONLY valid JSON. "
                "The JSON must be an object with a single key 'steps' whose value is an array "
                "of step objects. Each step: {\"joint\": \"<name>\", \"target\": <number>}. "
                "Example: {\"steps\": [{\"joint\": \"joint_2\", \"target\": 1.0}]}. "
                "No extra keys, no prose, no markdown \u2014 only the JSON object."
            )},
            {"role": "user", "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{bird_b64}",
                    "detail": "high"
                }},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{ee_b64}",
                    "detail": "high"
                }},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{iso_b64}",
                    "detail": "high"
                }},
            ]},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        console.write(f"[JARVIS2] HTTP {e.code}: {err_body[:500]}")
        return None
    except Exception as e:
        console.write(f"[JARVIS2] Request failed: {e}")
        return None

    try:
        finish = data["choices"][0].get("finish_reason", "unknown")
        console.write(f"[JARVIS2] Finish reason: {finish}")
    except (KeyError, IndexError):
        pass

    return _parse_llm_steps(data, movable_joints, console, tag="JARVIS2")


# ---------------------------------------------------------------------------
# Jarvis 3 — Dispatch prompt (code-as-policy)
# ---------------------------------------------------------------------------

def build_dispatch_prompt(user_request: str, ee_pos: list,
                          scene_objects: dict | None = None,
                          bbox_legend: str | None = None) -> str:
    """ONE-SHOT dispatch prompt.  LLM returns {"plan": [...]} -- an ordered
    array of primitives.  All motion is then executed by deterministic code.

    If *bbox_legend* is provided, the prompt notes that annotated camera
    images are attached and includes the colour legend so the LLM knows
    which bounding-box colour corresponds to which object."""

    objects_block = ""
    if scene_objects:
        objects_block = "OBJECTS IN SCENE (measured positions, world-frame metres):\n"
        for name, pos in scene_objects.items():
            desc = OBJECT_DESCRIPTIONS.get(name, name)
            objects_block += (f"  \u2022 \"{name}\" \u2014 {desc}\n"
                             f"      position: x={pos['x']:.4f}  y={pos['y']:.4f}  z={pos['z']:.4f}\n")
        objects_block += "\n"

    primitives_doc = (
        "AVAILABLE PRIMITIVES (use cmd names EXACTLY):\n\n"
        "GRASPING & RELEASING:\n"
        '  1.  pick          \u2014 Side-grasp an object.\n'
        '      {"cmd":"pick","object":"<name>"}\n'
        '  2.  place         \u2014 Place held object at XYZ.\n'
        '      {"cmd":"place","x":f,"y":f,"z":f}\n'
        '  3.  open_gripper  \u2014 Open gripper fully.\n'
        '      {"cmd":"open_gripper"}\n'
        '  4.  close_gripper \u2014 Close gripper fully.\n'
        '      {"cmd":"close_gripper"}\n'
        '  5.  drop          \u2014 Release from current height.\n'
        '      {"cmd":"drop"}\n'
        '  6.  top_grasp     \u2014 Top-down grasp (approach from above).\n'
        '      {"cmd":"top_grasp","object":"<name>"}\n'
        '  7.  regrasp       \u2014 Open gripper slightly, adjust, re-close.\n'
        '      {"cmd":"regrasp"}\n'
        '  8.  pick_and_place\u2014 Combined pick then place.\n'
        '      {"cmd":"pick_and_place","object":"<name>","x":f,"y":f,"z":f}\n\n'
        "MOVEMENT & POSITIONING:\n"
        '  9.  move_to       \u2014 Move EE to world XYZ.\n'
        '      {"cmd":"move_to","x":f,"y":f,"z":f}\n'
        '  10. home          \u2014 Return arm to upright rest pose.\n'
        '      {"cmd":"home"}\n'
        '  11. lift_high     \u2014 Lift whatever is held high above table.\n'
        '      {"cmd":"lift_high"}\n'
        '  12. move_above    \u2014 Move EE directly above an object at safe height.\n'
        '      {"cmd":"move_above","object":"<name>"}\n'
        '  13. approach_from_top \u2014 Descend straight down to object from above.\n'
        '      {"cmd":"approach_from_top","object":"<name>"}\n'
        '  14. approach_from_side\u2014 Move to object from the side at its height.\n'
        '      {"cmd":"approach_from_side","object":"<name>"}\n'
        '  15. retract       \u2014 Pull EE straight back from current position.\n'
        '      {"cmd":"retract","distance":f}\n'
        '  16. move_relative \u2014 Move EE by a relative offset (dx,dy,dz).\n'
        '      {"cmd":"move_relative","dx":f,"dy":f,"dz":f}\n'
        '  17. align_with    \u2014 Position EE above object at specified height.\n'
        '      {"cmd":"align_with","object":"<name>","height":f}\n'
        '  18. hover         \u2014 Hover at specified height above object.\n'
        '      {"cmd":"hover","object":"<name>","clearance":f}\n'
        '  19. park          \u2014 Tuck arm into a compact safe position.\n'
        '      {"cmd":"park"}\n\n'
        "PUSHING & CONTACT:\n"
        '  20. push          \u2014 Push object along (dx,dy) on table.\n'
        '      {"cmd":"push","object":"<name>","dx":f,"dy":f}\n'
        '  21. nudge         \u2014 Tiny push to fine-adjust position.\n'
        '      {"cmd":"nudge","object":"<name>","dx":f,"dy":f}\n'
        '  22. drag          \u2014 Grip & drag object across surface.\n'
        '      {"cmd":"drag","object":"<name>","dx":f,"dy":f}\n'
        '  23. tap           \u2014 Tap the top of an object once.\n'
        '      {"cmd":"tap","object":"<name>"}\n'
        '  24. knock_over    \u2014 Knock a standing object sideways.\n'
        '      {"cmd":"knock_over","object":"<name>"}\n'
        '  25. touch         \u2014 Gently touch top of object and hold.\n'
        '      {"cmd":"touch","object":"<name>"}\n'
        '  26. slide         \u2014 Grip object, slide it along table.\n'
        '      {"cmd":"slide","object":"<name>","dx":f,"dy":f}\n\n'
        "WRIST / IN-HAND:\n"
        '  27. rotate_object \u2014 Rotate held object about wrist axis.\n'
        '      {"cmd":"rotate_object","angle_deg":f}\n'
        '  28. flip_object   \u2014 Flip held object 180\u00b0.\n'
        '      {"cmd":"flip_object"}\n'
        '  29. tilt_object   \u2014 Tilt held object by angle_deg.\n'
        '      {"cmd":"tilt_object","angle_deg":f}\n'
        '  30. shake         \u2014 Shake held object back and forth.\n'
        '      {"cmd":"shake"}\n'
        '  31. pour          \u2014 Tilt held object ~90\u00b0 to pour.\n'
        '      {"cmd":"pour"}\n\n'
        "COMPOUND TASKS:\n"
        '  32. stack         \u2014 Pick first object, stack on second.\n'
        '      {"cmd":"stack","object":"<name>","target":"<name>"}\n'
        '  33. sort          \u2014 Move object to angle on table ring.\n'
        '      {"cmd":"sort","object":"<name>","angle_deg":f}\n'
        '  34. line_up       \u2014 Arrange listed objects in a line.\n'
        '      {"cmd":"line_up","objects":["<n1>","<n2>",...],"y":f}\n'
        '  35. handover      \u2014 Pick up & present towards human.\n'
        '      {"cmd":"handover","object":"<name>"}\n'
        '  36. deliver       \u2014 Pick object & place on target (e.g. person\'s tray).\n'
        '      {"cmd":"deliver","object":"<name>","target":"<name>"}\n'
        '  36b. present     \u2014 Pick object & hold near person (face height, above tray).\n'
        '                     Use for interactions: brush teeth, show item, etc.\n'
        '      {"cmd":"present","object":"<name>","target":"<name>"}\n'
        '  37. inspect       \u2014 Pick up & bring to EE camera.\n'
        '      {"cmd":"inspect","object":"<name>"}\n'
        '  38. pick_and_place\u2014 Pick object then place at XYZ.\n'
        '      {"cmd":"pick_and_place","object":"<name>","x":f,"y":f,"z":f}\n'
        '  39. swap          \u2014 Swap positions of two objects.\n'
        '      {"cmd":"swap","object_a":"<name>","object_b":"<name>"}\n'
        '  40. group         \u2014 Gather listed objects to a single XY.\n'
        '      {"cmd":"group","objects":["<n1>","<n2>",...],"x":f,"y":f}\n'
        '  41. scatter       \u2014 Spread objects apart from center.\n'
        '      {"cmd":"scatter","objects":["<n1>","<n2>",...]}\n'
        '  42. clear_area    \u2014 Push objects away from an XY region.\n'
        '      {"cmd":"clear_area","x":f,"y":f,"radius":f}\n\n'
        "SWEEPING:\n"
        '  43. sweep_table   \u2014 Arc sweep from start_deg to end_deg.\n'
        '      {"cmd":"sweep_table","start_deg":f,"end_deg":f}\n'
        '  44. circle_sweep  \u2014 Full 360\u00b0 sweep to clear table.\n'
        '      {"cmd":"circle_sweep"}\n\n'
        "GESTURES:\n"
        '  45. wave          \u2014 Friendly wave gesture.\n'
        '      {"cmd":"wave"}\n'
        '  46. point_at      \u2014 Point at an object.\n'
        '      {"cmd":"point_at","object":"<name>"}\n'
        '  47. guard         \u2014 Hover arm protectively over object.\n'
        '      {"cmd":"guard","object":"<name>"}\n\n'
        "NAVIGATION & PATHS:\n"
        '  48. orbit         \u2014 Move EE in a circle around an object.\n'
        '      {"cmd":"orbit","object":"<name>","radius":f,"height":f}\n'
        '  49. zigzag        \u2014 Zigzag EE between two XY points.\n'
        '      {"cmd":"zigzag","x1":f,"y1":f,"x2":f,"y2":f,"z":f}\n'
        '  50. patrol        \u2014 Visit a sequence of XYZ waypoints.\n'
        '      {"cmd":"patrol","waypoints":[{"x":f,"y":f,"z":f},...]}\n\n'
        "BASIC JOINT CONTROL:\n"
        '  51. rotate        \u2014 Rotate base (joint 1) by angle_deg degrees.\n'
        '      {"cmd":"rotate","angle_deg":f}\n'
        '  52. straighten    \u2014 Reset joints 1-5 to 0 (arm upright, keeps wrist).\n'
        '      {"cmd":"straighten"}\n'
    )

    rules = (
        "\nRULES:\n"
        '  - Return {"plan": [...]}.  plan is an array of primitives.\n'
        "  - Each primitive is a JSON object with \"cmd\" plus its args.\n"
        "  - Object names must EXACTLY match OBJECTS IN SCENE.\n"
        "  - Use the provided xyz coords \u2014 do NOT invent coordinates.\n"
        "  - 'The other side' = negate the object's x and y.\n"
        "  - Keep plans short (typically 2\u20136 steps).\n"
        "  - For pick-and-place: pick \u2192 place.\n"
        "  - 'bring me'/'give me' = deliver to person: use deliver with target='person'.\n"
        "  - End with home if the arm should park afterwards.\n"
    )

    examples = (
        "\nEXAMPLES:\n"
        '  "pick up the block and put it on the other side"\n'
        '  \u2192 {"plan":[{"cmd":"pick","object":"block"},{"cmd":"place","x":-0.60,"y":-0.73,"z":1.25},{"cmd":"home"}]}\n\n'
        '  "sweep everything off the table"\n'
        '  \u2192 {"plan":[{"cmd":"circle_sweep"},{"cmd":"home"}]}\n\n'
        '  "stack the cylinder on the block"\n'
        '  \u2192 {"plan":[{"cmd":"stack","object":"cylinder","target":"block"},{"cmd":"home"}]}\n\n'
        '  "wave at me then point at the bottle"\n'
        '  \u2192 {"plan":[{"cmd":"wave"},{"cmd":"point_at","object":"bottle"}]}\n'
    )

    # Vision context: if annotated images are attached, tell the LLM
    vision_block = ""
    if bbox_legend:
        vision_block = (
            "CAMERA IMAGES (attached):\n"
            "  Three annotated camera images are included with this request.\n"
            "  In every image, each detected object is surrounded by a\n"
            "  uniquely coloured bounding box so you can visually identify it.\n\n"
            "    IMAGE 1 \u2014 Birds-eye (top-down) view with bounding boxes.\n"
            "    IMAGE 2 \u2014 End-effector camera with bounding boxes.\n"
            "    IMAGE 3 \u2014 Isometric (3/4) view with bounding boxes.\n\n"
            "  Use the bounding boxes to confirm which object corresponds to\n"
            "  the user\u2019s description.  The colour legend below maps each\n"
            "  bounding-box colour to its object name:\n\n"
            f"{bbox_legend}\n\n"
        )

    prompt = (
        "=== KINOVA GEN 3 \u2014 TASK DISPATCHER ===\n\n"
        "You are a high-level task planner.  Translate the user\u2019s request\n"
        "into a sequence of motion primitives.  All low-level motion is code.\n\n"
        f"{vision_block}"
        "ROBOT INFO:\n"
        f"  Base: (0, 0, 1.18)  |  Reach: ~0.7 m  |  Table z \u2248 1.24 m\n"
        f"  Current EE: x={ee_pos[0]:.3f}  y={ee_pos[1]:.3f}  z={ee_pos[2]:.3f}\n\n"
        f"{objects_block}"
        f"{primitives_doc}"
        f"{rules}"
        f"{examples}\n"
        f"USER TASK:  \"{user_request}\"\n\n"
        "Return JSON:"
    )
    return prompt


# ---------------------------------------------------------------------------
# Jarvis 3 step display helpers
# ---------------------------------------------------------------------------

def step_summary(step: dict) -> str:
    """One-line summary of a jarvis3 step."""
    label = step.get("label", "")
    if label:
        return label
    stype = step.get("type", "joint")
    if stype == "pose":
        return "Move"
    elif stype == "gripper":
        return "Open gripper" if step.get("target", 0) > 0.5 else "Close gripper"
    elif stype == "gripper_force":
        return "Grip"
    elif stype == "servo_contact":
        tgt = step.get("grasp_target", "object")
        return f"Align {tgt}"
    elif stype == "creep_contact":
        return "Creep to contact"
    elif stype == "dwell":
        return "Wait"
    else:
        return stype


def format_step(num: int, step: dict) -> str:
    """Clean one-line step for plan display."""
    return f"  Step {num}: {step_summary(step)}"


# ---------------------------------------------------------------------------
# PRE-FILTER: fast LLM call to identify relevant objects before DINO
# ---------------------------------------------------------------------------

def _prefilter_objects(user_request: str, loaded_objects: dict,
                       console=None) -> list:
    """Use gpt-4o-mini to identify which object(s) the user wants to manipulate.

    Returns a list of canonical object names (subset of loaded_objects keys).
    On any failure, returns all loaded object names (graceful fallback).
    """
    all_names = sorted(loaded_objects.keys())
    # Not worth filtering if 2 or fewer objects
    if len(all_names) <= 2:
        return all_names

    # Build object library for the prompt
    library_lines = []
    for name in all_names:
        desc = _GDINO_QUERIES.get(name, name)
        library_lines.append(f"  - {name}: {desc}")
    library_text = "\n".join(library_lines)

    prompt = (
        f"Objects in the scene:\n{library_text}\n\n"
        f"User request: \"{user_request}\"\n\n"
        "Which object(s) does the user want the robot to manipulate? "
        "Return ONLY a JSON object with an \"objects\" key containing an "
        "array of object names from the list above. Do not include "
        "destinations, people, or locations — only the object(s) to be "
        "picked up or acted upon."
    )

    body = json.dumps({
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_completion_tokens": 256,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system",
             "content": "You identify which objects a user wants a robot to "
                        "manipulate. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"].strip()
        parsed = json.loads(content)
        names = parsed.get("objects", [])
        # Validate: only keep names that exist in loaded_objects
        valid = [n for n in names if n in loaded_objects]
        if valid:
            if console:
                console.write(f"[JARVIS] Pre-filter: {', '.join(valid)} "
                              f"({len(valid)} of {len(all_names)} objects)")
            return valid
        # LLM returned empty or invalid — fall back
        if console:
            console.write("[JARVIS] Pre-filter: no match, using all objects")
        return all_names
    except Exception as e:
        if console:
            console.write(f"[JARVIS] Pre-filter failed ({e}), using all objects")
        return all_names


# ---------------------------------------------------------------------------
# ONE-SHOT DISPATCH: user request -> LLM -> JSON plan -> expand -> flat steps
# ---------------------------------------------------------------------------

def call_jarvis3_dispatch(user_request: str, robot_id, ee_link_index: int,
                          movable_joints: list, console, gripper_info=None,
                          loaded_objects: dict | None = None,
                          compute_object_positions=None,
                          # Camera helpers (injected from simulate_kinova)
                          capture_ee_image=None,
                          capture_birdseye_image=None,
                          capture_isometric_image=None,
                          images_to_base64=None,
                          _write_png=None,
                          SNAP_DIR=None) -> list | None:
    """Single LLM call: interpret user request -> plan of primitives -> expand.

    When camera helpers are provided the function captures three camera
    views, runs them through *clip_bbox* to produce annotated images with
    coloured bounding boxes, and sends those images alongside the text
    prompt so the LLM can visually identify objects.

    Returns flat list of sub-steps for the execution engine, or None on error.
    """
    if API_KEY == "YOUR_API_KEY_HERE":
        console.write("[JARVIS3] ERROR: No API key configured.")
        return None

    # 0 — Pre-filter: identify which objects the user wants to manipulate
    #     so DINO only runs on the relevant subset (much faster).
    filtered_objects = loaded_objects
    if loaded_objects:
        relevant_names = _prefilter_objects(user_request, loaded_objects, console)
        # Always keep "person" (hardcoded position, zero DINO cost)
        if "person" in loaded_objects and "person" not in relevant_names:
            relevant_names.append("person")
        filtered_objects = {n: loaded_objects[n] for n in relevant_names
                           if n in loaded_objects}

    # 1 — Compute object positions (DINO runs only on filtered set)
    scene_objects = {}
    if filtered_objects and compute_object_positions:
        scene_objects = compute_object_positions(filtered_objects, console)
        for oname, opos in scene_objects.items():
            print(f"  [JARVIS3] {oname}: ({opos['x']:.4f}, {opos['y']:.4f}, {opos['z']:.4f})")

    # 2 — EE position
    ee_state = p.getLinkState(robot_id, ee_link_index, computeForwardKinematics=True)
    ee_pos = list(ee_state[4])

    # 3 — Capture camera images and annotate with clip_bbox
    has_vision = (capture_ee_image is not None
                  and capture_birdseye_image is not None
                  and capture_isometric_image is not None
                  and images_to_base64 is not None)
    bbox_legend = None
    image_payloads = []

    if has_vision and filtered_objects:
        ee_img   = capture_ee_image(robot_id, ee_link_index)
        bird_img = capture_birdseye_image()
        iso_img  = capture_isometric_image()

        # Run clip_bbox — annotates images with coloured bounding boxes
        ann_bird, ann_ee, ann_iso, bbox_legend = compute_clip_bboxes(
            filtered_objects, robot_id, ee_link_index,
            bird_img, ee_img, iso_img,
            console=console,
        )

        # Save annotated snapshots for debugging
        if _write_png and SNAP_DIR:
            os.makedirs(SNAP_DIR, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            for tag, img in [("bird_bbox", ann_bird), ("ee_bbox", ann_ee), ("iso_bbox", ann_iso)]:
                path = os.path.join(SNAP_DIR, f"jarvis3_{tag}_{timestamp}.png")
                _write_png(path, img)
            console.write(f"  Snapshots saved to {SNAP_DIR}")

        # Encode annotated images as base64 for the vision API
        for label, img in [("birds-eye", ann_bird), ("EE", ann_ee), ("isometric", ann_iso)]:
            b64 = images_to_base64(img)
            image_payloads.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
            })

    # 4 — Build prompt
    text_prompt = build_dispatch_prompt(user_request, ee_pos,
                                        scene_objects=scene_objects,
                                        bbox_legend=bbox_legend)

    # Suppress verbose LLM request message — user sees "Calculating..."

    # 5 — Build message content (multimodal if images available)
    if image_payloads:
        user_content = [{"type": "text", "text": text_prompt}] + image_payloads
        system_msg = (
            "You are a task planner for a Kinova Gen 3 robotic arm. "
            "You receive three annotated camera images of the workspace "
            "(birds-eye, end-effector, and isometric views). "
            "In each image, every detected object is highlighted with a "
            "uniquely coloured bounding box drawn around it. A colour "
            "legend is provided in the text prompt mapping each colour to "
            "an object name. Use these bounding boxes to visually confirm "
            "which object the user is referring to and to verify the "
            "spatial layout of the scene before choosing primitives. "
            "If the user refers to an object by description (e.g. 'the "
            "red one'), match it to the correct bounding-box label. "
            "Return a JSON object with a \"plan\" key containing an array "
            "of motion primitives.  Use ONLY the primitives documented in "
            "the prompt.  Use ONLY coordinates from the provided object "
            "positions \u2014 never invent coordinates."
        )
    else:
        user_content = text_prompt
        system_msg = (
            "You are a task planner for a Kinova Gen 3 robotic arm. "
            "Return a JSON object with a \"plan\" key containing an array "
            "of motion primitives.  Use ONLY the primitives documented in "
            "the prompt.  Use ONLY coordinates from the provided object "
            "positions \u2014 never invent coordinates."
        )

    # 6 — Single API call
    body = json.dumps({
        "model": API_MODEL,
        "temperature": 1,
        "max_completion_tokens": 4096,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        console.write(f"Jarvis: Error — HTTP {e.code}")
        return None
    except Exception as e:
        console.write(f"Jarvis: Request failed.")
        return None

    # 5 — Parse JSON response
    finish_reason = data.get("choices", [{}])[0].get("finish_reason", "?")
    print(f"[JARVIS3] finish_reason={finish_reason}")
    try:
        content = data["choices"][0]["message"]["content"]
        if not content or not content.strip():
            console.write(f"Jarvis: LLM returned empty response (finish_reason={finish_reason}).")
            print(f"[JARVIS3] Raw API data: {json.dumps(data, indent=2)[:1000]}")
            return None
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]
        parsed = json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        console.write(f"Jarvis: Failed to parse response.")
        try:
            raw = data["choices"][0]["message"]["content"]
            print(f"[JARVIS3] Raw: {str(raw)[:500]}")
        except Exception:
            pass
        return None

    # Accept {"plan": [...]} or bare [...]
    if isinstance(parsed, dict) and "plan" in parsed:
        plan_list = parsed["plan"]
    elif isinstance(parsed, list):
        plan_list = parsed
    else:
        console.write(f"Jarvis: Unexpected response format.")
        return None

    if not plan_list:
        console.write("Jarvis: Empty plan returned.")
        return None

    # Log primitives to stdout only (not user console)
    for i, prim in enumerate(plan_list):
        print(f"  [JARVIS3] [{i+1}] {prim.get('cmd','?')}  {json.dumps(prim)}")

    # 6 — Expand primitives into sub-steps
    # Pass loaded_objects into expand_plan so primitives can query body ids / AABBs
    flat_steps = expand_plan(plan_list, robot_id, ee_link_index, movable_joints,
                             scene_objects, console, loaded_objects=loaded_objects)
    if not flat_steps:
        console.write("Jarvis: Planning failed.")
        return None

    return flat_steps


# ---------------------------------------------------------------------------
# STUCK-STEP REQUERY — vision-assisted re-planning when a step stalls (7 s)
# ---------------------------------------------------------------------------

def requery_jarvis3_dispatch(
    original_request: str,
    robot_id, ee_link_index: int,
    movable_joints: list, console,
    gripper_info=None,
    loaded_objects: dict | None = None,
    compute_object_positions=None,
    # Camera helpers injected from simulate_kinova at call time
    capture_ee_image=None,
    capture_birdseye_image=None,
    capture_isometric_image=None,
    images_to_base64=None,
    # Context about what happened so far
    completed_steps: list | None = None,
    stuck_step: dict | None = None,
    remaining_steps: list | None = None,
) -> list | None:
    """Re-query the LLM with fresh camera images when a jarvis3 step is stuck.

    Sends:
      - The original user request
      - Fresh EE + birds-eye + isometric images
      - A summary of completed / stuck / remaining steps
    Returns a NEW flat list of sub-steps to replace the remaining plan,
    or None on failure.
    """
    if API_KEY == "YOUR_API_KEY_HERE":
        console.write("[REQUERY] ERROR: No API key configured.")
        return None

    # 1 — Capture fresh camera views
    console.write("[REQUERY] Capturing fresh camera views...")
    ee_img = capture_ee_image(robot_id, ee_link_index)
    bird_img = capture_birdseye_image()
    iso_img = capture_isometric_image()

    image_payloads = []
    for label, img in [("EE", ee_img), ("birds-eye", bird_img), ("isometric", iso_img)]:
        if img is not None:
            b64 = images_to_base64(img)
            image_payloads.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
            })

    # 2 — Object positions
    scene_objects = {}
    if loaded_objects and compute_object_positions:
        scene_objects = compute_object_positions(loaded_objects, console)

    # 3 — EE position
    ee_state = p.getLinkState(robot_id, ee_link_index, computeForwardKinematics=True)
    ee_pos = list(ee_state[4])

    # 4 — Build dispatch prompt (same as original, so LLM has full context)
    base_prompt = build_dispatch_prompt(original_request, ee_pos,
                                        scene_objects=scene_objects)

    # 5 — Build stuck-context addendum
    context_lines = [
        "\n\n=== REQUERY — STEP IS STUCK ===",
        "The robot attempted to execute the plan but a step has been stuck",
        "for 7 seconds with no meaningful progress.  Fresh camera images",
        "are attached so you can see the current state of the workspace.",
        "",
    ]
    if completed_steps:
        context_lines.append(f"COMPLETED STEPS ({len(completed_steps)}):")
        for i, s in enumerate(completed_steps):
            context_lines.append(f"  {i+1}. {step_summary(s)}")
        context_lines.append("")

    if stuck_step:
        context_lines.append(f"STUCK STEP: {step_summary(stuck_step)}")
        context_lines.append("  This step has not converged after 7 seconds.")
        context_lines.append("")

    if remaining_steps:
        context_lines.append(f"REMAINING STEPS ({len(remaining_steps)}):")
        for i, s in enumerate(remaining_steps):
            context_lines.append(f"  {i+1}. {step_summary(s)}")
        context_lines.append("")

    context_lines.append(
        "Re-evaluate the situation.  Return a NEW complete plan (JSON with "
        "\"plan\" key) to accomplish the ORIGINAL task from the current state. "
        "The stuck step will be skipped — plan from scratch for what remains."
    )
    stuck_addendum = "\n".join(context_lines)
    full_prompt = base_prompt + stuck_addendum

    console.write("[REQUERY] Sending requery with images to LLM...")

    # 6 — Build multimodal message
    user_content = [{"type": "text", "text": full_prompt}] + image_payloads

    body = json.dumps({
        "model": API_MODEL,
        "temperature": 1,
        "max_completion_tokens": 4096,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": (
                "You are a task planner for a Kinova Gen 3 robotic arm. "
                "A previous plan got stuck.  Fresh camera images show the "
                "current workspace state.  Return a NEW JSON plan to "
                "complete the original task.  Use ONLY the primitives "
                "documented in the prompt."
            )},
            {"role": "user", "content": user_content},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        console.write(f"[REQUERY] HTTP {e.code}: {err_body[:500]}")
        return None
    except Exception as e:
        console.write(f"[REQUERY] Request failed: {e}")
        return None

    # 7 — Parse response
    finish_reason = data.get("choices", [{}])[0].get("finish_reason", "?")
    print(f"[REQUERY] finish_reason={finish_reason}")
    try:
        content = data["choices"][0]["message"]["content"]
        if not content or not content.strip():
            console.write(f"[REQUERY] LLM returned empty content (finish_reason={finish_reason}).")
            return None
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]
        parsed = json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        console.write(f"[REQUERY] Failed to parse LLM response: {e}")
        return None

    if isinstance(parsed, dict) and "plan" in parsed:
        plan_list = parsed["plan"]
    elif isinstance(parsed, list):
        plan_list = parsed
    else:
        console.write(f"[REQUERY] Unexpected JSON shape: {str(parsed)[:300]}")
        return None

    if not plan_list:
        console.write("[REQUERY] Empty plan returned.")
        return None

    console.write(f"[REQUERY] New plan with {len(plan_list)} primitive(s):")
    for i, prim in enumerate(plan_list):
        console.write(f"  [{i+1}] {prim.get('cmd','?')}  {json.dumps(prim)}")

    # 8 — Expand into sub-steps
    flat_steps = expand_plan(plan_list, robot_id, ee_link_index, movable_joints,
                             scene_objects, console, loaded_objects=loaded_objects)
    if not flat_steps:
        console.write("[REQUERY] Plan expansion produced 0 sub-steps.")
        return None

    console.write(f"[REQUERY] New plan expanded to {len(flat_steps)} sub-step(s).")
    return flat_steps


# ---------------------------------------------------------------------------
# IDENTIFY OBJECTS — lightweight LLM call to determine GDINO search terms
# ---------------------------------------------------------------------------

def identify_objects_llm(user_request: str, console,
                         available_objects: list | None = None) -> dict | None:
    """Ask the LLM to interpret the user's request and return search terms.

    Sends a fast, text-only API call (no images) that asks the model:
      - What object(s) does the user want to interact with?
      - What is the best visual description for each, to feed to an
        open-vocabulary object detector (Grounding DINO)?
      - What action does the user want (pick, place, push, etc.)?

    Parameters
    ----------
    user_request : str
        The raw user command, e.g. "pick up the red block".
    console : object
        Console for status messages.
    available_objects : list or None
        Known object types on the table (e.g. ["block", "cylinder"]).
        Helps the LLM ground its answer to what's actually present.

    Returns
    -------
    dict or None
        On success::

            {
                "action": "pick",           # high-level intent
                "targets": [                # ordered list of objects
                    {
                        "name": "block",    # canonical name (or free-text)
                        "description": "a small red cube block"
                    }
                ],
                "search_queries": ["a small red cube block"],
                "reasoning": "User wants to pick up the red block..."
            }

        Returns None on API error.
    """
    if API_KEY == "YOUR_API_KEY_HERE":
        console.write("[IDENTIFY] ERROR: No API key configured — "
                      "falling back to text extraction.")
        return None

    avail_str = ""
    if available_objects:
        avail_str = (
            f"\n\nKnown objects currently on the table: "
            f"{', '.join(available_objects)}\n"
            "Prefer matching the user's description to one of these known "
            "objects when reasonable.  If the user describes something not "
            "in this list, use your best judgement for the search query."
        )

    system_msg = (
        "You are a vision-language assistant for a robotic arm.  "
        "Given a user command, determine:\n"
        "1. The high-level action (pick, place, push, point, inspect, etc.)\n"
        "2. Which object(s) the user is referring to\n"
        "3. A concise visual description of each object suitable for an "
        "open-vocabulary object detector (Grounding DINO).  The description "
        "should be a short noun phrase that would visually identify the "
        "object in a camera image — include colour, shape, and size when "
        "the user mentions them."
        + avail_str
    )

    user_msg = f"User command: \"{user_request}\""

    console.write("[IDENTIFY] Asking LLM to identify target objects...")

    _IDENTIFY_SCHEMA = {
        "type": "json_schema",
        "json_schema": {
            "name": "identify_objects",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "High-level verb: pick, place, push, point, stack, inspect, etc."
                    },
                    "targets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Short canonical name of the object, e.g. 'red block'"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Concise noun phrase for Grounding DINO detection"
                                }
                            },
                            "required": ["name", "description"],
                            "additionalProperties": False
                        },
                        "description": "Objects the user is referring to, in order of interaction"
                    },
                    "search_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Visual search queries for each target (same order as targets)"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "One sentence explaining interpretation"
                    }
                },
                "required": ["action", "targets", "search_queries", "reasoning"],
                "additionalProperties": False
            }
        }
    }

    body = json.dumps({
        "model": API_MODEL,
        "temperature": 1,
        "max_completion_tokens": 512,
        "response_format": _IDENTIFY_SCHEMA,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        console.write(f"[IDENTIFY] HTTP {e.code}: {err_body[:300]}")
        return None
    except Exception as e:
        console.write(f"[IDENTIFY] Request failed: {e}")
        return None

    try:
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        console.write(f"[IDENTIFY] Failed to parse LLM response: {e}")
        raw_msg = data.get("choices", [{}])[0].get("message", {})
        console.write(f"[IDENTIFY] Raw: {str(raw_msg)[:300]}")
        return None

    action = parsed["action"]
    targets = parsed["targets"]
    search_queries = parsed["search_queries"]
    reasoning = parsed["reasoning"]

    console.write(f"[IDENTIFY] Action: {action}")
    for t in targets:
        console.write(f"[IDENTIFY]   Target: '{t.get('name', '?')}' "
                      f"→ search: \"{t.get('description', '?')}\"")
    if reasoning:
        console.write(f"[IDENTIFY] Reasoning: {reasoning}")

    return parsed


# =========================================================================
#  JARVISPLAN  —  LLM task-planning test (no vision, no execution)
# =========================================================================

def _synthetic_scene(object_names: list) -> dict:
    """Generate fake ring-table positions for every object in *object_names*.

    Objects are evenly spaced around a ring of radius 0.945 m (TABLE_MID_R)
    at z = 1.24 m (approx table surface).  These are cosmetic — the plan is
    never executed, so exact positions don't matter.
    """
    import math as _math
    TABLE_Z = 1.24
    OBJ_R = 0.945
    n = len(object_names)
    scene = {}
    for i, name in enumerate(sorted(object_names)):
        angle = _math.radians(360 * i / max(n, 1))
        scene[name] = {
            "x": round(OBJ_R * _math.cos(angle), 4),
            "y": round(OBJ_R * _math.sin(angle), 4),
            "z": TABLE_Z,
        }
    return scene


def build_planning_prompt(user_request: str, scene_objects: dict,
                          object_library: list) -> str:
    """Build the jarvisplan prompt — like build_dispatch_prompt but with an
    extra ANALYSIS block that instructs the LLM to ask clarifying questions
    when the task is ambiguous or information is missing.

    No vision/bbox sections — text-only.
    """

    # --- Objects block (all library objects with synthetic positions) ---
    objects_block = "OBJECTS IN SCENE (all objects currently on the table):\n"
    for name, pos in scene_objects.items():
        desc = _GDINO_QUERIES.get(name, OBJECT_DESCRIPTIONS.get(name, name))
        objects_block += (f"  \u2022 \"{name}\" \u2014 {desc}\n"
                         f"      position: x={pos['x']:.4f}  y={pos['y']:.4f}  z={pos['z']:.4f}\n")
    objects_block += "\n"

    # --- Reuse the same primitives doc from build_dispatch_prompt ---
    primitives_doc = (
        "AVAILABLE PRIMITIVES (use cmd names EXACTLY):\n\n"
        "GRASPING & RELEASING:\n"
        '  1.  pick          \u2014 Side-grasp an object.\n'
        '      {"cmd":"pick","object":"<name>"}\n'
        '  2.  place         \u2014 Place held object at XYZ.\n'
        '      {"cmd":"place","x":f,"y":f,"z":f}\n'
        '  3.  open_gripper  \u2014 Open gripper fully.\n'
        '      {"cmd":"open_gripper"}\n'
        '  4.  close_gripper \u2014 Close gripper fully.\n'
        '      {"cmd":"close_gripper"}\n'
        '  5.  drop          \u2014 Release from current height.\n'
        '      {"cmd":"drop"}\n'
        '  6.  top_grasp     \u2014 Top-down grasp (approach from above).\n'
        '      {"cmd":"top_grasp","object":"<name>"}\n'
        '  7.  regrasp       \u2014 Open gripper slightly, adjust, re-close.\n'
        '      {"cmd":"regrasp"}\n'
        '  8.  pick_and_place\u2014 Combined pick then place.\n'
        '      {"cmd":"pick_and_place","object":"<name>","x":f,"y":f,"z":f}\n\n'
        "MOVEMENT & POSITIONING:\n"
        '  9.  move_to       \u2014 Move EE to world XYZ.\n'
        '      {"cmd":"move_to","x":f,"y":f,"z":f}\n'
        '  10. home          \u2014 Return arm to upright rest pose.\n'
        '      {"cmd":"home"}\n'
        '  11. lift_high     \u2014 Lift whatever is held high above table.\n'
        '      {"cmd":"lift_high"}\n'
        '  12. move_above    \u2014 Move EE directly above an object at safe height.\n'
        '      {"cmd":"move_above","object":"<name>"}\n'
        '  13. approach_from_top \u2014 Descend straight down to object from above.\n'
        '      {"cmd":"approach_from_top","object":"<name>"}\n'
        '  14. approach_from_side\u2014 Move to object from the side at its height.\n'
        '      {"cmd":"approach_from_side","object":"<name>"}\n'
        '  15. retract       \u2014 Pull EE straight back from current position.\n'
        '      {"cmd":"retract","distance":f}\n'
        '  16. move_relative \u2014 Move EE by a relative offset (dx,dy,dz).\n'
        '      {"cmd":"move_relative","dx":f,"dy":f,"dz":f}\n'
        '  17. align_with    \u2014 Position EE above object at specified height.\n'
        '      {"cmd":"align_with","object":"<name>","height":f}\n'
        '  18. hover         \u2014 Hover at specified height above object.\n'
        '      {"cmd":"hover","object":"<name>","clearance":f}\n'
        '  19. park          \u2014 Tuck arm into a compact safe position.\n'
        '      {"cmd":"park"}\n\n'
        "PUSHING & CONTACT:\n"
        '  20. push          \u2014 Push object along (dx,dy) on table.\n'
        '      {"cmd":"push","object":"<name>","dx":f,"dy":f}\n'
        '  21. nudge         \u2014 Tiny push to fine-adjust position.\n'
        '      {"cmd":"nudge","object":"<name>","dx":f,"dy":f}\n'
        '  22. drag          \u2014 Grip & drag object across surface.\n'
        '      {"cmd":"drag","object":"<name>","dx":f,"dy":f}\n'
        '  23. tap           \u2014 Tap the top of an object once.\n'
        '      {"cmd":"tap","object":"<name>"}\n'
        '  24. knock_over    \u2014 Knock a standing object sideways.\n'
        '      {"cmd":"knock_over","object":"<name>"}\n'
        '  25. touch         \u2014 Gently touch top of object and hold.\n'
        '      {"cmd":"touch","object":"<name>"}\n'
        '  26. slide         \u2014 Grip object, slide it along table.\n'
        '      {"cmd":"slide","object":"<name>","dx":f,"dy":f}\n\n'
        "WRIST / IN-HAND:\n"
        '  27. rotate_object \u2014 Rotate held object about wrist axis.\n'
        '      {"cmd":"rotate_object","angle_deg":f}\n'
        '  28. flip_object   \u2014 Flip held object 180\u00b0.\n'
        '      {"cmd":"flip_object"}\n'
        '  29. tilt_object   \u2014 Tilt held object by angle_deg.\n'
        '      {"cmd":"tilt_object","angle_deg":f}\n'
        '  30. shake         \u2014 Shake held object back and forth.\n'
        '      {"cmd":"shake"}\n'
        '  31. pour          \u2014 Tilt held object ~90\u00b0 to pour.\n'
        '      {"cmd":"pour"}\n\n'
        "COMPOUND TASKS:\n"
        '  32. stack         \u2014 Pick first object, stack on second.\n'
        '      {"cmd":"stack","object":"<name>","target":"<name>"}\n'
        '  33. sort          \u2014 Move object to angle on table ring.\n'
        '      {"cmd":"sort","object":"<name>","angle_deg":f}\n'
        '  34. line_up       \u2014 Arrange listed objects in a line.\n'
        '      {"cmd":"line_up","objects":["<n1>","<n2>",...],"y":f}\n'
        '  35. handover      \u2014 Pick up & present towards human.\n'
        '      {"cmd":"handover","object":"<name>"}\n'
        '  36. deliver       \u2014 Pick object & place on target (e.g. person\'s tray).\n'
        '      {"cmd":"deliver","object":"<name>","target":"<name>"}\n'
        '  36b. present     \u2014 Pick object & hold near person (face height, above tray).\n'
        '                     Use for interactions: brush teeth, show item, etc.\n'
        '      {"cmd":"present","object":"<name>","target":"<name>"}\n'
        '  37. inspect       \u2014 Pick up & bring to EE camera.\n'
        '      {"cmd":"inspect","object":"<name>"}\n'
        '  38. pick_and_place\u2014 Pick object then place at XYZ.\n'
        '      {"cmd":"pick_and_place","object":"<name>","x":f,"y":f,"z":f}\n'
        '  39. swap          \u2014 Swap positions of two objects.\n'
        '      {"cmd":"swap","object_a":"<name>","object_b":"<name>"}\n'
        '  40. group         \u2014 Gather listed objects to a single XY.\n'
        '      {"cmd":"group","objects":["<n1>","<n2>",...],"x":f,"y":f}\n'
        '  41. scatter       \u2014 Spread objects apart from center.\n'
        '      {"cmd":"scatter","objects":["<n1>","<n2>",...]}\n'
        '  42. clear_area    \u2014 Push objects away from an XY region.\n'
        '      {"cmd":"clear_area","x":f,"y":f,"radius":f}\n\n'
        "SWEEPING:\n"
        '  43. sweep_table   \u2014 Arc sweep from start_deg to end_deg.\n'
        '      {"cmd":"sweep_table","start_deg":f,"end_deg":f}\n'
        '  44. circle_sweep  \u2014 Full 360\u00b0 sweep to clear table.\n'
        '      {"cmd":"circle_sweep"}\n\n'
        "GESTURES:\n"
        '  45. wave          \u2014 Friendly wave gesture.\n'
        '      {"cmd":"wave"}\n'
        '  46. point_at      \u2014 Point at an object.\n'
        '      {"cmd":"point_at","object":"<name>"}\n'
        '  47. guard         \u2014 Hover arm protectively over object.\n'
        '      {"cmd":"guard","object":"<name>"}\n\n'
        "NAVIGATION & PATHS:\n"
        '  48. orbit         \u2014 Move EE in a circle around an object.\n'
        '      {"cmd":"orbit","object":"<name>","radius":f,"height":f}\n'
        '  49. zigzag        \u2014 Zigzag EE between two XY points.\n'
        '      {"cmd":"zigzag","x1":f,"y1":f,"x2":f,"y2":f,"z":f}\n'
        '  50. patrol        \u2014 Visit a sequence of XYZ waypoints.\n'
        '      {"cmd":"patrol","waypoints":[{"x":f,"y":f,"z":f},...]}\n\n'
        "BASIC JOINT CONTROL:\n"
        '  51. rotate        \u2014 Rotate base (joint 1) by angle_deg degrees.\n'
        '      {"cmd":"rotate","angle_deg":f}\n'
        '  52. straighten    \u2014 Reset joints 1-5 to 0 (arm upright, keeps wrist).\n'
        '      {"cmd":"straighten"}\n'
    )

    # --- Rules (same as normal + question-asking rule) ---
    rules = (
        "\nRULES:\n"
        '  - Object names must EXACTLY match OBJECTS IN SCENE.\n'
        "  - Use the provided xyz coords \u2014 do NOT invent coordinates.\n"
        "  - Keep plans short (typically 2\u20136 steps).\n"
        "  - 'bring me'/'give me' = deliver to person.\n"
        "  - End with home if the arm should park afterwards.\n"
    )

    # --- Examples ---
    examples = (
        "\nEXAMPLES:\n"
        '  "pick up the block" (straightforward)\n'
        '  → {"plan":[{"cmd":"pick","object":"block"},{"cmd":"home"}]}\n\n'
        '  "make me tea" (ambiguous — need preferences)\n'
        '  → {"questions":["What kind of tea would you like?","Do you take sugar or milk?"]}\n\n'
        '  "bring me something to drink"\n'
        '  → {"questions":["Would you like water from the water_cup or the water_bottle?"]}\n'
    )

    # --- The key analysis instruction ---
    analysis_block = (
        "TASK ANALYSIS (IMPORTANT):\n"
        "  Before generating a plan, analyse the user's request against the\n"
        "  objects currently in the scene and the capabilities of the robot.\n\n"
        "  Ask yourself:\n"
        "    1. Does the scene contain every object needed for this task?\n"
        "    2. Does the task require user preferences the request doesn't\n"
        "       specify (e.g. flavour, temperature, quantity)?\n"
        "    3. Are there any ambiguities about WHICH object to use?\n\n"
        "  IF information is missing or ambiguous:\n"
        '    Return {"questions": ["<q1>", "<q2>", ...]}  with the MINIMUM\n'
        "    number of short, specific clarifying questions needed.\n"
        "    Do NOT ask questions you can reasonably infer the answer to.\n"
        "    Do NOT ask more than 3 questions.\n\n"
        "  IF you have enough information:\n"
        '    Return {"plan": [...]}  as usual with motion primitives.\n\n'
        "  NEVER return both keys.  Return questions OR plan, not both.\n\n"
    )

    prompt = (
        "=== KINOVA GEN 3 — TASK PLANNER (analysis mode) ===\n\n"
        "You are a high-level task planner for a robotic arm in an assistive\n"
        "daily-living scenario.  You can see which objects are on the table\n"
        "and you must decide whether you have enough information to act.\n\n"
        f"{analysis_block}"
        "ROBOT INFO:\n"
        "  Base: (0, 0, 1.18)  |  Reach: ~0.7 m  |  Table z ≈ 1.24 m\n\n"
        f"{objects_block}"
        f"{primitives_doc}"
        f"{rules}"
        f"{examples}\n"
        f"USER TASK:  \"{user_request}\"\n\n"
        "Return a single JSON object.  If you need more info return "
        '{"questions": ["...", ...]}.  '
        'If you can act return {"plan": [{"cmd":"...", ...}, ...]}.'
    )
    return prompt


def call_jarvisplan(user_request: str, console,
                    object_names: list | None = None) -> dict | None:
    """LLM-only task planning test — no vision, no execution.

    Assumes all objects in *object_names* (defaults to full library from
    _GDINO_QUERIES) are present on the table with synthetic positions.
    Returns {"questions": [...]} or {"plan": [...]} or None on error.
    """
    if API_KEY == "YOUR_API_KEY_HERE":
        console.write("[JARVISPLAN] ERROR: No API key configured.")
        return None

    # Object library — use all known DINO objects if none specified
    if not object_names:
        object_names = list(_GDINO_QUERIES.keys())

    # Synthetic scene positions (evenly spaced on the ring table)
    scene = _synthetic_scene(object_names)

    # Build prompt
    text_prompt = build_planning_prompt(user_request, scene, object_names)

    system_msg = (
        "You are a task planner for a Kinova Gen 3 robotic arm in an "
        "assistive daily-living scenario. You must decide whether the "
        "user's request can be fulfilled with the objects in the scene "
        "and the information given. If anything is ambiguous or missing, "
        "ask the minimum clarifying questions. Otherwise return a plan. "
        "Return a JSON object with EITHER a \"questions\" key (array of "
        "strings) OR a \"plan\" key (array of motion primitives). "
        "Never return both."
    )

    body = json.dumps({
        "model": "gpt-5",
        "temperature": 1,
        "max_completion_tokens": 4096,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": text_prompt},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )

    console.write("[JARVISPLAN] Calling LLM...")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        console.write(f"[JARVISPLAN] HTTP {e.code}: {err_body[:300]}")
        return None
    except Exception as e:
        console.write(f"[JARVISPLAN] Request failed: {e}")
        return None

    # Parse response
    finish_reason = data.get("choices", [{}])[0].get("finish_reason", "?")
    print(f"[JARVISPLAN] finish_reason={finish_reason}")
    try:
        content = data["choices"][0]["message"]["content"]
        if not content or not content.strip():
            console.write(f"[JARVISPLAN] LLM returned empty response (finish_reason={finish_reason}).")
            # Dump raw response for debugging
            print(f"[JARVISPLAN] Raw API data: {json.dumps(data, indent=2)[:1000]}")
            return None
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]
        parsed = json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        console.write(f"[JARVISPLAN] Failed to parse response: {e}")
        try:
            raw = data["choices"][0]["message"]["content"]
            print(f"[JARVISPLAN] Raw: {str(raw)[:500]}")
        except Exception:
            pass
        return None

    # Accept {"questions": [...]} or {"plan": [...]}
    if isinstance(parsed, dict):
        if "questions" in parsed:
            return {"questions": parsed["questions"]}
        if "plan" in parsed:
            return {"plan": parsed["plan"]}

    console.write("[JARVISPLAN] Unexpected response format.")
    print(f"[JARVISPLAN] Parsed: {str(parsed)[:500]}")
    return None
