"""
Jarvis 3 — Primitive expansion functions and dispatch table.

All 50 expansion functions live here.  Each takes (cmd_dict, ctx_dict) and
returns a list of sub-step dicts (pose / gripper), or None on failure.

ctx keys:  robot_id, ee_link, joints, scene_objects, console
"""

import math
import numpy as np
import pybullet as p

from kinova_constants import REAL_JOINT_LIMITS

# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def _rotmat_to_quat(m):
    """Rotation matrix (3x3 numpy) -> quaternion [x, y, z, w] (Shepperd)."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        qx = (m[2, 1] - m[1, 2]) * s
        qy = (m[0, 2] - m[2, 0]) * s
        qz = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return [qx, qy, qz, w]


def _compute_side_grasp_orn(x: float, y: float):
    """Return quaternion [x,y,z,w] for a side-approach grasp toward (x,y)
    from the robot base at (0,0).  Camera nub faces UP."""
    dx, dy = x, y
    dist_xy = math.sqrt(dx * dx + dy * dy)
    if dist_xy < 1e-6:
        dx, dy, dist_xy = 1.0, 0.0, 1.0
    ax, ay = dx / dist_xy, dy / dist_xy
    R = np.array([[-ay, 0.0, ax],
                  [ ax, 0.0, ay],
                  [0.0, 1.0, 0.0]], dtype=np.float64)
    return _rotmat_to_quat(R)


# ---------------------------------------------------------------------------
# IK solver
# ---------------------------------------------------------------------------

def _solve_ik(robot_id, ee_link_index, target_pos, movable_joints,
              target_orn=None, max_iters=100) -> list:
    """Run PyBullet IK and return [{joint, target}, ...] for movable arm joints.

    Uses null-space parameters for the 7-DOF arm so the solver biases toward
    an elbow-up, collision-free configuration instead of picking an arbitrary
    solution from the infinite set of redundant configurations.
    """
    # Build null-space arrays for arm joints (skip gripper etc.)
    ns_lo, ns_hi, ns_rng, ns_rst = [], [], [], []
    for j in movable_joints:
        if j["name"].startswith("joint_"):
            lo, hi = REAL_JOINT_LIMITS.get(j["name"], (j["lower"], j["upper"]))
            ns_lo.append(lo)
            ns_hi.append(hi)
            ns_rng.append(hi - lo)
            if j["name"] == "joint_2":
                ns_rst.append(0.5)
            elif j["name"] == "joint_4":
                ns_rst.append(-1.0)
            else:
                ns_rst.append(0.0)

    kwargs = dict(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=ee_link_index,
        targetPosition=target_pos,
        maxNumIterations=max_iters,
        residualThreshold=1e-4,
        lowerLimits=ns_lo,
        upperLimits=ns_hi,
        jointRanges=ns_rng,
        restPoses=ns_rst,
    )
    if target_orn is not None:
        kwargs["targetOrientation"] = target_orn

    joint_angles = p.calculateInverseKinematics(**kwargs)

    # Map IK results to our movable arm joints (skip gripper)
    steps = []
    for j, angle in zip(movable_joints, joint_angles):
        if j["name"].startswith("joint_"):
            lo, hi = REAL_JOINT_LIMITS.get(j["name"], (j["lower"], j["upper"]))
            clamped = max(lo, min(hi, angle))
            steps.append({"joint": j["name"], "target": round(clamped, 4)})
    return steps


# ---------------------------------------------------------------------------
# IK convenience: returns dict {joint_name: target_rad} or None
# ---------------------------------------------------------------------------

def _ik_dict(robot_id, ee_link_index, pos, movable_joints,
             target_orn=None, console=None):
    """Solve IK and return {joint: target} dict, or None on failure."""
    ik = _solve_ik(robot_id, ee_link_index, pos, movable_joints,
                   target_orn=target_orn)
    if not ik:
        if console:
            console.write(f"[PLAN] IK failed for ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        return None
    return {s["joint"]: s["target"] for s in ik}


def _pose_step(targets: dict, label: str) -> dict:
    return {"type": "pose", "targets": targets, "label": label}

def _grip_step(val: float) -> dict:
    return {"type": "gripper", "target": val}

GRIP_OPEN = 1.2
GRIP_CLOSE = 0.0
TABLE_Z = 1.24          # safe object-grasp z (slightly above surface)
LIFT_Z  = 1.42          # comfortable above-table height
HIGH_Z  = 1.70          # "lift high" z
APPROACH_BACK = 0.12    # 12 cm offset for pre-grasp
GRIPPER_DEPTH = 0.00    # creep-to-contact handles alignment now
HOME_JOINTS = None       # resolved lazily (need movable_joints at runtime)

def _get_home_targets(movable_joints):
    """Return joint targets for the upright home pose."""
    return {j["name"]: 0.0 for j in movable_joints}


# ---------------------------------------------------------------------------
# 27 PRIMITIVE EXPANSION FUNCTIONS
# Each returns a list of sub-steps (pose / gripper dicts), or None on error.
# Ctx = dict with keys: robot_id, ee_link, joints, scene_objects, console
# ---------------------------------------------------------------------------

def _expand_pick(cmd, ctx):
    """Side-grasp pick sequence:  open -> pre-grasp -> approach -> close -> lift.

    Uses camera-derived position and width from scene_objects (no p.getAABB).
    """
    obj = cmd.get("object")
    pos = ctx["scene_objects"].get(obj)
    if not pos:
        ctx["console"].write(f"[PLAN] pick: unknown object '{obj}'")
        return None
    x, y, z = pos["x"], pos["y"], pos["z"]
    # Vision-estimated width from DINO bbox; default 0.05m if unavailable
    width = pos.get("width", 0.05)

    # Approach axes
    dx, dy = x, y
    d = math.sqrt(dx*dx + dy*dy) or 1.0
    ax, ay = dx/d, dy/d

    # Lateral axis (finger separation direction)
    lat = np.array([-ay, ax, 0.0], dtype=np.float64)
    lat_norm = np.linalg.norm(lat)
    if lat_norm < 1e-6:
        lat = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        lat = lat / lat_norm

    steps = [_grip_step(GRIP_OPEN)]
    steps[-1]["label"] = "Open gripper"

    # Use vision-estimated width for grasp planning
    centre = np.array([x, y, z], dtype=np.float64)

    # Pre-grasp: back off along approach direction at object height
    pre = [x - APPROACH_BACK*ax, y - APPROACH_BACK*ay, z + 0.04]

    # Build a look-at orientation where forward points to the object centre and
    # right aligns with the lateral axis, so the fingers face the grasp extents.
    ee_pos = np.array(pre, dtype=np.float64)
    target = centre.copy()
    f = target - ee_pos
    fnorm = np.linalg.norm(f)
    if fnorm < 1e-6:
        f = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        fnorm = 1.0
    f = f / fnorm
    # ensure lateral is orthogonal to forward
    if abs(np.dot(f, lat)) > 0.999:
        # pick an alternative lateral
        lat = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    r = lat / (np.linalg.norm(lat) or 1.0)
    # Tilt forward vector ~15° downward (Rodrigues around lateral axis)
    # so the EE camera nub above the gripper clears the table surface.
    _TILT = math.radians(15)
    f = f * math.cos(_TILT) + np.cross(r, f) * math.sin(_TILT)
    f = f / (np.linalg.norm(f) or 1.0)
    u = np.cross(f, r)
    R = np.column_stack((r, u, f))
    orn = _rotmat_to_quat(R)

    t = _ik_dict(ctx["robot_id"], ctx["ee_link"], pre, ctx["joints"],
                 target_orn=orn, console=ctx["console"])
    if not t:
        return None
    steps.append(_pose_step(t, f"Approach {obj}"))

    # Visual-servoing approach: from pre-grasp, re-detect object via EE camera
    # and correct the contact position before driving to it.
    # Offset contact BACKWARD so the fingertip grasp envelope (not the EE
    # origin) is centred on the object.
    contact_x = x - GRIPPER_DEPTH * ax
    contact_y = y - GRIPPER_DEPTH * ay
    contact = [contact_x, contact_y, z]
    t2 = _ik_dict(ctx["robot_id"], ctx["ee_link"], contact, ctx["joints"],
                  target_orn=orn, console=ctx["console"])
    if not t2:
        return None

    # Build null-space dict for IK re-solve during servo correction
    ns_lo, ns_hi, ns_rng, ns_rst = [], [], [], []
    for j in ctx["joints"]:
        if j["name"].startswith("joint_"):
            lo, hi = REAL_JOINT_LIMITS.get(j["name"], (j["lower"], j["upper"]))
            ns_lo.append(lo)
            ns_hi.append(hi)
            ns_rng.append(hi - lo)
            if j["name"] == "joint_2":
                ns_rst.append(0.5)
            elif j["name"] == "joint_4":
                ns_rst.append(-1.0)
            else:
                ns_rst.append(0.0)

    steps.append({
        "type": "servo_contact",
        "nominal_targets": t2,
        "quat": list(orn),
        "grasp_target": obj,
        "ax": ax, "ay": ay, "cz": z,
        "_cx": contact_x, "_cy": contact_y,
        "_null_space": {"lo": ns_lo, "hi": ns_hi, "rng": ns_rng, "rst": ns_rst},
        "label": f"Align {obj}",
    })

    # Creep forward until lightest contact with the object, then stop.
    # This ensures the fingertips are precisely touching before grip close.
    steps.append({
        "type": "creep_contact",
        "ax": ax, "ay": ay,
        "quat": list(orn),
        "max_mm": 30,              # safety cap: 3 cm max travel
        "label": f"Creep to {obj}",
    })

    # Force-sensing gripper close (detect contact instead of hard slam)
    steps.append({"type": "gripper_force", "force_threshold": 5.0,
                  "label": f"Grip {obj}"})

    # Micro-lift test: raise 2 cm, check if object slipped, re-grip if needed
    steps.append({
        "type": "grip_verify",
        "x": x, "y": y, "z": z,
        "lift_delta": 0.02,
        "max_retries": 2,
        "label": f"Verify grip {obj}",
    })

    # lift  (no orientation constraint — let IK find a clean upward path)
    t3 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x, y, LIFT_Z], ctx["joints"],
                  console=ctx["console"])
    if not t3:
        return None
    steps.append(_pose_step(t3, f"Lift {obj}"))
    return steps


def _expand_place(cmd, ctx):
    """Lower held object to XYZ -> open -> retract up."""
    x, y, z = cmd.get("x",0), cmd.get("y",0), cmd.get("z", TABLE_Z)
    orn = _compute_side_grasp_orn(x, y)
    steps = []
    # above target
    t1 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x, y, z+0.08], ctx["joints"],
                  target_orn=orn, console=ctx["console"])
    if not t1: return None
    steps.append(_pose_step(t1, f"place-above({x:.3f},{y:.3f},{z+0.08:.3f})"))

    # at target
    t2 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x, y, z], ctx["joints"],
                  target_orn=orn, console=ctx["console"])
    if not t2: return None
    steps.append(_pose_step(t2, f"place({x:.3f},{y:.3f},{z:.3f})"))

    steps.append(_grip_step(GRIP_OPEN))

    # retract up
    t3 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x, y, LIFT_Z], ctx["joints"],
                  console=ctx["console"])
    if not t3: return None
    steps.append(_pose_step(t3, f"retract({x:.3f},{y:.3f},{LIFT_Z:.3f})"))
    return steps


def _expand_move_to(cmd, ctx):
    """Simple EE move to XYZ."""
    x, y, z = cmd.get("x",0), cmd.get("y",0), cmd.get("z", LIFT_Z)
    t = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x,y,z], ctx["joints"],
                 console=ctx["console"])
    if not t: return None
    return [_pose_step(t, f"move_to({x:.3f},{y:.3f},{z:.3f})")]


def _expand_home(cmd, ctx):
    """Return to upright home pose."""
    targets = _get_home_targets(ctx["joints"])
    return [_pose_step(targets, "home")]


def _expand_open_gripper(cmd, ctx):
    return [_grip_step(GRIP_OPEN)]


def _expand_close_gripper(cmd, ctx):
    return [_grip_step(GRIP_CLOSE)]


def _expand_drop(cmd, ctx):
    """Just open the gripper where we are."""
    return [_grip_step(GRIP_OPEN)]


def _expand_lift_high(cmd, ctx):
    """Lift EE straight up to HIGH_Z."""
    ee = p.getLinkState(ctx["robot_id"], ctx["ee_link"], computeForwardKinematics=True)
    cx, cy = ee[4][0], ee[4][1]
    t = _ik_dict(ctx["robot_id"], ctx["ee_link"], [cx, cy, HIGH_Z], ctx["joints"],
                 console=ctx["console"])
    if not t: return None
    return [_pose_step(t, f"lift_high({cx:.3f},{cy:.3f},{HIGH_Z:.3f})")]


def _expand_push(cmd, ctx):
    """Move behind object, push in (dx,dy) direction."""
    obj = cmd.get("object")
    pos = ctx["scene_objects"].get(obj)
    if not pos:
        ctx["console"].write(f"[PLAN] push: unknown object '{obj}'")
        return None
    x, y, z = pos["x"], pos["y"], pos["z"]
    dx, dy = cmd.get("dx", 0), cmd.get("dy", 0)
    push_dist = math.sqrt(dx*dx + dy*dy) or 0.10
    ndx, ndy = dx/push_dist if push_dist > 0 else 1.0, dy/push_dist if push_dist > 0 else 0.0

    steps = [_grip_step(GRIP_CLOSE)]
    # behind object
    bx, by = x - 0.08*ndx, y - 0.08*ndy
    t1 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [bx, by, z], ctx["joints"],
                  console=ctx["console"])
    if not t1: return None
    steps.append(_pose_step(t1, f"push-start({bx:.3f},{by:.3f},{z:.3f})"))

    # push through
    fx, fy = x + dx, y + dy
    t2 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [fx, fy, z], ctx["joints"],
                  console=ctx["console"])
    if not t2: return None
    steps.append(_pose_step(t2, f"push-end({fx:.3f},{fy:.3f},{z:.3f})"))

    # retract up
    t3 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [fx, fy, LIFT_Z], ctx["joints"],
                  console=ctx["console"])
    if not t3: return None
    steps.append(_pose_step(t3, f"push-retract"))
    return steps


def _expand_nudge(cmd, ctx):
    """Tiny push (dx,dy) -- same as push but smaller motion."""
    return _expand_push(cmd, ctx)


def _expand_drag(cmd, ctx):
    """Grip object -> drag along table by (dx,dy) -> release."""
    obj = cmd.get("object")
    pos = ctx["scene_objects"].get(obj)
    if not pos:
        ctx["console"].write(f"[PLAN] drag: unknown object '{obj}'")
        return None
    x, y, z = pos["x"], pos["y"], pos["z"]
    dx, dy = cmd.get("dx", 0), cmd.get("dy", 0)
    orn = _compute_side_grasp_orn(x, y)
    d = math.sqrt(x*x + y*y) or 1.0
    ax, ay = x/d, y/d

    steps = [_grip_step(GRIP_OPEN)]
    # approach
    pre = [x - APPROACH_BACK*ax, y - APPROACH_BACK*ay, z]
    t1 = _ik_dict(ctx["robot_id"], ctx["ee_link"], pre, ctx["joints"],
                  target_orn=orn, console=ctx["console"])
    if not t1: return None
    steps.append(_pose_step(t1, "drag-pre"))
    t2 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x,y,z], ctx["joints"],
                  target_orn=orn, console=ctx["console"])
    if not t2: return None
    steps.append(_pose_step(t2, "drag-grab"))
    steps.append(_grip_step(GRIP_CLOSE))

    # drag to target
    tx, ty = x+dx, y+dy
    orn2 = _compute_side_grasp_orn(tx, ty)
    t3 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [tx, ty, z], ctx["joints"],
                  target_orn=orn2, console=ctx["console"])
    if not t3: return None
    steps.append(_pose_step(t3, f"drag-to({tx:.3f},{ty:.3f})"))
    steps.append(_grip_step(GRIP_OPEN))

    # retract
    t4 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [tx, ty, LIFT_Z], ctx["joints"],
                  console=ctx["console"])
    if not t4: return None
    steps.append(_pose_step(t4, "drag-retract"))
    return steps


def _expand_tap(cmd, ctx):
    """Tap the top of an object: move above -> descend -> retract."""
    obj = cmd.get("object")
    pos = ctx["scene_objects"].get(obj)
    if not pos:
        ctx["console"].write(f"[PLAN] tap: unknown object '{obj}'")
        return None
    x, y, z = pos["x"], pos["y"], pos["z"]
    steps = []
    # above
    t1 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x, y, z+0.15], ctx["joints"],
                  console=ctx["console"])
    if not t1: return None
    steps.append(_pose_step(t1, "tap-above"))
    # tap
    t2 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x, y, z+0.02], ctx["joints"],
                  console=ctx["console"])
    if not t2: return None
    steps.append(_pose_step(t2, "tap-contact"))
    # retract
    steps.append(_pose_step(t1, "tap-retract"))
    return steps


def _expand_knock_over(cmd, ctx):
    """Side-push an object to knock it over."""
    obj = cmd.get("object")
    pos = ctx["scene_objects"].get(obj)
    if not pos:
        ctx["console"].write(f"[PLAN] knock_over: unknown object '{obj}'")
        return None
    x, y, z = pos["x"], pos["y"], pos["z"]
    # push perpendicular to radial direction
    d = math.sqrt(x*x + y*y) or 1.0
    perp_x, perp_y = -y/d, x/d  # perpendicular in XY
    steps = [_grip_step(GRIP_CLOSE)]
    sx, sy = x - 0.10*perp_x, y - 0.10*perp_y
    t1 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [sx, sy, z], ctx["joints"],
                  console=ctx["console"])
    if not t1: return None
    steps.append(_pose_step(t1, "knock-wind-up"))
    ex, ey = x + 0.12*perp_x, y + 0.12*perp_y
    t2 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [ex, ey, z], ctx["joints"],
                  console=ctx["console"])
    if not t2: return None
    steps.append(_pose_step(t2, "knock-strike"))
    t3 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [ex, ey, LIFT_Z], ctx["joints"],
                  console=ctx["console"])
    if not t3: return None
    steps.append(_pose_step(t3, "knock-retract"))
    return steps


def _expand_rotate_object(cmd, ctx):
    """Rotate wrist (joint_7) by angle_deg while holding an object."""
    angle = math.radians(cmd.get("angle_deg", 90))
    # Find joint_7 in movable_joints
    j7 = next((j for j in ctx["joints"] if j["name"] == "joint_7"), None)
    if j7 is None:
        return None
    cur = p.getJointState(ctx["robot_id"], j7["index"])[0]
    new_target = cur + angle
    targets = {}
    for j in ctx["joints"]:
        jn = j["name"]
        targets[jn] = p.getJointState(ctx["robot_id"], j["index"])[0]
    targets["joint_7"] = new_target
    return [_pose_step(targets, f"rotate_wrist({math.degrees(angle):.0f}\u00b0)")]


def _expand_flip_object(cmd, ctx):
    """Flip 180 deg -- just rotate_object by 180."""
    cmd2 = dict(cmd)
    cmd2["angle_deg"] = 180
    return _expand_rotate_object(cmd2, ctx)


def _expand_stack(cmd, ctx):
    """Pick object, place on top of target object."""
    obj = cmd.get("object")
    target = cmd.get("target")
    tpos = ctx["scene_objects"].get(target)
    if not tpos:
        ctx["console"].write(f"[PLAN] stack: unknown target '{target}'")
        return None
    # pick the object
    pick_steps = _expand_pick({"cmd": "pick", "object": obj}, ctx)
    if not pick_steps:
        return None
    # place above target (stack height ~+0.06)
    px, py, pz = tpos["x"], tpos["y"], tpos["z"] + 0.06
    place_steps = _expand_place({"cmd": "place", "x": px, "y": py, "z": pz}, ctx)
    if not place_steps:
        return None
    return pick_steps + place_steps


def _expand_handover(cmd, ctx):
    """Pick up object -> present it forward (toward x>0, high)."""
    pick_steps = _expand_pick({"cmd": "pick", "object": cmd.get("object")}, ctx)
    if not pick_steps:
        return None
    # present position: in front, high
    t = _ik_dict(ctx["robot_id"], ctx["ee_link"], [0.45, 0.0, 1.55], ctx["joints"],
                 console=ctx["console"])
    if not t: return None
    return pick_steps + [_pose_step(t, "handover-present")]


def _expand_deliver(cmd, ctx):
    """Pick object, hover above tray, open gripper to drop.

    The person's tray position is hardcoded (mirrors create_person geometry)
    so delivery works even if vision can't detect the person.
    Hovers above the tray with enough clearance to avoid collisions,
    then simply opens the gripper to let the object fall onto the tray.
    """
    pick_steps = _expand_pick({"cmd": "pick", "object": cmd.get("object")}, ctx)
    if not pick_steps:
        return None

    # ---- Hardcoded tray position (mirrors compute_object_positions_vision) ----
    _ANG = math.radians(-100)
    _R   = 0.65
    _BZ  = 1.235
    _bx  = _R * math.cos(_ANG)
    _by  = _R * math.sin(_ANG)
    _facing = math.atan2(-_by, -_bx)
    _cf, _sf = math.cos(_facing), math.sin(_facing)
    _torso_hy = 0.035; _fore_arm = 0.14; _arm_r = 0.018; _tray_hz = 0.006
    _leg_h = 0.22; _torso_h = 0.24; _upper_arm = 0.12
    _torso_cz  = _BZ + _leg_h + _torso_h / 2
    _shoulder_z = _torso_cz + _torso_h / 2 - 0.03
    _elbow_z    = _shoulder_z - _upper_arm
    _tray_local_y = _torso_hy + _fore_arm / 2
    tray_x = _bx - _tray_local_y * _sf
    tray_y = _by + _tray_local_y * _cf
    tray_z = _elbow_z + _arm_r + _tray_hz

    # Drop from HIGH_Z (face height) — plenty of clearance above tray.
    drop_z = HIGH_Z

    steps = list(pick_steps)

    # Move above tray
    t_hover = _ik_dict(ctx["robot_id"], ctx["ee_link"],
                       [tray_x, tray_y, drop_z], ctx["joints"],
                       console=ctx["console"])
    if not t_hover:
        return None
    steps.append(_pose_step(t_hover, "Hover above tray"))

    # Open gripper to release
    steps.append(_grip_step(GRIP_OPEN))

    return steps


def _expand_present(cmd, ctx):
    """Pick object, lift, and hold at face height above the person's tray.

    Used for interaction requests (e.g. 'brush my teeth') where the object
    must be brought near the person WITHOUT being released on the tray.
    Reuses the same hardcoded tray XY as deliver, but targets HIGH_Z so
    the gripper clears the tray entirely.
    """
    pick_steps = _expand_pick({"cmd": "pick", "object": cmd.get("object")}, ctx)
    if not pick_steps:
        return None

    # ---- Hardcoded tray XY (same geometry as _expand_deliver) ----
    _ANG = math.radians(-100)
    _R   = 0.65
    _BZ  = 1.235
    _bx  = _R * math.cos(_ANG)
    _by  = _R * math.sin(_ANG)
    _facing = math.atan2(-_by, -_bx)
    _cf, _sf = math.cos(_facing), math.sin(_facing)
    _torso_hy = 0.035; _fore_arm = 0.14
    _tray_local_y = _torso_hy + _fore_arm / 2
    tray_x = _bx - _tray_local_y * _sf
    tray_y = _by + _tray_local_y * _cf

    # Target: tray XY but at HIGH_Z (face height, well above tray surface)
    t_present = _ik_dict(ctx["robot_id"], ctx["ee_link"],
                         [tray_x, tray_y, HIGH_Z], ctx["joints"],
                         console=ctx["console"])
    if not t_present:
        return None

    steps = list(pick_steps)
    steps.append(_pose_step(t_present, "Present to person (face height)"))
    return steps


def _expand_inspect(cmd, ctx):
    """Pick up -> bring close to EE camera (high, close to base)."""
    pick_steps = _expand_pick({"cmd": "pick", "object": cmd.get("object")}, ctx)
    if not pick_steps:
        return None
    t = _ik_dict(ctx["robot_id"], ctx["ee_link"], [0.20, 0.0, 1.65], ctx["joints"],
                 console=ctx["console"])
    if not t: return None
    return pick_steps + [_pose_step(t, "inspect-view")]


def _expand_sweep_table(cmd, ctx):
    """Sweep: lower arm (joint_2 = -1.87), then rotate joint_1 a full 360 deg.
    start_deg/end_deg control the arc (default full circle)."""
    start_deg = cmd.get("start_deg", 0)
    end_deg = cmd.get("end_deg", 360)
    arc = end_deg - start_deg
    # Number of waypoints: one every ~30 deg
    n_pts = max(4, int(abs(arc) / 30))

    # Read current joint positions
    cur = {}
    for j in ctx["joints"]:
        cur[j["name"]] = p.getJointState(ctx["robot_id"], j["index"])[0]

    # Step 1: close gripper
    steps = [_grip_step(GRIP_CLOSE)]

    # Step 2: lower arm -- set joint_2 to -1.87, keep others
    lower_targets = dict(cur)
    lower_targets["joint_2"] = -1.87
    steps.append(_pose_step(lower_targets, "sweep-lower(j2=-1.87)"))

    # Steps 3+: rotate joint_1 through the arc
    j1_start = cur.get("joint_1", 0.0) + math.radians(start_deg)
    for i in range(1, n_pts + 1):
        frac = i / n_pts
        j1_target = j1_start + math.radians(arc) * frac
        rot_targets = dict(lower_targets)
        rot_targets["joint_1"] = j1_target
        deg_now = start_deg + arc * frac
        steps.append(_pose_step(rot_targets, f"sweep-rotate({deg_now:.0f}\u00b0)"))

    return steps


def _expand_circle_sweep(cmd, ctx):
    """Full 360 deg sweep to clear the table."""
    return _expand_sweep_table({"cmd": "sweep_table", "start_deg": 0, "end_deg": 360},
                               ctx)


def _expand_wave(cmd, ctx):
    """Friendly wave gesture: move up, rock wrist back and forth."""
    steps = []
    wave_pos = [0.15, 0.0, 1.70]
    t = _ik_dict(ctx["robot_id"], ctx["ee_link"], wave_pos, ctx["joints"],
                 console=ctx["console"])
    if not t: return None
    steps.append(_pose_step(t, "wave-up"))
    # 3 wave motions via joint_6
    j6 = next((j for j in ctx["joints"] if j["name"] == "joint_6"), None)
    if j6 is not None:
        cur6 = p.getJointState(ctx["robot_id"], j6["index"])[0]
        for offset in [0.6, -0.6, 0.6, -0.6, 0.0]:
            wave_t = dict(t)
            wave_t["joint_6"] = cur6 + offset
            steps.append(_pose_step(wave_t, "wave-rock"))
    return steps


def _expand_point_at(cmd, ctx):
    """Point EE at an object (move toward it at lifted height)."""
    obj = cmd.get("object")
    pos = ctx["scene_objects"].get(obj)
    if not pos:
        ctx["console"].write(f"[PLAN] point_at: unknown object '{obj}'")
        return None
    x, y, z = pos["x"], pos["y"], pos["z"]
    # Point from a bit away and above
    d = math.sqrt(x*x + y*y) or 1.0
    px, py = x * 0.7 / d * d, y * 0.7 / d * d  # clamp to reach
    dist_to = math.sqrt(px*px + py*py)
    if dist_to > 0.65:
        px, py = px * 0.65 / dist_to, py * 0.65 / dist_to
    orn = _compute_side_grasp_orn(x, y)
    t = _ik_dict(ctx["robot_id"], ctx["ee_link"], [px, py, z + 0.10], ctx["joints"],
                 target_orn=orn, console=ctx["console"])
    if not t: return None
    return [_grip_step(GRIP_CLOSE), _pose_step(t, f"point_at({obj})")]


def _expand_guard(cmd, ctx):
    """Hover arm protectively above an object."""
    obj = cmd.get("object")
    pos = ctx["scene_objects"].get(obj)
    if not pos:
        ctx["console"].write(f"[PLAN] guard: unknown object '{obj}'")
        return None
    x, y = pos["x"], pos["y"]
    t = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x, y, LIFT_Z + 0.05], ctx["joints"],
                 console=ctx["console"])
    if not t: return None
    return [_pose_step(t, f"guard({obj})")]


def _expand_sort(cmd, ctx):
    """Pick object and place at angle_deg on the table ring."""
    obj = cmd.get("object")
    angle = math.radians(cmd.get("angle_deg", 0))
    r = 0.55  # mid table ring
    tx, ty = r * math.cos(angle), r * math.sin(angle)
    pick_steps = _expand_pick({"cmd": "pick", "object": obj}, ctx)
    if not pick_steps:
        return None
    place_steps = _expand_place({"cmd": "place", "x": tx, "y": ty, "z": TABLE_Z}, ctx)
    if not place_steps:
        return None
    return pick_steps + place_steps


def _expand_line_up(cmd, ctx):
    """Arrange listed objects in a horizontal line at y=<val>."""
    objects = cmd.get("objects", [])
    y_val = cmd.get("y", 0.0)
    spacing = 0.12
    start_x = -spacing * (len(objects) - 1) / 2.0 + 0.50  # offset from base
    all_steps = []
    for i, obj_name in enumerate(objects):
        tx = start_x + i * spacing
        pick_steps = _expand_pick({"cmd": "pick", "object": obj_name}, ctx)
        if not pick_steps:
            continue
        place_steps = _expand_place({"cmd": "place", "x": tx, "y": y_val, "z": TABLE_Z}, ctx)
        if not place_steps:
            continue
        all_steps.extend(pick_steps + place_steps)
    return all_steps if all_steps else None


def _expand_rotate(cmd, ctx):
    """Rotate the base (joint_1) by a given angle in degrees."""
    angle_deg = cmd.get("angle_deg", 90)
    cur = {}
    for j in ctx["joints"]:
        cur[j["name"]] = p.getJointState(ctx["robot_id"], j["index"])[0]
    target = dict(cur)
    target["joint_1"] = cur.get("joint_1", 0.0) + math.radians(angle_deg)
    return [_pose_step(target, f"rotate({angle_deg}\u00b0)")]


def _expand_straighten(cmd, ctx):
    """Reset joints 1-5 to 0 (upright with current wrist/gripper kept)."""
    cur = {}
    for j in ctx["joints"]:
        cur[j["name"]] = p.getJointState(ctx["robot_id"], j["index"])[0]
    target = dict(cur)
    for jn in ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]:
        if jn in target:
            target[jn] = 0.0
    return [_pose_step(target, "straighten(j1-j6\u21920)")]


# ---------------------------------------------------------------------------
# 23 NEW PRIMITIVES  (28-50)
# ---------------------------------------------------------------------------

def _expand_top_grasp(cmd, ctx):
    """Top-down grasp: approach from directly above, then close."""
    obj = cmd.get("object")
    pos = ctx["scene_objects"].get(obj)
    if not pos:
        ctx["console"].write(f"[PLAN] top_grasp: unknown object '{obj}'")
        return None
    x, y, z = pos["x"], pos["y"], pos["z"]
    # Orientation: gripper pointing straight down
    orn = [1.0, 0.0, 0.0, 0.0]  # 180° about X → Z-down
    steps = [_grip_step(GRIP_OPEN)]
    # above
    t1 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x, y, z + 0.15], ctx["joints"],
                  target_orn=orn, console=ctx["console"])
    if not t1: return None
    steps.append(_pose_step(t1, f"top-pre({x:.3f},{y:.3f},{z+0.15:.3f})"))
    # descend
    t2 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x, y, z], ctx["joints"],
                  target_orn=orn, console=ctx["console"])
    if not t2: return None
    steps.append(_pose_step(t2, f"top-grasp({x:.3f},{y:.3f},{z:.3f})"))
    steps.append(_grip_step(GRIP_CLOSE))
    # lift
    t3 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x, y, LIFT_Z], ctx["joints"],
                  target_orn=orn, console=ctx["console"])
    if not t3: return None
    steps.append(_pose_step(t3, f"top-lift({x:.3f},{y:.3f},{LIFT_Z:.3f})"))
    return steps


def _expand_regrasp(cmd, ctx):
    """Open gripper slightly, pause, then re-close for a better grip."""
    return [_grip_step(0.4), _grip_step(GRIP_CLOSE)]


def _expand_pick_and_place(cmd, ctx):
    """Combined pick then place in one primitive."""
    pick_steps = _expand_pick({"cmd": "pick", "object": cmd.get("object")}, ctx)
    if not pick_steps:
        return None
    x, y, z = cmd.get("x", 0), cmd.get("y", 0), cmd.get("z", TABLE_Z)
    place_steps = _expand_place({"cmd": "place", "x": x, "y": y, "z": z}, ctx)
    if not place_steps:
        return None
    return pick_steps + place_steps


def _expand_move_above(cmd, ctx):
    """Move EE directly above an object at LIFT_Z."""
    obj = cmd.get("object")
    pos = ctx["scene_objects"].get(obj)
    if not pos:
        ctx["console"].write(f"[PLAN] move_above: unknown object '{obj}'")
        return None
    x, y = pos["x"], pos["y"]
    t = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x, y, LIFT_Z], ctx["joints"],
                 console=ctx["console"])
    if not t: return None
    return [_pose_step(t, f"move_above({obj})")]


def _expand_approach_from_top(cmd, ctx):
    """Descend straight down from above to object height."""
    obj = cmd.get("object")
    pos = ctx["scene_objects"].get(obj)
    if not pos:
        ctx["console"].write(f"[PLAN] approach_from_top: unknown object '{obj}'")
        return None
    x, y, z = pos["x"], pos["y"], pos["z"]
    orn = [1.0, 0.0, 0.0, 0.0]
    steps = []
    # above
    t1 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x, y, z + 0.15], ctx["joints"],
                  target_orn=orn, console=ctx["console"])
    if not t1: return None
    steps.append(_pose_step(t1, f"top-above({x:.3f},{y:.3f},{z+0.15:.3f})"))
    # descend
    t2 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x, y, z + 0.02], ctx["joints"],
                  target_orn=orn, console=ctx["console"])
    if not t2: return None
    steps.append(_pose_step(t2, f"top-approach({x:.3f},{y:.3f},{z+0.02:.3f})"))
    return steps


def _expand_approach_from_side(cmd, ctx):
    """Approach object from the side at object height."""
    obj = cmd.get("object")
    pos = ctx["scene_objects"].get(obj)
    if not pos:
        ctx["console"].write(f"[PLAN] approach_from_side: unknown object '{obj}'")
        return None
    x, y, z = pos["x"], pos["y"], pos["z"]
    orn = _compute_side_grasp_orn(x, y)
    d = math.sqrt(x*x + y*y) or 1.0
    ax, ay = x/d, y/d
    # offset position
    ox, oy = x - APPROACH_BACK * ax, y - APPROACH_BACK * ay
    steps = []
    t1 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [ox, oy, z], ctx["joints"],
                  target_orn=orn, console=ctx["console"])
    if not t1: return None
    steps.append(_pose_step(t1, f"side-offset({ox:.3f},{oy:.3f},{z:.3f})"))
    t2 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x, y, z], ctx["joints"],
                  target_orn=orn, console=ctx["console"])
    if not t2: return None
    steps.append(_pose_step(t2, f"side-arrive({x:.3f},{y:.3f},{z:.3f})"))
    return steps


def _expand_retract(cmd, ctx):
    """Pull EE straight back toward the base by *distance* metres."""
    dist = cmd.get("distance", 0.12)
    ee = p.getLinkState(ctx["robot_id"], ctx["ee_link"], computeForwardKinematics=True)
    cx, cy, cz = ee[4][0], ee[4][1], ee[4][2]
    d = math.sqrt(cx*cx + cy*cy) or 1.0
    # move inward (toward base)
    nx, ny = -cx/d * dist, -cy/d * dist
    t = _ik_dict(ctx["robot_id"], ctx["ee_link"], [cx + nx, cy + ny, cz], ctx["joints"],
                 console=ctx["console"])
    if not t: return None
    return [_pose_step(t, f"retract({dist:.2f}m)")]


def _expand_move_relative(cmd, ctx):
    """Move EE by a relative offset (dx, dy, dz) from current position."""
    dx, dy, dz = cmd.get("dx", 0), cmd.get("dy", 0), cmd.get("dz", 0)
    ee = p.getLinkState(ctx["robot_id"], ctx["ee_link"], computeForwardKinematics=True)
    cx, cy, cz = ee[4][0], ee[4][1], ee[4][2]
    t = _ik_dict(ctx["robot_id"], ctx["ee_link"],
                 [cx + dx, cy + dy, cz + dz], ctx["joints"],
                 console=ctx["console"])
    if not t: return None
    return [_pose_step(t, f"move_rel(dx={dx:.3f},dy={dy:.3f},dz={dz:.3f})")]


def _expand_align_with(cmd, ctx):
    """Position EE above the object's XY at the given height."""
    obj = cmd.get("object")
    pos = ctx["scene_objects"].get(obj)
    if not pos:
        ctx["console"].write(f"[PLAN] align_with: unknown object '{obj}'")
        return None
    h = cmd.get("height", LIFT_Z)
    t = _ik_dict(ctx["robot_id"], ctx["ee_link"], [pos["x"], pos["y"], h], ctx["joints"],
                 console=ctx["console"])
    if not t: return None
    return [_pose_step(t, f"align_with({obj},h={h:.2f})")]


def _expand_hover(cmd, ctx):
    """Hover above an object with configurable clearance."""
    obj = cmd.get("object")
    pos = ctx["scene_objects"].get(obj)
    if not pos:
        ctx["console"].write(f"[PLAN] hover: unknown object '{obj}'")
        return None
    clearance = cmd.get("clearance", 0.10)
    x, y, z = pos["x"], pos["y"], pos["z"] + clearance
    t = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x, y, z], ctx["joints"],
                 console=ctx["console"])
    if not t: return None
    return [_pose_step(t, f"hover({obj},+{clearance:.2f})")]


def _expand_park(cmd, ctx):
    """Tuck arm into a compact safe position (all joints near zero,
    shoulder lowered, elbow bent)."""
    targets = {}
    for j in ctx["joints"]:
        targets[j["name"]] = 0.0
    targets["joint_2"] = -0.35   # shoulder slightly forward
    targets["joint_4"] = 0.70    # elbow tucked
    targets["joint_5"] = -1.57   # forearm down
    return [_grip_step(GRIP_CLOSE), _pose_step(targets, "park")]


def _expand_touch(cmd, ctx):
    """Gently descend to the top of an object and hold position."""
    obj = cmd.get("object")
    pos = ctx["scene_objects"].get(obj)
    if not pos:
        ctx["console"].write(f"[PLAN] touch: unknown object '{obj}'")
        return None
    x, y, z = pos["x"], pos["y"], pos["z"]
    orn = [1.0, 0.0, 0.0, 0.0]
    steps = []
    t1 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x, y, z + 0.12], ctx["joints"],
                  target_orn=orn, console=ctx["console"])
    if not t1: return None
    steps.append(_pose_step(t1, f"touch-above({x:.3f},{y:.3f},{z+0.12:.3f})"))
    t2 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x, y, z + 0.01], ctx["joints"],
                  target_orn=orn, console=ctx["console"])
    if not t2: return None
    steps.append(_pose_step(t2, f"touch({x:.3f},{y:.3f},{z+0.01:.3f})"))
    return steps


def _expand_slide(cmd, ctx):
    """Grip object and slide it along the table surface by (dx,dy)."""
    obj = cmd.get("object")
    pos = ctx["scene_objects"].get(obj)
    if not pos:
        ctx["console"].write(f"[PLAN] slide: unknown object '{obj}'")
        return None
    x, y, z = pos["x"], pos["y"], pos["z"]
    dx, dy = cmd.get("dx", 0), cmd.get("dy", 0)
    orn = _compute_side_grasp_orn(x, y)
    d = math.sqrt(x*x + y*y) or 1.0
    ax, ay = x/d, y/d

    steps = [_grip_step(GRIP_OPEN)]
    pre = [x - APPROACH_BACK*ax, y - APPROACH_BACK*ay, z]
    t1 = _ik_dict(ctx["robot_id"], ctx["ee_link"], pre, ctx["joints"],
                  target_orn=orn, console=ctx["console"])
    if not t1: return None
    steps.append(_pose_step(t1, "slide-pre"))
    t2 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x, y, z], ctx["joints"],
                  target_orn=orn, console=ctx["console"])
    if not t2: return None
    steps.append(_pose_step(t2, "slide-grip"))
    steps.append(_grip_step(GRIP_CLOSE))

    tx, ty = x + dx, y + dy
    orn2 = _compute_side_grasp_orn(tx, ty)
    t3 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [tx, ty, z], ctx["joints"],
                  target_orn=orn2, console=ctx["console"])
    if not t3: return None
    steps.append(_pose_step(t3, f"slide-to({tx:.3f},{ty:.3f})"))
    steps.append(_grip_step(GRIP_OPEN))
    t4 = _ik_dict(ctx["robot_id"], ctx["ee_link"], [tx, ty, LIFT_Z], ctx["joints"],
                  console=ctx["console"])
    if not t4: return None
    steps.append(_pose_step(t4, "slide-retract"))
    return steps


def _expand_tilt_object(cmd, ctx):
    """Tilt held object by angle_deg (wrist pitch rotation)."""
    angle = math.radians(cmd.get("angle_deg", 45))
    j6 = next((j for j in ctx["joints"] if j["name"] == "joint_6"), None)
    if j6 is None:
        return None
    cur = {}
    for j in ctx["joints"]:
        cur[j["name"]] = p.getJointState(ctx["robot_id"], j["index"])[0]
    targets = dict(cur)
    targets["joint_6"] = cur.get("joint_6", 0.0) + angle
    return [_pose_step(targets, f"tilt({math.degrees(angle):.0f}\u00b0)")]


def _expand_shake(cmd, ctx):
    """Shake held object: small rapid wrist oscillations."""
    j6 = next((j for j in ctx["joints"] if j["name"] == "joint_6"), None)
    if j6 is None:
        return None
    cur = {}
    for j in ctx["joints"]:
        cur[j["name"]] = p.getJointState(ctx["robot_id"], j["index"])[0]
    base6 = cur.get("joint_6", 0.0)
    steps = []
    for offset in [0.4, -0.4, 0.4, -0.4, 0.3, -0.3, 0.0]:
        t = dict(cur)
        t["joint_6"] = base6 + offset
        steps.append(_pose_step(t, "shake"))
    return steps


def _expand_pour(cmd, ctx):
    """Tilt held object ~90° about wrist to pour contents."""
    return _expand_tilt_object({"angle_deg": 90}, ctx)


def _expand_swap(cmd, ctx):
    """Swap positions of two objects: pick A, place at temp, pick B,
    place at A's original, pick A from temp, place at B's original."""
    a, b = cmd.get("object_a"), cmd.get("object_b")
    pos_a = ctx["scene_objects"].get(a)
    pos_b = ctx["scene_objects"].get(b)
    if not pos_a or not pos_b:
        ctx["console"].write(f"[PLAN] swap: unknown object(s)")
        return None
    ax, ay, az = pos_a["x"], pos_a["y"], pos_a["z"]
    bx, by, bz = pos_b["x"], pos_b["y"], pos_b["z"]
    # temp spot: midpoint, offset outward
    tx, ty = (ax + bx) / 2.0, (ay + by) / 2.0 + 0.15

    all_steps = []
    # A → temp
    s = _expand_pick({"cmd": "pick", "object": a}, ctx)
    if not s: return None
    all_steps.extend(s)
    s = _expand_place({"cmd": "place", "x": tx, "y": ty, "z": TABLE_Z}, ctx)
    if not s: return None
    all_steps.extend(s)
    # B → A's spot
    s = _expand_pick({"cmd": "pick", "object": b}, ctx)
    if not s: return None
    all_steps.extend(s)
    s = _expand_place({"cmd": "place", "x": ax, "y": ay, "z": az}, ctx)
    if not s: return None
    all_steps.extend(s)
    # temp (A) → B's spot
    # Update scene_objects to reflect temp position for A
    old_a = dict(ctx["scene_objects"].get(a, {}))
    ctx["scene_objects"][a] = {"x": tx, "y": ty, "z": TABLE_Z}
    s = _expand_pick({"cmd": "pick", "object": a}, ctx)
    if not s:
        ctx["scene_objects"][a] = old_a
        return None
    all_steps.extend(s)
    s = _expand_place({"cmd": "place", "x": bx, "y": by, "z": bz}, ctx)
    if not s:
        ctx["scene_objects"][a] = old_a
        return None
    all_steps.extend(s)
    ctx["scene_objects"][a] = old_a
    return all_steps


def _expand_group(cmd, ctx):
    """Gather listed objects to a single XY location."""
    objects = cmd.get("objects", [])
    gx, gy = cmd.get("x", 0.5), cmd.get("y", 0.0)
    all_steps = []
    for obj in objects:
        s = _expand_pick({"cmd": "pick", "object": obj}, ctx)
        if not s: continue
        all_steps.extend(s)
        s = _expand_place({"cmd": "place", "x": gx, "y": gy, "z": TABLE_Z}, ctx)
        if not s: continue
        all_steps.extend(s)
        # offset slightly so they don't stack exactly on top
        gx += 0.06
    return all_steps if all_steps else None


def _expand_scatter(cmd, ctx):
    """Spread objects apart from center in a radial pattern."""
    objects = cmd.get("objects", [])
    if not objects:
        return None
    r = 0.50
    all_steps = []
    for i, obj in enumerate(objects):
        angle = 2 * math.pi * i / len(objects)
        tx, ty = r * math.cos(angle), r * math.sin(angle)
        s = _expand_pick({"cmd": "pick", "object": obj}, ctx)
        if not s: continue
        all_steps.extend(s)
        s = _expand_place({"cmd": "place", "x": tx, "y": ty, "z": TABLE_Z}, ctx)
        if not s: continue
        all_steps.extend(s)
    return all_steps if all_steps else None


def _expand_clear_area(cmd, ctx):
    """Push every object within radius of (x,y) away from that center."""
    cx, cy = cmd.get("x", 0), cmd.get("y", 0)
    radius = cmd.get("radius", 0.20)
    steps = []
    for name, pos in ctx["scene_objects"].items():
        dx = pos["x"] - cx
        dy = pos["y"] - cy
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < radius and dist > 0.01:
            push_dx = dx / dist * 0.20  # push 20 cm outward
            push_dy = dy / dist * 0.20
            s = _expand_push({"cmd": "push", "object": name,
                              "dx": push_dx, "dy": push_dy}, ctx)
            if s:
                steps.extend(s)
    return steps if steps else None


def _expand_orbit(cmd, ctx):
    """Move EE in a circle around an object at constant radius and height."""
    obj = cmd.get("object")
    pos = ctx["scene_objects"].get(obj)
    if not pos:
        ctx["console"].write(f"[PLAN] orbit: unknown object '{obj}'")
        return None
    cx, cy = pos["x"], pos["y"]
    r = cmd.get("radius", 0.15)
    h = cmd.get("height", LIFT_Z)
    n_pts = 12
    steps = []
    for i in range(n_pts + 1):
        angle = 2 * math.pi * i / n_pts
        px = cx + r * math.cos(angle)
        py = cy + r * math.sin(angle)
        t = _ik_dict(ctx["robot_id"], ctx["ee_link"], [px, py, h], ctx["joints"],
                     console=ctx["console"])
        if not t: continue
        steps.append(_pose_step(t, f"orbit({i}/{n_pts})"))
    return steps if steps else None


def _expand_zigzag(cmd, ctx):
    """Move EE in a zigzag pattern between two XY points at height z."""
    x1, y1 = cmd.get("x1", 0), cmd.get("y1", 0)
    x2, y2 = cmd.get("x2", 0.5), cmd.get("y2", 0)
    z = cmd.get("z", LIFT_Z)
    n_zags = 6
    amplitude = 0.08  # lateral offset
    dx, dy = x2 - x1, y2 - y1
    length = math.sqrt(dx*dx + dy*dy) or 1.0
    # perpendicular direction
    px, py = -dy / length, dx / length
    steps = []
    for i in range(n_zags + 1):
        frac = i / n_zags
        mx = x1 + dx * frac
        my = y1 + dy * frac
        side = amplitude * (1 if i % 2 == 0 else -1)
        wx = mx + px * side
        wy = my + py * side
        t = _ik_dict(ctx["robot_id"], ctx["ee_link"], [wx, wy, z], ctx["joints"],
                     console=ctx["console"])
        if not t: continue
        steps.append(_pose_step(t, f"zigzag({i}/{n_zags})"))
    return steps if steps else None


def _expand_patrol(cmd, ctx):
    """Visit a sequence of XYZ waypoints in order."""
    waypoints = cmd.get("waypoints", [])
    steps = []
    for i, wp in enumerate(waypoints):
        x = wp.get("x", 0)
        y = wp.get("y", 0)
        z = wp.get("z", LIFT_Z)
        t = _ik_dict(ctx["robot_id"], ctx["ee_link"], [x, y, z], ctx["joints"],
                     console=ctx["console"])
        if not t: continue
        steps.append(_pose_step(t, f"patrol({i+1}/{len(waypoints)})"))
    return steps if steps else None


# ---------------------------------------------------------------------------
# New Primitive: point_gripper_at
# ---------------------------------------------------------------------------
def _expand_point_gripper_at(cmd, ctx):
    """Rotate and bend the gripper to point at the object from above, so it appears in the EE camera."""
    obj = cmd.get("object")
    pos = ctx["scene_objects"].get(obj)
    if not pos:
        ctx["console"].write(f"[PLAN] point_gripper_at: unknown object '{obj}'")
        return None
    x, y, z = pos["x"], pos["y"], pos["z"]
    # Approach from above, but tilt the gripper down to look at the object
    # We'll use a fixed offset above the object, and orient the gripper to point down
    approach_height = 0.25  # meters above object
    ee_pos = [x, y, z + approach_height]
    # Compute orientation: gripper Z axis points toward the object
    # We'll look straight down (pitch -90 deg), but yaw to face the object from the robot base
    dx, dy = x, y
    # Approach offset so the gripper can tilt down and "look" at the object
    dist_xy = math.sqrt(dx * dx + dy * dy) or 1.0
    ax, ay = dx / dist_xy, dy / dist_xy
    offset = 0.12
    approach_height = 0.25
    ee_pos = [x - offset * ax, y - offset * ay, z + approach_height]
    # Direction from EE approach pose to object
    vx = x - ee_pos[0]
    vy = y - ee_pos[1]
    vz = z - ee_pos[2]

    # Build a robust "look-at" orientation so the EE forward axis points at the object.
    # forward := normalized vector from EE pose to object
    f = np.array([vx, vy, vz], dtype=np.float64)
    fnorm = np.linalg.norm(f)
    if fnorm < 1e-6:
        f = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        fnorm = 1.0
    f = f / fnorm

    # world-up (Z)
    up_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    # if forward is nearly parallel to world-up, pick a different up vector
    if abs(np.dot(f, up_w)) > 0.999:
        up_w = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # right = normalize(cross(up, forward))
    r = np.cross(up_w, f)
    rnorm = np.linalg.norm(r)
    if rnorm < 1e-6:
        # degenerate: choose arbitrary right
        r = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        rnorm = 1.0
    r = r / rnorm

    # true up = cross(forward, right)
    u = np.cross(f, r)

    # Construct rotation matrix with columns [right, up, forward]
    R = np.column_stack((r, u, f))
    orn = _rotmat_to_quat(R)
    t = _ik_dict(ctx["robot_id"], ctx["ee_link"], ee_pos, ctx["joints"], target_orn=orn, console=ctx["console"])
    if not t:
        ctx["console"].write(f"[PLAN] point_gripper_at: IK failed for '{obj}'")
        return None
    return [_pose_step(t, f"point_gripper_at({x:.3f},{y:.3f},{z:.3f})")]

# Register the new primitive


# ---------------------------------------------------------------------------
# Dispatch table: cmd name -> expander function
# ---------------------------------------------------------------------------
PRIMITIVE_DISPATCH = {
    # Grasping & Releasing
    "pick":               _expand_pick,
    "place":              _expand_place,
    "open_gripper":       _expand_open_gripper,
    "close_gripper":      _expand_close_gripper,
    "drop":               _expand_drop,
    "top_grasp":          _expand_top_grasp,
    "regrasp":            _expand_regrasp,
    "pick_and_place":     _expand_pick_and_place,
    # Movement & Positioning
    "move_to":            _expand_move_to,
    "point_gripper_at":  _expand_point_gripper_at,
    "home":               _expand_home,
    "lift_high":          _expand_lift_high,
    "move_above":         _expand_move_above,
    "approach_from_top":  _expand_approach_from_top,
    "approach_from_side": _expand_approach_from_side,
    "retract":            _expand_retract,
    "move_relative":      _expand_move_relative,
    "align_with":         _expand_align_with,
    "hover":              _expand_hover,
    "park":               _expand_park,
    # Pushing & Contact
    "push":               _expand_push,
    "nudge":              _expand_nudge,
    "drag":               _expand_drag,
    "tap":                _expand_tap,
    "knock_over":         _expand_knock_over,
    "touch":              _expand_touch,
    "slide":              _expand_slide,
    # Wrist / In-Hand
    "rotate_object":      _expand_rotate_object,
    "flip_object":        _expand_flip_object,
    "tilt_object":        _expand_tilt_object,
    "shake":              _expand_shake,
    "pour":               _expand_pour,
    # Compound Tasks
    "stack":              _expand_stack,
    "sort":               _expand_sort,
    "line_up":            _expand_line_up,
    "handover":           _expand_handover,
    "deliver":            _expand_deliver,
    "present":            _expand_present,
    "inspect":            _expand_inspect,
    "swap":               _expand_swap,
    "group":              _expand_group,
    "scatter":            _expand_scatter,
    "clear_area":         _expand_clear_area,
    # Sweeping
    "sweep_table":        _expand_sweep_table,
    "circle_sweep":       _expand_circle_sweep,
    # Gestures
    "wave":               _expand_wave,
    "point_at":           _expand_point_at,
    "guard":              _expand_guard,
    # Navigation & Paths
    "orbit":              _expand_orbit,
    "zigzag":             _expand_zigzag,
    "patrol":             _expand_patrol,
    # Basic Joint Control
    "rotate":             _expand_rotate,
    "straighten":         _expand_straighten,
}


def expand_plan(plan_list: list, robot_id, ee_link_index, movable_joints,
                scene_objects: dict, console, loaded_objects: dict | None = None) -> list:
    """Expand a list of high-level primitives into flat sub-steps.
    Returns a list of pose/gripper dicts ready for the execution engine."""
    ctx = {
        "robot_id": robot_id,
        "ee_link": ee_link_index,
        "joints": movable_joints,
        "scene_objects": scene_objects or {},
        "console": console,
        "loaded_objects": loaded_objects or {},
    }
    all_steps = []
    for i, cmd in enumerate(plan_list):
        cmd_name = cmd.get("cmd", "")
        expander = PRIMITIVE_DISPATCH.get(cmd_name)
        if not expander:
            continue
        sub_steps = expander(cmd, ctx)
        if sub_steps:
            all_steps.extend(sub_steps)
    return all_steps
