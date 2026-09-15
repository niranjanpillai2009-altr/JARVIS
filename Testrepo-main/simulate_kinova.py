"""
PyBullet Simulation for Kinova Gen 3 7-DOF Robotic Arm
Loads the robot from the provided URDF and mesh files,
sets up a physics environment, and runs an interactive simulation.
"""

import pybullet as p
import pybullet_data
import time
import os
import sys
import math
import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import re
import difflib
from collections import deque
import struct
import zlib
import base64

# ---------------------------------------------------------------------------
# Module imports (split from this file for maintainability)
# ---------------------------------------------------------------------------
from kinova_constants import (
    API_KEY, API_URL, API_MODEL,
    JOINT_DESCRIPTIONS, REAL_JOINT_LIMITS,
    EXAMPLE_MOVEMENTS, MOTION_PRIMITIVES,
    OBJECT_DESCRIPTIONS,
    get_primitives_description,
)
from jarvis_prompts import (
    step_summary,
    format_step,
    identify_objects_llm,
    call_jarvis3_dispatch,
    requery_jarvis3_dispatch,
    call_jarvisplan,
)
from jarvis_primitives import (
    PRIMITIVE_DISPATCH,
    GRIP_OPEN,
    GRIP_CLOSE,
    LIFT_Z,
    TABLE_Z,
)
from clip_bbox import clip_bbox as compute_clip_bboxes
import gqcnn_wrapper
import grconvnet_wrapper

# ---------------------------------------------------------------------------
# Mesh download helper – ensures STL files are present before loading URDF
# ---------------------------------------------------------------------------
def ensure_meshes_exist():
    """Check for kinova_meshes folder; download if missing."""
    mesh_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kinova_meshes")
    expected_files = [
        "base_link.STL",
        "shoulder_link.STL",
        "half_arm_1_link.STL",
        "half_arm_2_link.STL",
        "forearm_link.STL",
        "spherical_wrist_1_link.STL",
        "spherical_wrist_2_link.STL",
        "bracelet_with_vision_link.STL",
    ]

    missing = [f for f in expected_files if not os.path.isfile(os.path.join(mesh_dir, f))]

    if missing:
        print(f"[INFO] Missing {len(missing)} mesh file(s). Running download script...")
        # Import the download helper that lives next to this file
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from download_kinova_meshes import download_kinova_meshes
        download_kinova_meshes()

        # Verify again
        still_missing = [f for f in expected_files if not os.path.isfile(os.path.join(mesh_dir, f))]
        if still_missing:
            print(f"[ERROR] Still missing meshes after download: {still_missing}")
            sys.exit(1)
    else:
        print("[INFO] All mesh files found.")


# ---------------------------------------------------------------------------
# Gripper loader — attach Robotiq 2F-85 to the end-effector
# ---------------------------------------------------------------------------
def load_gripper(robot_id, ee_link_index):
    """Load the Robotiq 2F-85 gripper URDF and attach it to the end-effector."""
    gripper_urdf = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "robotiq_2f85_gripper.urdf"
    )
    # Get end-effector world pose for initial placement
    ee_state = p.getLinkState(robot_id, ee_link_index, computeForwardKinematics=True)
    ee_pos = ee_state[4]
    ee_orn = ee_state[5]

    gripper_id = p.loadURDF(
        gripper_urdf,
        basePosition=ee_pos,
        baseOrientation=ee_orn,
        useFixedBase=False,
    )

    # Create a fixed constraint: EE link -> gripper base
    cid = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=ee_link_index,
        childBodyUniqueId=gripper_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=[0, 0, 0, 1],
        childFrameOrientation=[0, 0, 0, 1],
    )
    p.changeConstraint(cid, maxForce=500)

    # Discover movable gripper joints
    left_idx = right_idx = None
    for i in range(p.getNumJoints(gripper_id)):
        info = p.getJointInfo(gripper_id, i)
        name = info[1].decode("utf-8")
        if name == "gripper_left_finger_joint":
            left_idx = i
        elif name == "gripper_right_finger_joint":
            right_idx = i

    # Disable collisions between every gripper link and every robot link.
    # Without this, wrist rotations cause the gripper body to jam against
    # the forearm/upper-arm links (self-collision blocks the motion).
    num_gripper_joints = p.getNumJoints(gripper_id)
    num_robot_joints = p.getNumJoints(robot_id)
    gripper_links = list(range(-1, num_gripper_joints))  # -1 = base link
    robot_links = list(range(-1, num_robot_joints))
    for gl in gripper_links:
        for rl in robot_links:
            p.setCollisionFilterPair(gripper_id, robot_id, gl, rl, enableCollision=0)

    # Increase finger friction so gripped objects don't slip out
    for link_idx in range(num_gripper_joints):
        p.changeDynamics(gripper_id, link_idx, lateralFriction=5.0,
                         spinningFriction=2.0, rollingFriction=1.0,
                         contactStiffness=30000, contactDamping=1000)

    return {
        "id": gripper_id,
        "left_idx": left_idx,
        "right_idx": right_idx,
        "constraint": cid,
    }


# ---------------------------------------------------------------------------
# Random colour generator for spawned objects
# ---------------------------------------------------------------------------
import random as _random

_RANDOM_RGBAS = [
    [0.85, 0.15, 0.15, 1],
    [0.15, 0.55, 0.15, 1],
    [0.20, 0.40, 0.90, 1],
    [0.95, 0.82, 0.15, 1],
    [0.70, 0.30, 0.80, 1],
    [0.95, 0.55, 0.10, 1],
    [0.15, 0.80, 0.80, 1],
    [0.90, 0.45, 0.65, 1],
    [0.60, 0.30, 0.10, 1],
    [0.85, 0.85, 0.85, 1],
]


def _random_color():
    """Return a random RGBA colour list."""
    return list(_random.choice(_RANDOM_RGBAS))


# ---------------------------------------------------------------------------
# Spawnable scene objects  (colour is randomised per spawn)
# ---------------------------------------------------------------------------
def create_block(position=None, color=None):
    """Spawn a small block (5 cm cube)."""
    pos = position or [0.4, 0.0, 0.025]
    rgba = color or _random_color()
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025],
                              rgbaColor=rgba)
    body = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis, basePosition=pos)
    return body


def create_cylinder(position=None, color=None):
    """Spawn a cylinder (r=2 cm, h=8 cm)."""
    pos = position or [0.35, 0.2, 0.04]
    rgba = color or _random_color()
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.02, height=0.08)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.02, length=0.08,
                              rgbaColor=rgba)
    body = p.createMultiBody(baseMass=0.05, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis, basePosition=pos)
    return body


def create_remote(position=None, color=None):
    """Spawn a simple TV remote (flat box with button bumps)."""
    pos = position or [0.4, -0.2, 0.01]
    rgba = color or [0.08, 0.08, 0.08, 1.0]  # black
    col_body = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.075, 0.025, 0.015])
    vis_body = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.075, 0.025, 0.015],
                                   rgbaColor=rgba)
    remote_id = p.createMultiBody(baseMass=0.12, baseCollisionShapeIndex=col_body,
                                  baseVisualShapeIndex=vis_body, basePosition=pos)
    # High friction so gripper doesn't drop it
    p.changeDynamics(remote_id, -1, lateralFriction=5.0, spinningFriction=2.0,
                     rollingFriction=1.0, contactStiffness=30000, contactDamping=1000)

    # Button bumps — slightly lighter shade, fixed to remote body
    btn = [min(1.0, c + 0.3) for c in rgba[:3]] + [1]
    for bx in [-0.04, -0.015, 0.015, 0.04]:
        bvis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.008, 0.008, 0.003],
                                   rgbaColor=btn)
        bid = p.createMultiBody(baseMass=0.001, baseVisualShapeIndex=bvis,
                                basePosition=[pos[0] + bx, pos[1], pos[2] + 0.013])
        cid = p.createConstraint(remote_id, -1, bid, -1, p.JOINT_FIXED,
                                 [0, 0, 0], [bx, 0, 0.013], [0, 0, 0])
        p.changeConstraint(cid, maxForce=10000)
    # Power button (red dot)
    pwr_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.005,
                                  rgbaColor=[0.9, 0.1, 0.1, 1])
    pwr_id = p.createMultiBody(baseMass=0.001, baseVisualShapeIndex=pwr_vis,
                               basePosition=[pos[0] + 0.06, pos[1], pos[2] + 0.013])
    cid = p.createConstraint(remote_id, -1, pwr_id, -1, p.JOINT_FIXED,
                             [0, 0, 0], [0.06, 0, 0.013], [0, 0, 0])
    p.changeConstraint(cid, maxForce=10000)
    return remote_id


def create_mug(position=None, color=None):
    """Spawn a mug (cup body + handle)."""
    pos = position or [0.35, -0.15, 0.035]
    rgba = color or _random_color()
    col_cup = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.03, height=0.07)
    vis_cup = p.createVisualShape(p.GEOM_CYLINDER, radius=0.03, length=0.07,
                                  rgbaColor=rgba)
    mug_id = p.createMultiBody(baseMass=0.10, baseCollisionShapeIndex=col_cup,
                               baseVisualShapeIndex=vis_cup, basePosition=pos)
    # Handle — same colour, fixed to mug body
    h_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.005, 0.012, 0.018],
                                rgbaColor=rgba)
    h_id = p.createMultiBody(baseMass=0.001, baseVisualShapeIndex=h_vis,
                             basePosition=[pos[0] + 0.035, pos[1], pos[2] + 0.005])
    cid = p.createConstraint(mug_id, -1, h_id, -1, p.JOINT_FIXED,
                             [0, 0, 0], [0.035, 0, 0.005], [0, 0, 0])
    p.changeConstraint(cid, maxForce=10000)
    return mug_id


def create_box(position=None, color=None):
    """Spawn an open-top box (tray) large enough to hold all 3 objects.
    Interior: 25 cm × 25 cm, wall height 5 cm, wall thickness 1 cm.
    The returned body id is the floor piece; walls are static fixtures."""
    pos = position or [0.5, -0.5, 0.005]
    rgba = color or _random_color()
    # Dimensions
    inner = 0.125           # half interior width  (25 cm / 2)
    wall_t = 0.005          # wall half-thickness  (1 cm / 2)
    wall_h = 0.025          # wall half-height     (5 cm / 2)
    floor_h = 0.0025        # floor half-thickness (0.5 cm / 2)

    # Floor
    floor_col = p.createCollisionShape(p.GEOM_BOX,
                                       halfExtents=[inner + wall_t, inner + wall_t, floor_h])
    floor_vis = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=[inner + wall_t, inner + wall_t, floor_h],
                                    rgbaColor=rgba)
    box_id = p.createMultiBody(baseMass=0.5,
                               baseCollisionShapeIndex=floor_col,
                               baseVisualShapeIndex=floor_vis,
                               basePosition=pos)

    # 4 walls (static, mass=0) — positioned relative to floor center
    fx, fy, fz = pos
    wall_specs = [
        # (halfExtents, centre offset)
        ([inner + wall_t, wall_t, wall_h], [0,  inner + wall_t, wall_h + floor_h]),    # +Y wall
        ([inner + wall_t, wall_t, wall_h], [0, -inner - wall_t, wall_h + floor_h]),    # -Y wall
        ([wall_t, inner + wall_t, wall_h], [ inner + wall_t, 0, wall_h + floor_h]),    # +X wall
        ([wall_t, inner + wall_t, wall_h], [-inner - wall_t, 0, wall_h + floor_h]),    # -X wall
    ]
    for hext, off in wall_specs:
        wc = p.createCollisionShape(p.GEOM_BOX, halfExtents=hext)
        wv = p.createVisualShape(p.GEOM_BOX, halfExtents=hext, rgbaColor=rgba)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wc,
                          baseVisualShapeIndex=wv,
                          basePosition=[fx + off[0], fy + off[1], fz + off[2]])
    return box_id


def create_bottle(position=None, color=None):
    """Spawn a bottle (body + neck) on the table."""
    pos = position or [0.35, 0.0, 0.05]
    rgba = color or _random_color()
    # Body: r=2.5 cm, h=8 cm
    body_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.025, height=0.08)
    body_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.025, length=0.08,
                                   rgbaColor=rgba)
    bottle_id = p.createMultiBody(baseMass=0.08,
                                  baseCollisionShapeIndex=body_col,
                                  baseVisualShapeIndex=body_vis,
                                  basePosition=pos)
    # Neck — same colour, fixed to bottle body
    neck_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.01, length=0.04,
                                   rgbaColor=rgba)
    neck_id = p.createMultiBody(baseMass=0.001,
                                baseVisualShapeIndex=neck_vis,
                                basePosition=[pos[0], pos[1], pos[2] + 0.06])
    cid = p.createConstraint(bottle_id, -1, neck_id, -1, p.JOINT_FIXED,
                             [0, 0, 0], [0, 0, 0.06], [0, 0, 0])
    p.changeConstraint(cid, maxForce=10000)
    return bottle_id


def create_pyramid(position=None, color=None):
    """Spawn a 4-sided pyramid (5 cm base, 6 cm tall)."""
    pos = position or [0.4, 0.2, 0.0]
    rgba = color or _random_color()
    obj_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyramid.obj")
    col = p.createCollisionShape(p.GEOM_MESH, fileName=obj_file, meshScale=[1, 1, 1])
    vis = p.createVisualShape(p.GEOM_MESH, fileName=obj_file,
                              meshScale=[1, 1, 1], rgbaColor=rgba)
    body = p.createMultiBody(baseMass=0.08,
                             baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=pos)
    return body


def create_pill_bottle(position=None, color=None):
    """Spawn an orange prescription pill bottle (r=1.8 cm, h=7 cm) with white cap."""
    pos = position or [0.4, 0.2, 0.0]
    body_rgba = color or [0.93, 0.55, 0.15, 1.0]  # orange
    cap_rgba = [0.95, 0.95, 0.95, 1.0]             # white cap
    r, h = 0.018, 0.07
    cap_h = 0.012
    col_body = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
    vis_body = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=body_rgba)
    body = p.createMultiBody(baseMass=0.06,
                             baseCollisionShapeIndex=col_body,
                             baseVisualShapeIndex=vis_body,
                             basePosition=pos)
    col_cap = p.createCollisionShape(p.GEOM_CYLINDER, radius=r + 0.002, height=cap_h)
    vis_cap = p.createVisualShape(p.GEOM_CYLINDER, radius=r + 0.002, length=cap_h, rgbaColor=cap_rgba)
    cap_id = p.createMultiBody(baseMass=0.001,
                               baseCollisionShapeIndex=col_cap,
                               baseVisualShapeIndex=vis_cap,
                               basePosition=[pos[0], pos[1], pos[2] + h / 2 + cap_h / 2])
    cid = p.createConstraint(body, -1, cap_id, -1, p.JOINT_FIXED,
                             [0, 0, 0], [0, 0, h / 2 + cap_h / 2], [0, 0, 0])
    p.changeConstraint(cid, maxForce=10000)
    p.changeDynamics(body, -1, lateralFriction=5.0, spinningFriction=2.0,
                     rollingFriction=1.0, contactStiffness=30000, contactDamping=1000)
    return body


def create_phone(position=None, color=None):
    """Spawn a smartphone (14 x 7 x 0.8 cm)."""
    pos = position or [0.4, 0.2, 0.0]
    body_rgba = color or [0.15, 0.15, 0.18, 1.0]  # dark grey
    screen_rgba = [0.1, 0.1, 0.12, 1.0]
    lx, ly, lz = 0.14, 0.07, 0.008
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[lx/2, ly/2, lz/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[lx/2, ly/2, lz/2], rgbaColor=body_rgba)
    body = p.createMultiBody(baseMass=0.18,
                             baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=pos)
    # screen bezel, fixed to phone body
    vis_scr = p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=[lx/2 - 0.004, ly/2 - 0.004, lz/2 + 0.0002],
                                  rgbaColor=screen_rgba)
    scr_id = p.createMultiBody(baseMass=0.001,
                               baseCollisionShapeIndex=-1,
                               baseVisualShapeIndex=vis_scr,
                               basePosition=[pos[0], pos[1], pos[2] + 0.0003])
    cid = p.createConstraint(body, -1, scr_id, -1, p.JOINT_FIXED,
                             [0, 0, 0], [0, 0, 0.0003], [0, 0, 0])
    p.changeConstraint(cid, maxForce=10000)
    p.changeDynamics(body, -1, lateralFriction=5.0, spinningFriction=2.0,
                     rollingFriction=1.0, contactStiffness=30000, contactDamping=1000)
    return body


def create_book(position=None, color=None):
    """Spawn a hardcover book (14 x 9 x 3.5 cm) — compact enough for gripper."""
    pos = position or [0.4, 0.2, 0.0]
    body_rgba = color or [0.15, 0.30, 0.75, 1.0]  # blue
    lx, ly, lz = 0.14, 0.09, 0.035
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[lx/2, ly/2, lz/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[lx/2, ly/2, lz/2], rgbaColor=body_rgba)
    body = p.createMultiBody(baseMass=0.30,
                             baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=pos)
    # spine accent, fixed to book body
    vis_spine = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=[lx/2 + 0.0002, 0.003, lz/2 + 0.0003],
                                    rgbaColor=[0.85, 0.75, 0.45, 1.0])
    spine_id = p.createMultiBody(baseMass=0.001,
                                 baseCollisionShapeIndex=-1,
                                 baseVisualShapeIndex=vis_spine,
                                 basePosition=[pos[0], pos[1] - ly/2 + 0.003, pos[2]])
    cid = p.createConstraint(body, -1, spine_id, -1, p.JOINT_FIXED,
                             [0, 0, 0], [0, -ly/2 + 0.003, 0], [0, 0, 0])
    p.changeConstraint(cid, maxForce=10000)
    # title patch, fixed to book body
    vis_title = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=[0.03, 0.020, lz/2 + 0.0002],
                                    rgbaColor=[0.85, 0.75, 0.45, 1.0])
    title_id = p.createMultiBody(baseMass=0.001,
                                 baseCollisionShapeIndex=-1,
                                 baseVisualShapeIndex=vis_title,
                                 basePosition=[pos[0], pos[1] + 0.01, pos[2] + 0.0002])
    cid = p.createConstraint(body, -1, title_id, -1, p.JOINT_FIXED,
                             [0, 0, 0], [0, 0.01, 0.0002], [0, 0, 0])
    p.changeConstraint(cid, maxForce=10000)
    p.changeDynamics(body, -1, lateralFriction=5.0, spinningFriction=2.0,
                     rollingFriction=1.0, contactStiffness=30000, contactDamping=1000)
    return body


def create_cup(position=None, color=None):
    """Spawn a drinking cup (r=3.5 cm, h=10 cm)."""
    pos = position or [0.4, 0.2, 0.0]
    rgba = color or [0.75, 0.85, 0.95, 0.85]  # translucent light blue
    r, h = 0.035, 0.10
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=rgba)
    body = p.createMultiBody(baseMass=0.08,
                             baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=pos)
    return body


def create_water_cup(position=None, color=None):
    """Spawn a large crimson-red mug filled with water, with a visible handle.

    Big and distinct so DINO and the camera can't confuse it with
    any other object on the table.
    """
    pos = position or [0.4, 0.2, 0.0]
    cup_rgba = color or [0.80, 0.10, 0.10, 1.0]   # crimson red
    water_rgba = [0.30, 0.55, 0.90, 0.70]          # translucent blue water
    r, h = 0.045, 0.13                              # big mug (9 cm ø, 13 cm tall)
    water_h = 0.08  # water fills ~60 % of the mug

    # Mug body
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=cup_rgba)
    body = p.createMultiBody(baseMass=0.22,
                             baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=pos)
    p.changeDynamics(body, -1, lateralFriction=5.0, spinningFriction=2.0,
                     rollingFriction=1.0, contactStiffness=30000, contactDamping=1000)

    # Water fill (visual only — slightly smaller radius, sits inside mug)
    vis_water = p.createVisualShape(p.GEOM_CYLINDER, radius=r - 0.004,
                                    length=water_h, rgbaColor=water_rgba)
    water_id = p.createMultiBody(baseMass=0.001,
                                 baseVisualShapeIndex=vis_water,
                                 basePosition=[pos[0], pos[1],
                                               pos[2] - (h - water_h) / 2 + 0.002])
    cid = p.createConstraint(body, -1, water_id, -1, p.JOINT_FIXED,
                             [0, 0, 0], [0, 0, -(h - water_h) / 2 + 0.002], [0, 0, 0])
    p.changeConstraint(cid, maxForce=10000)

    # Handle (small box attached to the side of the mug)
    handle_w, handle_h, handle_d = 0.012, 0.05, 0.025
    handle_col = p.createCollisionShape(p.GEOM_BOX,
                                        halfExtents=[handle_w / 2, handle_d / 2, handle_h / 2])
    handle_vis = p.createVisualShape(p.GEOM_BOX,
                                     halfExtents=[handle_w / 2, handle_d / 2, handle_h / 2],
                                     rgbaColor=cup_rgba)
    handle_id = p.createMultiBody(baseMass=0.001,
                                  baseCollisionShapeIndex=handle_col,
                                  baseVisualShapeIndex=handle_vis,
                                  basePosition=[pos[0] + r + handle_w / 2,
                                                pos[1], pos[2]])
    hcid = p.createConstraint(body, -1, handle_id, -1, p.JOINT_FIXED,
                              [0, 0, 0], [r + handle_w / 2, 0, 0], [0, 0, 0])
    p.changeConstraint(hcid, maxForce=10000)
    return body


def create_water_bottle(position=None, color=None):
    """Spawn a sport water bottle (r=3 cm, h=20 cm) with grey cap."""
    pos = position or [0.4, 0.2, 0.0]
    body_rgba = color or [0.20, 0.45, 0.85, 1.0]   # blue
    cap_rgba = [0.55, 0.55, 0.55, 1.0]              # grey
    r, h = 0.03, 0.20
    cap_r, cap_h = 0.015, 0.03
    col_body = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
    vis_body = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=body_rgba)
    body = p.createMultiBody(baseMass=0.15,
                             baseCollisionShapeIndex=col_body,
                             baseVisualShapeIndex=vis_body,
                             basePosition=pos)
    col_cap = p.createCollisionShape(p.GEOM_CYLINDER, radius=cap_r, height=cap_h)
    vis_cap = p.createVisualShape(p.GEOM_CYLINDER, radius=cap_r, length=cap_h, rgbaColor=cap_rgba)
    cap_id = p.createMultiBody(baseMass=0.001,
                               baseCollisionShapeIndex=col_cap,
                               baseVisualShapeIndex=vis_cap,
                               basePosition=[pos[0], pos[1], pos[2] + h / 2 + cap_h / 2])
    cid = p.createConstraint(body, -1, cap_id, -1, p.JOINT_FIXED,
                             [0, 0, 0], [0, 0, h / 2 + cap_h / 2], [0, 0, 0])
    p.changeConstraint(cid, maxForce=10000)
    p.changeDynamics(body, -1, lateralFriction=5.0, spinningFriction=2.0,
                     rollingFriction=1.0, contactStiffness=30000, contactDamping=1000)
    return body


def create_toothbrush(position=None, color=None):
    """Spawn a toothbrush — thin cylinder handle (r=1 cm, h=16 cm) + box head."""
    pos = position or [0.4, 0.2, 0.0]
    handle_rgba = color or [0.92, 0.92, 0.95, 1.0]  # white
    head_rgba = [0.25, 0.55, 0.85, 1.0]             # blue bristles
    r_h, h_h = 0.01, 0.16
    col_handle = p.createCollisionShape(p.GEOM_CYLINDER, radius=r_h, height=h_h)
    vis_handle = p.createVisualShape(p.GEOM_CYLINDER, radius=r_h, length=h_h,
                                     rgbaColor=handle_rgba)
    body = p.createMultiBody(baseMass=0.03,
                             baseCollisionShapeIndex=col_handle,
                             baseVisualShapeIndex=vis_handle,
                             basePosition=pos)
    # Head — small box at the top
    head_hx, head_hy, head_hz = 0.012, 0.008, 0.015
    vis_head = p.createVisualShape(p.GEOM_BOX, halfExtents=[head_hx, head_hy, head_hz],
                                   rgbaColor=head_rgba)
    col_head = p.createCollisionShape(p.GEOM_BOX, halfExtents=[head_hx, head_hy, head_hz])
    head_id = p.createMultiBody(baseMass=0.001,
                                baseCollisionShapeIndex=col_head,
                                baseVisualShapeIndex=vis_head,
                                basePosition=[pos[0], pos[1], pos[2] + h_h / 2 + head_hz])
    cid = p.createConstraint(body, -1, head_id, -1, p.JOINT_FIXED,
                             [0, 0, 0], [0, 0, h_h / 2 + head_hz], [0, 0, 0])
    p.changeConstraint(cid, maxForce=10000)
    p.changeDynamics(body, -1, lateralFriction=5.0, spinningFriction=2.0,
                     rollingFriction=1.0, contactStiffness=30000, contactDamping=1000)
    return body


def create_soup_can(position=None, color=None):
    """Spawn a soup can (r=3.3 cm, h=10 cm) with gray label band and rim lips."""
    pos = position or [0.4, 0.2, 0.0]
    body_rgba = color or [0.10, 0.70, 0.15, 1.0]    # vivid green
    label_rgba = [0.50, 0.50, 0.50, 1.0]             # gray label
    lip_rgba  = [0.60, 0.60, 0.60, 1.0]              # metallic gray lip
    r, h = 0.033, 0.10
    lip_r, lip_h = r + 0.003, 0.006   # slight overhang lip
    col_body = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
    vis_body = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=body_rgba)
    body = p.createMultiBody(baseMass=0.35,
                             baseCollisionShapeIndex=col_body,
                             baseVisualShapeIndex=vis_body,
                             basePosition=pos)
    # Label band — slightly larger visual cylinder
    vis_label = p.createVisualShape(p.GEOM_CYLINDER, radius=r + 0.001, length=h * 0.6,
                                    rgbaColor=label_rgba)
    label_id = p.createMultiBody(baseMass=0.001,
                                 baseCollisionShapeIndex=-1,
                                 baseVisualShapeIndex=vis_label,
                                 basePosition=[pos[0], pos[1], pos[2]])
    cid = p.createConstraint(body, -1, label_id, -1, p.JOINT_FIXED,
                             [0, 0, 0], [0, 0, 0], [0, 0, 0])
    p.changeConstraint(cid, maxForce=10000)
    # Top lip
    col_top = p.createCollisionShape(p.GEOM_CYLINDER, radius=lip_r, height=lip_h)
    vis_top = p.createVisualShape(p.GEOM_CYLINDER, radius=lip_r, length=lip_h,
                                  rgbaColor=lip_rgba)
    top_id = p.createMultiBody(baseMass=0.001,
                               baseCollisionShapeIndex=col_top,
                               baseVisualShapeIndex=vis_top,
                               basePosition=[pos[0], pos[1],
                                             pos[2] + h / 2 + lip_h / 2])
    cid_t = p.createConstraint(body, -1, top_id, -1, p.JOINT_FIXED,
                               [0, 0, 0], [0, 0, h / 2 + lip_h / 2], [0, 0, 0])
    p.changeConstraint(cid_t, maxForce=10000)
    # Bottom lip
    col_bot = p.createCollisionShape(p.GEOM_CYLINDER, radius=lip_r, height=lip_h)
    vis_bot = p.createVisualShape(p.GEOM_CYLINDER, radius=lip_r, length=lip_h,
                                  rgbaColor=lip_rgba)
    bot_id = p.createMultiBody(baseMass=0.001,
                               baseCollisionShapeIndex=col_bot,
                               baseVisualShapeIndex=vis_bot,
                               basePosition=[pos[0], pos[1],
                                             pos[2] - h / 2 - lip_h / 2])
    cid_b = p.createConstraint(body, -1, bot_id, -1, p.JOINT_FIXED,
                               [0, 0, 0], [0, 0, -(h / 2 + lip_h / 2)], [0, 0, 0])
    p.changeConstraint(cid_b, maxForce=10000)
    p.changeDynamics(body, -1, lateralFriction=8.0, spinningFriction=3.0,
                     rollingFriction=2.0, contactStiffness=30000, contactDamping=1000)
    return body


def create_person(position=None, color=None, facing=None):
    """Spawn a large static stick-figure person with L-shaped arms and a tray.

    The *position* is the feet location on the table surface.
    *facing* is the angle (radians) the person faces — 0 = +Y, default
    faces inward toward the robot base at (0, 0).
    Arms bend at the elbow: upper arms hang down from shoulders,
    forearms extend forward, supporting a wide tray.
    Everything is static (mass=0) so the figure is immovable.
    The tray has a collision shape so dropped objects rest on it.
    """
    pos = position or [0.4, 0.2, 0.0]
    skin = [0.85, 0.70, 0.55, 1.0]
    shirt = color or [0.95, 0.75, 0.85, 1.0]
    pants = [0.25, 0.25, 0.30, 1.0]
    tray_c = [0.90, 0.75, 0.50, 1.0]      # bright tan tray

    bx, by, bz = pos  # feet position on table

    # Facing direction: default = toward robot base
    if facing is None:
        facing = math.atan2(-by, -bx)

    cf, sf = math.cos(facing), math.sin(facing)

    def _rot(lx, ly):
        """Rotate local (x, y) offset by facing angle, add to feet."""
        return bx + lx * cf - ly * sf, by + lx * sf + ly * cf

    def _rot_quat(local_euler):
        """Compose a local Euler rotation with the facing yaw."""
        local_q = p.getQuaternionFromEuler(local_euler)
        yaw_q = p.getQuaternionFromEuler([0, 0, facing])
        return p.multiplyTransforms([0, 0, 0], yaw_q,
                                    [0, 0, 0], local_q)[1]

    # -- Dimensions --
    leg_h    = 0.22
    torso_h  = 0.24
    upper_arm = 0.12    # vertical segment hanging from shoulder
    fore_arm  = 0.14    # horizontal segment pointing forward
    arm_r    = 0.018
    head_r   = 0.055
    torso_hx = 0.055    # torso half-width (local X)
    torso_hy = 0.035    # torso half-depth (local Y / forward)
    tray_hx  = 0.15     # tray half-width  (local X, spans between arms)
    tray_hy  = 0.125    # tray half-depth  (local Y, extends forward)
    tray_hz  = 0.006    # tray half-thickness

    torso_cz = bz + leg_h + torso_h / 2
    shoulder_z = torso_cz + torso_h / 2 - 0.03
    elbow_z  = shoulder_z - upper_arm     # bottom of upper arm

    def _static(vis_shape, col_shape, world_pos, orn=None):
        """Spawn a static body (mass=0) at world_pos."""
        kw = dict(baseMass=0.0,
                  baseCollisionShapeIndex=col_shape,
                  baseVisualShapeIndex=vis_shape,
                  basePosition=world_pos)
        if orn is not None:
            kw["baseOrientation"] = orn
        return p.createMultiBody(**kw)

    # Facing orientation for boxes (yaw around Z)
    yaw_quat = list(p.getQuaternionFromEuler([0, 0, facing]))

    # === Torso ===
    col_torso = p.createCollisionShape(p.GEOM_BOX,
                                       halfExtents=[torso_hx, torso_hy, torso_h / 2])
    vis_torso = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=[torso_hx, torso_hy, torso_h / 2],
                                    rgbaColor=shirt)
    tx, ty = _rot(0, 0)
    person_id = _static(vis_torso, col_torso, [tx, ty, torso_cz], orn=yaw_quat)

    # === Head ===
    vis_head = p.createVisualShape(p.GEOM_SPHERE, radius=head_r, rgbaColor=skin)
    col_head = p.createCollisionShape(p.GEOM_SPHERE, radius=head_r)
    hx2, hy2 = _rot(0, 0)
    _static(vis_head, col_head, [hx2, hy2, torso_cz + torso_h / 2 + head_r])

    # === Legs ===
    for side in [-0.025, 0.025]:
        vis_leg = p.createVisualShape(p.GEOM_CYLINDER, radius=0.022,
                                      length=leg_h, rgbaColor=pants)
        col_leg = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.022,
                                         height=leg_h)
        lx, ly = _rot(side, 0)
        _static(vis_leg, col_leg, [lx, ly, bz + leg_h / 2])

    # === Upper arms (vertical, hanging from shoulders) ===
    for sign in [-1, 1]:
        local_sx = sign * (torso_hx + arm_r)
        vis_ua = p.createVisualShape(p.GEOM_CYLINDER, radius=arm_r,
                                     length=upper_arm, rgbaColor=skin)
        col_ua = p.createCollisionShape(p.GEOM_CYLINDER, radius=arm_r,
                                        height=upper_arm)
        ux, uy = _rot(local_sx, 0)
        _static(vis_ua, col_ua, [ux, uy, shoulder_z - upper_arm / 2])

    # === Forearms (horizontal, pointing forward from body) ===
    fore_quat = _rot_quat([math.pi / 2, 0, 0])  # along local Y, rotated by facing
    fore_local_y = torso_hy + fore_arm / 2   # centre of forearm along local forward
    for sign in [-1, 1]:
        local_fx = sign * (torso_hx + arm_r)
        vis_fa = p.createVisualShape(p.GEOM_CYLINDER, radius=arm_r,
                                     length=fore_arm, rgbaColor=skin)
        col_fa = p.createCollisionShape(p.GEOM_CYLINDER, radius=arm_r,
                                        height=fore_arm)
        fx, fy = _rot(local_fx, fore_local_y)
        _static(vis_fa, col_fa, [fx, fy, elbow_z], orn=fore_quat)

    # === Hands (spheres at forearm tips) ===
    hand_local_y = torso_hy + fore_arm
    for sign in [-1, 1]:
        local_hx = sign * (torso_hx + arm_r)
        vis_hand = p.createVisualShape(p.GEOM_SPHERE, radius=0.022, rgbaColor=skin)
        col_hand = p.createCollisionShape(p.GEOM_SPHERE, radius=0.022)
        hhx, hhy = _rot(local_hx, hand_local_y)
        _static(vis_hand, col_hand, [hhx, hhy, elbow_z])

    # === Tray (wide platform resting on forearms — collision enabled) ===
    tray_local_y = torso_hy + fore_arm / 2   # centred on forearms
    tray_z = elbow_z + arm_r + tray_hz       # sits on top of forearms
    col_tray = p.createCollisionShape(p.GEOM_BOX,
                                      halfExtents=[tray_hx, tray_hy, tray_hz])
    vis_tray = p.createVisualShape(p.GEOM_BOX,
                                   halfExtents=[tray_hx, tray_hy, tray_hz],
                                   rgbaColor=tray_c)
    tray_wx, tray_wy = _rot(0, tray_local_y)
    _static(vis_tray, col_tray, [tray_wx, tray_wy, tray_z], orn=yaw_quat)

    # === Rim around tray ===
    rim_c = [0.65, 0.45, 0.25, 1.0]
    rim_h = 0.015  # rim half-height
    for hext, off in [
        ([tray_hx, 0.004, rim_h], [0,  tray_hy, rim_h]),
        ([tray_hx, 0.004, rim_h], [0, -tray_hy, rim_h]),
        ([0.004, tray_hy, rim_h], [ tray_hx, 0, rim_h]),
        ([0.004, tray_hy, rim_h], [-tray_hx, 0, rim_h]),
    ]:
        vis_rim = p.createVisualShape(p.GEOM_BOX, halfExtents=hext, rgbaColor=rim_c)
        col_rim = p.createCollisionShape(p.GEOM_BOX, halfExtents=hext)
        rim_wx, rim_wy = _rot(off[0], tray_local_y + off[1])
        _static(vis_rim, col_rim,
                [rim_wx, rim_wy, tray_z + off[2]], orn=yaw_quat)

    return person_id


OBJECT_CREATORS = {
    "block":         create_block,
    "pyramid":       create_pyramid,
    "bottle":        create_bottle,
    "cylinder":      create_cylinder,
    "remote":        create_remote,
    "mug":           create_mug,
    "box":           create_box,
    "pill_bottle":   create_pill_bottle,
    "phone":         create_phone,
    "book":          create_book,
    "cup":           create_cup,
    "water_cup":     create_water_cup,
    "water_bottle":  create_water_bottle,
    "toothbrush":    create_toothbrush,
    "soup_can":      create_soup_can,
    "person":        create_person,
}

# ---------------------------------------------------------------------------
# End-Effector Camera
# ---------------------------------------------------------------------------
CAM_WIDTH = 320
CAM_HEIGHT = 240
CAM_FOV = 90            # wider FOV for better object visibility
CAM_NEAR = 0.02
CAM_FAR = 5.0
CAM_UPDATE_INTERVAL = 6   # update every N sim ticks (~25 fps at 240 Hz)

# Snapshot resolution — 768x768 square for Grounding DINO
SNAP_WIDTH = 768
SNAP_HEIGHT = 768
SNAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
GRASP_SNAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp snapshots")


# ---------------------------------------------------------------------------
# Minimal PNG writer (no PIL dependency)
# ---------------------------------------------------------------------------
def _write_png(filepath: str, rgb_array: np.ndarray):
    """Write an HxWx3 uint8 numpy array as a PNG file using zlib."""
    h, w, _ = rgb_array.shape

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    ihdr = _chunk(b"IHDR", ihdr_data)
    # IDAT — raw image rows with filter byte 0 (None) per row
    raw_rows = b""
    for y in range(h):
        raw_rows += b"\x00" + rgb_array[y].tobytes()
    compressed = zlib.compress(raw_rows, 9)
    idat = _chunk(b"IDAT", compressed)
    # IEND
    iend = _chunk(b"IEND", b"")

    with open(filepath, "wb") as f:
        f.write(sig + ihdr + idat + iend)


def _draw_grasp_overlay(crop_rgb: np.ndarray, grasps: list,
                        bbox_in_crop=None) -> np.ndarray:
    """Draw top-K grasp candidates on a copy of the crop image.

    Returns an HxWx3 uint8 annotated image.  Pure numpy — no PIL/OpenCV.
    """
    img = crop_rgb.copy()
    ih, iw = img.shape[:2]

    RANK_COLORS = [
        (0, 255, 0),      # #1 — green
        (200, 255, 0),    # #2 — yellow-green
        (255, 255, 0),    # #3 — yellow
        (255, 165, 0),    # #4 — orange
        (255, 0, 0),      # #5 — red
    ]

    def _set_px(y, x, color):
        if 0 <= y < ih and 0 <= x < iw:
            img[y, x] = color

    def _draw_line(y0, x0, y1, x1, color, thickness=1):
        """Bresenham-ish line with optional thickness."""
        dx = abs(x1 - x0); dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            for t in range(-thickness // 2, thickness // 2 + 1):
                _set_px(y0 + t, x0, color)
                _set_px(y0, x0 + t, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; x0 += sx
            if e2 < dx:
                err += dx; y0 += sy

    # --- Draw cyan bounding box rectangle ---
    if bbox_in_crop is not None:
        bx1, by1, bx2, by2 = [int(round(v)) for v in bbox_in_crop]
        cyan = (0, 255, 255)
        for x in range(max(0, bx1), min(iw, bx2 + 1)):
            for t in range(2):
                _set_px(by1 + t, x, cyan)
                _set_px(by2 - t, x, cyan)
        for y in range(max(0, by1), min(ih, by2 + 1)):
            for t in range(2):
                _set_px(y, bx1 + t, cyan)
                _set_px(y, bx2 - t, cyan)

    # --- Draw each grasp (worst first so best is on top) ---
    for rank, g in reversed(list(enumerate(grasps))):
        cx, cy = int(round(g["pixel"][0])), int(round(g["pixel"][1]))
        angle = float(g.get("angle", 0.0))
        half_w = float(g.get("width", 30.0)) / 2.0
        color = RANK_COLORS[min(rank, len(RANK_COLORS) - 1)]

        # Crosshair (±6 px)
        for d in range(-6, 7):
            _set_px(cy + d, cx, color)
            _set_px(cy, cx + d, color)

        # Oriented angle line (length = gripper width)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        ex1 = int(round(cx - half_w * cos_a))
        ey1 = int(round(cy - half_w * sin_a))
        ex2 = int(round(cx + half_w * cos_a))
        ey2 = int(round(cy + half_w * sin_a))
        _draw_line(ey1, ex1, ey2, ex2, color, thickness=2)

        # Perpendicular gripper-jaw ticks (±5 px) at each end
        perp_x = -sin_a
        perp_y = cos_a
        for ex, ey in [(ex1, ey1), (ex2, ey2)]:
            jx1 = int(round(ex - 5 * perp_x))
            jy1 = int(round(ey - 5 * perp_y))
            jx2 = int(round(ex + 5 * perp_x))
            jy2 = int(round(ey + 5 * perp_y))
            _draw_line(jy1, jx1, jy2, jx2, color, thickness=2)

    return img


# ---------------------------------------------------------------------------
# 3-D object position computation via depth + segmentation buffers
# ---------------------------------------------------------------------------
def compute_object_positions(loaded_objects: dict,
                             width: int = SNAP_WIDTH,
                             height: int = SNAP_HEIGHT) -> dict:
    """Render a depth + segmentation image and compute the world-frame 3-D
    centre of every loaded object.

    Returns  {name: {"x": float, "y": float, "z": float}} for each object
    that is visible (has at least 1 pixel in the segmentation mask).
    """
    if not loaded_objects:
        return {}

    # Camera parameters (birds-eye, same as capture_birdseye_image)
    target = [0, 0, 1.24]
    eye = [target[0], target[1], 3.0]     # high up for full table coverage
    up = [0, 1, 0]
    fov = 90
    near_val = 0.05
    far_val = 5.0
    aspect = width / height

    view = p.computeViewMatrix(
        cameraEyePosition=eye,
        cameraTargetPosition=target,
        cameraUpVector=up,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=fov, aspect=aspect, nearVal=near_val, farVal=far_val,
    )

    _, _, _rgba, depth_buf, seg_buf = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,  # Tiny renderer is reliable for seg
    )

    depth_arr = np.array(depth_buf, dtype=np.float32).reshape(height, width)
    seg_arr = np.array(seg_buf, dtype=np.int32).reshape(height, width)

    # Build view / projection matrices as 4x4 numpy arrays (column-major from
    # PyBullet → row-major numpy after reshape transpose).
    view_mat = np.array(view, dtype=np.float64).reshape(4, 4).T
    proj_mat = np.array(proj, dtype=np.float64).reshape(4, 4).T

    # Pre-compute inverse(proj @ view) for un-projection
    vp = proj_mat @ view_mat
    vp_inv = np.linalg.inv(vp)

    results = {}
    for obj_name, body_id in loaded_objects.items():
        mask = seg_arr == body_id
        n_pixels = mask.sum()
        if n_pixels == 0:
            # Fallback: use PyBullet ground-truth position
            pos, _ = p.getBasePositionAndOrientation(body_id)
            results[obj_name] = {"x": round(pos[0], 4),
                                 "y": round(pos[1], 4),
                                 "z": round(pos[2], 4)}
            continue

        # Pixel coords of the object (row, col)
        rows, cols = np.where(mask)
        # Average depth for these pixels (OpenGL non-linear depth buffer)
        avg_depth_ndc = float(depth_arr[mask].mean())
        # Convert OpenGL depth buffer → linear depth
        linear_depth = far_val * near_val / (
            far_val - (far_val - near_val) * avg_depth_ndc
        )

        # Average pixel centre
        avg_col = float(cols.mean())
        avg_row = float(rows.mean())

        # Pixel → NDC  (Normalised Device Coordinates [-1, 1])
        ndc_x = (2.0 * avg_col / width) - 1.0
        ndc_y = 1.0 - (2.0 * avg_row / height)  # flip Y
        ndc_z = 2.0 * avg_depth_ndc - 1.0        # OpenGL NDC z

        # Un-project: NDC → world
        clip = np.array([ndc_x, ndc_y, ndc_z, 1.0])
        world_h = vp_inv @ clip
        world_h /= world_h[3]

        results[obj_name] = {
            "x": round(float(world_h[0]), 4),
            "y": round(float(world_h[1]), 4),
            "z": round(float(world_h[2]), 4),
        }

    return results


# ---------------------------------------------------------------------------
# Camera snapshot helpers
# ---------------------------------------------------------------------------
def _get_renderer():
    """Return the best available PyBullet renderer."""
    if hasattr(p, "ER_BULLET_HARDWARE_OPENGL_ACCELERATED"):
        return p.ER_BULLET_HARDWARE_OPENGL_ACCELERATED
    return p.ER_TINY_RENDERER


def capture_ee_image(robot_id: int, ee_link_index: int,
                     width: int = SNAP_WIDTH, height: int = SNAP_HEIGHT) -> np.ndarray:
    """Render a snapshot from the end-effector camera. Returns HxWx3 uint8."""
    ee_state = p.getLinkState(robot_id, ee_link_index,
                              computeForwardKinematics=True)
    ee_pos = np.array(ee_state[4])
    ee_orn = ee_state[5]
    rot = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
    cam_forward = rot[:, 2]
    cam_up = -rot[:, 1]
    target_pos = ee_pos + 0.1 * cam_forward

    view = p.computeViewMatrix(
        cameraEyePosition=ee_pos.tolist(),
        cameraTargetPosition=target_pos.tolist(),
        cameraUpVector=cam_up.tolist(),
    )
    proj = p.computeProjectionMatrixFOV(
        fov=CAM_FOV, aspect=width / height, nearVal=CAM_NEAR, farVal=CAM_FAR,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view, projectionMatrix=proj,
        renderer=_get_renderer(),
    )
    img = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)
    return img[:, :, :3]


def capture_ee_rgbd(robot_id: int, ee_link_index: int,
                     width: int = SNAP_WIDTH, height: int = SNAP_HEIGHT):
    """Capture RGB and linear depth from the end-effector camera.

    Returns: (rgb_uint8 HxWx3, depth_meters HxW float32, view, proj)
    """
    ee_state = p.getLinkState(robot_id, ee_link_index,
                              computeForwardKinematics=True)
    ee_pos = np.array(ee_state[4])
    ee_orn = ee_state[5]
    rot = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
    cam_forward = rot[:, 2]
    cam_up = -rot[:, 1]
    target_pos = ee_pos + 0.1 * cam_forward

    view = p.computeViewMatrix(
        cameraEyePosition=ee_pos.tolist(),
        cameraTargetPosition=target_pos.tolist(),
        cameraUpVector=cam_up.tolist(),
    )
    proj = p.computeProjectionMatrixFOV(
        fov=CAM_FOV, aspect=width / height, nearVal=CAM_NEAR, farVal=CAM_FAR,
    )
    _, _, rgba, depth_buf, _ = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view, projectionMatrix=proj,
        renderer=_get_renderer(),
    )
    img = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    depth_raw = np.array(depth_buf, dtype=np.float32).reshape(height, width)
    # Convert OpenGL non-linear depth buffer to linear meters
    near_val = CAM_NEAR
    far_val = CAM_FAR
    linear_depth = far_val * near_val / (far_val - (far_val - near_val) * depth_raw)
    return img, linear_depth.astype(np.float32), view, proj


def capture_birdseye_image(target_pos=None,
                           width: int = SNAP_WIDTH, height: int = SNAP_HEIGHT) -> np.ndarray:
    """Render a top-down birds-eye view of the scene. Returns HxWx3 uint8."""
    target = target_pos or [0, 0, 1.24]   # table surface height
    eye = [target[0], target[1], 3.0]     # high enough to see entire ring table

    view = p.computeViewMatrix(
        cameraEyePosition=eye,
        cameraTargetPosition=target,
        cameraUpVector=[0, 1, 0],      # Y-forward in the top-down image
    )
    proj = p.computeProjectionMatrixFOV(
        fov=90, aspect=width / height, nearVal=0.05, farVal=5.0,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view, projectionMatrix=proj,
        renderer=_get_renderer(),
    )
    img = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)
    return img[:, :, :3]


def capture_isometric_image(target_pos=None,
                            width: int = SNAP_WIDTH, height: int = SNAP_HEIGHT) -> np.ndarray:
    """Render an isometric (3/4) view of the scene, zoomed to
    show the table-top workspace clearly. Returns HxWx3 uint8."""
    target = target_pos or [0, 0, 0.8]
    distance = 2.125        # 15% closer for better zoom (2.5 * 0.85)
    yaw = 45                # degrees — same as default camera
    pitch = -30             # degrees — same as default camera

    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=0,
        upAxisIndex=2,       # Z-up
    )
    proj = p.computeProjectionMatrixFOV(
        fov=60, aspect=width / height, nearVal=0.1, farVal=20.0,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view, projectionMatrix=proj,
        renderer=_get_renderer(),
    )
    img = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)
    return img[:, :, :3]


# ---------------------------------------------------------------------------
# RGBD variants of the fixed cameras (for vision-based localisation)
# ---------------------------------------------------------------------------

def capture_birdseye_rgbd(target_pos=None,
                          width: int = SNAP_WIDTH, height: int = SNAP_HEIGHT):
    """Birdseye RGBD. Returns (rgb HxWx3, depth_linear HxW float32, view, proj, near, far)."""
    target = target_pos or [0, 0, 1.24]   # table surface height
    eye = [target[0], target[1], 3.0]     # high enough for full ring table
    near_val, far_val = 0.05, 5.0

    view = p.computeViewMatrix(
        cameraEyePosition=eye,
        cameraTargetPosition=target,
        cameraUpVector=[0, 1, 0],
    )
    proj = p.computeProjectionMatrixFOV(
        fov=90, aspect=width / height, nearVal=near_val, farVal=far_val,
    )
    _, _, rgba, depth_buf, _ = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view, projectionMatrix=proj,
        renderer=_get_renderer(),
    )
    img = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    depth_raw = np.array(depth_buf, dtype=np.float32).reshape(height, width)
    linear_depth = far_val * near_val / (far_val - (far_val - near_val) * depth_raw)
    return img, linear_depth.astype(np.float32), view, proj, near_val, far_val


def capture_isometric_rgbd(target_pos=None,
                           width: int = SNAP_WIDTH, height: int = SNAP_HEIGHT):
    """Isometric RGBD. Returns (rgb HxWx3, depth_linear HxW float32, view, proj, near, far)."""
    target = target_pos or [0, 0, 0.8]
    distance = 2.125
    yaw, pitch = 45, -30
    near_val, far_val = 0.1, 20.0

    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=distance, yaw=yaw, pitch=pitch, roll=0, upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=60, aspect=width / height, nearVal=near_val, farVal=far_val,
    )
    _, _, rgba, depth_buf, _ = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view, projectionMatrix=proj,
        renderer=_get_renderer(),
    )
    img = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    depth_raw = np.array(depth_buf, dtype=np.float32).reshape(height, width)
    linear_depth = far_val * near_val / (far_val - (far_val - near_val) * depth_raw)
    return img, linear_depth.astype(np.float32), view, proj, near_val, far_val


def capture_tabletop_rgbd(angle_deg: float = 0.0,
                          width: int = SNAP_WIDTH, height: int = SNAP_HEIGHT):
    """Table-level camera looking inward at the ring table from a given angle.

    Positioned at the outer edge of the table at object height, looking
    toward the centre — good for detecting objects from the side when
    birdseye or isometric views struggle.

    Returns (rgb HxWx3, depth_linear HxW float32, view, proj, near, far).
    """
    r = 1.8                   # just beyond the table outer edge (1.35)
    table_z = 1.24
    obj_h = 0.06              # typical half-height of objects on the table
    cam_z = table_z + obj_h   # camera at object mid-height
    near_val, far_val = 0.05, 5.0

    rad = math.radians(angle_deg)
    eye = [r * math.cos(rad), r * math.sin(rad), cam_z]
    target = [0.0, 0.0, cam_z]   # look toward centre at same height

    view = p.computeViewMatrix(
        cameraEyePosition=eye,
        cameraTargetPosition=target,
        cameraUpVector=[0, 0, 1],
    )
    proj = p.computeProjectionMatrixFOV(
        fov=70, aspect=width / height, nearVal=near_val, farVal=far_val,
    )
    _, _, rgba, depth_buf, _ = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view, projectionMatrix=proj,
        renderer=_get_renderer(),
    )
    img = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    depth_raw = np.array(depth_buf, dtype=np.float32).reshape(height, width)
    linear_depth = far_val * near_val / (far_val - (far_val - near_val) * depth_raw)
    return img, linear_depth.astype(np.float32), view, proj, near_val, far_val


def _unproject_bbox_centre(bbox, depth, view_flat, proj_flat, near_val, far_val, w, h):
    """Unproject the centre of a bounding box to world XYZ using the depth buffer.

    bbox: (x1, y1, x2, y2) in pixel coords.
    depth: HxW linear depth (metres).
    Returns (x, y, z) world coords or None if depth is invalid.
    """
    x1, y1, x2, y2 = bbox
    cu = (x1 + x2) / 2.0
    cv = (y1 + y2) / 2.0
    ui = int(max(0, min(w - 1, round(cu))))
    vi = int(max(0, min(h - 1, round(cv))))
    Z = float(depth[vi, ui])
    if Z <= 0 or Z > far_val * 0.99 or math.isinf(Z) or math.isnan(Z):
        return None

    # Convert linear metres back to non-linear buffer value for NDC
    d_raw = (far_val * (Z - near_val)) / (Z * (far_val - near_val))
    ndc_x = (2.0 * cu / w) - 1.0
    ndc_y = 1.0 - (2.0 * cv / h)
    ndc_z = 2.0 * d_raw - 1.0

    view_mat = np.array(view_flat, dtype=np.float64).reshape(4, 4).T
    proj_mat = np.array(proj_flat, dtype=np.float64).reshape(4, 4).T
    vp_inv = np.linalg.inv(proj_mat @ view_mat)
    clip_pt = np.array([ndc_x, ndc_y, ndc_z, 1.0])
    world_h = vp_inv @ clip_pt
    world_h /= world_h[3]
    return (round(float(world_h[0]), 4),
            round(float(world_h[1]), 4),
            round(float(world_h[2]), 4))


# --- Ring-table spatial bounds (used to reject robot-arm false positives) ---
_TABLE_SURFACE_Z = 1.22
_TABLE_INNER_R   = 0.54
_TABLE_OUTER_R   = 1.35


def _is_plausible_table_position(pos):
    """Return True if *pos* could be an object on the ring table.

    Rejects positions inside the table hole (robot zone), beyond the
    table edge, or at an implausible height.
    """
    x, y, z = pos
    r = math.sqrt(x * x + y * y)
    if r < _TABLE_INNER_R * 0.7:       # inside robot zone
        return False
    if r > _TABLE_OUTER_R * 1.3:       # well beyond the table
        return False
    if z > _TABLE_SURFACE_Z + 0.35:    # way above any table object
        return False
    if z < _TABLE_SURFACE_Z - 0.10:    # below the table
        return False
    return True


def _best_valid_detection(det, depth, view, proj, near, far, w, h, console=None, label=""):
    """Try each detection candidate (best score first); return the first
    whose unprojected world position passes table-plausibility check.

    Returns (x, y, z) or None.
    """
    candidates = det.get("all_boxes", [])
    # Fallback for older _detect_objects without all_boxes
    if not candidates and det.get("visible") and det.get("bbox"):
        candidates = [(*det["bbox"], det.get("score", 0))]
    for bbox_x1, bbox_y1, bbox_x2, bbox_y2, _score in candidates:
        pos = _unproject_bbox_centre(
            (bbox_x1, bbox_y1, bbox_x2, bbox_y2),
            depth, view, proj, near, far, w, h)
        if pos is None:
            continue
        if _is_plausible_table_position(pos):
            return pos
        # Rejected — log if this was the top detection
        if _score == candidates[0][4]:
            print(f"  [VISION] {label}: rejected ({pos[0]:.3f}, "
                  f"{pos[1]:.3f}, {pos[2]:.3f}) — likely robot arm")
    return None


def compute_object_positions_vision(loaded_objects: dict,
                                    console=None) -> dict:
    """Locate every loaded object using ONLY camera RGB-D + Grounding DINO.

    Pipeline:
      1. Capture birdseye RGBD.
      2. Run Grounding DINO on the birdseye RGB for each object name.
      3. For detected objects, unproject bbox centre via depth → world XYZ,
         rejecting positions that lie on the robot arm.
      4. For objects not found in birdseye, retry with isometric RGBD.

    Returns  {name: {"x": float, "y": float, "z": float}}  (same interface
    as the old compute_object_positions).
    """
    if not loaded_objects:
        return {}

    from clip_bbox import _ensure_model, _detect_objects
    _ensure_model(console)

    object_names = list(loaded_objects.keys())
    results = {}
    remaining = list(object_names)

    # --- Person: hardcoded tray position (mirrors create_person with facing) ---
    if "person" in remaining:
        _ANG = math.radians(-100)          # spawn angle on ring
        _R   = 0.65                        # person ring radius (closer to base)
        _BZ  = 1.235                       # OBJ_Z = TABLE_SURFACE_Z + THICKNESS/2
        _bx  = _R * math.cos(_ANG)
        _by  = _R * math.sin(_ANG)
        _facing = math.atan2(-_by, -_bx)   # face toward robot base
        _cf, _sf = math.cos(_facing), math.sin(_facing)
        # Person geometry (mirrors create_person constants)
        _leg_h = 0.22; _torso_h = 0.24; _torso_hy = 0.035
        _fore_arm = 0.14; _arm_r = 0.018; _tray_hz = 0.006; _upper_arm = 0.12
        _torso_cz  = _BZ + _leg_h + _torso_h / 2
        _shoulder_z = _torso_cz + _torso_h / 2 - 0.03
        _elbow_z    = _shoulder_z - _upper_arm
        _tray_local_y = _torso_hy + _fore_arm / 2   # 0.105 m forward
        _tray_x = _bx + 0 * _cf - _tray_local_y * _sf
        _tray_y = _by + 0 * _sf + _tray_local_y * _cf
        _tray_z = _elbow_z + _arm_r + _tray_hz
        results["person"] = {"x": _tray_x, "y": _tray_y, "z": _tray_z}
        remaining.remove("person")
        print(f"  [VISION] person: hardcoded tray ({_tray_x:.3f}, {_tray_y:.3f}, {_tray_z:.3f})")

    # --- Pass 1: birdseye ---
    bird_rgb, bird_depth, bird_view, bird_proj, bird_near, bird_far = \
        capture_birdseye_rgbd()
    h, w = bird_depth.shape

    bird_dets = _detect_objects(bird_rgb, remaining)
    for name in list(remaining):
        det = bird_dets.get(name, {})
        pos = _best_valid_detection(
            det, bird_depth, bird_view, bird_proj,
            bird_near, bird_far, w, h,
            console=console, label=f"{name} birdseye")
        if pos:
            entry = {"x": pos[0], "y": pos[1], "z": pos[2]}
            # Estimate object width from DINO bbox at detected depth
            bbox = det.get("bbox")
            if bbox:
                bx1, by1, bx2, by2 = bbox
                bbox_w_px = bx2 - bx1
                bbox_h_px = by2 - by1
                # For birdseye, compute world-per-pixel at detected depth
                ui = int(max(0, min(w - 1, round((bx1 + bx2) / 2.0))))
                vi = int(max(0, min(h - 1, round((by1 + by2) / 2.0))))
                Z = float(bird_depth[vi, ui])
                if Z > 0 and Z < bird_far * 0.99:
                    fov_rad = math.radians(70)  # birdseye FOV
                    wpp = (2.0 * Z * math.tan(fov_rad / 2.0)) / w
                    entry["width"] = bbox_w_px * wpp
                    entry["height"] = bbox_h_px * wpp
            results[name] = entry
            remaining.remove(name)
            w_str = f", ~{entry['width']*100:.1f}cm" if "width" in entry else ""
            print(f"  [VISION] {name}: birdseye → "
                  f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}){w_str}")

    if not remaining:
        return results

    # --- Pass 2: isometric (for objects occluded from above) ---
    iso_rgb, iso_depth, iso_view, iso_proj, iso_near, iso_far = \
        capture_isometric_rgbd()
    h2, w2 = iso_depth.shape

    iso_dets = _detect_objects(iso_rgb, remaining)
    for name in list(remaining):
        det = iso_dets.get(name, {})
        pos = _best_valid_detection(
            det, iso_depth, iso_view, iso_proj,
            iso_near, iso_far, w2, h2,
            console=console, label=f"{name} isometric")
        if pos:
            entry = {"x": pos[0], "y": pos[1], "z": pos[2]}
            bbox = det.get("bbox")
            if bbox:
                bx1, by1, bx2, by2 = bbox
                bbox_w_px = bx2 - bx1
                ui = int(max(0, min(w2 - 1, round((bx1 + bx2) / 2.0))))
                vi = int(max(0, min(h2 - 1, round((by1 + by2) / 2.0))))
                Z = float(iso_depth[vi, ui])
                if Z > 0 and Z < iso_far * 0.99:
                    fov_rad = math.radians(70)  # isometric FOV
                    wpp = (2.0 * Z * math.tan(fov_rad / 2.0)) / w2
                    entry["width"] = bbox_w_px * wpp
            results[name] = entry
            remaining.remove(name)
            w_str = f", ~{entry['width']*100:.1f}cm" if "width" in entry else ""
            print(f"  [VISION] {name}: isometric → "
                  f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}){w_str}")

    # Objects still not found get no entry (skip silently)
    for name in remaining:
        print(f"  [VISION] {name}: not detected in any view")

    return results


def locate_single_object_vision(query: str, console=None):
    """Locate a single object by natural-language description using Grounding DINO.

    Unlike compute_object_positions_vision this does NOT require the object
    to be in loaded_objects — it searches for whatever the caller describes.
    Tries birdseye first, then isometric.  Iterates through all detection
    candidates (best score first) and rejects positions that lie on the
    robot arm rather than the table.

    Returns {"x": float, "y": float, "z": float} or None.
    """
    from clip_bbox import _ensure_model, _detect_objects
    _ensure_model(console)

    # --- Pass 1: birdseye ---
    bird_rgb, bird_depth, bird_view, bird_proj, bird_near, bird_far = \
        capture_birdseye_rgbd()
    h, w = bird_depth.shape
    bird_dets = _detect_objects(bird_rgb, [query])
    det = bird_dets.get(query, {})
    pos = _best_valid_detection(
        det, bird_depth, bird_view, bird_proj,
        bird_near, bird_far, w, h,
        console=console, label=f"'{query}' birdseye")
    if pos:
        print(f"  [VISION] '{query}': birdseye → "
              f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        return {"x": pos[0], "y": pos[1], "z": pos[2]}

    # --- Pass 2: isometric ---
    iso_rgb, iso_depth, iso_view, iso_proj, iso_near, iso_far = \
        capture_isometric_rgbd()
    h2, w2 = iso_depth.shape
    iso_dets = _detect_objects(iso_rgb, [query])
    det2 = iso_dets.get(query, {})
    pos = _best_valid_detection(
        det2, iso_depth, iso_view, iso_proj,
        iso_near, iso_far, w2, h2,
        console=console, label=f"'{query}' isometric")
    if pos:
        print(f"  [VISION] '{query}': isometric → "
              f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        return {"x": pos[0], "y": pos[1], "z": pos[2]}

    # --- Pass 3: table-level cameras (4 views around the table) ---
    for tt_angle in [0, 90, 180, 270]:
        tt_rgb, tt_depth, tt_view, tt_proj, tt_near, tt_far = \
            capture_tabletop_rgbd(angle_deg=tt_angle)
        h3, w3 = tt_depth.shape
        tt_dets = _detect_objects(tt_rgb, [query])
        det3 = tt_dets.get(query, {})
        pos = _best_valid_detection(
            det3, tt_depth, tt_view, tt_proj,
            tt_near, tt_far, w3, h3,
            console=console, label=f"'{query}' tabletop-{tt_angle}°")
        if pos:
            print(f"  [VISION] '{query}': tabletop-{tt_angle}° → "
                  f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            return {"x": pos[0], "y": pos[1], "z": pos[2]}

    print(f"  [VISION] '{query}': not detected in any view")
    return None


def save_snapshots(robot_id: int, ee_link_index: int, console) -> dict:
    """Capture & save both camera views. Returns dict with file paths."""
    os.makedirs(SNAP_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    ee_img = capture_ee_image(robot_id, ee_link_index)
    ee_path = os.path.join(SNAP_DIR, f"ee_cam_{timestamp}.png")
    _write_png(ee_path, ee_img)

    bird_img = capture_birdseye_image()
    bird_path = os.path.join(SNAP_DIR, f"birdseye_{timestamp}.png")
    _write_png(bird_path, bird_img)

    iso_img = capture_isometric_image()
    iso_path = os.path.join(SNAP_DIR, f"isometric_{timestamp}.png")
    _write_png(iso_path, iso_img)

    console.write(f"Snapshots saved ({SNAP_WIDTH}x{SNAP_HEIGHT})")

    return {"ee": ee_path, "birdseye": bird_path, "isometric": iso_path,
            "ee_rgb": ee_img, "birdseye_rgb": bird_img, "isometric_rgb": iso_img}


def images_to_base64(rgb_array: np.ndarray) -> str:
    """Encode an HxWx3 uint8 numpy array as a base64 PNG string (for API use)."""
    import io
    buf = io.BytesIO()
    h, w, _ = rgb_array.shape

    def _chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    raw = b""
    for y in range(h):
        raw += b"\x00" + rgb_array[y].tobytes()
    idat = _chunk(b"IDAT", zlib.compress(raw, 9))
    iend = _chunk(b"IEND", b"")

    buf.write(sig + ihdr + idat + iend)
    return base64.b64encode(buf.getvalue()).decode("ascii")



class EndEffectorCamera:
    """Synthetic camera rigidly attached to the robot's end-effector.

    Opens a separate tkinter Toplevel window that shows a live feed
    rendered by PyBullet's built-in getCameraImage.
    """

    def __init__(self, robot_id: int, ee_link_index: int, console_root):
        self.robot_id = robot_id
        self.ee_link_index = ee_link_index
        self._tick_counter = 0
        self.enabled = True

        # Pre-compute projection matrix (constant)
        self._proj_matrix = p.computeProjectionMatrixFOV(
            fov=CAM_FOV,
            aspect=CAM_WIDTH / CAM_HEIGHT,
            nearVal=CAM_NEAR,
            farVal=CAM_FAR,
        )

        # --- tkinter window for the feed ---
        self._win = tk.Toplevel(console_root)
        self._win.title("End-Effector Camera")
        self._win.geometry(f"{CAM_WIDTH}x{CAM_HEIGHT}")
        self._win.resizable(False, False)
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._canvas = tk.Canvas(self._win, width=CAM_WIDTH, height=CAM_HEIGHT,
                                 bg="black", highlightthickness=0)
        self._canvas.pack()
        self._photo = None  # will hold the current tk.PhotoImage

    # -- public API ---------------------------------------------------------

    def tick(self):
        """Called every sim tick.  Only renders at CAM_UPDATE_INTERVAL."""
        if not self.enabled:
            return
        self._tick_counter += 1
        if self._tick_counter % CAM_UPDATE_INTERVAL != 0:
            return
        self._render()

    def destroy(self):
        self.enabled = False
        try:
            self._win.destroy()
        except Exception:
            pass

    # -- internal -----------------------------------------------------------

    def _on_close(self):
        self.enabled = False
        self._win.destroy()

    def _render(self):
        """Compute view matrix from EE pose and blit the image."""
        # Get end-effector world pose
        ee_state = p.getLinkState(self.robot_id, self.ee_link_index,
                                  computeForwardKinematics=True)
        ee_pos = np.array(ee_state[4])   # world position
        ee_orn = ee_state[5]             # world orientation quaternion

        # Rotation matrix from quaternion
        rot = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)

        # Camera convention: camera looks along -Z in its local frame.
        # The EE's local Z axis typically points outward (approach axis).
        cam_forward = rot[:, 2]          # local Z -> world
        cam_up = -rot[:, 1]              # local -Y -> world up

        # Target point a short distance ahead of the camera
        target_pos = ee_pos + 0.1 * cam_forward

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=ee_pos.tolist(),
            cameraTargetPosition=target_pos.tolist(),
            cameraUpVector=cam_up.tolist(),
        )

        # Render
        _, _, rgba, _, _ = p.getCameraImage(
            width=CAM_WIDTH,
            height=CAM_HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL_ACCELERATED
                     if hasattr(p, "ER_BULLET_HARDWARE_OPENGL_ACCELERATED")
                     else p.ER_TINY_RENDERER,
        )

        # Convert to tkinter-compatible PhotoImage (PPM via numpy)
        img_array = np.array(rgba, dtype=np.uint8).reshape(CAM_HEIGHT, CAM_WIDTH, 4)
        rgb = img_array[:, :, :3]  # drop alpha

        # Build a PPM header in-memory (fast, no PIL needed)
        header = f"P6 {CAM_WIDTH} {CAM_HEIGHT} 255 ".encode()
        ppm_data = header + rgb.tobytes()

        try:
            self._photo = tk.PhotoImage(data=ppm_data, format="PPM")
            self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
        except tk.TclError:
            self.enabled = False


# ---------------------------------------------------------------------------
# Command Console (tkinter window)
# ---------------------------------------------------------------------------
class CommandConsole:
    """Tkinter command-line window.

    Uses root.update() called from the simulation while-loop instead of
    root.mainloop().  This is the only approach that reliably accepts
    keyboard input on Windows when PyBullet's OpenGL window is also open.
    """

    def __init__(self):
        self._command_queue = deque()
        self.alive = True               # set to False when window is closed

        # ---- Build the window ----
        self._root = tk.Tk()
        self._root.title("Kinova Gen 3 — Command Console")
        self._root.geometry("700x420")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Output area
        out_frame = tk.Frame(self._root)
        out_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._output = scrolledtext.ScrolledText(
            out_frame, height=18, width=80, font=("Consolas", 10),
            bg="#1e1e1e", fg="#d4d4d4", insertbackground="#d4d4d4",
            state=tk.DISABLED,
        )
        self._output.pack(fill=tk.BOTH, expand=True)

        # Input row
        input_frame = tk.Frame(self._root)
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(input_frame, text=">", font=("Consolas", 11, "bold")).pack(side=tk.LEFT)

        self._entry = tk.Entry(input_frame, font=("Consolas", 10))
        self._entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        self._entry.bind("<Return>", self._on_submit)
        self._entry.focus()

        # Welcome
        self._append_text("Kinova Gen 3 Command Console")
        self._append_text("=" * 40)
        self._append_text("Commands:")
        self._append_text("  jarvis <request>   — Vision-based grasp planner (Grounding DINO + GR-ConvNet)")
        self._append_text("    j go / j 1-3 / j clear — execute, select range, or discard")
        self._append_text("  jarvistest <obj>   — Full vision grasp without LLM")
        self._append_text("  go                 — Execute the pending movement plan")
        self._append_text("  plan               — Re-display the pending plan")
        self._append_text("  clear              — Discard the pending plan")
        self._append_text("  snapshot           — Save EE + birds-eye camera images")
        self._append_text("  camera             — Toggle the end-effector camera")
        self._append_text("  bbox_screenshot    — Run CLIP detection and save annotated images")
        self._append_text("  help               — Show this help message")
        # List spawnable object types
        try:
            objs = ", ".join(sorted(OBJECT_CREATORS.keys()))
        except Exception:
            objs = "block, cylinder, bottle, mug, remote, box"
        self._append_text(f"  spawnable objects: {objs}\n")

    # -- public API ---------------------------------------------------------

    def update(self):
        """Pump tkinter events.  Call this once per simulation tick."""
        try:
            self._root.update()
        except tk.TclError:
            self.alive = False

    def has_command(self) -> bool:
        return len(self._command_queue) > 0

    def pop_command(self) -> str:
        return self._command_queue.popleft()

    def write(self, text: str):
        self._append_text(text)
        # Immediately process so the text shows up
        try:
            self._root.update_idletasks()
        except tk.TclError:
            pass

    def stop(self):
        try:
            self._root.destroy()
        except Exception:
            pass
        self.alive = False

    # -- internal -----------------------------------------------------------

    def _append_text(self, text: str):
        self._output.configure(state=tk.NORMAL)
        self._output.insert(tk.END, text + "\n")
        self._output.see(tk.END)
        self._output.configure(state=tk.DISABLED)

    def _on_submit(self, _event=None):
        cmd = self._entry.get().strip()
        if cmd:
            self._append_text(f"> {cmd}")
            self._command_queue.append(cmd)
            self._entry.delete(0, tk.END)

    def _on_close(self):
        self.alive = False
        self._root.destroy()


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def main():
    # Make sure meshes are available
    ensure_meshes_exist()

    # ---- Start PyBullet ----
    physics_client = p.connect(p.GUI)

    # Set additional search path so PyBullet can find built-in assets (plane, etc.)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Gravity
    p.setGravity(0, 0, -9.81)

    # Time step
    time_step = 1.0 / 240.0
    p.setTimeStep(time_step)

    # ---- Load ground plane ----
    plane_id = p.loadURDF("plane.urdf")

    # ---- Spawn pedestal (skinny cylinder the height of the robot) ----
    PEDESTAL_HEIGHT = 1.18      # metres — roughly the robot's full upright height
    PEDESTAL_RADIUS = 0.06      # skinny
    ped_col = p.createCollisionShape(p.GEOM_CYLINDER,
                                     radius=PEDESTAL_RADIUS,
                                     height=PEDESTAL_HEIGHT)
    ped_vis = p.createVisualShape(p.GEOM_CYLINDER,
                                  radius=PEDESTAL_RADIUS,
                                  length=PEDESTAL_HEIGHT,
                                  rgbaColor=[0.25, 0.25, 0.25, 1])
    pedestal_id = p.createMultiBody(
        baseMass=0,                     # static / immovable
        baseCollisionShapeIndex=ped_col,
        baseVisualShapeIndex=ped_vis,
        basePosition=[0, 0, PEDESTAL_HEIGHT / 2],  # centre at half-height
    )

    # ---- Load Kinova Gen 3 ----
    urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KinovaGen3_7DOF_Meshes.urdf")
    start_pos = [0, 0, PEDESTAL_HEIGHT]   # base sits on top of pedestal
    start_orn = p.getQuaternionFromEuler([0, 0, 0])

    robot_id = p.loadURDF(
        urdf_path,
        basePosition=start_pos,
        baseOrientation=start_orn,
        useFixedBase=True,          # robot base bolted to the world
        flags=p.URDF_USE_SELF_COLLISION,
    )

    # ---- Print robot info ----
    num_joints = p.getNumJoints(robot_id)
    print(f"\n{'='*50}")
    print(f"  Kinova Gen 3 loaded  —  {num_joints} joints")
    print(f"{'='*50}")

    joint_info = []
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode("utf-8")
        joint_type = info[2]
        lower = info[8]
        upper = info[9]
        max_force = info[10]
        max_vel = info[11]
        type_str = {0: "revolute", 1: "prismatic", 2: "spherical",
                    3: "planar", 4: "fixed", 5: "continuous"}.get(joint_type, "unknown")
        print(f"  Joint {i}: {name:35s}  type={type_str:10s}  limits=[{lower:.3f}, {upper:.3f}]")
        joint_info.append({
            "index": i,
            "name": name,
            "type": joint_type,
            "lower": lower,
            "upper": upper,
            "max_force": max_force,
            "max_vel": max_vel,
        })

    # Identify movable (non-fixed) joints
    movable_joints = [j for j in joint_info if j["type"] != 4]
    print(f"\n  Movable joints: {len(movable_joints)}")

    # ---- Find end-effector link index (for camera) ----
    ee_link_index = None
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        link_name = info[12].decode("utf-8")
        if link_name == "end_effector_link":
            ee_link_index = i
            break
    if ee_link_index is None:
        # Fallback: use the last link
        ee_link_index = num_joints - 1
    print(f"  End-effector link index: {ee_link_index}")

    # ---- Load and attach gripper ----
    gripper_info = load_gripper(robot_id, ee_link_index)
    print(f"\n  Robotiq 2F-85 gripper attached (body id: {gripper_info['id']})")
    print(f"    Left finger joint idx:  {gripper_info['left_idx']}")
    print(f"    Right finger joint idx: {gripper_info['right_idx']}")

    # ---- Add GUI sliders for each movable joint ----
    sliders = []
    for j in movable_joints:
        lo, hi = REAL_JOINT_LIMITS.get(j["name"], (j["lower"], j["upper"]))
        # For continuous joints the URDF reports (0, -1); use our real limits
        if lo >= hi:
            lo, hi = -math.pi, math.pi
        slider_id = p.addUserDebugParameter(j["name"], lo, hi, 0.0)
        sliders.append((j["index"], slider_id, j["max_force"]))

    # Gripper slider
    gripper_slider_id = p.addUserDebugParameter("gripper", 0.0, 1.2, 0.0)

    # ---- Configure rendering ----
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)  # don't steal keyboard
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
    p.resetDebugVisualizerCamera(
        cameraDistance=2.5,
        cameraYaw=350,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.8],
    )
    cam_yaw = 350  # mutable — arrow keys rotate this

    # ---- Spawn ring table around the robot ----
    TABLE_SURFACE_Z = 1.22          # 4 cm above pedestal top
    TABLE_THICKNESS = 0.03          # 3 cm thick surface
    TABLE_INNER_R = 0.54            # 3x original — clears pedestal + arm base
    TABLE_OUTER_R = 1.35            # 3x original — comfortable reaching distance
    TABLE_MID_R = (TABLE_INNER_R + TABLE_OUTER_R) / 2
    TABLE_RADIAL_HALF = (TABLE_OUTER_R - TABLE_INNER_R) / 2
    N_TABLE_SEGMENTS = 48           # more segments for the larger ring
    table_color = [0.45, 0.30, 0.15, 1]  # warm wood tone

    for seg_i in range(N_TABLE_SEGMENTS):
        angle = 2 * math.pi * seg_i / N_TABLE_SEGMENTS
        cx = TABLE_MID_R * math.cos(angle)
        cy = TABLE_MID_R * math.sin(angle)
        tangent_half = TABLE_MID_R * math.tan(math.pi / N_TABLE_SEGMENTS) * 1.02
        seg_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[TABLE_RADIAL_HALF, tangent_half, TABLE_THICKNESS / 2])
        seg_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[TABLE_RADIAL_HALF, tangent_half, TABLE_THICKNESS / 2],
            rgbaColor=table_color)
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=seg_col,
            baseVisualShapeIndex=seg_vis,
            basePosition=[cx, cy, TABLE_SURFACE_Z],
            baseOrientation=p.getQuaternionFromEuler([0, 0, angle]),
        )

    # ---- 4 small legs under the table ----
    LEG_RADIUS = 0.015
    LEG_HEIGHT = TABLE_SURFACE_Z - TABLE_THICKNESS / 2  # floor to underside
    for leg_angle in [0, math.pi / 2, math.pi, 3 * math.pi / 2]:
        lx = TABLE_MID_R * math.cos(leg_angle)
        ly = TABLE_MID_R * math.sin(leg_angle)
        leg_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=LEG_RADIUS, height=LEG_HEIGHT)
        leg_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=LEG_RADIUS, length=LEG_HEIGHT,
                                      rgbaColor=[0.3, 0.2, 0.1, 1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=leg_col,
                          baseVisualShapeIndex=leg_vis,
                          basePosition=[lx, ly, LEG_HEIGHT / 2])

    # ---- Auto-spawn objects on the table ----
    OBJ_Z = TABLE_SURFACE_Z + TABLE_THICKNESS / 2  # top of table surface
    OBJ_R = TABLE_MID_R                                # centre of the ring surface
    # ADL assistive-robotics test scene
    auto_objects = {
        "pill_bottle": [OBJ_R * math.cos(math.radians(-60)),
                        OBJ_R * math.sin(math.radians(-60)),
                        OBJ_Z + 0.035],       # half-height of 7cm body
        "water_bottle": [OBJ_R * math.cos(math.radians(20)),
                        OBJ_R * math.sin(math.radians(20)),
                        OBJ_Z + 0.10],        # half-height of 20cm bottle
        "toothbrush":  [OBJ_R * math.cos(math.radians(-20)),
                        OBJ_R * math.sin(math.radians(-20)),
                        OBJ_Z + 0.08],        # half-height of 16cm handle
        "soup_can":    [OBJ_R * math.cos(math.radians(40)),
                        OBJ_R * math.sin(math.radians(40)),
                        OBJ_Z + 0.05],        # half-height of 10cm can
        "water_cup":   [OBJ_R * math.cos(math.radians(80)),
                        OBJ_R * math.sin(math.radians(80)),
                        OBJ_Z + 0.065],       # half-height of 13cm mug
        "person":      [0.65 * math.cos(math.radians(-100)),
                        0.65 * math.sin(math.radians(-100)),
                        OBJ_Z],               # feet on table surface (closer radius)
    }
    preloaded_objects = {}
    for obj_name, obj_pos in auto_objects.items():
        if obj_name == "person":
            # Person needs facing angle to orient toward robot base
            body_id = OBJECT_CREATORS[obj_name](position=obj_pos,
                                                 facing=math.atan2(-obj_pos[1], -obj_pos[0]))
        else:
            body_id = OBJECT_CREATORS[obj_name](position=obj_pos)
        preloaded_objects[obj_name] = body_id

    # ---- Launch Command Console ----
    console = CommandConsole()
    console.write("Simulation ready.")
    console.write(f"  {len(movable_joints)} joints, Robotiq 2F-85 gripper")
    console.write(f"  Ring table at z={TABLE_SURFACE_Z:.2f}m, {len(preloaded_objects)} objects")

    # ---- End-Effector Camera ----
    ee_camera = EndEffectorCamera(robot_id, ee_link_index, console._root)
    console.write("Camera ready.")

    # ---- Movement execution state ----
    state = {
        "pending_plan": [],
        "executing": False,
        "exec_step_index": 0,
        "exec_settle_ticks": 0,
        "exec_step_ticks": 0,     # total ticks spent on current step
        "exec_prev_positions": None,  # snapshot for stall detection
        "exec_stall_ticks": 0,       # ticks since last meaningful movement
        "original_request": "",       # user's natural-language request for replanning
        "_requery_fired": False,      # limit one requery per stuck step
    }
    STEP_TIMEOUT_TICKS = 2400       # ~10 seconds at 240 Hz before auto-skip
    STALL_TICKS = 720               # ~3 seconds at 240 Hz — auto-advance if stuck
    STALL_THRESHOLD = 0.001         # rad — movement below this counts as stalled
    loaded_objects = dict(preloaded_objects)   # name -> pybullet body id (start with auto-spawned)
    POSITION_TOLERANCE = 0.05   # rad
    AI_MAX_VELOCITY = 1.0       # rad/s — slower than default for smooth motion

    # Positions to hold after AI execution (joint_index -> angle)
    held_positions = {}         # empty = use sliders; populated = hold AI pose
    held_gripper = None         # None = use slider; float = hold AI gripper pose

    # quick lookup: joint name -> (index, max_force)
    joint_lookup = {j["name"]: (j["index"], j["max_force"]) for j in movable_joints}

    # ------------------------------------------------------------------
    # Shared helpers used by jarvis, jarvistest, and refine
    # ------------------------------------------------------------------

    def _extract_object_phrase(text: str) -> str:
        """Extract the object noun-phrase from a user command.

        Strips grasp verbs, filler words, prepositions, and articles so that
        'pick up the red block' → 'red block',
        'grab that tall green bottle' → 'tall green bottle',
        'point yourself at the cylinder' → 'cylinder'.
        The result is passed directly to Grounding DINO as the query.
        """
        # Strip punctuation and extra whitespace
        txt = re.sub(r"[,;!?.]+", " ", text).strip()
        strip_words = {
            # verbs / verb phrases
            "pick", "up", "grab", "grasp", "get", "move", "lift", "take",
            "fetch", "snatch", "seize", "hold", "place", "put", "drop",
            "point", "aim", "look", "find", "locate", "detect", "show",
            "reach", "approach", "touch", "push", "pull",
            # polite / modal
            "please", "can", "you", "could", "would", "will", "shall",
            "should", "let", "lets", "try", "to", "me", "i", "want",
            # reflexive / pronouns
            "yourself", "itself", "myself", "it", "them", "that", "this",
            "those", "these", "there", "here",
            # prepositions / conjunctions
            "at", "on", "in", "into", "onto", "from", "for", "of", "with",
            "toward", "towards", "over", "near", "by", "and", "or",
            # articles
            "the", "a", "an",
            # misc
            "go", "now", "just", "also", "then", "do", "does",
        }
        words = txt.split()
        # Strip from the front
        while words and words[0].lower() in strip_words:
            words.pop(0)
        # Strip from the back
        while words and words[-1].lower() in strip_words:
            words.pop()
        phrase = " ".join(words).strip()
        return phrase if phrase else text.strip()

    def _find_best_object_match(text: str, names: list):
        """Fuzzy object matching that prioritises grasp targets."""
        txt = text.lower()
        words = re.findall(r"\w+", txt)
        names_lower = {n: n.lower() for n in names}

        color_map = {
            "block": ["red"],
            "cylinder": ["blue"],
            "bottle": ["green"],
            "mug": ["yellow"],
            "remote": ["remote", "gray", "grey", "dark", "black"],
            "box": ["box", "brown", "wood"],
        }

        def _match_after(keyword_idx, words_list):
            tail = " ".join(words_list[keyword_idx + 1:])
            for n in names:
                for col in color_map.get(n, []):
                    if col in tail:
                        return n
            for n in names:
                if names_lower[n] in tail:
                    return n
            for w in words_list[keyword_idx + 1:]:
                for n in names:
                    nl = names_lower[n]
                    if w == nl or w in nl or nl in w:
                        return n
            return None

        # 0) Grasp-verb priority
        grasp_verbs = {"pick", "grab", "grasp", "get", "move", "lift",
                       "take", "fetch", "snatch", "seize", "hold"}
        for i, w in enumerate(words):
            if w in grasp_verbs:
                hit = _match_after(i, words)
                if hit:
                    return hit

        # 1) Colour-based
        color_matches = []
        for n in names:
            for col in color_map.get(n, []):
                if col in txt:
                    color_matches.append(n)
                    break
        if len(color_matches) == 1:
            return color_matches[0]
        if len(color_matches) > 1:
            for n in color_matches:
                if names_lower[n] in txt:
                    return n
            return color_matches[0]

        # 2) Exact substring
        for n in names:
            if names_lower[n] in txt:
                return n

        # 3) Token matching
        for w in words:
            for n in names:
                nl = names_lower[n]
                if w == nl or w in nl or nl in w:
                    return n

        # 4) difflib on whole text
        best = difflib.get_close_matches(txt, [n.lower() for n in names], n=1, cutoff=0.6)
        if best:
            b = best[0]
            for n in names:
                if names_lower[n] == b:
                    return n

        # 5) difflib on individual words
        for w in words:
            best = difflib.get_close_matches(w, [n.lower() for n in names], n=1, cutoff=0.6)
            if best:
                b = best[0]
                for n in names:
                    if names_lower[n] == b:
                        return n
        return None

    # ------------------------------------------------------------------
    # Rotation matrix → quaternion (shared helper)
    # ------------------------------------------------------------------
    def _rotmat_to_quat_shared(m):
        """Convert a 3×3 rotation matrix to [x, y, z, w] quaternion."""
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = 0.5 / math.sqrt(tr + 1.0); wq = 0.25 / s
            return [(m[2,1]-m[1,2])*s, (m[0,2]-m[2,0])*s,
                    (m[1,0]-m[0,1])*s, wq]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
            return [0.25*s, (m[0,1]+m[1,0])/s, (m[0,2]+m[2,0])/s,
                    (m[2,1]-m[1,2])/s]
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
            return [(m[0,1]+m[1,0])/s, 0.25*s, (m[1,2]+m[2,1])/s,
                    (m[0,2]-m[2,0])/s]
        else:
            s = 2.0 * math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
            return [(m[0,2]+m[2,0])/s, (m[1,2]+m[2,1])/s, 0.25*s,
                    (m[1,0]-m[0,1])/s]

    # ------------------------------------------------------------------
    # Null-space IK helper (shared)
    # ------------------------------------------------------------------
    def _ns_ik(pos, orn):
        """Solve IK with null-space bias for consistent arm configuration."""
        ns_lo, ns_hi, ns_rng, ns_rst = [], [], [], []
        for j in movable_joints:
            if j["name"].startswith("joint_"):
                lo, hi = REAL_JOINT_LIMITS.get(
                    j["name"], (j["lower"], j["upper"]))
                ns_lo.append(lo); ns_hi.append(hi)
                ns_rng.append(hi - lo)
                if j["name"] == "joint_2":     ns_rst.append(0.5)
                elif j["name"] == "joint_4":   ns_rst.append(-1.0)
                else:                          ns_rst.append(0.0)
        ik_raw = p.calculateInverseKinematics(
            robot_id, ee_link_index, pos, orn,
            lowerLimits=ns_lo, upperLimits=ns_hi,
            jointRanges=ns_rng, restPoses=ns_rst)
        targets = {}
        for j, ang in zip(movable_joints, ik_raw):
            if j["name"].startswith("joint_"):
                lo, hi = REAL_JOINT_LIMITS.get(
                    j["name"], (j["lower"], j["upper"]))
                targets[j["name"]] = float(max(lo, min(hi, ang)))
        ns_dict = {"lo": ns_lo, "hi": ns_hi, "rng": ns_rng, "rst": ns_rst}
        return targets, ns_dict

    # ------------------------------------------------------------------
    # Look-at orientation helper (shared)
    # ------------------------------------------------------------------
    def _look_at_quat(eye_pos, target_pos):
        """Build an EE quaternion so the camera (local Z) looks at target.

        Convention: X=right, Y=-cam_up, Z=forward.
        """
        fwd = np.array([target_pos[0] - eye_pos[0],
                         target_pos[1] - eye_pos[1],
                         target_pos[2] - eye_pos[2]], dtype=np.float64)
        fn = np.linalg.norm(fwd)
        if fn < 1e-6:
            fwd = np.array([1.0, 0.0, 0.0])
        else:
            fwd = fwd / fn
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(fwd, world_up)) > 0.95:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, world_up)
        right = right / (np.linalg.norm(right) or 1.0)
        up = np.cross(right, fwd)
        R = np.column_stack((right, -up, fwd))
        return _rotmat_to_quat_shared(R)

    # ------------------------------------------------------------------
    # Vision-only grasp pipeline (shared by jarvistest and jarvis)
    # ------------------------------------------------------------------
    def _vision_grasp_pipeline(grasp_target: str):
        """Full vision-only grasp pipeline: locate → point → centre → detect → plan.

        Returns (steps, labels) where *steps* is a list of executable sub-step
        dicts ready for ``_exec_steps_immediate`` and *labels* is human-readable
        descriptions of each step.  Returns (None, None) on failure.

        All spatial data comes purely from camera images — no p.getAABB or
        ground-truth object positions.
        """
        from clip_bbox import _ensure_model, _detect_objects
        _ensure_model()

        VIEW_STANDOFF = 0.30   # metres back from object

        # 1 — Locate object from overhead cameras
        print(f"[GRASP] Locating '{grasp_target}' from overhead cameras...")
        obj_pos = locate_single_object_vision(grasp_target, console)
        if not obj_pos:
            console.write(f"Could not locate '{grasp_target}'.")
            return None, None
        ox, oy, oz = obj_pos["x"], obj_pos["y"], obj_pos["z"]

        # 2 — Point EE camera at the object
        od = math.sqrt(ox*ox + oy*oy) or 1.0
        oax, oay = ox / od, oy / od
        look_pos = [ox - VIEW_STANDOFF * oax,
                    oy - VIEW_STANDOFF * oay,
                    oz + 0.05]
        look_quat = _look_at_quat(look_pos, [ox, oy, oz])
        look_targets, _ = _ns_ik(look_pos, look_quat)

        print(f"[GRASP] Pointing at '{grasp_target}'...")
        _exec_steps_immediate([{"type": "pose", "targets": look_targets}])
        for joint_idx_s, slider_id_s, max_force_s in sliders:
            held_positions[joint_idx_s] = p.getJointState(robot_id, joint_idx_s)[0]

        # 3 — Centering refinement loop (up to 6 iterations)
        CENTRE_TOL_PX = 20
        _consecutive_misses = 0
        for _refine_iter in range(6):
            ref_rgb, ref_depth, _, _ = capture_ee_rgbd(robot_id, ee_link_index)
            rh, rw = ref_depth.shape
            ref_dets = _detect_objects(ref_rgb, [grasp_target])
            ref_det = ref_dets.get(grasp_target, {})
            ref_bbox = ref_det.get("bbox")

            if not ref_bbox:
                _consecutive_misses += 1
                if _consecutive_misses >= 2:
                    print(f"  Centering: re-localizing '{grasp_target}' from overhead...")
                    reloc = locate_single_object_vision(grasp_target)
                    if reloc:
                        ox, oy, oz = reloc["x"], reloc["y"], reloc["z"]
                        od2 = math.sqrt(ox*ox + oy*oy) or 1.0
                        oax, oay = ox / od2, oy / od2
                        look_pos = [ox - VIEW_STANDOFF * oax,
                                    oy - VIEW_STANDOFF * oay,
                                    oz + 0.05]
                    else:
                        print(f"  [GRASP] Centering: '{grasp_target}' not found in any camera")
                        break
                    _consecutive_misses = 0
                else:
                    print(f"  Centering: '{grasp_target}' not visible — backing up")
                    look_pos[0] -= 0.05 * oax
                    look_pos[1] -= 0.05 * oay
                    look_pos[2] += 0.03

                fwd2_q = _look_at_quat(look_pos, [ox, oy, oz])
                lt2, _ = _ns_ik(look_pos, fwd2_q)
                _exec_steps_immediate([{"type": "pose", "targets": lt2}])
                for ji, si, mf in sliders:
                    held_positions[ji] = p.getJointState(robot_id, ji)[0]
                continue

            _consecutive_misses = 0
            rb1, rb2, rb3, rb4 = ref_bbox
            act_u = (rb1 + rb3) / 2.0
            act_v = (rb2 + rb4) / 2.0
            du = act_u - rw / 2.0
            dv = act_v - rh / 2.0
            if abs(du) < CENTRE_TOL_PX and abs(dv) < CENTRE_TOL_PX:
                print(f"  [GRASP] Centred on '{grasp_target}' (err {abs(du):.0f}/{abs(dv):.0f} px)")
                break

            roi = ref_depth[rb2:rb4+1, rb1:rb3+1]
            valid_d = roi[(roi > 0) & np.isfinite(roi)]
            rZ = float(np.median(valid_d)) if valid_d.size > 0 else 0
            if rZ <= 0 or not math.isfinite(rZ):
                break

            r_aspect = rw / rh
            r_hvfov = math.radians(CAM_FOV / 2.0)
            r_hhfov = math.atan(r_aspect * math.tan(r_hvfov))
            wpp_h = (2.0 * rZ * math.tan(r_hhfov)) / rw
            wpp_v = (2.0 * rZ * math.tan(r_hvfov)) / rh
            ee_st = p.getLinkState(robot_id, ee_link_index,
                                   computeForwardKinematics=True)
            ee_pos_ref = np.array(ee_st[4])
            ee_orn_ref = ee_st[5]
            rot_ref = np.array(p.getMatrixFromQuaternion(ee_orn_ref)).reshape(3, 3)
            cam_right = rot_ref[:, 0]
            cam_down = rot_ref[:, 1]
            shift = du * wpp_h * cam_right + dv * wpp_v * cam_down
            new_ee = (ee_pos_ref + shift).tolist()

            obj_est = (new_ee[0] + rZ * rot_ref[:, 2][0],
                       new_ee[1] + rZ * rot_ref[:, 2][1],
                       new_ee[2] + rZ * rot_ref[:, 2][2])
            lq_c = _look_at_quat(new_ee, obj_est)
            ref_tgts, _ = _ns_ik(new_ee, lq_c)
            if ref_tgts:
                _exec_steps_immediate([{"type": "pose", "targets": ref_tgts}])
                for ji, si, mf in sliders:
                    held_positions[ji] = p.getJointState(robot_id, ji)[0]
                print(f"  [GRASP] Centering: shifted {np.linalg.norm(shift)*1000:.1f} mm "
                      f"(du={du:.0f}px dv={dv:.0f}px)")
            else:
                break

        # 4 — Capture EE RGB-D and detect object + grasp
        print("[GRASP] Analysing scene...")
        ee_rgb, ee_depth, view_mat_flat, proj_mat_flat = capture_ee_rgbd(
            robot_id, ee_link_index)
        h, w = ee_depth.shape

        ee_dets = _detect_objects(ee_rgb, [grasp_target])
        det = ee_dets.get(grasp_target, {})
        bbox = det.get("bbox")

        if bbox:
            bx1, by1, bx2, by2 = bbox
            print(f"[GRASP] Object '{grasp_target}' detected in EE camera.")
            bw = bx2 - bx1; bh = by2 - by1
            pad_x = max(int(bw * 0.5), 20)
            pad_y = max(int(bh * 0.5), 20)
            crop_x1 = max(0, bx1 - pad_x)
            crop_y1 = max(0, by1 - pad_y)
            crop_x2 = min(w - 1, bx2 + pad_x)
            crop_y2 = min(h - 1, by2 + pad_y)
            crop_rgb = ee_rgb[crop_y1:crop_y2+1, crop_x1:crop_x2+1].copy()
            crop_depth = ee_depth[crop_y1:crop_y2+1, crop_x1:crop_x2+1].copy()
        else:
            print(f"[GRASP] Grounding DINO couldn't locate '{grasp_target}' — using full view.")
            crop_rgb = ee_rgb
            crop_depth = ee_depth
            crop_x1, crop_y1 = 0, 0

        # 5 — Run GR-ConvNet for grasp detection
        grasps = None
        grasp_source = None
        try:
            grasps = grconvnet_wrapper.detect_grconvnet_grasps(
                crop_rgb, crop_depth, top_k=5)
            if grasps:
                grasp_source = "GR-ConvNet"
        except Exception as e:
            print(f"[GRASP] GR-ConvNet error: {e}")

        if not grasps and bbox:
            try:
                grasps = grconvnet_wrapper.detect_grconvnet_grasps(
                    ee_rgb, ee_depth, top_k=5)
                if grasps:
                    grasp_source = "GR-ConvNet"
                    crop_x1, crop_y1 = 0, 0
            except Exception:
                pass

        if not grasps:
            try:
                import tempfile as _tmpmod
                tmpf = _tmpmod.NamedTemporaryFile(
                    prefix="gq_depth_", suffix=".npy", delete=False)
                np.save(tmpf.name, crop_depth)
                tmpf.close()
                grasps = gqcnn_wrapper.detect_gqcnn_grasps(tmpf.name, None, None)
                if grasps:
                    grasp_source = "GQ-CNN"
            except Exception:
                pass

        if not grasps:
            console.write("No grasp candidates found.")
            return None, None

        best = grasps[0]
        cu, cv = best["pixel"]

        # ---- Grasp visualisation snapshot ----
        try:
            # Pick the image the grasp net actually ran on
            if crop_x1 == 0 and crop_y1 == 0:
                draw_img = ee_rgb          # full-view fallback
            else:
                draw_img = crop_rgb        # DINO crop
            bbox_in_crop = None
            if bbox:
                bbx1, bby1, bbx2, bby2 = bbox
                bbox_in_crop = (bbx1 - crop_x1, bby1 - crop_y1,
                                bbx2 - crop_x1, bby2 - crop_y1)
            overlay = _draw_grasp_overlay(draw_img, grasps,
                                          bbox_in_crop=bbox_in_crop)
            ts = time.strftime("%Y%m%d_%H%M%S")
            snap_name = f"grasp_{grasp_target}_{ts}.png"
            os.makedirs(GRASP_SNAP_DIR, exist_ok=True)
            snap_path = os.path.join(GRASP_SNAP_DIR, snap_name)
            _write_png(snap_path, overlay)
            print(f"[GRASP] Snapshot saved: {snap_path}")
            for i, g in enumerate(grasps):
                q = g.get('quality', 0.0)
                px = g['pixel']
                print(f"  #{i+1}  quality={q:.4f}  pixel=({px[0]:.0f},{px[1]:.0f})  "
                      f"angle={g.get('angle',0):.2f}  width={g.get('width',0):.1f}  "
                      f"[{grasp_source}]")
            os.startfile(snap_path)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[GRASP] Snapshot error (non-fatal): {e}")

        u = cu + crop_x1
        v = cv + crop_y1
        grasp_angle = float(best.get("angle", 0.0))
        grasp_width_px = float(best.get("width", 30.0))

        # 6 — Robust depth (median over bbox)
        if bbox:
            orig_bx1, orig_by1, orig_bx2, orig_by2 = det["bbox"]
            roi = ee_depth[orig_by1:orig_by2+1, orig_bx1:orig_bx2+1]
            valid = roi[(roi > 0) & np.isfinite(roi)]
            Z = float(np.median(valid)) if valid.size > 0 else float(
                ee_depth[int(max(0, min(h-1, round(v)))),
                         int(max(0, min(w-1, round(u))))])
        else:
            Z = float(ee_depth[int(max(0, min(h-1, round(v)))),
                               int(max(0, min(w-1, round(u))))])
        if Z <= 0 or math.isinf(Z) or math.isnan(Z):
            console.write("Could not determine object depth.")
            return None, None
        print(f"  [GRASP] Depth (median over bbox): {Z:.4f} m")

        # 7 — Unproject grasp point to world coordinates
        d_raw = (CAM_FAR * (Z - CAM_NEAR)) / (Z * (CAM_FAR - CAM_NEAR))
        ndc_x = (2.0 * u / w) - 1.0
        ndc_y = 1.0 - (2.0 * v / h)
        ndc_z = 2.0 * d_raw - 1.0

        view_mat = np.array(view_mat_flat, dtype=np.float64).reshape(4, 4).T
        proj_mat = np.array(proj_mat_flat, dtype=np.float64).reshape(4, 4).T
        vp_inv = np.linalg.inv(proj_mat @ view_mat)
        clip_pt = np.array([ndc_x, ndc_y, ndc_z, 1.0])
        world_h = vp_inv @ clip_pt
        world_h /= world_h[3]
        gx, gy, gz = float(world_h[0]), float(world_h[1]), float(world_h[2])

        # 8 — Build side-grasp orientation
        dx, dy = gx, gy
        d = math.sqrt(dx*dx + dy*dy) or 1.0
        ax, ay = dx / d, dy / d

        lat = np.array([-ay, ax, 0.0], dtype=np.float64)
        lat_norm = np.linalg.norm(lat)
        if lat_norm < 1e-6:
            lat = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            lat = lat / lat_norm

        if abs(grasp_angle) > 0.01:
            ca, sa = math.cos(grasp_angle), math.sin(grasp_angle)
            lx, ly = lat[0], lat[1]
            lat[0] = ca * lx - sa * ly
            lat[1] = sa * lx + ca * ly

        # Object size estimation from bbox + depth
        aspect = w / h
        half_vfov = math.radians(CAM_FOV / 2.0)
        half_hfov = math.atan(aspect * math.tan(half_vfov))
        world_per_px = (2.0 * Z * math.tan(half_hfov)) / w

        if bbox:
            orig_bx1_s, orig_by1_s, orig_bx2_s, orig_by2_s = det["bbox"]
            bbox_w_px = orig_bx2_s - orig_bx1_s
            bbox_h_px = orig_by2_s - orig_by1_s
            obj_width = bbox_w_px * world_per_px
            obj_height = bbox_h_px * world_per_px
        else:
            obj_width = grasp_width_px * world_per_px
            obj_height = obj_width

        obj_radius = obj_width / 2.0
        cx = gx + obj_radius * ax
        cy = gy + obj_radius * ay
        cz = max(TABLE_Z, gz)

        print(f"  [GRASP] Surface  ({gx:.3f}, {gy:.3f}, {gz:.3f})  "
              f"obj ~{obj_width:.3f}m wide, ~{obj_height:.3f}m tall")
        print(f"  [GRASP] Centroid ({cx:.3f}, {cy:.3f}, {cz:.3f})")

        # 9 — Pre-shape gripper width
        ROBOTIQ_MAX_OPENING = 0.085
        grasp_width_m = grasp_width_px * world_per_px
        pre_grip = min(GRIP_OPEN,
                       (grasp_width_m / ROBOTIQ_MAX_OPENING) * GRIP_OPEN * 1.3)
        pre_grip = max(pre_grip, 0.2)
        print(f"  [GRASP] Gripper pre-shape: {pre_grip:.2f} rad "
              f"(obj ~{grasp_width_m*1000:.0f} mm wide)")

        # 10 — IK for pre-grasp / contact / lift
        APPROACH_BACK = 0.12
        pre_pos = [cx - APPROACH_BACK * ax,
                   cy - APPROACH_BACK * ay,
                   cz + 0.04]
        contact_pos = [cx, cy, cz]
        lift_pos = [cx, cy, LIFT_Z]

        fwd_la = np.array([cx - pre_pos[0], cy - pre_pos[1],
                           cz - pre_pos[2]], dtype=np.float64)
        fwd_la_norm = np.linalg.norm(fwd_la)
        if fwd_la_norm > 1e-6:
            fwd_la = fwd_la / fwd_la_norm
        else:
            fwd_la = np.array([ax, ay, 0.0], dtype=np.float64)
        r_vec = lat / (np.linalg.norm(lat) or 1.0)
        if abs(np.dot(fwd_la, r_vec)) > 0.999:
            r_vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u_vec = np.cross(fwd_la, r_vec)
        u_vec = u_vec / (np.linalg.norm(u_vec) or 1.0)
        R_mat = np.column_stack((r_vec, u_vec, fwd_la))
        quat = _rotmat_to_quat_shared(R_mat)

        pre_targets, ns_dict = _ns_ik(pre_pos, quat)
        contact_targets, _ = _ns_ik(contact_pos, quat)
        lift_targets, _ = _ns_ik(lift_pos, quat)

        if not pre_targets or not contact_targets or not lift_targets:
            console.write("IK failed for grasp waypoints.")
            return None, None

        # 11 — Build executable step list
        obj_label = grasp_target
        det_info = "Grounding DINO + " if bbox else ""
        steps = [
            {"type": "gripper", "target": pre_grip, "label": "Open gripper"},
            {"type": "pose", "targets": pre_targets, "label": f"Approach {obj_label}"},
            {"type": "servo_contact",
             "nominal_targets": contact_targets,
             "quat": quat,
             "grasp_target": grasp_target,
             "ax": ax, "ay": ay, "cz": cz,
             "_cx": cx, "_cy": cy,
             "_null_space": ns_dict,
             "label": f"Align {obj_label}"},
            {"type": "gripper_force", "force_threshold": 5.0, "label": f"Grip {obj_label}"},
            {"type": "pose", "targets": lift_targets, "label": f"Lift {obj_label}"},
        ]
        labels = [
            "Open gripper",
            f"Approach {obj_label}",
            f"Align {obj_label}",
            f"Grip {obj_label}",
            f"Lift {obj_label}",
        ]

        print(f"[GRASP] Plan for '{obj_label}' ({det_info}{grasp_source}): {len(steps)} steps")

        return steps, labels

    def _exec_steps_immediate(steps_list):
        """Execute primitive sub-steps synchronously (blocking convergence loop)."""
        nonlocal held_gripper
        for st in steps_list:
            stype = st.get("type", "joint")
            if stype == "pose":
                targets = st["targets"]
                active_idxs = []
                for jname, (jidx, max_force) in joint_lookup.items():
                    if jname in targets:
                        tgt = targets[jname]
                        active_idxs.append((jname, jidx, tgt, max_force))
                        p.setJointMotorControl2(bodyUniqueId=robot_id,
                                                jointIndex=jidx,
                                                controlMode=p.POSITION_CONTROL,
                                                targetPosition=tgt,
                                                force=max_force,
                                                maxVelocity=AI_MAX_VELOCITY)
                    else:
                        cur = p.getJointState(robot_id, joint_lookup[jname][0])[0]
                        p.setJointMotorControl2(bodyUniqueId=robot_id,
                                                jointIndex=joint_lookup[jname][0],
                                                controlMode=p.POSITION_CONTROL,
                                                targetPosition=cur,
                                                force=joint_lookup[jname][1])
                grip_hold = held_gripper if held_gripper is not None else p.readUserDebugParameter(gripper_slider_id)
                p.setJointMotorControl2(gripper_info["id"], gripper_info["left_idx"],
                                        p.POSITION_CONTROL, targetPosition=grip_hold, force=50)
                p.setJointMotorControl2(gripper_info["id"], gripper_info["right_idx"],
                                        p.POSITION_CONTROL, targetPosition=-grip_hold, force=50)
                settle = 0
                ticks = 0
                while settle < 20 and ticks < 2400:
                    p.stepSimulation()
                    max_err = 0.0
                    for (jname, jidx, tgt, _) in active_idxs:
                        cur = p.getJointState(robot_id, jidx)[0]
                        max_err = max(max_err, abs(cur - tgt))
                    if max_err < POSITION_TOLERANCE:
                        settle += 1
                    else:
                        settle = 0
                    ticks += 1
                    time.sleep(1.0 / 240.0)
            elif stype == "gripper":
                tgt = st["target"]
                g_id = gripper_info["id"]
                p.setJointMotorControl2(g_id, gripper_info["left_idx"], p.POSITION_CONTROL,
                                        targetPosition=tgt, force=20, maxVelocity=AI_MAX_VELOCITY)
                p.setJointMotorControl2(g_id, gripper_info["right_idx"], p.POSITION_CONTROL,
                                        targetPosition=-tgt, force=20, maxVelocity=AI_MAX_VELOCITY)
                settle = 0
                ticks = 0
                while settle < 20 and ticks < 960:
                    p.stepSimulation()
                    cur = p.getJointState(g_id, gripper_info["left_idx"])[0]
                    if abs(cur - tgt) < POSITION_TOLERANCE:
                        settle += 1
                    else:
                        settle = 0
                    ticks += 1
                    time.sleep(1.0 / 240.0)
            elif stype == "dwell":
                dwell_ticks = st.get("ticks", 480)
                for _ in range(dwell_ticks):
                    p.stepSimulation()
                    time.sleep(1.0 / 240.0)
            elif stype == "gripper_force":
                # Close gripper in two phases:
                #   Phase A (approach): close quickly, ignore torque noise
                #   Phase B (squeeze): close slowly, stop on sustained stall
                g_id = gripper_info["id"]
                l_idx = gripper_info["left_idx"]
                r_idx = gripper_info["right_idx"]
                force_threshold = st.get("force_threshold", 5.0)
                max_ticks = st.get("max_ticks", 1200)

                cur_grip = p.getJointState(g_id, l_idx)[0]
                FAST_STEP = 0.015       # rad/tick — fast approach
                SLOW_STEP = 0.003       # rad/tick — fine squeeze
                MIN_TICKS_BEFORE_CHECK = 120  # ignore torque during first ~0.5s
                STALL_WINDOW = 15       # consecutive stalled ticks to confirm grip
                stall_count = 0
                contact_detected = False
                phase = "approach"

                for tick in range(max_ticks):
                    step_size = FAST_STEP if phase == "approach" else SLOW_STEP
                    cur_grip = max(cur_grip - step_size, GRIP_CLOSE)

                    p.setJointMotorControl2(g_id, l_idx, p.POSITION_CONTROL,
                                            targetPosition=cur_grip, force=50,
                                            maxVelocity=2.0)
                    p.setJointMotorControl2(g_id, r_idx, p.POSITION_CONTROL,
                                            targetPosition=-cur_grip, force=50,
                                            maxVelocity=2.0)
                    for jname_f, (jidx_f, mf_f) in joint_lookup.items():
                        cur_j = p.getJointState(robot_id, jidx_f)[0]
                        p.setJointMotorControl2(robot_id, jidx_f, p.POSITION_CONTROL,
                                                targetPosition=cur_j, force=mf_f)
                    p.stepSimulation()
                    time.sleep(1.0 / 240.0)

                    left_state = p.getJointState(g_id, l_idx)
                    actual_pos = left_state[0]
                    joint_motor_torque = abs(left_state[3])
                    pos_err = abs(actual_pos - cur_grip)

                    # Phase A → B: switch on first resistance after warm-up
                    if phase == "approach" and tick > MIN_TICKS_BEFORE_CHECK:
                        if joint_motor_torque > force_threshold or pos_err > 0.01:
                            phase = "squeeze"

                    # Phase B: count consecutive stalled ticks
                    if phase == "squeeze":
                        if pos_err > 0.005:
                            stall_count += 1
                        else:
                            stall_count = 0
                        if stall_count >= STALL_WINDOW:
                            contact_detected = True
                            break

                    if cur_grip <= GRIP_CLOSE:
                        break

                # Over-squeeze: push 15 extra SLOW_STEPs past the stall point
                if contact_detected:
                    for _osq in range(15):
                        cur_grip = max(cur_grip - SLOW_STEP, GRIP_CLOSE)
                        p.setJointMotorControl2(g_id, l_idx, p.POSITION_CONTROL,
                                                targetPosition=cur_grip, force=50,
                                                maxVelocity=1.0)
                        p.setJointMotorControl2(g_id, r_idx, p.POSITION_CONTROL,
                                                targetPosition=-cur_grip, force=50,
                                                maxVelocity=1.0)
                        for jname_f, (jidx_f, mf_f) in joint_lookup.items():
                            cur_j = p.getJointState(robot_id, jidx_f)[0]
                            p.setJointMotorControl2(robot_id, jidx_f, p.POSITION_CONTROL,
                                                    targetPosition=cur_j, force=mf_f)
                        p.stepSimulation()
                        time.sleep(1.0 / 240.0)

                # Longer dwell to let the grip fully settle
                for _ in range(240):
                    p.setJointMotorControl2(g_id, l_idx, p.POSITION_CONTROL,
                                            targetPosition=cur_grip, force=50)
                    p.setJointMotorControl2(g_id, r_idx, p.POSITION_CONTROL,
                                            targetPosition=-cur_grip, force=50)
                    for jname_f, (jidx_f, mf_f) in joint_lookup.items():
                        cur_j = p.getJointState(robot_id, jidx_f)[0]
                        p.setJointMotorControl2(robot_id, jidx_f, p.POSITION_CONTROL,
                                                targetPosition=cur_j, force=mf_f)
                    p.stepSimulation()
                    time.sleep(1.0 / 240.0)

                if contact_detected:
                    print(f"  [GRIP] secured at {cur_grip:.3f} rad")
                else:
                    print(f"  [GRIP] closed to {cur_grip:.3f} rad (no sustained contact)")
                held_gripper = cur_grip
            elif stype == "creep_contact":
                # Slowly creep the EE forward along the approach axis
                # until ANY gripper link touches a non-robot body, then
                # stop immediately so the fingertips are perfectly positioned.
                g_id = gripper_info["id"]
                c_ax = st.get("ax", 0.0)
                c_ay = st.get("ay", 0.0)
                c_quat = st.get("quat")
                c_max_mm = st.get("max_mm", 30)
                c_max_ticks = int(c_max_mm)   # ~1 mm per tick
                CREEP_STEP = 0.001            # 1 mm per tick

                num_g_links = p.getNumJoints(g_id)
                gripper_link_set = set(range(-1, num_g_links))

                ee_st = p.getLinkState(robot_id, ee_link_index,
                                       computeForwardKinematics=True)
                cx, cy, cz = ee_st[4]

                contact_hit = False
                for ctick in range(c_max_ticks):
                    # Advance EE position 1 mm forward
                    cx += CREEP_STEP * c_ax
                    cy += CREEP_STEP * c_ay
                    # Solve IK for the new position (keep orientation)
                    kw = dict(bodyUniqueId=robot_id,
                              endEffectorLinkIndex=ee_link_index,
                              targetPosition=[cx, cy, cz],
                              maxNumIterations=50,
                              residualThreshold=1e-4)
                    if c_quat:
                        kw["targetOrientation"] = c_quat
                    ik_ans = p.calculateInverseKinematics(**kw)
                    # Drive joints
                    for j, ang in zip(movable_joints, ik_ans):
                        if j["name"].startswith("joint_"):
                            jidx = j["index"]
                            lo, hi = REAL_JOINT_LIMITS.get(j["name"],
                                                           (j["lower"], j["upper"]))
                            tgt = max(lo, min(hi, ang))
                            p.setJointMotorControl2(robot_id, jidx,
                                                    p.POSITION_CONTROL,
                                                    targetPosition=tgt,
                                                    force=50,
                                                    maxVelocity=0.3)
                    # Hold gripper open
                    grip_hold = held_gripper if held_gripper is not None else GRIP_OPEN
                    p.setJointMotorControl2(g_id, gripper_info["left_idx"],
                                            p.POSITION_CONTROL,
                                            targetPosition=grip_hold, force=20)
                    p.setJointMotorControl2(g_id, gripper_info["right_idx"],
                                            p.POSITION_CONTROL,
                                            targetPosition=-grip_hold, force=20)
                    p.stepSimulation()
                    time.sleep(1.0 / 240.0)

                    # Check for contact between any gripper link and world
                    contacts = p.getContactPoints(bodyA=g_id)
                    for cp in contacts:
                        other_body = cp[2]  # bodyB
                        link_a = cp[3]      # link index on gripper
                        if other_body != robot_id and link_a in gripper_link_set:
                            contact_hit = True
                            break
                    if contact_hit:
                        break

                if contact_hit:
                    print(f"  [CREEP] contact after {ctick+1} mm")
                else:
                    print(f"  [CREEP] no contact in {c_max_mm} mm, proceeding")

            elif stype == "grip_verify":
                # Micro-lift test: raise EE 2 cm, check if the object slipped
                # by seeing if the gripper fingers opened, re-grip if needed.
                g_id = gripper_info["id"]
                l_idx = gripper_info["left_idx"]
                r_idx = gripper_info["right_idx"]
                vx = st.get("x", 0.0)
                vy = st.get("y", 0.0)
                vz = st.get("z", 1.24)
                lift_delta = st.get("lift_delta", 0.02)
                max_retries = st.get("max_retries", 2)
                SLIP_THRESHOLD = 0.02   # rad — if fingers open this much, object slipped

                grip_ok = False
                for attempt in range(max_retries + 1):
                    pre_grip = p.getJointState(g_id, l_idx)[0]

                    # Get current EE position and lift up by lift_delta
                    ee_st = p.getLinkState(robot_id, ee_link_index,
                                           computeForwardKinematics=True)
                    lx0, ly0, lz0 = ee_st[4]
                    lz_up = lz0 + lift_delta

                    # IK for lifted position (keep current orientation)
                    ik_up = p.calculateInverseKinematics(
                        robot_id, ee_link_index,
                        targetPosition=[lx0, ly0, lz_up],
                        maxNumIterations=50, residualThreshold=1e-4)
                    for j, ang in zip(movable_joints, ik_up):
                        if j["name"].startswith("joint_"):
                            jidx = j["index"]
                            lo, hi = REAL_JOINT_LIMITS.get(j["name"],
                                                           (j["lower"], j["upper"]))
                            tgt = max(lo, min(hi, ang))
                            p.setJointMotorControl2(robot_id, jidx,
                                                    p.POSITION_CONTROL,
                                                    targetPosition=tgt,
                                                    force=50, maxVelocity=0.5)
                    # Hold gripper tight during lift
                    for _lt in range(360):  # ~1.5s
                        p.setJointMotorControl2(g_id, l_idx, p.POSITION_CONTROL,
                                                targetPosition=pre_grip, force=50)
                        p.setJointMotorControl2(g_id, r_idx, p.POSITION_CONTROL,
                                                targetPosition=-pre_grip, force=50)
                        p.stepSimulation()
                        time.sleep(1.0 / 240.0)

                    # Check if fingers opened (object slipped out)
                    post_grip = p.getJointState(g_id, l_idx)[0]
                    slipped = (post_grip - pre_grip) > SLIP_THRESHOLD

                    if not slipped:
                        grip_ok = True
                        print(f"  [GRIP_VERIFY] hold confirmed (delta={post_grip - pre_grip:.4f} rad)")
                        break

                    print(f"  [GRIP_VERIFY] slip detected (delta={post_grip - pre_grip:.4f} rad), attempt {attempt+1}/{max_retries+1}")

                    if attempt < max_retries:
                        # Lower back to grasp height
                        ik_dn = p.calculateInverseKinematics(
                            robot_id, ee_link_index,
                            targetPosition=[lx0, ly0, lz0],
                            maxNumIterations=50, residualThreshold=1e-4)
                        for j, ang in zip(movable_joints, ik_dn):
                            if j["name"].startswith("joint_"):
                                jidx = j["index"]
                                lo, hi = REAL_JOINT_LIMITS.get(j["name"],
                                                               (j["lower"], j["upper"]))
                                tgt = max(lo, min(hi, ang))
                                p.setJointMotorControl2(robot_id, jidx,
                                                        p.POSITION_CONTROL,
                                                        targetPosition=tgt,
                                                        force=50, maxVelocity=0.5)
                        for _ld in range(360):
                            p.stepSimulation()
                            time.sleep(1.0 / 240.0)

                        # Re-grip: open slightly then force-close again
                        reopen = pre_grip + 0.1  # open a bit wider
                        p.setJointMotorControl2(g_id, l_idx, p.POSITION_CONTROL,
                                                targetPosition=reopen, force=50)
                        p.setJointMotorControl2(g_id, r_idx, p.POSITION_CONTROL,
                                                targetPosition=-reopen, force=50)
                        for _ro in range(120):
                            p.stepSimulation()
                            time.sleep(1.0 / 240.0)

                        # Force-close again (squeeze phase only)
                        regrip = reopen
                        SLOW_STEP = 0.003
                        rg_stall = 0
                        for _rg in range(600):
                            regrip = max(regrip - SLOW_STEP, GRIP_CLOSE)
                            p.setJointMotorControl2(g_id, l_idx, p.POSITION_CONTROL,
                                                    targetPosition=regrip, force=50,
                                                    maxVelocity=1.0)
                            p.setJointMotorControl2(g_id, r_idx, p.POSITION_CONTROL,
                                                    targetPosition=-regrip, force=50,
                                                    maxVelocity=1.0)
                            for jname_f, (jidx_f, mf_f) in joint_lookup.items():
                                cur_j = p.getJointState(robot_id, jidx_f)[0]
                                p.setJointMotorControl2(robot_id, jidx_f, p.POSITION_CONTROL,
                                                        targetPosition=cur_j, force=mf_f)
                            p.stepSimulation()
                            time.sleep(1.0 / 240.0)
                            rg_pos = p.getJointState(g_id, l_idx)[0]
                            if abs(rg_pos - regrip) > 0.005:
                                rg_stall += 1
                            else:
                                rg_stall = 0
                            if rg_stall >= 15:
                                break
                        # Over-squeeze
                        for _osq in range(5):
                            regrip = max(regrip - SLOW_STEP, GRIP_CLOSE)
                            p.setJointMotorControl2(g_id, l_idx, p.POSITION_CONTROL,
                                                    targetPosition=regrip, force=50,
                                                    maxVelocity=1.0)
                            p.setJointMotorControl2(g_id, r_idx, p.POSITION_CONTROL,
                                                    targetPosition=-regrip, force=50,
                                                    maxVelocity=1.0)
                            p.stepSimulation()
                            time.sleep(1.0 / 240.0)
                        held_gripper = regrip
                        print(f"  [GRIP_VERIFY] re-gripped at {regrip:.3f} rad")

                if not grip_ok:
                    print(f"  [GRIP_VERIFY] WARNING: grip may be unreliable after {max_retries+1} attempts")
                held_gripper = p.getJointState(g_id, l_idx)[0]

            elif stype == "deliver_swing":
                # Drive toward tray position; release INSTANTLY on collision
                # with any body that wasn't already in contact (person / tray).
                targets = st["targets"]
                release_target = st.get("release_target", 1.2)
                g_id = gripper_info["id"]
                l_idx = gripper_info["left_idx"]
                r_idx = gripper_info["right_idx"]

                # Snapshot which bodies are already contacting the gripper
                # (held object, its cap / constraint children, etc.)
                pre_contacts = set()
                for c in p.getContactPoints(bodyA=g_id):
                    other = c[2] if c[1] == g_id else c[1]
                    pre_contacts.add(other)
                pre_contacts.add(robot_id)   # never trigger on self
                pre_contacts.add(g_id)

                # Also snapshot robot-arm contacts
                pre_robot = set()
                for c in p.getContactPoints(bodyA=robot_id):
                    other = c[2] if c[1] == robot_id else c[1]
                    pre_robot.add(other)
                pre_robot.add(g_id)

                # Set arm motor targets
                active_idxs = []
                for jname, (jidx, max_force) in joint_lookup.items():
                    if jname in targets:
                        tgt = targets[jname]
                        active_idxs.append((jname, jidx, tgt, max_force))
                        p.setJointMotorControl2(robot_id, jidx,
                                                p.POSITION_CONTROL,
                                                targetPosition=tgt,
                                                force=max(max_force, 50.0),
                                                maxVelocity=AI_MAX_VELOCITY)
                    else:
                        cur = p.getJointState(robot_id, jidx)[0]
                        p.setJointMotorControl2(robot_id, jidx,
                                                p.POSITION_CONTROL,
                                                targetPosition=cur,
                                                force=max_force)

                grip_hold = held_gripper if held_gripper is not None else 0.0
                released = False
                settle = 0
                for tick in range(2400):
                    # Hold gripper tight
                    p.setJointMotorControl2(g_id, l_idx, p.POSITION_CONTROL,
                                            targetPosition=grip_hold, force=50)
                    p.setJointMotorControl2(g_id, r_idx, p.POSITION_CONTROL,
                                            targetPosition=-grip_hold, force=50)
                    p.stepSimulation()
                    time.sleep(1.0 / 240.0)

                    # Convergence check
                    max_err = 0.0
                    for (jn, ji, tg, _) in active_idxs:
                        cur = p.getJointState(robot_id, ji)[0]
                        max_err = max(max_err, abs(cur - tg))
                    if max_err < POSITION_TOLERANCE:
                        settle += 1
                    else:
                        settle = 0

                    # --- Collision check (gripper) ---
                    for c in p.getContactPoints(bodyA=g_id):
                        other = c[2] if c[1] == g_id else c[1]
                        if other not in pre_contacts:
                            released = True
                            break
                    if released:
                        break
                    # --- Collision check (robot arm) ---
                    for c in p.getContactPoints(bodyA=robot_id):
                        other = c[2] if c[1] == robot_id else c[1]
                        if other not in pre_robot:
                            released = True
                            break
                    if released:
                        break

                    if settle >= 20:
                        break

                # Release gripper
                p.setJointMotorControl2(g_id, l_idx, p.POSITION_CONTROL,
                                        targetPosition=release_target, force=50)
                p.setJointMotorControl2(g_id, r_idx, p.POSITION_CONTROL,
                                        targetPosition=-release_target, force=50)
                for _ in range(180):   # brief dwell for object to drop
                    p.stepSimulation()
                    time.sleep(1.0 / 240.0)
                held_gripper = release_target

                if released:
                    print("  [DELIVER] Contact detected — released on tray")
                else:
                    print("  [DELIVER] Reached tray — released")

            elif stype == "servo_contact":
                # Visual servoing: re-capture from current pre-grasp pose,
                # re-detect the object, compute pixel error vs expected
                # centre, convert to world-space correction, then drive
                # to the corrected contact pose.
                nom_targets = st["nominal_targets"]
                s_quat = st["quat"]
                s_target = st["grasp_target"]
                s_ax = st["ax"]
                s_ay = st["ay"]
                s_cz = st["cz"]

                # 1) Re-capture EE RGB-D from current pose
                s_rgb, s_depth, s_view, s_proj = capture_ee_rgbd(robot_id, ee_link_index)
                s_h, s_w = s_depth.shape

                # 2) Re-detect the target with Grounding DINO
                from clip_bbox import _ensure_model, _detect_objects
                _ensure_model()
                s_dets = _detect_objects(s_rgb, [s_target])
                s_det = s_dets.get(s_target, {})
                s_bbox = s_det.get("bbox")

                correction = np.zeros(3)
                if s_bbox:
                    # Save annotated EE image showing what we're targeting
                    from clip_bbox import annotate_image
                    _servo_dets = {s_target: {"visible": True, "bbox": s_bbox,
                                             "score": s_det.get("score", 0.0)}}
                    _servo_ann = annotate_image(s_rgb, _servo_dets,
                                               {s_target: (0, 255, 0)})
                    os.makedirs(SNAP_DIR, exist_ok=True)
                    _servo_path = os.path.join(SNAP_DIR, f"servo_{s_target}.png")
                    _write_png(_servo_path, _servo_ann)
                    print(f"  [SERVO] bbox saved: {_servo_path}")

                    sb1, sb2, sb3, sb4 = s_bbox
                    # Expected pixel: image centre (we pointed right at it)
                    exp_u, exp_v = s_w / 2.0, s_h / 2.0
                    # Actual pixel: bbox centre
                    act_u = (sb1 + sb3) / 2.0
                    act_v = (sb2 + sb4) / 2.0
                    # Pixel error
                    du = act_u - exp_u
                    dv = act_v - exp_v

                    # Median depth over detected bbox for robust Z
                    roi = s_depth[sb2:sb4+1, sb1:sb3+1]
                    valid_d = roi[(roi > 0) & np.isfinite(roi)]
                    s_Z = float(np.median(valid_d)) if valid_d.size > 0 else float(
                        s_depth[int(s_h/2), int(s_w/2)])

                    if s_Z > 0 and math.isfinite(s_Z):
                        # Convert pixel error to world delta
                        aspect = s_w / s_h
                        half_vfov = math.radians(CAM_FOV / 2.0)
                        half_hfov = math.atan(aspect * math.tan(half_vfov))
                        wpp = (2.0 * s_Z * math.tan(half_hfov)) / s_w
                        wpp_v = (2.0 * s_Z * math.tan(half_vfov)) / s_h

                        # EE camera axes: we need to map pixel delta
                        # to world delta through the camera frame
                        ee_st = p.getLinkState(robot_id, ee_link_index,
                                               computeForwardKinematics=True)
                        ee_orn = ee_st[5]
                        rot = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
                        cam_right = rot[:, 0]   # local X
                        cam_down = rot[:, 1]    # local Y (down in image)

                        # du > 0 means object is to the right in image
                        # dv > 0 means object is lower in image
                        correction = (du * wpp * cam_right
                                      + dv * wpp_v * cam_down)

                        err_mm = np.linalg.norm(correction) * 1000
                        print(f"  [SERVO] correction: {err_mm:.1f} mm "
                              f"(du={du:.1f}px, dv={dv:.1f}px)")
                    else:
                        print("  [SERVO] depth invalid, using nominal pose")
                else:
                    print(f"  [SERVO] lost '{s_target}', using nominal pose")

                # ---- GR-ConvNet grasp visualisation snapshot ----
                try:
                    if s_bbox:
                        _sb1, _sb2, _sb3, _sb4 = s_bbox
                        _bw = _sb3 - _sb1; _bh = _sb4 - _sb2
                        _px = max(int(_bw * 0.5), 20)
                        _py = max(int(_bh * 0.5), 20)
                        _cx1 = max(0, _sb1 - _px)
                        _cy1 = max(0, _sb2 - _py)
                        _cx2 = min(s_w - 1, _sb3 + _px)
                        _cy2 = min(s_h - 1, _sb4 + _py)
                        _gcrop_rgb = s_rgb[_cy1:_cy2+1, _cx1:_cx2+1].copy()
                        _gcrop_dep = s_depth[_cy1:_cy2+1, _cx1:_cx2+1].copy()
                        _bbox_in_crop = (_sb1 - _cx1, _sb2 - _cy1,
                                         _sb3 - _cx1, _sb4 - _cy1)
                    else:
                        _gcrop_rgb = s_rgb.copy()
                        _gcrop_dep = s_depth.copy()
                        _bbox_in_crop = None

                    _gts = time.strftime("%Y%m%d_%H%M%S")
                    os.makedirs(GRASP_SNAP_DIR, exist_ok=True)

                    _ggrasps = None
                    try:
                        _ggrasps = grconvnet_wrapper.detect_grconvnet_grasps(
                            _gcrop_rgb, _gcrop_dep, top_k=5)
                    except Exception as _grce:
                        console.write(f"  [GRASP] GR-ConvNet error: {_grce}")

                    if _ggrasps:
                        _govl = _draw_grasp_overlay(
                            _gcrop_rgb, _ggrasps, bbox_in_crop=_bbox_in_crop)
                        _gpath = os.path.join(
                            GRASP_SNAP_DIR, f"grasp_{s_target}_{_gts}.png")
                        _write_png(_gpath, _govl)
                        console.write(f"  [GRASP] Snapshot saved → {_gpath}")
                        for _gi, _gg in enumerate(_ggrasps):
                            _gq = _gg.get('quality', 0.0)
                            _gpx = _gg['pixel']
                            console.write(f"    #{_gi+1}  q={_gq:.4f}  px=({_gpx[0]:.0f},{_gpx[1]:.0f})  "
                                          f"a={_gg.get('angle',0):.2f}  w={_gg.get('width',0):.1f}")
                        os.startfile(_gpath)
                    else:
                        # Still save raw EE crop so user can see what the camera saw
                        _gpath = os.path.join(
                            GRASP_SNAP_DIR, f"grasp_{s_target}_{_gts}_nocandidates.png")
                        _write_png(_gpath, _gcrop_rgb)
                        console.write(f"  [GRASP] No candidates — raw crop saved → {_gpath}")
                        os.startfile(_gpath)
                except Exception as _ge:
                    console.write(f"  [GRASP] Snapshot error: {_ge}")

                # 3) Compute corrected contact position
                # Apply the servo correction to the centroid position
                # and re-solve IK for the corrected contact.
                nom_cx = st.get("_cx", None)
                nom_cy = st.get("_cy", None)
                if nom_cx is not None and nom_cy is not None:
                    new_contact = [nom_cx + correction[0],
                                   nom_cy + correction[1],
                                   s_cz + correction[2]]
                    # Use null-space IK to keep a consistent arm
                    # configuration (7-DOF arm is redundant).
                    s_ns = st.get("_null_space")
                    if s_ns:
                        ik_corr = p.calculateInverseKinematics(
                            robot_id, ee_link_index, new_contact, s_quat,
                            lowerLimits=s_ns["lo"],
                            upperLimits=s_ns["hi"],
                            jointRanges=s_ns["rng"],
                            restPoses=s_ns["rst"])
                    else:
                        ik_corr = p.calculateInverseKinematics(
                            robot_id, ee_link_index, new_contact, s_quat)
                    corr_targets = {}
                    for j, angle_val in zip(movable_joints, ik_corr):
                        if j["name"].startswith("joint_"):
                            lo, hi = REAL_JOINT_LIMITS.get(j["name"],
                                                           (j["lower"], j["upper"]))
                            corr_targets[j["name"]] = float(max(lo, min(hi, angle_val)))
                else:
                    corr_targets = dict(nom_targets)

                # 4) Execute the corrected contact pose
                active_idxs = []
                for jname, (jidx, max_force) in joint_lookup.items():
                    if jname in corr_targets:
                        tgt = corr_targets[jname]
                        active_idxs.append((jname, jidx, tgt, max_force))
                        p.setJointMotorControl2(bodyUniqueId=robot_id,
                                                jointIndex=jidx,
                                                controlMode=p.POSITION_CONTROL,
                                                targetPosition=tgt,
                                                force=max_force,
                                                maxVelocity=AI_MAX_VELOCITY)
                    else:
                        cur = p.getJointState(robot_id, jidx)[0]
                        p.setJointMotorControl2(bodyUniqueId=robot_id,
                                                jointIndex=jidx,
                                                controlMode=p.POSITION_CONTROL,
                                                targetPosition=cur, force=max_force)
                grip_hold = held_gripper if held_gripper is not None else p.readUserDebugParameter(gripper_slider_id)
                p.setJointMotorControl2(gripper_info["id"], gripper_info["left_idx"],
                                        p.POSITION_CONTROL, targetPosition=grip_hold, force=20)
                p.setJointMotorControl2(gripper_info["id"], gripper_info["right_idx"],
                                        p.POSITION_CONTROL, targetPosition=-grip_hold, force=20)
                # Proportional speed: start fast, creep near contact
                RAMP_THRESHOLD = 0.5   # rad — below this, start slowing
                MIN_VELOCITY = 0.1     # rad/s — minimum approach speed
                settle = 0
                ticks = 0
                while settle < 20 and ticks < 2400:
                    p.stepSimulation()
                    max_err = 0.0
                    for (jname, jidx, tgt, _) in active_idxs:
                        cur = p.getJointState(robot_id, jidx)[0]
                        max_err = max(max_err, abs(cur - tgt))
                    # Scale velocity proportionally to remaining error
                    vel = max(MIN_VELOCITY,
                              AI_MAX_VELOCITY * min(1.0, max_err / RAMP_THRESHOLD))
                    for (jname, jidx, tgt, mf) in active_idxs:
                        p.setJointMotorControl2(robot_id, jidx,
                                                p.POSITION_CONTROL,
                                                targetPosition=tgt, force=mf,
                                                maxVelocity=vel)
                    if max_err < POSITION_TOLERANCE:
                        settle += 1
                    else:
                        settle = 0
                    ticks += 1
                    time.sleep(1.0 / 240.0)

    # ---- Main simulation loop (while-loop + root.update) ----
    print("\n[SIM] Running — use the GUI sliders to move joints. Close the console window to exit.\n")
    try:
        while console.alive and p.isConnected():
            # ---- Process console commands ----
            while console.has_command():
                raw = console.pop_command()
                cmd_lower = raw.lower().strip()
                # --- jarvisplan — LLM task-planning test (no vision/execution) ---
                if cmd_lower.startswith("jarvisplan"):
                    try:
                        req = raw[len("jarvisplan"):].strip()
                        if not req:
                            console.write("Usage: jarvisplan <request>")
                            console.write("  Tests LLM planning for ambiguous tasks.")
                            console.write("  Example: jarvisplan make me tea")
                            continue
                        console.write(f"[JARVISPLAN] Request: '{req}'")
                        result = call_jarvisplan(req, console)
                        if result is None:
                            console.write("[JARVISPLAN] Planning failed.")
                        elif "questions" in result:
                            qs = result["questions"]
                            console.write(f"\nJarvis needs to ask {len(qs)} question(s):")
                            for qi, q in enumerate(qs, 1):
                                console.write(f"  {qi}. {q}")
                            console.write("\nRe-run with answers: jarvisplan <original request>, <your answers>")
                        elif "plan" in result:
                            plan = result["plan"]
                            console.write(f"\nJarvis plan ({len(plan)} steps):")
                            for pi, prim in enumerate(plan, 1):
                                cmd_name = prim.get('cmd', '?')
                                args = {k: v for k, v in prim.items() if k != 'cmd'}
                                if args:
                                    console.write(f"  {pi}. {cmd_name}  {args}")
                                else:
                                    console.write(f"  {pi}. {cmd_name}")
                            console.write("\n(Plan-only mode — not executed.)")
                    except Exception as e:
                        import traceback
                        console.write(f"[JARVISPLAN] ERROR: {e}")
                        console.write(traceback.format_exc()[-400:])

                # --- Test command: full vision grasp (LLM identifies objects) ---
                elif cmd_lower.startswith("jarvistest"):
                    try:
                        test_text = raw[len("jarvistest"):].strip()
                        if not test_text:
                            console.write("[TEST] Usage: jarvistest <object description>")
                            continue

                        # Ask LLM what to search for
                        avail = list(loaded_objects.keys()) or list(OBJECT_CREATORS.keys())
                        llm_id = identify_objects_llm(test_text, console,
                                                     available_objects=avail)
                        if llm_id and llm_id.get("search_queries"):
                            grasp_phrase = llm_id["search_queries"][0]
                            console.write(f"[TEST] LLM search query: '{grasp_phrase}'")
                        else:
                            grasp_phrase = _extract_object_phrase(test_text)
                            console.write(f"[TEST] Fallback phrase: '{grasp_phrase}'")

                        if not grasp_phrase:
                            console.write("[TEST] Could not identify object to grasp.")
                            continue
                        console.write(f"[TEST] Running vision grasp pipeline for '{grasp_phrase}'...")
                        steps, labels = _vision_grasp_pipeline(grasp_phrase)
                        if not steps:
                            console.write(f"[TEST] Grasp pipeline failed for '{grasp_phrase}'.")
                            continue
                        console.write(f"[TEST] Executing {len(steps)} steps...")
                        _exec_steps_immediate(steps)
                        for joint_idx, slider_id, max_force in sliders:
                            held_positions[joint_idx] = p.getJointState(robot_id, joint_idx)[0]
                        held_gripper = p.getJointState(gripper_info["id"], gripper_info["left_idx"])[0]
                        console.write(f"[TEST] Done — '{grasp_phrase}' grasp complete.")
                    except Exception as e:
                        import traceback
                        console.write(f"[TEST] ERROR: {e}")
                        console.write(traceback.format_exc()[-400:])
                # --- jarvis — LLM task planner (jarvis3 dispatch) ---
                elif cmd_lower.startswith("jarvis"):
                    try:
                        req = raw[len("jarvis"):].strip()
                        if not req:
                            console.write("Usage: jarvis <command>")
                            console.write("  Example: jarvis pick up the bottle")
                            console.write("  Example: jarvis stack the pyramid on the block")
                            console.write("  After the plan is shown, type:")
                            console.write("    go           — execute all steps")
                            console.write("    plan         — re-display the plan")
                            console.write("    clear        — discard the plan")
                            continue

                        console.write(f"Jarvis: \"{req}\"")
                        console.write("Jarvis: Calculating...")

                        # Detect all objects via camera before planning
                        scene_positions = compute_object_positions_vision(
                            loaded_objects, console)
                        if not scene_positions:
                            console.write("Jarvis: No objects detected.")
                            continue

                        # Call LLM task planner with camera images + scene positions
                        flat_steps = call_jarvis3_dispatch(
                            user_request=req,
                            robot_id=robot_id,
                            ee_link_index=ee_link_index,
                            movable_joints=movable_joints,
                            console=console,
                            gripper_info=gripper_info,
                            loaded_objects=loaded_objects,
                            compute_object_positions=compute_object_positions_vision,
                            capture_ee_image=capture_ee_image,
                            capture_birdseye_image=capture_birdseye_image,
                            capture_isometric_image=capture_isometric_image,
                            images_to_base64=images_to_base64,
                            _write_png=_write_png,
                            SNAP_DIR=SNAP_DIR,
                        )
                        if not flat_steps:
                            console.write("Jarvis: Planning failed.")
                            continue

                        # Store in execution state
                        state["pending_plan"] = flat_steps
                        state["original_request"] = req
                        state["executing"] = False
                        state["exec_step_index"] = 0

                        # Display plan for confirmation
                        console.write(f"\nJarvis: Ready! ({len(flat_steps)} steps)")
                        for i, s in enumerate(flat_steps):
                            console.write(format_step(i + 1, s))
                        console.write(f"\nType 'go' to execute, 'plan' to review, or 'clear' to discard.")

                    except Exception as e:
                        import traceback
                        console.write(f"Jarvis: ERROR \u2014 {e}")
                        console.write(traceback.format_exc()[-400:])

                elif cmd_lower == "go" or cmd_lower.startswith("go "):
                    if state["executing"]:
                        console.write("Already executing — please wait.")
                    elif not state["pending_plan"]:
                        console.write("No pending plan. Use 'jarvis <request>' first.")
                    else:
                        state["executing"] = True
                        state["exec_step_index"] = 0
                        state["exec_settle_ticks"] = 0
                        state["exec_step_ticks"] = 0
                        state["exec_stall_ticks"] = 0
                        state["exec_prev_positions"] = None
                        console.write(f"Jarvis: Executing ({len(state['pending_plan'])} steps)...")

                elif cmd_lower == "plan":
                    if not state["pending_plan"]:
                        console.write("No pending plan.")
                    else:
                        console.write(f"\nPlan ({len(state['pending_plan'])} steps):")
                        for i, s in enumerate(state["pending_plan"]):
                            console.write(format_step(i + 1, s))

                elif cmd_lower == "skip":
                    if not state["executing"]:
                        console.write("Not executing — nothing to skip.")
                    else:
                        idx = state["exec_step_index"]
                        step = state["pending_plan"][idx]
                        console.write(f"  \u23ed Skipping step {idx+1}: {step_summary(step)}")
                        state["exec_step_index"] += 1
                        state["exec_settle_ticks"] = 0
                        state["exec_step_ticks"] = 0
                        state["exec_stall_ticks"] = 0
                        state["exec_prev_positions"] = None
                        if state["exec_step_index"] >= len(state["pending_plan"]):
                            console.write("\u2713 Plan complete (last step skipped).")
                            state["pending_plan"] = []
                            state["executing"] = False
                        else:
                            nxt = state["pending_plan"][state["exec_step_index"]]
                            console.write(f"  Continuing: {step_summary(nxt)}")

                elif cmd_lower == "clear":
                    state["pending_plan"] = []
                    state["executing"] = False
                    state["exec_step_index"] = 0
                    state["original_request"] = ""
                    held_positions.clear()
                    held_gripper = None
                    console.write("Plan cleared.")

                elif cmd_lower.startswith("load"):
                    obj_type = cmd_lower[len("load"):].strip()
                    if obj_type not in OBJECT_CREATORS:
                        console.write(f"[OBJ] Unknown object '{obj_type}'.")
                        console.write(f"       Available: {', '.join(OBJECT_CREATORS.keys())}")
                    elif obj_type in loaded_objects:
                        console.write(f"[OBJ] '{obj_type}' is already loaded. Unload it first.")
                    else:
                        body_id = OBJECT_CREATORS[obj_type]()
                        loaded_objects[obj_type] = body_id
                        pos, _ = p.getBasePositionAndOrientation(body_id)
                        console.write(f"[OBJ] Loaded '{obj_type}' at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}).")

                elif cmd_lower.startswith("unload"):
                    obj_type = cmd_lower[len("unload"):].strip()
                    if obj_type not in loaded_objects:
                        console.write(f"[OBJ] '{obj_type}' is not loaded.")
                    else:
                        p.removeBody(loaded_objects.pop(obj_type))
                        console.write(f"[OBJ] Removed '{obj_type}'.")

                elif cmd_lower == "objects":
                    if not loaded_objects:
                        console.write("[OBJ] No objects loaded.")
                    else:
                        console.write(f"[OBJ] {len(loaded_objects)} object(s):")
                        for name, bid in loaded_objects.items():
                            pos, _ = p.getBasePositionAndOrientation(bid)
                            console.write(f"  {name}  id={bid}  pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

                elif cmd_lower == "camera":
                    if ee_camera.enabled:
                        ee_camera.destroy()
                        console.write("Camera closed.")
                    else:
                        ee_camera = EndEffectorCamera(robot_id, ee_link_index, console._root)
                        console.write("Camera opened.")

                elif cmd_lower == "snapshot":
                    console.write("Capturing snapshots...")
                    snap_result = save_snapshots(robot_id, ee_link_index, console)
                    console.write("Snapshots saved.")


                elif cmd_lower == "bbox_screenshot" or cmd_lower == "bbox":
                    if not loaded_objects:
                        console.write("[BBOX] No objects loaded — nothing to detect.")
                    else:
                        console.write("[BBOX] Step 1: Capturing birdseye and isometric views...")
                        try:
                            bb_bird = capture_birdseye_image()
                            bb_iso  = capture_isometric_image()
                            # --- Step 2: Use these views to point the EE camera at the object ---
                            # Placeholder: In a real implementation, analyze bb_bird/bb_iso to find object and move EE.
                            # For now, just print a message.
                            console.write("[BBOX] (TODO: Move end-effector to point at object using birdseye/isometric)")
                            # --- Step 3: Capture EE image after aiming ---
                            bb_ee   = capture_ee_image(robot_id, ee_link_index)
                            # --- Step 4: Run CLIP detection ---
                            ann_bird, ann_ee, ann_iso, legend = compute_clip_bboxes(
                                loaded_objects, robot_id, ee_link_index,
                                bb_bird, bb_ee, bb_iso,
                                console=console,
                            )
                            os.makedirs(SNAP_DIR, exist_ok=True)
                            ts = time.strftime("%Y%m%d_%H%M%S")
                            for tag, img in [("bird_bbox", ann_bird),
                                             ("ee_bbox", ann_ee),
                                             ("iso_bbox", ann_iso)]:
                                path = os.path.join(SNAP_DIR, f"{tag}_{ts}.png")
                                _write_png(path, img)
                                console.write(f"  Saved: {os.path.basename(path)}")
                            console.write(f"[BBOX] Done — annotated images in snapshots/ folder.")
                            console.write(legend)
                        except Exception as e:
                            import traceback
                            console.write(f"[BBOX] ERROR: {e}")
                            console.write(traceback.format_exc()[-400:])

                # --- detect <query> — point EE, show DINO bounding boxes, open image ---
                elif cmd_lower.startswith("detect"):
                    det_query = raw[len("detect"):].strip()
                    if not det_query:
                        console.write("[DETECT] Usage: detect <object description>")
                        console.write("  Example: detect bottle")
                        console.write("  Example: detect bottle, box")
                        continue
                    try:
                        from clip_bbox import (
                            _ensure_model, _detect_objects,
                            _draw_rect, _draw_corner_marks, _draw_label,
                            _bbox_color, _bbox_color_name,
                        )
                        _ensure_model(console)

                        # Split on commas or "and" for multi-object queries
                        import re as _re
                        raw_items = _re.split(r',|\band\b', det_query)
                        search_items = [s.strip() for s in raw_items if s.strip()]
                        if not search_items:
                            search_items = [det_query]

                        # Use first item to locate from isometric view
                        primary = search_items[0]
                        VIEW_STANDOFF_DET = 0.30

                        console.write(f"[DETECT] Locating '{primary}' from isometric view...")
                        iso_rgb, iso_dep, iso_v, iso_p, iso_n, iso_f = \
                            capture_isometric_rgbd()
                        ih, iw = iso_dep.shape
                        iso_dets = _detect_objects(iso_rgb, [primary])
                        iso_det = iso_dets.get(primary, {})
                        iso_pos = _best_valid_detection(
                            iso_det, iso_dep, iso_v, iso_p,
                            iso_n, iso_f, iw, ih,
                            console=console, label=f"'{primary}' isometric")
                        if not iso_pos:
                            console.write(f"[DETECT] Could not locate '{primary}' in isometric view.")
                            continue
                        ox, oy, oz = iso_pos
                        console.write(f"  [DETECT] '{primary}': isometric → "
                                      f"({ox:.3f}, {oy:.3f}, {oz:.3f})")

                        # Open gripper fully so fingers don't obstruct the EE camera
                        _exec_steps_immediate([{"type": "gripper", "target": GRIP_OPEN}])
                        held_gripper = GRIP_OPEN

                        # Point EE at the object
                        od = math.sqrt(ox*ox + oy*oy) or 1.0
                        oax, oay = ox / od, oy / od
                        look_pos = [ox - VIEW_STANDOFF_DET * oax,
                                    oy - VIEW_STANDOFF_DET * oay,
                                    oz + 0.05]
                        look_quat = _look_at_quat(look_pos, [ox, oy, oz])
                        look_targets, _ = _ns_ik(look_pos, look_quat)

                        console.write(f"[DETECT] Pointing EE at '{primary}'...")
                        _exec_steps_immediate([{"type": "pose", "targets": look_targets}])
                        for joint_idx_s, slider_id_s, max_force_s in sliders:
                            held_positions[joint_idx_s] = p.getJointState(robot_id, joint_idx_s)[0]

                        # Centering refinement loop
                        CENTRE_TOL_DET = 20
                        _cmiss = 0
                        for _ri in range(3):
                            r_rgb, r_dep, _, _ = capture_ee_rgbd(robot_id, ee_link_index)
                            rh, rw = r_dep.shape
                            r_dets = _detect_objects(r_rgb, [primary])
                            r_det = r_dets.get(primary, {})
                            r_bbox = r_det.get("bbox")

                            if not r_bbox:
                                _cmiss += 1
                                if _cmiss >= 2:
                                    ri_rgb, ri_dep, ri_v, ri_p, ri_n, ri_f = \
                                        capture_isometric_rgbd()
                                    ri_h, ri_w = ri_dep.shape
                                    ri_dets = _detect_objects(ri_rgb, [primary])
                                    ri_det = ri_dets.get(primary, {})
                                    ri_pos = _best_valid_detection(
                                        ri_det, ri_dep, ri_v, ri_p,
                                        ri_n, ri_f, ri_w, ri_h)
                                    if ri_pos:
                                        ox, oy, oz = ri_pos
                                        od2 = math.sqrt(ox*ox + oy*oy) or 1.0
                                        oax, oay = ox / od2, oy / od2
                                        look_pos = [ox - VIEW_STANDOFF_DET * oax,
                                                    oy - VIEW_STANDOFF_DET * oay,
                                                    oz + 0.05]
                                    else:
                                        break
                                    _cmiss = 0
                                else:
                                    look_pos[0] -= 0.05 * oax
                                    look_pos[1] -= 0.05 * oay
                                    look_pos[2] += 0.03
                                fq = _look_at_quat(look_pos, [ox, oy, oz])
                                lt, _ = _ns_ik(look_pos, fq)
                                _exec_steps_immediate([{"type": "pose", "targets": lt}])
                                for ji, si, mf in sliders:
                                    held_positions[ji] = p.getJointState(robot_id, ji)[0]
                                continue

                            _cmiss = 0
                            rb1, rb2, rb3, rb4 = r_bbox
                            du = (rb1 + rb3) / 2.0 - rw / 2.0
                            dv = (rb2 + rb4) / 2.0 - rh / 2.0
                            if abs(du) < CENTRE_TOL_DET and abs(dv) < CENTRE_TOL_DET:
                                print(f"  Centred on '{primary}' (err {abs(du):.0f}/{abs(dv):.0f} px)")
                                break
                            roi = r_dep[rb2:rb4+1, rb1:rb3+1]
                            vd = roi[(roi > 0) & np.isfinite(roi)]
                            rZ = float(np.median(vd)) if vd.size > 0 else 0
                            if rZ <= 0 or not math.isfinite(rZ):
                                break
                            asp = rw / rh
                            hvfov = math.radians(CAM_FOV / 2.0)
                            hhfov = math.atan(asp * math.tan(hvfov))
                            wpp_h = (2.0 * rZ * math.tan(hhfov)) / rw
                            wpp_v = (2.0 * rZ * math.tan(hvfov)) / rh
                            es = p.getLinkState(robot_id, ee_link_index,
                                                computeForwardKinematics=True)
                            epr = np.array(es[4])
                            rot_r = np.array(p.getMatrixFromQuaternion(es[5])).reshape(3, 3)
                            shift = du * wpp_h * rot_r[:, 0] + dv * wpp_v * rot_r[:, 1]
                            ne = (epr + shift).tolist()
                            oe = (ne[0] + rZ * rot_r[:, 2][0],
                                  ne[1] + rZ * rot_r[:, 2][1],
                                  ne[2] + rZ * rot_r[:, 2][2])
                            lqc = _look_at_quat(ne, oe)
                            rt, _ = _ns_ik(ne, lqc)
                            if rt:
                                _exec_steps_immediate([{"type": "pose", "targets": rt}])
                                for ji, si, mf in sliders:
                                    held_positions[ji] = p.getJointState(robot_id, ji)[0]
                                console.write(f"  Centering: shifted {np.linalg.norm(shift)*1000:.1f} mm")
                            else:
                                break

                        # Final EE capture — run DINO on all search items
                        console.write(f"[DETECT] Capturing EE view for: {search_items}")
                        ee_rgb, _, _, _ = capture_ee_rgbd(robot_id, ee_link_index)
                        ee_dets = _detect_objects(ee_rgb, search_items)

                        annotated = ee_rgb.copy()
                        found_any = False
                        for idx, name in enumerate(search_items):
                            info = ee_dets.get(name, {})
                            if not info.get("visible"):
                                console.write(f"  [DETECT] '{name}': not detected")
                                continue
                            found_any = True
                            bx1, by1, bx2, by2 = info["bbox"]
                            color = _bbox_color(idx)
                            cname = _bbox_color_name(idx)
                            _draw_rect(annotated, bx1, by1, bx2, by2, color, thickness=3)
                            _draw_corner_marks(annotated, bx1, by1, bx2, by2, color,
                                               length=14, thickness=4)
                            _draw_label(annotated, bx1, by1, color,
                                        f"{name} {info['score']:.2f}")
                            console.write(f"  [DETECT] '{name}': {cname} box "
                                          f"({bx1},{by1})-({bx2},{by2}) "
                                          f"score={info['score']:.3f}")

                        if not found_any:
                            console.write("[DETECT] Nothing detected in EE camera view.")
                            continue

                        os.makedirs(SNAP_DIR, exist_ok=True)
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        det_path = os.path.join(SNAP_DIR, f"detect_{ts}.png")
                        _write_png(det_path, annotated)
                        console.write(f"[DETECT] Saved: {det_path}")
                        os.startfile(det_path)

                    except Exception as e:
                        import traceback
                        console.write(f"[DETECT] ERROR: {e}")
                        console.write(traceback.format_exc()[-400:])

                elif cmd_lower == "help":
                    console.write("\nCommands:")
                    console.write("  jarvis <request>        — LLM task planner (natural language → plan → execute)")
                    console.write("    Example: jarvis pick up the bottle")
                    console.write("    Example: jarvis stack the pyramid on the block")
                    console.write("  go                      — Execute the pending plan")
                    console.write("  plan                    — Re-display the pending plan")
                    console.write("  clear                   — Discard plan, release joints")
                    console.write("  skip                    — Skip the current stuck step")
                    console.write("  jarvistest <obj>        — Direct vision grasp (no LLM)")
                    # dynamic object list
                    try:
                        obj_list = ", ".join(sorted(OBJECT_CREATORS.keys()))
                    except Exception:
                        obj_list = "block, cylinder, bottle, mug, remote, box"
                    console.write(f"  load <type>             — Spawn object (available: {obj_list})")
                    console.write("  unload <type>           — Remove a loaded object")
                    console.write("  objects                 — List currently loaded objects")
                    console.write("  camera                  — Toggle the end-effector camera")
                    console.write("  snapshot                — Save camera images")
                    console.write("  detect <query>          — Point EE at object, show DINO bounding box")
                    console.write("  bbox_screenshot         — Run CLIP detection and save annotated images")
                    console.write("  help                    — Show this help message")
                    console.write("")
                    console.write("Notes:")
                    console.write("  - 'jarvis' scans the scene with cameras, sends images to the LLM,")
                    console.write("    and builds an executable plan using 50+ motion primitives.")
                    console.write("  - If a step gets stuck during execution, the system will attempt")
                    console.write("    to replan automatically using fresh camera images.")
                    console.write("  - Use 'snapshot' to save debug images to the snapshots/ folder.")

                else:
                    console.write(f"[CMD] Unknown command: {raw}")
                    console.write("       Type 'help' for available commands.")

            # ---- Joint control ----
            if state["executing"] and state["pending_plan"]:
                step = state["pending_plan"][state["exec_step_index"]]
                stype = step.get("type", "joint")

                # ---- Determine convergence flag and drive motors ----
                converged = False

                if stype == "pose":
                    # SIMULTANEOUS pose: drive ALL target joints at once
                    targets = step["targets"]  # dict  joint_name -> angle
                    active_idxs = set()
                    max_err = 0.0
                    for jname, jtarget in targets.items():
                        jidx, jforce = joint_lookup[jname]
                        active_idxs.add(jidx)
                        # Use higher force during AI execution so the arm can
                        # push through light objects (e.g. sweep manoeuvres)
                        exec_force = max(jforce, 50.0)
                        p.setJointMotorControl2(
                            bodyUniqueId=robot_id,
                            jointIndex=jidx,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=jtarget,
                            force=exec_force,
                            maxVelocity=AI_MAX_VELOCITY,
                        )
                        cur = p.getJointState(robot_id, jidx)[0]
                        max_err = max(max_err, abs(cur - jtarget))
                    # Hold non-active arm joints
                    for joint_idx, slider_id, max_force in sliders:
                        if joint_idx in active_idxs:
                            continue
                        hold_target = held_positions.get(joint_idx, p.readUserDebugParameter(slider_id))
                        p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL,
                                                targetPosition=hold_target, force=max_force)
                    # Hold gripper (force=50 to match gripper_force close strength)
                    grip_hold = held_gripper if held_gripper is not None else p.readUserDebugParameter(gripper_slider_id)
                    p.setJointMotorControl2(gripper_info["id"], gripper_info["left_idx"],
                                            p.POSITION_CONTROL, targetPosition=grip_hold, force=50)
                    p.setJointMotorControl2(gripper_info["id"], gripper_info["right_idx"],
                                            p.POSITION_CONTROL, targetPosition=-grip_hold, force=50)
                    converged = max_err < POSITION_TOLERANCE

                elif stype == "gripper":
                    target_angle = step["target"]
                    g_id = gripper_info["id"]
                    p.setJointMotorControl2(g_id, gripper_info["left_idx"],
                                            p.POSITION_CONTROL, targetPosition=target_angle,
                                            force=20, maxVelocity=AI_MAX_VELOCITY)
                    p.setJointMotorControl2(g_id, gripper_info["right_idx"],
                                            p.POSITION_CONTROL, targetPosition=-target_angle,
                                            force=20, maxVelocity=AI_MAX_VELOCITY)
                    cur_angle = p.getJointState(g_id, gripper_info["left_idx"])[0]
                    # Hold all arm joints
                    for joint_idx, slider_id, max_force in sliders:
                        hold_target = held_positions.get(joint_idx, p.readUserDebugParameter(slider_id))
                        p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL,
                                                targetPosition=hold_target, force=max_force)
                    converged = abs(cur_angle - target_angle) < POSITION_TOLERANCE

                else:  # single joint step OR complex types (dwell/gripper_force/servo_contact/creep_contact)
                    if stype in ("dwell", "gripper_force", "servo_contact", "creep_contact", "grip_verify", "deliver_swing"):
                        # Complex step types have internal convergence loops —
                        # execute synchronously via _exec_steps_immediate, then advance.
                        console.write(f"  Step {state['exec_step_index']+1}: {step_summary(step)}...")
                        try:
                            _exec_steps_immediate([step])
                        except Exception as _step_err:
                            import traceback as _tb
                            console.write(f"  Step {state['exec_step_index']+1}: {step_summary(step)} \u2717")
                            console.write(_tb.format_exc()[-300:])
                        # Record held positions after sync execution
                        for joint_idx_s, slider_id_s, max_force_s in sliders:
                            held_positions[joint_idx_s] = p.getJointState(robot_id, joint_idx_s)[0]
                        held_gripper = p.getJointState(gripper_info["id"], gripper_info["left_idx"])[0]
                        # Advance to next step
                        state["exec_step_index"] += 1
                        state["exec_settle_ticks"] = 0
                        state["exec_step_ticks"] = 0
                        state["exec_stall_ticks"] = 0
                        state["exec_prev_positions"] = None
                        state["_requery_fired"] = False
                        if state["exec_step_index"] >= len(state["pending_plan"]):
                            console.write("Jarvis: Done!")
                            state["pending_plan"] = []
                            state["executing"] = False
                        continue
                    # Legacy single-joint step
                    target_angle = step["target"]
                    jidx, jforce = joint_lookup[step["joint"]]
                    p.setJointMotorControl2(
                        bodyUniqueId=robot_id,
                        jointIndex=jidx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=target_angle,
                        force=jforce,
                        maxVelocity=AI_MAX_VELOCITY,
                    )
                    cur_angle = p.getJointState(robot_id, jidx)[0]
                    # Hold other arm joints
                    for joint_idx, slider_id, max_force in sliders:
                        if joint_idx == jidx:
                            continue
                        hold_target = held_positions.get(joint_idx, p.readUserDebugParameter(slider_id))
                        p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL,
                                                targetPosition=hold_target, force=max_force)
                    # Hold gripper (force=50 to match gripper_force close strength)
                    grip_hold = held_gripper if held_gripper is not None else p.readUserDebugParameter(gripper_slider_id)
                    p.setJointMotorControl2(gripper_info["id"], gripper_info["left_idx"],
                                            p.POSITION_CONTROL, targetPosition=grip_hold, force=50)
                    p.setJointMotorControl2(gripper_info["id"], gripper_info["right_idx"],
                                            p.POSITION_CONTROL, targetPosition=-grip_hold, force=50)
                    converged = abs(cur_angle - target_angle) < POSITION_TOLERANCE

                # ---- Tick counting & settle / stall detection ----
                state["exec_step_ticks"] += 1

                if converged:
                    state["exec_settle_ticks"] += 1
                else:
                    state["exec_settle_ticks"] = 0

                # Stall detection: snapshot all arm joint positions every tick
                # and check if anything has moved in the last STALL_TICKS.
                cur_positions = tuple(p.getJointState(robot_id, ji)[0] for ji, _, _ in sliders)
                if state["exec_prev_positions"] is not None:
                    max_delta = max(abs(a - b) for a, b in zip(cur_positions, state["exec_prev_positions"]))
                    if max_delta < STALL_THRESHOLD:
                        state["exec_stall_ticks"] += 1
                    else:
                        state["exec_stall_ticks"] = 0
                state["exec_prev_positions"] = cur_positions

                # ---- Helper: finish current sub-step (success or timeout) ----
                def _finish_step(timed_out=False):
                    """Common bookkeeping after a sub-step completes or times out."""
                    # Record held positions
                    if stype == "pose":
                        for jname, jtarget in step["targets"].items():
                            jidx_done = joint_lookup[jname][0]
                            if timed_out:
                                held_positions[jidx_done] = p.getJointState(robot_id, jidx_done)[0]
                            else:
                                held_positions[jidx_done] = jtarget
                    elif stype == "gripper":
                        if timed_out:
                            held_gripper_val = p.getJointState(gripper_info["id"], gripper_info["left_idx"])[0]
                        else:
                            held_gripper_val = step["target"]
                        nonlocal held_gripper
                        held_gripper = held_gripper_val
                    else:  # single joint (legacy)
                        jidx_done = joint_lookup[step["joint"]][0]
                        if timed_out:
                            held_positions[jidx_done] = p.getJointState(robot_id, jidx_done)[0]
                        else:
                            held_positions[jidx_done] = step["target"]
                    state["exec_step_index"] += 1
                    state["exec_settle_ticks"] = 0
                    state["exec_step_ticks"] = 0
                    state["exec_stall_ticks"] = 0
                    state["exec_prev_positions"] = None
                    state["_requery_fired"] = False

                # ---- Stall or Timeout ----
                stalled = (state["exec_stall_ticks"] >= STALL_TICKS and
                           state["exec_step_ticks"] > STALL_TICKS)
                timed_out_hard = state["exec_step_ticks"] >= STEP_TIMEOUT_TICKS
                if stalled or timed_out_hard:
                    reason = "STALLED" if stalled else "TIMEOUT"
                    idx = state["exec_step_index"]
                    console.write(f"  Step {idx+1}: {step_summary(step)} \u2717 ({reason.lower()})")

                    # Try replanning once per stuck step
                    if (state.get("original_request")
                            and not state.get("_requery_fired")):
                        state["_requery_fired"] = True
                        console.write("Jarvis: Replanning...")
                        completed = state["pending_plan"][:idx]
                        remaining = state["pending_plan"][idx+1:]
                        new_steps = requery_jarvis3_dispatch(
                            original_request=state["original_request"],
                            robot_id=robot_id,
                            ee_link_index=ee_link_index,
                            movable_joints=movable_joints,
                            console=console,
                            gripper_info=gripper_info,
                            loaded_objects=loaded_objects,
                            compute_object_positions=compute_object_positions_vision,
                            capture_ee_image=capture_ee_image,
                            capture_birdseye_image=capture_birdseye_image,
                            capture_isometric_image=capture_isometric_image,
                            images_to_base64=images_to_base64,
                            completed_steps=completed,
                            stuck_step=step,
                            remaining_steps=remaining,
                        )
                        if new_steps:
                            console.write(f"Jarvis: New plan ({len(new_steps)} steps)")
                            state["pending_plan"] = completed + new_steps
                            state["exec_step_index"] = len(completed)
                            state["exec_settle_ticks"] = 0
                            state["exec_step_ticks"] = 0
                            state["exec_stall_ticks"] = 0
                            state["exec_prev_positions"] = None
                            continue
                        else:
                            console.write("Jarvis: Replan failed, skipping.")

                    _finish_step(timed_out=True)

                    if state["exec_step_index"] >= len(state["pending_plan"]):
                        console.write("Jarvis: Done!")
                        state["pending_plan"] = []
                        state["executing"] = False
                    else:
                        nxt = state["pending_plan"][state["exec_step_index"]]
                        console.write(f"  Step {state['exec_step_index']+1}: {step_summary(nxt)}...")

                # ---- Settled / complete ----
                elif state["exec_settle_ticks"] > 20:
                    console.write(f"  Step {state['exec_step_index']+1}: "
                                  f"{step_summary(step)} \u2713")
                    _finish_step(timed_out=False)

                    if state["exec_step_index"] >= len(state["pending_plan"]):
                        console.write("Jarvis: Done!")
                        state["pending_plan"] = []
                        state["executing"] = False
            else:
                for joint_idx, slider_id, max_force in sliders:
                    # Use held position if AI has set one, otherwise follow slider
                    if joint_idx in held_positions:
                        target = held_positions[joint_idx]
                    else:
                        target = p.readUserDebugParameter(slider_id)
                    p.setJointMotorControl2(
                        bodyUniqueId=robot_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=target,
                        force=max_force,
                    )
                # Gripper: hold AI position or follow slider
                if held_gripper is not None:
                    gripper_target = held_gripper
                    grip_force = 50   # match gripper_force close strength
                else:
                    gripper_target = p.readUserDebugParameter(gripper_slider_id)
                    grip_force = 20   # light force for manual slider control
                p.setJointMotorControl2(gripper_info["id"], gripper_info["left_idx"],
                                        p.POSITION_CONTROL, targetPosition=gripper_target, force=grip_force)
                p.setJointMotorControl2(gripper_info["id"], gripper_info["right_idx"],
                                        p.POSITION_CONTROL, targetPosition=-gripper_target, force=grip_force)

                # end command processing

            p.stepSimulation()

            # Update end-effector camera feed
            ee_camera.tick()

            # Pump tkinter events (this is what makes keyboard input work)
            console.update()

            # Arrow-key camera rotation (10° per press)
            keys = p.getKeyboardEvents()
            if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_WAS_TRIGGERED:
                cam_yaw = (cam_yaw + 10) % 360
                p.resetDebugVisualizerCamera(2.5, cam_yaw, -30, [0, 0, 0.8])
            if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_WAS_TRIGGERED:
                cam_yaw = (cam_yaw - 10) % 360
                p.resetDebugVisualizerCamera(2.5, cam_yaw, -30, [0, 0, 0.8])

            time.sleep(1 / 240)

    except KeyboardInterrupt:
        print("\n[SIM] Interrupted by user.")
    except Exception as e:
        import traceback
        print(f"\n[SIM] FATAL ERROR: {e}")
        traceback.print_exc()
    finally:
        ee_camera.destroy()
        console.stop()
        p.disconnect()
        print("[SIM] PyBullet disconnected. Goodbye!")


if __name__ == "__main__":
    main()
