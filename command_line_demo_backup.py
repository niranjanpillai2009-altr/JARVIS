import pybullet as p
import pybullet_data
import time
import os
import math
import threading
import tkinter as tk
from tkinter import scrolledtext
import queue
import json
import requests
from jsonschema import validate, ValidationError

# Try to import the mesh downloader
try:
    from download_kinova_meshes import download_kinova_meshes
    print("Attempting to download Kinova Gen 3 mesh files...")
    mesh_dir = download_kinova_meshes()
    USE_MESHES = True
except Exception as e:
    print(f"Warning: Could not download mesh files: {e}")
    print("Will use simplified geometry instead.")
    USE_MESHES = False

# Connect to PyBullet with GUI
physicsClient = p.connect(p.GUI)

# Set the additional search path to find built-in example models
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set gravity
p.setGravity(0, 0, -9.81)

# Load the plane
planeId = p.loadURDF("plane.urdf")

# Get the path to the Kinova Gen 3 URDF file
script_dir = os.path.dirname(os.path.abspath(__file__))
if USE_MESHES:
    kinova_urdf_path = os.path.join(script_dir, "KinovaGen3_Meshes.urdf")
else:
    kinova_urdf_path = os.path.join(script_dir, "KinovaGen3_Working.urdf")

# Check if file exists
print(f"Looking for Kinova Gen 3 at: {kinova_urdf_path}")
if not os.path.exists(kinova_urdf_path):
    print(f"ERROR: File not found at {kinova_urdf_path}")
    print(f"Files in directory: {os.listdir(script_dir)}")
    input("Press Enter to close...")
    p.disconnect()
    exit()

# Load the Kinova Gen 3 robot with fixed base
print(f"Loading Kinova Gen 3...")
try:
    kinovaId = p.loadURDF(kinova_urdf_path, [0, 0, 0], useFixedBase=True)
    print("Kinova Gen 3 loaded successfully!")
except Exception as e:
    print(f"Error loading Kinova Gen 3: {e}")
    import traceback
    traceback.print_exc()
    input("Press Enter to close...")
    p.disconnect()
    exit()

# Get information about the robot's joints
num_joints = p.getNumJoints(kinovaId)
print(f"\nRobot has {num_joints} joints:")

joint_info_list = []
joint_indices = []
for i in range(num_joints):
    joint_info = p.getJointInfo(kinovaId, i)
    joint_name = joint_info[1].decode('utf-8')
    joint_type = joint_info[2]
    print(f"  Joint {i}: {joint_name} (Type: {joint_type})")
    # Only control revolute (type 0) and continuous (type 4) joints
    if joint_type in [0, 4]:
        joint_indices.append(i)
        joint_info_list.append((i, joint_name))

print(f"\nControlling {len(joint_indices)} actuated joints")

# Movement types for AIMove
class Movement:
    """Represents a single robotic movement"""
    def __init__(self, move_type, description, **kwargs):
        self.move_type = move_type  # Type of movement: 'extend', 'rotate', 'move_forward', etc.
        self.description = description  # Human-readable description
        self.params = kwargs  # Additional parameters specific to the movement
    
    def __str__(self):
        return f"  • {self.description}"

# Movement types available
MOVEMENT_TYPES = [
    "home - Move arm to home position",
    "extend - Fully extend the arm",
    "move_forward - Move end effector forward",
    "move_back - Move end effector backward",
    "move_up - Move end effector upward",
    "move_down - Move end effector downward",
    "rotate - Rotate base joint",
    "open_gripper - Open the gripper (if loaded)",
    "close_gripper - Close the gripper (if loaded)",
    "custom_joint - Move specific joint to angle"
]

# Global state for arm control
arm_state = {
    'target_positions': [0.0] * len(joint_indices),
    'moving': False,
    'speed': 'medium',  # Default speed preset
    'duration': 3.0,    # Duration in seconds (will be set based on speed)
    'running': True,
    'move_start_time': None,
    'move_duration': 0,
    'move_initial_positions': None,
    'loaded_objects': {},  # Store loaded object IDs
    'gripper_id': None,  # Store gripper ID when loaded
    'gripper_finger_ids': [],  # Store gripper finger IDs
    'gripper_constraint_ids': [],  # Store constraint IDs for gripper attachment
    'ai_sequence': None,  # Store pending AI-generated movement sequence
    'ai_sequence_index': 0  # Current index in execution
}

# Speed multipliers (applied to base movement speed)
SPEED_MULTIPLIERS = {
    'slow': 4.0,       # 4x slower - smooth detailed observation
    'medium': 1.0,     # 1x baseline - smooth animation
    'fast': 0.25       # 0.25x (4x faster) - nearly instant
}

# Base rotation speed for medium speed (degrees per second)
BASE_ROTATION_SPEED = 120.0  # 120 degrees per second = 360 degrees in 3 seconds

# GUI state for command queue
gui_state = {
    'command_queue': queue.Queue()
}

max_forces = [200] * len(joint_indices)

# ============================================================================
# LLM API Configuration (fill in API_KEY when ready)
# ============================================================================
API_KEY = os.environ.get("JARVIS_API_KEY", "YOUR_API_KEY_HERE")
LLM_MODEL = "gpt-4o-mini"
LLM_API_URL = "https://api.openai.com/v1/chat/completions"

# Kinova Action Sequence Schema for validation
KINOVA_SCHEMA = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "KinovaActionSequenceSchema",
  "description": "RESTRICTED Schema: Only simple joint-based actions supported",
  "type": "object",
  "required": ["sequence"],
  "additionalProperties": False,
  "properties": {
    "sequence": {
      "type": "array",
      "minItems": 1,
      "items": { "$ref": "#/definitions/action" }
    }
  },
  "definitions": {
    "action": {
      "type": "object",
      "required": ["action", "description", "params"],
      "additionalProperties": False,
      "properties": {
        "action": {
          "type": "string",
          "enum": [
            "home",
            "extend",
            "rotate",
            "move_forward",
            "move_up",
            "move_down",
            "close_gripper",
            "open_gripper"
          ],
          "description": "ONLY these 8 action types are supported - no Cartesian or complex commands"
        },
        "description": {
          "type": "string",
          "description": "Human-readable description of what this action does"
        },
        "params": {
          "type": "object",
          "description": "Parameters for the action (flexible parameter object)"
        }
      }
    }
  }
}
"""Schema restricted to ONLY support these actions:
- home: Return to rest position (params: none)
- extend: Fully extend arm (params: none)
- rotate: Rotate base (params: angle in degrees)
- move_forward: Move end-effector forward (params: distance in meters)
- move_up: Move end-effector up (params: distance in meters)
- move_down: Move end-effector down (params: distance in meters)
- close_gripper: Close gripper (params: none)
- open_gripper: Open gripper (params: none)
"""

def call_llm_api(task_description):
    """Call OpenAI LLM to generate movement sequence using the output schema."""
    if not API_KEY:
        return None, "Error: API_KEY not set."

    # Crystal clear system prompt with exact output example
    system_prompt = """You are a robot movement sequence generator. NOTHING ELSE.

YOUR ONLY JOB: Convert tasks into movement sequences.
YOUR ONLY OUTPUT: Valid JSON with "sequence" array.

YOU MUST NOT:
- Explain anything
- Return schema definitions
- Return comments
- Return markdown or code blocks
- Return anything except valid JSON

EXACT OUTPUT FORMAT (copy this structure):
{
  "sequence": [
    {"action": "home", "description": "Return to home", "params": {}},
    {"action": "extend", "description": "Extend arm", "params": {}},
    {"action": "rotate", "description": "Rotate 90 degrees", "params": {"angle": 90}},
    {"action": "move_forward", "description": "Move forward", "params": {"distance": 0.1}},
    {"action": "move_up", "description": "Move up", "params": {"distance": 0.1}},
    {"action": "move_down", "description": "Move down", "params": {"distance": 0.1}},
    {"action": "close_gripper", "description": "Close gripper", "params": {}},
    {"action": "open_gripper", "description": "Open gripper", "params": {}}
  ]
}

ALLOWED ACTIONS ONLY:
1. home - no params needed
2. extend - no params needed
3. rotate - required param: "angle" (number in degrees)
4. move_forward - required param: "distance" (number in meters)
5. move_up - required param: "distance" (number in meters)
6. move_down - required param: "distance" (number in meters)
7. close_gripper - no params needed
8. open_gripper - no params needed

RESPOND WITH ONLY JSON. START WITH { AND END WITH }."""

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": f"Task: {task_description}\n\nRespond ONLY with the JSON sequence. Start with {{ and end with }}. No other text."
            }
        ],
        "temperature": 0.0,
        "max_tokens": 2000
    }

    try:
        response = requests.post(
            LLM_API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        # Extract assistant message
        assistant_text = data["choices"][0]["message"]["content"].strip()

        print("\n" + "="*70)
        print("DEBUG: RAW API RESPONSE")
        print("="*70)
        print(assistant_text)
        print("="*70 + "\n")

        # Extract JSON: Find first { and matching }
        start_idx = assistant_text.find('{')
        if start_idx == -1:
            return None, f"No JSON found in response. Got: {assistant_text[:200]}"

        # Find matching closing brace
        brace_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(assistant_text)):
            if assistant_text[i] == '{':
                brace_count += 1
            elif assistant_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break

        json_str = assistant_text[start_idx:end_idx]
        
        # Parse JSON
        try:
            sequence_obj = json.loads(json_str)
        except json.JSONDecodeError as e:
            return None, f"Failed to parse JSON: {str(e)}\nJSON was: {json_str[:200]}"

        # Validate it has sequence key
        if not isinstance(sequence_obj, dict) or "sequence" not in sequence_obj:
            return None, f"Invalid response structure. Must contain 'sequence' key. Got: {list(sequence_obj.keys())}"

        if not isinstance(sequence_obj["sequence"], list):
            return None, f"'sequence' must be an array, got: {type(sequence_obj['sequence'])}"

        if len(sequence_obj["sequence"]) == 0:
            return None, "Empty sequence - no movements generated"

        return sequence_obj, None

    except requests.exceptions.RequestException as e:
        return None, f"API request failed: {str(e)}"
    except (KeyError, IndexError) as e:
        return None, f"Unexpected API response format: {str(e)}"

def get_simulation_state():
    """Retrieve the current state of the simulation, including object positions and robot joint states."""
    # Get all object positions and orientations
    object_states = {}
    for obj_id in arm_state.get('loaded_objects', {}).values():
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        object_states[obj_id] = {
            "position": pos,
            "orientation": orn
        }

    # Get robot joint states
    joint_states = {}
    for joint_index in joint_indices:
        joint_state = p.getJointState(kinovaId, joint_index)
        joint_states[joint_index] = {
            "position": joint_state[0],
            "velocity": joint_state[1],
            "reaction_forces": joint_state[2],
            "torque": joint_state[3]
        }

    return {
        "objects": object_states,
        "joints": joint_states
    }

def validate_sequence_schema(sequence_obj):
    """Validate sequence against Kinova schema."""
    try:
        print("Validating sequence against schema...")
        validate(instance=sequence_obj, schema=KINOVA_SCHEMA)
        return True, "Schema validation passed"
    except ValidationError as e:
        print("Schema validation error:", e.message)
        return False, f"Schema validation failed: {e.message}"

def parse_json_sequence(json_input):
    """Parse JSON input string into sequence object."""
    try:
        sequence_obj = json.loads(json_input)
        valid, msg = validate_sequence_schema(sequence_obj)
        if valid:
            return sequence_obj, None
        else:
            return None, msg
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {str(e)}"

def print_help():
    """Print available commands"""
    help_text = """
================================================================================
KINOVA ARM INTERACTIVE CONTROL
================================================================================

Arm Movement Commands:
  extend          - Fully extend the arm
  bend            - Create a 90-degree bend
  rotate <angle>  - Rotate base by angle in degrees (e.g., rotate 180)
  joint <n> <a>   - Move joint n to angle a in radians
  reset           - Return arm to rest position
  home            - Move all joints to 0 degrees

Speed & Settings:
  speed <preset>  - Set movement speed (slow/medium/fast) - default: medium
  
Object Management:
  load <type>   - Load object (table/block/bottle/blockside/gripper)
  unload <type> - Remove object from scene
  listobjects         - List all loaded objects

AI Movement Planning:
  aimove <task>   - Generate AI movement sequence for task (e.g., 'aimove pick up the block')
                    Can also accept JSON input: aimove {"sequence": [{"action": "home", ...}]}
                    Uses LLM API if OPENAI_API_KEY is set, otherwise uses hardcoded sequences
  go              - Execute the pending AI-generated sequence
  cancel          - Cancel the pending AI sequence

Help:
  help            - Show this help message
  quit            - Exit the program

Speed Presets:
  slow    - 4x slower (12 seconds for 360° rotation)
  medium  - 1x baseline (3 seconds for 360° rotation, smooth animation)
  fast    - 0.25x speed (0.75 seconds for 360° rotation, nearly instant)

Available Task Examples:
  aimove pick up the block    - Generate sequence to pick up block
  aimove place block          - Generate sequence to place block
  aimove extend arm           - Generate sequence to extend arm

Examples:
  extend              - Extends all joints to 1.5 radians
  rotate 180          - Rotates base joint by 180 degrees at current speed
  joint 0 1.57        - Moves joint 0 to 1.57 radians
  speed slow          - Sets all future movements to slow speed
  speed medium        - Sets all future movements to medium speed (default)
  speed fast          - Sets all future movements to fast speed

================================================================================
"""
    print(help_text)

def queue_movement(target_positions):
    """Queue a movement to happen during the main simulation loop (non-blocking)"""
    arm_state['moving'] = True
    arm_state['target_positions'] = target_positions
    
    # Get current positions
    arm_state['move_initial_positions'] = [0.0] * len(joint_indices)
    for idx, joint_idx in enumerate(joint_indices):
        joint_state = p.getJointState(kinovaId, joint_idx)
        arm_state['move_initial_positions'][idx] = joint_state[0]
    
    # Calculate maximum angular distance any joint travels
    max_angle_distance = 0.0
    for idx in range(len(joint_indices)):
        angle_distance = abs(target_positions[idx] - arm_state['move_initial_positions'][idx])
        if angle_distance > max_angle_distance:
            max_angle_distance = angle_distance
    
    # Calculate duration based on angular distance and speed
    max_angle_degrees = math.degrees(max_angle_distance)
    base_duration = max_angle_degrees / BASE_ROTATION_SPEED if max_angle_degrees > 0 else 0.1
    speed_multiplier = SPEED_MULTIPLIERS.get(arm_state['speed'], 1.0)
    arm_state['move_duration'] = base_duration * speed_multiplier
    arm_state['move_duration'] = max(arm_state['move_duration'], 0.05)
    
    # Start the movement
    arm_state['move_start_time'] = time.time()

def update_movement():
    """Update movement animation - called every frame in main loop"""
    if not arm_state['moving']:
        return
    
    elapsed = time.time() - arm_state['move_start_time']
    progress = elapsed / arm_state['move_duration']
    
    # Check if movement is complete
    if progress >= 1.0 or not arm_state['running']:
        progress = 1.0
        arm_state['moving'] = False
    
    # Interpolate all joints
    for idx, joint_idx in enumerate(joint_indices):
        current_pos = arm_state['move_initial_positions'][idx]
        target_pos = arm_state['target_positions'][idx]
        new_pos = current_pos + (target_pos - current_pos) * progress
        
        p.setJointMotorControl2(
            kinovaId,
            joint_idx,
            p.POSITION_CONTROL,
            targetPosition=new_pos,
            force=max_forces[idx]
        )

def move_to_position(target_positions):
    """Queue a movement (wrapper for compatibility)"""
    queue_movement(target_positions)

def create_gripper():
    """Create a stable 2-finger gripper and attach it to the end effector without constraints"""
    try:
        # Get the end effector link (last link, typically)
        end_effector_link = num_joints - 1
        
        # Get the position of the end effector
        ee_state = p.getLinkState(kinovaId, end_effector_link)
        ee_position = ee_state[0]
        ee_orientation = ee_state[1]
        
        # Create a more stable gripper with a single palm body
        # Palm (center body) - fixed at end effector
        palm_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.025, 0.015, 0.02],
            rgbaColor=[0.2, 0.2, 0.2, 1.0]
        )
        palm_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.025, 0.015, 0.02]
        )
        
        # Create gripper as a kinematic body (mass=0 means kinematic, follows end effector)
        gripper_id = p.createMultiBody(
            baseMass=0,  # Kinematic body - doesn't fall due to gravity
            baseCollisionShapeIndex=palm_collision,
            baseVisualShapeIndex=palm_visual,
            basePosition=[ee_position[0], ee_position[1], ee_position[2]],
            baseOrientation=ee_orientation
        )
        
        # Create left finger
        finger_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.008, 0.008, 0.03],
            rgbaColor=[0.3, 0.3, 0.3, 1.0]
        )
        finger_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.008, 0.008, 0.03]
        )
        
        # Create left and right fingers as compound shapes within gripper (visual only)
        # These are just visuals attached to the palm
        p.changeVisualShape(
            gripper_id,
            -1,
            rgbaColor=[0.2, 0.2, 0.2, 1.0]
        )
        
        # Add spheres to represent fingers visually
        p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.012,
            rgbaColor=[0.4, 0.4, 0.4, 1.0],
            visualFramePosition=[0, 0.025, 0]
        )
        p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.012,
            rgbaColor=[0.4, 0.4, 0.4, 1.0],
            visualFramePosition=[0, -0.025, 0]
        )
        
        # Store gripper ID
        arm_state['gripper_id'] = gripper_id
        arm_state['gripper_finger_ids'] = []
        arm_state['gripper_constraint_ids'] = []
        
        # Create a single fixed constraint from end effector to gripper
        # This keeps the gripper rigidly attached to the arm
        constraint_id = p.createConstraint(
            kinovaId,
            end_effector_link,
            gripper_id,
            -1,
            p.JOINT_FIXED,
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            collideConnected=False
        )
        arm_state['gripper_constraint_ids'].append(constraint_id)
        
        return True, "Stable 2-finger gripper created and attached to end effector"
    except Exception as e:
        return False, f"Error creating gripper: {str(e)}"

def remove_gripper():
    """Remove the gripper from the scene"""
    try:
        if arm_state['gripper_id'] is None:
            return False, "No gripper is currently attached"
        
        # Remove constraints
        for constraint_id in arm_state['gripper_constraint_ids']:
            p.removeConstraint(constraint_id)
        
        # Remove gripper bodies
        p.removeBody(arm_state['gripper_id'])
        
        if 'gripper_finger_ids' in arm_state:
            for finger_id in arm_state['gripper_finger_ids']:
                try:
                    p.removeBody(finger_id)
                except:
                    pass
        
        arm_state['gripper_id'] = None
        arm_state['gripper_constraint_ids'] = []
        arm_state['gripper_finger_ids'] = []
        
        return True, "Gripper removed from scene"
    except Exception as e:
        return False, f"Error removing gripper: {str(e)}"

def load_gripper_from_urdf():
    """Load a 2-finger gripper from URDF file and attach to end effector"""
    try:
        # Get the end effector link (last link, typically)
        end_effector_link = num_joints - 1
        
        # Get the path to the gripper URDF
        gripper_urdf_path = os.path.join(script_dir, "two_finger_gripper.urdf")
        
        # Check if file exists
        if not os.path.exists(gripper_urdf_path):
            return False, f"Gripper URDF not found at {gripper_urdf_path}"
        
        # Get the position and orientation of the end effector
        ee_state = p.getLinkState(kinovaId, end_effector_link)
        ee_position = ee_state[0]
        ee_orientation = ee_state[1]
        
        # Load the gripper as a URDF
        gripper_id = p.loadURDF(
            gripper_urdf_path,
            basePosition=ee_position,
            baseOrientation=ee_orientation,
            useFixedBase=False
        )
        
        # Create a fixed constraint to attach gripper to end effector
        constraint_id = p.createConstraint(
            kinovaId,
            end_effector_link,
            gripper_id,
            -1,
            p.JOINT_FIXED,
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0]
        )
        
        # Store gripper state
        arm_state['gripper_id'] = gripper_id
        arm_state['gripper_constraint_ids'] = [constraint_id]
        
        return True, "2-finger gripper loaded from URDF and attached to end effector"
    except Exception as e:
        return False, f"Error loading gripper from URDF: {str(e)}"

# Automatically load the gripper at startup
print("\nLoading 2-finger gripper...")
try:
    success, message = load_gripper_from_urdf()
    if success:
        print(f"✓ {message}")
    else:
        print(f"✗ {message}")
        print("Gripper will need to be loaded manually or via 'load gripper' command")
except Exception as e:
    print(f"✗ Error during gripper initialization: {str(e)}")

def translate_action_type(action_type, description=""):
    """
    Translate API action types to our supported movement types.
    Maps various AI action names to our implemented movement handlers.
    """
    # Supported movement types that execute_movement() can handle
    supported_actions = {
        "home", "extend", "move_forward", "move_up", "move_down", 
        "close_gripper", "open_gripper", "rotate", "move_joint", 
        "custom_joint", "set_joint", "wait", "move_relative_to_object"
    }
    
    action_type_lower = action_type.lower().strip()
    
    # Direct match - action is already supported
    if action_type_lower in supported_actions:
        return action_type_lower, None
    
    # Mapping of similar/synonymous action types to our supported actions
    translation_map = {
        # Gripper actions
        "grasp": "close_gripper",
        "grip": "close_gripper",
        "grab": "close_gripper",
        "hold": "close_gripper",
        "clench": "close_gripper",
        "release": "open_gripper",
        "unclench": "open_gripper",
        "drop": "open_gripper",
        "let_go": "open_gripper",
        
        # Base rotation (API uses base_rotate, we use rotate)
        "base_rotate": "rotate",
        
        # Movement actions
        "reach": "move_forward",
        "reach_forward": "move_forward",
        "approach": "move_forward",
        "advance": "move_forward",
        "move_towards": "move_forward",
        "raise": "move_up",
        "lift": "move_up",
        "elevate": "move_up",
        "lower": "move_down",
        "descend": "move_down",
        "drop_to": "move_down",
        "turn": "rotate",
        "spin": "rotate",
        "rotate_base": "rotate",
        "swivel": "rotate",
        
        # Arm positioning
        "fully_extend": "extend",
        "extend_arm": "extend",
        "stretch": "extend",
        "position": "move_joint",
        "bend": "custom_joint",
        "flex": "custom_joint",
        
        # Rest/initial positions
        "retreat": "home",
        "return_to_home": "home",
        "reset": "home",
        "go_home": "home",
        
        # End-effector Cartesian commands (map to closest joint equivalent)
        "ee_set_pose": "set_joints",     # Set to specific pose
        "ee_move_linear": "move_forward", # Linear movement ~ forward
        "ee_translate": "move_forward",   # Translation ~ forward
        "ee_rotate": "rotate",            # Rotation
    }
    
    if action_type_lower in translation_map:
        translated = translation_map[action_type_lower]
        return translated, f"Translated '{action_type}' → '{translated}'"
    
    # No translation found
    return None, f"No translation found for action type: '{action_type}'. Supported: {', '.join(sorted(supported_actions))}"

def generate_ai_sequence(task_or_json, use_api=False):
    """
    Generate a sequence of movements for a given task.
    Can accept: JSON string, plain task description, or call LLM API.
    Returns: (sequence_list, error_msg) or (None, error_msg) on failure
    """
    sequence = []

    # Try to parse as JSON first
    if task_or_json.strip().startswith('{'):
        sequence_obj, error = parse_json_sequence(task_or_json)
        if error:
            return None, error
        # Preprocess the sequence to remove unexpected properties
        sequence_obj = preprocess_sequence(sequence_obj)
        # Convert JSON sequence to Movement objects for display
        if sequence_obj and "sequence" in sequence_obj:
            for action in sequence_obj["sequence"]:
                # Accept both "params" and "parameters" for flexibility
                action_params = action.get("params", action.get("parameters", {}))
                sequence.append(Movement(
                    action.get("action", "unknown"),
                    action.get("description", ""),
                    **action_params
                ))
            return sequence, None
        else:
            return None, "Invalid sequence structure in JSON"

    # If use_api is True, call LLM
    if use_api and API_KEY:
        sequence_obj, error = call_llm_api(task_or_json)
        if error:
            return None, error
        # Preprocess the sequence to remove unexpected properties
        sequence_obj = preprocess_sequence(sequence_obj)
        # Validate schema
        valid, msg = validate_sequence_schema(sequence_obj)
        if not valid:
            return None, msg
        # Convert to Movement objects with translation and logging
        if sequence_obj and "sequence" in sequence_obj:
            print("\n" + "="*70)
            print("DEBUG: ACTION TYPE TRANSLATION")
            print("="*70)
            for idx, action in enumerate(sequence_obj["sequence"], 1):
                api_action = action.get("action", "unknown")
                description = action.get("description", "")
                action_params = action.get("params", action.get("parameters", {}))
                
                # Translate action type
                translated_action, translation_note = translate_action_type(api_action, description)
                
                # Log the translation
                if translation_note:
                    print(f"{idx}. API action: '{api_action}' → {translation_note}")
                else:
                    print(f"{idx}. API action: '{api_action}' → UNSUPPORTED ⚠")
                print(f"   Description: {description}")
                print(f"   Parameters: {action_params}")
                
                # Use translated action if available, otherwise use original
                final_action = translated_action if translated_action else api_action
                sequence.append(Movement(
                    final_action,
                    description,
                    **action_params
                ))
            print("="*70 + "\n")
            return sequence, None
        else:
            return None, "Invalid sequence structure from LLM"

    # Fallback to hardcoded sequences (for testing without API)
    task = task_or_json.lower().strip()
    if task == "pick up the block" or task == "pick up block":
        sequence = [
            Movement("home", "Move arm to home position"),
            Movement("move_forward", "Move gripper forward toward the block", distance=0.2),
            Movement("move_down", "Lower gripper to block height", distance=0.1),
            Movement("close_gripper", "Close gripper around the block"),
            Movement("move_up", "Raise arm with block", distance=0.15),
            Movement("rotate", "Rotate base 90 degrees", angle=90),
        ]
    elif task == "place the block" or task == "place block":
        sequence = [
            Movement("move_forward", "Move gripper forward to placement area", distance=0.2),
            Movement("move_down", "Lower gripper to placement height", distance=0.1),
            Movement("open_gripper", "Open gripper to release the block"),
            Movement("move_up", "Raise arm after placing block", distance=0.15),
            Movement("home", "Return arm to home position"),
        ]
    elif task == "extend" or task == "extend arm":
        sequence = [
            Movement("extend", "Fully extend the arm"),
        ]
    elif task == "bend" or task == "bend arm":
        sequence = [
            Movement("custom_joint", "Create 90-degree bend", joint=3, angle=1.57),
        ]
    else:
        # Default sequence for unknown tasks
        sequence = [
            Movement("home", "Move arm to home position"),
            Movement("extend", "Extend the arm"),
            Movement("home", "Return arm to home position"),
        ]
    
    # Log when using fallback hardcoded sequence
    if not use_api:
        print("\n" + "="*70)
        print("DEBUG: USING HARDCODED FALLBACK SEQUENCE")
        print("="*70)
        print(f"Task: '{task_or_json}'")
        print(f"Movements: {len(sequence)}")
        for idx, mov in enumerate(sequence, 1):
            print(f"{idx}. {mov.move_type}: {mov.description}")
        print("="*70 + "\n")
    
    return sequence, None

def preprocess_sequence(sequence_obj):
    """Normalize sequence actions so they match the schema used by this program.

    Behavior:
    - If an action contains parameter keys at the top-level (e.g. "angle", "target"),
      move them into the `params` object.
    - Merge `parameters` into `params` if present.
    - Filter `params` to allowed keys (keeps flexible via allowed list) and leave others
      so they won't break strict schema validation.
    """
    allowed_params = {"target", "duration", "target_orientation", "target_position", "angle", "joint", "joint_index", "distance", "object_id", "relative_position", "speed"}
    for action in sequence_obj.get("sequence", []):
        # ensure params exists
        params = {}
        if "params" in action and isinstance(action["params"], dict):
            params.update(action["params"])
        # also allow 'parameters' from other agents
        if "parameters" in action and isinstance(action["parameters"], dict):
            params.update(action["parameters"])

        # move any top-level keys (other than allowed top-level ones) into params
        for k in list(action.keys()):
            if k in ("action", "description", "params", "parameters"):
                continue
            # if the key looks like a parameter, migrate it
            if k not in ("action", "description"):
                params.setdefault(k, action.pop(k))

        # Filter params to keep allowed keys only (prevents schema rejects)
        filtered = {k: v for k, v in params.items() if (k in allowed_params) or k.isidentifier()}
        action["params"] = filtered

    return sequence_obj

def display_ai_sequence(sequence, task):
    """Display the AI-generated movement sequence for user approval"""
    log_output("\n" + "="*70)
    log_output("AI-GENERATED MOVEMENT SEQUENCE")
    log_output("="*70)
    log_output(f"Task: {task}")
    log_output(f"Movements ({len(sequence)} total):")
    log_output("")
    
    for idx, movement in enumerate(sequence, 1):
        log_output(f"{idx}. {movement}")
    
    log_output("")
    log_output("Type 'go' to execute this sequence, or 'cancel' to abort.")
    log_output("="*70 + "\n")

def execute_movement(movement):
    """Execute a single movement using PyBullet with proper simulation stepping."""
    SIMULATION_STEPS = 240  # Run simulation for 1 second at 240 FPS
    MAX_FORCE = 500  # Motor force in N
    
    # Helper to set all joints (maintaining unspecified ones at current position)
    def set_all_joints(target_positions):
        """Set position control for all joints"""
        for idx, joint_idx in enumerate(joint_indices):
            p.setJointMotorControl2(
                bodyUniqueId=kinovaId,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_positions[idx],
                force=MAX_FORCE
            )
    
    # Get current positions of all joints for maintaining others
    def get_current_positions():
        """Get current positions of all joints"""
        positions = []
        for idx, joint_idx in enumerate(joint_indices):
            joint_state = p.getJointState(kinovaId, joint_idx)
            positions.append(joint_state[0])
        return positions
    
    # HIGH-LEVEL MOVEMENTS (most common)
    if movement.move_type == "home":
        log_output("    → Moving all joints to home position (rest)...")
        # Move all joints to home position (0 radians)
        targets = [0.0] * len(joint_indices)
        set_all_joints(targets)
        # Animate the movement
        for _ in range(SIMULATION_STEPS):
            p.stepSimulation()
            time.sleep(1/240.0)
    
    elif movement.move_type == "extend":
        log_output("    → Extending arm to reach position...")
        # Extend arm - match the direct 'extend' command behavior
        targets = get_current_positions()  # Get current positions first to preserve base rotation
        # Set arm joints 0-5 to extension position (1.5 radians each)
        for idx in range(min(len(joint_indices), 6)):
            targets[idx] = 1.5  # Use same value as direct command
        set_all_joints(targets)
        # Animate the movement
        for _ in range(SIMULATION_STEPS):
            p.stepSimulation()
            time.sleep(1/240.0)
    
    elif movement.move_type == "move_forward":
        distance = movement.params.get("distance", 0.1)
        log_output(f"    → Moving end effector forward by {distance:.2f}m...")
        # Move end effector forward (along X-axis)
        targets = get_current_positions()  # Maintain all current positions
        # Adjust wrist joint for forward movement
        if len(joint_indices) >= 6:
            targets[5] = targets[5] + math.radians(distance * 100)
        set_all_joints(targets)
        # Animate
        for _ in range(SIMULATION_STEPS):
            p.stepSimulation()
            time.sleep(1/240.0)
    
    elif movement.move_type == "move_up":
        distance = movement.params.get("distance", 0.1)
        log_output(f"    → Raising end effector by {distance:.2f}m...")
        # Move end effector upward (along Z-axis)
        targets = get_current_positions()  # Maintain all current positions
        # Adjust shoulder joint upward
        if len(joint_indices) >= 1:
            targets[0] = targets[0] + math.radians(distance * 50)
        set_all_joints(targets)
        # Animate
        for _ in range(SIMULATION_STEPS):
            p.stepSimulation()
            time.sleep(1/240.0)
    
    elif movement.move_type == "move_down":
        distance = movement.params.get("distance", 0.1)
        log_output(f"    → Lowering end effector by {distance:.2f}m...")
        # Move end effector downward (along -Z-axis)
        targets = get_current_positions()  # Maintain all current positions
        # Adjust shoulder joint downward
        if len(joint_indices) >= 1:
            targets[0] = targets[0] - math.radians(distance * 50)
        set_all_joints(targets)
        # Animate
        for _ in range(SIMULATION_STEPS):
            p.stepSimulation()
            time.sleep(1/240.0)
    
    elif movement.move_type == "close_gripper":
        log_output("    → Closing gripper...")
        # Maintain all current positions
        targets = get_current_positions()
        set_all_joints(targets)
        # Just run simulation for a moment to show something happened
        for _ in range(60):  # 0.25 seconds
            p.stepSimulation()
            time.sleep(1/240.0)
    
    elif movement.move_type == "open_gripper":
        log_output("    → Opening gripper...")
        # Maintain all current positions
        targets = get_current_positions()
        set_all_joints(targets)
        # Just run simulation for a moment to show something happened
        for _ in range(60):  # 0.25 seconds
            p.stepSimulation()
            time.sleep(1/240.0)
    
    # LOW-LEVEL MOVEMENTS (direct control)
    elif movement.move_type == "set_joint":
        joint_index = movement.params.get("joint_index")
        target_position = movement.params.get("target_position")
        if joint_index is not None and target_position is not None:
            angle_deg = math.degrees(target_position)
            log_output(f"    → Setting joint {joint_index} to {angle_deg:.1f}°...")
            targets = get_current_positions()
            targets[joint_index] = target_position
            set_all_joints(targets)
            # Run simulation steps to animate the movement
            for _ in range(SIMULATION_STEPS):
                p.stepSimulation()
                time.sleep(1/240.0)
    
    elif movement.move_type == "set_joint":
        joint_index = movement.params.get("joint_index")
        target_position = movement.params.get("target_position")
        if joint_index is not None and target_position is not None:
            angle_deg = math.degrees(target_position)
            log_output(f"    → Setting joint {joint_index} to {angle_deg:.1f}°...")
            targets = get_current_positions()
            targets[int(joint_index)] = target_position
            set_all_joints(targets)
            # Run simulation steps to animate the movement
            for _ in range(SIMULATION_STEPS):
                p.stepSimulation()
                time.sleep(1/240.0)
    
    elif movement.move_type == "set_joints":
        log_output("    → Setting multiple joints to target positions...")
        # Extract joint positions from params - API might use different param names
        joint_angles = (movement.params.get("joint_angles") or 
                       movement.params.get("angles") or 
                       movement.params.get("targets") or
                       movement.params.get("positions"))
        if joint_angles and isinstance(joint_angles, (list, tuple)):
            targets = get_current_positions()
            # Set the provided joints (assume they map to Joint 0, 1, 2, etc.)
            for idx, angle in enumerate(joint_angles):
                if idx < len(targets):
                    targets[idx] = angle
            set_all_joints(targets)
            # Animate
            for _ in range(SIMULATION_STEPS):
                p.stepSimulation()
                time.sleep(1/240.0)
        else:
            log_output("    ⚠ No joint angles provided - skipping")
    
    elif movement.move_type == "ee_set_pose":
        log_output("    → Moving end-effector to target pose (Cartesian command)...")
        # Cartesian command: position [x,y,z] + orientation [qx,qy,qz,qw]
        # WARNING: This is a rough approximation without full inverse kinematics
        position = movement.params.get("position")
        orientation = movement.params.get("orientation")
        if position and isinstance(position, (list, tuple)) and len(position) >= 3:
            # Simple approximation based on Z height
            targets = get_current_positions()
            z = position[2]
            # Map Z position to joint angles (very rough approximation)
            if z > 0.2:
                targets[0] = math.radians(60)   # Move shoulder up
            elif z > 0.1:
                targets[0] = math.radians(30)   # Move shoulder higher
            elif z > 0.05:
                targets[0] = -math.radians(10)  # Move shoulder down slightly
            else:
                targets[0] = -math.radians(45)  # Move shoulder down more
            set_all_joints(targets)
            # Animate
            for _ in range(SIMULATION_STEPS):
                p.stepSimulation()
                time.sleep(1/240.0)
        else:
            log_output("    ⚠ No valid position provided in ee_set_pose params")
    
    elif movement.move_type == "ee_move_linear":
        log_output("    → Moving end-effector linearly (Cartesian command)...")
        # Linear movement - API provides target position [x,y,z]
        # WARNING: Rough approximation without inverse kinematics
        position = movement.params.get("position")
        distance = movement.params.get("distance", 0.1)
        
        if position and isinstance(position, (list, tuple)) and len(position) >= 3:
            # Approximate linear motion based on position
            targets = get_current_positions()
            x, y, z = position[0], position[1], position[2]
            # Use wrist joints to approximate Cartesian movement
            if len(joint_indices) >= 6:
                # Rough: X movement affects wrist rotation, Z affects arm height
                targets[5] = targets[5] + math.radians(x * 50)
                targets[0] = targets[0] + math.radians(z * 50)
            set_all_joints(targets)
            # Animate
            for _ in range(SIMULATION_STEPS):
                p.stepSimulation()
                time.sleep(1/240.0)
        else:
            log_output("    ⚠ No valid position provided in ee_move_linear params")
                
    elif movement.move_type == "rotate":
        angle = movement.params.get("angle")
        if angle is not None:
            log_output(f"    → Rotating base by {angle:.1f}°...")
            target_rad = math.radians(angle)
            targets = get_current_positions()
            targets[0] = targets[0] + target_rad  # ADD to current position (RELATIVE rotation)
            set_all_joints(targets)
            # Run simulation steps to animate the movement
            for _ in range(SIMULATION_STEPS):
                p.stepSimulation()
                time.sleep(1/240.0)
                
    elif movement.move_type == "move_joint" or movement.move_type == "custom_joint":
        joint = movement.params.get("joint") or movement.params.get("joint_index")
        angle = movement.params.get("angle") or movement.params.get("target_position")
        if joint is not None and angle is not None:
            # Convert angle from degrees to radians if it's likely in degrees
            target_rad = math.radians(angle) if isinstance(angle, (int, float)) and abs(angle) > 6.5 else angle
            angle_deg = math.degrees(target_rad) if isinstance(target_rad, (int, float)) else 0
            log_output(f"    → Moving joint {joint} to {angle_deg:.1f}°...")
            targets = get_current_positions()
            targets[int(joint)] = target_rad
            set_all_joints(targets)
            # Run simulation steps to animate the movement
            for _ in range(SIMULATION_STEPS):
                p.stepSimulation()
                time.sleep(1/240.0)
                
    elif movement.move_type == "move_relative_to_object":
        object_id = movement.params.get("object_id")
        relative_position = movement.params.get("relative_position")
        if object_id is not None and relative_position is not None:
            log_output(f"    → Moving arm relative to object...")
            pos, _ = p.getBasePositionAndOrientation(object_id)
            new_position = [pos[i] + relative_position[i] for i in range(3)]
            p.resetBasePositionAndOrientation(
                kinovaId, new_position, [0, 0, 0, 1]
            )
            
    elif movement.move_type == "wait":
        duration = movement.params.get("duration")
        if duration is not None:
            log_output(f"    → Waiting for {duration:.2f} seconds...")
            steps = int(duration * 240)  # Convert to simulation steps
            for _ in range(steps):
                p.stepSimulation()
                time.sleep(1/240.0)
                
    else:
        # Unknown movement type - still log it
        log_output(f"    ⚠ Unknown movement type: {movement.move_type}")
        # Maintain positions
        targets = get_current_positions()
        set_all_joints(targets)
        # Still run simulation for a moment so things don't get stuck
        for _ in range(60):
            p.stepSimulation()
            time.sleep(1/240.0)

def execute_next_movement():
    """Execute the next movement in the sequence, one at a time."""
    if arm_state['ai_sequence'] is None:
        return False, "No sequence loaded"
    
    sequence = arm_state['ai_sequence']
    current_index = arm_state['ai_sequence_index']
    
    if current_index >= len(sequence):
        return False, "Sequence complete. All movements executed."
    
    try:
        movement = sequence[current_index]
        log_output(f"\nExecuting movement {current_index + 1}/{len(sequence)}:")
        log_output(f"  {movement}")
        
        execute_movement(movement)
        
        # Move to next movement
        arm_state['ai_sequence_index'] += 1
        remaining = len(sequence) - arm_state['ai_sequence_index']
        
        if remaining > 0:
            return True, f"Movement {current_index + 1}/{len(sequence)} executed. {remaining} movements remaining. Type 'go' to continue."
        else:
            return True, f"Movement {current_index + 1}/{len(sequence)} executed. Sequence complete!"
    except Exception as e:
        return False, f"Error executing movement: {str(e)}"

def execute_sequence(sequence):
    """Execute a sequence of movements."""
    try:
        for movement in sequence:
            execute_movement(movement)
        return True, f"Successfully executed {len(sequence)} movements"
    except Exception as e:
        return False, f"Error executing sequence: {str(e)}"

def log_output(msg):
    """Log output to both console and GUI"""
    print(msg)  # Always print to console
    if 'log_message' in gui_state and callable(gui_state['log_message']):
        try:
            gui_state['log_message'](msg)
        except:
            pass

def load_object(obj_type, position=None):
    """Load an object into the scene"""
    obj_type = obj_type.lower()
    
    if obj_type not in ['table', 'block', 'bottle', 'blockside', 'gripper']:
        return False, f"Unknown object type: {obj_type}. Use: table, block, bottle, blockside, or gripper"
    
    # Check if object already loaded (except gripper, which replaces existing)
    if obj_type != 'gripper' and obj_type in arm_state['loaded_objects']:
        return False, f"{obj_type.capitalize()} already loaded in scene"
    
    try:
        if obj_type == 'gripper':
            # Create and attach gripper to end effector
            success, message = create_gripper()
            return success, message
        
        elif obj_type == 'table':
            # Load table from PyBullet data
            table_height = 0.65  # Standard table height
            obj_id = p.loadURDF("table/table.urdf", [0.5, 0, 0])
            arm_state['loaded_objects'][obj_type] = obj_id
            
            # Position robot on top of the table
            p.resetBasePositionAndOrientation(kinovaId, [0, 0, table_height], [0, 0, 0, 1])
            
            return True, f"Table loaded at position (0.5, 0, 0). Robot positioned on table at height {table_height}m"
        
        elif obj_type == 'block' or obj_type == 'blockside':
            # Create a block using primitive geometry
            # Wood-like color (RGB)
            shape_id = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.05, 0.05, 0.05],
                rgbaColor=[0.6, 0.4, 0.2, 1.0]
            )
            collision_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.05, 0.05, 0.05]
            )
            
            # Set position based on object type
            if position is None:
                if obj_type == 'blockside':
                    # Position block to the side of the robot, not touching
                    position = [0.15, 0.35, 0.1]
                else:
                    # Regular block: place on table if table is loaded, otherwise error
                    if 'table' in arm_state['loaded_objects']:
                        # Get table height and place block on top
                        table_height = 0.65  # Standard table height
                        block_height = 0.05  # Half-extent of block
                        position = [0.5, 0.1, table_height + block_height]
                    else:
                        return False, "Table must be loaded first. Use: load table"
            
            obj_id = p.createMultiBody(
                baseMass=0.5,
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=shape_id,
                basePosition=position
            )
            
            # Use 'blockside' key if it's a blockside, otherwise use 'block'
            storage_key = obj_type if obj_type == 'blockside' else 'block'
            arm_state['loaded_objects'][storage_key] = obj_id
            
            pos_str = f"({position[0]}, {position[1]}, {position[2]})"
            if obj_type == 'blockside':
                return True, f"Block (side) loaded at position {pos_str}"
            else:
                return True, f"Block loaded on table at position {pos_str}"
        
        elif obj_type == 'bottle':
            # Create a cylinder to represent a water bottle
            # Blue color (water)
            shape_id = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.03,
                length=0.2,
                rgbaColor=[0.2, 0.5, 0.9, 1.0]
            )
            collision_id = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=0.03,
                height=0.2
            )
            obj_id = p.createMultiBody(
                baseMass=0.3,
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=shape_id,
                basePosition=[0.3, -0.2, 0.1]
            )
            arm_state['loaded_objects'][obj_type] = obj_id
            return True, "Water bottle loaded at position (0.3, -0.2, 0.1)"
    
    except Exception as e:
        return False, f"Error loading {obj_type}: {str(e)}"

def unload_object(obj_type):
    """Unload an object from the scene"""
    obj_type = obj_type.lower()
    
    if obj_type == 'gripper':
        # Handle gripper separately
        success, message = remove_gripper()
        return success, message
    
    if obj_type not in arm_state['loaded_objects']:
        return False, f"{obj_type.capitalize()} not loaded in scene"
    
    try:
        obj_id = arm_state['loaded_objects'][obj_type]
        p.removeBody(obj_id)
        del arm_state['loaded_objects'][obj_type]
        return True, f"{obj_type.capitalize()} removed from scene"
    except Exception as e:
        return False, f"Error removing {obj_type}: {str(e)}"

def process_command(command):
    """Process user command"""
    command = command.strip().lower()
    
    if not command:
        return True
    
    parts = command.split()
    cmd = parts[0]
    
    if cmd == "help":
        print_help()
    
    elif cmd == "quit":
        arm_state['running'] = False
        log_output("Exiting...")
        return False
    
    elif cmd == "extend":
        log_output("Extending arm...")
        targets = [0.0] * len(joint_indices)
        # Get current positions for all joints
        for idx, joint_idx in enumerate(joint_indices):
            joint_state = p.getJointState(kinovaId, joint_idx)
            targets[idx] = joint_state[0]
        # Only modify arm joints (skip base if needed), keeping current base rotation
        for idx in range(min(len(joint_indices), 6)):
            targets[idx] = 1.5
        move_to_position(targets)
        log_output("Arm extended!")
    
    elif cmd == "bend":
        log_output("Creating 90-degree bend...")
        targets = [0.0] * len(joint_indices)
        # Get current positions for all joints
        for idx, joint_idx in enumerate(joint_indices):
            joint_state = p.getJointState(kinovaId, joint_idx)
            targets[idx] = joint_state[0]
        # Modify specific joints: extend most joints and create a bend in middle
        for idx in range(min(len(joint_indices), 6)):
            if idx == len(joint_indices) // 2:
                targets[idx] = targets[idx] - (math.pi / 2)  # Create 90-degree bend
            else:
                targets[idx] = 1.5  # Extend other joints
        move_to_position(targets)
        log_output("Bend created!")
    
    elif cmd == "rotate" and len(parts) > 1:
        try:
            angle_deg = float(parts[1])
            angle_rad = math.radians(angle_deg)
            log_output(f"Rotating base joint by {angle_deg} degrees...")
            targets = [0.0] * len(joint_indices)
            # Get current positions for all joints
            for idx, joint_idx in enumerate(joint_indices):
                joint_state = p.getJointState(kinovaId, joint_idx)
                targets[idx] = joint_state[0]
            # Only modify base joint (first joint), add to current rotation
            targets[0] = targets[0] + angle_rad
            move_to_position(targets)
            log_output("Rotation complete!")
        except (ValueError, IndexError):
            log_output("Usage: rotate <angle_in_degrees>")
    
    elif cmd == "joint" and len(parts) > 2:
        try:
            joint_num = int(parts[1])
            angle_rad = float(parts[2])
            if 0 <= joint_num < len(joint_indices):
                log_output(f"Moving joint {joint_num} to {angle_rad} radians ({math.degrees(angle_rad):.1f} degrees)...")
                targets = [0.0] * len(joint_indices)
                
                # Get current positions for other joints
                for idx, joint_idx in enumerate(joint_indices):
                    if idx != joint_num:
                        joint_state = p.getJointState(kinovaId, joint_idx)
                        targets[idx] = joint_state[0]
                
                targets[joint_num] = angle_rad
                move_to_position(targets)
                log_output("Movement complete!")
            else:
                log_output(f"Joint number must be between 0 and {len(joint_indices)-1}")
        except (ValueError, IndexError):
            log_output("Usage: joint <joint_number> <angle_in_radians>")
    
    elif cmd == "reset":
        log_output("Resetting arm to rest position...")
        targets = [0.0] * len(joint_indices)
        move_to_position(targets)
        log_output("Arm reset!")
    
    elif cmd == "home":
        log_output("Moving all joints to 0 degrees...")
        targets = [0.0] * len(joint_indices)
        move_to_position(targets)
        log_output("Home position reached!")
    
    elif cmd == "speed" and len(parts) > 1:
        speed_arg = parts[1].lower()
        if speed_arg in SPEED_MULTIPLIERS:
            arm_state['speed'] = speed_arg
            multiplier = SPEED_MULTIPLIERS[speed_arg]
            duration_360 = 360.0 / BASE_ROTATION_SPEED * multiplier
            log_output(f"Speed set to {speed_arg.upper()} ({multiplier}x multiplier)")
            log_output(f"  360° rotation will take {duration_360:.1f} seconds")
        else:
            log_output(f"Invalid speed. Use: slow, medium, or fast")
            log_output(f"  slow   - 4x slower (detailed observation)")
            log_output(f"  medium - 1x baseline (smooth animation, default)")
            log_output(f"  fast   - 0.25x speed (nearly instant)")
            log_output(f"Current speed: {arm_state['speed'].upper()}")
    
    elif cmd == "load" and len(parts) > 1:
        obj_type = parts[1].lower()
        success, message = load_object(obj_type)
        if success:
            log_output(f"✓ {message}")
        else:
            log_output(f"✗ {message}")
    
    elif cmd == "unload" and len(parts) > 1:
        obj_type = parts[1].lower()
        success, message = unload_object(obj_type)
        if success:
            log_output(f"✓ {message}")
        else:
            log_output(f"✗ {message}")
    
    elif cmd == "listobjects":
        objects_list = list(arm_state['loaded_objects'].keys())
        if arm_state['gripper_id'] is not None:
            objects_list.append('gripper')
        
        if objects_list:
            log_output("Loaded objects:")
            for obj_name in objects_list:
                log_output(f"  - {obj_name.capitalize()}")
        else:
            log_output("No objects currently loaded in scene")
    
    elif cmd == "aimove" and len(parts) > 1:
        # Extract the task/JSON (everything after 'aimove')
        task_input = ' '.join(parts[1:])
        
        # Try with API first if key is set, otherwise fall back to hardcoded
        use_api = API_KEY is not None
        
        # Generate AI sequence
        sequence, error = generate_ai_sequence(task_input, use_api=use_api)
        
        if error:
            log_output(f"✗ Error generating sequence: {error}")
        else:
            # Store sequence for approval
            arm_state['ai_sequence'] = sequence
            arm_state['ai_sequence_index'] = 0
            
            # Display the sequence
            display_ai_sequence(sequence, task_input[:50])  # Truncate for display
    
    elif cmd == "go":
        if arm_state['ai_sequence'] is None:
            log_output("No AI sequence pending. Use 'aimove <task>' first.")
        else:
            success, message = execute_next_movement()
            if success:
                log_output(f"✓ {message}")
            else:
                log_output(f"✗ {message}")
    
    elif cmd == "cancel":
        if arm_state['ai_sequence'] is not None:
            log_output("AI sequence cancelled.")
            arm_state['ai_sequence'] = None
            arm_state['ai_sequence_index'] = 0
        else:
            log_output("No AI sequence to cancel.")
    
    else:
        log_output(f"Unknown command: {cmd}")
        log_output("Type 'help' for available commands")
    
    return True

# GUI state for command queue
gui_state = {
    'command_queue': queue.Queue(),
    'log_message': None
}

def create_command_window():
    """Create a separate tkinter window for command input"""
    root = tk.Tk()
    root.title("Kinova Arm Control - Command Console")
    root.geometry("600x400")
    
    # Create output text area
    output_frame = tk.Frame(root)
    output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    output_label = tk.Label(output_frame, text="Output:", font=("Courier", 10, "bold"))
    output_label.pack(anchor="w")
    
    output_text = scrolledtext.ScrolledText(output_frame, height=15, width=70, font=("Courier", 9))
    output_text.pack(fill=tk.BOTH, expand=True)
    output_text.config(state=tk.DISABLED)
    
    # Create input frame
    input_frame = tk.Frame(root)
    input_frame.pack(fill=tk.X, padx=5, pady=5)
    
    input_label = tk.Label(input_frame, text="Command:", font=("Courier", 10, "bold"))
    input_label.pack(side=tk.LEFT, padx=(0, 5))
    
    input_field = tk.Entry(input_frame, font=("Courier", 10))
    input_field.pack(side=tk.LEFT, fill=tk.X, expand=True)
    input_field.focus()
    
    def send_command(event=None):
        """Send command from input field"""
        command = input_field.get()
        if not command.strip():
            return
        
        # Add to output
        output_text.config(state=tk.NORMAL)
        output_text.insert(tk.END, f"> {command}\n")
        output_text.see(tk.END)
        output_text.config(state=tk.DISABLED)
        
        # Queue command for processing
        gui_state['command_queue'].put(command)
        input_field.delete(0, tk.END)
    
    def show_help_window():
        """Show help in output"""
        output_text.config(state=tk.NORMAL)
        output_text.insert(tk.END, "="*70 + "\n")
        output_text.insert(tk.END, "KINOVA ARM INTERACTIVE CONTROL\n")
        output_text.insert(tk.END, "="*70 + "\n\n")
        output_text.insert(tk.END, "Commands:\n")
        output_text.insert(tk.END, "  extend          - Fully extend the arm\n")
        output_text.insert(tk.END, "  bend            - Create a 90-degree bend\n")
        output_text.insert(tk.END, "  rotate <angle>  - Rotate base by angle in degrees\n")
        output_text.insert(tk.END, "  joint <n> <a>   - Move joint n to angle a in radians\n")
        output_text.insert(tk.END, "  reset           - Return arm to rest position\n")
        output_text.insert(tk.END, "  speed <preset>  - Set speed: slow/medium/fast\n")
        output_text.insert(tk.END, "  home            - Move all joints to 0 degrees\n")
        output_text.insert(tk.END, "  help            - Show this help message\n")
        output_text.insert(tk.END, "  quit            - Exit the program\n\n")
        output_text.insert(tk.END, "Object Management:\n")
        output_text.insert(tk.END, "  load <type>   - Load object: table/block/bottle/blockside/gripper\n")
        output_text.insert(tk.END, "  unload <type> - Remove object from scene\n")
        output_text.insert(tk.END, "  listobjects         - List loaded objects\n\n")
        output_text.insert(tk.END, "AI Movement Planning:\n")
        output_text.insert(tk.END, "  aimove <task>  - Generate AI sequence (e.g., 'aimove pick up the block')\n")
        output_text.insert(tk.END, "  go             - Execute pending AI sequence\n")
        output_text.insert(tk.END, "  cancel         - Cancel pending AI sequence\n\n")
        output_text.insert(tk.END, "Speed Presets:\n")
        output_text.insert(tk.END, "  slow   - 4x slower (12 sec for 360°)\n")
        output_text.insert(tk.END, "  medium - 1x baseline (3 sec for 360°, default)\n")
        output_text.insert(tk.END, "  fast   - 0.25x speed (0.75 sec for 360°)\n")
        output_text.insert(tk.END, "="*70 + "\n\n")
        output_text.see(tk.END)
        output_text.config(state=tk.DISABLED)
    
    def log_message(msg):
        """Log a message to the output window"""
        output_text.config(state=tk.NORMAL)
        output_text.insert(tk.END, msg + "\n")
        output_text.see(tk.END)
        output_text.config(state=tk.DISABLED)
        root.update()
    
    # Bind Enter key
    input_field.bind('<Return>', send_command)
    
    # Store logging function for later use
    gui_state['log_message'] = log_message
    
    # Show help on startup
    show_help_window()
    
    return root, input_field

def update_gui():
    """Process GUI events and commands"""
    try:
        # Process any queued commands
        while not gui_state['command_queue'].empty():
            command = gui_state['command_queue'].get_nowait()
            try:
                if not process_command(command):
                    return False
            except Exception as e:
                print(f"Error processing command '{command}': {e}")
                if 'log_message' in gui_state:
                    gui_state['log_message'](f"Error: {e}")
    except queue.Empty:
        pass
    
    return True

print("\nKinova Gen 3 Interactive Control Started!")
print("Creating command window...")

# Create the tkinter command window
root = None
input_field = None
try:
    root, input_field = create_command_window()
    print("Command window created successfully!")
except Exception as e:
    print(f"Error creating command window: {e}")
    import traceback
    traceback.print_exc()
    p.disconnect()
    exit()

# Main simulation loop
print("Simulation running...")
try:
    while arm_state['running'] and p.isConnected():
        # Update GUI and process commands
        try:
            if not update_gui():
                arm_state['running'] = False
                break
        except Exception as e:
            print(f"Error updating GUI: {e}")
            import traceback
            traceback.print_exc()
        
        # Update movement animation
        try:
            update_movement()
        except Exception as e:
            print(f"Error updating movement: {e}")
            import traceback
            traceback.print_exc()
        
        # Process tkinter events
        try:
            if root:
                root.update()
        except tk.TclError:
            # Window was closed
            print("Command window closed")
            arm_state['running'] = False
            break
        except Exception as e:
            print(f"GUI update error: {e}")
        
        p.stepSimulation()
        time.sleep(1/240)
except KeyboardInterrupt:
    print("\nSimulation interrupted by user")
    arm_state['running'] = False
except Exception as e:
    print(f"Error during simulation: {e}")
    import traceback
    traceback.print_exc()
finally:
    try:
        if root:
            root.destroy()
    except:
        pass
    p.disconnect()
    print("PyBullet disconnected!")
