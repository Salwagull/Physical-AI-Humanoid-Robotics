# VLA (Vision-Language-Action) Examples

This directory contains Python examples demonstrating Vision-Language-Action system integration for intelligent robot control.

## Overview

VLA systems enable robots to understand natural language commands in the context of their visual environment and execute appropriate actions. The components work together:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Vision    │────▶│  Language   │────▶│   Action    │
│  Detector   │     │   Parser    │     │   Planner   │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
   Objects &           Intent &            Motion
   Locations           Entities           Commands
```

## Prerequisites

### Required Packages

```bash
# Core dependencies
pip install numpy opencv-python

# For vision detection
pip install ultralytics  # YOLOv8

# For language parsing (optional but recommended)
pip install spacy
python -m spacy download en_core_web_sm

# For ROS 2 integration
# Requires ROS 2 Humble or later
```

### Hardware Requirements

- Webcam or camera for vision module
- NVIDIA GPU (optional, for faster YOLO inference)

## Files

### 1. `vision_detector.py` - Object Detection

Real-time object detection using YOLOv8 for robotic perception.

**Features:**
- Detect 80 COCO object classes
- Find specific objects by name
- Track object positions
- Annotate images with detections

**Usage:**

```bash
# Live webcam demo
python vision_detector.py

# Process single image
python vision_detector.py path/to/image.jpg
```

**API Example:**

```python
from vision_detector import VisionDetector

detector = VisionDetector()

# Detect all objects
detections = detector.detect(camera_image)

# Find specific object
cup = detector.find_object(camera_image, "cup")
if cup:
    print(f"Cup found at {cup.center}")
```

### 2. `language_parser.py` - Command Understanding

Natural language parser for robot commands.

**Features:**
- Action verb extraction (pick, place, move, etc.)
- Object and location identification
- Modifier parsing (colors, sizes)
- Compound command handling

**Usage:**

```bash
# Demo with test commands
python language_parser.py

# Interactive mode
python language_parser.py --interactive
```

**API Example:**

```python
from language_parser import LanguageParser, RobotAction

parser = LanguageParser()

# Parse single command
result = parser.parse("Pick up the red cup")
print(result.action)         # RobotAction.PICK
print(result.target_object)  # "cup"
print(result.modifiers)      # {"color": "red"}

# Parse compound command
commands = parser.parse_compound("Grab the bottle and put it on the shelf")
for cmd in commands:
    print(cmd.describe())
```

**Supported Actions:**

| Action | Example Commands |
|--------|-----------------|
| PICK   | "pick up", "grab", "take", "get" |
| PLACE  | "place", "put", "set down" |
| MOVE   | "move to", "go to", "navigate" |
| LOOK   | "look for", "find", "search" |
| OPEN   | "open the door/drawer" |
| CLOSE  | "close", "shut" |
| POUR   | "pour water into" |
| STOP   | "stop", "halt", "freeze" |

### 3. `voice_robot_control.py` - Complete Voice-Controlled Robot

End-to-end voice command to robot action system integrating all VLA components.

**Features:**
- Speech recognition with OpenAI Whisper
- LLM planning (OpenAI GPT or local Ollama)
- Ambiguity detection and resolution
- Task decomposition and planning
- ROS 2 integration

**Usage:**

```bash
# Demo mode (no hardware required)
python voice_robot_control.py --demo

# Interactive mode
python voice_robot_control.py --interactive

# With local LLM (Ollama)
python voice_robot_control.py --local --interactive

# As ROS 2 node
python voice_robot_control.py --ros2
```

**API Example:**

```python
from voice_robot_control import VoiceControlledRobot, SceneObject

robot = VoiceControlledRobot(use_openai=True)

# Set up scene
robot.update_scene([
    SceneObject("cup", (0.5, 0.2, 0.1), 0.95, color="red"),
    SceneObject("bottle", (0.3, 0.0, 0.15), 0.88),
])

# Process voice command
plan = robot.process_command("Pick up the red cup")

# Execute plan
robot.execute_plan(plan)
```

**ROS 2 Topics:**
- `/speech/text` (subscribe) - Voice commands as text
- `/camera/rgb/image_raw` (subscribe) - Camera input
- `/cmd_vel` (publish) - Velocity commands
- `/arm/goal` (publish) - Arm pose goals
- `/robot/speech` (publish) - Robot speech output
- `/robot/status` (publish) - Execution status

## Quick Start

### 1. Test Vision Detection

```bash
# Requires webcam
python vision_detector.py
```

Press 'q' to quit. The detector will show bounding boxes around detected objects.

### 2. Test Language Parsing

```bash
python language_parser.py --interactive
```

Type commands like:
- "Pick up the red cup"
- "Move to the kitchen"
- "Find the bottle and bring it here"

### 3. Combine Vision + Language

```python
from vision_detector import VisionDetector
from language_parser import LanguageParser

detector = VisionDetector()
parser = LanguageParser()

# User says: "Pick up the bottle"
command = parser.parse("Pick up the bottle")

# Find the bottle in camera image
bottle = detector.find_object(camera_image, command.target_object)

if bottle:
    print(f"Found {command.target_object} at pixel {bottle.center}")
    # Convert to 3D position and execute pick action
else:
    print(f"Cannot find {command.target_object}")
```

## VLA Pipeline Architecture

```
User Voice Command: "Pick up the red cup on the left"
                            │
                            ▼
                ┌───────────────────────┐
                │   Speech Recognition  │
                │   (Google/Whisper)    │
                └───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   Language Parser     │
                │   ─────────────────   │
                │   Action: PICK        │
                │   Object: cup         │
                │   Color: red          │
                │   Position: left      │
                └───────────────────────┘
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
          ▼                                   ▼
┌─────────────────────┐           ┌─────────────────────┐
│   Vision Detector   │           │    World Model      │
│   ─────────────────│           │   ─────────────────  │
│   Find all cups     │           │   Known locations   │
│   Filter by color   │           │   Object positions  │
│   Get 3D position   │           │   Robot state       │
└─────────────────────┘           └─────────────────────┘
          │                                   │
          └─────────────────┬─────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │    Action Planner     │
                │   ─────────────────   │
                │   1. Look at cup      │
                │   2. Approach         │
                │   3. Pre-grasp pose   │
                │   4. Grasp            │
                │   5. Lift             │
                └───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │    Robot Controller   │
                │   (ROS 2 / MoveIt2)   │
                └───────────────────────┘
                            │
                            ▼
                    Robot Executes
```

## Integration with Isaac Sim

For simulation-based VLA testing, see the `isaac_sim_examples/` directory.

```python
# In Isaac Sim
from omni.isaac.sensor import Camera

# Get camera image
camera = Camera(prim_path="/World/Camera")
rgba = camera.get_rgba()
rgb = rgba[:, :, :3]

# Use with VLA detector
from vla_examples.vision_detector import VisionDetector
detector = VisionDetector()
detections = detector.detect(rgb)
```

## Extending the System

### Adding New Actions

Edit `language_parser.py`:

```python
# In ACTION_PATTERNS dictionary
RobotAction.DANCE: r"\b(dance|boogie|move to music)\b",

# Add to RobotAction enum
DANCE = "dance"
```

### Adding New Objects

Edit `vision_detector.py`:

```python
# In ROBOT_TARGET_CLASSES list
ROBOT_TARGET_CLASSES = [
    # ... existing objects ...
    "custom_object",
]
```

### Custom Object Detection

Train a custom YOLO model:

```bash
# Using ultralytics
yolo train data=custom.yaml model=yolov8n.pt epochs=100
```

Then load in detector:

```python
detector = VisionDetector(model_path="path/to/best.pt")
```

## Troubleshooting

### "ultralytics not installed"

```bash
pip install ultralytics
```

### "spaCy model not found"

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### "Camera not found"

- Check webcam is connected: `ls /dev/video*` (Linux)
- Try different camera index: `cv2.VideoCapture(1)`

### "YOLO model download failed"

Models download automatically on first use. If blocked:
1. Download manually from https://github.com/ultralytics/assets/releases
2. Place `yolov8n.pt` in working directory

## Resources

- [Chapter 8: Vision-Language-Action Systems](../docs/chapter8_vla_systems.md)
- [Chapter 9: LLM Planning and Voice Commands](../docs/chapter9_llm_voice_commands.md)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [spaCy Documentation](https://spacy.io/)
- [Google RT-2 Paper](https://robotics-transformer2.github.io/)

## License

These examples are provided as educational material for the Physical AI & Humanoid Robotics book.
