---
sidebar_position: 8
title: "Chapter 8: Vision-Language-Action Systems"
description: "Build VLA systems combining computer vision, natural language processing, and action planning for intelligent robot control"
keywords: [vla, vision, language, action, yolo, object detection, nlp, spacy, robot control, ros2]
---

# Chapter 8: Vision-Language-Action Systems

## Learning Objectives

By the end of this chapter, you will:

- Understand the architecture and components of Vision-Language-Action (VLA) systems
- Implement vision processing pipelines for robotic perception
- Integrate language models for understanding natural language commands
- Build action planning systems that translate commands into robot motions
- Connect VLA components with ROS 2 for end-to-end robot control

## Introduction to VLA Systems

**Vision-Language-Action (VLA)** systems represent the cutting edge of Physical AI, enabling robots to understand their environment through vision, receive instructions in natural language, and execute appropriate actions. These systems bridge the gap between human intent and robot behavior.

### Why VLA Matters for Physical AI

Traditional robots require precise programming for every task. VLA systems enable:

- **Natural interaction**: Communicate with robots using everyday language
- **Flexible task execution**: Handle novel situations without reprogramming
- **Contextual understanding**: Consider visual context when interpreting commands
- **Generalization**: Apply learned behaviors to new environments and objects

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Vision-Language-Action Pipeline                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────┐      ┌──────────────┐      ┌─────────────┐                │
│  │  Vision │─────▶│   Language   │─────▶│   Action    │                │
│  │ Module  │      │    Model     │      │  Planner    │                │
│  └─────────┘      └──────────────┘      └─────────────┘                │
│       │                 │                     │                         │
│       ▼                 ▼                     ▼                         │
│  ┌─────────┐      ┌──────────────┐      ┌─────────────┐                │
│  │ Object  │      │   Intent     │      │   Motion    │                │
│  │Detection│      │ Recognition  │      │  Commands   │                │
│  └─────────┘      └──────────────┘      └─────────────┘                │
│       │                 │                     │                         │
│       └─────────────────┴─────────────────────┘                         │
│                         │                                                │
│                         ▼                                                │
│                  ┌─────────────┐                                        │
│                  │  ROS 2      │                                        │
│                  │  Robot      │                                        │
│                  │  Control    │                                        │
│                  └─────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## VLA Architecture Overview

A complete VLA system consists of three interconnected modules:

### 1. Vision Module

The vision module processes visual input from cameras to understand the robot's environment.

**Key capabilities:**
- **Object detection**: Identify and locate objects (YOLO, DETR)
- **Semantic segmentation**: Classify every pixel in an image
- **Depth estimation**: Understand 3D scene structure
- **Scene understanding**: Recognize spatial relationships

### 2. Language Module

The language module interprets natural language commands and extracts actionable intent.

**Key capabilities:**
- **Command parsing**: Extract verbs, objects, and modifiers
- **Intent classification**: Determine what action is requested
- **Entity extraction**: Identify referenced objects and locations
- **Disambiguation**: Resolve unclear references using context

### 3. Action Module

The action module translates understood intent into executable robot commands.

**Key capabilities:**
- **Task planning**: Break commands into action sequences
- **Motion planning**: Generate collision-free trajectories
- **Skill execution**: Perform atomic actions (grasp, move, place)
- **Feedback processing**: Adjust based on execution results

## Vision Processing for Robotic Perception

### Object Detection with YOLO

**YOLO (You Only Look Once)** is a real-time object detection system ideal for robotics.

```python
#!/usr/bin/env python3
"""
Vision module for VLA system using YOLOv8 for object detection.

Prerequisites:
    pip install ultralytics opencv-python

Usage:
    python vision_detector.py
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Try to import ultralytics, provide fallback
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


@dataclass
class Detection:
    """Represents a detected object."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int


class VisionDetector:
    """
    Vision module for detecting objects in camera images.

    Uses YOLOv8 for real-time object detection with support for
    80 COCO classes including common household objects.
    """

    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initialize the vision detector.

        Args:
            model_path: Path to YOLO model (downloads automatically if not found)
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics package required. Install with: pip install ultralytics")

        self.model = YOLO(model_path)
        self.confidence_threshold = 0.5

        # Common objects robots interact with
        self.target_classes = [
            "bottle", "cup", "bowl", "apple", "banana", "orange",
            "book", "keyboard", "mouse", "remote", "cell phone",
            "chair", "couch", "potted plant", "bed", "dining table",
            "laptop", "scissors", "toothbrush", "teddy bear"
        ]

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect objects in an image.

        Args:
            image: BGR image from camera (OpenCV format)

        Returns:
            List of Detection objects
        """
        results = self.model(image, verbose=False)[0]
        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])

            if confidence < self.confidence_threshold:
                continue

            # Get bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Calculate center and area
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)

            detections.append(Detection(
                class_name=class_name,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                center=(center_x, center_y),
                area=area
            ))

        return detections

    def find_object(self, image: np.ndarray, object_name: str) -> Optional[Detection]:
        """
        Find a specific object in the image.

        Args:
            image: Camera image
            object_name: Name of object to find (e.g., "cup", "bottle")

        Returns:
            Detection if found, None otherwise
        """
        detections = self.detect(image)

        # Find best match for requested object
        matches = [d for d in detections if object_name.lower() in d.class_name.lower()]

        if not matches:
            return None

        # Return highest confidence match
        return max(matches, key=lambda d: d.confidence)

    def annotate_image(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return annotated


def demo_vision_detector():
    """Demonstrate the vision detector with webcam."""
    detector = VisionDetector()
    cap = cv2.VideoCapture(0)

    print("Vision Detector Demo")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        detections = detector.detect(frame)

        # Annotate frame
        annotated = detector.annotate_image(frame, detections)

        # Display
        cv2.imshow("VLA Vision Module", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo_vision_detector()
```

### Depth Perception for 3D Understanding

Robots need depth information to plan movements in 3D space.

```python
"""
Depth estimation for VLA systems using MiDaS or camera depth sensors.
"""

import numpy as np
from typing import Tuple, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DepthEstimator:
    """
    Estimates depth from RGB images using MiDaS or processes
    depth camera data directly.
    """

    def __init__(self, use_gpu: bool = True):
        """Initialize depth estimator."""
        self.device = "cuda" if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.model = None
        self.transform = None

    def load_midas(self, model_type: str = "MiDaS_small"):
        """
        Load MiDaS depth estimation model.

        Args:
            model_type: "DPT_Large", "DPT_Hybrid", or "MiDaS_small"
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for MiDaS. Install with: pip install torch")

        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device).eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

    def estimate_depth(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from RGB image.

        Args:
            rgb_image: RGB image (H, W, 3)

        Returns:
            Depth map (H, W) with relative depth values
        """
        if self.model is None:
            self.load_midas()

        import cv2

        # Transform image
        input_batch = self.transform(rgb_image).to(self.device)

        # Predict
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb_image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map

    def get_3d_position(
        self,
        pixel: Tuple[int, int],
        depth_value: float,
        camera_intrinsics: dict
    ) -> Tuple[float, float, float]:
        """
        Convert 2D pixel + depth to 3D position.

        Args:
            pixel: (x, y) pixel coordinates
            depth_value: Depth in meters
            camera_intrinsics: Camera parameters (fx, fy, cx, cy)

        Returns:
            (X, Y, Z) position in camera frame (meters)
        """
        x, y = pixel
        fx = camera_intrinsics["fx"]
        fy = camera_intrinsics["fy"]
        cx = camera_intrinsics["cx"]
        cy = camera_intrinsics["cy"]

        # Back-project to 3D
        Z = depth_value
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy

        return (X, Y, Z)
```

## Language Understanding for Robot Commands

### Intent Classification and Entity Extraction

The language module parses natural language commands to understand what the user wants.

```python
#!/usr/bin/env python3
"""
Language understanding module for VLA systems.

Parses natural language commands into structured intents and entities.

Prerequisites:
    pip install spacy
    python -m spacy download en_core_web_sm
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class RobotAction(Enum):
    """Supported robot actions."""
    PICK = "pick"
    PLACE = "place"
    MOVE = "move"
    PUSH = "push"
    POUR = "pour"
    OPEN = "open"
    CLOSE = "close"
    POINT = "point"
    LOOK = "look"
    STOP = "stop"
    UNKNOWN = "unknown"


@dataclass
class ParsedCommand:
    """Structured representation of a parsed command."""
    action: RobotAction
    target_object: Optional[str]
    target_location: Optional[str]
    modifiers: Dict[str, Any]
    confidence: float
    original_text: str


class LanguageParser:
    """
    Parses natural language commands for robot control.

    Supports commands like:
    - "Pick up the red cup"
    - "Move to the table"
    - "Place the bottle on the shelf"
    - "Pour water into the glass"
    """

    # Action verb patterns
    ACTION_PATTERNS = {
        RobotAction.PICK: r"\b(pick|grab|grasp|take|get|lift)\b",
        RobotAction.PLACE: r"\b(place|put|set|drop|release)\b",
        RobotAction.MOVE: r"\b(move|go|navigate|drive|travel)\b",
        RobotAction.PUSH: r"\b(push|shove|slide)\b",
        RobotAction.POUR: r"\b(pour|fill|empty)\b",
        RobotAction.OPEN: r"\b(open|unlock)\b",
        RobotAction.CLOSE: r"\b(close|shut|lock)\b",
        RobotAction.POINT: r"\b(point|indicate|show)\b",
        RobotAction.LOOK: r"\b(look|see|find|locate|search)\b",
        RobotAction.STOP: r"\b(stop|halt|freeze|pause)\b",
    }

    # Location prepositions
    LOCATION_PREPS = ["on", "to", "into", "onto", "in", "at", "near", "by", "beside", "next to"]

    # Color modifiers
    COLORS = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "brown"]

    # Size modifiers
    SIZES = ["big", "small", "large", "tiny", "medium", "tall", "short"]

    def __init__(self):
        """Initialize the language parser."""
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy model not found. Using regex-only parsing.")

    def parse(self, command: str) -> ParsedCommand:
        """
        Parse a natural language command.

        Args:
            command: Natural language command (e.g., "Pick up the red cup")

        Returns:
            ParsedCommand with extracted action, object, location, and modifiers
        """
        command_lower = command.lower().strip()

        # Extract action
        action = self._extract_action(command_lower)

        # Extract modifiers (colors, sizes)
        modifiers = self._extract_modifiers(command_lower)

        # Extract target object and location
        target_object = self._extract_object(command_lower, action)
        target_location = self._extract_location(command_lower)

        # Calculate confidence based on extraction success
        confidence = self._calculate_confidence(action, target_object, target_location)

        return ParsedCommand(
            action=action,
            target_object=target_object,
            target_location=target_location,
            modifiers=modifiers,
            confidence=confidence,
            original_text=command
        )

    def _extract_action(self, text: str) -> RobotAction:
        """Extract the primary action from the command."""
        for action, pattern in self.ACTION_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return action
        return RobotAction.UNKNOWN

    def _extract_modifiers(self, text: str) -> Dict[str, Any]:
        """Extract color, size, and other modifiers."""
        modifiers = {}

        for color in self.COLORS:
            if color in text:
                modifiers["color"] = color
                break

        for size in self.SIZES:
            if size in text:
                modifiers["size"] = size
                break

        return modifiers

    def _extract_object(self, text: str, action: RobotAction) -> Optional[str]:
        """Extract the target object from the command."""
        if self.nlp:
            return self._extract_object_spacy(text)
        return self._extract_object_regex(text, action)

    def _extract_object_regex(self, text: str, action: RobotAction) -> Optional[str]:
        """Regex-based object extraction."""
        # Common objects
        objects = [
            "cup", "bottle", "glass", "mug", "bowl", "plate",
            "book", "phone", "remote", "pen", "pencil",
            "apple", "banana", "orange", "ball", "box",
            "chair", "table", "door", "drawer", "shelf"
        ]

        for obj in objects:
            if obj in text:
                return obj

        # Pattern: "the [adjective] [noun]"
        pattern = r"the\s+(?:\w+\s+)?(\w+)(?:\s|$)"
        match = re.search(pattern, text)
        if match:
            return match.group(1)

        return None

    def _extract_object_spacy(self, text: str) -> Optional[str]:
        """spaCy-based object extraction using NLP."""
        doc = self.nlp(text)

        # Find direct objects
        for token in doc:
            if token.dep_ == "dobj" or token.dep_ == "pobj":
                return token.text

        # Fall back to nouns
        nouns = [token.text for token in doc if token.pos_ == "NOUN"]
        return nouns[0] if nouns else None

    def _extract_location(self, text: str) -> Optional[str]:
        """Extract target location from the command."""
        for prep in self.LOCATION_PREPS:
            pattern = rf"{prep}\s+(?:the\s+)?(\w+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _calculate_confidence(
        self,
        action: RobotAction,
        target_object: Optional[str],
        target_location: Optional[str]
    ) -> float:
        """Calculate confidence score for the parse."""
        score = 0.0

        if action != RobotAction.UNKNOWN:
            score += 0.5

        if target_object:
            score += 0.3

        if target_location:
            score += 0.2

        return min(score, 1.0)


def demo_language_parser():
    """Demonstrate the language parser."""
    parser = LanguageParser()

    test_commands = [
        "Pick up the red cup",
        "Move to the kitchen table",
        "Place the bottle on the shelf",
        "Find the blue ball",
        "Pour water into the glass",
        "Open the drawer",
        "Look at the apple",
    ]

    print("Language Parser Demo")
    print("=" * 60)

    for cmd in test_commands:
        result = parser.parse(cmd)
        print(f"\nCommand: \"{cmd}\"")
        print(f"  Action: {result.action.value}")
        print(f"  Object: {result.target_object}")
        print(f"  Location: {result.target_location}")
        print(f"  Modifiers: {result.modifiers}")
        print(f"  Confidence: {result.confidence:.2f}")


if __name__ == "__main__":
    demo_language_parser()
```

## Action Planning and Execution

### From Intent to Robot Motion

The action planner translates parsed commands into executable robot behaviors.

```python
#!/usr/bin/env python3
"""
Action planning module for VLA systems.

Translates parsed commands into sequences of robot actions.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import numpy as np


class SkillType(Enum):
    """Primitive robot skills."""
    NAVIGATE = "navigate"
    APPROACH = "approach"
    GRASP = "grasp"
    LIFT = "lift"
    MOVE_ARM = "move_arm"
    RELEASE = "release"
    ROTATE = "rotate"
    LOOK_AT = "look_at"
    WAIT = "wait"


@dataclass
class Skill:
    """A primitive robot skill with parameters."""
    skill_type: SkillType
    parameters: dict
    duration_estimate: float  # seconds
    preconditions: List[str]
    effects: List[str]


@dataclass
class ActionPlan:
    """A sequence of skills to execute."""
    skills: List[Skill]
    total_duration: float
    success_probability: float
    fallback_plan: Optional['ActionPlan'] = None


class ActionPlanner:
    """
    Plans sequences of robot actions to accomplish goals.

    Uses a skill-based approach where complex tasks are decomposed
    into primitive skills that the robot can execute.
    """

    def __init__(self, robot_capabilities: dict = None):
        """
        Initialize the action planner.

        Args:
            robot_capabilities: Dict describing robot's capabilities
        """
        self.capabilities = robot_capabilities or self._default_capabilities()
        self.world_state = {}  # Tracked world state

    def _default_capabilities(self) -> dict:
        """Default capabilities for a mobile manipulator."""
        return {
            "has_gripper": True,
            "has_mobile_base": True,
            "max_reach": 0.8,  # meters
            "gripper_width": 0.1,  # meters
            "navigation_speed": 0.5,  # m/s
            "arm_speed": 0.3,  # m/s
        }

    def plan_pick(
        self,
        object_name: str,
        object_position: Tuple[float, float, float]
    ) -> ActionPlan:
        """
        Plan a pick/grasp action.

        Args:
            object_name: Name of object to pick
            object_position: (x, y, z) position of object

        Returns:
            ActionPlan for picking the object
        """
        skills = []

        # 1. Look at the object
        skills.append(Skill(
            skill_type=SkillType.LOOK_AT,
            parameters={"target": object_position},
            duration_estimate=0.5,
            preconditions=[],
            effects=["object_visible"]
        ))

        # 2. Approach if needed (check if object is within reach)
        distance = np.linalg.norm(np.array(object_position[:2]))
        if distance > self.capabilities["max_reach"]:
            approach_pos = self._calculate_approach_position(object_position)
            skills.append(Skill(
                skill_type=SkillType.NAVIGATE,
                parameters={"target": approach_pos},
                duration_estimate=distance / self.capabilities["navigation_speed"],
                preconditions=[],
                effects=["at_approach_position"]
            ))

        # 3. Move arm to pre-grasp pose
        pre_grasp = (object_position[0], object_position[1], object_position[2] + 0.1)
        skills.append(Skill(
            skill_type=SkillType.MOVE_ARM,
            parameters={"target": pre_grasp, "gripper_open": True},
            duration_estimate=1.5,
            preconditions=["at_approach_position"],
            effects=["arm_at_pregrasp"]
        ))

        # 4. Move to grasp pose
        skills.append(Skill(
            skill_type=SkillType.MOVE_ARM,
            parameters={"target": object_position, "gripper_open": True},
            duration_estimate=0.5,
            preconditions=["arm_at_pregrasp"],
            effects=["arm_at_grasp"]
        ))

        # 5. Close gripper
        skills.append(Skill(
            skill_type=SkillType.GRASP,
            parameters={"force": 20.0},  # Newtons
            duration_estimate=0.5,
            preconditions=["arm_at_grasp"],
            effects=["object_grasped"]
        ))

        # 6. Lift object
        lift_pos = (object_position[0], object_position[1], object_position[2] + 0.15)
        skills.append(Skill(
            skill_type=SkillType.LIFT,
            parameters={"target": lift_pos},
            duration_estimate=0.5,
            preconditions=["object_grasped"],
            effects=["object_lifted"]
        ))

        total_duration = sum(s.duration_estimate for s in skills)

        return ActionPlan(
            skills=skills,
            total_duration=total_duration,
            success_probability=0.85
        )

    def plan_place(
        self,
        target_position: Tuple[float, float, float]
    ) -> ActionPlan:
        """Plan a place action (assumes object is already grasped)."""
        skills = []

        # 1. Navigate if needed
        distance = np.linalg.norm(np.array(target_position[:2]))
        if distance > self.capabilities["max_reach"]:
            approach_pos = self._calculate_approach_position(target_position)
            skills.append(Skill(
                skill_type=SkillType.NAVIGATE,
                parameters={"target": approach_pos},
                duration_estimate=distance / self.capabilities["navigation_speed"],
                preconditions=["object_grasped"],
                effects=["at_place_position"]
            ))

        # 2. Move arm to pre-place pose
        pre_place = (target_position[0], target_position[1], target_position[2] + 0.1)
        skills.append(Skill(
            skill_type=SkillType.MOVE_ARM,
            parameters={"target": pre_place},
            duration_estimate=1.5,
            preconditions=["object_grasped"],
            effects=["arm_at_preplace"]
        ))

        # 3. Move to place pose
        skills.append(Skill(
            skill_type=SkillType.MOVE_ARM,
            parameters={"target": target_position},
            duration_estimate=0.5,
            preconditions=["arm_at_preplace"],
            effects=["arm_at_place"]
        ))

        # 4. Release object
        skills.append(Skill(
            skill_type=SkillType.RELEASE,
            parameters={},
            duration_estimate=0.3,
            preconditions=["arm_at_place"],
            effects=["object_placed"]
        ))

        # 5. Retract arm
        retract_pos = (target_position[0], target_position[1], target_position[2] + 0.15)
        skills.append(Skill(
            skill_type=SkillType.MOVE_ARM,
            parameters={"target": retract_pos},
            duration_estimate=0.5,
            preconditions=["object_placed"],
            effects=["arm_retracted"]
        ))

        total_duration = sum(s.duration_estimate for s in skills)

        return ActionPlan(
            skills=skills,
            total_duration=total_duration,
            success_probability=0.90
        )

    def plan_navigate(
        self,
        target_location: str,
        location_positions: dict
    ) -> ActionPlan:
        """Plan navigation to a named location."""
        if target_location not in location_positions:
            # Return a search plan
            return self._plan_search(target_location)

        target_pos = location_positions[target_location]

        skills = [
            Skill(
                skill_type=SkillType.NAVIGATE,
                parameters={"target": target_pos, "avoid_obstacles": True},
                duration_estimate=5.0,  # Estimate
                preconditions=[],
                effects=[f"at_{target_location}"]
            )
        ]

        return ActionPlan(
            skills=skills,
            total_duration=5.0,
            success_probability=0.95
        )

    def _calculate_approach_position(
        self,
        target: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Calculate position to approach an object from."""
        # Position robot 0.5m from target
        approach_distance = 0.5
        direction = np.array(target[:2]) / (np.linalg.norm(target[:2]) + 1e-6)
        approach_pos = np.array(target[:2]) - direction * approach_distance
        return (approach_pos[0], approach_pos[1], 0.0)

    def _plan_search(self, target: str) -> ActionPlan:
        """Plan a search behavior for an unknown location."""
        skills = [
            Skill(
                skill_type=SkillType.ROTATE,
                parameters={"angle": 360, "search_for": target},
                duration_estimate=10.0,
                preconditions=[],
                effects=["searched_area"]
            )
        ]
        return ActionPlan(
            skills=skills,
            total_duration=10.0,
            success_probability=0.7
        )


def demo_action_planner():
    """Demonstrate the action planner."""
    planner = ActionPlanner()

    print("Action Planner Demo")
    print("=" * 60)

    # Plan a pick action
    object_pos = (0.5, 0.3, 0.1)
    pick_plan = planner.plan_pick("cup", object_pos)

    print("\nPick Plan for 'cup' at (0.5, 0.3, 0.1):")
    print(f"  Total duration: {pick_plan.total_duration:.1f}s")
    print(f"  Success probability: {pick_plan.success_probability:.0%}")
    print(f"  Skills ({len(pick_plan.skills)}):")
    for i, skill in enumerate(pick_plan.skills):
        print(f"    {i+1}. {skill.skill_type.value}: {skill.parameters}")


if __name__ == "__main__":
    demo_action_planner()
```

## Integrating VLA with ROS 2

The complete VLA system runs as interconnected ROS 2 nodes:

```python
#!/usr/bin/env python3
"""
Complete VLA ROS 2 Integration Node

This node integrates vision, language, and action modules
for end-to-end voice-controlled robot operation.

Topics:
    Subscribed:
        /camera/rgb/image_raw (sensor_msgs/Image)
        /speech/text (std_msgs/String)

    Published:
        /vla/command (std_msgs/String)
        /vla/status (std_msgs/String)
        /cmd_vel (geometry_msgs/Twist)
        /arm/target_pose (geometry_msgs/Pose)

Prerequisites:
    pip install ultralytics opencv-python
    ros2 run your_package vla_node

Author: Physical AI & Humanoid Robotics Book
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from cv_bridge import CvBridge
import numpy as np
from typing import Optional

# Import our VLA modules (from previous code)
# from .vision_detector import VisionDetector, Detection
# from .language_parser import LanguageParser, ParsedCommand, RobotAction
# from .action_planner import ActionPlanner, ActionPlan


class VLANode(Node):
    """
    Vision-Language-Action ROS 2 node.

    Processes camera images and voice commands to control
    a robot using the VLA pipeline.
    """

    def __init__(self):
        super().__init__("vla_node")

        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()

        # Initialize VLA modules
        # self.vision = VisionDetector()
        # self.language = LanguageParser()
        # self.planner = ActionPlanner()

        # State
        self.latest_image: Optional[np.ndarray] = None
        self.latest_detections = []
        self.current_plan: Optional[any] = None
        self.executing = False

        # Known locations (would be learned or configured)
        self.locations = {
            "table": (1.0, 0.0, 0.0),
            "shelf": (0.0, 1.0, 0.5),
            "kitchen": (2.0, 2.0, 0.0),
        }

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            "/camera/rgb/image_raw",
            self.image_callback,
            10
        )

        self.speech_sub = self.create_subscription(
            String,
            "/speech/text",
            self.speech_callback,
            10
        )

        # Publishers
        self.status_pub = self.create_publisher(String, "/vla/status", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.arm_pose_pub = self.create_publisher(Pose, "/arm/target_pose", 10)

        # Timer for vision processing (10 Hz)
        self.vision_timer = self.create_timer(0.1, self.process_vision)

        self.get_logger().info("VLA Node initialized")
        self.publish_status("Ready for commands")

    def image_callback(self, msg: Image):
        """Handle incoming camera images."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    def process_vision(self):
        """Process latest image for object detection."""
        if self.latest_image is None:
            return

        # Run object detection
        # self.latest_detections = self.vision.detect(self.latest_image)
        pass

    def speech_callback(self, msg: String):
        """Handle voice commands."""
        command_text = msg.data
        self.get_logger().info(f"Received command: {command_text}")

        # Parse the command
        # parsed = self.language.parse(command_text)
        # self.handle_parsed_command(parsed)

        # Simplified demo handling
        self.handle_command_demo(command_text)

    def handle_command_demo(self, command: str):
        """Simplified command handling for demonstration."""
        command_lower = command.lower()

        if "stop" in command_lower:
            self.stop_robot()
            self.publish_status("Stopped")
            return

        if "forward" in command_lower or "move" in command_lower:
            twist = Twist()
            twist.linear.x = 0.5
            self.cmd_vel_pub.publish(twist)
            self.publish_status("Moving forward")
            return

        if "turn" in command_lower or "rotate" in command_lower:
            twist = Twist()
            twist.angular.z = 0.5 if "left" in command_lower else -0.5
            self.cmd_vel_pub.publish(twist)
            self.publish_status("Turning")
            return

        if "pick" in command_lower or "grab" in command_lower:
            self.publish_status("Planning pick action...")
            # Would trigger pick action plan
            return

        self.publish_status(f"Unknown command: {command}")

    def stop_robot(self):
        """Stop all robot motion."""
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        self.executing = False

    def publish_status(self, status: str):
        """Publish VLA system status."""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f"Status: {status}")


def main(args=None):
    rclpy.init(args=args)
    node = VLANode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

## VLA System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        VLA System Architecture                            │
└──────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   User Voice    │
                              │    Command      │
                              └────────┬────────┘
                                       │
                                       ▼
┌────────────────┐           ┌─────────────────┐           ┌────────────────┐
│   Camera       │──────────▶│   VLA Node      │──────────▶│   Robot        │
│   /camera/rgb  │           │                 │           │   /cmd_vel     │
└────────────────┘           │  ┌───────────┐  │           │   /arm/pose    │
                             │  │  Vision   │  │           └────────────────┘
                             │  │  Detector │  │
                             │  └─────┬─────┘  │
                             │        │        │
                             │  ┌─────▼─────┐  │
┌────────────────┐           │  │ Language  │  │
│   Microphone   │──────────▶│  │  Parser   │  │
│   /speech/text │           │  └─────┬─────┘  │
└────────────────┘           │        │        │
                             │  ┌─────▼─────┐  │
                             │  │  Action   │  │
                             │  │  Planner  │  │
                             │  └───────────┘  │
                             │                 │
                             └─────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   ROS 2 Topics  │
                              │   & Services    │
                              └─────────────────┘
```

## Summary

In this chapter, you learned:

✅ **VLA architecture**: Vision, Language, and Action modules work together for intelligent robot control

✅ **Vision processing**: Object detection with YOLO provides scene understanding for robots

✅ **Language parsing**: Extract actions, objects, and locations from natural language commands

✅ **Action planning**: Decompose high-level commands into executable robot skills

✅ **ROS 2 integration**: Connect VLA components through topics and services

✅ **End-to-end pipeline**: Camera input → Command parsing → Action execution

### Key Takeaways

- VLA systems enable **natural human-robot interaction** through language
- **Object detection** provides spatial grounding for language references
- **Intent classification** maps natural language to robot actions
- **Skill-based planning** breaks complex tasks into primitive actions
- ROS 2 provides the **communication infrastructure** for VLA components

## Exercises

### Exercise 1: Extend Object Detection

1. Modify `VisionDetector` to track objects across frames
2. Add distance estimation using object size
3. Implement a "find all red objects" capability

### Exercise 2: Improve Language Understanding

1. Add support for compound commands: "Pick up the cup and place it on the table"
2. Handle pronouns: "Pick it up" (referring to previously mentioned object)
3. Add confirmation for ambiguous commands

### Exercise 3: Build a Pick-and-Place Demo

1. Create a complete pick-and-place pipeline
2. Use simulated vision (hardcoded detections) if camera unavailable
3. Test with commands: "Pick up the bottle", "Place it on the shelf"

### Challenge: Add Error Recovery

Implement a robust VLA system that:
1. Detects when grasp fails (object not acquired)
2. Re-plans and retries with different approach
3. Reports failure to user after max retries

## Up Next

In **Chapter 9: LLM Planning and Voice Commands**, we'll enhance our VLA system with:
- Large Language Model (LLM) integration for complex reasoning
- Voice command processing with speech recognition
- Multi-step task planning using LLM capabilities
- Handling ambiguous and context-dependent commands

## Additional Resources

- [OpenAI CLIP](https://openai.com/research/clip) - Vision-Language Foundation Models
- [RT-2: Vision-Language-Action Models](https://robotics-transformer2.github.io/) - Google DeepMind
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [spaCy NLP Library](https://spacy.io/)
- [ROS 2 Navigation Stack](https://navigation.ros.org/)

---

**Sources:**
- [Physical Intelligence Pi0](https://www.physicalintelligence.company/) - VLA Research
- [Google RT-2 Paper](https://arxiv.org/abs/2307.15818) - Vision-Language-Action Models
- [OpenAI Research](https://openai.com/research) - Multimodal Learning
- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
