#!/usr/bin/env python3
"""
Voice-Controlled Robot System
============================

Complete VLA integration example combining:
- Speech recognition (Whisper)
- LLM planning (OpenAI GPT / Ollama)
- Vision detection (YOLO)
- Language parsing
- ROS 2 robot control

This module demonstrates end-to-end voice command to robot action.

Chapter 9: LLM Planning and Voice Commands for Robots
Physical AI & Humanoid Robotics Book

Usage:
    # Basic demo (no ROS 2)
    python voice_robot_control.py --demo

    # With ROS 2 (requires ROS 2 environment)
    ros2 run vla_examples voice_robot_control

    # Interactive mode
    python voice_robot_control.py --interactive

Requirements:
    pip install openai-whisper sounddevice numpy opencv-python
    pip install openai  # For OpenAI LLM
    # OR
    # Ollama installed locally for local LLM
"""

import os
import sys
import json
import time
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Callable
from abc import ABC, abstractmethod

import numpy as np

# Optional imports with graceful fallback
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: whisper not installed. Speech recognition disabled.")

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not installed. Microphone input disabled.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# =============================================================================
# Data Classes and Enums
# =============================================================================

class RobotAction(Enum):
    """Robot action types."""
    PICK = "pick"
    PLACE = "place"
    MOVE = "move"
    LOOK = "look"
    NAVIGATE = "navigate"
    OPEN = "open"
    CLOSE = "close"
    POUR = "pour"
    PUSH = "push"
    PULL = "pull"
    ROTATE = "rotate"
    WAVE = "wave"
    STOP = "stop"
    WAIT = "wait"
    UNKNOWN = "unknown"


@dataclass
class ParsedCommand:
    """Parsed robot command from natural language."""
    action: RobotAction
    target_object: Optional[str] = None
    destination: Optional[str] = None
    modifiers: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    original_text: str = ""

    def describe(self) -> str:
        """Human-readable description."""
        parts = [f"Action: {self.action.value}"]
        if self.target_object:
            parts.append(f"Target: {self.target_object}")
        if self.destination:
            parts.append(f"Destination: {self.destination}")
        if self.modifiers:
            mods = ", ".join(f"{k}={v}" for k, v in self.modifiers.items())
            parts.append(f"Modifiers: {mods}")
        return " | ".join(parts)


@dataclass
class TaskStep:
    """Single step in a task plan."""
    action: str
    parameters: Dict[str, Any]
    description: str
    preconditions: List[str] = field(default_factory=list)
    expected_outcome: str = ""


@dataclass
class TaskPlan:
    """Complete task plan from LLM."""
    goal: str
    steps: List[TaskStep]
    estimated_duration: float = 0.0
    confidence: float = 1.0
    fallback_plan: Optional['TaskPlan'] = None


@dataclass
class SceneObject:
    """Object detected in the scene."""
    name: str
    position: tuple  # (x, y, z) in robot frame
    confidence: float
    color: Optional[str] = None
    size: Optional[str] = None


@dataclass
class SceneState:
    """Current state of the robot's environment."""
    objects: List[SceneObject] = field(default_factory=list)
    robot_position: tuple = (0.0, 0.0, 0.0)
    gripper_state: str = "open"
    held_object: Optional[str] = None
    timestamp: float = 0.0


# =============================================================================
# Speech Recognition
# =============================================================================

class SpeechRecognizer:
    """Whisper-based speech recognition."""

    def __init__(self, model_size: str = "base"):
        """
        Initialize speech recognizer.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.model = None
        self.model_size = model_size
        self.sample_rate = 16000
        self.audio_queue = queue.Queue()
        self.is_listening = False

        if WHISPER_AVAILABLE:
            print(f"Loading Whisper {model_size} model...")
            self.model = whisper.load_model(model_size)
            print("Whisper model loaded.")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio samples as numpy array (float32, mono, 16kHz)

        Returns:
            Transcribed text
        """
        if self.model is None:
            return ""

        # Ensure correct format
        audio = audio.astype(np.float32)

        # Run transcription
        result = self.model.transcribe(
            audio,
            language="en",
            fp16=False  # CPU compatible
        )

        return result["text"].strip()

    def listen_once(self, duration: float = 5.0) -> str:
        """
        Record and transcribe a single utterance.

        Args:
            duration: Recording duration in seconds

        Returns:
            Transcribed text
        """
        if not SOUNDDEVICE_AVAILABLE:
            return input("Speech input unavailable. Type command: ")

        print(f"Listening for {duration} seconds...")

        # Record audio
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()

        # Flatten to 1D
        audio = audio.flatten()

        # Transcribe
        text = self.transcribe(audio)
        print(f"Heard: '{text}'")

        return text

    def start_continuous(self, callback: Callable[[str], None]):
        """Start continuous listening with callback for each utterance."""
        self.is_listening = True

        def listen_loop():
            while self.is_listening:
                text = self.listen_once(duration=3.0)
                if text:
                    callback(text)

        thread = threading.Thread(target=listen_loop, daemon=True)
        thread.start()

    def stop_continuous(self):
        """Stop continuous listening."""
        self.is_listening = False


# =============================================================================
# LLM Planning
# =============================================================================

class LLMPlanner(ABC):
    """Abstract base class for LLM-based planners."""

    @abstractmethod
    def plan(self, command: str, scene: SceneState) -> TaskPlan:
        """Generate a task plan from natural language command."""
        pass

    @abstractmethod
    def clarify(self, command: str, ambiguities: List[str]) -> str:
        """Generate clarification question for ambiguous command."""
        pass


class OpenAIPlanner(LLMPlanner):
    """Task planner using OpenAI GPT models."""

    SYSTEM_PROMPT = """You are a robot task planner. Given a natural language command and scene description,
generate a step-by-step plan the robot can execute.

Output JSON with this structure:
{
    "goal": "high-level goal description",
    "steps": [
        {
            "action": "action_name",
            "parameters": {"param1": "value1"},
            "description": "what this step does",
            "preconditions": ["condition1"],
            "expected_outcome": "what should happen"
        }
    ],
    "confidence": 0.95
}

Available actions: pick, place, move, look, navigate, open, close, pour, push, pull, rotate, wave, stop, wait
Always check preconditions before actions (e.g., gripper must be empty before pick)."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Initialize OpenAI planner.

        Args:
            model: OpenAI model to use
            api_key: API key (or set OPENAI_API_KEY env var)
        """
        self.model = model
        self.client = None

        if OPENAI_AVAILABLE:
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)

    def _format_scene(self, scene: SceneState) -> str:
        """Format scene state for prompt."""
        lines = ["Current scene:"]
        lines.append(f"- Robot position: {scene.robot_position}")
        lines.append(f"- Gripper: {scene.gripper_state}")
        if scene.held_object:
            lines.append(f"- Holding: {scene.held_object}")

        if scene.objects:
            lines.append("- Visible objects:")
            for obj in scene.objects:
                pos_str = f"({obj.position[0]:.2f}, {obj.position[1]:.2f}, {obj.position[2]:.2f})"
                lines.append(f"  - {obj.name} at {pos_str} (conf: {obj.confidence:.2f})")
        else:
            lines.append("- No objects detected")

        return "\n".join(lines)

    def plan(self, command: str, scene: SceneState) -> TaskPlan:
        """Generate task plan using GPT."""
        if self.client is None:
            # Return simple fallback plan
            return self._fallback_plan(command)

        scene_desc = self._format_scene(scene)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Command: {command}\n\n{scene_desc}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )

        result = json.loads(response.choices[0].message.content)

        steps = [
            TaskStep(
                action=s["action"],
                parameters=s.get("parameters", {}),
                description=s.get("description", ""),
                preconditions=s.get("preconditions", []),
                expected_outcome=s.get("expected_outcome", "")
            )
            for s in result.get("steps", [])
        ]

        return TaskPlan(
            goal=result.get("goal", command),
            steps=steps,
            confidence=result.get("confidence", 0.8)
        )

    def clarify(self, command: str, ambiguities: List[str]) -> str:
        """Generate clarification question."""
        if self.client is None:
            return f"Please clarify: {', '.join(ambiguities)}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Generate a brief, friendly clarification question."},
                {"role": "user", "content": f"Command: {command}\nAmbiguous aspects: {ambiguities}"}
            ],
            temperature=0.7,
            max_tokens=100
        )

        return response.choices[0].message.content

    def _fallback_plan(self, command: str) -> TaskPlan:
        """Simple rule-based fallback when LLM unavailable."""
        command_lower = command.lower()

        steps = []
        if "pick" in command_lower or "grab" in command_lower:
            steps = [
                TaskStep("look", {"target": "object"}, "Locate target object"),
                TaskStep("navigate", {"target": "object"}, "Move to object"),
                TaskStep("pick", {"target": "object"}, "Grasp object"),
            ]
        elif "place" in command_lower or "put" in command_lower:
            steps = [
                TaskStep("navigate", {"target": "destination"}, "Move to destination"),
                TaskStep("place", {}, "Release object"),
            ]
        elif "move" in command_lower or "go" in command_lower:
            steps = [
                TaskStep("navigate", {"target": "destination"}, "Navigate to target"),
            ]
        else:
            steps = [
                TaskStep("wait", {}, "Unknown command - waiting"),
            ]

        return TaskPlan(goal=command, steps=steps, confidence=0.5)


class OllamaPlanner(LLMPlanner):
    """Task planner using local Ollama models."""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama planner.

        Args:
            model: Ollama model name
            base_url: Ollama API URL
        """
        self.model = model
        self.base_url = base_url

    def plan(self, command: str, scene: SceneState) -> TaskPlan:
        """Generate task plan using Ollama."""
        if not REQUESTS_AVAILABLE:
            return TaskPlan(goal=command, steps=[], confidence=0.0)

        prompt = f"""Generate a robot task plan for: "{command}"

Output as JSON with steps array containing action, parameters, description."""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=30
            )

            result = response.json()
            plan_data = json.loads(result.get("response", "{}"))

            steps = [
                TaskStep(
                    action=s.get("action", "unknown"),
                    parameters=s.get("parameters", {}),
                    description=s.get("description", "")
                )
                for s in plan_data.get("steps", [])
            ]

            return TaskPlan(goal=command, steps=steps)

        except Exception as e:
            print(f"Ollama error: {e}")
            return TaskPlan(goal=command, steps=[], confidence=0.0)

    def clarify(self, command: str, ambiguities: List[str]) -> str:
        """Generate clarification using Ollama."""
        return f"Could you clarify: {', '.join(ambiguities)}?"


# =============================================================================
# Ambiguity Detection
# =============================================================================

class AmbiguityDetector:
    """Detect and resolve ambiguous commands."""

    def __init__(self):
        self.ambiguity_patterns = {
            "missing_object": ["pick", "grab", "take", "get"],
            "missing_location": ["put", "place", "move", "go"],
            "vague_reference": ["it", "that", "this", "there", "here"],
        }

    def detect(self, command: str, scene: SceneState) -> List[str]:
        """
        Detect ambiguities in command.

        Args:
            command: Natural language command
            scene: Current scene state

        Returns:
            List of detected ambiguities
        """
        ambiguities = []
        words = command.lower().split()

        # Check for missing object
        for action in self.ambiguity_patterns["missing_object"]:
            if action in words:
                # Check if any object is mentioned
                has_object = any(
                    obj.name.lower() in command.lower()
                    for obj in scene.objects
                )
                if not has_object and "it" not in words and "that" not in words:
                    ambiguities.append(f"What should I {action}?")

        # Check for vague references
        for vague in self.ambiguity_patterns["vague_reference"]:
            if vague in words:
                # Count matching objects
                matching = [o for o in scene.objects]
                if len(matching) > 1:
                    ambiguities.append(f"'{vague}' is ambiguous - multiple objects visible")

        # Check for duplicate objects
        object_counts = {}
        for obj in scene.objects:
            name = obj.name.lower()
            object_counts[name] = object_counts.get(name, 0) + 1

        for name, count in object_counts.items():
            if count > 1 and name in command.lower():
                # Check if distinguishing modifier present
                has_modifier = any(
                    mod in command.lower()
                    for mod in ["red", "blue", "green", "left", "right", "big", "small"]
                )
                if not has_modifier:
                    ambiguities.append(f"Which {name}? There are {count} visible.")

        return ambiguities

    def resolve(self, ambiguities: List[str], user_response: str) -> Dict[str, str]:
        """
        Resolve ambiguities based on user response.

        Args:
            ambiguities: List of ambiguity questions
            user_response: User's clarification

        Returns:
            Resolution mapping
        """
        resolutions = {}
        response_lower = user_response.lower()

        # Extract color if mentioned
        colors = ["red", "blue", "green", "yellow", "orange", "white", "black"]
        for color in colors:
            if color in response_lower:
                resolutions["color"] = color

        # Extract position if mentioned
        positions = ["left", "right", "front", "back", "closest", "farthest"]
        for pos in positions:
            if pos in response_lower:
                resolutions["position"] = pos

        # Extract object name
        words = response_lower.split()
        for word in words:
            if word not in colors + positions + ["the", "a", "an", "one"]:
                resolutions["object"] = word
                break

        return resolutions


# =============================================================================
# Voice-Controlled Robot (Main Integration)
# =============================================================================

class VoiceControlledRobot:
    """
    Complete voice-controlled robot system.

    Integrates speech recognition, LLM planning, and command execution.
    """

    def __init__(
        self,
        use_openai: bool = True,
        whisper_model: str = "base",
        llm_model: str = "gpt-4o-mini"
    ):
        """
        Initialize voice-controlled robot.

        Args:
            use_openai: Use OpenAI (True) or Ollama (False)
            whisper_model: Whisper model size
            llm_model: LLM model name
        """
        # Initialize components
        self.speech = SpeechRecognizer(model_size=whisper_model)

        if use_openai:
            self.planner = OpenAIPlanner(model=llm_model)
        else:
            self.planner = OllamaPlanner(model=llm_model)

        self.ambiguity_detector = AmbiguityDetector()

        # State
        self.scene = SceneState()
        self.current_plan: Optional[TaskPlan] = None
        self.is_executing = False

        # Callbacks
        self.on_command: Optional[Callable[[str], None]] = None
        self.on_plan: Optional[Callable[[TaskPlan], None]] = None
        self.on_execute: Optional[Callable[[TaskStep], None]] = None
        self.on_clarify: Optional[Callable[[str], str]] = None

    def update_scene(self, objects: List[SceneObject]):
        """Update scene with detected objects."""
        self.scene.objects = objects
        self.scene.timestamp = time.time()

    def process_command(self, command: str) -> Optional[TaskPlan]:
        """
        Process a voice command end-to-end.

        Args:
            command: Natural language command

        Returns:
            Generated task plan (or None if clarification needed)
        """
        print(f"\n[VoiceRobot] Processing: '{command}'")

        if self.on_command:
            self.on_command(command)

        # Check for ambiguities
        ambiguities = self.ambiguity_detector.detect(command, self.scene)

        if ambiguities:
            print(f"[VoiceRobot] Ambiguities detected: {ambiguities}")

            # Get clarification
            question = self.planner.clarify(command, ambiguities)
            print(f"[VoiceRobot] Asking: {question}")

            if self.on_clarify:
                response = self.on_clarify(question)
                resolutions = self.ambiguity_detector.resolve(ambiguities, response)

                # Enhance command with resolutions
                enhanced = command
                if "color" in resolutions:
                    enhanced = f"{resolutions['color']} " + enhanced
                if "object" in resolutions:
                    enhanced = enhanced.replace("it", resolutions["object"])

                command = enhanced
                print(f"[VoiceRobot] Enhanced command: '{command}'")

        # Generate plan
        plan = self.planner.plan(command, self.scene)
        self.current_plan = plan

        print(f"[VoiceRobot] Plan generated: {plan.goal}")
        for i, step in enumerate(plan.steps):
            print(f"  {i+1}. {step.action}: {step.description}")

        if self.on_plan:
            self.on_plan(plan)

        return plan

    def execute_plan(self, plan: Optional[TaskPlan] = None):
        """
        Execute a task plan.

        Args:
            plan: Plan to execute (or use current_plan)
        """
        plan = plan or self.current_plan
        if not plan:
            print("[VoiceRobot] No plan to execute")
            return

        self.is_executing = True
        print(f"\n[VoiceRobot] Executing plan: {plan.goal}")

        for i, step in enumerate(plan.steps):
            if not self.is_executing:
                print("[VoiceRobot] Execution stopped")
                break

            print(f"[VoiceRobot] Step {i+1}/{len(plan.steps)}: {step.action}")
            print(f"  Description: {step.description}")
            print(f"  Parameters: {step.parameters}")

            if self.on_execute:
                self.on_execute(step)

            # Simulate execution time
            time.sleep(0.5)

        self.is_executing = False
        print("[VoiceRobot] Plan execution complete")

    def stop(self):
        """Stop current execution."""
        self.is_executing = False
        print("[VoiceRobot] Stop requested")

    def run_interactive(self):
        """Run interactive command loop."""
        print("\n" + "="*60)
        print("Voice-Controlled Robot - Interactive Mode")
        print("="*60)
        print("Commands: Type or speak commands. Type 'quit' to exit.")
        print("Example: 'Pick up the red cup'")
        print("="*60 + "\n")

        # Set up clarification callback
        self.on_clarify = lambda q: input(f"Robot asks: {q}\nYour response: ")

        # Add some test objects
        self.update_scene([
            SceneObject("cup", (0.5, 0.2, 0.1), 0.95, color="red"),
            SceneObject("cup", (0.3, -0.1, 0.1), 0.90, color="blue"),
            SceneObject("bottle", (0.4, 0.0, 0.15), 0.88),
            SceneObject("book", (0.6, 0.3, 0.05), 0.92),
        ])

        while True:
            try:
                # Get command (text input for demo)
                command = input("\nYou: ").strip()

                if not command:
                    continue

                if command.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                if command.lower() == "scene":
                    print("\nCurrent scene:")
                    for obj in self.scene.objects:
                        print(f"  - {obj.name} ({obj.color or 'unknown'}) at {obj.position}")
                    continue

                if command.lower() == "listen":
                    command = self.speech.listen_once(duration=5.0)
                    if not command:
                        print("No speech detected.")
                        continue

                # Process and execute
                plan = self.process_command(command)
                if plan and plan.steps:
                    execute = input("\nExecute plan? (y/n): ").strip().lower()
                    if execute == "y":
                        self.execute_plan(plan)

            except KeyboardInterrupt:
                print("\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


# =============================================================================
# ROS 2 Integration (Optional)
# =============================================================================

def create_ros2_node():
    """
    Create ROS 2 node for voice-controlled robot.

    Returns ROS 2 node class if rclpy available, else None.
    """
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
        from geometry_msgs.msg import Twist, PoseStamped
        from sensor_msgs.msg import Image
    except ImportError:
        return None

    class VoiceRobotNode(Node):
        """ROS 2 node for voice-controlled robot."""

        def __init__(self):
            super().__init__('voice_robot_node')

            # Initialize robot system
            self.robot = VoiceControlledRobot(use_openai=True)

            # Publishers
            self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
            self.arm_goal_pub = self.create_publisher(PoseStamped, '/arm/goal', 10)
            self.speech_pub = self.create_publisher(String, '/robot/speech', 10)
            self.status_pub = self.create_publisher(String, '/robot/status', 10)

            # Subscribers
            self.create_subscription(String, '/speech/text', self.speech_callback, 10)
            self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)

            # Set up callbacks
            self.robot.on_execute = self.execute_step
            self.robot.on_plan = self.publish_plan

            self.get_logger().info('Voice Robot Node initialized')

        def speech_callback(self, msg: String):
            """Handle incoming speech text."""
            command = msg.data
            self.get_logger().info(f'Received command: {command}')

            plan = self.robot.process_command(command)
            if plan:
                self.robot.execute_plan(plan)

        def image_callback(self, msg: Image):
            """Handle camera images for object detection."""
            # Convert ROS Image to numpy (simplified)
            pass

        def execute_step(self, step: TaskStep):
            """Execute a single plan step."""
            self.get_logger().info(f'Executing: {step.action}')

            # Publish status
            status = String()
            status.data = f"Executing: {step.description}"
            self.status_pub.publish(status)

            # Execute based on action type
            if step.action == "navigate":
                twist = Twist()
                twist.linear.x = 0.2
                self.cmd_vel_pub.publish(twist)

            elif step.action in ["pick", "place"]:
                pose = PoseStamped()
                pose.header.frame_id = "base_link"
                # Set pose from parameters
                self.arm_goal_pub.publish(pose)

        def publish_plan(self, plan: TaskPlan):
            """Publish plan for visualization."""
            msg = String()
            msg.data = json.dumps({
                "goal": plan.goal,
                "steps": [s.description for s in plan.steps]
            })
            self.status_pub.publish(msg)

    return VoiceRobotNode


# =============================================================================
# Demo and Main
# =============================================================================

def run_demo():
    """Run demonstration without hardware."""
    print("\n" + "="*60)
    print("Voice-Controlled Robot - Demo Mode")
    print("="*60)

    # Create robot without requiring APIs
    robot = VoiceControlledRobot(use_openai=False)

    # Set up mock scene
    robot.update_scene([
        SceneObject("cup", (0.5, 0.2, 0.1), 0.95, color="red"),
        SceneObject("cup", (0.3, -0.1, 0.1), 0.90, color="blue"),
        SceneObject("bottle", (0.4, 0.0, 0.15), 0.88),
    ])

    # Demo commands
    demo_commands = [
        "Pick up the red cup",
        "Move to the kitchen",
        "Find the bottle",
    ]

    for cmd in demo_commands:
        print(f"\n{'='*40}")
        print(f"Demo command: '{cmd}'")
        print('='*40)

        plan = robot.process_command(cmd)
        if plan:
            print(f"\nGenerated plan ({plan.confidence:.0%} confidence):")
            for i, step in enumerate(plan.steps, 1):
                print(f"  {i}. [{step.action}] {step.description}")

        time.sleep(1)

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Voice-Controlled Robot System")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--ros2", action="store_true", help="Run as ROS 2 node")
    parser.add_argument("--whisper", default="base", help="Whisper model size")
    parser.add_argument("--llm", default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--local", action="store_true", help="Use local Ollama instead of OpenAI")

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.ros2:
        NodeClass = create_ros2_node()
        if NodeClass is None:
            print("Error: ROS 2 (rclpy) not available")
            sys.exit(1)

        import rclpy
        rclpy.init()
        node = NodeClass()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        # Default to interactive mode
        robot = VoiceControlledRobot(
            use_openai=not args.local,
            whisper_model=args.whisper,
            llm_model=args.llm
        )
        robot.run_interactive()


if __name__ == "__main__":
    main()
