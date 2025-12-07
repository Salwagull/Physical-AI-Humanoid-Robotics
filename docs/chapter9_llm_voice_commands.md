---
sidebar_position: 9
title: "Chapter 9: LLM Planning and Voice Commands"
description: "Integrate LLMs for task planning and build voice-controlled robot systems with speech recognition"
keywords: [llm, voice commands, speech recognition, whisper, openai, ollama, task planning, ros2]
---

# Chapter 9: LLM Planning and Voice Commands for Robots

## Learning Objectives

By the end of this chapter, you will:

- Integrate Large Language Models (LLMs) for high-level robot task planning
- Build a voice command processing pipeline with speech recognition
- Translate natural language into executable robot actions
- Handle ambiguous and context-dependent commands
- Create a complete voice-controlled robot system with ROS 2

## Introduction

While Chapter 8 introduced the fundamentals of Vision-Language-Action systems, this chapter takes robot intelligence to the next level by integrating **Large Language Models (LLMs)** for sophisticated reasoning and **voice commands** for natural human-robot interaction.

### Why LLMs for Robotics?

Traditional robot programming requires explicit instructions for every scenario. LLMs enable:

- **Complex reasoning**: Break down multi-step tasks automatically
- **Context understanding**: Remember conversation history and scene state
- **Ambiguity resolution**: Ask clarifying questions when commands are unclear
- **Generalization**: Handle novel situations without reprogramming
- **Natural dialogue**: Communicate with users conversationally

```
┌─────────────────────────────────────────────────────────────────────────┐
│              LLM-Powered Voice Control Pipeline                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│   │  Voice  │───▶│  Speech │───▶│   LLM   │───▶│ Action  │             │
│   │  Input  │    │  to Text│    │ Planner │    │Executor │             │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘             │
│                                      │                                   │
│                                      ▼                                   │
│                               ┌─────────────┐                           │
│                               │   Context   │                           │
│                               │   Memory    │                           │
│                               │ (Scene, History)                        │
│                               └─────────────┘                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Voice Command Processing Pipeline

### Speech Recognition with Whisper

**OpenAI Whisper** provides state-of-the-art speech-to-text for robot commands.

```python
#!/usr/bin/env python3
"""
Speech Recognition Module for Voice-Controlled Robots

Uses OpenAI Whisper for accurate speech-to-text conversion.

Prerequisites:
    pip install openai-whisper sounddevice numpy

Usage:
    python speech_recognition.py
"""

import numpy as np
import queue
import threading
from dataclasses import dataclass
from typing import Optional, Callable

try:
    import whisper
    import sounddevice as sd
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: Install dependencies with:")
    print("  pip install openai-whisper sounddevice numpy")


@dataclass
class SpeechResult:
    """Result from speech recognition."""
    text: str
    confidence: float
    language: str
    duration: float


class SpeechRecognizer:
    """
    Real-time speech recognition using Whisper.

    Supports continuous listening with wake word detection
    or push-to-talk operation.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str = "en",
        wake_word: Optional[str] = None
    ):
        """
        Initialize speech recognizer.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            language: Target language code
            wake_word: Optional wake word to trigger listening (e.g., "robot")
        """
        if not WHISPER_AVAILABLE:
            raise RuntimeError("Whisper not installed. Run: pip install openai-whisper")

        print(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        self.language = language
        self.wake_word = wake_word.lower() if wake_word else None

        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.float32

        # Recording state
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.is_recording = False

        print(f"Speech recognizer ready (language: {language})")

    def transcribe(self, audio: np.ndarray) -> SpeechResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio samples as numpy array (16kHz, mono)

        Returns:
            SpeechResult with transcription
        """
        # Normalize audio
        audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0

        # Run Whisper
        result = self.model.transcribe(
            audio,
            language=self.language,
            fp16=False  # Use FP32 for CPU compatibility
        )

        return SpeechResult(
            text=result["text"].strip(),
            confidence=1.0,  # Whisper doesn't provide confidence
            language=result.get("language", self.language),
            duration=len(audio) / self.sample_rate
        )

    def record_audio(self, duration: float = 5.0) -> np.ndarray:
        """
        Record audio from microphone.

        Args:
            duration: Recording duration in seconds

        Returns:
            Audio samples as numpy array
        """
        print(f"Recording for {duration}s... Speak now!")

        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype
        )
        sd.wait()

        print("Recording complete")
        return recording.flatten()

    def listen_once(self, duration: float = 5.0) -> SpeechResult:
        """
        Record and transcribe a single utterance.

        Args:
            duration: Max recording duration

        Returns:
            SpeechResult with transcription
        """
        audio = self.record_audio(duration)
        return self.transcribe(audio)

    def start_continuous_listening(
        self,
        callback: Callable[[str], None],
        chunk_duration: float = 3.0
    ):
        """
        Start continuous listening with callback.

        Args:
            callback: Function called with transcribed text
            chunk_duration: Duration of each audio chunk
        """
        self.is_listening = True

        def listen_thread():
            while self.is_listening:
                try:
                    audio = self.record_audio(chunk_duration)
                    result = self.transcribe(audio)

                    if result.text:
                        # Check for wake word if configured
                        if self.wake_word:
                            if self.wake_word in result.text.lower():
                                # Remove wake word and process command
                                command = result.text.lower().replace(self.wake_word, "").strip()
                                if command:
                                    callback(command)
                        else:
                            callback(result.text)

                except Exception as e:
                    print(f"Listening error: {e}")

        thread = threading.Thread(target=listen_thread, daemon=True)
        thread.start()

    def stop_listening(self):
        """Stop continuous listening."""
        self.is_listening = False


def demo_speech_recognition():
    """Demonstrate speech recognition."""
    print("\n" + "=" * 60)
    print("Speech Recognition Demo")
    print("=" * 60)

    recognizer = SpeechRecognizer(model_size="base")

    print("\nSpeak a command (5 seconds)...")
    result = recognizer.listen_once(duration=5.0)

    print(f"\nRecognized: \"{result.text}\"")
    print(f"Language: {result.language}")
    print(f"Duration: {result.duration:.1f}s")


if __name__ == "__main__":
    demo_speech_recognition()
```

### Alternative: Google Speech Recognition

For lighter-weight speech recognition:

```python
"""
Lightweight speech recognition using Google Speech API.

Prerequisites:
    pip install SpeechRecognition pyaudio
"""

import speech_recognition as sr
from typing import Optional


class GoogleSpeechRecognizer:
    """Simple speech recognition using Google's API."""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

    def listen(self, timeout: float = 5.0) -> Optional[str]:
        """
        Listen for speech and return transcription.

        Args:
            timeout: Max listening time

        Returns:
            Transcribed text or None if failed
        """
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=timeout)

            print("Processing...")
            text = self.recognizer.recognize_google(audio)
            return text

        except sr.WaitTimeoutError:
            print("No speech detected")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"API error: {e}")
            return None
```

## LLM Integration for Task Planning

### Using LLMs for Robot Reasoning

LLMs can decompose complex commands into executable steps.

```python
#!/usr/bin/env python3
"""
LLM-based Task Planner for Robots

Uses OpenAI GPT or local LLMs for high-level task planning.

Prerequisites:
    pip install openai  # For OpenAI API
    # OR
    pip install ollama  # For local LLMs

Usage:
    export OPENAI_API_KEY="your-key"
    python llm_planner.py
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

# Try to import LLM libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class PlanStep(Enum):
    """Types of plan steps."""
    MOVE = "move"
    PICK = "pick"
    PLACE = "place"
    LOOK = "look"
    SAY = "say"
    WAIT = "wait"
    ASK = "ask"  # Ask for clarification


@dataclass
class TaskPlan:
    """A plan generated by the LLM."""
    goal: str
    steps: List[Dict[str, Any]]
    reasoning: str
    confidence: float
    needs_clarification: bool = False
    clarification_question: Optional[str] = None


class LLMPlanner:
    """
    LLM-based task planner for robots.

    Converts natural language goals into executable step-by-step plans.
    Supports both cloud APIs (OpenAI) and local models (Ollama).
    """

    # System prompt for the robot planner
    SYSTEM_PROMPT = """You are a helpful robot assistant that plans tasks.
Given a user command and the current scene, generate a step-by-step plan.

Available actions:
- MOVE(location): Move to a location (e.g., "kitchen", "table")
- PICK(object): Pick up an object
- PLACE(location): Place held object at location
- LOOK(target): Look at or search for something
- SAY(message): Speak a message to the user
- WAIT(seconds): Wait for specified time
- ASK(question): Ask the user for clarification

Current robot capabilities:
- Can navigate to known locations
- Can pick up objects within reach
- Can place objects on surfaces
- Has camera for object detection

Respond with a JSON object containing:
{
    "goal": "interpreted goal",
    "reasoning": "why this plan makes sense",
    "steps": [
        {"action": "ACTION_TYPE", "params": {...}, "description": "what this does"}
    ],
    "needs_clarification": false,
    "clarification_question": null
}

If the command is ambiguous, set needs_clarification to true and provide a question."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize LLM planner.

        Args:
            provider: "openai" or "ollama"
            model: Model name (e.g., "gpt-4o-mini", "llama3.2")
        """
        self.provider = provider
        self.model = model
        self.conversation_history = []

        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI not installed. Run: pip install openai")
            self.client = openai.OpenAI()
        elif provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise RuntimeError("Ollama not installed. Run: pip install ollama")
        else:
            raise ValueError(f"Unknown provider: {provider}")

        print(f"LLM Planner initialized with {provider}/{model}")

    def plan(
        self,
        command: str,
        scene_context: Optional[Dict] = None,
        conversation_history: Optional[List] = None
    ) -> TaskPlan:
        """
        Generate a plan for the given command.

        Args:
            command: Natural language command
            scene_context: Current scene state (objects, locations)
            conversation_history: Previous conversation for context

        Returns:
            TaskPlan with steps to execute
        """
        # Build context message
        context_parts = [f"User command: \"{command}\""]

        if scene_context:
            context_parts.append(f"\nCurrent scene:")
            if "objects" in scene_context:
                context_parts.append(f"  Visible objects: {scene_context['objects']}")
            if "robot_location" in scene_context:
                context_parts.append(f"  Robot location: {scene_context['robot_location']}")
            if "held_object" in scene_context:
                context_parts.append(f"  Currently holding: {scene_context['held_object']}")

        user_message = "\n".join(context_parts)

        # Call LLM
        if self.provider == "openai":
            response = self._call_openai(user_message)
        else:
            response = self._call_ollama(user_message)

        # Parse response
        return self._parse_response(response, command)

    def _call_openai(self, user_message: str) -> str:
        """Call OpenAI API."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        return response.choices[0].message.content

    def _call_ollama(self, user_message: str) -> str:
        """Call Ollama local model."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        response = ollama.chat(
            model=self.model,
            messages=messages
        )

        return response["message"]["content"]

    def _parse_response(self, response: str, original_command: str) -> TaskPlan:
        """Parse LLM response into TaskPlan."""
        try:
            # Extract JSON from response
            data = json.loads(response)

            return TaskPlan(
                goal=data.get("goal", original_command),
                steps=data.get("steps", []),
                reasoning=data.get("reasoning", ""),
                confidence=0.9 if data.get("steps") else 0.5,
                needs_clarification=data.get("needs_clarification", False),
                clarification_question=data.get("clarification_question")
            )

        except json.JSONDecodeError:
            # Fallback for non-JSON response
            return TaskPlan(
                goal=original_command,
                steps=[],
                reasoning=response,
                confidence=0.3,
                needs_clarification=True,
                clarification_question="I couldn't understand that command. Could you rephrase?"
            )

    def explain_plan(self, plan: TaskPlan) -> str:
        """Generate human-readable explanation of plan."""
        lines = [f"Goal: {plan.goal}", "", "Plan:"]

        for i, step in enumerate(plan.steps, 1):
            action = step.get("action", "UNKNOWN")
            desc = step.get("description", "")
            lines.append(f"  {i}. {action}: {desc}")

        if plan.reasoning:
            lines.extend(["", f"Reasoning: {plan.reasoning}"])

        return "\n".join(lines)


class ConversationalPlanner(LLMPlanner):
    """
    LLM planner with conversation memory for multi-turn interactions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history = []
        self.scene_state = {}

    def chat(self, user_input: str) -> str:
        """
        Process user input in conversational context.

        Handles both commands and questions.
        """
        # Add to history
        self.history.append({"role": "user", "content": user_input})

        # Check if this is a clarification response
        if self._is_clarification_response(user_input):
            return self._handle_clarification(user_input)

        # Generate plan
        plan = self.plan(user_input, self.scene_state, self.history)

        if plan.needs_clarification:
            response = plan.clarification_question
        else:
            # Execute plan (in real system)
            response = self.explain_plan(plan)
            response += "\n\nExecuting plan..."

        self.history.append({"role": "assistant", "content": response})
        return response

    def _is_clarification_response(self, text: str) -> bool:
        """Check if input is answering a previous question."""
        # Simple heuristic - improve with NLU
        short_responses = ["yes", "no", "the red one", "on the table", "left", "right"]
        return text.lower().strip() in short_responses or len(text.split()) < 5

    def _handle_clarification(self, response: str) -> str:
        """Handle clarification response."""
        # Re-plan with additional context
        if len(self.history) >= 2:
            original_command = self.history[-2]["content"]
            augmented = f"{original_command}. User clarified: {response}"
            plan = self.plan(augmented, self.scene_state)
            return self.explain_plan(plan)
        return "I'm not sure what you're referring to. Could you repeat your request?"

    def update_scene(self, objects: List[str], robot_location: str, held_object: Optional[str] = None):
        """Update scene state for context-aware planning."""
        self.scene_state = {
            "objects": objects,
            "robot_location": robot_location,
            "held_object": held_object
        }


def demo_llm_planner():
    """Demonstrate LLM planning."""
    print("\n" + "=" * 60)
    print("LLM Task Planner Demo")
    print("=" * 60)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY") and not OLLAMA_AVAILABLE:
        print("\nNote: Set OPENAI_API_KEY or install Ollama for full demo")
        print("Showing example output instead:\n")

        # Example output
        example_plan = TaskPlan(
            goal="Get the red cup from the kitchen and bring it to me",
            steps=[
                {"action": "MOVE", "params": {"location": "kitchen"}, "description": "Navigate to kitchen"},
                {"action": "LOOK", "params": {"target": "red cup"}, "description": "Search for red cup"},
                {"action": "PICK", "params": {"object": "red cup"}, "description": "Pick up the red cup"},
                {"action": "MOVE", "params": {"location": "user"}, "description": "Return to user"},
                {"action": "SAY", "params": {"message": "Here is your cup"}, "description": "Confirm delivery"},
            ],
            reasoning="The user wants the red cup from the kitchen. I need to navigate there, find and pick up the cup, then return.",
            confidence=0.9
        )

        print("Command: 'Get the red cup from the kitchen and bring it to me'")
        print("\nGenerated Plan:")
        for i, step in enumerate(example_plan.steps, 1):
            print(f"  {i}. {step['action']}: {step['description']}")
        print(f"\nReasoning: {example_plan.reasoning}")
        return

    # Real demo with API
    try:
        planner = LLMPlanner(provider="openai", model="gpt-4o-mini")

        test_commands = [
            "Get me a drink from the fridge",
            "Clean up the living room",
            "Find my keys",
        ]

        for cmd in test_commands:
            print(f"\nCommand: \"{cmd}\"")
            plan = planner.plan(cmd, scene_context={
                "objects": ["cup", "bottle", "book", "remote"],
                "robot_location": "living room",
                "held_object": None
            })
            print(planner.explain_plan(plan))

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    demo_llm_planner()
```

## Handling Ambiguous Commands

Real-world commands are often ambiguous. The system must detect and resolve ambiguity.

```python
"""
Ambiguity Detection and Resolution for Robot Commands

Detects unclear commands and generates clarifying questions.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class AmbiguityType(Enum):
    """Types of command ambiguity."""
    MULTIPLE_OBJECTS = "multiple_objects"  # "Pick up the cup" (which cup?)
    UNCLEAR_LOCATION = "unclear_location"  # "Put it there" (where?)
    MISSING_OBJECT = "missing_object"      # "Pick it up" (what?)
    VAGUE_ACTION = "vague_action"          # "Do something with the cup"
    INCOMPLETE = "incomplete"               # "The red..." (unfinished)


@dataclass
class AmbiguityResult:
    """Result of ambiguity detection."""
    is_ambiguous: bool
    ambiguity_type: Optional[AmbiguityType]
    clarifying_question: Optional[str]
    options: List[str]


class AmbiguityResolver:
    """
    Detects and helps resolve ambiguous robot commands.
    """

    # Pronouns that need resolution
    PRONOUNS = ["it", "that", "this", "them", "those", "there", "here"]

    # Vague action words
    VAGUE_ACTIONS = ["do", "something", "stuff", "thing", "handle", "deal with"]

    def __init__(self, scene_objects: List[str] = None):
        """
        Initialize resolver.

        Args:
            scene_objects: List of objects currently visible
        """
        self.scene_objects = scene_objects or []
        self.last_mentioned_object = None
        self.last_mentioned_location = None

    def check_ambiguity(
        self,
        command: str,
        parsed_object: Optional[str] = None,
        parsed_location: Optional[str] = None
    ) -> AmbiguityResult:
        """
        Check if command is ambiguous.

        Args:
            command: Original command text
            parsed_object: Extracted object (may be pronoun)
            parsed_location: Extracted location

        Returns:
            AmbiguityResult with detection and clarification
        """
        command_lower = command.lower()

        # Check for pronouns without clear reference
        if self._has_unresolved_pronoun(command_lower, parsed_object):
            return self._create_pronoun_ambiguity(parsed_object)

        # Check for multiple matching objects in scene
        if parsed_object:
            matches = self._find_matching_objects(parsed_object)
            if len(matches) > 1:
                return self._create_multiple_objects_ambiguity(parsed_object, matches)

        # Check for vague actions
        if self._has_vague_action(command_lower):
            return AmbiguityResult(
                is_ambiguous=True,
                ambiguity_type=AmbiguityType.VAGUE_ACTION,
                clarifying_question=f"What would you like me to do with the {parsed_object or 'object'}?",
                options=["pick it up", "move it", "examine it", "describe it"]
            )

        # Check for unclear locations
        if parsed_location in self.PRONOUNS:
            return AmbiguityResult(
                is_ambiguous=True,
                ambiguity_type=AmbiguityType.UNCLEAR_LOCATION,
                clarifying_question="Where exactly should I put it?",
                options=["on the table", "on the shelf", "on the floor", "give it to you"]
            )

        # No ambiguity detected
        return AmbiguityResult(
            is_ambiguous=False,
            ambiguity_type=None,
            clarifying_question=None,
            options=[]
        )

    def _has_unresolved_pronoun(self, command: str, parsed_object: Optional[str]) -> bool:
        """Check if command uses pronouns without context."""
        if parsed_object and parsed_object.lower() in self.PRONOUNS:
            # Check if we have context from previous interaction
            return self.last_mentioned_object is None
        return False

    def _create_pronoun_ambiguity(self, pronoun: str) -> AmbiguityResult:
        """Create ambiguity result for unresolved pronoun."""
        if self.scene_objects:
            options = self.scene_objects[:4]  # Limit options
            return AmbiguityResult(
                is_ambiguous=True,
                ambiguity_type=AmbiguityType.MISSING_OBJECT,
                clarifying_question=f"What would you like me to pick up?",
                options=options
            )
        return AmbiguityResult(
            is_ambiguous=True,
            ambiguity_type=AmbiguityType.MISSING_OBJECT,
            clarifying_question="I don't see what you're referring to. What object do you mean?",
            options=[]
        )

    def _find_matching_objects(self, object_name: str) -> List[str]:
        """Find objects in scene matching the description."""
        matches = []
        for obj in self.scene_objects:
            if object_name.lower() in obj.lower() or obj.lower() in object_name.lower():
                matches.append(obj)
        return matches

    def _create_multiple_objects_ambiguity(
        self,
        object_name: str,
        matches: List[str]
    ) -> AmbiguityResult:
        """Create ambiguity result for multiple matching objects."""
        return AmbiguityResult(
            is_ambiguous=True,
            ambiguity_type=AmbiguityType.MULTIPLE_OBJECTS,
            clarifying_question=f"I see multiple {object_name}s. Which one do you mean?",
            options=matches
        )

    def _has_vague_action(self, command: str) -> bool:
        """Check if command has vague action words."""
        return any(word in command for word in self.VAGUE_ACTIONS)

    def resolve_with_context(
        self,
        ambiguity: AmbiguityResult,
        user_response: str
    ) -> Tuple[str, str]:
        """
        Resolve ambiguity using user's clarifying response.

        Args:
            ambiguity: The detected ambiguity
            user_response: User's response to clarifying question

        Returns:
            Tuple of (resolved_object, resolved_location)
        """
        response_lower = user_response.lower().strip()

        # Check if response matches any option
        for option in ambiguity.options:
            if option.lower() in response_lower or response_lower in option.lower():
                if ambiguity.ambiguity_type == AmbiguityType.MULTIPLE_OBJECTS:
                    self.last_mentioned_object = option
                    return (option, None)
                elif ambiguity.ambiguity_type == AmbiguityType.UNCLEAR_LOCATION:
                    self.last_mentioned_location = option
                    return (None, option)

        # Use response directly if no match
        return (response_lower, None)


def demo_ambiguity():
    """Demonstrate ambiguity detection."""
    print("\n" + "=" * 60)
    print("Ambiguity Detection Demo")
    print("=" * 60)

    resolver = AmbiguityResolver(
        scene_objects=["red cup", "blue cup", "water bottle", "book"]
    )

    test_cases = [
        ("Pick up the cup", "cup", None),
        ("Put it there", "it", "there"),
        ("Get that thing", "thing", None),
        ("Pick up the water bottle", "water bottle", None),  # Not ambiguous
    ]

    for command, obj, loc in test_cases:
        result = resolver.check_ambiguity(command, obj, loc)
        print(f"\nCommand: \"{command}\"")
        print(f"  Ambiguous: {result.is_ambiguous}")
        if result.is_ambiguous:
            print(f"  Type: {result.ambiguity_type.value}")
            print(f"  Question: {result.clarifying_question}")
            print(f"  Options: {result.options}")


if __name__ == "__main__":
    demo_ambiguity()
```

## Complete Voice-Controlled Robot System

Now let's integrate everything into a complete ROS 2 system:

```python
#!/usr/bin/env python3
"""
Voice-Controlled Robot System - Complete ROS 2 Integration

Integrates:
- Speech recognition (Whisper/Google)
- LLM planning (GPT/Ollama)
- Vision (from Chapter 8)
- Action execution

ROS 2 Topics:
    Subscribed:
        /camera/rgb/image_raw - Camera images
    Published:
        /cmd_vel - Velocity commands
        /arm/goal - Arm movement goals
        /robot/speech - Text-to-speech output

Prerequisites:
    pip install openai-whisper sounddevice numpy
    pip install openai  # or ollama

Usage:
    ros2 run vla_package voice_robot_control

Author: Physical AI & Humanoid Robotics Book
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image
import threading
import queue
from typing import Optional
import json

# Import our modules (from previous sections)
# from .speech_recognition import SpeechRecognizer
# from .llm_planner import LLMPlanner, TaskPlan
# from .ambiguity_resolver import AmbiguityResolver


class VoiceControlledRobot(Node):
    """
    Complete voice-controlled robot ROS 2 node.

    Listens for voice commands, plans using LLM, and executes actions.
    """

    def __init__(self):
        super().__init__("voice_robot_control")

        # Parameters
        self.declare_parameter("wake_word", "robot")
        self.declare_parameter("llm_provider", "openai")
        self.declare_parameter("llm_model", "gpt-4o-mini")
        self.declare_parameter("whisper_model", "base")

        wake_word = self.get_parameter("wake_word").value
        llm_provider = self.get_parameter("llm_provider").value
        llm_model = self.get_parameter("llm_model").value

        # Initialize components
        self.get_logger().info("Initializing voice control system...")

        # Speech recognizer (simplified for demo)
        self.speech_queue = queue.Queue()
        self.is_listening = True

        # LLM Planner (simplified initialization)
        self.llm_available = False
        try:
            # self.planner = LLMPlanner(provider=llm_provider, model=llm_model)
            self.llm_available = True
            self.get_logger().info(f"LLM planner ready: {llm_provider}/{llm_model}")
        except Exception as e:
            self.get_logger().warn(f"LLM not available: {e}. Using rule-based fallback.")

        # Scene state
        self.visible_objects = []
        self.robot_location = "home"
        self.held_object = None

        # Conversation state
        self.conversation_history = []
        self.pending_clarification = None

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.arm_goal_pub = self.create_publisher(PoseStamped, "/arm/goal", 10)
        self.speech_pub = self.create_publisher(String, "/robot/speech", 10)
        self.status_pub = self.create_publisher(String, "/robot/status", 10)

        # Subscribers
        self.speech_sub = self.create_subscription(
            String, "/speech/text", self.speech_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, "/camera/rgb/image_raw", self.image_callback, 10
        )

        # Processing timer
        self.timer = self.create_timer(0.1, self.process_commands)

        self.get_logger().info("Voice-controlled robot ready!")
        self.speak("Hello! I'm ready for your commands.")

    def speech_callback(self, msg: String):
        """Handle incoming speech text."""
        text = msg.data.strip()
        if text:
            self.get_logger().info(f"Received: '{text}'")
            self.speech_queue.put(text)

    def image_callback(self, msg: Image):
        """Process camera images for scene understanding."""
        # In real implementation, run object detection here
        # self.visible_objects = self.detector.detect(image)
        pass

    def process_commands(self):
        """Process queued voice commands."""
        try:
            command = self.speech_queue.get_nowait()
            self.handle_command(command)
        except queue.Empty:
            pass

    def handle_command(self, command: str):
        """Process a voice command."""
        self.publish_status(f"Processing: {command}")

        # Check for system commands
        if self._is_system_command(command):
            self._handle_system_command(command)
            return

        # Check if this is a clarification response
        if self.pending_clarification:
            self._handle_clarification(command)
            return

        # Plan with LLM or fallback
        if self.llm_available:
            plan = self._plan_with_llm(command)
        else:
            plan = self._plan_rule_based(command)

        # Check for ambiguity
        if plan.get("needs_clarification"):
            self.pending_clarification = plan
            self.speak(plan["clarification_question"])
            return

        # Execute plan
        self._execute_plan(plan)

    def _is_system_command(self, command: str) -> bool:
        """Check if command is a system command."""
        system_words = ["stop", "cancel", "pause", "help", "status", "quit"]
        return any(word in command.lower() for word in system_words)

    def _handle_system_command(self, command: str):
        """Handle system commands."""
        cmd_lower = command.lower()

        if "stop" in cmd_lower or "cancel" in cmd_lower:
            self._stop_robot()
            self.speak("Stopping.")

        elif "help" in cmd_lower:
            self.speak("I can pick up objects, navigate to locations, and follow your instructions. Try saying: pick up the cup, or go to the kitchen.")

        elif "status" in cmd_lower:
            status = f"I'm at {self.robot_location}."
            if self.held_object:
                status += f" Holding {self.held_object}."
            if self.visible_objects:
                status += f" I see: {', '.join(self.visible_objects[:3])}."
            self.speak(status)

    def _handle_clarification(self, response: str):
        """Handle clarification response."""
        # Combine with original command
        original = self.pending_clarification.get("original_command", "")
        clarified = f"{original}. Specifically: {response}"

        self.pending_clarification = None

        # Re-plan with clarification
        if self.llm_available:
            plan = self._plan_with_llm(clarified)
        else:
            plan = self._plan_rule_based(clarified)

        self._execute_plan(plan)

    def _plan_with_llm(self, command: str) -> dict:
        """Generate plan using LLM."""
        # In real implementation:
        # plan = self.planner.plan(command, {
        #     "objects": self.visible_objects,
        #     "robot_location": self.robot_location,
        #     "held_object": self.held_object
        # })
        # return plan.__dict__

        # Simplified for demo
        return self._plan_rule_based(command)

    def _plan_rule_based(self, command: str) -> dict:
        """Rule-based planning fallback."""
        cmd_lower = command.lower()
        steps = []

        # Parse action
        if any(w in cmd_lower for w in ["pick", "grab", "get", "take"]):
            # Extract object (simplified)
            obj = self._extract_object(cmd_lower)
            if obj:
                steps = [
                    {"action": "LOOK", "params": {"target": obj}},
                    {"action": "PICK", "params": {"object": obj}},
                    {"action": "SAY", "params": {"message": f"I have the {obj}"}},
                ]
            else:
                return {
                    "needs_clarification": True,
                    "clarification_question": "What should I pick up?",
                    "original_command": command
                }

        elif any(w in cmd_lower for w in ["go", "move", "navigate"]):
            loc = self._extract_location(cmd_lower)
            if loc:
                steps = [
                    {"action": "MOVE", "params": {"location": loc}},
                    {"action": "SAY", "params": {"message": f"I'm at the {loc}"}},
                ]
            else:
                return {
                    "needs_clarification": True,
                    "clarification_question": "Where should I go?",
                    "original_command": command
                }

        elif any(w in cmd_lower for w in ["put", "place", "set"]):
            loc = self._extract_location(cmd_lower)
            if loc and self.held_object:
                steps = [
                    {"action": "PLACE", "params": {"location": loc}},
                    {"action": "SAY", "params": {"message": f"Placed {self.held_object} on {loc}"}},
                ]
            elif not self.held_object:
                return {
                    "needs_clarification": True,
                    "clarification_question": "I'm not holding anything. What should I pick up first?",
                    "original_command": command
                }

        elif any(w in cmd_lower for w in ["find", "look", "search", "where"]):
            obj = self._extract_object(cmd_lower)
            steps = [
                {"action": "LOOK", "params": {"target": obj or "around"}},
            ]

        else:
            return {
                "needs_clarification": True,
                "clarification_question": "I didn't understand that. Could you try rephrasing?",
                "original_command": command
            }

        return {
            "goal": command,
            "steps": steps,
            "needs_clarification": False
        }

    def _extract_object(self, text: str) -> Optional[str]:
        """Extract object from command text."""
        objects = ["cup", "bottle", "book", "phone", "ball", "box", "apple", "remote"]
        for obj in objects:
            if obj in text:
                return obj
        return None

    def _extract_location(self, text: str) -> Optional[str]:
        """Extract location from command text."""
        locations = ["table", "kitchen", "living room", "shelf", "floor", "counter", "desk"]
        for loc in locations:
            if loc in text:
                return loc
        # Check for prepositions
        import re
        match = re.search(r"(?:to|on|at|in)\s+(?:the\s+)?(\w+)", text)
        if match:
            return match.group(1)
        return None

    def _execute_plan(self, plan: dict):
        """Execute a generated plan."""
        steps = plan.get("steps", [])

        self.speak(f"Executing: {plan.get('goal', 'task')}")

        for step in steps:
            action = step.get("action")
            params = step.get("params", {})

            self.get_logger().info(f"Executing: {action} with {params}")

            if action == "MOVE":
                self._execute_move(params.get("location"))
            elif action == "PICK":
                self._execute_pick(params.get("object"))
            elif action == "PLACE":
                self._execute_place(params.get("location"))
            elif action == "LOOK":
                self._execute_look(params.get("target"))
            elif action == "SAY":
                self.speak(params.get("message", ""))
            elif action == "WAIT":
                import time
                time.sleep(params.get("seconds", 1))

    def _execute_move(self, location: str):
        """Execute navigation to location."""
        self.publish_status(f"Moving to {location}")

        # Send velocity command (simplified)
        twist = Twist()
        twist.linear.x = 0.3
        self.cmd_vel_pub.publish(twist)

        # In real implementation: use Nav2 action
        self.robot_location = location

    def _execute_pick(self, obj: str):
        """Execute pick action."""
        self.publish_status(f"Picking up {obj}")

        # In real implementation: use MoveIt2
        self.held_object = obj

    def _execute_place(self, location: str):
        """Execute place action."""
        self.publish_status(f"Placing on {location}")

        # In real implementation: use MoveIt2
        placed = self.held_object
        self.held_object = None

    def _execute_look(self, target: str):
        """Execute look/search action."""
        self.publish_status(f"Looking for {target}")

        # In real implementation: pan camera, run detection
        if target in self.visible_objects:
            self.speak(f"I see the {target}")
        else:
            self.speak(f"I don't see a {target} right now")

    def _stop_robot(self):
        """Emergency stop."""
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

    def speak(self, text: str):
        """Publish text for text-to-speech."""
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)
        self.get_logger().info(f"Speaking: {text}")

    def publish_status(self, status: str):
        """Publish status update."""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VoiceControlledRobot()

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

## Summary

In this chapter, you learned:

✅ **Speech Recognition**: Convert voice to text using Whisper or Google Speech API

✅ **LLM Integration**: Use GPT or local LLMs for intelligent task planning

✅ **Ambiguity Handling**: Detect unclear commands and ask clarifying questions

✅ **Conversational Context**: Maintain conversation history for multi-turn interactions

✅ **ROS 2 Integration**: Build a complete voice-controlled robot system

✅ **End-to-End Pipeline**: Voice → Text → Plan → Execute → Feedback

### Key Takeaways

- **LLMs enable complex reasoning** beyond simple keyword matching
- **Speech recognition** is now accurate enough for robot control
- **Ambiguity resolution** is critical for natural interaction
- **Context memory** enables multi-turn conversations
- **Fallback strategies** ensure robustness when AI fails

## Exercises

### Exercise 1: Add Wake Word Detection

Modify the speech recognizer to only process commands after hearing "Hey Robot":

```python
# Hint: Listen continuously, check for wake word, then process next utterance
```

### Exercise 2: Implement Text-to-Speech

Add speech synthesis for robot responses:

```python
# Use pyttsx3 or gTTS for text-to-speech
pip install pyttsx3
```

### Exercise 3: Multi-Object Commands

Extend the planner to handle commands like:
- "Pick up all the cups"
- "Move the red and blue bottles to the shelf"

### Exercise 4: Error Recovery

Add error handling when actions fail:
- Retry logic with different approach
- Ask user for help
- Report what went wrong

### Challenge: Teach New Commands

Build a system where users can teach the robot new tasks:
- "When I say 'clean up', pick up all objects and put them on the shelf"
- Store learned behaviors for later use

## Up Next

In **Chapter 10: Building the Humanoid Robot Capstone**, you'll combine everything:
- ROS 2 control from Chapters 3-4
- Isaac Sim simulation from Chapters 6-7
- VLA systems from Chapters 8-9
- Build a complete voice-controlled humanoid robot!

## Additional Resources

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech Recognition
- [Ollama](https://ollama.ai/) - Local LLM Hosting
- [OpenAI API](https://platform.openai.com/docs) - GPT Integration
- [ROS 2 Audio Common](https://github.com/ros-drivers/audio_common) - Audio in ROS 2
- [MoveIt 2](https://moveit.ros.org/) - Motion Planning
- [Nav2](https://navigation.ros.org/) - Navigation Stack

---

**Sources:**
- [OpenAI Whisper Paper](https://arxiv.org/abs/2212.04356)
- [SayCan: Google Research](https://say-can.github.io/) - LLM + Robot Affordances
- [Language Models as Zero-Shot Planners](https://arxiv.org/abs/2201.07207)
- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
