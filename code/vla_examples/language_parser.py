#!/usr/bin/env python3
"""
Language Parser Module for VLA Systems

Natural language understanding for robot command parsing.
This module provides the "ears and brain" of the VLA system.

Prerequisites:
    pip install spacy
    python -m spacy download en_core_web_sm

Usage:
    from language_parser import LanguageParser
    parser = LanguageParser()
    command = parser.parse("Pick up the red cup")
    print(f"Action: {command.action}, Object: {command.target_object}")

Features:
    - Action verb extraction (pick, place, move, etc.)
    - Target object identification
    - Location extraction
    - Modifier parsing (colors, sizes)
    - Confidence scoring

Author: Physical AI & Humanoid Robotics Book
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

# Check for spaCy availability
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not installed. Using regex-only parsing.")
    print("For better results: pip install spacy && python -m spacy download en_core_web_sm")


class RobotAction(Enum):
    """Supported robot action types."""
    PICK = "pick"
    PLACE = "place"
    MOVE = "move"
    PUSH = "push"
    PULL = "pull"
    POUR = "pour"
    OPEN = "open"
    CLOSE = "close"
    POINT = "point"
    LOOK = "look"
    STOP = "stop"
    WAIT = "wait"
    TURN = "turn"
    WAVE = "wave"
    UNKNOWN = "unknown"

    @classmethod
    def describe(cls, action: 'RobotAction') -> str:
        """Get human-readable description of action."""
        descriptions = {
            cls.PICK: "Pick up or grasp an object",
            cls.PLACE: "Place or put down an object",
            cls.MOVE: "Move to a location",
            cls.PUSH: "Push an object",
            cls.PULL: "Pull an object",
            cls.POUR: "Pour contents from container",
            cls.OPEN: "Open a door, drawer, or container",
            cls.CLOSE: "Close a door, drawer, or container",
            cls.POINT: "Point at an object or location",
            cls.LOOK: "Look at or search for something",
            cls.STOP: "Stop current action",
            cls.WAIT: "Wait or pause",
            cls.TURN: "Turn or rotate",
            cls.WAVE: "Wave hand gesture",
            cls.UNKNOWN: "Unrecognized action",
        }
        return descriptions.get(action, "Unknown action")


@dataclass
class ParsedCommand:
    """
    Structured representation of a parsed robot command.

    Attributes:
        action: The primary action to perform
        target_object: Object to manipulate (if any)
        target_location: Destination location (if any)
        modifiers: Descriptive modifiers (color, size, etc.)
        quantity: Number of objects (default 1)
        confidence: Parse confidence score (0-1)
        original_text: Original command text
        is_compound: Whether this is part of a compound command
    """
    action: RobotAction
    target_object: Optional[str] = None
    target_location: Optional[str] = None
    modifiers: Dict[str, Any] = field(default_factory=dict)
    quantity: int = 1
    confidence: float = 0.0
    original_text: str = ""
    is_compound: bool = False

    def __repr__(self):
        parts = [f"action={self.action.value}"]
        if self.target_object:
            parts.append(f"object='{self.target_object}'")
        if self.target_location:
            parts.append(f"location='{self.target_location}'")
        if self.modifiers:
            parts.append(f"modifiers={self.modifiers}")
        parts.append(f"confidence={self.confidence:.2f}")
        return f"ParsedCommand({', '.join(parts)})"

    def describe(self) -> str:
        """Generate human-readable description of the command."""
        desc = f"{self.action.value.capitalize()}"
        if self.target_object:
            obj = self.target_object
            if self.modifiers.get("color"):
                obj = f"{self.modifiers['color']} {obj}"
            if self.modifiers.get("size"):
                obj = f"{self.modifiers['size']} {obj}"
            desc += f" the {obj}"
        if self.target_location:
            desc += f" to/on the {self.target_location}"
        return desc


class LanguageParser:
    """
    Natural language parser for robot commands.

    Parses commands like:
    - "Pick up the red cup"
    - "Move to the kitchen table"
    - "Place the bottle on the shelf"
    - "Pour water into the glass"
    - "Find the blue ball"

    Example:
        parser = LanguageParser()
        result = parser.parse("Pick up the large red cup")
        print(result.action)  # RobotAction.PICK
        print(result.target_object)  # "cup"
        print(result.modifiers)  # {"color": "red", "size": "large"}
    """

    # Action verb patterns - map verbs to actions
    ACTION_PATTERNS = {
        RobotAction.PICK: r"\b(pick|grab|grasp|take|get|lift|fetch|retrieve)\b",
        RobotAction.PLACE: r"\b(place|put|set|drop|release|leave|deposit)\b",
        RobotAction.MOVE: r"\b(move|go|navigate|drive|travel|walk|come|approach)\b",
        RobotAction.PUSH: r"\b(push|shove|slide)\b",
        RobotAction.PULL: r"\b(pull|drag)\b",
        RobotAction.POUR: r"\b(pour|fill|empty|transfer)\b",
        RobotAction.OPEN: r"\b(open|unlock)\b",
        RobotAction.CLOSE: r"\b(close|shut|lock)\b",
        RobotAction.POINT: r"\b(point|indicate|show|gesture)\b",
        RobotAction.LOOK: r"\b(look|see|find|locate|search|spot|identify)\b",
        RobotAction.STOP: r"\b(stop|halt|freeze|pause|cancel)\b",
        RobotAction.WAIT: r"\b(wait|hold|stay)\b",
        RobotAction.TURN: r"\b(turn|rotate|spin|face)\b",
        RobotAction.WAVE: r"\b(wave|greet|hello)\b",
    }

    # Location prepositions
    LOCATION_PREPS = [
        "on", "onto", "to", "into", "in", "at", "near", "by",
        "beside", "next to", "toward", "towards", "over", "under"
    ]

    # Common modifiers
    COLORS = [
        "red", "blue", "green", "yellow", "black", "white",
        "orange", "purple", "pink", "brown", "gray", "grey"
    ]

    SIZES = [
        "big", "small", "large", "tiny", "medium", "tall",
        "short", "huge", "little", "massive"
    ]

    POSITIONS = [
        "left", "right", "front", "back", "top", "bottom",
        "upper", "lower", "middle", "center"
    ]

    # Common objects robots interact with
    KNOWN_OBJECTS = [
        "cup", "bottle", "glass", "mug", "bowl", "plate", "box",
        "book", "phone", "remote", "pen", "pencil", "laptop",
        "apple", "banana", "orange", "ball", "toy",
        "chair", "table", "door", "drawer", "shelf", "counter"
    ]

    def __init__(self, use_spacy: bool = True):
        """
        Initialize the language parser.

        Args:
            use_spacy: Whether to use spaCy for enhanced NLP
        """
        self.nlp = None

        if use_spacy and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("Language parser initialized with spaCy NLP")
            except OSError:
                print("Warning: spaCy model not found. Using regex parsing.")
                print("Install model: python -m spacy download en_core_web_sm")
        else:
            print("Language parser initialized with regex-only parsing")

    def parse(self, command: str) -> ParsedCommand:
        """
        Parse a natural language command.

        Args:
            command: Natural language command string

        Returns:
            ParsedCommand with extracted components
        """
        if not command or not command.strip():
            return ParsedCommand(
                action=RobotAction.UNKNOWN,
                confidence=0.0,
                original_text=command
            )

        command_lower = command.lower().strip()

        # Extract action
        action = self._extract_action(command_lower)

        # Extract modifiers (colors, sizes, positions)
        modifiers = self._extract_modifiers(command_lower)

        # Extract target object
        target_object = self._extract_object(command_lower)

        # Extract target location
        target_location = self._extract_location(command_lower)

        # Extract quantity
        quantity = self._extract_quantity(command_lower)

        # Calculate confidence
        confidence = self._calculate_confidence(
            action, target_object, target_location, modifiers
        )

        return ParsedCommand(
            action=action,
            target_object=target_object,
            target_location=target_location,
            modifiers=modifiers,
            quantity=quantity,
            confidence=confidence,
            original_text=command
        )

    def parse_compound(self, command: str) -> List[ParsedCommand]:
        """
        Parse a compound command with multiple actions.

        Args:
            command: Command that may contain multiple actions
                    (e.g., "Pick up the cup and place it on the table")

        Returns:
            List of ParsedCommands
        """
        # Split on common conjunctions
        parts = re.split(r'\s+(?:and|then|,)\s+', command, flags=re.IGNORECASE)

        commands = []
        for part in parts:
            parsed = self.parse(part.strip())
            parsed.is_compound = len(parts) > 1
            commands.append(parsed)

        # Resolve pronouns ("it", "them") to previous objects
        for i in range(1, len(commands)):
            if commands[i].target_object in ["it", "them", "that", "this"]:
                # Reference previous object
                for j in range(i - 1, -1, -1):
                    if commands[j].target_object:
                        commands[i].target_object = commands[j].target_object
                        commands[i].modifiers.update(commands[j].modifiers)
                        break

        return commands

    def _extract_action(self, text: str) -> RobotAction:
        """Extract the primary action from the command."""
        for action, pattern in self.ACTION_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return action
        return RobotAction.UNKNOWN

    def _extract_modifiers(self, text: str) -> Dict[str, Any]:
        """Extract color, size, and position modifiers."""
        modifiers = {}

        for color in self.COLORS:
            if re.search(rf"\b{color}\b", text):
                modifiers["color"] = color
                break

        for size in self.SIZES:
            if re.search(rf"\b{size}\b", text):
                modifiers["size"] = size
                break

        for position in self.POSITIONS:
            if re.search(rf"\b{position}\b", text):
                modifiers["position"] = position
                break

        return modifiers

    def _extract_object(self, text: str) -> Optional[str]:
        """Extract the target object from the command."""
        if self.nlp:
            return self._extract_object_spacy(text)
        return self._extract_object_regex(text)

    def _extract_object_regex(self, text: str) -> Optional[str]:
        """Regex-based object extraction."""
        # First check for known objects
        for obj in self.KNOWN_OBJECTS:
            if re.search(rf"\b{obj}s?\b", text):
                return obj

        # Pattern: "the [adjective] [noun]"
        pattern = r"\bthe\s+(?:\w+\s+)*?(\w+)(?:\s|$)"
        match = re.search(pattern, text)
        if match:
            word = match.group(1)
            # Filter out prepositions and common words
            if word not in ["to", "on", "in", "at", "and", "or"]:
                return word

        return None

    def _extract_object_spacy(self, text: str) -> Optional[str]:
        """spaCy-based object extraction using dependency parsing."""
        doc = self.nlp(text)

        # Find direct objects
        for token in doc:
            if token.dep_ in ("dobj", "pobj") and token.pos_ == "NOUN":
                return token.text

        # Fall back to first noun
        nouns = [t.text for t in doc if t.pos_ == "NOUN"]
        if nouns:
            return nouns[0]

        return None

    def _extract_location(self, text: str) -> Optional[str]:
        """Extract target location from the command."""
        for prep in self.LOCATION_PREPS:
            # Pattern: "preposition [the] location"
            pattern = rf"\b{prep}\s+(?:the\s+)?(\w+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location = match.group(1)
                # Filter out pronouns
                if location not in ["it", "them", "this", "that"]:
                    return location
        return None

    def _extract_quantity(self, text: str) -> int:
        """Extract quantity from the command."""
        # Number words
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "a": 1, "an": 1, "the": 1, "all": -1, "every": -1
        }

        for word, num in number_words.items():
            if re.search(rf"\b{word}\b", text):
                return num

        # Numeric digits
        match = re.search(r"\b(\d+)\b", text)
        if match:
            return int(match.group(1))

        return 1

    def _calculate_confidence(
        self,
        action: RobotAction,
        target_object: Optional[str],
        target_location: Optional[str],
        modifiers: Dict
    ) -> float:
        """Calculate confidence score based on extraction completeness."""
        score = 0.0

        # Action is most important
        if action != RobotAction.UNKNOWN:
            score += 0.4

        # Object adds confidence
        if target_object:
            score += 0.3
            # Known objects are more reliable
            if target_object.lower() in [o.lower() for o in self.KNOWN_OBJECTS]:
                score += 0.1

        # Location adds confidence
        if target_location:
            score += 0.15

        # Modifiers add small confidence boost
        if modifiers:
            score += 0.05

        return min(score, 1.0)

    def get_action_suggestions(self, partial_command: str) -> List[str]:
        """
        Get action suggestions for autocomplete.

        Args:
            partial_command: Partial command text

        Returns:
            List of suggested completions
        """
        suggestions = []
        partial_lower = partial_command.lower()

        action_verbs = {
            RobotAction.PICK: ["pick up", "grab", "take"],
            RobotAction.PLACE: ["place", "put down", "set"],
            RobotAction.MOVE: ["move to", "go to", "navigate to"],
            RobotAction.LOOK: ["look for", "find", "search for"],
            RobotAction.OPEN: ["open"],
            RobotAction.CLOSE: ["close"],
        }

        for action, verbs in action_verbs.items():
            for verb in verbs:
                if verb.startswith(partial_lower):
                    suggestions.append(verb)

        return suggestions


def demo_parser():
    """Demonstrate the language parser."""
    parser = LanguageParser()

    test_commands = [
        "Pick up the red cup",
        "Move to the kitchen table",
        "Place the bottle on the shelf",
        "Find the blue ball",
        "Pour water into the glass",
        "Open the drawer",
        "Look at the apple on the left",
        "Grab the large box and put it on the floor",
        "Stop",
        "Turn left",
        "Get three apples from the bowl",
    ]

    print("\n" + "=" * 70)
    print("VLA Language Parser Demo")
    print("=" * 70)

    for cmd in test_commands:
        result = parser.parse(cmd)
        print(f"\nInput: \"{cmd}\"")
        print(f"  → {result}")
        print(f"  Description: {result.describe()}")

    # Compound command demo
    print("\n" + "-" * 70)
    print("Compound Command Demo")
    print("-" * 70)

    compound = "Pick up the cup and place it on the table"
    commands = parser.parse_compound(compound)
    print(f"\nInput: \"{compound}\"")
    for i, cmd in enumerate(commands):
        print(f"  Step {i+1}: {cmd.describe()}")


def interactive_demo():
    """Interactive demo for testing commands."""
    parser = LanguageParser()

    print("\n" + "=" * 70)
    print("VLA Language Parser - Interactive Mode")
    print("=" * 70)
    print("Enter robot commands (or 'quit' to exit)")
    print("Examples: 'Pick up the red cup', 'Move to the table'")
    print("-" * 70)

    while True:
        try:
            cmd = input("\nCommand: ").strip()
            if cmd.lower() in ["quit", "exit", "q"]:
                break

            if not cmd:
                continue

            # Check for compound command
            if " and " in cmd.lower() or " then " in cmd.lower():
                commands = parser.parse_compound(cmd)
                print("Parsed as compound command:")
                for i, c in enumerate(commands):
                    print(f"  {i+1}. {c.describe()} (conf: {c.confidence:.2f})")
            else:
                result = parser.parse(cmd)
                print(f"\n{result}")
                print(f"Description: {result.describe()}")

        except KeyboardInterrupt:
            break

    print("\nGoodbye!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_demo()
    else:
        demo_parser()
