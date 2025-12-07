#!/usr/bin/env python3
"""
Vision Detector Module for VLA Systems

Real-time object detection for robotic perception using YOLOv8.
This module provides the "eyes" of the VLA system.

Prerequisites:
    pip install ultralytics opencv-python numpy

Usage:
    # As a module
    from vision_detector import VisionDetector
    detector = VisionDetector()
    detections = detector.detect(image)

    # Standalone demo
    python vision_detector.py

Features:
    - Real-time object detection with YOLOv8
    - Support for 80 COCO object classes
    - Bounding box and center point extraction
    - Confidence filtering
    - Object search by name

Author: Physical AI & Humanoid Robotics Book
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

# Check for ultralytics availability
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed.")
    print("Install with: pip install ultralytics")


@dataclass
class Detection:
    """
    Represents a detected object in an image.

    Attributes:
        class_name: Name of the detected object class
        confidence: Detection confidence score (0-1)
        bbox: Bounding box as (x1, y1, x2, y2)
        center: Center point as (x, y)
        area: Bounding box area in pixels
    """
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    area: int

    def __repr__(self):
        return f"Detection({self.class_name}, conf={self.confidence:.2f}, center={self.center})"


class VisionDetector:
    """
    Vision module for detecting objects in camera images.

    Uses YOLOv8 for real-time object detection with support for
    80 COCO classes including common household objects.

    Example:
        detector = VisionDetector()
        detections = detector.detect(camera_image)
        for det in detections:
            print(f"Found {det.class_name} at {det.center}")
    """

    # Objects commonly manipulated by robots
    ROBOT_TARGET_CLASSES = [
        "bottle", "cup", "bowl", "apple", "banana", "orange",
        "book", "keyboard", "mouse", "remote", "cell phone",
        "chair", "couch", "potted plant", "dining table",
        "laptop", "scissors", "toothbrush", "teddy bear",
        "fork", "knife", "spoon", "wine glass", "vase"
    ]

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        target_classes: Optional[List[str]] = None
    ):
        """
        Initialize the vision detector.

        Args:
            model_path: Path to YOLO model weights.
                       Options: yolov8n.pt (nano), yolov8s.pt (small),
                               yolov8m.pt (medium), yolov8l.pt (large)
            confidence_threshold: Minimum confidence for detections (0-1)
            target_classes: List of class names to detect (None = all)
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError(
                "ultralytics package required.\n"
                "Install with: pip install ultralytics"
            )

        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes

        # Cache class names
        self.class_names = self.model.names

        print(f"Model loaded with {len(self.class_names)} classes")

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect objects in an image.

        Args:
            image: BGR image from camera (OpenCV format), shape (H, W, 3)

        Returns:
            List of Detection objects sorted by confidence (highest first)
        """
        # Run inference
        results = self.model(image, verbose=False)[0]
        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]
            confidence = float(box.conf[0])

            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                continue

            # Filter by target classes if specified
            if self.target_classes and class_name not in self.target_classes:
                continue

            # Get bounding box coordinates
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

        # Sort by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def find_object(
        self,
        image: np.ndarray,
        object_name: str
    ) -> Optional[Detection]:
        """
        Find a specific object in the image.

        Args:
            image: Camera image (BGR format)
            object_name: Name of object to find (e.g., "cup", "bottle")

        Returns:
            Detection if found, None otherwise
        """
        detections = self.detect(image)

        # Find matches (case-insensitive, partial match)
        matches = [
            d for d in detections
            if object_name.lower() in d.class_name.lower()
        ]

        if not matches:
            return None

        # Return highest confidence match
        return matches[0]

    def find_all_objects(
        self,
        image: np.ndarray,
        object_name: str
    ) -> List[Detection]:
        """
        Find all instances of a specific object.

        Args:
            image: Camera image
            object_name: Name of object to find

        Returns:
            List of matching detections
        """
        detections = self.detect(image)
        return [
            d for d in detections
            if object_name.lower() in d.class_name.lower()
        ]

    def find_nearest_object(
        self,
        image: np.ndarray,
        object_name: str,
        reference_point: Tuple[int, int]
    ) -> Optional[Detection]:
        """
        Find the object nearest to a reference point.

        Args:
            image: Camera image
            object_name: Name of object to find
            reference_point: (x, y) reference point in image

        Returns:
            Nearest detection, or None if not found
        """
        matches = self.find_all_objects(image, object_name)
        if not matches:
            return None

        def distance(det: Detection) -> float:
            dx = det.center[0] - reference_point[0]
            dy = det.center[1] - reference_point[1]
            return (dx * dx + dy * dy) ** 0.5

        return min(matches, key=distance)

    def annotate_image(
        self,
        image: np.ndarray,
        detections: List[Detection],
        show_labels: bool = True,
        box_color: Tuple[int, int, int] = (0, 255, 0),
        text_color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on image.

        Args:
            image: Input image
            detections: List of detections to draw
            show_labels: Whether to show class names and confidence
            box_color: BGR color for bounding boxes
            text_color: BGR color for text

        Returns:
            Annotated image copy
        """
        annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Draw center point
            cv2.circle(annotated, det.center, 5, (0, 0, 255), -1)

            if show_labels:
                # Draw label background
                label = f"{det.class_name}: {det.confidence:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w, y1),
                    box_color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )

        return annotated

    def get_detection_summary(self, detections: List[Detection]) -> str:
        """Generate a human-readable summary of detections."""
        if not detections:
            return "No objects detected"

        # Count objects by class
        counts = {}
        for det in detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1

        parts = [f"{count} {name}{'s' if count > 1 else ''}"
                 for name, count in counts.items()]
        return "Detected: " + ", ".join(parts)


def demo_webcam():
    """Run live demo with webcam."""
    print("\n" + "=" * 60)
    print("VLA Vision Detector Demo")
    print("=" * 60)
    print("Press 'q' to quit")
    print("Press 's' to search for specific object")
    print("=" * 60 + "\n")

    detector = VisionDetector(confidence_threshold=0.4)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Performance tracking
    frame_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Detect objects
        detections = detector.detect(frame)

        # Annotate frame
        annotated = detector.annotate_image(frame, detections)

        # Calculate FPS
        frame_time = time.time() - start_time
        frame_times.append(frame_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times))

        # Draw info
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f} | Objects: {len(detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Print detection summary
        summary = detector.get_detection_summary(detections)
        cv2.putText(
            annotated,
            summary[:80],  # Truncate long summaries
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.imshow("VLA Vision Detector", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Search mode
            object_name = input("Enter object to search for: ")
            result = detector.find_object(frame, object_name)
            if result:
                print(f"Found {result.class_name} at {result.center}")
            else:
                print(f"Object '{object_name}' not found")

    cap.release()
    cv2.destroyAllWindows()


def demo_single_image(image_path: str):
    """Run demo on a single image."""
    print(f"\nProcessing: {image_path}")

    detector = VisionDetector()
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    detections = detector.detect(image)

    print(f"\n{detector.get_detection_summary(detections)}")
    for det in detections:
        print(f"  - {det}")

    annotated = detector.annotate_image(image, detections)
    cv2.imshow("Detections", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Process provided image
        demo_single_image(sys.argv[1])
    else:
        # Live webcam demo
        demo_webcam()
