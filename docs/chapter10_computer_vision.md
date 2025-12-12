---
sidebar_position: 10
title: "Chapter 10: Computer Vision for Robotics"
description: "Master computer vision techniques for robotic perception including image processing, feature detection, depth estimation, and 3D reconstruction"
keywords: [computer vision, robotics, opencv, perception, image processing, depth estimation, 3d reconstruction, ros2, camera calibration]
---

# Chapter 10: Computer Vision for Robotics

## Learning Objectives

By the end of this chapter, you will:

- Understand the fundamentals of computer vision for robotic systems
- Implement image processing pipelines using OpenCV and ROS 2
- Perform camera calibration for accurate measurements
- Apply feature detection and matching for visual navigation
- Estimate depth and reconstruct 3D scenes from images
- Integrate computer vision with robot control systems

## Introduction to Robotic Vision

**Computer vision** is the field of AI that enables machines to interpret and understand visual information from the world. For robots, vision is a primary sensing modality that provides rich information about the environment, enabling navigation, manipulation, and human-robot interaction.

### Why Vision Matters for Physical AI

Robots operating in the real world must perceive their surroundings to act intelligently. Computer vision provides:

- **Environmental awareness**: Understand the layout and contents of a space
- **Object recognition**: Identify and classify objects for manipulation
- **Obstacle detection**: Navigate safely around barriers
- **Pose estimation**: Determine object positions and orientations
- **Human understanding**: Recognize gestures, faces, and intentions

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Computer Vision Pipeline for Robotics                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  Camera  │───▶│   Image      │───▶│   Feature    │───▶│   Object     │ │
│  │  Input   │    │ Processing   │    │  Extraction  │    │  Detection   │ │
│  └──────────┘    └──────────────┘    └──────────────┘    └──────────────┘ │
│                                                                   │        │
│                                                                   ▼        │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  Robot   │◀───│   Motion     │◀───│   Depth      │◀───│   3D Scene   │ │
│  │  Action  │    │  Planning    │    │  Estimation  │    │ Reconstruction│ │
│  └──────────┘    └──────────────┘    └──────────────┘    └──────────────┘ │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

## Vision System Architecture

A complete robotic vision system consists of several interconnected components working together to transform raw sensor data into actionable information.

### Camera Types for Robotics

Different camera technologies serve different purposes in robotics:

| Camera Type | Best For | Typical Use |
|-------------|----------|-------------|
| **RGB Camera** | Color perception, object recognition | General-purpose vision |
| **Depth Camera** | 3D perception, obstacle avoidance | Navigation, manipulation |
| **Stereo Camera** | Outdoor depth estimation | Mobile robots, drones |
| **Event Camera** | High-speed motion tracking | Dynamic environments |
| **Thermal Camera** | Heat detection, night vision | Rescue robots, inspection |

### ROS 2 Vision Stack

ROS 2 provides standardized interfaces for camera data:

```python
#!/usr/bin/env python3
"""
Camera interface node for ROS 2 robotic vision systems.

This node demonstrates how to subscribe to camera topics,
process images, and publish results for downstream nodes.

Topics:
    Subscribed:
        /camera/image_raw (sensor_msgs/Image)
        /camera/camera_info (sensor_msgs/CameraInfo)
        /camera/depth/image_raw (sensor_msgs/Image)

    Published:
        /vision/processed_image (sensor_msgs/Image)
        /vision/detections (vision_msgs/Detection2DArray)

Prerequisites:
    pip install opencv-python numpy
    sudo apt install ros-humble-cv-bridge ros-humble-vision-msgs
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import Optional, Tuple


class VisionSystemNode(Node):
    """
    Core vision processing node for robotic perception.

    Handles camera input, image processing, and publishes
    processed results for other ROS 2 nodes to consume.
    """

    def __init__(self):
        super().__init__('vision_system_node')

        # Initialize CV bridge for ROS <-> OpenCV conversion
        self.bridge = CvBridge()

        # Camera intrinsics (populated from CameraInfo)
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.image_size: Optional[Tuple[int, int]] = None

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )

        # Publishers
        self.processed_pub = self.create_publisher(
            Image,
            '/vision/processed_image',
            10
        )

        # State
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None

        # Processing timer (30 Hz)
        self.timer = self.create_timer(1/30.0, self.process_frame)

        self.get_logger().info('Vision System Node initialized')

    def camera_info_callback(self, msg: CameraInfo):
        """Store camera calibration parameters."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
        self.image_size = (msg.width, msg.height)

    def image_callback(self, msg: Image):
        """Convert and store incoming RGB images."""
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')

    def depth_callback(self, msg: Image):
        """Convert and store incoming depth images."""
        try:
            # Depth images are typically 16UC1 (millimeters) or 32FC1 (meters)
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth conversion error: {e}')

    def process_frame(self):
        """Main processing loop - called at 30 Hz."""
        if self.latest_rgb is None:
            return

        # Example processing pipeline
        processed = self.preprocess_image(self.latest_rgb)

        # Publish processed image
        try:
            msg = self.bridge.cv2_to_imgmsg(processed, 'bgr8')
            self.processed_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Publishing error: {e}')

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply standard preprocessing for robotic vision.

        Args:
            image: Raw BGR image from camera

        Returns:
            Preprocessed image ready for further analysis
        """
        # Undistort if calibration available
        if self.camera_matrix is not None:
            image = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

        # Denoise
        image = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)

        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return image


def main(args=None):
    rclpy.init(args=args)
    node = VisionSystemNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Camera Calibration

Accurate camera calibration is essential for measuring real-world distances and performing 3D reconstruction.

### Understanding Camera Models

Cameras introduce distortions that must be corrected for precise measurements:

```
┌───────────────────────────────────────────────────────────────────┐
│                      Camera Pinhole Model                          │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│    3D World Point                      2D Image Point              │
│    (X, Y, Z)                           (u, v)                      │
│         │                                  ▲                       │
│         │                                  │                       │
│         ▼                                  │                       │
│    ┌─────────┐    Projection Matrix    ┌───────┐                  │
│    │  World  │ ───────────────────────▶│ Image │                  │
│    │  Frame  │    K[R|t]               │ Plane │                  │
│    └─────────┘                         └───────┘                  │
│                                                                    │
│    Intrinsic Matrix K:    │ fx  0  cx │                           │
│                           │ 0  fy  cy │                           │
│                           │ 0   0   1 │                           │
│                                                                    │
│    fx, fy = focal lengths (pixels)                                 │
│    cx, cy = principal point (image center)                         │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

### Calibration Implementation

```python
#!/usr/bin/env python3
"""
Camera calibration utilities for robotic vision systems.

Uses a checkerboard pattern to compute camera intrinsics
and distortion coefficients for accurate measurements.

Prerequisites:
    pip install opencv-python numpy

Usage:
    python camera_calibration.py --images ./calibration_images/
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json


@dataclass
class CalibrationResult:
    """Stores camera calibration results."""
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvecs: List[np.ndarray]
    tvecs: List[np.ndarray]
    reprojection_error: float
    image_size: Tuple[int, int]

    def save(self, filepath: str):
        """Save calibration to JSON file."""
        data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'reprojection_error': self.reprojection_error,
            'image_size': list(self.image_size)
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'CalibrationResult':
        """Load calibration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(
            camera_matrix=np.array(data['camera_matrix']),
            dist_coeffs=np.array(data['dist_coeffs']),
            rvecs=[],
            tvecs=[],
            reprojection_error=data['reprojection_error'],
            image_size=tuple(data['image_size'])
        )


class CameraCalibrator:
    """
    Performs camera calibration using checkerboard pattern.

    The calibration process:
    1. Detect checkerboard corners in multiple images
    2. Compute intrinsic matrix and distortion coefficients
    3. Validate with reprojection error
    """

    def __init__(
        self,
        board_size: Tuple[int, int] = (9, 6),
        square_size: float = 0.025  # meters
    ):
        """
        Initialize calibrator.

        Args:
            board_size: Number of inner corners (columns, rows)
            square_size: Size of each square in meters
        """
        self.board_size = board_size
        self.square_size = square_size

        # Prepare 3D object points for the checkerboard
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        self.objp *= square_size

        # Detection criteria for corner refinement
        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

    def detect_corners(
        self,
        image: np.ndarray,
        visualize: bool = False
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect checkerboard corners in an image.

        Args:
            image: Input image (BGR or grayscale)
            visualize: Whether to show detection result

        Returns:
            (success, corners) tuple
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            self.board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            # Refine corner positions to sub-pixel accuracy
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), self.criteria
            )

            if visualize:
                vis_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(vis_image, self.board_size, corners, ret)
                cv2.imshow('Detected Corners', vis_image)
                cv2.waitKey(500)

        return ret, corners if ret else None

    def calibrate(
        self,
        images: List[np.ndarray],
        verbose: bool = True
    ) -> Optional[CalibrationResult]:
        """
        Perform camera calibration from a list of images.

        Args:
            images: List of calibration images
            verbose: Print progress information

        Returns:
            CalibrationResult or None if calibration failed
        """
        object_points = []  # 3D points in world coordinates
        image_points = []   # 2D points in image coordinates
        image_size = None

        for i, image in enumerate(images):
            if image_size is None:
                image_size = (image.shape[1], image.shape[0])

            ret, corners = self.detect_corners(image)

            if ret:
                object_points.append(self.objp)
                image_points.append(corners)
                if verbose:
                    print(f'Image {i+1}/{len(images)}: corners detected')
            else:
                if verbose:
                    print(f'Image {i+1}/{len(images)}: no corners found')

        if len(object_points) < 3:
            print('Error: Need at least 3 valid calibration images')
            return None

        if verbose:
            print(f'\nCalibrating with {len(object_points)} images...')

        # Perform calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None
        )

        if not ret:
            print('Calibration failed')
            return None

        # Calculate reprojection error
        total_error = 0
        for i in range(len(object_points)):
            projected, _ = cv2.projectPoints(
                object_points[i], rvecs[i], tvecs[i],
                camera_matrix, dist_coeffs
            )
            error = cv2.norm(image_points[i], projected, cv2.NORM_L2)
            total_error += error ** 2

        reprojection_error = np.sqrt(total_error / len(object_points))

        if verbose:
            print(f'\nCalibration successful!')
            print(f'Reprojection error: {reprojection_error:.4f} pixels')
            print(f'\nCamera matrix:\n{camera_matrix}')
            print(f'\nDistortion coefficients: {dist_coeffs.ravel()}')

        return CalibrationResult(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            rvecs=rvecs,
            tvecs=tvecs,
            reprojection_error=reprojection_error,
            image_size=image_size
        )

    def undistort(
        self,
        image: np.ndarray,
        calibration: CalibrationResult
    ) -> np.ndarray:
        """Remove lens distortion from an image."""
        return cv2.undistort(
            image,
            calibration.camera_matrix,
            calibration.dist_coeffs
        )


def calibrate_from_folder(folder_path: str, output_path: str = 'calibration.json'):
    """
    Convenience function to calibrate from a folder of images.

    Args:
        folder_path: Path to folder containing calibration images
        output_path: Where to save calibration results
    """
    calibrator = CameraCalibrator()
    folder = Path(folder_path)

    # Load all images
    images = []
    for ext in ['*.jpg', '*.png', '*.bmp']:
        for img_path in folder.glob(ext):
            image = cv2.imread(str(img_path))
            if image is not None:
                images.append(image)

    print(f'Found {len(images)} images in {folder_path}')

    result = calibrator.calibrate(images)

    if result:
        result.save(output_path)
        print(f'Calibration saved to {output_path}')

    return result


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        calibrate_from_folder(sys.argv[1])
    else:
        print('Usage: python camera_calibration.py <image_folder>')
```

## Feature Detection and Matching

Feature detection enables visual odometry, SLAM, and object tracking by finding distinctive points in images.

### Common Feature Detectors

```python
#!/usr/bin/env python3
"""
Feature detection and matching for robotic vision.

Implements ORB, SIFT, and custom feature pipelines for
visual navigation and object tracking.

Prerequisites:
    pip install opencv-python opencv-contrib-python numpy
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class FeatureType(Enum):
    """Supported feature detector types."""
    ORB = "orb"       # Fast, rotation-invariant, open-source
    SIFT = "sift"     # Scale-invariant, more accurate but slower
    AKAZE = "akaze"   # Modern alternative to SIFT, open-source
    BRISK = "brisk"   # Binary descriptor, very fast


@dataclass
class FeatureMatch:
    """Represents a feature match between two images."""
    pt1: Tuple[float, float]  # Point in image 1
    pt2: Tuple[float, float]  # Point in image 2
    distance: float           # Match quality (lower = better)


class FeatureDetector:
    """
    Unified interface for feature detection and matching.

    Supports multiple feature types with consistent API
    for robotic vision applications.
    """

    def __init__(self, feature_type: FeatureType = FeatureType.ORB):
        """
        Initialize feature detector.

        Args:
            feature_type: Type of features to detect
        """
        self.feature_type = feature_type
        self.detector = self._create_detector()
        self.matcher = self._create_matcher()

    def _create_detector(self):
        """Create the appropriate feature detector."""
        if self.feature_type == FeatureType.ORB:
            return cv2.ORB_create(
                nfeatures=1000,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=31,
                patchSize=31
            )
        elif self.feature_type == FeatureType.SIFT:
            return cv2.SIFT_create(
                nfeatures=1000,
                contrastThreshold=0.04,
                edgeThreshold=10
            )
        elif self.feature_type == FeatureType.AKAZE:
            return cv2.AKAZE_create()
        elif self.feature_type == FeatureType.BRISK:
            return cv2.BRISK_create()
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

    def _create_matcher(self):
        """Create appropriate feature matcher."""
        if self.feature_type in [FeatureType.ORB, FeatureType.BRISK, FeatureType.AKAZE]:
            # Binary descriptors use Hamming distance
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            # Float descriptors use L2 distance
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def detect(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detect features in an image.

        Args:
            image: Input image (BGR or grayscale)
            mask: Optional mask specifying where to detect

        Returns:
            (keypoints, descriptors) tuple
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect and compute descriptors
        keypoints, descriptors = self.detector.detectAndCompute(gray, mask)

        return keypoints, descriptors

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_threshold: float = 0.75
    ) -> List[cv2.DMatch]:
        """
        Match features using Lowe's ratio test.

        Args:
            desc1: Descriptors from image 1
            desc2: Descriptors from image 2
            ratio_threshold: Ratio test threshold (0.7-0.8 typical)

        Returns:
            List of good matches
        """
        if desc1 is None or desc2 is None:
            return []

        # Find k=2 nearest neighbors
        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        return good_matches

    def find_matches(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        ratio_threshold: float = 0.75
    ) -> Tuple[List[FeatureMatch], np.ndarray]:
        """
        Complete pipeline: detect features in both images and match.

        Args:
            image1: First image
            image2: Second image
            ratio_threshold: Match quality threshold

        Returns:
            (matches, visualization) tuple
        """
        # Detect features
        kp1, desc1 = self.detect(image1)
        kp2, desc2 = self.detect(image2)

        # Match features
        raw_matches = self.match(desc1, desc2, ratio_threshold)

        # Convert to FeatureMatch objects
        matches = []
        for m in raw_matches:
            matches.append(FeatureMatch(
                pt1=kp1[m.queryIdx].pt,
                pt2=kp2[m.trainIdx].pt,
                distance=m.distance
            ))

        # Create visualization
        vis = cv2.drawMatches(
            image1, kp1,
            image2, kp2,
            raw_matches,
            None,
            matchColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        return matches, vis

    def compute_homography(
        self,
        matches: List[FeatureMatch],
        ransac_threshold: float = 5.0
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Compute homography from matches using RANSAC.

        Args:
            matches: List of feature matches
            ransac_threshold: RANSAC inlier threshold in pixels

        Returns:
            (homography_matrix, inlier_mask) tuple
        """
        if len(matches) < 4:
            return None, np.array([])

        pts1 = np.float32([m.pt1 for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([m.pt2 for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_threshold)

        return H, mask.ravel() if mask is not None else np.array([])


def demo_feature_matching():
    """Demonstrate feature detection and matching."""
    # Create synthetic test images
    img1 = np.zeros((480, 640, 3), dtype=np.uint8)
    img2 = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add some shapes for features
    cv2.rectangle(img1, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.circle(img1, (400, 300), 50, (0, 255, 255), -1)
    cv2.putText(img1, 'Test', (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    # Create transformed version (rotated and translated)
    M = cv2.getRotationMatrix2D((320, 240), 15, 1.0)
    img2 = cv2.warpAffine(img1, M, (640, 480))

    # Test different feature types
    for ftype in [FeatureType.ORB, FeatureType.AKAZE]:
        detector = FeatureDetector(ftype)

        matches, vis = detector.find_matches(img1, img2)

        print(f"\n{ftype.value.upper()} Features:")
        print(f"  Found {len(matches)} matches")

        if len(matches) >= 4:
            H, inliers = detector.compute_homography(matches)
            if H is not None:
                print(f"  Homography inliers: {inliers.sum()}/{len(matches)}")

        cv2.imshow(f'{ftype.value} Matches', vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    demo_feature_matching()
```

## Depth Estimation and 3D Reconstruction

Robots need to understand the 3D structure of their environment for navigation and manipulation.

### Stereo Vision

```python
#!/usr/bin/env python3
"""
Stereo vision for depth estimation in robotic systems.

Computes depth maps from stereo camera pairs for
3D perception and obstacle avoidance.

Prerequisites:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class StereoCalibration:
    """Stereo camera calibration parameters."""
    camera_matrix_left: np.ndarray
    dist_coeffs_left: np.ndarray
    camera_matrix_right: np.ndarray
    dist_coeffs_right: np.ndarray
    R: np.ndarray  # Rotation between cameras
    T: np.ndarray  # Translation between cameras
    baseline: float  # Distance between cameras (meters)


class StereoDepthEstimator:
    """
    Computes depth from stereo image pairs.

    Uses semi-global block matching (SGBM) for dense
    depth estimation with sub-pixel accuracy.
    """

    def __init__(
        self,
        calibration: Optional[StereoCalibration] = None,
        image_size: Tuple[int, int] = (640, 480)
    ):
        """
        Initialize stereo depth estimator.

        Args:
            calibration: Stereo calibration parameters
            image_size: Expected image size (width, height)
        """
        self.calibration = calibration
        self.image_size = image_size

        # Stereo matcher parameters
        self.num_disparities = 128  # Must be divisible by 16
        self.block_size = 11       # Odd number >= 1

        # Create stereo matcher
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=8 * 3 * self.block_size ** 2,
            P2=32 * 3 * self.block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Rectification maps (computed from calibration)
        self.map_left = None
        self.map_right = None
        self.Q = None  # Disparity-to-depth mapping matrix

        if calibration:
            self._compute_rectification_maps()

    def _compute_rectification_maps(self):
        """Compute stereo rectification maps from calibration."""
        cal = self.calibration

        # Stereo rectification
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            cal.camera_matrix_left,
            cal.dist_coeffs_left,
            cal.camera_matrix_right,
            cal.dist_coeffs_right,
            self.image_size,
            cal.R,
            cal.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )

        # Compute rectification maps
        self.map_left = cv2.initUndistortRectifyMap(
            cal.camera_matrix_left,
            cal.dist_coeffs_left,
            R1, P1,
            self.image_size,
            cv2.CV_32FC1
        )

        self.map_right = cv2.initUndistortRectifyMap(
            cal.camera_matrix_right,
            cal.dist_coeffs_right,
            R2, P2,
            self.image_size,
            cv2.CV_32FC1
        )

        self.Q = Q

    def rectify(
        self,
        left: np.ndarray,
        right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo image pair.

        Args:
            left: Left camera image
            right: Right camera image

        Returns:
            (rectified_left, rectified_right) tuple
        """
        if self.map_left is None:
            return left, right

        rect_left = cv2.remap(
            left,
            self.map_left[0],
            self.map_left[1],
            cv2.INTER_LINEAR
        )

        rect_right = cv2.remap(
            right,
            self.map_right[0],
            self.map_right[1],
            cv2.INTER_LINEAR
        )

        return rect_left, rect_right

    def compute_disparity(
        self,
        left: np.ndarray,
        right: np.ndarray
    ) -> np.ndarray:
        """
        Compute disparity map from stereo pair.

        Args:
            left: Left camera image (grayscale or BGR)
            right: Right camera image (grayscale or BGR)

        Returns:
            Disparity map (float32, in pixels)
        """
        # Convert to grayscale if needed
        if len(left.shape) == 3:
            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left
            right_gray = right

        # Rectify images
        rect_left, rect_right = self.rectify(left_gray, right_gray)

        # Compute disparity
        disparity = self.stereo_matcher.compute(rect_left, rect_right)

        # Convert to float (disparity is in fixed-point with 4 fractional bits)
        disparity = disparity.astype(np.float32) / 16.0

        return disparity

    def disparity_to_depth(
        self,
        disparity: np.ndarray,
        baseline: Optional[float] = None,
        focal_length: Optional[float] = None
    ) -> np.ndarray:
        """
        Convert disparity map to depth map.

        Args:
            disparity: Disparity map in pixels
            baseline: Camera baseline in meters (optional if calibrated)
            focal_length: Focal length in pixels (optional if calibrated)

        Returns:
            Depth map in meters
        """
        # Use calibration values if available
        if baseline is None and self.calibration:
            baseline = self.calibration.baseline
        if focal_length is None and self.calibration:
            focal_length = self.calibration.camera_matrix_left[0, 0]

        if baseline is None or focal_length is None:
            raise ValueError("Baseline and focal length required for depth computation")

        # Avoid division by zero
        disparity_safe = np.where(disparity > 0, disparity, 0.1)

        # depth = baseline * focal_length / disparity
        depth = (baseline * focal_length) / disparity_safe

        # Mask invalid disparities
        depth = np.where(disparity > 0, depth, 0)

        return depth

    def compute_point_cloud(
        self,
        disparity: np.ndarray,
        image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate 3D point cloud from disparity map.

        Args:
            disparity: Disparity map
            image: Optional RGB image for coloring points

        Returns:
            Point cloud as Nx3 (or Nx6 with color) array
        """
        if self.Q is None:
            raise ValueError("Calibration required for point cloud generation")

        # Reproject to 3D
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)

        # Create mask for valid points
        mask = disparity > 0

        # Extract valid points
        points = points_3d[mask]

        # Add color if image provided
        if image is not None:
            if len(image.shape) == 3:
                colors = image[mask]
            else:
                colors = np.stack([image[mask]] * 3, axis=-1)
            points = np.hstack([points, colors])

        return points


def visualize_disparity(disparity: np.ndarray) -> np.ndarray:
    """Create colorized visualization of disparity map."""
    # Normalize to 0-255
    disp_vis = disparity.copy()
    disp_vis = np.clip(disp_vis, 0, disp_vis.max())
    disp_vis = (disp_vis / disp_vis.max() * 255).astype(np.uint8)

    # Apply colormap
    colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    # Mask invalid regions
    colored[disparity <= 0] = 0

    return colored


def demo_stereo_depth():
    """Demonstrate stereo depth estimation."""
    # Create synthetic stereo pair
    # In practice, these would come from real stereo cameras

    # Create left image with objects at different depths
    left = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(left, (100, 100), (200, 300), (255, 255, 255), -1)  # Near
    cv2.rectangle(left, (400, 150), (500, 350), (128, 128, 128), -1)  # Far

    # Create right image (shifted version simulating stereo)
    right = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(right, (80, 100), (180, 300), (255, 255, 255), -1)   # 20px shift (near)
    cv2.rectangle(right, (395, 150), (495, 350), (128, 128, 128), -1)  # 5px shift (far)

    # Compute depth
    estimator = StereoDepthEstimator()
    disparity = estimator.compute_disparity(left, right)

    # Visualize
    vis_disparity = visualize_disparity(disparity)

    combined = np.vstack([
        np.hstack([left, right]),
        np.hstack([vis_disparity, np.zeros_like(left)])
    ])

    cv2.putText(combined, 'Left', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, 'Right', (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, 'Disparity', (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Stereo Depth Demo', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    demo_stereo_depth()
```

## Visual Odometry

Visual odometry estimates robot motion from camera images, essential for navigation when wheel encoders are unreliable.

```python
#!/usr/bin/env python3
"""
Visual odometry for robot pose estimation.

Tracks camera motion through feature matching between
consecutive frames for localization and mapping.

Prerequisites:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Pose:
    """Robot pose in 3D space."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    def transform_matrix(self) -> np.ndarray:
        """Get 4x4 transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.position
        return T


class VisualOdometry:
    """
    Monocular visual odometry using feature tracking.

    Estimates camera motion by tracking features between
    consecutive frames and computing the essential matrix.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None
    ):
        """
        Initialize visual odometry.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        self.K = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)

        # Feature detector
        self.detector = cv2.ORB_create(nfeatures=2000)

        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # State
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_keypoints: Optional[List[cv2.KeyPoint]] = None
        self.prev_descriptors: Optional[np.ndarray] = None

        # Accumulated pose
        self.pose = Pose()
        self.trajectory: List[np.ndarray] = [self.pose.position.copy()]

    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[Pose, Optional[np.ndarray]]:
        """
        Process a new frame and update pose estimate.

        Args:
            frame: New camera frame (BGR or grayscale)

        Returns:
            (current_pose, visualization) tuple
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Undistort
        gray = cv2.undistort(gray, self.K, self.dist_coeffs)

        # Detect features
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        vis = None

        if self.prev_frame is not None and descriptors is not None:
            # Match features
            matches = self._match_features(
                self.prev_descriptors,
                descriptors,
                self.prev_keypoints,
                keypoints
            )

            if len(matches) >= 8:
                # Estimate motion
                R, t, inliers = self._estimate_motion(matches)

                if R is not None:
                    # Update pose
                    self.pose.position += self.pose.rotation @ t.ravel()
                    self.pose.rotation = R @ self.pose.rotation

                    self.trajectory.append(self.pose.position.copy())

                # Create visualization
                vis = self._visualize_matches(
                    self.prev_frame, gray,
                    self.prev_keypoints, keypoints,
                    matches, inliers
                )

        # Update state
        self.prev_frame = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

        return self.pose, vis

    def _match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint]
    ) -> List[Tuple[cv2.KeyPoint, cv2.KeyPoint, cv2.DMatch]]:
        """Match features between frames using ratio test."""
        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append((
                        kp1[m.queryIdx],
                        kp2[m.trainIdx],
                        m
                    ))

        return good_matches

    def _estimate_motion(
        self,
        matches: List[Tuple[cv2.KeyPoint, cv2.KeyPoint, cv2.DMatch]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Estimate camera motion from matched features."""
        # Extract point correspondences
        pts1 = np.float32([m[0].pt for m in matches])
        pts2 = np.float32([m[1].pt for m in matches])

        # Compute essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            return None, None, None

        # Recover pose from essential matrix
        _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, self.K, mask)

        return R, t, pose_mask

    def _visualize_matches(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        kp1: List[cv2.KeyPoint],
        kp2: List[cv2.KeyPoint],
        matches: List[Tuple],
        inlier_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """Create visualization of feature matches."""
        h, w = img1.shape[:2]
        vis = np.zeros((h, w * 2, 3), dtype=np.uint8)

        # Convert grayscale to color
        vis[:, :w] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        vis[:, w:] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        # Draw matches
        for i, (kp_prev, kp_curr, _) in enumerate(matches):
            color = (0, 255, 0) if inlier_mask is None or inlier_mask[i] else (0, 0, 255)

            pt1 = (int(kp_prev.pt[0]), int(kp_prev.pt[1]))
            pt2 = (int(kp_curr.pt[0]) + w, int(kp_curr.pt[1]))

            cv2.circle(vis, pt1, 3, color, -1)
            cv2.circle(vis, pt2, 3, color, -1)
            cv2.line(vis, pt1, pt2, color, 1)

        # Add pose info
        pos = self.pose.position
        cv2.putText(
            vis,
            f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return vis

    def get_trajectory_image(self, scale: float = 100.0) -> np.ndarray:
        """Create top-down view of trajectory."""
        # Create image
        img_size = 800
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        center = img_size // 2

        # Draw trajectory
        for i in range(1, len(self.trajectory)):
            p1 = self.trajectory[i - 1]
            p2 = self.trajectory[i]

            pt1 = (int(center + p1[0] * scale), int(center - p1[2] * scale))
            pt2 = (int(center + p2[0] * scale), int(center - p2[2] * scale))

            cv2.line(img, pt1, pt2, (0, 255, 0), 2)

        # Draw current position
        curr = self.trajectory[-1]
        curr_pt = (int(center + curr[0] * scale), int(center - curr[2] * scale))
        cv2.circle(img, curr_pt, 5, (0, 0, 255), -1)

        # Draw start
        start_pt = (center, center)
        cv2.circle(img, start_pt, 5, (255, 0, 0), -1)

        # Labels
        cv2.putText(img, "Start", (center + 10, center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(img, "Current", (curr_pt[0] + 10, curr_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return img


def demo_visual_odometry():
    """Demonstrate visual odometry with webcam."""
    # Approximate camera matrix (adjust for your camera)
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)

    vo = VisualOdometry(K)
    cap = cv2.VideoCapture(0)

    print("Visual Odometry Demo")
    print("Move the camera slowly. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pose, vis = vo.process_frame(frame)

        if vis is not None:
            # Show feature matches
            cv2.imshow('Feature Matches', vis)

        # Show trajectory
        traj = vo.get_trajectory_image()
        cv2.imshow('Trajectory', traj)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    demo_visual_odometry()
```

## Integration with ROS 2

Integrating vision pipelines with ROS 2 enables modular, reusable perception systems.

```python
#!/usr/bin/env python3
"""
Complete ROS 2 vision pipeline node.

Integrates camera input, feature detection, depth estimation,
and publishes results for downstream navigation and manipulation.

Topics:
    Subscribed:
        /camera/image_raw (sensor_msgs/Image)
        /camera/depth/image_raw (sensor_msgs/Image)
        /camera/camera_info (sensor_msgs/CameraInfo)

    Published:
        /vision/features (custom feature message)
        /vision/depth_colorized (sensor_msgs/Image)
        /vision/point_cloud (sensor_msgs/PointCloud2)

Services:
    /vision/detect_object (custom service)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import Optional
import struct


class VisionPipelineNode(Node):
    """
    Comprehensive vision pipeline for robotic perception.

    Processes camera data to extract features, compute depth,
    and generate 3D point clouds for navigation and manipulation.
    """

    def __init__(self):
        super().__init__('vision_pipeline_node')

        # Parameters
        self.declare_parameter('feature_type', 'orb')
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('enable_depth', True)

        self.feature_type = self.get_parameter('feature_type').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.enable_depth = self.get_parameter('enable_depth').value

        # Initialize components
        self.bridge = CvBridge()
        self.detector = cv2.ORB_create(nfeatures=1000)

        # Camera data
        self.camera_matrix: Optional[np.ndarray] = None
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/image_raw',
            self.rgb_callback, 10
        )

        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw',
            self.depth_callback, 10
        )

        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info',
            self.info_callback, 10
        )

        # Publishers
        self.depth_vis_pub = self.create_publisher(
            Image, '/vision/depth_colorized', 10
        )

        self.pointcloud_pub = self.create_publisher(
            PointCloud2, '/vision/point_cloud', 10
        )

        self.features_pub = self.create_publisher(
            Image, '/vision/features', 10
        )

        # Processing timer
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.process_callback
        )

        self.get_logger().info('Vision Pipeline Node started')

    def rgb_callback(self, msg: Image):
        """Handle incoming RGB images."""
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB conversion error: {e}')

    def depth_callback(self, msg: Image):
        """Handle incoming depth images."""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth conversion error: {e}')

    def info_callback(self, msg: CameraInfo):
        """Store camera calibration."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def process_callback(self):
        """Main processing loop."""
        if self.latest_rgb is None:
            return

        # Detect and visualize features
        gray = cv2.cvtColor(self.latest_rgb, cv2.COLOR_BGR2GRAY)
        keypoints, _ = self.detector.detectAndCompute(gray, None)

        # Draw features
        features_img = cv2.drawKeypoints(
            self.latest_rgb, keypoints, None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # Publish feature visualization
        try:
            msg = self.bridge.cv2_to_imgmsg(features_img, 'bgr8')
            self.features_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Feature publish error: {e}')

        # Process depth if available
        if self.enable_depth and self.latest_depth is not None:
            self.process_depth()

    def process_depth(self):
        """Process and publish depth data."""
        # Colorize depth for visualization
        depth_norm = cv2.normalize(
            self.latest_depth, None, 0, 255, cv2.NORM_MINMAX
        )
        depth_color = cv2.applyColorMap(
            depth_norm.astype(np.uint8),
            cv2.COLORMAP_JET
        )

        # Publish colorized depth
        try:
            msg = self.bridge.cv2_to_imgmsg(depth_color, 'bgr8')
            self.depth_vis_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Depth vis publish error: {e}')

        # Generate and publish point cloud
        if self.camera_matrix is not None:
            cloud_msg = self.create_point_cloud()
            if cloud_msg:
                self.pointcloud_pub.publish(cloud_msg)

    def create_point_cloud(self) -> Optional[PointCloud2]:
        """Create PointCloud2 message from depth data."""
        if self.latest_depth is None or self.camera_matrix is None:
            return None

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        h, w = self.latest_depth.shape

        # Create point cloud
        points = []

        # Downsample for performance
        step = 4
        for v in range(0, h, step):
            for u in range(0, w, step):
                z = self.latest_depth[v, u]

                if z <= 0 or z > 10000:  # Invalid or too far
                    continue

                # Convert to meters if in millimeters
                z_m = z / 1000.0 if z > 100 else z

                # Back-project to 3D
                x = (u - cx) * z_m / fx
                y = (v - cy) * z_m / fy

                # Get color
                if self.latest_rgb is not None:
                    b, g, r = self.latest_rgb[v, u]
                else:
                    r = g = b = 255

                # Pack RGB into single float
                rgb = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]

                points.append([x, y, z_m, rgb])

        if not points:
            return None

        # Create PointCloud2 message
        header = Header()
        header.frame_id = 'camera_optical_frame'
        header.stamp = self.get_clock().now().to_msg()

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        cloud_data = np.array(points, dtype=np.float32).tobytes()

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * len(points)
        msg.data = cloud_data
        msg.is_dense = True

        return msg


def main(args=None):
    rclpy.init(args=args)
    node = VisionPipelineNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Computer Vision Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                Complete Robotic Vision Pipeline                               │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Camera    │────▶│   Image     │────▶│   Feature   │────▶│   Object    │
│   Driver    │     │ Preprocessing│     │  Detection  │     │  Detection  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │                    │
      │                   │                   │                    │
      ▼                   ▼                   ▼                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Depth     │────▶│   Stereo    │────▶│   Visual    │────▶│    SLAM     │
│   Sensor    │     │  Matching   │     │  Odometry   │     │   Mapping   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │                    │
      │                   │                   │                    │
      └───────────────────┴───────────────────┴────────────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │   Robot Control │
                          │   Navigation &  │
                          │   Manipulation  │
                          └─────────────────┘
```

## Summary

In this chapter, you learned:

- **Vision system architecture**: How camera data flows through processing pipelines to robot actions
- **Camera calibration**: Computing intrinsics and removing distortion for accurate measurements
- **Feature detection**: Using ORB, SIFT, and other detectors for visual tracking
- **Depth estimation**: Stereo matching and depth cameras for 3D perception
- **Visual odometry**: Estimating robot motion from image sequences
- **ROS 2 integration**: Building modular vision pipelines with standard interfaces

### Key Takeaways

- Proper **camera calibration** is essential for accurate 3D measurements
- **Feature detection and matching** enables tracking, localization, and mapping
- **Stereo vision** provides dense depth information for navigation
- **Visual odometry** complements wheel encoders for robust localization
- ROS 2 provides **standardized interfaces** for building reusable vision components

## Exercises

### Exercise 1: Camera Calibration Practice

1. Print a checkerboard pattern (9x6 inner corners recommended)
2. Capture 15-20 images from different angles
3. Run the calibration script and evaluate reprojection error
4. Test undistortion on a new image

### Exercise 2: Feature Tracking Application

1. Modify the feature detector to track features across multiple frames
2. Implement a simple tracker that follows a specific object
3. Visualize the tracking trajectory over time

### Exercise 3: Depth-Based Obstacle Detection

1. Use the depth estimation code with a depth camera
2. Implement a simple obstacle detector (find regions closer than threshold)
3. Publish obstacle locations as ROS 2 markers

### Challenge: Build a Visual SLAM System

Extend the visual odometry code to:
1. Maintain a map of observed features
2. Perform loop closure detection
3. Optimize the trajectory using bundle adjustment

## Up Next

In the **Capstone Project**, we'll integrate all components from this book to build a complete humanoid robot system that combines ROS 2 communication, simulation, perception, and voice-controlled action.

## Additional Resources

- [OpenCV Documentation](https://docs.opencv.org/) - Comprehensive computer vision library
- [ROS 2 Vision Packages](https://github.com/ros-perception) - Pre-built vision components
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) - State-of-the-art visual SLAM
- [Open3D](http://www.open3d.org/) - 3D data processing library
- [Multiple View Geometry Book](https://www.robots.ox.ac.uk/~vgg/hzbook/) - Foundational computer vision theory

---

**Sources:**
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
- [ROS 2 Image Pipeline](https://github.com/ros-perception/image_pipeline)
- [Camera Calibration Theory](https://www.mathworks.com/help/vision/ug/camera-calibration.html)
- [Visual SLAM Overview](https://arxiv.org/abs/2102.04060)
