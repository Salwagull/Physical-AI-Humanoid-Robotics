---
sidebar_position: 11
title: "Chapter 11: 3D Perception & Depth Sensing"
description: "Master 3D perception technologies including depth cameras, LiDAR, point cloud processing, and spatial understanding for robotic applications"
keywords: [3d perception, depth sensing, lidar, point cloud, rgb-d, realsense, spatial understanding, ros2, pcl, open3d]
---

# Chapter 11: 3D Perception & Depth Sensing

## Learning Objectives

By the end of this chapter, you will:

- Understand different depth sensing technologies and their trade-offs
- Work with RGB-D cameras and LiDAR sensors in ROS 2
- Process and filter point cloud data using PCL and Open3D
- Implement plane detection, clustering, and object segmentation
- Build spatial maps for robot navigation and manipulation
- Integrate 3D perception with robot control systems

## Introduction to 3D Perception

**3D perception** enables robots to understand the spatial structure of their environment. Unlike 2D images that capture appearance, 3D data provides geometric information essential for physical interaction with the world.

### Why 3D Perception Matters for Physical AI

Robots operating in the real world must reason about space and geometry to:

- **Navigate safely**: Detect obstacles and plan collision-free paths
- **Manipulate objects**: Understand object shapes for grasping
- **Build maps**: Create spatial representations for localization
- **Understand scenes**: Recognize surfaces, objects, and their relationships

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     3D Perception Pipeline                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │    Depth     │    │    Point     │    │   Spatial    │                  │
│  │   Sensors    │───▶│    Cloud     │───▶│  Analysis    │                  │
│  │ RGB-D/LiDAR  │    │  Generation  │    │              │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                   │                   │                          │
│         │                   │                   │                          │
│         ▼                   ▼                   ▼                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │    Depth     │    │  Filtering   │    │   Plane &    │                  │
│  │     Map      │    │ Downsampling │    │   Object     │                  │
│  │              │    │   Outliers   │    │  Detection   │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                 │                          │
│                                                 ▼                          │
│                                          ┌──────────────┐                  │
│                                          │    Robot     │                  │
│                                          │   Control    │                  │
│                                          │  Navigation  │                  │
│                                          │ Manipulation │                  │
│                                          └──────────────┘                  │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

## Depth Sensing Technologies

Different sensors provide 3D data with varying characteristics suitable for different applications.

### Comparison of Depth Sensors

| Technology | Range | Resolution | Outdoor | Cost | Best For |
|------------|-------|------------|---------|------|----------|
| **Structured Light** | 0.2-4m | High | Poor | Low | Indoor manipulation |
| **Time-of-Flight (ToF)** | 0.1-10m | Medium | Fair | Medium | Mobile robots |
| **Stereo Vision** | 0.5-20m | Variable | Good | Low | Outdoor navigation |
| **LiDAR** | 1-200m | High | Excellent | High | Autonomous vehicles |
| **Active Stereo** | 0.3-10m | High | Fair | Medium | General purpose |

### RGB-D Cameras

RGB-D cameras combine color imaging with depth sensing, providing rich perceptual data.

```python
#!/usr/bin/env python3
"""
RGB-D camera interface for 3D perception.

Demonstrates working with Intel RealSense D435/D455 cameras
for robotic perception applications.

Prerequisites:
    pip install pyrealsense2 opencv-python numpy

Hardware:
    Intel RealSense D435, D455, or similar RGB-D camera
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Optional
import time

# Try to import RealSense SDK
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not installed. Using simulated data.")


@dataclass
class DepthFrame:
    """Container for synchronized RGB-D data."""
    color: np.ndarray          # RGB image (H, W, 3)
    depth: np.ndarray          # Depth in meters (H, W)
    depth_raw: np.ndarray      # Raw depth in millimeters (H, W)
    timestamp: float           # Frame timestamp
    intrinsics: dict           # Camera parameters


class RGBDCamera:
    """
    Interface for RGB-D camera sensors.

    Provides synchronized color and depth streams with
    proper calibration for 3D reconstruction.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        align_depth: bool = True
    ):
        """
        Initialize RGB-D camera.

        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second
            align_depth: Align depth to color frame
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.align_depth = align_depth

        self.pipeline = None
        self.align = None
        self.intrinsics = None

        if REALSENSE_AVAILABLE:
            self._init_realsense()
        else:
            self._init_simulated()

    def _init_realsense(self):
        """Initialize Intel RealSense camera."""
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Enable streams
        config.enable_stream(
            rs.stream.color, self.width, self.height,
            rs.format.bgr8, self.fps
        )
        config.enable_stream(
            rs.stream.depth, self.width, self.height,
            rs.format.z16, self.fps
        )

        # Start pipeline
        profile = self.pipeline.start(config)

        # Get depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Set up alignment
        if self.align_depth:
            self.align = rs.align(rs.stream.color)

        # Get intrinsics
        depth_profile = profile.get_stream(rs.stream.depth)
        intr = depth_profile.as_video_stream_profile().get_intrinsics()

        self.intrinsics = {
            'fx': intr.fx,
            'fy': intr.fy,
            'cx': intr.ppx,
            'cy': intr.ppy,
            'width': intr.width,
            'height': intr.height
        }

    def _init_simulated(self):
        """Initialize simulated depth data for testing."""
        self.depth_scale = 0.001  # 1mm per unit

        # Simulated intrinsics
        self.intrinsics = {
            'fx': 600.0,
            'fy': 600.0,
            'cx': self.width / 2,
            'cy': self.height / 2,
            'width': self.width,
            'height': self.height
        }

    def get_frame(self) -> Optional[DepthFrame]:
        """
        Capture synchronized RGB-D frame.

        Returns:
            DepthFrame with color and depth data, or None if capture failed
        """
        if REALSENSE_AVAILABLE and self.pipeline:
            return self._get_realsense_frame()
        else:
            return self._get_simulated_frame()

    def _get_realsense_frame(self) -> Optional[DepthFrame]:
        """Get frame from RealSense camera."""
        frames = self.pipeline.wait_for_frames()

        if self.align:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None

        # Convert to numpy arrays
        color = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth = depth_raw.astype(np.float32) * self.depth_scale

        return DepthFrame(
            color=color,
            depth=depth,
            depth_raw=depth_raw,
            timestamp=time.time(),
            intrinsics=self.intrinsics
        )

    def _get_simulated_frame(self) -> DepthFrame:
        """Generate simulated RGB-D data for testing."""
        # Simulated color image
        color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.rectangle(color, (200, 150), (440, 330), (0, 255, 0), -1)
        cv2.circle(color, (500, 350), 80, (255, 0, 0), -1)

        # Simulated depth (objects at different distances)
        depth = np.ones((self.height, self.width), dtype=np.float32) * 3.0
        depth[150:330, 200:440] = 1.5  # Rectangle at 1.5m
        cv2.circle(depth, (500, 350), 80, 1.0, -1)  # Circle at 1.0m

        depth_raw = (depth / self.depth_scale).astype(np.uint16)

        return DepthFrame(
            color=color,
            depth=depth,
            depth_raw=depth_raw,
            timestamp=time.time(),
            intrinsics=self.intrinsics
        )

    def depth_to_pointcloud(
        self,
        depth: np.ndarray,
        color: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert depth image to 3D point cloud.

        Args:
            depth: Depth image in meters (H, W)
            color: Optional RGB image for colored points

        Returns:
            Point cloud as (N, 3) or (N, 6) array with XYZ [RGB]
        """
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        cx = self.intrinsics['cx']
        cy = self.intrinsics['cy']

        # Create pixel coordinate grids
        h, w = depth.shape
        u = np.arange(w)
        v = np.arange(h)
        u, v = np.meshgrid(u, v)

        # Back-project to 3D
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Stack into point cloud
        points = np.stack([x, y, z], axis=-1)

        # Filter invalid points
        valid = (z > 0.1) & (z < 10.0)
        points = points[valid]

        if color is not None:
            colors = color[valid]
            points = np.hstack([points, colors])

        return points

    def close(self):
        """Release camera resources."""
        if self.pipeline:
            self.pipeline.stop()


def visualize_depth(depth: np.ndarray, max_depth: float = 5.0) -> np.ndarray:
    """Create colorized visualization of depth map."""
    depth_normalized = np.clip(depth / max_depth, 0, 1)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    colorized = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    colorized[depth <= 0] = 0  # Mask invalid regions
    return colorized


def demo_rgbd_camera():
    """Demonstrate RGB-D camera usage."""
    camera = RGBDCamera()

    print("RGB-D Camera Demo")
    print("Press 'q' to quit, 's' to save point cloud")

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        # Visualize
        depth_vis = visualize_depth(frame.depth)
        combined = np.hstack([frame.color, depth_vis])

        # Add info overlay
        cv2.putText(
            combined, f"Depth range: {frame.depth[frame.depth > 0].min():.2f}m - {frame.depth.max():.2f}m",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        cv2.imshow('RGB-D Camera', combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            points = camera.depth_to_pointcloud(frame.depth, frame.color)
            np.save('pointcloud.npy', points)
            print(f"Saved {len(points)} points to pointcloud.npy")

    camera.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    demo_rgbd_camera()
```

### LiDAR Sensors

LiDAR (Light Detection and Ranging) provides precise, long-range 3D measurements.

```python
#!/usr/bin/env python3
"""
LiDAR interface for 3D perception.

Demonstrates processing LiDAR point clouds for
robotic navigation and obstacle detection.

Prerequisites:
    pip install numpy open3d

ROS 2 Integration:
    Subscribes to sensor_msgs/PointCloud2 topics
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import struct


@dataclass
class LiDARScan:
    """Container for LiDAR scan data."""
    points: np.ndarray         # (N, 3) XYZ coordinates
    intensities: np.ndarray    # (N,) Intensity values
    ring_ids: np.ndarray       # (N,) Ring/layer indices
    timestamps: np.ndarray     # (N,) Point timestamps
    frame_id: str              # Coordinate frame


class LiDARProcessor:
    """
    Process LiDAR point cloud data for robotics.

    Handles common LiDAR processing tasks including
    filtering, ground removal, and obstacle detection.
    """

    def __init__(
        self,
        min_range: float = 0.5,
        max_range: float = 100.0,
        min_height: float = -2.0,
        max_height: float = 3.0
    ):
        """
        Initialize LiDAR processor.

        Args:
            min_range: Minimum valid range (meters)
            max_range: Maximum valid range (meters)
            min_height: Minimum height filter (meters)
            max_height: Maximum height filter (meters)
        """
        self.min_range = min_range
        self.max_range = max_range
        self.min_height = min_height
        self.max_height = max_height

    def filter_by_range(self, points: np.ndarray) -> np.ndarray:
        """
        Filter points by distance from sensor.

        Args:
            points: (N, 3) point cloud

        Returns:
            Filtered point cloud
        """
        distances = np.linalg.norm(points[:, :2], axis=1)  # XY distance
        mask = (distances >= self.min_range) & (distances <= self.max_range)
        return points[mask]

    def filter_by_height(self, points: np.ndarray) -> np.ndarray:
        """
        Filter points by height (Z coordinate).

        Args:
            points: (N, 3) point cloud

        Returns:
            Filtered point cloud
        """
        mask = (points[:, 2] >= self.min_height) & (points[:, 2] <= self.max_height)
        return points[mask]

    def remove_ground_ransac(
        self,
        points: np.ndarray,
        distance_threshold: float = 0.15,
        max_iterations: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove ground plane using RANSAC.

        Args:
            points: (N, 3) point cloud
            distance_threshold: Max distance from plane to be inlier
            max_iterations: RANSAC iterations

        Returns:
            (non_ground_points, ground_points, plane_coefficients)
        """
        best_inliers = None
        best_plane = None
        n_points = len(points)

        for _ in range(max_iterations):
            # Sample 3 random points
            indices = np.random.choice(n_points, 3, replace=False)
            sample = points[indices]

            # Fit plane through 3 points
            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            normal = np.cross(v1, v2)

            if np.linalg.norm(normal) < 1e-6:
                continue

            normal = normal / np.linalg.norm(normal)

            # Plane equation: ax + by + cz + d = 0
            d = -np.dot(normal, sample[0])
            plane = np.append(normal, d)

            # Count inliers
            distances = np.abs(np.dot(points, normal) + d)
            inliers = distances < distance_threshold

            if best_inliers is None or inliers.sum() > best_inliers.sum():
                best_inliers = inliers
                best_plane = plane

        if best_inliers is None:
            return points, np.array([]), None

        ground_points = points[best_inliers]
        non_ground_points = points[~best_inliers]

        return non_ground_points, ground_points, best_plane

    def cluster_obstacles(
        self,
        points: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 10
    ) -> List[np.ndarray]:
        """
        Cluster points into distinct obstacles using DBSCAN.

        Args:
            points: (N, 3) point cloud
            eps: Maximum distance between neighbors
            min_samples: Minimum points to form cluster

        Returns:
            List of point cloud clusters
        """
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            print("sklearn required for clustering")
            return [points]

        if len(points) < min_samples:
            return []

        # Cluster using DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_

        # Extract clusters (ignore noise label -1)
        clusters = []
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                continue  # Skip noise
            cluster_mask = labels == label
            clusters.append(points[cluster_mask])

        return clusters

    def compute_obstacle_bboxes(
        self,
        clusters: List[np.ndarray]
    ) -> List[dict]:
        """
        Compute 3D bounding boxes for obstacle clusters.

        Args:
            clusters: List of point cloud clusters

        Returns:
            List of bounding box dictionaries
        """
        bboxes = []

        for cluster in clusters:
            if len(cluster) < 3:
                continue

            # Axis-aligned bounding box
            min_pt = cluster.min(axis=0)
            max_pt = cluster.max(axis=0)
            center = (min_pt + max_pt) / 2
            dimensions = max_pt - min_pt

            bboxes.append({
                'center': center,
                'dimensions': dimensions,
                'min_point': min_pt,
                'max_point': max_pt,
                'num_points': len(cluster)
            })

        return bboxes

    def process_scan(
        self,
        points: np.ndarray
    ) -> dict:
        """
        Full processing pipeline for LiDAR scan.

        Args:
            points: (N, 3) raw point cloud

        Returns:
            Dictionary with processed results
        """
        # Apply filters
        filtered = self.filter_by_range(points)
        filtered = self.filter_by_height(filtered)

        # Remove ground
        obstacles, ground, plane = self.remove_ground_ransac(filtered)

        # Cluster obstacles
        clusters = self.cluster_obstacles(obstacles)

        # Compute bounding boxes
        bboxes = self.compute_obstacle_bboxes(clusters)

        return {
            'filtered_points': filtered,
            'obstacle_points': obstacles,
            'ground_points': ground,
            'ground_plane': plane,
            'clusters': clusters,
            'bounding_boxes': bboxes
        }


def generate_simulated_lidar_scan(
    num_rings: int = 16,
    points_per_ring: int = 1800,
    max_range: float = 50.0
) -> np.ndarray:
    """
    Generate simulated LiDAR scan data for testing.

    Args:
        num_rings: Number of vertical rings
        points_per_ring: Horizontal resolution
        max_range: Maximum range

    Returns:
        (N, 3) point cloud
    """
    points = []

    # Vertical angles (typical 16-beam LiDAR)
    vertical_angles = np.linspace(-15, 15, num_rings) * np.pi / 180

    for v_angle in vertical_angles:
        # Horizontal sweep
        h_angles = np.linspace(0, 2 * np.pi, points_per_ring, endpoint=False)

        for h_angle in h_angles:
            # Simulate ground plane
            if v_angle < 0:
                # Range to ground at this angle
                ground_height = -1.5  # Sensor 1.5m above ground
                range_to_ground = abs(ground_height / np.sin(v_angle))

                if range_to_ground < max_range:
                    x = range_to_ground * np.cos(v_angle) * np.cos(h_angle)
                    y = range_to_ground * np.cos(v_angle) * np.sin(h_angle)
                    z = range_to_ground * np.sin(v_angle)
                    points.append([x, y, z])

            # Add some obstacles
            # Obstacle 1: Wall-like structure
            if 0.3 < h_angle < 0.8:
                wall_dist = 5.0
                x = wall_dist * np.cos(h_angle)
                y = wall_dist * np.sin(h_angle)
                z = np.sin(v_angle) * wall_dist + 0.5
                if -1 < z < 2:
                    points.append([x, y, z])

            # Obstacle 2: Cylindrical object
            obstacle_pos = np.array([3.0, -2.0])
            obstacle_radius = 0.5
            ray_dir = np.array([np.cos(h_angle), np.sin(h_angle)])

            # Ray-cylinder intersection
            a = np.dot(ray_dir, ray_dir)
            b = 2 * np.dot(ray_dir, -obstacle_pos)
            c = np.dot(obstacle_pos, obstacle_pos) - obstacle_radius**2
            discriminant = b**2 - 4*a*c

            if discriminant > 0:
                t = (-b - np.sqrt(discriminant)) / (2*a)
                if 0 < t < max_range:
                    x = t * np.cos(h_angle)
                    y = t * np.sin(h_angle)
                    z = t * np.sin(v_angle)
                    if -0.5 < z < 1.5:
                        points.append([x, y, z])

    return np.array(points)


def demo_lidar_processor():
    """Demonstrate LiDAR processing pipeline."""
    print("LiDAR Processing Demo")
    print("=" * 50)

    # Generate simulated data
    print("Generating simulated LiDAR scan...")
    raw_points = generate_simulated_lidar_scan()
    print(f"Raw points: {len(raw_points)}")

    # Process scan
    processor = LiDARProcessor()
    results = processor.process_scan(raw_points)

    print(f"\nProcessing Results:")
    print(f"  Filtered points: {len(results['filtered_points'])}")
    print(f"  Ground points: {len(results['ground_points'])}")
    print(f"  Obstacle points: {len(results['obstacle_points'])}")
    print(f"  Detected clusters: {len(results['clusters'])}")

    print(f"\nBounding Boxes:")
    for i, bbox in enumerate(results['bounding_boxes']):
        print(f"  Obstacle {i+1}:")
        print(f"    Center: ({bbox['center'][0]:.2f}, {bbox['center'][1]:.2f}, {bbox['center'][2]:.2f})")
        print(f"    Size: ({bbox['dimensions'][0]:.2f}, {bbox['dimensions'][1]:.2f}, {bbox['dimensions'][2]:.2f})")
        print(f"    Points: {bbox['num_points']}")


if __name__ == '__main__':
    demo_lidar_processor()
```

## Point Cloud Processing with Open3D

Open3D provides efficient algorithms for point cloud manipulation and analysis.

```python
#!/usr/bin/env python3
"""
Advanced point cloud processing using Open3D.

Demonstrates filtering, registration, and surface
reconstruction for 3D perception applications.

Prerequisites:
    pip install open3d numpy
"""

import numpy as np
from typing import Tuple, Optional, List

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not installed. Some features unavailable.")


class PointCloudProcessor:
    """
    Advanced point cloud processing using Open3D.

    Provides filtering, registration, segmentation,
    and surface reconstruction capabilities.
    """

    def __init__(self):
        """Initialize point cloud processor."""
        if not OPEN3D_AVAILABLE:
            raise RuntimeError("Open3D required. Install with: pip install open3d")

    @staticmethod
    def from_numpy(points: np.ndarray, colors: Optional[np.ndarray] = None) -> 'o3d.geometry.PointCloud':
        """
        Create Open3D point cloud from numpy array.

        Args:
            points: (N, 3) XYZ coordinates
            colors: Optional (N, 3) RGB colors [0-1]

        Returns:
            Open3D PointCloud object
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            # Normalize colors if needed
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    @staticmethod
    def to_numpy(pcd: 'o3d.geometry.PointCloud') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convert Open3D point cloud to numpy arrays."""
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        return points, colors

    def voxel_downsample(
        self,
        pcd: 'o3d.geometry.PointCloud',
        voxel_size: float = 0.05
    ) -> 'o3d.geometry.PointCloud':
        """
        Downsample point cloud using voxel grid.

        Args:
            pcd: Input point cloud
            voxel_size: Voxel edge length (meters)

        Returns:
            Downsampled point cloud
        """
        return pcd.voxel_down_sample(voxel_size)

    def remove_outliers_statistical(
        self,
        pcd: 'o3d.geometry.PointCloud',
        nb_neighbors: int = 20,
        std_ratio: float = 2.0
    ) -> Tuple['o3d.geometry.PointCloud', np.ndarray]:
        """
        Remove outliers using statistical analysis.

        Args:
            pcd: Input point cloud
            nb_neighbors: Number of neighbors for analysis
            std_ratio: Standard deviation threshold

        Returns:
            (filtered_pcd, inlier_indices)
        """
        filtered, indices = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        return filtered, np.asarray(indices)

    def remove_outliers_radius(
        self,
        pcd: 'o3d.geometry.PointCloud',
        nb_points: int = 16,
        radius: float = 0.1
    ) -> Tuple['o3d.geometry.PointCloud', np.ndarray]:
        """
        Remove outliers using radius search.

        Args:
            pcd: Input point cloud
            nb_points: Minimum neighbors in radius
            radius: Search radius (meters)

        Returns:
            (filtered_pcd, inlier_indices)
        """
        filtered, indices = pcd.remove_radius_outlier(
            nb_points=nb_points,
            radius=radius
        )
        return filtered, np.asarray(indices)

    def estimate_normals(
        self,
        pcd: 'o3d.geometry.PointCloud',
        search_radius: float = 0.1,
        max_nn: int = 30
    ) -> None:
        """
        Estimate surface normals for each point.

        Args:
            pcd: Point cloud (modified in place)
            search_radius: KNN search radius
            max_nn: Maximum neighbors to consider
        """
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=search_radius,
                max_nn=max_nn
            )
        )
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=15)

    def segment_plane(
        self,
        pcd: 'o3d.geometry.PointCloud',
        distance_threshold: float = 0.02,
        ransac_n: int = 3,
        num_iterations: int = 1000
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Segment the dominant plane (e.g., ground, table).

        Args:
            pcd: Input point cloud
            distance_threshold: RANSAC distance threshold
            ransac_n: Points to sample per iteration
            num_iterations: RANSAC iterations

        Returns:
            (plane_model, inlier_indices)
            plane_model: [a, b, c, d] where ax + by + cz + d = 0
        """
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        return np.array(plane_model), list(inliers)

    def cluster_dbscan(
        self,
        pcd: 'o3d.geometry.PointCloud',
        eps: float = 0.05,
        min_points: int = 10
    ) -> List['o3d.geometry.PointCloud']:
        """
        Cluster point cloud using DBSCAN.

        Args:
            pcd: Input point cloud
            eps: Cluster distance threshold
            min_points: Minimum points per cluster

        Returns:
            List of clustered point clouds
        """
        labels = np.array(pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points
        ))

        max_label = labels.max()
        clusters = []

        for i in range(max_label + 1):
            mask = labels == i
            cluster = pcd.select_by_index(np.where(mask)[0])
            clusters.append(cluster)

        return clusters

    def compute_convex_hull(
        self,
        pcd: 'o3d.geometry.PointCloud'
    ) -> 'o3d.geometry.TriangleMesh':
        """
        Compute convex hull of point cloud.

        Args:
            pcd: Input point cloud

        Returns:
            Convex hull as triangle mesh
        """
        hull, _ = pcd.compute_convex_hull()
        return hull

    def surface_reconstruction_poisson(
        self,
        pcd: 'o3d.geometry.PointCloud',
        depth: int = 9
    ) -> 'o3d.geometry.TriangleMesh':
        """
        Reconstruct surface using Poisson reconstruction.

        Requires normals to be computed first.

        Args:
            pcd: Point cloud with normals
            depth: Octree depth (higher = more detail)

        Returns:
            Reconstructed triangle mesh
        """
        if not pcd.has_normals():
            self.estimate_normals(pcd)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth
        )

        # Remove low-density vertices (outlier artifacts)
        densities = np.asarray(densities)
        threshold = np.quantile(densities, 0.01)
        vertices_to_remove = densities < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        return mesh

    def icp_registration(
        self,
        source: 'o3d.geometry.PointCloud',
        target: 'o3d.geometry.PointCloud',
        init_transform: Optional[np.ndarray] = None,
        max_distance: float = 0.1
    ) -> Tuple[np.ndarray, float]:
        """
        Align two point clouds using ICP.

        Args:
            source: Source point cloud (to be transformed)
            target: Target point cloud (reference)
            init_transform: Initial transformation guess
            max_distance: Max correspondence distance

        Returns:
            (transformation_matrix, fitness_score)
        """
        if init_transform is None:
            init_transform = np.eye(4)

        reg = o3d.pipelines.registration.registration_icp(
            source, target, max_distance, init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=50
            )
        )

        return reg.transformation, reg.fitness


def demo_point_cloud_processing():
    """Demonstrate Open3D point cloud processing."""
    if not OPEN3D_AVAILABLE:
        print("Open3D not available")
        return

    print("Point Cloud Processing Demo")
    print("=" * 50)

    processor = PointCloudProcessor()

    # Create sample point cloud (table with objects)
    # Table surface
    table_points = np.random.uniform(
        [-0.5, -0.3, 0],
        [0.5, 0.3, 0.01],
        (5000, 3)
    )

    # Object 1: Box on table
    box_points = np.random.uniform(
        [-0.1, -0.1, 0.01],
        [0.1, 0.1, 0.15],
        (2000, 3)
    )

    # Object 2: Cylinder
    theta = np.random.uniform(0, 2*np.pi, 1500)
    r = 0.05
    cylinder_x = 0.3 + r * np.cos(theta)
    cylinder_y = 0.0 + r * np.sin(theta)
    cylinder_z = np.random.uniform(0.01, 0.2, 1500)
    cylinder_points = np.stack([cylinder_x, cylinder_y, cylinder_z], axis=1)

    # Combine all points
    all_points = np.vstack([table_points, box_points, cylinder_points])

    # Add noise
    all_points += np.random.normal(0, 0.002, all_points.shape)

    print(f"Total points: {len(all_points)}")

    # Create Open3D point cloud
    pcd = processor.from_numpy(all_points)

    # Downsample
    pcd_down = processor.voxel_downsample(pcd, voxel_size=0.01)
    print(f"After downsampling: {len(np.asarray(pcd_down.points))}")

    # Remove outliers
    pcd_filtered, _ = processor.remove_outliers_statistical(pcd_down)
    print(f"After outlier removal: {len(np.asarray(pcd_filtered.points))}")

    # Segment plane (table)
    plane_model, inliers = processor.segment_plane(pcd_filtered)
    print(f"\nPlane detected: {plane_model}")
    print(f"Plane inliers: {len(inliers)}")

    # Extract objects (non-plane points)
    points_np, _ = processor.to_numpy(pcd_filtered)
    outlier_mask = np.ones(len(points_np), dtype=bool)
    outlier_mask[inliers] = False
    object_points = points_np[outlier_mask]

    print(f"Object points: {len(object_points)}")

    # Cluster objects
    object_pcd = processor.from_numpy(object_points)
    clusters = processor.cluster_dbscan(object_pcd, eps=0.05, min_points=50)

    print(f"\nDetected objects: {len(clusters)}")
    for i, cluster in enumerate(clusters):
        cluster_points, _ = processor.to_numpy(cluster)
        bbox = cluster.get_axis_aligned_bounding_box()
        print(f"  Object {i+1}: {len(cluster_points)} points")
        print(f"    Size: {bbox.get_extent()}")

    # Visualize (if display available)
    try:
        # Color the point cloud
        pcd_filtered.paint_uniform_color([0.7, 0.7, 0.7])

        # Color clusters
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
        for i, cluster in enumerate(clusters):
            cluster.paint_uniform_color(colors[i % len(colors)])

        print("\nVisualization window opened (close to continue)")
        o3d.visualization.draw_geometries([pcd_filtered] + clusters)
    except Exception as e:
        print(f"Visualization skipped: {e}")


if __name__ == '__main__':
    demo_point_cloud_processing()
```

## ROS 2 Integration for 3D Perception

```python
#!/usr/bin/env python3
"""
ROS 2 node for 3D perception and depth sensing.

Integrates depth cameras and LiDAR for comprehensive
spatial perception in robotic systems.

Topics:
    Subscribed:
        /camera/depth/image_raw (sensor_msgs/Image)
        /camera/depth/camera_info (sensor_msgs/CameraInfo)
        /velodyne_points (sensor_msgs/PointCloud2)

    Published:
        /perception_3d/obstacles (visualization_msgs/MarkerArray)
        /perception_3d/ground_plane (visualization_msgs/Marker)
        /perception_3d/processed_cloud (sensor_msgs/PointCloud2)

Services:
    /perception_3d/get_obstacle_positions
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA, Header
from cv_bridge import CvBridge
import numpy as np
from typing import List, Optional
import struct


class Perception3DNode(Node):
    """
    Comprehensive 3D perception node for robotics.

    Processes depth camera and LiDAR data to detect
    obstacles, ground plane, and spatial features.
    """

    def __init__(self):
        super().__init__('perception_3d_node')

        # Parameters
        self.declare_parameter('depth_max_range', 5.0)
        self.declare_parameter('ground_height_threshold', 0.1)
        self.declare_parameter('obstacle_min_height', 0.1)
        self.declare_parameter('cluster_tolerance', 0.1)
        self.declare_parameter('min_cluster_size', 50)

        self.max_range = self.get_parameter('depth_max_range').value
        self.ground_threshold = self.get_parameter('ground_height_threshold').value
        self.min_obstacle_height = self.get_parameter('obstacle_min_height').value
        self.cluster_tolerance = self.get_parameter('cluster_tolerance').value
        self.min_cluster_size = self.get_parameter('min_cluster_size').value

        # State
        self.bridge = CvBridge()
        self.camera_info: Optional[CameraInfo] = None
        self.obstacles: List[dict] = []

        # Subscribers
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw',
            self.depth_callback, 10
        )

        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/depth/camera_info',
            self.camera_info_callback, 10
        )

        self.lidar_sub = self.create_subscription(
            PointCloud2, '/velodyne_points',
            self.lidar_callback, 10
        )

        # Publishers
        self.obstacle_marker_pub = self.create_publisher(
            MarkerArray, '/perception_3d/obstacles', 10
        )

        self.ground_marker_pub = self.create_publisher(
            Marker, '/perception_3d/ground_plane', 10
        )

        self.processed_cloud_pub = self.create_publisher(
            PointCloud2, '/perception_3d/processed_cloud', 10
        )

        self.get_logger().info('3D Perception Node initialized')

    def camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsics."""
        self.camera_info = msg

    def depth_callback(self, msg: Image):
        """Process depth image from RGB-D camera."""
        try:
            # Convert depth image
            depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

            # Convert to meters if needed
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0

            # Generate point cloud
            if self.camera_info:
                points = self.depth_to_points(depth)

                # Process for obstacles
                self.process_point_cloud(points, msg.header)

        except Exception as e:
            self.get_logger().error(f'Depth processing error: {e}')

    def depth_to_points(self, depth: np.ndarray) -> np.ndarray:
        """Convert depth image to 3D points."""
        if self.camera_info is None:
            return np.array([])

        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        h, w = depth.shape
        u = np.arange(w)
        v = np.arange(h)
        u, v = np.meshgrid(u, v)

        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack([x, y, z], axis=-1)

        # Filter valid points
        valid = (z > 0.1) & (z < self.max_range)
        return points[valid]

    def lidar_callback(self, msg: PointCloud2):
        """Process LiDAR point cloud."""
        points = self.pointcloud2_to_numpy(msg)
        self.process_point_cloud(points, msg.header)

    def pointcloud2_to_numpy(self, msg: PointCloud2) -> np.ndarray:
        """Convert PointCloud2 message to numpy array."""
        # Find XYZ field offsets
        x_offset = y_offset = z_offset = None

        for field in msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset

        if None in [x_offset, y_offset, z_offset]:
            return np.array([])

        # Parse points
        points = []
        for i in range(0, len(msg.data), msg.point_step):
            x = struct.unpack_from('f', msg.data, i + x_offset)[0]
            y = struct.unpack_from('f', msg.data, i + y_offset)[0]
            z = struct.unpack_from('f', msg.data, i + z_offset)[0]

            if not np.isnan(x) and not np.isinf(x):
                points.append([x, y, z])

        return np.array(points)

    def process_point_cloud(self, points: np.ndarray, header: Header):
        """Process point cloud for obstacle detection."""
        if len(points) == 0:
            return

        # Filter by height (remove ground)
        non_ground_mask = points[:, 2] > self.ground_threshold
        obstacles_points = points[non_ground_mask]

        # Simple clustering (in production, use DBSCAN or similar)
        obstacles = self.simple_cluster(obstacles_points)

        # Publish markers
        self.publish_obstacle_markers(obstacles, header)

        # Publish processed cloud
        self.publish_point_cloud(obstacles_points, header)

    def simple_cluster(self, points: np.ndarray) -> List[dict]:
        """Simple grid-based clustering for obstacles."""
        if len(points) == 0:
            return []

        # Grid-based clustering
        grid_size = self.cluster_tolerance
        grid_points = np.floor(points[:, :2] / grid_size).astype(int)

        # Group by grid cell
        clusters = {}
        for i, cell in enumerate(grid_points):
            key = (cell[0], cell[1])
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(points[i])

        # Convert to obstacle list
        obstacles = []
        for cell, pts in clusters.items():
            pts = np.array(pts)
            if len(pts) >= self.min_cluster_size:
                center = pts.mean(axis=0)
                size = pts.max(axis=0) - pts.min(axis=0)

                if size[2] > self.min_obstacle_height:
                    obstacles.append({
                        'center': center,
                        'size': size,
                        'num_points': len(pts)
                    })

        return obstacles

    def publish_obstacle_markers(self, obstacles: List[dict], header: Header):
        """Publish obstacle bounding box markers."""
        marker_array = MarkerArray()

        for i, obs in enumerate(obstacles):
            marker = Marker()
            marker.header = header
            marker.ns = 'obstacles'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = float(obs['center'][0])
            marker.pose.position.y = float(obs['center'][1])
            marker.pose.position.z = float(obs['center'][2])
            marker.pose.orientation.w = 1.0

            marker.scale.x = float(max(obs['size'][0], 0.1))
            marker.scale.y = float(max(obs['size'][1], 0.1))
            marker.scale.z = float(max(obs['size'][2], 0.1))

            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)
            marker.lifetime.sec = 1

            marker_array.markers.append(marker)

        self.obstacle_marker_pub.publish(marker_array)

    def publish_point_cloud(self, points: np.ndarray, header: Header):
        """Publish processed point cloud."""
        if len(points) == 0:
            return

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * len(points)
        msg.data = points.astype(np.float32).tobytes()
        msg.is_dense = True

        self.processed_cloud_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Perception3DNode()

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

## Spatial Understanding and Scene Reconstruction

```python
#!/usr/bin/env python3
"""
Spatial understanding and scene reconstruction.

Builds volumetric maps and semantic representations
of the robot's environment for navigation and planning.

Prerequisites:
    pip install numpy scipy
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


class VoxelState(Enum):
    """State of a voxel in the occupancy grid."""
    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2


@dataclass
class Voxel:
    """Single voxel with occupancy and semantic info."""
    state: VoxelState = VoxelState.UNKNOWN
    probability: float = 0.5  # Occupancy probability [0, 1]
    observations: int = 0
    semantic_label: Optional[str] = None
    color: Optional[np.ndarray] = None


class OccupancyGrid3D:
    """
    3D occupancy grid for spatial mapping.

    Efficiently represents the robot's environment
    using a volumetric grid with probabilistic updates.
    """

    def __init__(
        self,
        resolution: float = 0.05,
        origin: Tuple[float, float, float] = (-5.0, -5.0, -1.0),
        dimensions: Tuple[int, int, int] = (200, 200, 60)
    ):
        """
        Initialize occupancy grid.

        Args:
            resolution: Voxel size in meters
            origin: World coordinates of grid origin
            dimensions: Grid size (x, y, z) in voxels
        """
        self.resolution = resolution
        self.origin = np.array(origin)
        self.dimensions = dimensions

        # Log-odds representation for efficient updates
        # P(occupied) = 1 / (1 + exp(-log_odds))
        self.log_odds = np.zeros(dimensions, dtype=np.float32)

        # Sensor model parameters
        self.log_odds_hit = 0.7   # Log-odds update for hit
        self.log_odds_miss = -0.4  # Log-odds update for miss
        self.log_odds_max = 3.5
        self.log_odds_min = -3.5

    def world_to_grid(self, point: np.ndarray) -> Tuple[int, int, int]:
        """Convert world coordinates to grid indices."""
        grid_coords = (point - self.origin) / self.resolution
        return tuple(grid_coords.astype(int))

    def grid_to_world(self, indices: Tuple[int, int, int]) -> np.ndarray:
        """Convert grid indices to world coordinates (voxel center)."""
        return self.origin + (np.array(indices) + 0.5) * self.resolution

    def is_valid_index(self, indices: Tuple[int, int, int]) -> bool:
        """Check if grid indices are within bounds."""
        return all(
            0 <= indices[i] < self.dimensions[i]
            for i in range(3)
        )

    def get_occupancy(self, point: np.ndarray) -> float:
        """
        Get occupancy probability at world coordinates.

        Args:
            point: World coordinates

        Returns:
            Probability [0, 1] that the point is occupied
        """
        indices = self.world_to_grid(point)

        if not self.is_valid_index(indices):
            return 0.5  # Unknown

        log_odds = self.log_odds[indices]
        return 1.0 / (1.0 + np.exp(-log_odds))

    def update_ray(
        self,
        origin: np.ndarray,
        endpoint: np.ndarray,
        hit: bool = True
    ):
        """
        Update occupancy along a ray using ray casting.

        Args:
            origin: Ray origin (sensor position)
            endpoint: Ray endpoint (measurement point)
            hit: Whether the endpoint is an obstacle
        """
        # Convert to grid coordinates
        start = self.world_to_grid(origin)
        end = self.world_to_grid(endpoint)

        # Ray cast using 3D Bresenham
        ray_voxels = self._bresenham_3d(start, end)

        # Mark all voxels except last as free
        for voxel in ray_voxels[:-1]:
            if self.is_valid_index(voxel):
                self.log_odds[voxel] = np.clip(
                    self.log_odds[voxel] + self.log_odds_miss,
                    self.log_odds_min,
                    self.log_odds_max
                )

        # Mark endpoint as occupied or free
        if ray_voxels and self.is_valid_index(ray_voxels[-1]):
            update = self.log_odds_hit if hit else self.log_odds_miss
            self.log_odds[ray_voxels[-1]] = np.clip(
                self.log_odds[ray_voxels[-1]] + update,
                self.log_odds_min,
                self.log_odds_max
            )

    def _bresenham_3d(
        self,
        start: Tuple[int, int, int],
        end: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int]]:
        """3D Bresenham line algorithm for ray casting."""
        voxels = []

        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        dz = abs(end[2] - start[2])

        sx = 1 if end[0] > start[0] else -1
        sy = 1 if end[1] > start[1] else -1
        sz = 1 if end[2] > start[2] else -1

        x, y, z = start

        # Determine driving axis
        if dx >= dy and dx >= dz:
            err_y = 2 * dy - dx
            err_z = 2 * dz - dx

            for _ in range(dx + 1):
                voxels.append((x, y, z))
                if err_y > 0:
                    y += sy
                    err_y -= 2 * dx
                if err_z > 0:
                    z += sz
                    err_z -= 2 * dx
                err_y += 2 * dy
                err_z += 2 * dz
                x += sx

        elif dy >= dx and dy >= dz:
            err_x = 2 * dx - dy
            err_z = 2 * dz - dy

            for _ in range(dy + 1):
                voxels.append((x, y, z))
                if err_x > 0:
                    x += sx
                    err_x -= 2 * dy
                if err_z > 0:
                    z += sz
                    err_z -= 2 * dy
                err_x += 2 * dx
                err_z += 2 * dz
                y += sy

        else:
            err_x = 2 * dx - dz
            err_y = 2 * dy - dz

            for _ in range(dz + 1):
                voxels.append((x, y, z))
                if err_x > 0:
                    x += sx
                    err_x -= 2 * dz
                if err_y > 0:
                    y += sy
                    err_y -= 2 * dz
                err_x += 2 * dx
                err_y += 2 * dy
                z += sz

        return voxels

    def integrate_point_cloud(
        self,
        points: np.ndarray,
        sensor_origin: np.ndarray
    ):
        """
        Integrate a point cloud into the occupancy grid.

        Args:
            points: (N, 3) point cloud
            sensor_origin: Sensor position
        """
        for point in points:
            self.update_ray(sensor_origin, point, hit=True)

    def get_occupied_voxels(
        self,
        threshold: float = 0.7
    ) -> np.ndarray:
        """
        Get world coordinates of occupied voxels.

        Args:
            threshold: Occupancy probability threshold

        Returns:
            (N, 3) array of occupied voxel centers
        """
        # Convert log-odds to probability
        prob = 1.0 / (1.0 + np.exp(-self.log_odds))

        # Find occupied indices
        occupied = np.where(prob > threshold)
        indices = np.stack(occupied, axis=1)

        # Convert to world coordinates
        return self.origin + (indices + 0.5) * self.resolution

    def extract_2d_slice(
        self,
        z_range: Tuple[float, float] = (0.0, 2.0)
    ) -> np.ndarray:
        """
        Extract 2D occupancy map from 3D grid.

        Args:
            z_range: Height range to project (meters)

        Returns:
            2D occupancy probability map
        """
        z_min = int((z_range[0] - self.origin[2]) / self.resolution)
        z_max = int((z_range[1] - self.origin[2]) / self.resolution)

        z_min = max(0, z_min)
        z_max = min(self.dimensions[2], z_max)

        # Max occupancy over z range
        slice_log_odds = np.max(self.log_odds[:, :, z_min:z_max], axis=2)
        prob_2d = 1.0 / (1.0 + np.exp(-slice_log_odds))

        return prob_2d


def demo_occupancy_grid():
    """Demonstrate 3D occupancy grid mapping."""
    print("3D Occupancy Grid Demo")
    print("=" * 50)

    # Create grid
    grid = OccupancyGrid3D(
        resolution=0.1,
        origin=(-5.0, -5.0, -1.0),
        dimensions=(100, 100, 30)
    )

    # Simulate sensor at origin
    sensor_origin = np.array([0.0, 0.0, 0.5])

    # Simulate some obstacles
    # Wall in front
    wall_points = []
    for y in np.linspace(-2, 2, 50):
        for z in np.linspace(0, 2, 20):
            wall_points.append([3.0, y, z])

    # Box obstacle
    box_points = []
    for x in np.linspace(1.5, 2.0, 10):
        for y in np.linspace(-0.5, 0.5, 10):
            for z in np.linspace(0, 0.8, 10):
                box_points.append([x, y, z])

    all_points = np.array(wall_points + box_points)
    print(f"Integrating {len(all_points)} points...")

    # Integrate point cloud
    grid.integrate_point_cloud(all_points, sensor_origin)

    # Get statistics
    occupied = grid.get_occupied_voxels(threshold=0.6)
    print(f"Occupied voxels: {len(occupied)}")

    # Extract 2D map
    map_2d = grid.extract_2d_slice(z_range=(0.1, 1.5))
    print(f"2D map shape: {map_2d.shape}")
    print(f"2D map occupancy range: [{map_2d.min():.2f}, {map_2d.max():.2f}]")

    # Visualize (simple ASCII art)
    print("\n2D Occupancy Map (50x50 region):")
    map_vis = map_2d[25:75, 25:75]  # Center region
    for row in map_vis[::2]:  # Subsample for display
        line = ""
        for val in row[::2]:
            if val > 0.7:
                line += "##"
            elif val > 0.3:
                line += ".."
            else:
                line += "  "
        print(line)


if __name__ == '__main__':
    demo_occupancy_grid()
```

## 3D Perception Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Complete 3D Perception System                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              Sensor Layer                                    │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────┤
│   RGB-D Camera  │  Stereo Camera  │     LiDAR       │   Ultrasonic        │
│   (RealSense)   │  (ZED, Intel)   │  (Velodyne)     │   (Short-range)     │
└────────┬────────┴────────┬────────┴────────┬────────┴──────────┬──────────┘
         │                 │                 │                   │
         ▼                 ▼                 ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Point Cloud Generation                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Depth-to-3D projection     • Stereo matching                             │
│  • Sensor calibration         • Time synchronization                        │
│  • Coordinate transforms      • Multi-sensor fusion                         │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Point Cloud Processing                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │  Filtering    │  │  Downsampling │  │   Segmentation│  │  Clustering  │ │
│  │  (Outliers)   │  │  (Voxel Grid) │  │   (RANSAC)    │  │  (DBSCAN)    │ │
│  └───────────────┘  └───────────────┘  └───────────────┘  └──────────────┘ │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Spatial Understanding                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │  Occupancy    │  │   Surface     │  │    Object     │  │   Semantic   │ │
│  │    Grid       │  │ Reconstruction│  │   Detection   │  │   Mapping    │ │
│  └───────────────┘  └───────────────┘  └───────────────┘  └──────────────┘ │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Robot Applications                              │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────┤
│   Navigation    │   Manipulation  │   Localization  │   Scene Analysis    │
│   Path Planning │   Grasp Planning│     SLAM        │   Understanding     │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────┘
```

## Summary

In this chapter, you learned:

- **Depth sensing technologies**: RGB-D cameras, LiDAR, stereo vision and their trade-offs
- **Point cloud generation**: Converting depth data to 3D representations
- **Filtering and preprocessing**: Voxel downsampling, outlier removal, and noise reduction
- **Segmentation**: RANSAC plane fitting and DBSCAN clustering for object detection
- **Spatial mapping**: 3D occupancy grids for navigation and planning
- **ROS 2 integration**: Building perception pipelines with standard interfaces

### Key Takeaways

- **RGB-D cameras** provide dense, colored depth for short-range manipulation
- **LiDAR** excels at long-range, accurate outdoor perception
- **Point cloud filtering** is essential for robust perception
- **Occupancy grids** enable safe navigation and collision avoidance
- **Open3D** provides efficient algorithms for complex 3D processing
- ROS 2 standardizes **sensor interfaces** for reusable perception systems

## Exercises

### Exercise 1: Depth Camera Calibration

1. Set up an Intel RealSense camera
2. Capture depth frames and verify accuracy at known distances
3. Compare depth quality in different lighting conditions

### Exercise 2: Ground Plane Removal

1. Implement RANSAC ground plane detection from LiDAR data
2. Visualize ground vs. obstacle points with different colors
3. Test with tilted sensor orientations

### Exercise 3: Object Detection from Point Clouds

1. Create a scene with multiple objects on a table
2. Segment the table surface and cluster objects
3. Compute bounding boxes and publish as ROS 2 markers

### Challenge: Multi-Sensor Fusion

Build a system that:
1. Fuses RGB-D and LiDAR data into a unified point cloud
2. Handles different frame rates and coordinate systems
3. Maintains temporal consistency for moving objects

## Up Next

In the **Capstone Project**, we'll combine all the concepts from this book to build a complete humanoid robot system with integrated perception, planning, and control.

## Additional Resources

- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense) - RGB-D camera drivers
- [Open3D Documentation](http://www.open3d.org/docs/) - Point cloud processing
- [PCL (Point Cloud Library)](https://pointclouds.org/) - C++ point cloud algorithms
- [Octomap](https://octomap.github.io/) - Efficient 3D mapping
- [ROS 2 Perception Packages](https://github.com/ros-perception) - Pre-built perception nodes

---

**Sources:**
- [Intel RealSense Documentation](https://dev.intelrealsense.com/)
- [Velodyne LiDAR User Guide](https://velodynelidar.com/documentation/)
- [Open3D Tutorials](http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html)
- [RANSAC Algorithm](https://en.wikipedia.org/wiki/Random_sample_consensus)
- [OctoMap: 3D Occupancy Mapping](https://arxiv.org/abs/1010.1202)
