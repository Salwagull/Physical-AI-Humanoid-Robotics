#!/usr/bin/env python3
"""
Isaac Sim Perception Visualization Example

This script demonstrates how to use Isaac Sim's RTX sensors for perception:
- RGB camera
- Depth camera
- RTX LiDAR
- Semantic segmentation

The data is published to ROS 2 for visualization in RViz.

Prerequisites:
- NVIDIA Isaac Sim 4.5+ with RTX GPU
- ROS 2 Humble
- RViz2 for visualization

Usage:
    python perception_viz.py

Then visualize in RViz:
    rviz2
"""

from omni.isaac.kit import SimulationApp

# Launch Isaac Sim with RTX rendering enabled
simulation_app = SimulationApp({
    "headless": False,
    "renderer": "RayTracedLighting",  # Enable RTX
})

import omni
import omni.graph.core as og
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.sensor import Camera
from omni.isaac.range_sensor import LidarRtx
import omni.isaac.core.utils.prims as prim_utils
import numpy as np

# Create world
world = World()
world.scene.add_default_ground_plane()

# Add some objects to perceive
assets_root = get_assets_root_path()

# Add warehouse environment
warehouse_usd = f"{assets_root}/Isaac/Environments/Simple_Warehouse/warehouse.usd"
world.scene.add_reference_to_stage(warehouse_usd, "/World/Warehouse")

print("Creating perception sensors...")

# 1. RGB Camera
rgb_camera = Camera(
    prim_path="/World/Sensors/RGBCamera",
    position=np.array([3.0, 3.0, 2.0]),
    frequency=20,  # 20 Hz
    resolution=(1280, 720),
)
world.scene.add(rgb_camera)
print("✓ RGB camera created at /World/Sensors/RGBCamera")

# 2. Depth Camera
depth_camera = Camera(
    prim_path="/World/Sensors/DepthCamera",
    position=np.array([3.0, -3.0, 2.0]),
    frequency=20,
    resolution=(640, 480),
)
world.scene.add(depth_camera)
print("✓ Depth camera created at /World/Sensors/DepthCamera")

# 3. RTX LiDAR
lidar = LidarRtx(
    prim_path="/World/Sensors/Lidar",
    name="rtx_lidar",
    config={
        "minRange": 0.4,
        "maxRange": 100.0,
        "horizontalFov": 360.0,
        "verticalFov": 30.0,
        "horizontalResolution": 512,
        "verticalResolution": 32,
        "rotationRate": 20.0,
        "drawPoints": False,
        "drawLines": False,
    },
)
# Position LiDAR
prim_utils.set_prim_property(
    lidar.prim_path,
    "xformOp:translate",
    np.array([0.0, 0.0, 2.0])
)
print("✓ RTX LiDAR created at /World/Sensors/Lidar")

print("\nSetting up ROS 2 publishers for sensor data...")

# Create Action Graph for ROS 2 sensor publishing
keys = og.Controller.Keys
graph_path = "/World/PerceptionGraph"

(graph, nodes, _, _) = og.Controller.edit(
    {"graph_path": graph_path, "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            # Trigger
            ("OnTick", "omni.graph.action.OnPlaybackTick"),

            # Clock
            ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),

            # RGB Camera publisher
            ("RGBCameraHelper", "omni.isaac.ros2_bridge.ROS2CameraHelper"),

            # Depth Camera publisher
            ("DepthCameraHelper", "omni.isaac.ros2_bridge.ROS2CameraHelper"),

            # LiDAR publisher
            ("LidarHelper", "omni.isaac.ros2_bridge.ROS2RtxLidarHelper"),

            # TF publisher
            ("PublishTF", "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
        ],
        keys.SET_VALUES: [
            # RGB Camera configuration
            ("RGBCameraHelper.inputs:topicName", "/camera/rgb/image_raw"),
            ("RGBCameraHelper.inputs:frameId", "camera_rgb_frame"),
            ("RGBCameraHelper.inputs:type", "rgb"),
            ("RGBCameraHelper.inputs:cameraPrim", ["/World/Sensors/RGBCamera"]),

            # Depth Camera configuration
            ("DepthCameraHelper.inputs:topicName", "/camera/depth/image_raw"),
            ("DepthCameraHelper.inputs:frameId", "camera_depth_frame"),
            ("DepthCameraHelper.inputs:type", "depth"),
            ("DepthCameraHelper.inputs:cameraPrim", ["/World/Sensors/DepthCamera"]),

            # LiDAR configuration
            ("LidarHelper.inputs:topicName", "/scan"),
            ("LidarHelper.inputs:frameId", "lidar_frame"),
            ("LidarHelper.inputs:lidarPrim", ["/World/Sensors/Lidar"]),
            ("LidarHelper.inputs:publishFullScan", True),
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick", "PublishClock.inputs:execIn"),
            ("OnTick.outputs:tick", "RGBCameraHelper.inputs:execIn"),
            ("OnTick.outputs:tick", "DepthCameraHelper.inputs:execIn"),
            ("OnTick.outputs:tick", "LidarHelper.inputs:execIn"),
            ("OnTick.outputs:tick", "PublishTF.inputs:execIn"),
        ],
    },
)

print("✓ ROS 2 Action Graph created")

print("\n" + "="*60)
print("PERCEPTION VISUALIZATION READY")
print("="*60)
print("\nPublished ROS 2 Topics:")
print("  - /camera/rgb/image_raw (sensor_msgs/Image)")
print("  - /camera/rgb/camera_info (sensor_msgs/CameraInfo)")
print("  - /camera/depth/image_raw (sensor_msgs/Image)")
print("  - /camera/depth/camera_info (sensor_msgs/CameraInfo)")
print("  - /scan (sensor_msgs/LaserScan)")
print("  - /clock (rosgraph_msgs/Clock)")
print("  - /tf (tf2_msgs/TFMessage)")

print("\nVisualize in RViz2:")
print("  1. Run: rviz2")
print("  2. Set Fixed Frame to 'world'")
print("  3. Add → Image → Topic: /camera/rgb/image_raw")
print("  4. Add → Image → Topic: /camera/depth/image_raw")
print("  5. Add → LaserScan → Topic: /scan")

print("\nOr check topics in terminal:")
print("  ros2 topic list")
print("  ros2 topic echo /camera/rgb/image_raw")
print("  ros2 topic hz /scan")

# Reset world
world.reset()

# Main simulation loop
frame = 0
try:
    print("\nSimulation running (press Ctrl+C to stop)...")

    while simulation_app.is_running():
        # Step simulation
        world.step(render=True)

        # Every 100 frames, capture and display sensor info
        if frame % 100 == 0:
            # Get RGB image
            rgb_data = rgb_camera.get_rgba()
            if rgb_data is not None:
                print(f"\n[Frame {frame}]")
                print(f"  RGB Camera: {rgb_data.shape} pixels")

            # Get depth image
            depth_data = depth_camera.get_depth()
            if depth_data is not None:
                print(f"  Depth Camera: min={depth_data.min():.2f}m, max={depth_data.max():.2f}m")

            # Get LiDAR point cloud
            point_cloud = lidar.get_point_cloud()
            if point_cloud is not None and len(point_cloud) > 0:
                print(f"  LiDAR: {len(point_cloud)} points")

        frame += 1

except KeyboardInterrupt:
    print("\n\nShutting down perception visualization...")

# Cleanup
simulation_app.close()
print("Isaac Sim closed")
