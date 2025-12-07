#!/usr/bin/env python3
"""
Isaac Sim ROS 2 Control Example

This script demonstrates how to control a robot in Isaac Sim via ROS 2 messages.
It subscribes to /cmd_vel and publishes odometry and joint states.

Prerequisites:
- NVIDIA Isaac Sim 4.5+ installed
- ROS 2 Humble installed
- Isaac Sim ROS 2 bridge enabled

Usage:
    python ros2_control.py

Then control the robot from another terminal:
    ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
      "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}"
"""

from omni.isaac.kit import SimulationApp

# Launch Isaac Sim (headless=False shows GUI, True runs without window)
simulation_app = SimulationApp({"headless": False})

import omni
import omni.graph.core as og
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots import WheeledRobot
import numpy as np

# Create simulation world
world = World()
world.scene.add_default_ground_plane()

# Load Carter robot from Isaac Sim assets
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    print("Error: Could not find Isaac Sim assets")
    simulation_app.close()
    exit()

carter_usd = f"{assets_root_path}/Isaac/Robots/Carter/carter_v2.usd"
robot_prim_path = "/World/Carter"

# Add robot to scene
carter = world.scene.add_reference_to_stage(carter_usd, robot_prim_path)

# Create differential drive robot controller
robot = WheeledRobot(
    prim_path=robot_prim_path,
    name="carter",
    wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
    position=np.array([0, 0, 0]),
)
world.scene.add(robot)

print("Setting up ROS 2 Action Graph...")

# Create Action Graph for ROS 2 integration
keys = og.Controller.Keys
graph_path = "/World/ROS2_Control_Graph"

(graph, nodes, _, _) = og.Controller.edit(
    {"graph_path": graph_path, "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            # Simulation tick trigger
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),

            # ROS 2 clock publisher
            ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),

            # ROS 2 TF publisher (robot transforms)
            ("PublishTF", "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),

            # Subscribe to /cmd_vel
            ("SubscribeTwist", "omni.isaac.ros2_bridge.ROS2SubscribeTwist"),

            # Differential drive controller
            ("DiffController", "omni.isaac.wheeled_robots.DifferentialController"),

            # Articulation controller (applies commands to joints)
            ("ArticController", "omni.isaac.core_nodes.IsaacArticulationController"),

            # Publish odometry
            ("PublishOdom", "omni.isaac.ros2_bridge.ROS2PublishOdometry"),
        ],
        keys.SET_VALUES: [
            # Configure /cmd_vel subscriber
            ("SubscribeTwist.inputs:topicName", "/cmd_vel"),
            ("SubscribeTwist.inputs:qosProfile", "Sensor Data"),

            # Configure odometry publisher
            ("PublishOdom.inputs:topicName", "/odom"),
            ("PublishOdom.inputs:chassisPrim", robot_prim_path),
            ("PublishOdom.inputs:frameId", "odom"),
            ("PublishOdom.inputs:childFrameId", "base_link"),

            # Configure differential controller
            # Carter wheel parameters
            ("DiffController.inputs:wheelDistance", 0.4132),  # meters between wheels
            ("DiffController.inputs:wheelRadius", 0.0775),    # wheel radius in meters
            ("DiffController.inputs:maxLinearSpeed", 2.0),    # m/s
            ("DiffController.inputs:maxAngularSpeed", 3.14),  # rad/s

            # Configure articulation controller
            ("ArticController.inputs:robotPath", robot_prim_path),
            ("ArticController.inputs:jointNames", ["joint_wheel_left", "joint_wheel_right"]),
        ],
        keys.CONNECT: [
            # Connect tick to publishers
            ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "PublishOdom.inputs:execIn"),

            # Connect /cmd_vel to differential controller
            ("SubscribeTwist.outputs:angularVelocity", "DiffController.inputs:angularVelocity"),
            ("SubscribeTwist.outputs:linearVelocity", "DiffController.inputs:linearVelocity"),

            # Connect differential controller to articulation controller
            ("DiffController.outputs:velocityCommand", "ArticController.inputs:velocityCommand"),
        ],
    },
)

print("ROS 2 Action Graph created successfully")
print("\nStarting simulation...")
print("\nControl the robot using ROS 2:")
print("  ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \\")
print('    "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"')
print("\nAvailable ROS 2 topics:")
print("  - /cmd_vel (subscribe) - Twist commands")
print("  - /odom (publish) - Odometry")
print("  - /clock (publish) - Simulation clock")
print("  - /tf (publish) - Transform tree")

# Reset world to initialize physics
world.reset()

# Main simulation loop
frame = 0
try:
    while simulation_app.is_running():
        # Step physics and rendering
        world.step(render=True)

        # Print status every 100 frames
        if frame % 100 == 0:
            position, orientation = robot.get_world_pose()
            print(f"Frame {frame}: Robot at position {position}")

        frame += 1

except KeyboardInterrupt:
    print("\nShutting down...")

# Cleanup
simulation_app.close()
print("Isaac Sim closed")
