# Chapter 7: Isaac Sim for Perception and Navigation

## Learning Objectives

By the end of this chapter, you will:

- Configure and use Isaac Sim's RTX sensors (cameras, LiDAR, depth)
- Generate synthetic training datasets with Omniverse Replicator
- Integrate Isaac Sim with ROS 2 Nav2 for autonomous navigation
- Implement Isaac Perceptor for vision-based SLAM
- Build a complete perception and navigation pipeline

## RTX Sensor Simulation

Isaac Sim's **RTX-accelerated sensors** provide physically accurate simulation using ray tracing.

### Camera Sensors

Isaac Sim supports multiple camera types with realistic optics.

#### Creating an RGB Camera

**Method 1: GUI**
1. **Create → Isaac → Sensors → Camera**
2. Position camera at desired location
3. Configure properties in **Property** panel:
   - Resolution: 1280 x 720
   - Focal Length: 24mm
   - F-Stop: 1.8 (depth of field)
   - Focus Distance: 2.0m

**Method 2: Python Script**

```python
from omni.isaac.core import World
from omni.isaac.sensor import Camera
import numpy as np

# Create world
world = World()
world.scene.add_default_ground_plane()

# Create RGB camera
camera = Camera(
    prim_path="/World/Camera",
    position=np.array([2.0, 2.0, 1.5]),
    frequency=30,  # 30 Hz
    resolution=(1280, 720),
)

# Add camera to scene
world.scene.add(camera)
world.reset()

# Capture image
for _ in range(10):
    world.step(render=True)

# Get camera data
rgba = camera.get_rgba()  # Shape: (720, 1280, 4)
rgb = rgba[:, :, :3]

print(f"Captured image shape: {rgb.shape}")
```

#### Stereo Camera Setup

```python
from omni.isaac.sensor import Camera
import numpy as np

# Left camera
camera_left = Camera(
    prim_path="/World/StereoRig/CameraLeft",
    position=np.array([0.0, -0.06, 0.0]),  # 12cm baseline
    frequency=20,
    resolution=(1280, 720),
)

# Right camera
camera_right = Camera(
    prim_path="/World/StereoRig/CameraRight",
    position=np.array([0.0, 0.06, 0.0]),
    frequency=20,
    resolution=(1280, 720),
)

# Compute disparity (simplified)
left_img = camera_left.get_rgba()
right_img = camera_right.get_rgba()

# Use stereo matching algorithm
# (In practice, use OpenCV or Isaac Sim's built-in depth camera)
```

### Depth Cameras

Depth cameras provide direct 3D measurements.

```python
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.types import CameraTypes

# Create depth camera
depth_camera = Camera(
    prim_path="/World/DepthCamera",
    position=np.array([2.0, 0.0, 1.0]),
    frequency=30,
    resolution=(640, 480),
)

world.scene.add(depth_camera)
world.reset()
world.step(render=True)

# Get depth data
depth = depth_camera.get_depth()  # Shape: (480, 640), values in meters
print(f"Min depth: {depth.min():.2f}m, Max depth: {depth.max():.2f}m")

# Get point cloud
points = depth_camera.get_pointcloud()  # Shape: (N, 3)
print(f"Point cloud has {len(points)} points")
```

### LiDAR Sensors

Isaac Sim provides **RTX-accelerated LiDAR** with realistic physics.

#### Rotating LiDAR (e.g., Velodyne VLP-16)

```python
import omni.isaac.core.utils.nucleus as nucleus
from pxr import UsdGeom

# Load LiDAR sensor from USD
lidar_path = "/World/Lidar"
nucleus_server = nucleus.get_assets_root_path()
lidar_usd = f"{nucleus_server}/Isaac/Sensors/Lidar/Rotary/OS1_64ch20hz1024res.usd"

# Add LiDAR to stage
prim = stage.DefinePrim(lidar_path, "Xform")
prim.GetReferences().AddReference(lidar_usd)

# Or use Python API
from omni.isaac.range_sensor import LidarRtx

lidar = LidarRtx(
    prim_path="/World/Lidar",
    config={
        "minRange": 0.4,          # 0.4m minimum range
        "maxRange": 100.0,        # 100m maximum range
        "drawPoints": False,      # Don't visualize points
        "drawLines": False,       # Don't visualize rays
        "horizontalFov": 360.0,   # Full 360° scan
        "verticalFov": 30.0,      # ±15° vertical
        "horizontalResolution": 1024,  # 1024 points/rotation
        "verticalResolution": 64,      # 64 laser channels
        "rotationRate": 20.0,     # 20 Hz rotation
    },
)
```

#### Reading LiDAR Data

```python
# Capture LiDAR scan
world.step(render=True)

# Get point cloud
point_cloud = lidar.get_point_cloud()
# Returns: numpy array of shape (N, 3) with XYZ coordinates

# Optionally get structured data
linear_depth = lidar.get_linear_depth_data()
# Returns: 2D array (vertical_res, horizontal_res)

intensities = lidar.get_intensity_data()
# Returns: Reflectance values

print(f"LiDAR captured {len(point_cloud)} points")
print(f"Point cloud range: [{point_cloud.min():.2f}, {point_cloud.max():.2f}]")
```

### Publishing Sensors to ROS 2

Use **Action Graphs** to publish sensor data to ROS 2.

#### RGB Camera to ROS 2

```python
import omni.graph.core as og

# Create Action Graph for camera publishing
keys = og.Controller.Keys
graph_path = "/World/CameraGraph"

(graph, nodes, _, _) = og.Controller.edit(
    {"graph_path": graph_path, "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnPlaybackTick"),
            ("CameraHelper", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
        ],
        keys.SET_VALUES: [
            ("CameraHelper.inputs:topicName", "/camera/image_raw"),
            ("CameraHelper.inputs:frameId", "camera_link"),
            ("CameraHelper.inputs:type", "rgb"),
            ("CameraHelper.inputs:cameraPrim", [usd.get_stage_next_free_path("/World/Camera", "")]),
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick", "CameraHelper.inputs:execIn"),
        ],
    },
)
```

**Verify in ROS 2:**

```bash
ros2 topic list | grep camera
# /camera/image_raw
# /camera/camera_info

ros2 topic hz /camera/image_raw
# Should show ~30 Hz

# View in RViz
rviz2
# Add → Image → Topic: /camera/image_raw
```

#### LiDAR to ROS 2

```python
keys = og.Controller.Keys
(graph, nodes, _, _) = og.Controller.edit(
    {"graph_path": "/World/LidarGraph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnPlaybackTick"),
            ("LidarHelper", "omni.isaac.ros2_bridge.ROS2RtxLidarHelper"),
        ],
        keys.SET_VALUES: [
            ("LidarHelper.inputs:topicName", "/scan"),
            ("LidarHelper.inputs:frameId", "lidar_link"),
            ("LidarHelper.inputs:lidarPrim", [usd.get_stage_next_free_path("/World/Lidar", "")]),
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick", "LidarHelper.inputs:execIn"),
        ],
    },
)
```

**Verify:**

```bash
ros2 topic echo /scan --once
# Should see LaserScan or PointCloud2 message

# Visualize in RViz
rviz2
# Add → LaserScan → Topic: /scan
# Or Add → PointCloud2 → Topic: /lidar/points
```

## Synthetic Data Generation with Replicator

**Omniverse Replicator** enables automated dataset creation for ML training.

### Domain Randomization Workflow

#### Step 1: Create Scene with Randomizable Elements

```python
import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid

# Create world
world = World()
world.scene.add_default_ground_plane()

# Create multiple objects to randomize
for i in range(10):
    cube = DynamicCuboid(
        prim_path=f"/World/Objects/Cube_{i}",
        position=[i * 0.5 - 2.5, 0, 0.5],
        size=0.3,
        color=[1.0, 0.0, 0.0],
    )
    world.scene.add(cube)

world.reset()
```

#### Step 2: Setup Camera and Render Product

```python
# Create camera
camera = rep.create.camera(position=(5, 5, 3), look_at=(0, 0, 0))

# Create render product (what to capture)
render_product = rep.create.render_product(camera, (1280, 720))
```

#### Step 3: Define Randomization

```python
# Get objects to randomize
objects = rep.get.prims(path_pattern="/World/Objects/Cube_*")

def randomize_scene():
    with objects:
        # Randomize positions
        rep.modify.pose(
            position=rep.distribution.uniform((-3, -3, 0.5), (3, 3, 2.0)),
            rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360)),
        )

        # Randomize scale
        rep.modify.attribute(
            "xformOp:scale",
            rep.distribution.uniform((0.5, 0.5, 0.5), (2.0, 2.0, 2.0))
        )

    # Randomize lighting
    lights = rep.get.light()
    with lights:
        rep.modify.attribute(
            "intensity",
            rep.distribution.uniform(1000, 10000)
        )
        rep.modify.attribute(
            "color",
            rep.distribution.uniform((0.8, 0.8, 0.8), (1.0, 1.0, 1.0))
        )

    return True  # Indicates randomization succeeded
```

#### Step 4: Setup Data Writer

```python
# Initialize writer for annotations
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir="~/isaac_sim_data/cubes",
    rgb=True,                    # Save RGB images
    bounding_box_2d_tight=True,  # Save 2D bounding boxes
    semantic_segmentation=True,  # Save segmentation masks
    distance_to_camera=True,     # Save depth maps
)

# Attach writer to render product
writer.attach([render_product])
```

#### Step 5: Run Randomization Loop

```python
# Register randomization trigger
with rep.trigger.on_frame(num_frames=1000):
    rep.randomizer.register(randomize_scene)

# Run simulation and capture data
rep.orchestrator.run()

# This will generate 1000 images with annotations:
# - rgb/0000.png, rgb/0001.png, ...
# - bounding_box_2d_tight/0000.json, 0001.json, ...
# - semantic_segmentation/0000.png, ...
# - distance_to_camera/0000.npy, ...
```

### Advanced Randomization: Textures and Materials

```python
# Randomize object materials
def randomize_materials():
    materials = rep.get.prims(semantics=[("class", "material")])
    with materials:
        rep.randomizer.materials(
            textures=[
                "~/textures/wood.jpg",
                "~/textures/metal.jpg",
                "~/textures/concrete.jpg",
            ]
        )

with rep.trigger.on_frame():
    rep.randomizer.register(randomize_materials)
```

### Use Case: Training Object Detection

```python
# Generate dataset for YOLOv8 training
import omni.replicator.core as rep

camera = rep.create.camera(position=(3, 3, 2))
render_product = rep.create.render_product(camera, (640, 640))

# Writer for YOLO format
writer = rep.WriterRegistry.get("YOLOWriter")
writer.initialize(
    output_dir="~/datasets/robot_objects_yolo",
    classes=["robot", "obstacle", "target"],
)
writer.attach([render_product])

# Randomize robot and obstacles
with rep.trigger.on_frame(num_frames=10000):
    robots = rep.get.prims(semantics=[("class", "robot")])
    with robots:
        rep.modify.pose(
            position=rep.distribution.uniform((-2, -2, 0), (2, 2, 0)),
            rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
        )

rep.orchestrator.run()
print("Generated 10,000 training images for YOLOv8!")
```

## ROS 2 Navigation with Isaac Sim

Isaac Sim integrates seamlessly with **ROS 2 Nav2** stack.

### Prerequisites

```bash
# Install Nav2
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup

# Install Isaac Sim ROS 2 packages
cd ~/.local/share/ov/pkg/isaac-sim-*/ros2_workspace
source /opt/ros/humble/setup.bash
colcon build --packages-select carter_navigation isaac_ros_navigation_goal
source install/setup.bash
```

### Navigation Setup: Carter Robot

#### Step 1: Launch Isaac Sim with Carter

```bash
# Launch Isaac Sim with hospital environment
~/.local/share/ov/pkg/isaac-sim-*/isaac-sim.sh
```

**In Isaac Sim GUI:**
1. **File → Open** → Select hospital world USD
2. **Create → Isaac → Robots → Carter v2**
3. Position Carter at (0, 0, 0)

#### Step 2: Configure ROS 2 Bridge

Create Action Graph for Nav2 integration:

```python
import omni.graph.core as og

keys = og.Controller.Keys
(graph, nodes, _, _) = og.Controller.edit(
    {"graph_path": "/World/Nav2Graph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnPlaybackTick"),
            ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
            ("PublishTF", "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
            ("SubscribeTwist", "omni.isaac.ros2_bridge.ROS2SubscribeTwist"),
            ("DiffController", "omni.isaac.wheeled_robots.DifferentialController"),
            ("ArticController", "omni.isaac.core_nodes.IsaacArticulationController"),
            ("PublishOdom", "omni.isaac.ros2_bridge.ROS2PublishOdometry"),
            ("PublishLaser", "omni.isaac.ros2_bridge.ROS2RtxLidarHelper"),
        ],
        keys.SET_VALUES: [
            ("SubscribeTwist.inputs:topicName", "/cmd_vel"),
            ("PublishOdom.inputs:topicName", "/odom"),
            ("PublishOdom.inputs:chassisPrim", "/World/Carter"),
            ("PublishLaser.inputs:topicName", "/scan"),
            ("DiffController.inputs:wheelDistance", 0.4132),
            ("DiffController.inputs:wheelRadius", 0.0775),
            ("ArticController.inputs:robotPath", "/World/Carter"),
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick", "PublishClock.inputs:execIn"),
            ("OnTick.outputs:tick", "PublishTF.inputs:execIn"),
            ("OnTick.outputs:tick", "PublishOdom.inputs:execIn"),
            ("OnTick.outputs:tick", "PublishLaser.inputs:execIn"),
            ("SubscribeTwist.outputs:angularVelocity", "DiffController.inputs:angularVelocity"),
            ("SubscribeTwist.outputs:linearVelocity", "DiffController.inputs:linearVelocity"),
            ("DiffController.outputs:velocityCommand", "ArticController.inputs:velocityCommand"),
        ],
    },
)
```

#### Step 3: Launch Nav2

```bash
# Terminal 1: Launch Nav2
source ~/ros2_workspace/install/setup.bash
ros2 launch carter_navigation carter_navigation.launch.py

# This starts:
# - AMCL (localization)
# - Map server
# - Planner server
# - Controller server
# - Behavior server
```

#### Step 4: Set Navigation Goal

**Option 1: RViz**

```bash
# Terminal 2: Launch RViz with Nav2 config
rviz2 -d $(ros2 pkg prefix carter_navigation)/share/carter_navigation/rviz/carter_nav2.rviz

# In RViz:
# - Click "2D Pose Estimate" to set initial pose
# - Click "Nav2 Goal" to set goal location
# - Robot navigates autonomously!
```

**Option 2: Python Script**

```python
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

class GoalPublisher(Node):
    def __init__(self):
        super().__init__('goal_publisher')
        self.publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)

    def send_goal(self, x, y, theta):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.z = theta
        goal.pose.orientation.w = 1.0
        self.publisher.publish(goal)
        self.get_logger().info(f'Sent goal: ({x}, {y}, {theta})')

rclpy.init()
node = GoalPublisher()
node.send_goal(5.0, 3.0, 0.0)  # Navigate to (5, 3)
rclpy.spin_once(node)
node.destroy_node()
rclpy.shutdown()
```

### Creating a Custom Map

#### Step 1: Generate Occupancy Grid

```python
# Use Isaac Sim's occupancy map generator
from omni.isaac.occupancy_map import OccupancyMapGenerator

# Create generator
map_generator = OccupancyMapGenerator(
    cell_size=0.05,          # 5cm resolution
    bounds_min=[-10, -10],   # Map bounds
    bounds_max=[10, 10],
    height_threshold=0.1,    # Consider obstacles above 10cm
)

# Generate map from current scene
occupancy_grid = map_generator.generate()

# Save as ROS map format (.pgm + .yaml)
map_generator.save_map("~/maps/hospital_map")
```

#### Step 2: Use Map in Nav2

```yaml
# hospital_map.yaml
image: hospital_map.pgm
resolution: 0.05
origin: [-10.0, -10.0, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
```

```bash
# Launch Nav2 with custom map
ros2 launch carter_navigation carter_navigation.launch.py \
  map:=$HOME/maps/hospital_map.yaml
```

## Isaac Perceptor Integration

**Isaac Perceptor** is NVIDIA's vision-based SLAM and perception stack.

### What is Isaac Perceptor?

Isaac Perceptor combines:
- **nvblox**: GPU-accelerated 3D reconstruction
- **cuVSLAM**: Visual-inertial SLAM
- **cuMotion**: Trajectory optimization
- **Depth estimation**: AI-based depth from stereo or monocular cameras

### Running Perceptor in Isaac Sim

#### Prerequisites

```bash
# Install Isaac ROS Perceptor
sudo apt install ros-humble-isaac-ros-perceptor

# Or build from source
cd ~/ros2_workspace/src
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_perceptor.git
cd ~/ros2_workspace
colcon build --packages-up-to isaac_perceptor
```

#### Step 1: Setup Nova Carter with Cameras

Nova Carter is NVIDIA's reference robot with **6 stereo cameras** for 360° perception.

**Load in Isaac Sim:**
```python
# Nova Carter has built-in camera rig
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
import omni.isaac.core.utils.nucleus as nucleus

world = World()

# Load Nova Carter robot
nucleus_server = nucleus.get_assets_root_path()
nova_carter_usd = f"{nucleus_server}/Isaac/Robots/Carter/nova_carter_sensors.usd"
world.scene.add_reference_to_stage(nova_carter_usd, "/World/NovaCarter")

world.reset()
```

#### Step 2: Launch Perceptor Stack

```bash
# Terminal 1: Launch Isaac Perceptor
ros2 launch isaac_perceptor isaac_perceptor.launch.py

# This starts:
# - Visual SLAM (cuVSLAM)
# - 3D reconstruction (nvblox)
# - Costmap generation
# - Localization
```

#### Step 3: Visualize in RViz

```bash
# Terminal 2: RViz
rviz2 -d $(ros2 pkg prefix isaac_perceptor)/share/isaac_perceptor/rviz/perceptor.rviz

# You'll see:
# - Live camera feeds (6 stereo pairs)
# - 3D mesh reconstruction (nvblox)
# - Robot trajectory
# - Costmap for navigation
```

#### Step 4: Navigate with Vision

Perceptor provides `/costmap` for Nav2:

```bash
# Use Perceptor costmap instead of LiDAR-based costmap
ros2 launch nav2_bringup navigation_launch.py \
  use_sim_time:=True \
  params_file:=~/isaac_perceptor_nav_params.yaml
```

### Perceptor Features

**1. Real-time 3D Reconstruction (nvblox)**
- Builds voxel-based 3D map from camera depth
- Updates at 10-30 Hz
- GPU-accelerated fusion

**2. Visual SLAM (cuVSLAM)**
- Tracks robot pose using camera features
- No LiDAR required
- Works in GPS-denied environments

**3. Semantic Costmap**
- AI-based obstacle detection
- Classifies: floor, walls, obstacles, dynamic objects
- Safer than purely geometric methods

## Complete Example: Warehouse Navigation

Let's build an end-to-end navigation system.

### Scenario
A mobile robot navigates a warehouse, avoids obstacles, and reaches shelf locations.

### Implementation

```python
# warehouse_navigation.py
import omni.isaac.core.utils.nucleus as nucleus
from omni.isaac.core import World
from omni.isaac.wheeled_robots import WheeledRobot
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np

# Create simulation
world = World()

# Load warehouse environment
nucleus_server = nucleus.get_assets_root_path()
warehouse_usd = f"{nucleus_server}/Isaac/Environments/Simple_Warehouse/warehouse.usd"
world.scene.add_reference_to_stage(warehouse_usd, "/World/Warehouse")

# Add mobile robot
robot = world.scene.add(
    WheeledRobot(
        prim_path="/World/Robot",
        name="warehouse_robot",
        wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
        position=np.array([0, 0, 0.2]),
    )
)

# Add sensors (camera + LiDAR)
# ... (omitted for brevity, see previous camera/LiDAR examples)

world.reset()

# Simple navigation loop
goal_positions = [
    (5.0, 0.0),
    (5.0, 5.0),
    (0.0, 5.0),
    (0.0, 0.0),
]

for goal_x, goal_y in goal_positions:
    print(f"Navigating to ({goal_x}, {goal_y})")

    # In practice, use Nav2 for planning
    # Here: simple proportional controller
    while True:
        robot_pos, _ = robot.get_world_pose()
        dx = goal_x - robot_pos[0]
        dy = goal_y - robot_pos[1]
        distance = np.sqrt(dx**2 + dy**2)

        if distance < 0.5:  # Reached goal
            break

        # Compute velocities
        linear_vel = min(1.0, distance * 0.5)
        angle_to_goal = np.arctan2(dy, dx)
        angular_vel = angle_to_goal * 0.5

        # Apply to robot
        robot.apply_wheel_actions(
            ArticulationAction(joint_velocities=[linear_vel - angular_vel, linear_vel + angular_vel])
        )

        world.step(render=True)

    print(f"Reached ({goal_x}, {goal_y})")

print("Warehouse patrol complete!")
```

## Summary

In this chapter, you learned:

✅ **RTX Sensors**: Configured cameras, depth sensors, and LiDAR with realistic physics

✅ **ROS 2 Publishing**: Published sensor data to ROS 2 topics for external processing

✅ **Synthetic Data**: Used Replicator for domain randomization and dataset generation

✅ **ROS 2 Nav2**: Integrated Isaac Sim with Nav2 for autonomous navigation

✅ **Isaac Perceptor**: Implemented vision-based SLAM and perception

### Key Takeaways

- Isaac Sim's **RTX sensors** provide physically accurate simulations
- **Omniverse Replicator** enables automated synthetic dataset creation
- **Nav2 integration** allows testing navigation algorithms in realistic environments
- **Isaac Perceptor** provides cutting-edge camera-based perception

## Exercises

### Exercise 1: Multi-Sensor Fusion

1. Create a robot with RGB camera, depth camera, and LiDAR
2. Publish all three to ROS 2
3. Use `sensor_msgs::PointCloud2` fusion in ROS 2 to combine data

### Exercise 2: Dataset Generation

1. Create a scene with 20 random objects
2. Use Replicator to randomize poses, scales, and lighting
3. Generate 1000 images with bounding box annotations
4. Train a simple YOLOv8 model on the synthetic data

### Exercise 3: Navigation Challenge

1. Load a complex environment (e.g., hospital or warehouse)
2. Configure Nav2 with Isaac Sim
3. Set 5 waypoints and autonomously navigate between them
4. Handle dynamic obstacles (add moving objects)

### Challenge: Vision-Based Navigation

Use **only cameras** (no LiDAR) with Isaac Perceptor:
1. Setup Nova Carter with 6 stereo cameras
2. Launch Isaac Perceptor for 3D reconstruction
3. Navigate to goal using vision-based costmap
4. Record trajectory and compare with LiDAR-based navigation

## Up Next

In **Chapter 8: Vision-Language-Action (VLA) Systems**, we'll explore:
- Integrating vision models with language understanding
- Using LLMs for high-level task planning
- Voice-commanded robot control
- Multimodal perception and reasoning

## Additional Resources

- [Isaac Sim Sensors Documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/features/sensors_simulation/index.html)
- [Omniverse Replicator Documentation](https://docs.omniverse.nvidia.com/py/replicator/index.html)
- [ROS 2 Nav2 Documentation](https://navigation.ros.org/)
- [Isaac Perceptor Tutorial](https://nvidia-isaac-ros.github.io/reference_workflows/isaac_perceptor/run_perceptor_in_sim.html)
- [Isaac ROS GitHub](https://github.com/NVIDIA-ISAAC-ROS)

---

**Sources:**
- [Isaac Sim ROS 2 Integration - Marvik](https://www.marvik.ai/blog/isaac-sim-integration-with-ros-2)
- [ROS 2 Navigation Tutorial - Isaac Sim Docs](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/ros2_tutorials/tutorial_ros2_navigation.html)
- [Isaac Perceptor in Sim - Isaac ROS Docs](https://nvidia-isaac-ros.github.io/reference_workflows/isaac_perceptor/run_perceptor_in_sim.html)
- [ROS 2 Cameras - Isaac Sim Docs](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/ros2_tutorials/tutorial_ros2_camera.html)
