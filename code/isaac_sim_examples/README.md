># Isaac Sim Examples

This directory contains Python examples demonstrating NVIDIA Isaac Sim capabilities for robotics simulation, perception, ROS 2 integration, and **machine learning for Physical AI**.

## Prerequisites

### Software Requirements

1. **NVIDIA Isaac Sim 4.5 or 5.0**
   - Download from [NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/)
   - Or install via Docker: `docker pull nvcr.io/nvidia/isaac-sim:4.5.0`

2. **ROS 2 Humble or Jazzy**
   ```bash
   sudo apt update
   sudo apt install ros-humble-desktop
   source /opt/ros/humble/setup.bash
   ```

3. **Isaac Sim ROS 2 Workspace**
   ```bash
   cd ~/.local/share/ov/pkg/isaac-sim-*/ros2_workspace
   colcon build
   source install/setup.bash
   ```

4. **Isaac Lab (for RL training)**
   ```bash
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab
   conda create -n isaaclab python=3.10
   conda activate isaaclab
   pip install -e .[rl_games,sb3,skrl]
   ```

5. **Additional Python Packages (for ML examples)**
   ```bash
   pip install stable-baselines3 onnx onnxruntime torch tensorboard
   ```

### Hardware Requirements

- **GPU**: NVIDIA RTX 2060 or higher (RTX 4090/A6000 recommended)
- **VRAM**: 8 GB minimum (24 GB+ recommended for RL training)
- **RAM**: 32 GB minimum (64 GB recommended)
- **OS**: Ubuntu 22.04 or Windows 10/11

## Files

### 1. `ros2_control.py` - Robot Control via ROS 2

Controls a Carter mobile robot in Isaac Sim using ROS 2 `/cmd_vel` messages.

**Features:**
- Subscribes to `/cmd_vel` (Twist messages)
- Publishes odometry to `/odom`
- Publishes TF transforms
- Publishes simulation clock

**Usage:**

```bash
# Terminal 1: Run Isaac Sim simulation
python ros2_control.py

# Terminal 2: Control robot
source /opt/ros/humble/setup.bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}"

# View odometry
ros2 topic echo /odom

# List all topics
ros2 topic list
```

**Expected Output:**
- Robot moves forward at 1 m/s and rotates at 0.5 rad/s
- Odometry published at simulation rate
- TF tree shows robot transforms

### 2. `perception_viz.py` - Sensor Visualization

Demonstrates RTX-accelerated sensors: RGB camera, depth camera, and LiDAR.

**Features:**
- RGB camera (1280x720 @ 20Hz)
- Depth camera (640x480 @ 20Hz)
- RTX LiDAR (512x32 resolution, 360° FOV)
- Publishes all sensor data to ROS 2
- Visualizable in RViz2

**Usage:**

```bash
# Terminal 1: Run perception simulation
python perception_viz.py

# Terminal 2: Visualize in RViz2
rviz2
```

**RViz2 Configuration:**
1. Set **Fixed Frame** to `world`
2. **Add → Image**
   - Topic: `/camera/rgb/image_raw`
3. **Add → Image** (in new display)
   - Topic: `/camera/depth/image_raw`
4. **Add → LaserScan**
   - Topic: `/scan`
5. **Add → TF** to see sensor frames

**Expected Output:**
- RGB image shows warehouse environment
- Depth image shows distance measurements
- LiDAR shows 360° point cloud

### Inspection Commands

```bash
# Check camera images
ros2 topic hz /camera/rgb/image_raw
# Expected: ~20 Hz

# Echo depth data
ros2 topic echo /camera/depth/image_raw --once

# Check LiDAR points
ros2 topic echo /scan

# View camera info
ros2 topic echo /camera/rgb/camera_info --once
```

## Common Isaac Sim Code Patterns

### 1. Basic Simulation Setup

```python
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
world = World()
world.scene.add_default_ground_plane()

# Add objects, sensors, robots...

world.reset()

# Simulation loop
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
```

### 2. Adding a Robot

```python
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots import WheeledRobot
import numpy as np

assets_root = get_assets_root_path()
robot_usd = f"{assets_root}/Isaac/Robots/Carter/carter_v2.usd"

world.scene.add_reference_to_stage(robot_usd, "/World/Robot")

robot = WheeledRobot(
    prim_path="/World/Robot",
    name="carter",
    wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
    position=np.array([0, 0, 0]),
)
world.scene.add(robot)
```

### 3. Creating ROS 2 Action Graph

```python
import omni.graph.core as og

keys = og.Controller.Keys
(graph, nodes, _, _) = og.Controller.edit(
    {"graph_path": "/World/MyGraph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnPlaybackTick"),
            ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick", "PublishClock.inputs:execIn"),
        ],
    },
)
```

### 4. Adding a Camera

```python
from omni.isaac.sensor import Camera
import numpy as np

camera = Camera(
    prim_path="/World/Camera",
    position=np.array([2.0, 2.0, 1.5]),
    frequency=30,
    resolution=(1280, 720),
)
world.scene.add(camera)

# Capture image
world.step(render=True)
rgba = camera.get_rgba()  # Shape: (720, 1280, 4)
```

### 5. Adding RTX LiDAR

```python
from omni.isaac.range_sensor import LidarRtx

lidar = LidarRtx(
    prim_path="/World/Lidar",
    config={
        "minRange": 0.4,
        "maxRange": 100.0,
        "horizontalFov": 360.0,
        "verticalFov": 30.0,
        "horizontalResolution": 1024,
        "verticalResolution": 64,
        "rotationRate": 20.0,
    },
)

# Get point cloud
world.step(render=True)
points = lidar.get_point_cloud()  # Shape: (N, 3)
```

## Troubleshooting

### Problem: "Could not find Isaac Sim assets"

**Solution:**
- Verify Isaac Sim is installed correctly
- Check `~/.local/share/ov/pkg/isaac-sim-*/` exists
- Run Isaac Sim once from Omniverse Launcher to download assets

### Problem: "ROS 2 topics not appearing"

**Solution:**
1. Enable ROS 2 Bridge extension:
   - Window → Extensions → Search "ROS2 Bridge" → Enable
2. Source ROS 2 workspace:
   ```bash
   source /opt/ros/humble/setup.bash
   source ~/.local/share/ov/pkg/isaac-sim-*/ros2_workspace/install/setup.bash
   ```
3. Check domain ID matches:
   ```bash
   echo $ROS_DOMAIN_ID  # Should be same in both terminals
   ```

### Problem: "GPU out of memory"

**Solution:**
- Reduce camera resolution: `resolution=(640, 480)`
- Lower LiDAR resolution: `horizontalResolution=256, verticalResolution=16`
- Enable headless mode: `SimulationApp({"headless": True})`
- Close other GPU applications

### Problem: "Simulation running slowly"

**Solution:**
- Reduce sensor frequencies: `frequency=10` (10 Hz instead of 30 Hz)
- Disable real-time mode: `world.set_simulation_dt(0.01, use_fabric=True)`
- Use GPU acceleration: Ensure RTX GPU drivers are up-to-date
- Reduce scene complexity

### Problem: Robot doesn't move with `/cmd_vel`

**Solution:**
1. Check topic subscription:
   ```bash
   ros2 topic info /cmd_vel
   # Should show subscriber: /isaac_sim
   ```
2. Verify Action Graph connections
3. Check wheel joint names match robot URDF
4. Ensure `world.reset()` was called before simulation

---

## Machine Learning Examples

### 3. `rl_locomotion_training.py` - Reinforcement Learning Training

Train a quadruped robot (ANYmal) to walk using GPU-accelerated RL with Isaac Lab.

**Features:**
- PPO training with Stable Baselines3
- Parallel simulation of 1000s of robots
- Configurable terrain (flat/rough)
- TensorBoard logging
- Checkpoint saving/resuming

**Usage:**

```bash
# Basic training (2048 parallel environments)
python rl_locomotion_training.py

# Train with more environments (faster, needs more VRAM)
python rl_locomotion_training.py --num_envs 4096

# Headless training (faster)
python rl_locomotion_training.py --headless --timesteps 10000000

# Resume from checkpoint
python rl_locomotion_training.py --checkpoint ./checkpoints/model_1000000_steps.zip
```

**Expected Output:**
- Training logs in TensorBoard (`./tensorboard_logs/`)
- Checkpoints saved every 500k steps
- Final model: `anymal_locomotion_policy.zip`

**Training Time (RTX 4090):**
| Timesteps | Time | Expected Performance |
|-----------|------|---------------------|
| 1M | ~10 min | Basic walking |
| 5M | ~45 min | Stable locomotion |
| 10M | ~90 min | Robust traversal |

### 4. `synthetic_data_generation.py` - Dataset Generation for Perception

Generate synthetic training data with domain randomization for object detection models.

**Features:**
- RGB, depth, and segmentation images
- 2D bounding box annotations
- Domain randomization (lighting, poses, textures)
- COCO-format export

**Usage:**

```bash
# Generate 1000 images
python synthetic_data_generation.py --num_frames 1000

# Generate headless (faster)
python synthetic_data_generation.py --headless --num_frames 10000

# Custom output directory and resolution
python synthetic_data_generation.py --output_dir ./my_dataset --resolution 1920 1080
```

**Output Structure:**
```
./synthetic_data/
├── rgb/                    # RGB images
├── depth/                  # Depth images
├── semantic_segmentation/  # Semantic masks
├── instance_segmentation/  # Instance masks
├── bounding_box_2d_tight/  # Bounding boxes (JSON)
└── annotations_coco.json   # COCO-format annotations
```

### 5. `export_policy_onnx.py` - Export for Real Robot Deployment

Export trained policies to ONNX format for deployment on real robots.

**Features:**
- ONNX export with validation
- Inference benchmarking
- Optional TensorRT conversion
- ROS 2 deployment example generation

**Usage:**

```bash
# Export trained model to ONNX
python export_policy_onnx.py --model anymal_locomotion_policy.zip

# Export with TensorRT optimization
python export_policy_onnx.py --tensorrt

# Validate output accuracy
python export_policy_onnx.py --validate
```

**Output:**
- `locomotion_policy.onnx` - ONNX model
- `deployment_example_ros2.py` - ROS 2 deployment template
- (Optional) `locomotion_policy.trt` - TensorRT engine

---

## ML Training Workflow

The complete workflow for training robot policies:

```
1. Define Task → 2. Train in Sim → 3. Validate → 4. Export → 5. Deploy
      ↓                ↓              ↓           ↓           ↓
   Reward         Isaac Lab        Test in      ONNX/TRT    Real
   Design         + SB3/PPO         Sim                    Robot
```

### Quick Start: Train and Deploy a Walking Robot

```bash
# Step 1: Train policy (takes ~45 min on RTX 4090)
python rl_locomotion_training.py --headless --timesteps 5000000

# Step 2: Monitor training
tensorboard --logdir ./tensorboard_logs/

# Step 3: Export to ONNX
python export_policy_onnx.py --model anymal_locomotion_policy.zip

# Step 4: Deploy on robot (adapt deployment_example_ros2.py)
ros2 run my_robot_pkg locomotion_node
```

---

## Advanced Examples (Not Included)

To extend these examples, consider implementing:

### 1. Synthetic Data Generation

```python
import omni.replicator.core as rep

camera = rep.create.camera()
render_product = rep.create.render_product(camera, (1280, 720))

# Domain randomization
with rep.trigger.on_frame(num_frames=1000):
    objects = rep.get.prims(path_pattern="/World/Objects/*")
    with objects:
        rep.modify.pose(
            position=rep.distribution.uniform((-2, -2, 0), (2, 2, 2))
        )

rep.orchestrator.run()
```

### 2. Navigation with Nav2

```python
# Launch Nav2 stack
# Terminal 1: Isaac Sim with Carter
python ros2_control.py

# Terminal 2: Nav2
ros2 launch carter_navigation carter_navigation.launch.py

# Terminal 3: Set goal
ros2 topic pub /goal_pose geometry_msgs/PoseStamped \
  "{header: {frame_id: 'map'}, pose: {position: {x: 5.0, y: 3.0, z: 0.0}}}"
```

### 3. Multi-Robot Simulation

```python
# Spawn multiple robots with namespaces
for i in range(4):
    robot = WheeledRobot(
        prim_path=f"/World/Robot_{i}",
        name=f"carter_{i}",
        position=np.array([i * 2.0, 0, 0]),
    )
    # Setup ROS 2 with namespace /robot_0, /robot_1, etc.
```

### 4. Reinforcement Learning with Isaac Lab

```python
import gymnasium as gym
from isaaclab import ManagerBasedRLEnv

env = gym.make("Isaac-Velocity-Rough-Anymal-C-v0", num_envs=1024)
obs = env.reset()

for _ in range(10000):
    actions = policy(obs)
    obs, rewards, dones, info = env.step(actions)
```

## Next Steps

After mastering these examples:

1. **Chapter 8**: Integrate VLA (Vision-Language-Action) systems
2. **Chapter 9**: Add voice commands with LLM planning
3. **Chapter 10**: Build capstone humanoid robot project

## Resources

- [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/)
- [Isaac Sim Python API Reference](https://docs.isaacsim.omniverse.nvidia.com/latest/api.html)
- [ROS 2 Tutorials for Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/index.html)
- [Isaac Sim GitHub](https://github.com/isaac-sim/IsaacSim)
- [NVIDIA Omniverse Forums](https://forums.developer.nvidia.com/c/omniverse/)

## License

These examples are provided as educational material for the Physical AI & Humanoid Robotics book.
Isaac Sim is licensed by NVIDIA. See [NVIDIA Omniverse License](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html).
