# Chapter 6: Introduction to NVIDIA Isaac Sim & Machine Learning for Physical AI

## Learning Objectives

By the end of this chapter, you will:

- Understand what NVIDIA Isaac Sim is and how it differs from Gazebo and Unity
- Learn the core architecture and capabilities of Isaac Sim
- Set up Isaac Sim for robotics development
- Explore Isaac Sim's integration with ROS 2
- Create your first robot simulation in Isaac Sim
- **Train robot policies using GPU-accelerated reinforcement learning with Isaac Lab**
- **Generate synthetic datasets for perception model training**
- **Understand sim-to-real transfer techniques for deploying trained policies**

## What is NVIDIA Isaac Sim?

**NVIDIA Isaac Sim™** is an open-source reference framework built on **NVIDIA Omniverse™** that enables developers to simulate and test AI-driven robotics solutions in physically based virtual environments. Released in 2025 as version 5.0, Isaac Sim represents the cutting edge of robotics simulation technology.

### Why Isaac Sim for Physical AI?

While Gazebo excels at physics simulation and Unity provides photorealistic rendering, Isaac Sim combines the best of both worlds with additional AI-native capabilities:

1. **GPU-Accelerated Physics**: Built on NVIDIA PhysX with GPU acceleration for massive parallel simulations
2. **RTX Ray-Traced Rendering**: Photorealistic sensor simulation for cameras and LiDAR
3. **Synthetic Data Generation**: Built-in tools for generating training data for ML models
4. **AI Integration**: Native support for reinforcement learning and imitation learning
5. **ROS 2 Integration**: Seamless connectivity with ROS 2 ecosystems
6. **OpenUSD Standard**: Industry-standard format for 3D scene representation

### Isaac Sim vs. Other Simulators

| Feature | Gazebo | Unity | Isaac Sim |
|---------|--------|-------|-----------|
| **Physics Accuracy** | High | Medium | Very High (GPU) |
| **Visual Fidelity** | Low | Very High | Very High (RTX) |
| **GPU Acceleration** | No | Partial | Full |
| **ROS 2 Integration** | Native | Via bridge | Native |
| **Synthetic Data** | Limited | Manual | Built-in |
| **ML/RL Training** | External | ML-Agents | Isaac Lab |
| **Multi-Robot** | Yes | Yes | Massive Scale |
| **Sensor Simulation** | Good | Good | Excellent (RTX) |
| **License** | Open Source | Proprietary | Open Source |
| **Hardware Required** | CPU | GPU (Optional) | RTX GPU |

**When to use Isaac Sim:**
- Training perception models with synthetic data
- GPU-accelerated reinforcement learning
- Photorealistic camera/LiDAR simulation
- Large-scale multi-robot testing
- Hardware-in-the-loop testing
- Humanoid robot development

## Isaac Sim Architecture

Isaac Sim is built on a layered architecture:

```
┌─────────────────────────────────────────────────────────┐
│           User Applications & Extensions                │
│  (Custom Scripts, Isaac Lab, Replicator, ROS 2 Bridge) │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Isaac Sim Core Framework                    │
│   (Robot APIs, Sensor APIs, Scene Graph, Simulation)    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│             NVIDIA Omniverse Platform                    │
│      (USD Scene Graph, Rendering, Collaboration)        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│          Core Engines (GPU-Accelerated)                  │
│  PhysX (Physics) │ RTX (Rendering) │ cuOpt (Planning)   │
└─────────────────────────────────────────────────────────┘
```

### Key Components

**1. USD (Universal Scene Description)**
- Industry-standard 3D scene representation developed by Pixar
- Enables collaboration and asset sharing
- Supports composition, layering, and variants

**2. PhysX 5**
- GPU-accelerated rigid body dynamics
- Supports articulations (robot joints)
- Collision detection and contact simulation
- Soft body and cloth simulation

**3. RTX Rendering**
- Ray-traced lighting and shadows
- Physically accurate camera simulation
- LiDAR point cloud generation
- Material-based rendering (PBR)

**4. Isaac Sim Extensions**
- **Isaac Core**: Robot APIs, sensors, actuators
- **Isaac Replicator**: Synthetic data generation
- **Isaac Lab**: GPU-accelerated RL framework
- **Isaac ROS Bridge**: ROS 2 connectivity

## Core Features

### 1. Robot Model Support

Isaac Sim supports importing robots from multiple formats:

- **URDF** (Unified Robot Description Format) - ROS standard
- **MJCF** (MuJoCo XML format) - Reinforcement learning
- **USD** (Universal Scene Description) - Native format
- **CAD** (STEP, STL) - Engineering models

**Supported robot types:**
- **Manipulators**: Fanuc, KUKA, Universal Robots, Franka Emika
- **Mobile Robots**: Carter, Jetbot, TurtleBot
- **Quadrupeds**: ANYmal, Spot, Unitree
- **Humanoids**: 1X, Agility Digit, Fourier GR-1, Sanctuary Phoenix

### 2. Sensor Simulation

Isaac Sim provides **physically accurate** sensor models:

**Vision Sensors:**
- RGB cameras with lens distortion
- Depth cameras (stereo, structured light)
- Semantic segmentation
- Instance segmentation
- Bounding box detection

**Range Sensors:**
- RTX-accelerated LiDAR (rotating, solid-state)
- Ultrasonic sensors
- Radar simulation

**Proprioceptive Sensors:**
- IMU (Inertial Measurement Unit)
- Joint encoders
- Force/torque sensors
- Contact sensors

**Example: RTX LiDAR provides:**
- Physically accurate ray tracing
- Material-based reflectance
- Multiple returns per ray
- Beam divergence modeling

### 3. Synthetic Data Generation

One of Isaac Sim's most powerful features is **Omniverse Replicator** for generating training data:

```python
# Example: Generate 10,000 images with domain randomization
import omni.replicator.core as rep

camera = rep.create.camera()
render_product = rep.create.render_product(camera, (1280, 720))

with rep.new_layer():
    # Randomize object poses
    objects = rep.get.prims(path_pattern="/World/Objects/*")
    with objects:
        rep.modify.pose(
            position=rep.distribution.uniform((-1, 0, 0), (1, 0, 2)),
            rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360))
        )

    # Randomize lighting
    lights = rep.get.light()
    with lights:
        rep.modify.attribute("intensity", rep.distribution.uniform(1000, 5000))

    # Randomize textures
    materials = rep.get.prims(semantics=[("class", "material")])
    with materials:
        rep.randomizer.materials()

# Capture 10,000 frames
with rep.trigger.on_frame(num_frames=10000):
    rep.writers.get("BasicWriter").write()
```

**Use cases:**
- Training object detection models
- Generating depth datasets
- Semantic segmentation training
- Sim-to-real transfer learning

### 4. GPU-Accelerated RL with Isaac Lab

**Isaac Lab 2.2** (released 2025) provides a framework for reinforcement learning:

- Train 1000+ robots in parallel on a single GPU
- Built-in environments (locomotion, manipulation, navigation)
- Integration with popular RL libraries (Stable Baselines3, RL Games, SKRL)
- Domain randomization for robust policies

```python
# Example: Train quadruped locomotion
from isaaclab import ManagerBasedRLEnv
from isaaclab.envs import DirectRLEnv

env = gym.make("Isaac-Velocity-Rough-Anymal-C-v0", num_envs=4096)
obs = env.reset()

for _ in range(1000000):
    actions = policy(obs)  # Your RL policy
    obs, rewards, dones, info = env.step(actions)
```

## Installing Isaac Sim

### System Requirements

**Minimum:**
- **OS**: Ubuntu 22.04 or Windows 10/11
- **GPU**: NVIDIA RTX 2060 or higher
- **VRAM**: 8 GB
- **RAM**: 32 GB
- **Storage**: 50 GB SSD

**Recommended:**
- **GPU**: NVIDIA RTX 4090 / A6000 / H100
- **VRAM**: 24 GB+
- **RAM**: 64 GB

### Installation Steps

**Option 1: Omniverse Launcher (Easiest)**

1. Download NVIDIA Omniverse Launcher:
```bash
# Linux
wget https://install.launcher.omniverse.nvidia.com/installers/omniverse-launcher-linux.AppImage
chmod +x omniverse-launcher-linux.AppImage
./omniverse-launcher-linux.AppImage
```

2. In Omniverse Launcher:
   - Navigate to **Exchange** tab
   - Search for "Isaac Sim"
   - Click **Install** (version 4.5 or 5.0)

3. Launch Isaac Sim from the **Library** tab

**Option 2: Docker Container (For CI/CD)**

```bash
# Pull Isaac Sim container
docker pull nvcr.io/nvidia/isaac-sim:4.5.0

# Run container with GPU support
docker run --name isaac-sim --entrypoint bash -it --gpus all \
  -e "ACCEPT_EULA=Y" --rm --network=host \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  nvcr.io/nvidia/isaac-sim:4.5.0
```

**Option 3: pip Install (Python Only)**

```bash
# Install Isaac Sim Python package
pip install isaacsim==4.5.0.0

# Or install from source
git clone https://github.com/isaac-sim/IsaacSim.git
cd IsaacSim
./build.sh --install-release
```

### Verifying Installation

Launch Isaac Sim and verify GPU acceleration:

```bash
# From terminal
~/.local/share/ov/pkg/isaac-sim-*/isaac-sim.sh

# Check GPU usage
nvidia-smi
```

You should see Isaac Sim using GPU memory and compute resources.

## ROS 2 Integration Setup

Isaac Sim provides native ROS 2 support through the **ROS 2 Bridge** extension.

### Install ROS 2 Packages

```bash
# Isaac Sim is compatible with ROS 2 Humble and Jazzy
sudo apt install ros-humble-desktop

# Install Isaac Sim ROS 2 workspace
cd ~/.local/share/ov/pkg/isaac-sim-*/ros2_workspace
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

### Enable ROS 2 Bridge in Isaac Sim

1. **Window → Extensions**
2. Search for **"ROS2 Bridge"**
3. Enable **omni.isaac.ros2_bridge**

### Test ROS 2 Connection

**In Isaac Sim (Python script):**

```python
import omni
from omni.isaac.kit import SimulationApp

# Launch Isaac Sim headless
simulation_app = SimulationApp({"headless": False})

import omni.graph.core as og
from omni.isaac.core import World

# Create world
world = World()
world.scene.add_default_ground_plane()

# Create ROS 2 clock publisher
keys = og.Controller.Keys
og.Controller.edit(
    {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
        ],
        keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
        ],
    },
)

world.reset()
simulation_app.update()

# Run simulation
for _ in range(1000):
    world.step(render=True)

simulation_app.close()
```

**Verify in terminal:**

```bash
source /opt/ros/humble/setup.bash
ros2 topic list
# Should see /clock topic

ros2 topic echo /clock
# Should see simulation time updates
```

## Your First Isaac Sim Robot

Let's create a simple differential drive robot in Isaac Sim.

### Step 1: Create Robot in GUI

1. Launch Isaac Sim
2. **Create → Isaac → Robots → Carter v1**
   - Carter is NVIDIA's reference mobile robot
3. The robot appears in the viewport

### Step 2: Add ROS 2 Control

Create an Action Graph to control the robot via ROS 2:

1. **Window → Visual Scripting → Action Graph**
2. **Create new graph**: `/World/ROS2_Control`
3. Add nodes:
   - **On Playback Tick** (trigger)
   - **ROS2 Subscribe Twist** (subscribe to `/cmd_vel`)
   - **Differential Controller** (convert Twist to wheel commands)
   - **Articulation Controller** (apply to robot wheels)

4. Connect nodes:
```
OnPlaybackTick → ROS2SubscribeTwist → DifferentialController → ArticulationController
```

5. Configure **ROS2SubscribeTwist**:
   - Topic: `/cmd_vel`
   - QoS Profile: `Sensor Data`

6. Configure **ArticulationController**:
   - Target: `/World/Carter`
   - Joint names: `joint_wheel_left`, `joint_wheel_right`

### Step 3: Control Robot from ROS 2

```bash
# In terminal, publish velocity commands
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

The Carter robot should move forward in Isaac Sim!

---

## Machine Learning for Physical AI

One of Isaac Sim's most powerful capabilities is its deep integration with machine learning workflows. This section explores how to leverage Isaac Sim for training intelligent robot behaviors using reinforcement learning, supervised learning from synthetic data, and sim-to-real transfer.

### Why ML + Simulation?

Training robots in the real world is:
- **Expensive**: Hardware wear, human supervision costs
- **Slow**: Physical time constraints, safety protocols
- **Dangerous**: Failed policies can damage hardware or cause injury
- **Limited**: Single robot, single environment at a time

Isaac Sim solves these challenges by enabling:
- **Massive parallelism**: Train 1000+ robots simultaneously on a single GPU
- **Safe exploration**: Robots can fail without consequences
- **Domain randomization**: Expose policies to infinite environment variations
- **Accelerated time**: Run simulation 10-100x faster than real-time

### The Isaac Lab Framework

**Isaac Lab** (formerly Isaac Gym) is NVIDIA's GPU-accelerated reinforcement learning framework built on Isaac Sim. It provides:

```
┌─────────────────────────────────────────────────────────────┐
│                    Isaac Lab Architecture                    │
├─────────────────────────────────────────────────────────────┤
│   Training Frameworks (Stable Baselines3, RL Games, SKRL)   │
├─────────────────────────────────────────────────────────────┤
│           Isaac Lab Manager-Based RL Environments            │
│    (Task definitions, Rewards, Observations, Terminations)   │
├─────────────────────────────────────────────────────────────┤
│              Isaac Sim Simulation Engine                     │
│        (PhysX GPU, Articulations, Sensors, Rendering)        │
├─────────────────────────────────────────────────────────────┤
│                 NVIDIA GPU Hardware                          │
│              (RTX/Tesla for compute + graphics)              │
└─────────────────────────────────────────────────────────────┘
```

**Key Isaac Lab Capabilities:**

1. **Vectorized Environments**: Run thousands of parallel environments
2. **Built-in Tasks**: Locomotion, manipulation, navigation
3. **Domain Randomization**: Physics, visuals, dynamics
4. **Curriculum Learning**: Progressive difficulty scaling
5. **Multi-GPU Training**: Scale across GPU clusters

### Installing Isaac Lab

```bash
# Clone Isaac Lab repository
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Create conda environment
conda create -n isaaclab python=3.10
conda activate isaaclab

# Install Isaac Lab with RL dependencies
pip install -e .[rl_games,sb3,skrl]

# Verify installation
python -c "import isaaclab; print(isaaclab.__version__)"
```

### Reinforcement Learning with Isaac Lab

#### Understanding the RL Training Loop

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│    Policy    │────▶│ Environments  │────▶│   Rewards    │
│  (Neural Net)│     │  (Parallel)   │     │ Observations │
└──────────────┘     └───────────────┘     └──────────────┘
       ▲                                          │
       │              Gradient Update             │
       └──────────────────────────────────────────┘
```

1. **Policy** outputs actions for each parallel environment
2. **Environments** simulate physics and return observations
3. **Rewards** signal what behaviors to learn
4. **Gradients** update policy to maximize cumulative reward

#### Training a Quadruped to Walk

Let's train an ANYmal quadruped robot to walk over rough terrain:

```python
"""
Isaac Lab Quadruped Locomotion Training

This script trains an ANYmal robot to walk using PPO.
Training takes 10-30 minutes on RTX 4090.
"""

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.locomotion.velocity import velocity_env_cfg

# Configure training
env_cfg = velocity_env_cfg.AnymalCRoughEnvCfg()
env_cfg.scene.num_envs = 4096  # Run 4096 robots in parallel!

# Create vectorized environment
env = gym.make("Isaac-Velocity-Rough-Anymal-C-v0", cfg=env_cfg)

print(f"Environment created with {env.num_envs} parallel instances")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Reset all environments
obs, info = env.reset()

# Training loop (simplified - use RL library in practice)
for step in range(10000):
    # Sample random actions (replace with policy inference)
    actions = env.action_space.sample()

    # Step all environments simultaneously
    obs, rewards, terminated, truncated, info = env.step(actions)

    # Log progress
    if step % 1000 == 0:
        mean_reward = rewards.mean().item()
        print(f"Step {step}: Mean reward = {mean_reward:.3f}")

env.close()
```

#### Training with Stable Baselines3

For production training, use an RL library:

```python
"""
Train ANYmal with PPO using Stable Baselines3

Expected training time: 2-4 hours for converged policy
Expected performance: Robot walks at 1+ m/s over rough terrain
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

# Create environment
env = gym.make("Isaac-Velocity-Rough-Anymal-C-v0", num_envs=4096)

# Configure PPO with hyperparameters optimized for locomotion
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=24,  # Steps per environment before update
    batch_size=4096 * 24,  # All data in one batch
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=1.0,
    verbose=1,
    tensorboard_log="./tensorboard_logs/",
)

# Train for 10M timesteps
model.learn(
    total_timesteps=10_000_000,
    callback=EvalCallback(env, eval_freq=10000),
    progress_bar=True,
)

# Save trained policy
model.save("anymal_locomotion_policy")

print("Training complete! Policy saved to anymal_locomotion_policy.zip")
```

#### Reward Function Design

The key to successful RL is reward function design. Isaac Lab provides composable reward terms:

```python
from isaaclab.managers import RewardTermCfg

# Example reward configuration for locomotion
reward_terms = {
    # Primary goal: Track velocity command
    "track_lin_vel_xy_exp": RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,  # Main objective
        params={"std": 0.25},
    ),
    "track_ang_vel_z_exp": RewardTermCfg(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"std": 0.25},
    ),

    # Regularization: Smooth, efficient motion
    "lin_vel_z_l2": RewardTermCfg(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,  # Penalize vertical motion
    ),
    "action_rate_l2": RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.01,  # Penalize jerky actions
    ),
    "joint_torques_l2": RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-0.0001,  # Energy efficiency
    ),

    # Safety: Avoid failures
    "feet_air_time": RewardTermCfg(
        func=mdp.feet_air_time,
        weight=0.125,  # Encourage foot lifting
        params={"threshold": 0.5},
    ),
    "undesired_contacts": RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-1.0,  # Penalize body collisions
    ),
}
```

### Synthetic Data Generation for Perception

Beyond RL, Isaac Sim excels at generating training data for perception models (object detection, segmentation, depth estimation).

#### Omniverse Replicator

**Replicator** is Isaac Sim's synthetic data generation framework:

```python
"""
Generate 10,000 images with domain randomization for object detection training.

Output: RGB images, segmentation masks, bounding boxes in COCO format.
"""

import omni.replicator.core as rep
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

# Create camera
camera = rep.create.camera(position=(3, 3, 2), look_at=(0, 0, 0))
render_product = rep.create.render_product(camera, (1280, 720))

# Setup scene with objects to detect
with rep.new_layer():
    # Add table
    table = rep.create.cube(
        position=(0, 0, 0.5),
        scale=(1, 0.6, 0.05),
        semantics=[("class", "table")],
    )

    # Add objects on table with domain randomization
    objects = rep.create.from_usd(
        usd_files=[
            "omniverse://localhost/NVIDIA/Assets/Isaac/Props/YCB/003_cracker_box.usd",
            "omniverse://localhost/NVIDIA/Assets/Isaac/Props/YCB/005_tomato_soup_can.usd",
            "omniverse://localhost/NVIDIA/Assets/Isaac/Props/YCB/006_mustard_bottle.usd",
        ],
        semantics=[("class", "object")],
        count=5,  # 5 objects per scene
    )

    # Randomize object positions on table
    with objects:
        rep.modify.pose(
            position=rep.distribution.uniform((-0.4, -0.25, 0.55), (0.4, 0.25, 0.55)),
            rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
        )

    # Randomize lighting
    lights = rep.create.light(
        light_type="dome",
        rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360)),
        intensity=rep.distribution.uniform(500, 2000),
    )

    # Randomize camera position
    with camera:
        rep.modify.pose(
            position=rep.distribution.uniform((2, 2, 1.5), (4, 4, 3)),
            look_at=(0, 0, 0.5),
        )

# Configure output writers
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir="./synthetic_data/",
    rgb=True,
    semantic_segmentation=True,
    instance_segmentation=True,
    bounding_box_2d_tight=True,
)
writer.attach([render_product])

# Generate 10,000 frames
print("Generating synthetic data...")
for i in range(10000):
    rep.orchestrator.step()
    if i % 100 == 0:
        print(f"Generated {i} / 10000 images")

print("Synthetic data generation complete!")
print("Data saved to ./synthetic_data/")

simulation_app.close()
```

#### Domain Randomization Strategies

Effective domain randomization is crucial for sim-to-real transfer:

| Randomization Type | Parameters | Purpose |
|-------------------|------------|---------|
| **Lighting** | Intensity, color, position | Handle varied illumination |
| **Textures** | Colors, patterns, materials | Reduce texture bias |
| **Camera** | Position, FOV, distortion | Generalize viewpoints |
| **Objects** | Scale, pose, quantity | Handle object variation |
| **Distractors** | Background objects | Improve robustness |
| **Physics** | Mass, friction, damping | Transfer to real dynamics |

```python
# Example: Comprehensive domain randomization
with rep.trigger.on_frame(num_frames=10000):
    with objects:
        # Pose randomization
        rep.modify.pose(
            position=rep.distribution.uniform((-1, -1, 0), (1, 1, 0.5)),
            rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360)),
            scale=rep.distribution.uniform((0.8, 0.8, 0.8), (1.2, 1.2, 1.2)),
        )

    # Material randomization
    with rep.get.prims(semantics=[("class", "object")]):
        rep.randomizer.materials(
            materials=rep.distribution.choice([
                "omniverse://localhost/NVIDIA/Materials/Base/Wood/Wood_Walnut.mdl",
                "omniverse://localhost/NVIDIA/Materials/Base/Metals/Steel.mdl",
                "omniverse://localhost/NVIDIA/Materials/Base/Plastics/Plastic_ABS.mdl",
            ])
        )

    # Lighting randomization
    with lights:
        rep.modify.attribute("intensity", rep.distribution.uniform(500, 3000))
        rep.modify.attribute("color", rep.distribution.uniform((0.8, 0.8, 0.8), (1.2, 1.2, 1.0)))
```

### Sim-to-Real Transfer

The ultimate goal of simulation-based training is deploying policies to real robots. This section covers techniques for successful sim-to-real transfer.

#### The Sim-to-Real Gap

```
┌─────────────────────────────────────────────────────────────┐
│                   Sim-to-Real Gap Sources                    │
├─────────────────────────────────────────────────────────────┤
│  Dynamics Gap         │  Perception Gap                      │
│  ─────────────        │  ──────────────                      │
│  • Mass/inertia       │  • Lighting conditions               │
│  • Friction coeffs    │  • Sensor noise                      │
│  • Motor response     │  • Camera calibration                │
│  • Contact dynamics   │  • Object appearance                 │
│  • Actuator delays    │  • Background complexity             │
└─────────────────────────────────────────────────────────────┘
```

#### Domain Randomization for Dynamics

Randomize physics parameters during training to create robust policies:

```python
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

@configclass
class RobotDomainRandomization:
    """Domain randomization for sim-to-real transfer."""

    # Randomize robot dynamics
    randomize_robot_mass: bool = True
    mass_range: tuple = (0.8, 1.2)  # ±20% mass variation

    randomize_friction: bool = True
    friction_range: tuple = (0.5, 1.5)  # Ground friction

    randomize_motor_strength: bool = True
    motor_strength_range: tuple = (0.8, 1.2)  # ±20% torque

    randomize_pd_gains: bool = True
    kp_range: tuple = (0.9, 1.1)  # PD controller gains
    kd_range: tuple = (0.9, 1.1)

    # Randomize observations (add noise)
    add_observation_noise: bool = True
    joint_pos_noise: float = 0.01  # radians
    joint_vel_noise: float = 0.15  # rad/s
    imu_noise: float = 0.1  # orientation noise

# Apply during environment reset
def randomize_physics(env, cfg: RobotDomainRandomization):
    """Apply domain randomization to all environments."""
    import torch

    if cfg.randomize_robot_mass:
        mass_scale = torch.empty(env.num_envs).uniform_(*cfg.mass_range)
        env.robot.body_masses *= mass_scale.unsqueeze(-1)

    if cfg.randomize_friction:
        friction = torch.empty(env.num_envs).uniform_(*cfg.friction_range)
        env.terrain.static_friction = friction
        env.terrain.dynamic_friction = friction * 0.8

    if cfg.randomize_motor_strength:
        strength = torch.empty(env.num_envs, env.num_joints).uniform_(*cfg.motor_strength_range)
        env.robot.actuator_effort_limit *= strength
```

#### Real-World Deployment Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│   Train in  │───▶│  Export to   │───▶│  Deploy on  │───▶│   Test &   │
│  Isaac Lab  │    │   ONNX/TRT   │    │ Real Robot  │    │   Iterate  │
└─────────────┘    └──────────────┘    └─────────────┘    └────────────┘
```

**Step 1: Export Trained Policy**

```python
import torch
from stable_baselines3 import PPO

# Load trained model
model = PPO.load("anymal_locomotion_policy")

# Export policy to ONNX for deployment
dummy_obs = torch.zeros(1, model.observation_space.shape[0])
torch.onnx.export(
    model.policy,
    dummy_obs,
    "locomotion_policy.onnx",
    input_names=["observation"],
    output_names=["action"],
    dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
)

print("Policy exported to locomotion_policy.onnx")
```

**Step 2: Deploy on Real Robot (ROS 2)**

```python
#!/usr/bin/env python3
"""
Real robot deployment node using ONNX policy.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import onnxruntime as ort
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class PolicyDeploymentNode(Node):
    def __init__(self):
        super().__init__("policy_deployment")

        # Load ONNX policy
        self.session = ort.InferenceSession("locomotion_policy.onnx")
        self.input_name = self.session.get_inputs()[0].name

        # Subscribe to robot state
        self.state_sub = self.create_subscription(
            JointState, "/joint_states", self.state_callback, 10
        )

        # Publish actions
        self.action_pub = self.create_publisher(
            Float64MultiArray, "/joint_commands", 10
        )

        # Control rate: 50 Hz
        self.timer = self.create_timer(0.02, self.control_loop)
        self.latest_obs = None

    def state_callback(self, msg: JointState):
        # Build observation from sensor data
        self.latest_obs = np.array([
            *msg.position,  # Joint positions
            *msg.velocity,  # Joint velocities
            # Add more sensor data as needed
        ], dtype=np.float32)

    def control_loop(self):
        if self.latest_obs is None:
            return

        # Run policy inference
        obs = self.latest_obs.reshape(1, -1)
        action = self.session.run(None, {self.input_name: obs})[0]

        # Publish action
        msg = Float64MultiArray()
        msg.data = action.flatten().tolist()
        self.action_pub.publish(msg)

def main():
    rclpy.init()
    node = PolicyDeploymentNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
```

### ML Training Best Practices

| Practice | Description | Impact |
|----------|-------------|--------|
| **Start Simple** | Begin with flat terrain, add complexity gradually | Faster convergence |
| **Reward Shaping** | Use dense rewards, avoid sparse signals | More stable training |
| **Curriculum Learning** | Increase task difficulty over time | Learn harder tasks |
| **Regularization** | Penalize energy, jerk, collisions | Smoother real deployment |
| **Seed Variation** | Train with multiple random seeds | More robust policies |
| **Extensive DR** | Randomize everything plausible | Better transfer |
| **Action Smoothing** | Low-pass filter outputs | Reduce actuator stress |
| **Observation Normalization** | Standardize inputs | Stable gradients |

---

## Summary

In this chapter, you learned:

✅ **Isaac Sim** is an open-source robotics simulator built on NVIDIA Omniverse with GPU-accelerated physics and RTX rendering

✅ **Key advantages**: Synthetic data generation, massive parallelism, photorealistic sensors, and native AI/ML integration

✅ **Architecture**: USD scene graph, PhysX physics, RTX rendering, and extensible framework

✅ **Installation**: Via Omniverse Launcher, Docker, or pip

✅ **ROS 2 integration**: Native support via ROS 2 Bridge extension

✅ **First robot**: Created and controlled a Carter robot via ROS 2 `/cmd_vel` topic

✅ **Isaac Lab**: GPU-accelerated reinforcement learning with 1000s of parallel environments

✅ **Synthetic data**: Omniverse Replicator for generating perception training datasets

✅ **Sim-to-real**: Domain randomization and policy export for real robot deployment

### Key Takeaways

- Isaac Sim excels at **AI-first robotics** workflows
- Use for **synthetic data**, **RL training**, and **photorealistic simulation**
- Requires **RTX GPU** for full capabilities
- **Complements** Gazebo (physics) and Unity (game dev) for specific use cases
- **Isaac Lab** enables training policies 10-100x faster than real-time
- **Domain randomization** is essential for policies that transfer to real robots
- The **ML pipeline**: simulate → train → validate → deploy → iterate

## Exercises

### Exercise 1: Explore Isaac Sim Assets

1. Launch Isaac Sim
2. Browse **Create → Isaac → Robots** and spawn 3 different robots
3. Observe differences in joint structure using **Window → Simulation → Stage**

### Exercise 2: ROS 2 Topic Inspection

1. Launch Carter robot with ROS 2 bridge
2. List all published ROS 2 topics: `ros2 topic list`
3. Echo odometry: `ros2 topic echo /odom`
4. Drive the robot in a square pattern using `/cmd_vel`

### Exercise 3: Sensor Visualization

1. Add a camera to Carter: **Create → Isaac → Sensors → Camera**
2. Create Action Graph to publish camera to ROS 2:
   - **ROS2 Publish Camera Info**
   - **ROS2 Publish RGB**
3. View in RViz2:
```bash
rviz2
# Add → By Topic → /rgb → Image
```

### Exercise 4: Isaac Lab Environment Exploration

1. Install Isaac Lab following the instructions above
2. Run the pre-built ANYmal locomotion environment:
```bash
python -m isaaclab.scripts.run_env --task Isaac-Velocity-Flat-Anymal-C-v0 --num_envs 64
```
3. Observe how 64 robots train simultaneously
4. Modify `num_envs` to 256, 1024, 4096 - observe GPU memory and frame rate

### Exercise 5: Synthetic Data Generation

1. Create a simple scene with 5 YCB objects on a table
2. Use Omniverse Replicator to generate 100 images with:
   - Random object positions
   - Random lighting
   - Bounding box annotations
3. Export to COCO format
4. Visualize the bounding boxes on a few sample images

### Challenge: Multi-Robot Simulation

Create 4 Carter robots in Isaac Sim with namespaced ROS 2 topics (`/robot1/cmd_vel`, `/robot2/cmd_vel`, etc.) and control them independently.

**Hint**: Use Action Graph's **namespace** parameter for ROS 2 nodes.

### Challenge: Train a Custom Policy

1. Start with the flat-terrain ANYmal environment
2. Modify the reward function to encourage:
   - Higher forward speed (increase `track_lin_vel_xy` weight)
   - Lower energy consumption (increase `joint_torques_l2` penalty)
3. Train for 5M timesteps
4. Compare your policy's gait to the baseline

## Up Next

In **Chapter 7: Isaac Sim for Perception and Navigation**, we'll dive deeper into:
- Configuring camera, LiDAR, and depth sensors
- Generating synthetic datasets for ML training
- Integrating with ROS 2 Nav2 for autonomous navigation
- Using Isaac Perceptor for vision-based SLAM

## Additional Resources

- [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/)
- [Isaac Sim GitHub Repository](https://github.com/isaac-sim/IsaacSim)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [NVIDIA Isaac Platform Overview](https://developer.nvidia.com/isaac)
- [Omniverse USD Documentation](https://docs.omniverse.nvidia.com/usd/latest/index.html)

---

**Sources:**
- [Isaac Sim - NVIDIA Developer](https://developer.nvidia.com/isaac/sim)
- [Isaac Sim 5.0 Release - NVIDIA Blog](https://developer.nvidia.com/blog/isaac-sim-and-isaac-lab-are-now-available-for-early-developer-preview/)
- [Isaac Sim ROS 2 Integration - Marvik](https://www.marvik.ai/blog/isaac-sim-integration-with-ros-2)
- [ROS 2 Navigation Tutorial - Isaac Sim Docs](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/ros2_tutorials/tutorial_ros2_navigation.html)
