---
sidebar_position: 5
---

# Chapter 4: Gazebo for Robot Simulation

## Introduction

Gazebo is the industry-standard robot simulator, used by robotics researchers and developers worldwide. It provides a powerful physics engine, realistic sensor simulation, and seamless integration with ROS 2. Before deploying to real hardware—which is expensive, time-consuming, and sometimes dangerous—you can test your algorithms in Gazebo's virtual environments.

**Learning Objectives:**
- Understand Gazebo's architecture and physics engine
- Learn about robot modeling with SDF and URDF formats
- Create and spawn robots in simulated worlds
- Integrate Gazebo with ROS 2 for robot control
- Simulate sensors (cameras, lidar) and actuators
- Build reproducible testing environments

**Prerequisites:** Chapter 3 (ROS 2 Basics), familiarity with XML, basic physics concepts

**Why This Matters:** Simulation is essential for modern robotics development. Gazebo lets you iterate rapidly, test edge cases safely, and generate synthetic training data—all before touching real hardware. It's the bridge between theory and practice.

## Conceptual Overview

### What is Gazebo?

**Gazebo** is an open-source 3D robot simulator that provides:

- **Physics Simulation**: Realistic dynamics, collisions, friction, and gravity
- **Sensor Simulation**: Cameras, lidar, IMU, GPS, force/torque sensors
- **Actuator Simulation**: Motors, grippers, wheels with realistic response
- **Rendering**: High-quality 3D visualization of robots and environments
- **ROS 2 Integration**: Bidirectional communication with ROS 2 nodes
- **Plugin System**: Extend functionality with custom C++ or Python plugins

**Common Use Cases:**
- Algorithm development and testing
- Reinforcement learning for robot control
- Multi-robot coordination experiments
- Sensor fusion validation
- Hardware-in-the-loop testing

### Gazebo Architecture

Gazebo is built on several core components:

```
┌──────────────────────────────────────────────────────────┐
│                   Gazebo Simulator                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────┐      ┌─────────────────┐           │
│  │  Physics       │      │   Rendering     │           │
│  │  Engine        │      │   Engine        │           │
│  │  (ODE/Bullet/  │      │   (OGRE)        │           │
│  │   Simbody)     │      │                 │           │
│  └────────┬───────┘      └────────┬────────┘           │
│           │                       │                     │
│           └───────────┬───────────┘                     │
│                       │                                 │
│               ┌───────▼────────┐                        │
│               │   World Model  │                        │
│               │   (SDF/URDF)   │                        │
│               └───────┬────────┘                        │
│                       │                                 │
│        ┌──────────────┼──────────────┐                 │
│        │              │              │                 │
│    ┌───▼───┐    ┌────▼────┐    ┌────▼────┐            │
│    │Sensors│    │ Robots  │    │ Objects │            │
│    └───┬───┘    └────┬────┘    └────┬────┘            │
│        │             │              │                  │
│        └─────────────┼──────────────┘                  │
│                      │                                 │
│              ┌───────▼────────┐                        │
│              │   ROS 2 Bridge │                        │
│              │   (ros_gz)     │                        │
│              └───────┬────────┘                        │
└──────────────────────┼─────────────────────────────────┘
                       │
                ┌──────▼──────┐
                │  ROS 2 Nodes│
                └─────────────┘
```

**Key Components:**

1. **Physics Engine**: Simulates real-world physics (gravity, collisions, forces)
2. **Rendering Engine**: Creates 3D visualizations
3. **World Model**: Defines environments, robots, and their properties
4. **Sensors & Actuators**: Simulated hardware components
5. **ROS 2 Bridge**: Connects Gazebo to ROS 2 ecosystem

### Robot Modeling: SDF vs URDF

Robots in Gazebo are described using model files:

**SDF (Simulation Description Format)**
- Gazebo's native format
- Supports advanced features (nested models, multiple robots)
- More expressive than URDF
- XML-based

**URDF (Unified Robot Description Format)**
- Originally from ROS 1
- Widely used in robotics community
- Focuses on kinematics (links and joints)
- Can be converted to SDF

**When to use what:**
- **SDF**: Complex simulations, multiple robots, advanced sensors
- **URDF**: ROS 2 integration, MoveIt motion planning, existing robot descriptions

For this chapter, we'll primarily use SDF for its simplicity and Gazebo-native features.

### Physics Simulation

Gazebo uses physics engines to simulate real-world dynamics:

**Supported Physics Engines:**
1. **ODE (Open Dynamics Engine)** - Default, fast, good for most use cases
2. **Bullet** - Better collision detection, soft body physics
3. **Simbody** - High accuracy, slower, used for biomechanics
4. **DART** - Differentiable physics, good for machine learning

**What Physics Simulates:**
- **Gravity**: Objects fall, robots need to maintain balance
- **Collisions**: Robots can't pass through walls
- **Friction**: Wheels slip on smooth surfaces
- **Inertia**: Heavy objects are harder to accelerate
- **Joint Dynamics**: Motors have torque limits, gears have backlash

**Physics Configuration:**
You can adjust simulation parameters:
- Time step (smaller = more accurate, slower)
- Solver iterations (more = stable, slower)
- Gravity direction and magnitude
- Contact properties (friction coefficients, restitution)

### Sensor Simulation

Gazebo simulates a wide range of sensors:

**Vision Sensors:**
- **RGB Camera**: Color images
- **Depth Camera**: Distance to objects
- **Stereo Camera**: Two cameras for stereo vision
- **360° Camera**: Panoramic views

**Range Sensors:**
- **Lidar**: Laser range finder (2D or 3D)
- **Sonar**: Ultrasonic distance sensor
- **Radar**: Long-range detection

**Motion Sensors:**
- **IMU**: Accelerometer + gyroscope + magnetometer
- **GPS**: Global positioning
- **Odometry**: Wheel encoder-based position estimation

**Contact Sensors:**
- **Bumper**: Detects collisions
- **Force/Torque**: Measures forces on joints

Each sensor publishes data to ROS 2 topics, exactly like real hardware would.

## Technical Implementation

### Installing Gazebo with ROS 2

Gazebo Harmonic (latest) works with ROS 2 Humble:

```bash
# Install Gazebo Harmonic
sudo apt-get update
sudo apt-get install gz-harmonic

# Install ROS 2 - Gazebo bridge
sudo apt-get install ros-humble-ros-gz
```

**Verify installation:**
```bash
gz sim --version
# Should output: Gazebo Sim, version 8.x.x
```

### Creating Your First World

A Gazebo world defines the environment. Let's create a simple world with a ground plane and some obstacles.

**File: `simple_world.sdf`**

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="simple_world">

    <!-- Physics settings -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>

    <!-- Lighting -->
    <light type="directional" name="sun">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.5 0.5 0.5 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Obstacle: Box -->
    <model name="box_obstacle">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

  </world>
</sdf>
```

**Running the world:**
```bash
gz sim simple_world.sdf
```

This opens the Gazebo GUI with a ground plane and a red box obstacle.

**Key SDF Elements:**
- `<world>`: Container for the entire simulation
- `<physics>`: Defines physics engine and time step
- `<light>`: Illumination sources
- `<model>`: Entities in the world (static or dynamic)
- `<link>`: Physical bodies with collision and visual geometry
- `<pose>`: Position (x, y, z) and orientation (roll, pitch, yaw)

### Creating a Simple Mobile Robot

Now let's create a differential drive robot—two wheels and a caster.

**File: `diff_drive_robot.sdf`**

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="diff_drive_robot">

    <!-- Robot base -->
    <link name="base_link">
      <pose>0 0 0.1 0 0 0</pose>

      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <iyy>0.1</iyy>
          <izz>0.1</izz>
        </inertia>
      </inertial>

      <collision name="collision">
        <geometry>
          <box>
            <size>0.5 0.3 0.1</size>
          </box>
        </geometry>
      </collision>

      <visual name="visual">
        <geometry>
          <box>
            <size>0.5 0.3 0.1</size>
          </box>
        </geometry>
        <material>
          <ambient>0 0 1 1</ambient>
          <diffuse>0 0 1 1</diffuse>
        </material>
      </visual>
    </link>

    <!-- Left wheel -->
    <link name="left_wheel">
      <pose>0 0.2 0.1 -1.5707 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <iyy>0.01</iyy>
          <izz>0.01</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.2 1</ambient>
          <diffuse>0.2 0.2 0.2 1</diffuse>
        </material>
      </visual>
    </link>

    <!-- Right wheel -->
    <link name="right_wheel">
      <pose>0 -0.2 0.1 -1.5707 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <iyy>0.01</iyy>
          <izz>0.01</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.2 1</ambient>
          <diffuse>0.2 0.2 0.2 1</diffuse>
        </material>
      </visual>
    </link>

    <!-- Joints: Connect wheels to base -->
    <joint name="left_wheel_joint" type="revolute">
      <parent>base_link</parent>
      <child>left_wheel</child>
      <axis>
        <xyz>0 1 0</xyz>
      </axis>
    </joint>

    <joint name="right_wheel_joint" type="revolute">
      <parent>base_link</parent>
      <child>right_wheel</child>
      <axis>
        <xyz>0 1 0</xyz>
      </axis>
    </joint>

    <!-- Differential drive plugin for ROS 2 control -->
    <plugin
      filename="gz-sim-diff-drive-system"
      name="gz::sim::systems::DiffDrive">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.4</wheel_separation>
      <wheel_radius>0.1</wheel_radius>
      <topic>/cmd_vel</topic>
    </plugin>

  </model>
</sdf>
```

**Key Concepts:**

1. **Links**: Rigid bodies with mass, inertia, collision, and visual geometry
2. **Joints**: Connections between links (revolute = rotating, like wheels)
3. **Inertia**: Mass distribution affects how the robot moves
4. **Plugins**: Add behavior (here, differential drive controller)

**Spawning the robot:**

Create a world file that includes the robot:

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="robot_world">
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>

    <light type="directional" name="sun">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>file://diff_drive_robot.sdf</uri>
      <pose>0 0 0.2 0 0 0</pose>
    </include>
  </world>
</sdf>
```

```bash
gz sim robot_world.sdf
```

### Controlling the Robot with ROS 2

The differential drive plugin subscribes to `/cmd_vel` (Twist messages). Let's control it from ROS 2.

**Test with command-line:**
```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.5}, angular: {z: 0.0}}"
```

The robot should move forward!

**Control with Python:**

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class GazeboRobotController(Node):
    def __init__(self):
        super().__init__('gazebo_robot_controller')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.move_robot)
        self.get_logger().info('Gazebo Robot Controller started!')

    def move_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.5  # Move forward at 0.5 m/s
        cmd.angular.z = 0.2  # Turn slightly
        self.publisher.publish(cmd)

def main():
    rclpy.init()
    node = GazeboRobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Adding a Lidar Sensor

Let's add a lidar to the robot for obstacle detection.

Add this to the robot SDF (inside the `<model>` tag):

```xml
<!-- Lidar sensor -->
<link name="lidar_link">
  <pose>0.2 0 0.2 0 0 0</pose>
  <collision name="collision">
    <geometry>
      <cylinder>
        <radius>0.05</radius>
        <length>0.1</length>
      </cylinder>
    </geometry>
  </collision>
  <visual name="visual">
    <geometry>
      <cylinder>
        <radius>0.05</radius>
        <length>0.1</length>
      </cylinder>
    </geometry>
  </visual>

  <!-- Lidar sensor definition -->
  <sensor name="lidar" type="gpu_lidar">
    <topic>/scan</topic>
    <update_rate>10</update_rate>
    <lidar>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1.0</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>10.0</max>
        <resolution>0.01</resolution>
      </range>
    </lidar>
  </sensor>
</link>

<!-- Joint: Attach lidar to base -->
<joint name="lidar_joint" type="fixed">
  <parent>base_link</parent>
  <child>lidar_link</child>
</joint>
```

**Now the robot publishes lidar data to `/scan`!**

Subscribe to it from ROS 2:
```bash
ros2 topic echo /scan
```

## Practical Example: Autonomous Navigation in Gazebo

Let's combine everything: a robot with lidar, obstacle avoidance, and autonomous movement.

**Scenario:** Robot navigates around obstacles using lidar feedback.

**Python Controller:**

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class AutonomousNavigator(Node):
    def __init__(self):
        super().__init__('autonomous_navigator')

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.safe_distance = 1.0  # meters
        self.get_logger().info('Autonomous Navigator started!')

    def scan_callback(self, msg):
        # Find closest obstacle
        valid_ranges = [r for r in msg.ranges if 0.1 < r < 10.0]
        if not valid_ranges:
            return

        min_distance = min(valid_ranges)
        min_index = msg.ranges.index(min_distance)

        cmd = Twist()

        if min_distance < self.safe_distance:
            # Obstacle ahead! Turn away
            # Determine turn direction based on obstacle position
            if min_index < len(msg.ranges) / 2:
                # Obstacle on the left, turn right
                cmd.angular.z = -0.5
            else:
                # Obstacle on the right, turn left
                cmd.angular.z = 0.5
            cmd.linear.x = 0.1  # Slow down

            self.get_logger().info(
                f'Avoiding obstacle at {min_distance:.2f}m (turning)'
            )
        else:
            # Path clear, move forward
            cmd.linear.x = 0.5
            cmd.angular.z = 0.0
            self.get_logger().info('Path clear, moving forward')

        self.cmd_pub.publish(cmd)

def main():
    rclpy.init()
    node = AutonomousNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Expected Behavior:**
1. Robot moves forward when path is clear
2. When lidar detects obstacle < 1m, robot slows and turns
3. Robot chooses turn direction based on obstacle position
4. Continues navigating autonomously

## Visual Aids

### Gazebo Simulation Pipeline

```
Design Robot    →    Create SDF Model    →    Spawn in Gazebo
(CAD, Sketch)        (XML description)        (Simulation)
                                                    ↓
Test IRL       ←    Transfer Learning    ←    Sim with ROS 2
(Real Robot)         (Sim-to-Real)            (Algorithm Dev)
```

### SDF Model Structure

```
<model>
 ├── <link> (base_link)
 │    ├── <collision> (physics shape)
 │    ├── <visual> (what you see)
 │    ├── <inertial> (mass, inertia)
 │    └── <sensor> (optional)
 │
 ├── <link> (wheel, arm, etc.)
 │    └── ...
 │
 └── <joint> (connects links)
      ├── <parent> link
      ├── <child> link
      └── <axis> (rotation/translation direction)
```

## Summary and Next Steps

**Key Takeaways:**
- Gazebo provides realistic physics and sensor simulation for robots
- Robot models are defined using SDF (or URDF) XML files
- Models consist of links (bodies), joints (connections), and plugins (behaviors)
- Gazebo integrates seamlessly with ROS 2 via topics and services
- Sensors publish data just like real hardware
- Simulation enables rapid iteration and safe testing

**What You've Learned:**
You can now create Gazebo worlds, design robot models with SDF, spawn robots in simulation, control them via ROS 2, and add sensors like lidar. You understand the relationship between physics simulation and real-world robotics.

**Up Next:**
In [Chapter 5: Unity for Robotics Simulation](./chapter5_unity_simulation.md), we'll explore an alternative simulation platform. Unity offers high-quality rendering, VR/AR support, and a massive asset library—ideal for human-robot interaction, synthetic data generation, and visually rich environments.

## Exercises and Challenges

**Exercise 1: Modify the Robot**
Change the differential drive robot to have:
- 4 wheels instead of 2 (add two more wheel links and joints)
- A different color (modify the `<material>` tags)
- A camera sensor (research `<sensor type="camera">`)

**Exercise 2: Build a Custom World**
Create a Gazebo world with:
- Multiple obstacles of different shapes (boxes, cylinders, spheres)
- Varying terrain (add ramps, stairs, or uneven ground)
- Multiple light sources

**Exercise 3: Sensor Fusion**
Add both a camera and lidar to the robot. Write a ROS 2 node that:
- Subscribes to both `/scan` and `/camera/image`
- Logs when obstacles are detected by lidar
- Saves camera images when obstacles are nearby

**Challenge: Wall-Following Robot**
Implement a wall-following algorithm:
- Robot should stay 0.5m from the right wall
- Use lidar data to measure distance to wall
- Implement PD control for smooth following
- Handle corners and gaps in walls

## Further Reading

- [Gazebo Documentation](https://gazebosim.org/docs) - Official Gazebo tutorials and API reference
- [SDF Specification](http://sdformat.org/) - Complete SDF format documentation
- [ros_gz Bridge](https://github.com/gazebosim/ros_gz) - ROS 2 to Gazebo integration
- [Gazebo Models Database](https://app.gazebosim.org/fuel/models) - Community-contributed robot models

---

**Ready to continue?** Chapter 5: Unity for Robotics Simulation will be available soon!
