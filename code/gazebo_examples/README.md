# Gazebo Simulation Examples

This directory contains Gazebo simulation examples from Chapter 4: Gazebo for Robot Simulation.

## Prerequisites

- **Gazebo Harmonic** (or newer) installed
- **ROS 2 Humble** (or newer) installed
- **ros_gz bridge** for ROS 2 - Gazebo integration

### Installation

```bash
# Install Gazebo Harmonic
sudo apt-get update
sudo apt-get install gz-harmonic

# Install ROS 2 - Gazebo bridge
sudo apt-get install ros-humble-ros-gz

# Verify installation
gz sim --version
```

## Files

### 1. `simple_robot.sdf` - Basic Differential Drive Robot

A simple two-wheeled mobile robot with:
- Rectangular base (60cm x 40cm)
- Two driven wheels (10cm radius each)
- One caster wheel for stability
- Differential drive controller plugin
- Odometry publishing

**Run the robot (GUI):**
```bash
gz sim simple_robot.sdf
```

**Control the robot from ROS 2:**

In a separate terminal (source ROS 2 first):
```bash
source /opt/ros/humble/setup.bash

# Move forward
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

# Turn in place
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}"

# Stop
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

### 2. `simple_world.sdf` - Complete Simulation Environment

A Gazebo world including:
- Ground plane
- Lighting (directional sun)
- Obstacles (red box, green cylinder, yellow sphere)
- The differential drive robot

**Run the complete world:**
```bash
gz sim simple_world.sdf
```

**What you'll see:**
- 3D visualization of the world
- Robot spawned at the origin
- Static obstacles around the environment
- Real-time physics simulation

**Interact with the simulation:**
- **Rotate view**: Click and drag with mouse
- **Zoom**: Scroll wheel
- **Pan**: Middle-click and drag
- **Play/Pause**: Use GUI controls
- **Adjust simulation speed**: Real-time factor slider

## Using with ROS 2

### Check Available Topics

While the simulation is running:

```bash
# List all topics
ros2 topic list

# Expected topics:
# /cmd_vel       - Robot velocity commands (input)
# /odom          - Robot odometry (output)
# /clock         - Simulation time
```

### Monitor Odometry

```bash
# See robot position and velocity in real-time
ros2 topic echo /odom
```

### Control with Python

See Chapter 4 examples for Python control scripts. Basic template:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.control_loop)

    def control_loop(self):
        cmd = Twist()
        cmd.linear.x = 0.5  # Forward speed
        cmd.angular.z = 0.0  # Turn rate
        self.publisher.publish(cmd)

def main():
    rclpy.init()
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Model Structure

### SDF Format

Gazebo uses SDF (Simulation Description Format) to define models and worlds.

**Key SDF Elements:**

```xml
<model> - Container for a robot or object
  <link> - A rigid body (has mass and geometry)
    <collision> - Shape for physics (what can't pass through)
    <visual> - Shape for rendering (what you see)
    <inertial> - Mass and inertia properties
    <sensor> - Sensors attached to this link
  </link>
  <joint> - Connection between two links
    <parent> - Parent link name
    <child> - Child link name
    <axis> - Rotation/translation direction
  </joint>
  <plugin> - Behavior (controllers, sensors)
</model>
```

### Coordinate System

Gazebo uses right-handed coordinate system:
- **X**: Forward (red axis)
- **Y**: Left (green axis)
- **Z**: Up (blue axis)

**Pose**: `<pose>x y z roll pitch yaw</pose>`
- Position: meters
- Orientation: radians

## Customization

### Modify Robot Appearance

Edit `simple_robot.sdf` and change `<material>` tags:

```xml
<material>
  <ambient>R G B A</ambient>  <!-- RGB values 0-1, A=alpha -->
  <diffuse>R G B A</diffuse>
</material>
```

### Adjust Physics

Edit `simple_world.sdf`:

```xml
<physics name="1ms" type="ode">
  <real_time_factor>1.0</real_time_factor>  <!-- Speed: 1.0 = real-time -->
  <max_step_size>0.001</max_step_size>      <!-- Accuracy: smaller = better -->
</physics>
```

### Add Sensors

Common sensors to add to the robot:

**Lidar:**
```xml
<sensor name="lidar" type="gpu_lidar">
  <topic>/scan</topic>
  <update_rate>10</update_rate>
  <lidar>
    <scan>
      <horizontal>
        <samples>360</samples>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>10.0</max>
    </range>
  </lidar>
</sensor>
```

**Camera:**
```xml
<sensor name="camera" type="camera">
  <topic>/camera/image</topic>
  <update_rate>30</update_rate>
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
    </image>
  </camera>
</sensor>
```

## Troubleshooting

**Problem:** `gz sim` command not found
- **Solution:** Make sure Gazebo is installed and in your PATH:
  ```bash
  which gz
  sudo apt-get install gz-harmonic
  ```

**Problem:** Robot doesn't move when publishing to `/cmd_vel`
- **Solution:** Check that:
  1. Simulation is running (not paused)
  2. ROS 2 is sourced: `source /opt/ros/humble/setup.bash`
  3. Topic exists: `ros2 topic list | grep cmd_vel`
  4. Differential drive plugin is included in robot SDF

**Problem:** Simulation runs very slowly
- **Solution:** Reduce graphics quality or increase time step:
  - Lower physics accuracy: `<max_step_size>0.01</max_step_size>`
  - Reduce real-time factor: `<real_time_factor>0.5</real_time_factor>`
  - Run headless (no GUI): `gz sim -s simple_world.sdf`

**Problem:** Robot falls through the ground
- **Solution:** Check collision geometry and ensure robot is spawned above ground:
  ```xml
  <pose>0 0 0.15 0 0 0</pose>  <!-- Z=0.15 is above ground -->
  ```

## Next Steps

- Add more complex robot models (arms, grippers)
- Implement autonomous navigation algorithms
- Create custom plugins for sensors or controllers
- Test multi-robot simulations
- Integrate with perception stacks (object detection, SLAM)

## Learning Resources

- [Gazebo Documentation](https://gazebosim.org/docs)
- [SDF Format Specification](http://sdformat.org/)
- [Gazebo Tutorials](https://gazebosim.org/docs/harmonic/tutorials)
- [ros_gz Bridge](https://github.com/gazebosim/ros_gz)
