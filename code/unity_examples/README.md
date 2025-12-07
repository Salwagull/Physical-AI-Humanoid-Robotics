# Unity Robotics Examples

This directory contains Unity C# scripts for robot simulation examples from Chapter 5: Unity for Robotics Simulation.

## Prerequisites

- **Unity Hub** 2021.3 LTS or newer
- **Unity Robotics Hub** packages installed
- **ROS 2 Humble** (or newer) installed
- **ROS TCP Endpoint** package

### Installation

**1. Install Unity:**
- Download Unity Hub from https://unity.com/
- Install Unity Editor 2021.3 LTS

**2. Install Unity Robotics Hub:**

In Unity Editor:
- Window → Package Manager
- Click "+" → Add package from git URL
- Add: `https://github.com/Unity-Technologies/Unity-Robotics-Hub.git?path=/com.unity.robotics.ros-tcp-connector`
- Also add: URDF Importer package

**3. Install ROS 2 Endpoint:**
```bash
sudo apt-get install ros-humble-ros-tcp-endpoint

# Or via pip:
pip3 install roslibpy
```

## Files

### `RobotController.cs` - Robot Movement Controller

Controls a robot in Unity by subscribing to ROS 2 `/cmd_vel` topic.

**Features:**
- Subscribes to `/cmd_vel` (Twist messages)
- Applies linear and angular velocities to robot
- Configurable speed parameters
- Proper ROS ↔ Unity coordinate conversion

**Usage:**

1. Create a Unity project (3D template)
2. Create a robot GameObject (or import URDF)
3. Add Rigidbody component to robot
4. Attach `RobotController.cs` script
5. Configure ROS TCP Connector:
   - Robotics → ROS Settings
   - ROS IP: `127.0.0.1`
   - Port: `10000`
   - Protocol: ROS 2

6. Start ROS endpoint:
```bash
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0
```

7. Press Play in Unity

8. Control from ROS 2:
```bash
# Move forward
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

# Turn
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}"

# Stop
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

## Unity-ROS Coordinate System Conversion

**ROS Convention:**
- X: Forward
- Y: Left
- Z: Up

**Unity Convention:**
- X: Right
- Y: Up
- Z: Forward

**Conversion in Code:**
```csharp
// ROS (x,y,z) → Unity (x,y,z)
Vector3 unityPos = new Vector3(-rosMsg.y, rosMsg.z, rosMsg.x);

// For rotations around vertical axis:
float unityRotation = -rosMsg.angular.z; // Negative for convention
```

## Project Structure

Recommended Unity project structure for robotics:

```
Assets/
├── Scripts/
│   ├── RobotController.cs
│   ├── CameraPublisher.cs (Chapter 5)
│   └── SensorSimulator.cs
├── Robots/
│   └── imported_urdf/  (from URDF importer)
├── Scenes/
│   └── RoboticsLab.unity
└── ROS/
    └── ROSSettings.asset
```

## Additional Examples (Not Included)

To extend your Unity robotics project, consider adding:

**1. Camera Publisher** - Publish camera images to ROS
```csharp
// Pseudo-code
void PublishCameraImage() {
    Texture2D image = CaptureCamera();
    ImageMsg msg = ConvertToROSImage(image);
    ros.Publish("/camera/image_raw", msg);
}
```

**2. Lidar Simulator** - Raycast-based lidar simulation
```csharp
// Pseudo-code
void SimulateLidar() {
    LaserScanMsg scan = new LaserScanMsg();
    for (int i = 0; i < 360; i++) {
        RaycastHit hit;
        float angle = i * Mathf.Deg2Rad;
        Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
        if (Physics.Raycast(transform.position, direction, out hit, maxRange)) {
            scan.ranges[i] = hit.distance;
        }
    }
    ros.Publish("/scan", scan);
}
```

**3. Joint State Publisher** - Publish robot joint states
```csharp
// Pseudo-code
void PublishJointStates() {
    JointStateMsg msg = new JointStateMsg();
    msg.name = GetJointNames();
    msg.position = GetJointPositions();
    msg.velocity = GetJointVelocities();
    ros.Publish("/joint_states", msg);
}
```

## Troubleshooting

**Problem:** "ROSConnection could not be found"
- **Solution:** Install ROS TCP Connector package via Package Manager

**Problem:** "Connection to ROS endpoint failed"
- **Solution:**
  1. Check ROS endpoint is running: `ros2 run ros_tcp_endpoint default_server_endpoint`
  2. Verify IP and port in Robotics → ROS Settings
  3. Check firewall isn't blocking port 10000

**Problem:** Robot doesn't move in Unity
- **Solution:**
  1. Ensure Rigidbody is attached to robot
  2. Check "Is Kinematic" is unchecked
  3. Verify `/cmd_vel` messages are being received (check Unity console)
  4. Check mass and drag aren't too high

**Problem:** Coordinate system mismatch (robot moves wrong direction)
- **Solution:** Review coordinate conversion in `ReceiveVelocityCommand`:
  ```csharp
  // Try swapping axes or negating values
  currentLinearVelocity = new Vector3(-linearY, 0, linearX);
  currentAngularVelocity = -angularZ;
  ```

## Next Steps

- Import actual robot URDF files
- Add camera and lidar sensors
- Integrate with Unity ML-Agents for reinforcement learning
- Create procedural environments for training
- Generate synthetic datasets for computer vision

## Learning Resources

- [Unity Robotics Hub Tutorials](https://github.com/Unity-Technologies/Unity-Robotics-Hub/blob/main/tutorials/README.md)
- [ROS TCP Connector Documentation](https://github.com/Unity-Technologies/ROS-TCP-Connector)
- [URDF Importer Guide](https://github.com/Unity-Technologies/URDF-Importer/blob/main/com.unity.robotics.urdf-importer/Documentation~/UR DFImporter.md)
- [Unity Learn Robotics](https://learn.unity.com/course/unity-robotics-hub)
