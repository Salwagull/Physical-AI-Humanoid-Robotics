---
sidebar_position: 6
---

# Chapter 5: Unity for Robotics Simulation

## Introduction

Unity is a powerful game engine that has found a second life in robotics. While Gazebo excels at physics accuracy, Unity brings stunning visuals, VR/AR support, massive asset libraries, and machine learning integration. Unity is ideal for human-robot interaction research, synthetic data generation for AI training, and creating photorealistic simulations.

**Learning Objectives:**
- Understand Unity's role in robotics simulation
- Learn Unity Robotics Hub and ROS integration
- Create robot scenes with Unity's physics engine
- Generate synthetic training data for ML models
- Compare Unity vs Gazebo for different use cases
- Build interactive robot demonstrations

**Prerequisites:** Chapter 4 (Gazebo), basic understanding of game engines, C# programming basics

**Why This Matters:** Unity enables simulations that Gazebo can't match—photorealistic rendering for computer vision, VR/AR interfaces for teleoperation, and massive procedural environments for reinforcement learning. It's becoming the standard for ML-driven robotics research.

## Conceptual Overview

### Why Unity for Robotics?

**Unity's Strengths:**

1. **Photorealistic Rendering**: High-quality graphics for vision systems
2. **Asset Ecosystem**: Thousands of 3D models, textures, and environments
3. **ML Integration**: Unity ML-Agents for reinforcement learning
4. **VR/AR Support**: Native support for immersive interfaces
5. **Cross-Platform**: Deploy to desktop, mobile, web, VR headsets
6. **Procedural Generation**: Create infinite training scenarios
7. **Real-Time Performance**: Optimized for interactive experiences

**Unity vs Gazebo**:

| Feature | Unity | Gazebo |
|---------|-------|--------|
| **Physics Accuracy** | Good | Excellent |
| **Visual Quality** | Excellent | Good |
| **Asset Library** | Massive | Limited |
| **ML Integration** | Built-in | External |
| **VR/AR** | Native | Limited |
| **ROS Integration** | Via Unity Robotics Hub | Native |
| **Learning Curve** | Moderate (C#) | Moderate (SDF/C++) |
| **Use Case** | Vision, HRI, ML training | Navigation, manipulation |

**When to Use Unity:**
- Computer vision algorithm development
- Synthetic data generation for deep learning
- Human-robot interaction studies
- VR/AR teleoperation
- Photorealistic product demos
- Procedural environment generation

**When to Use Gazebo:**
- High-fidelity physics simulation
- ROS-centric workflows
- Real-world robot validation
- Navigation stack testing
- Multi-robot coordination

### Unity Robotics Hub

**Unity Robotics Hub** is Unity's official solution for robotics development:

**Components:**
1. **Unity ROS Integration**: Bidirectional communication with ROS/ROS 2
2. **URDF Importer**: Import robot models from ROS
3. **Articulation Body**: Advanced physics for robotic joints
4. **Computer Vision Tools**: Sensor simulation and labeling
5. **Example Projects**: Pre-built robotics scenarios

**Architecture:**

```
┌─────────────────────────────────────────────────────┐
│              Unity Simulation                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌────────────┐         ┌─────────────┐           │
│  │   Robot    │         │  Environment│           │
│  │  (URDF)    │         │  (Sensors)  │           │
│  └──────┬─────┘         └──────┬──────┘           │
│         │                      │                   │
│         └──────────┬───────────┘                   │
│                    │                               │
│           ┌────────▼─────────┐                     │
│           │ ROS TCP Connector│                     │
│           └────────┬─────────┘                     │
└────────────────────┼─────────────────────────────── ┘
                     │ TCP/IP
              ┌──────▼──────┐
              │ ROS 2 Node  │
              │(Endpoint)   │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │ Your ROS 2  │
              │ Application │
              └─────────────┘
```

**Communication Flow:**
1. Unity publishes sensor data (cameras, lidar) → ROS 2 topics
2. ROS 2 publishes commands (Twist, JointState) → Unity subscribes
3. All communication via TCP using ROS messages

## Technical Implementation

### Setting Up Unity for Robotics

**Prerequisites:**
- Unity Hub (latest)
- Unity Editor 2021.3 LTS or newer
- ROS 2 Humble installed

**Installation Steps:**

1. **Create Unity Project:**
   - Open Unity Hub → New Project
   - Template: 3D (URP for better graphics)
   - Name: "RoboticsSimulation"

2. **Install Unity Robotics Hub:**
   - Window → Package Manager
   - Add package from git URL:
     ```
     https://github.com/Unity-Technologies/Unity-Robotics-Hub.git?path=/com.unity.robotics.ros-tcp-connector
     ```
   - Also install: URDF Importer package

3. **Configure ROS Connection:**
   - Robotics → ROS Settings
   - ROS IP Address: `127.0.0.1` (localhost)
   - ROS Port: `10000`
   - Protocol: ROS 2

### Creating a Simple Robot Scene

**Step 1: Create the Environment**

```csharp
// GroundPlane.cs - Simple ground plane
using UnityEngine;

public class GroundPlane : MonoBehaviour
{
    void Start()
    {
        // Create ground plane
        GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.transform.position = Vector3.zero;
        ground.transform.localScale = new Vector3(10, 1, 10);

        // Add physics
        ground.AddComponent<MeshCollider>();
    }
}
```

**Step 2: Import Robot from URDF**

Unity can import ROS URDF files directly:

1. Assets → Import Robot from URDF
2. Select your robot's URDF file
3. Unity creates GameObjects for each link
4. Joints are configured automatically

**Step 3: Add ROS Communication**

```csharp
// RobotController.cs - Subscribe to /cmd_vel and move robot
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class RobotController : MonoBehaviour
{
    private ROSConnection ros;
    private Rigidbody rb;

    [SerializeField] private float speed = 2.0f;
    [SerializeField] private float turnSpeed = 100.0f;

    void Start()
    {
        // Get ROS connection
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<TwistMsg>("/cmd_vel", ReceiveVelocityCommand);

        // Get physics component
        rb = GetComponent<Rigidbody>();

        Debug.Log("Robot Controller: Subscribed to /cmd_vel");
    }

    void ReceiveVelocityCommand(TwistMsg twist)
    {
        // Extract linear and angular velocities
        float linear = (float)twist.linear.x;
        float angular = (float)twist.angular.z;

        // Apply to robot (Unity uses different coordinate system)
        Vector3 movement = transform.forward * linear * speed;
        rb.velocity = movement;

        float turnAmount = angular * turnSpeed * Time.deltaTime;
        transform.Rotate(0, turnAmount, 0);
    }
}
```

**Step 4: Publish Sensor Data**

```csharp
// CameraPublisher.cs - Publish camera images to ROS
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class CameraPublisher : MonoBehaviour
{
    private ROSConnection ros;
    private Camera cam;

    [SerializeField] private string topicName = "/camera/image_raw";
    [SerializeField] private float publishRate = 10.0f; // Hz

    private float timer;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImageMsg>(topicName);

        cam = GetComponent<Camera>();
        timer = 0f;
    }

    void Update()
    {
        timer += Time.deltaTime;

        if (timer >= 1.0f / publishRate)
        {
            PublishImage();
            timer = 0f;
        }
    }

    void PublishImage()
    {
        // Capture camera frame
        RenderTexture rt = new RenderTexture(640, 480, 24);
        cam.targetTexture = rt;
        cam.Render();

        Texture2D image = new Texture2D(640, 480, TextureFormat.RGB24, false);
        RenderTexture.active = rt;
        image.ReadPixels(new Rect(0, 0, 640, 480), 0, 0);
        image.Apply();

        // Convert to ROS message
        ImageMsg msg = new ImageMsg
        {
            header = new RosMessageTypes.Std.HeaderMsg
            {
                stamp = new RosMessageTypes.BuiltinInterfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                }
            },
            height = 480,
            width = 640,
            encoding = "rgb8",
            step = 640 * 3,
            data = image.GetRawTextureData()
        };

        ros.Publish(topicName, msg);

        // Cleanup
        cam.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);
    }
}
```

### Running the Simulation

**Terminal 1: Start ROS 2 Endpoint**
```bash
# Install Unity ROS endpoint
pip3 install roslibpy

# Run endpoint
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0
```

**Unity: Press Play**

**Terminal 2: Control the Robot**
```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 1.0}, angular: {z: 0.0}}"
```

The robot moves in Unity, and camera images are published to ROS!

## Practical Example: Synthetic Data Generation

Unity excels at generating training data for machine learning.

**Scenario:** Generate labeled images of objects for object detection training.

**Setup:**
1. Create scene with random objects
2. Vary lighting, camera angles, backgrounds
3. Automatically label bounding boxes
4. Export thousands of images

**Perception Camera Setup:**

```csharp
// DatasetGenerator.cs
using UnityEngine;
using UnityEngine.Perception.GroundTruth;

public class DatasetGenerator : MonoBehaviour
{
    void Start()
    {
        // Add perception camera component
        var perceptionCamera = gameObject.AddComponent<PerceptionCamera>();

        // Configure labelers
        var labelConfig = ScriptableObject.CreateInstance<IdLabelConfig>();
        // Add labels for objects (e.g., "robot", "obstacle", "target")

        // Add bounding box labeler
        var boundingBoxLabeler = new BoundingBox2DLabeler(labelConfig);
        perceptionCamera.AddLabeler(boundingBoxLabeler);

        Debug.Log("Dataset generation configured");
    }
}
```

Unity automatically generates:
- RGB images
- Bounding box annotations (COCO format)
- Semantic segmentation masks
- Instance segmentation
- Depth maps

Perfect for training YOLO, Faster R-CNN, or other vision models!

## Visual Aids

### Unity vs Gazebo Workflow

```
Gazebo Workflow:
Design → SDF Model → Spawn → Test Physics → Transfer

Unity Workflow:
Design → Scene Setup → Import URDF → Add Sensors → ML Training → Transfer
```

### Unity Robotics Architecture

```
Unity Scene
  ├── Robot (URDF)
  │    ├── Links (Rigid Bodies)
  │    ├── Joints (Articulations)
  │    └── Sensors (Cameras, Lidar)
  ├── Environment (Assets)
  └── ROS TCP Connector
       ├── Publishers (Sensors → ROS)
       └── Subscribers (ROS → Actuators)
```

## Summary and Next Steps

**Key Takeaways:**
- Unity provides photorealistic rendering and massive asset libraries
- Unity Robotics Hub enables ROS/ROS 2 integration
- URDF importer brings ROS robots into Unity
- Unity excels at synthetic data generation for ML
- Use Unity for vision, HRI, and ML; use Gazebo for physics accuracy
- C# scripts control Unity-ROS communication

**What You've Learned:**
You understand Unity's role in robotics, how to integrate Unity with ROS 2, create robot scenes, and generate synthetic training data. You can choose between Unity and Gazebo based on your project needs.

**Up Next:**
In [Chapter 6: Introduction to NVIDIA Isaac Sim](chapter6_intro_isaac_sim.md), we'll explore the most advanced robotics simulator available. Isaac Sim combines photorealistic rendering, accurate physics, and GPU acceleration—bringing together the best of Unity and Gazebo, plus AI-powered perception and synthetic data generation at massive scale.

## Exercises and Challenges

**Exercise 1: Basic Scene**
Create a Unity scene with:
- Ground plane
- Three obstacles (cubes, spheres)
- Directional light
- A simple robot (box with wheels)

**Exercise 2: ROS Integration**
Set up ROS TCP Connector and:
- Subscribe to `/cmd_vel`
- Move a cube based on Twist messages
- Publish cube position to `/object/pose`

**Exercise 3: Camera Sensor**
Add a camera to your scene and:
- Publish images to `/camera/image_raw`
- View images in RViz: `ros2 run rviz2 rviz2`
- Adjust camera FOV and resolution

**Challenge: Object Detection Dataset**
Generate a synthetic dataset:
- Random object placement (10 objects)
- Random lighting (3 light sources at different angles)
- Capture 1000 images with bounding boxes
- Export in COCO format

## Further Reading

- [Unity Robotics Hub](https://github.com/Unity-Technologies/Unity-Robotics-Hub) - Official Unity robotics tools
- [Unity Perception Package](https://github.com/Unity-Technologies/com.unity.perception) - Synthetic data generation
- [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) - Reinforcement learning in Unity
- [URDF Importer Guide](https://github.com/Unity-Technologies/URDF-Importer) - Import ROS robots

---

**Ready to continue?** Proceed to [Chapter 6: Introduction to NVIDIA Isaac Sim](chapter6_intro_isaac_sim.md)!
