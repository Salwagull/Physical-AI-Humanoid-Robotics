---
sidebar_position: 4
---

# Chapter 3: ROS 2 Basics - Nodes, Topics, Services

## Introduction

The Robot Operating System 2 (ROS 2) is the de facto standard middleware for modern robotics development. It provides the communication infrastructure, tools, and libraries that enable you to build complex, distributed robotic systems. Think of ROS 2 as the "nervous system" of your robot—it connects sensors, actuators, planning algorithms, and control systems into a cohesive whole.

**Learning Objectives:**
- Understand ROS 2 architecture and its core concepts
- Learn about nodes, topics, messages, and the publish-subscribe pattern
- Explore services for request-response communication
- Create and run your first ROS 2 Python nodes
- Master the fundamental patterns for robot communication

**Prerequisites:** Chapter 1 (Physical AI), Chapter 2 (Embodied Intelligence), basic Python programming

**Why This Matters:** ROS 2 is the foundation for nearly all modern robotics software. Whether you're building autonomous vehicles, industrial robots, or research platforms, ROS 2 provides the essential infrastructure. Mastering ROS 2 is non-negotiable for serious robotics development.

## Conceptual Overview

### What is ROS 2?

ROS 2 (Robot Operating System 2) is not actually an operating system in the traditional sense—it's a middleware framework and set of tools for building robot applications. ROS 2 provides:

- **Communication infrastructure**: Nodes can exchange data seamlessly
- **Hardware abstraction**: Write code once, run on different robots
- **Standard message types**: Common data formats for sensors and actuators
- **Tools and libraries**: Visualization, simulation, debugging, and more
- **Package management**: Organize and share robot software

**Key improvement in ROS 2**: Unlike ROS 1, ROS 2 is built on DDS (Data Distribution Service), an industry-standard protocol. This makes ROS 2 more reliable, secure, and suitable for production environments.

### Core Concepts: The ROS 2 Graph

ROS 2 applications are built as a **computation graph**—a network of processes (nodes) that communicate via topics and services.

**The ROS 2 Graph consists of:**

1. **Nodes**: Independent processes that perform specific tasks
2. **Topics**: Named channels for asynchronous data streaming
3. **Messages**: Data structures sent over topics
4. **Services**: Request-response communication between nodes
5. **Actions**: Long-running tasks with feedback (we'll cover this in later chapters)
6. **Parameters**: Configuration values for nodes

### Nodes: The Building Blocks

A **node** is a single-purpose executable process. In ROS 2, you design your robot software as a collection of nodes, each responsible for one aspect of functionality.

**Examples of nodes:**
- Camera driver node (publishes images)
- Object detection node (subscribes to images, publishes detections)
- Path planner node (computes navigation paths)
- Motor controller node (sends commands to actuators)

**Why use nodes?**
- **Modularity**: Each node does one thing well
- **Reusability**: Nodes can be reused across different robots
- **Fault isolation**: If one node crashes, others continue running
- **Distributed computing**: Nodes can run on different machines

### Topics and the Publish-Subscribe Pattern

**Topics** are the primary mechanism for data flow in ROS 2. They implement the **publish-subscribe** pattern:

- **Publishers** produce data and send it to a topic
- **Subscribers** consume data from a topic
- Publishers and subscribers don't know about each other—they're **decoupled**

**Example:**
```
Camera Node (Publisher) ──[/camera/image]──> Object Detector (Subscriber)
                                         └──> Display Node (Subscriber)
```

The camera node publishes images to `/camera/image`. Both the object detector and display node subscribe to that topic. The camera doesn't know or care who's listening.

**Benefits of pub-sub:**
- **Loose coupling**: Add/remove nodes without changing others
- **Many-to-many**: Multiple publishers, multiple subscribers
- **Asynchronous**: Non-blocking communication

### Messages: Structured Data

**Messages** define the data structure sent over topics. ROS 2 provides many standard message types:

- `std_msgs/String`: Simple text messages
- `sensor_msgs/Image`: Camera images
- `sensor_msgs/LaserScan`: Lidar data
- `geometry_msgs/Twist`: Velocity commands for mobile robots
- `nav_msgs/Odometry`: Robot position and velocity

You can also define custom messages for your specific needs.

**Message Definition Example (`geometry_msgs/Twist`):**
```
Vector3 linear    # Linear velocity (x, y, z)
Vector3 angular   # Angular velocity (roll, pitch, yaw)
```

### Services: Request-Response Communication

While topics are great for continuous data streams, sometimes you need **request-response** communication. That's where **services** come in.

**Service pattern:**
1. Client sends a request to a service
2. Server processes the request
3. Server sends a response back to the client

**Example use cases:**
- "Calculate the inverse kinematics for this pose" → response with joint angles
- "Is the path clear?" → response: yes/no
- "Save the current map" → response: success/failure

Services are **synchronous and blocking**: the client waits for the response.

### ROS 2 Communication Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     ROS 2 Graph                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   ┌─────────┐   /cmd_vel    ┌─────────────┐           │
│   │ Planner │ ─────────────> │   Robot     │           │
│   │  Node   │   (Twist msg)  │  Controller │           │
│   └─────────┘                └─────────────┘           │
│        ↑                                                │
│        │                                                │
│        │ /scan                                          │
│        │ (LaserScan)                                    │
│        │                                                │
│   ┌─────────┐                                           │
│   │  Lidar  │                                           │
│   │  Node   │                                           │
│   └─────────┘                                           │
│                                                          │
│   [All connected via DDS middleware]                    │
└──────────────────────────────────────────────────────────┘
```

**Data flow:**
1. Lidar node publishes scan data to `/scan` topic
2. Planner node subscribes to `/scan`, processes obstacle data
3. Planner computes safe velocity and publishes to `/cmd_vel`
4. Robot controller subscribes to `/cmd_vel` and moves the robot

All of this happens asynchronously, in real-time, with minimal latency.

## Technical Implementation

### Setting Up a ROS 2 Workspace

Before writing nodes, you need a ROS 2 workspace. A workspace organizes your packages.

**Workspace structure:**
```
ros2_workspace/
├── src/              # Source code for packages
│   └── my_package/
│       ├── my_package/
│       │   └── node.py
│       ├── package.xml
│       └── setup.py
├── build/            # Build artifacts
├── install/          # Installed packages
└── log/              # Build logs
```

**Creating a workspace:**
```bash
mkdir -p ~/ros2_workspace/src
cd ~/ros2_workspace
colcon build
source install/setup.bash
```

### Your First ROS 2 Node: Hello World Publisher

Let's create a simple publisher that sends "Hello, ROS 2!" messages.

**File: `hello_publisher.py`**

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class HelloPublisher(Node):
    def __init__(self):
        # Initialize the node with a name
        super().__init__('hello_publisher')

        # Create a publisher
        # Parameters: message type, topic name, queue size
        self.publisher = self.create_publisher(String, 'hello_topic', 10)

        # Create a timer that calls timer_callback every 1 second
        self.timer = self.create_timer(1.0, self.timer_callback)

        # Counter for messages
        self.counter = 0

        self.get_logger().info('Hello Publisher has started!')

    def timer_callback(self):
        """
        Called every 1 second to publish a message
        """
        # Create a message
        msg = String()
        msg.data = f'Hello, ROS 2! Message #{self.counter}'

        # Publish the message
        self.publisher.publish(msg)

        # Log what we published
        self.get_logger().info(f'Published: "{msg.data}"')

        self.counter += 1

def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Create the node
    node = HelloPublisher()

    # Keep the node running
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # Cleanup
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Key Components:**

1. **Node Initialization**: `super().__init__('hello_publisher')` creates a node named "hello_publisher"
2. **Publisher Creation**: `self.create_publisher(String, 'hello_topic', 10)` creates a publisher for String messages on the 'hello_topic' topic
3. **Timer**: `self.create_timer(1.0, self.timer_callback)` calls `timer_callback()` every second
4. **Message Publishing**: `self.publisher.publish(msg)` sends the message
5. **Spinning**: `rclpy.spin(node)` keeps the node running and processing callbacks

**Running the node:**
```bash
python3 hello_publisher.py
```

**Expected Output:**
```
[INFO] [hello_publisher]: Hello Publisher has started!
[INFO] [hello_publisher]: Published: "Hello, ROS 2! Message #0"
[INFO] [hello_publisher]: Published: "Hello, ROS 2! Message #1"
[INFO] [hello_publisher]: Published: "Hello, ROS 2! Message #2"
...
```

### Your First Subscriber: Hello World Listener

Now let's create a subscriber that listens to the messages published by our publisher.

**File: `hello_subscriber.py`**

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class HelloSubscriber(Node):
    def __init__(self):
        # Initialize the node
        super().__init__('hello_subscriber')

        # Create a subscriber
        # Parameters: message type, topic name, callback function, queue size
        self.subscription = self.create_subscription(
            String,
            'hello_topic',
            self.listener_callback,
            10
        )

        self.get_logger().info('Hello Subscriber has started! Waiting for messages...')

    def listener_callback(self, msg):
        """
        Called whenever a message arrives on hello_topic
        """
        self.get_logger().info(f'Received: "{msg.data}"')

def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Create the node
    node = HelloSubscriber()

    # Keep the node running
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # Cleanup
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Key Components:**

1. **Subscription**: `self.create_subscription(String, 'hello_topic', self.listener_callback, 10)` subscribes to 'hello_topic'
2. **Callback Function**: `listener_callback(self, msg)` is called automatically whenever a message arrives
3. **Message Access**: `msg.data` contains the string sent by the publisher

**Running the subscriber** (in a separate terminal):
```bash
python3 hello_subscriber.py
```

**Expected Output:**
```
[INFO] [hello_subscriber]: Hello Subscriber has started! Waiting for messages...
[INFO] [hello_subscriber]: Received: "Hello, ROS 2! Message #5"
[INFO] [hello_subscriber]: Received: "Hello, ROS 2! Message #6"
[INFO] [hello_subscriber]: Received: "Hello, ROS 2! Message #7"
...
```

### Understanding the Communication Flow

When both nodes are running:

1. **Publisher** creates messages every second
2. **DDS Middleware** transmits messages over the network (even on localhost)
3. **Subscriber** receives messages and invokes the callback
4. **Callback** processes the message (in this case, logs it)

The beauty: publisher and subscriber don't know about each other. You can:
- Start/stop them in any order
- Run multiple subscribers on the same topic
- Run publishers and subscribers on different machines

### Introspection with ROS 2 CLI Tools

ROS 2 provides powerful command-line tools for debugging and introspection.

**List all nodes:**
```bash
ros2 node list
```
Output:
```
/hello_publisher
/hello_subscriber
```

**List all topics:**
```bash
ros2 topic list
```
Output:
```
/hello_topic
/parameter_events
/rosout
```

**See information about a topic:**
```bash
ros2 topic info /hello_topic
```
Output:
```
Type: std_msgs/msg/String
Publisher count: 1
Subscription count: 1
```

**Echo messages on a topic** (see live data):
```bash
ros2 topic echo /hello_topic
```

**Check the message rate:**
```bash
ros2 topic hz /hello_topic
```
Output:
```
average rate: 1.000
```

**Publish a message manually:**
```bash
ros2 topic pub /hello_topic std_msgs/msg/String "{data: 'Manual message'}"
```

These tools are invaluable for debugging and understanding your ROS 2 system.

## Practical Example: Robot Velocity Controller

Let's build something more realistic: a node that controls a robot's velocity based on sensor input.

**Scenario:** A mobile robot uses lidar to detect obstacles. If an obstacle is too close, the robot slows down.

**File: `obstacle_avoider.py`**

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class ObstacleAvoider(Node):
    def __init__(self):
        super().__init__('obstacle_avoider')

        # Subscribe to lidar data
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Publish velocity commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Safety parameters
        self.safe_distance = 1.0  # meters
        self.max_speed = 0.5      # m/s

        self.get_logger().info('Obstacle Avoider started. Safe distance: 1.0m')

    def scan_callback(self, msg):
        """
        Process lidar data and adjust robot speed
        """
        # Find minimum distance from lidar scan
        # Filter out invalid readings (0.0 or inf)
        valid_ranges = [r for r in msg.ranges if r > 0.1 and r < 10.0]

        if not valid_ranges:
            self.get_logger().warn('No valid lidar data!')
            return

        min_distance = min(valid_ranges)

        # Create velocity command
        cmd = Twist()

        if min_distance < self.safe_distance:
            # Too close! Slow down proportionally
            speed_factor = min_distance / self.safe_distance
            cmd.linear.x = self.max_speed * speed_factor

            self.get_logger().info(
                f'Obstacle at {min_distance:.2f}m - Slowing to {cmd.linear.x:.2f}m/s'
            )
        else:
            # Clear path, full speed ahead
            cmd.linear.x = self.max_speed
            self.get_logger().info(
                f'Path clear ({min_distance:.2f}m) - Full speed: {self.max_speed}m/s'
            )

        # No rotation for this simple example
        cmd.angular.z = 0.0

        # Publish the velocity command
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoider()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**How it works:**

1. **Subscribe to `/scan`**: Receives lidar data (LaserScan messages)
2. **Process data**: Finds the minimum distance to any obstacle
3. **Compute velocity**: Slows down proportionally as obstacles get closer
4. **Publish to `/cmd_vel`**: Sends Twist (velocity) commands to the robot

**Expected Behavior:**
- **No obstacles nearby** (> 1.0m): Robot moves at 0.5 m/s
- **Obstacle at 0.5m**: Robot slows to 0.25 m/s (50% speed)
- **Obstacle at 0.2m**: Robot slows to 0.1 m/s (20% speed)

This demonstrates the classic ROS 2 pattern: **sense → process → act**, all through pub-sub communication.

## Visual Aids

### ROS 2 Node and Topic Structure

```
┌─────────────────────────────────────────────────────────────┐
│                     ROS 2 Computation Graph                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐                    ┌──────────────┐     │
│  │   Publisher  │                    │  Subscriber  │     │
│  │     Node     │                    │     Node     │     │
│  └──────┬───────┘                    └───────┬──────┘     │
│         │                                    │             │
│         │ publish()                subscribe()│             │
│         ↓                                    ↓             │
│    ┌────────────────────────────────────────────┐          │
│    │         Topic: /hello_topic                │          │
│    │    Message Type: std_msgs/String           │          │
│    └────────────────────────────────────────────┘          │
│                                                             │
│              [DDS Middleware Layer]                         │
└─────────────────────────────────────────────────────────────┘
```

### Publish-Subscribe vs Service Communication

| Aspect | Topics (Pub-Sub) | Services (Req-Res) |
|--------|------------------|---------------------|
| Pattern | Asynchronous streaming | Synchronous request-response |
| Direction | One-to-many | One-to-one |
| Use Case | Continuous data (sensors) | Occasional queries |
| Blocking | Non-blocking | Blocking (client waits) |
| Examples | Camera images, lidar scans | Path planning, calculations |

## Summary and Next Steps

**Key Takeaways:**
- ROS 2 is a middleware framework that provides communication infrastructure for robots
- **Nodes** are independent processes that perform specific tasks
- **Topics** enable asynchronous pub-sub communication between nodes
- **Messages** define the data structures sent over topics
- **Services** provide synchronous request-response communication
- The **rclpy** library is the Python client for ROS 2
- ROS 2 CLI tools (ros2 node, ros2 topic) are essential for debugging

**What You've Learned:**
You now understand the fundamental architecture of ROS 2 and how nodes communicate via topics and services. You've created your first publisher and subscriber nodes, and you've seen a practical example of robot control using ROS 2 patterns.

**Up Next:**
In [Chapter 4: Gazebo for Robot Simulation](./chapter4_gazebo_simulation.md), we'll learn how to simulate robots in virtual environments using Gazebo. You'll apply your ROS 2 knowledge to control simulated robots, test algorithms safely, and visualize sensor data—all before touching real hardware.

## Exercises and Challenges

**Exercise 1: Custom Publisher**
Modify the `HelloPublisher` to publish random numbers instead of strings. Use the `std_msgs/Int32` message type.

**Exercise 2: Multi-Topic Subscriber**
Create a node that subscribes to TWO topics simultaneously:
- `/sensor_a` (String messages)
- `/sensor_b` (Int32 messages)

Log when messages arrive from each sensor.

**Exercise 3: Service Caller**
Research ROS 2 services. Create a simple service that:
- **Request**: Two integers (a, b)
- **Response**: Their sum (a + b)

Write both the service server and a client that calls it.

**Challenge: Velocity Smoother**
The `ObstacleAvoider` node changes velocity instantly, which can be jerky. Modify it to:
- Gradually increase/decrease speed (smooth acceleration)
- Use a low-pass filter or exponential smoothing

Hint: Maintain a `current_speed` variable and adjust it incrementally each callback.

## Further Reading

- [ROS 2 Official Documentation](https://docs.ros.org/en/humble/) - Comprehensive ROS 2 tutorials and API reference
- [rclpy API Documentation](https://docs.ros2.org/latest/api/rclpy/) - Python client library reference
- [ROS 2 Design](https://design.ros2.org/) - Architectural decisions and rationale behind ROS 2
- [Understanding ROS 2 Topics](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Topics/Understanding-ROS2-Topics.html) - In-depth topic tutorial

---

**Ready to continue?** Chapter 4: Gazebo for Robot Simulation will be available soon, where you'll bring your ROS 2 knowledge into the world of simulation!
