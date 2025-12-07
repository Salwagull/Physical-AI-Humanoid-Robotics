# ROS 2 Examples

This directory contains ROS 2 Python code examples from Chapter 3: ROS 2 Basics.

## Prerequisites

- ROS 2 Humble (or later) installed
- Python 3.8+
- rclpy (ROS 2 Python client library)

## Installation

If you have ROS 2 installed, rclpy should already be available. To verify:

```bash
python3 -c "import rclpy; print('rclpy is installed!')"
```

If not installed, follow the [ROS 2 installation guide](https://docs.ros.org/en/humble/Installation.html).

## Examples

### 1. Simple Publisher (`simple_publisher.py`)

A basic publisher that sends "Hello, ROS 2!" messages to the 'chatter' topic at 2Hz.

**Run:**
```bash
python3 simple_publisher.py
```

**Expected Output:**
```
[INFO] [simple_publisher]: Simple Publisher Node has been started!
[INFO] [simple_publisher]: Publishing to topic: chatter at 2.0Hz
[INFO] [simple_publisher]: Publishing: "Hello, ROS 2! Message count: 0"
[INFO] [simple_publisher]: Publishing: "Hello, ROS 2! Message count: 1"
[INFO] [simple_publisher]: Publishing: "Hello, ROS 2! Message count: 2"
...
```

**What it does:**
- Creates a ROS 2 node named 'simple_publisher'
- Publishes String messages to the 'chatter' topic
- Uses a timer to publish every 0.5 seconds

### 2. Simple Subscriber (`simple_subscriber.py`)

A basic subscriber that listens to messages on the 'chatter' topic and logs them.

**Run** (in a separate terminal):
```bash
python3 simple_subscriber.py
```

**Expected Output:**
```
[INFO] [simple_subscriber]: Simple Subscriber Node has been started!
[INFO] [simple_subscriber]: Waiting for messages on topic: chatter
[INFO] [simple_subscriber]: Received message #1: "Hello, ROS 2! Message count: 5"
[INFO] [simple_subscriber]: Received message #2: "Hello, ROS 2! Message count: 6"
[INFO] [simple_subscriber]: Received message #3: "Hello, ROS 2! Message count: 7"
...
```

**What it does:**
- Creates a ROS 2 node named 'simple_subscriber'
- Subscribes to the 'chatter' topic
- Logs each received message via the callback function

## Running Both Together

To see pub-sub in action, run both nodes simultaneously:

**Terminal 1:**
```bash
python3 simple_publisher.py
```

**Terminal 2:**
```bash
python3 simple_subscriber.py
```

You should see the publisher sending messages and the subscriber receiving them!

## Using ROS 2 CLI Tools

While the nodes are running, try these commands in a third terminal:

**List all nodes:**
```bash
ros2 node list
```

**List all topics:**
```bash
ros2 topic list
```

**See topic info:**
```bash
ros2 topic info /chatter
```

**Echo messages (see live data):**
```bash
ros2 topic echo /chatter
```

**Check message rate:**
```bash
ros2 topic hz /chatter
```

**Publish a manual message:**
```bash
ros2 topic pub /chatter std_msgs/msg/String "{data: 'Manual test message'}"
```

## Troubleshooting

**Problem:** `ModuleNotFoundError: No module named 'rclpy'`
- **Solution:** Make sure ROS 2 is properly installed and sourced:
  ```bash
  source /opt/ros/humble/setup.bash
  ```

**Problem:** Subscriber doesn't receive messages
- **Solution:** Check that both publisher and subscriber are running and using the same topic name
- Verify with `ros2 topic list` and `ros2 topic echo /chatter`

**Problem:** "Cannot communicate with ROS master"
- **Solution:** ROS 2 doesn't use a master (unlike ROS 1), so this shouldn't happen. If you see this, you might be mixing ROS 1 and ROS 2 commands.

## Next Steps

- Modify the publisher to send different message types (e.g., Int32, Float32)
- Create a subscriber that processes messages and triggers actions
- Experiment with multiple publishers/subscribers on the same topic
- Try creating a service instead of a topic (request-response pattern)

## Learning Resources

- [ROS 2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
- [rclpy API](https://docs.ros2.org/latest/api/rclpy/)
- [ROS 2 Examples Repository](https://github.com/ros2/examples)
