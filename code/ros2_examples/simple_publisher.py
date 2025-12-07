#!/usr/bin/env python3
"""
Simple ROS 2 Publisher Example

This node demonstrates basic ROS 2 publishing by sending
"Hello, ROS 2!" messages to a topic at regular intervals.

Usage:
    python3 simple_publisher.py

Author: Physical AI & Humanoid Robotics Book
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimplePublisher(Node):
    """
    A basic ROS 2 publisher node that sends string messages.
    """

    def __init__(self):
        """
        Initialize the publisher node.
        """
        # Initialize the node with a unique name
        super().__init__('simple_publisher')

        # Create a publisher
        # - Message type: String
        # - Topic name: 'chatter'
        # - Queue size: 10 (buffer for outgoing messages)
        self.publisher = self.create_publisher(String, 'chatter', 10)

        # Create a timer that triggers the callback every 0.5 seconds
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Counter to track how many messages we've sent
        self.message_count = 0

        # Log that the node has started
        self.get_logger().info('Simple Publisher Node has been started!')
        self.get_logger().info(f'Publishing to topic: chatter at {1/timer_period}Hz')

    def timer_callback(self):
        """
        Timer callback function - called every timer period.
        Creates and publishes a message.
        """
        # Create a new String message
        msg = String()
        msg.data = f'Hello, ROS 2! Message count: {self.message_count}'

        # Publish the message to the 'chatter' topic
        self.publisher.publish(msg)

        # Log the published message
        self.get_logger().info(f'Publishing: "{msg.data}"')

        # Increment the counter
        self.message_count += 1


def main(args=None):
    """
    Main entry point for the node.
    """
    # Initialize the ROS 2 Python client library
    rclpy.init(args=args)

    # Create an instance of the SimplePublisher node
    simple_publisher = SimplePublisher()

    # Spin the node to keep it alive and process callbacks
    # This will run until interrupted (Ctrl+C)
    try:
        rclpy.spin(simple_publisher)
    except KeyboardInterrupt:
        simple_publisher.get_logger().info('Keyboard interrupt, shutting down.')

    # Clean up and shutdown
    simple_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
