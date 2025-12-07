#!/usr/bin/env python3
"""
Simple ROS 2 Subscriber Example

This node demonstrates basic ROS 2 subscription by listening
to messages on a topic and processing them.

Usage:
    python3 simple_subscriber.py

Author: Physical AI & Humanoid Robotics Book
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimpleSubscriber(Node):
    """
    A basic ROS 2 subscriber node that receives string messages.
    """

    def __init__(self):
        """
        Initialize the subscriber node.
        """
        # Initialize the node with a unique name
        super().__init__('simple_subscriber')

        # Create a subscription
        # - Message type: String
        # - Topic name: 'chatter' (same as the publisher)
        # - Callback function: listener_callback
        # - Queue size: 10 (buffer for incoming messages)
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10
        )

        # Track how many messages we've received
        self.message_count = 0

        # Log that the node has started
        self.get_logger().info('Simple Subscriber Node has been started!')
        self.get_logger().info('Waiting for messages on topic: chatter')

    def listener_callback(self, msg):
        """
        Callback function that processes received messages.

        Args:
            msg (String): The message received from the topic
        """
        # Increment the receive counter
        self.message_count += 1

        # Log the received message
        self.get_logger().info(f'Received message #{self.message_count}: "{msg.data}"')

        # You can add any processing logic here
        # For example, parse the message, trigger actions, etc.


def main(args=None):
    """
    Main entry point for the node.
    """
    # Initialize the ROS 2 Python client library
    rclpy.init(args=args)

    # Create an instance of the SimpleSubscriber node
    simple_subscriber = SimpleSubscriber()

    # Spin the node to keep it alive and process incoming messages
    # This will run until interrupted (Ctrl+C)
    try:
        rclpy.spin(simple_subscriber)
    except KeyboardInterrupt:
        simple_subscriber.get_logger().info('Keyboard interrupt, shutting down.')

    # Clean up and shutdown
    simple_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
