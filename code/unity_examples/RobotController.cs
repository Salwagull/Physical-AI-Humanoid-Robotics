/*
 * RobotController.cs
 *
 * Unity script for controlling a robot via ROS 2 /cmd_vel topic
 *
 * Usage:
 * 1. Attach this script to your robot GameObject in Unity
 * 2. Ensure ROS TCP Connector is configured
 * 3. Start ROS 2 endpoint: ros2 run ros_tcp_endpoint default_server_endpoint
 * 4. Publish to /cmd_vel to move the robot
 *
 * Author: Physical AI & Humanoid Robotics Book
 */

using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class RobotController : MonoBehaviour
{
    // ROS connection
    private ROSConnection ros;

    // Physics component
    private Rigidbody rb;

    // Movement parameters
    [Header("Movement Settings")]
    [SerializeField] private float linearSpeed = 2.0f;
    [SerializeField] private float angularSpeed = 100.0f;

    // ROS topic
    [Header("ROS Settings")]
    [SerializeField] private string velocityTopic = "/cmd_vel";

    // Current command
    private Vector3 currentLinearVelocity = Vector3.zero;
    private float currentAngularVelocity = 0f;

    void Start()
    {
        // Get or create ROS connection
        ros = ROSConnection.GetOrCreateInstance();

        // Subscribe to velocity commands
        ros.Subscribe<TwistMsg>(velocityTopic, ReceiveVelocityCommand);

        // Get Rigidbody component
        rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            Debug.LogError("RobotController requires a Rigidbody component!");
        }

        Debug.Log($"RobotController: Subscribed to {velocityTopic}");
    }

    void ReceiveVelocityCommand(TwistMsg twist)
    {
        // Extract velocities from ROS message
        // Note: ROS uses x-forward, Unity uses z-forward
        float linearX = (float)twist.linear.x;
        float linearY = (float)twist.linear.y;
        float angularZ = (float)twist.angular.z;

        // Store for FixedUpdate
        currentLinearVelocity = new Vector3(linearY, 0, linearX);
        currentAngularVelocity = angularZ;

        Debug.Log($"Received cmd_vel: linear={linearX:F2}, angular={angularZ:F2}");
    }

    void FixedUpdate()
    {
        if (rb == null) return;

        // Apply linear velocity (in robot's local frame)
        Vector3 worldVelocity = transform.TransformDirection(currentLinearVelocity * linearSpeed);
        rb.velocity = new Vector3(worldVelocity.x, rb.velocity.y, worldVelocity.z);

        // Apply angular velocity (rotation around Y-axis)
        float turnAmount = currentAngularVelocity * angularSpeed * Time.fixedDeltaTime;
        transform.Rotate(0, -turnAmount, 0); // Negative for ROS convention
    }

    void OnDestroy()
    {
        // Unsubscribe when destroyed
        if (ros != null)
        {
            ros.Unsubscribe(velocityTopic);
        }
    }
}
