#!/usr/bin/env python3
"""
Export Trained Policy to ONNX for Real Robot Deployment

This script exports a trained Stable Baselines3 policy to ONNX format
for deployment on real robots using efficient inference engines like
ONNX Runtime or TensorRT.

Prerequisites:
- Trained model from rl_locomotion_training.py
- torch and onnx installed

Usage:
    # Export default model
    python export_policy_onnx.py

    # Export specific model
    python export_policy_onnx.py --model ./checkpoints/model.zip

    # Export with TensorRT optimization
    python export_policy_onnx.py --tensorrt

Output:
    - locomotion_policy.onnx (ONNX model)
    - locomotion_policy.trt (TensorRT model, if --tensorrt)

Deployment:
    The exported ONNX model can be used with:
    - ONNX Runtime (Python, C++, ROS 2)
    - TensorRT (NVIDIA Jetson, desktop GPU)
    - OpenVINO (Intel hardware)

Author: Physical AI & Humanoid Robotics Book
"""

import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description="Export policy to ONNX")
parser.add_argument("--model", type=str, default="anymal_locomotion_policy.zip",
                    help="Path to trained model")
parser.add_argument("--output", type=str, default="locomotion_policy.onnx",
                    help="Output ONNX file path")
parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
parser.add_argument("--tensorrt", action="store_true",
                    help="Also export TensorRT engine")
parser.add_argument("--validate", action="store_true",
                    help="Validate ONNX model outputs match original")
args = parser.parse_args()

print("="*60)
print("Policy Export to ONNX")
print("="*60)

# Import PyTorch and ONNX
import torch
import torch.nn as nn

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("Please install onnx and onnxruntime:")
    print("  pip install onnx onnxruntime")
    exit(1)

# Load the trained model
try:
    from stable_baselines3 import PPO
    print(f"\nLoading model from: {args.model}")
    model = PPO.load(args.model)
    print("  ✓ Model loaded successfully")
except FileNotFoundError:
    print(f"Error: Model not found at {args.model}")
    print("Please train a model first with rl_locomotion_training.py")
    exit(1)


class PolicyWrapper(nn.Module):
    """
    Wrapper to extract just the policy network for inference.

    The full SB3 model includes value function, but for deployment
    we only need the policy (actor) network.
    """

    def __init__(self, sb3_model):
        super().__init__()
        self.policy = sb3_model.policy
        self.observation_space = sb3_model.observation_space

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through policy network.

        Args:
            observation: Robot state observation [batch, obs_dim]

        Returns:
            action: Motor commands [batch, action_dim]
        """
        # Get deterministic action (mean of action distribution)
        # For deployment, we typically use deterministic actions
        with torch.no_grad():
            features = self.policy.extract_features(observation)
            if hasattr(self.policy, 'mlp_extractor'):
                latent_pi, _ = self.policy.mlp_extractor(features)
            else:
                latent_pi = features
            action = self.policy.action_net(latent_pi)
        return action


def export_to_onnx(model, output_path, opset_version):
    """Export the policy to ONNX format."""

    print(f"\nExporting to ONNX...")

    # Wrap policy for clean inference
    policy_wrapper = PolicyWrapper(model)
    policy_wrapper.eval()

    # Create dummy input matching observation space
    obs_dim = model.observation_space.shape[0]
    dummy_input = torch.randn(1, obs_dim)

    print(f"  Observation dimension: {obs_dim}")
    print(f"  Action dimension: {model.action_space.shape[0]}")

    # Export to ONNX
    torch.onnx.export(
        policy_wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
    )

    print(f"  ✓ ONNX model saved to: {output_path}")

    # Get file size
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"  ✓ Model size: {file_size:.2f} MB")

    return policy_wrapper, dummy_input


def validate_onnx(policy_wrapper, dummy_input, onnx_path):
    """Validate that ONNX model produces same outputs as PyTorch."""

    print(f"\nValidating ONNX model...")

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model passed validation")

    # Create ONNX Runtime session
    session = ort.InferenceSession(onnx_path)

    # Run inference with both models
    policy_wrapper.eval()
    with torch.no_grad():
        pytorch_output = policy_wrapper(dummy_input).numpy()

    onnx_output = session.run(
        None,
        {"observation": dummy_input.numpy()}
    )[0]

    # Compare outputs
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    print(f"  Max difference: {max_diff:.2e}")

    if max_diff < 1e-5:
        print("  ✓ ONNX output matches PyTorch (within tolerance)")
    else:
        print("  ⚠ Warning: Outputs differ significantly")

    return session


def benchmark_inference(session, obs_dim, num_runs=1000):
    """Benchmark ONNX Runtime inference speed."""

    print(f"\nBenchmarking inference speed ({num_runs} runs)...")

    import time

    # Warmup
    dummy = np.random.randn(1, obs_dim).astype(np.float32)
    for _ in range(10):
        session.run(None, {"observation": dummy})

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {"observation": dummy})
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    avg_time = np.mean(times)
    std_time = np.std(times)
    p99_time = np.percentile(times, 99)

    print(f"  Average: {avg_time:.3f} ms")
    print(f"  Std dev: {std_time:.3f} ms")
    print(f"  P99: {p99_time:.3f} ms")
    print(f"  Throughput: {1000/avg_time:.0f} inferences/sec")

    # Check if suitable for real-time control
    control_rate = 50  # Hz (typical robot control rate)
    max_allowed = 1000 / control_rate  # ms per inference

    if avg_time < max_allowed:
        print(f"  ✓ Suitable for {control_rate} Hz control (need < {max_allowed:.1f} ms)")
    else:
        print(f"  ⚠ May be too slow for {control_rate} Hz control")


def export_tensorrt(onnx_path, trt_path):
    """Export to TensorRT for optimized GPU inference."""

    print(f"\nExporting to TensorRT...")

    try:
        import tensorrt as trt
    except ImportError:
        print("  TensorRT not available. Install with:")
        print("  pip install tensorrt")
        return

    # TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Build engine
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        # Parse ONNX model
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(f"  Error: {parser.get_error(error)}")
                return

        # Build config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        # Enable FP16 if available
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 mode enabled")

        # Build engine
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            print("  Failed to build TensorRT engine")
            return

        # Save engine
        with open(trt_path, "wb") as f:
            f.write(engine)

        print(f"  ✓ TensorRT engine saved to: {trt_path}")
        file_size = os.path.getsize(trt_path) / 1024 / 1024
        print(f"  ✓ Engine size: {file_size:.2f} MB")


def create_deployment_example():
    """Create example deployment code."""

    example_code = '''#!/usr/bin/env python3
"""
Example: Deploy ONNX policy on real robot with ROS 2

This node subscribes to joint states and publishes motor commands
using the trained locomotion policy.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import onnxruntime as ort
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray


class LocomotionPolicyNode(Node):
    """ROS 2 node for running locomotion policy."""

    def __init__(self):
        super().__init__("locomotion_policy")

        # Load ONNX model
        self.session = ort.InferenceSession("locomotion_policy.onnx")
        self.input_name = self.session.get_inputs()[0].name
        self.get_logger().info("Loaded ONNX policy model")

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, "/imu/data", self.imu_callback, 10
        )

        # Publisher
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, "/joint_commands", 10
        )

        # State buffers
        self.joint_positions = None
        self.joint_velocities = None
        self.imu_orientation = None
        self.imu_angular_vel = None

        # Control loop at 50 Hz
        self.timer = self.create_timer(0.02, self.control_callback)

        # Velocity command (from higher-level planner)
        self.velocity_cmd = np.array([0.5, 0.0, 0.0])  # [vx, vy, yaw_rate]

    def joint_callback(self, msg: JointState):
        self.joint_positions = np.array(msg.position)
        self.joint_velocities = np.array(msg.velocity)

    def imu_callback(self, msg: Imu):
        self.imu_orientation = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])
        self.imu_angular_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

    def control_callback(self):
        """Run policy inference and publish commands."""
        if self.joint_positions is None or self.imu_orientation is None:
            return

        # Build observation vector
        # Note: Order must match training observation space!
        observation = np.concatenate([
            self.velocity_cmd,           # 3: velocity command
            self.imu_orientation,        # 4: base orientation (quaternion)
            self.imu_angular_vel,        # 3: base angular velocity
            self.joint_positions,        # 12: joint positions
            self.joint_velocities,       # 12: joint velocities
            # Add previous actions if used in training
        ]).astype(np.float32).reshape(1, -1)

        # Run policy inference
        action = self.session.run(None, {self.input_name: observation})[0]

        # Publish motor commands
        msg = Float64MultiArray()
        msg.data = action.flatten().tolist()
        self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = LocomotionPolicyNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
'''

    example_path = "deployment_example_ros2.py"
    with open(example_path, "w") as f:
        f.write(example_code)

    print(f"\n  Created deployment example: {example_path}")


def main():
    """Main export function."""

    # Export to ONNX
    policy_wrapper, dummy_input = export_to_onnx(
        model, args.output, args.opset
    )

    # Validate
    if args.validate or True:  # Always validate
        session = validate_onnx(policy_wrapper, dummy_input, args.output)

        # Benchmark
        benchmark_inference(
            session,
            model.observation_space.shape[0]
        )

    # Export to TensorRT if requested
    if args.tensorrt:
        trt_path = args.output.replace(".onnx", ".trt")
        export_tensorrt(args.output, trt_path)

    # Create deployment example
    create_deployment_example()

    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print(f"\nONNX model: {args.output}")
    print("\nTo use in ROS 2:")
    print("  1. Copy locomotion_policy.onnx to robot")
    print("  2. Adapt deployment_example_ros2.py for your robot")
    print("  3. Source ROS 2 workspace and run node")
    print("\nTo convert to TensorRT (on target device):")
    print("  trtexec --onnx=locomotion_policy.onnx --saveEngine=policy.trt")


if __name__ == "__main__":
    main()
