#!/usr/bin/env python3
"""
Isaac Lab Reinforcement Learning Locomotion Training

This script demonstrates how to train a quadruped robot (ANYmal) to walk using
GPU-accelerated reinforcement learning with Isaac Lab and Stable Baselines3.

Prerequisites:
- NVIDIA Isaac Sim 4.5+ or 5.0 installed
- Isaac Lab installed (see README.md)
- RTX GPU with 8GB+ VRAM

Usage:
    # Basic training with default settings
    python rl_locomotion_training.py

    # Training with more environments (requires more VRAM)
    python rl_locomotion_training.py --num_envs 4096

    # Headless training (faster, no rendering)
    python rl_locomotion_training.py --headless

    # Resume from checkpoint
    python rl_locomotion_training.py --checkpoint ./checkpoints/model_1000000_steps.zip

Expected Output:
    - Training logs to console and TensorBoard
    - Saved model checkpoints to ./checkpoints/
    - Final model saved as anymal_locomotion_policy.zip

Training Time (RTX 4090):
    - 1M steps: ~10 minutes
    - 5M steps: ~45 minutes
    - 10M steps: ~90 minutes

Author: Physical AI & Humanoid Robotics Book
"""

import argparse
import os
from datetime import datetime

# Parse arguments before importing heavy libraries
parser = argparse.ArgumentParser(description="Train quadruped locomotion with Isaac Lab")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of parallel environments")
parser.add_argument("--timesteps", type=int, default=5_000_000, help="Total training timesteps")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
parser.add_argument("--terrain", choices=["flat", "rough"], default="flat", help="Terrain type")
args = parser.parse_args()

# Initialize Isaac Sim (must be done before other imports)
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless,
    "width": 1280,
    "height": 720,
})

print("="*60)
print("Isaac Lab Locomotion Training")
print("="*60)
print(f"Environment count: {args.num_envs}")
print(f"Total timesteps: {args.timesteps:,}")
print(f"Terrain: {args.terrain}")
print(f"Headless: {args.headless}")
print("="*60)

# Now import the rest
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure

# Import Isaac Lab
try:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.manager_based.locomotion.velocity import velocity_env_cfg
except ImportError as e:
    print(f"Error importing Isaac Lab: {e}")
    print("\nPlease install Isaac Lab:")
    print("  git clone https://github.com/isaac-sim/IsaacLab.git")
    print("  cd IsaacLab && pip install -e .[rl_games,sb3]")
    simulation_app.close()
    exit(1)


def create_environment(num_envs: int, terrain: str):
    """Create Isaac Lab locomotion environment."""

    # Select environment configuration based on terrain
    if terrain == "flat":
        env_id = "Isaac-Velocity-Flat-Anymal-C-v0"
        env_cfg = velocity_env_cfg.AnymalCFlatEnvCfg()
    else:
        env_id = "Isaac-Velocity-Rough-Anymal-C-v0"
        env_cfg = velocity_env_cfg.AnymalCRoughEnvCfg()

    # Configure number of parallel environments
    env_cfg.scene.num_envs = num_envs

    # Create gymnasium-compatible environment
    env = gym.make(env_id, cfg=env_cfg)

    print(f"\nEnvironment created: {env_id}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Number of environments: {env.num_envs}")

    return env


def create_ppo_model(env, checkpoint: str = None):
    """Create or load PPO model with optimized hyperparameters."""

    # Hyperparameters optimized for quadruped locomotion
    hyperparams = {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 24,  # Steps per environment before update
        "batch_size": env.num_envs * 24,  # Use all data
        "n_epochs": 5,
        "gamma": 0.99,  # Discount factor
        "gae_lambda": 0.95,  # GAE lambda
        "clip_range": 0.2,  # PPO clip range
        "ent_coef": 0.01,  # Entropy bonus (exploration)
        "vf_coef": 0.5,  # Value function coefficient
        "max_grad_norm": 1.0,  # Gradient clipping
        "verbose": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    if checkpoint and os.path.exists(checkpoint):
        print(f"\nLoading model from checkpoint: {checkpoint}")
        model = PPO.load(checkpoint, env=env)
    else:
        print("\nCreating new PPO model...")
        print(f"  Device: {hyperparams['device']}")
        print(f"  Batch size: {hyperparams['batch_size']:,}")
        model = PPO(env=env, **hyperparams)

    return model


def setup_callbacks(env, log_dir: str, checkpoint_dir: str):
    """Setup training callbacks for logging and checkpointing."""

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Checkpoint callback - save every 500k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000 // env.num_envs,  # Adjust for vectorized env
        save_path=checkpoint_dir,
        name_prefix="model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # Evaluation callback (optional, uses main env)
    eval_callback = EvalCallback(
        env,
        best_model_save_path=checkpoint_dir,
        log_path=log_dir,
        eval_freq=100_000 // env.num_envs,
        n_eval_episodes=10,
        deterministic=True,
    )

    return CallbackList([checkpoint_callback, eval_callback])


def main():
    """Main training loop."""

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./tensorboard_logs/locomotion_{timestamp}"
    checkpoint_dir = f"./checkpoints/locomotion_{timestamp}"

    print(f"\nLog directory: {log_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Create environment
    env = create_environment(args.num_envs, args.terrain)

    # Create model
    model = create_ppo_model(env, args.checkpoint)

    # Setup logging
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # Setup callbacks
    callbacks = setup_callbacks(env, log_dir, checkpoint_dir)

    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Training for {args.timesteps:,} timesteps...")
    print("Monitor progress with TensorBoard:")
    print(f"  tensorboard --logdir {log_dir}")
    print("="*60 + "\n")

    try:
        # Train
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=args.checkpoint is None,
        )

        # Save final model
        final_model_path = os.path.join(checkpoint_dir, "final_model.zip")
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")

        # Also save as convenient name
        model.save("anymal_locomotion_policy")
        print("Also saved as: anymal_locomotion_policy.zip")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        # Save checkpoint on interrupt
        interrupt_path = os.path.join(checkpoint_dir, "interrupted_model.zip")
        model.save(interrupt_path)
        print(f"Partial model saved to: {interrupt_path}")

    finally:
        env.close()
        simulation_app.close()

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nTo evaluate the trained policy:")
    print(f"  python evaluate_policy.py --model {checkpoint_dir}/final_model.zip")
    print("\nTo export for real robot deployment:")
    print("  python export_policy_onnx.py --model anymal_locomotion_policy.zip")


if __name__ == "__main__":
    main()
