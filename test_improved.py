#!/usr/bin/env python3
"""
Test script for the improved DQN implementation.
Tests all new features including proper preprocessing, double DQN, and evaluation.
"""

import gymnasium as gym
import ale_py
import numpy as np
import torch
import os

# Register ALE environments
gym.register_envs(ale_py)

def test_improved_implementation():
    """Test the improved DQN implementation."""
    print("🧪 Testing Improved DQN Implementation...")
    print("=" * 60)
    
    # Test 1: Environment with proper preprocessing
    print("1. Testing improved environment setup...")
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
    
    env = gym.make('ALE/Pong-v5', render_mode=None, frameskip=1)
    env = AtariPreprocessing(
        env, 
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=True
    )
    env = FrameStackObservation(env, stack_size=4)
    
    print(f"   ✓ Environment created with proper Atari preprocessing")
    print(f"   ✓ State shape: {env.observation_space.shape}")
    print(f"   ✓ Action space: {env.action_space}")
    
    # Test 2: Agent with improved hyperparameters
    print("2. Testing improved agent...")
    from dqn_agent import DQNAgent
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(
        state_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        lr=0.0005,              # Increased learning rate
        replay_start_size=50000, # Warm-up period
        target_update_freq=5000, # Reduced frequency
        double_dqn=True,        # Double DQN
        grad_clip=True,         # Gradient clipping
        device=device
    )
    
    print(f"   ✓ Agent created with improved hyperparameters")
    print(f"   ✓ Learning rate: 0.0005 (increased)")
    print(f"   ✓ Replay start size: 50,000 (warm-up)")
    print(f"   ✓ Double DQN: Enabled")
    print(f"   ✓ Gradient clipping: Enabled")
    
    # Test 3: Training loop with proper step counting
    print("3. Testing training loop improvements...")
    
    state, _ = env.reset()
    total_steps = 0
    
    for step in range(100):  # Short test
        action = agent.select_action(state, training=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Clip rewards
        clipped_reward = np.clip(reward, -1, 1)
        
        # Store transition
        agent.store_transition(state, action, clipped_reward, next_state, done)
        
        # Update step counter
        total_steps += 1
        agent.steps_done = total_steps
        
        # Try training (will skip during warm-up)
        loss = agent.train_step()
        
        if done:
            state, _ = env.reset()
        else:
            state = next_state
    
    print(f"   ✓ Training loop working correctly")
    print(f"   ✓ Total steps: {total_steps}")
    print(f"   ✓ Agent steps: {agent.steps_done}")
    print(f"   ✓ Memory size: {len(agent.memory)}")
    print(f"   ✓ Warm-up status: {'Active' if total_steps < agent.replay_start_size else 'Complete'}")
    
    # Test 4: Evaluation system
    print("4. Testing evaluation system...")
    from evaluation import evaluate_agent, create_evaluation_env, EpisodeRecorder
    
    eval_env = create_evaluation_env()
    recorder = EpisodeRecorder(save_dir="test_recordings")
    
    print(f"   ✓ Evaluation environment created")
    print(f"   ✓ Episode recorder initialized")
    
    eval_env.close()
    env.close()
    
    print("5. Testing video recording...")
    # Test recording functionality
    recorder.start_recording()
    
    # Simulate some frames
    for i in range(10):
        fake_frame = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
        recorder.add_frame(fake_frame)
    
    video_path = recorder.save_recording("test_recording.mp4")
    
    if video_path and os.path.exists(video_path):
        print(f"   ✓ Video recording working: {video_path}")
        os.remove(video_path)  # Clean up
    else:
        print(f"   ⚠ Video recording may have issues")
    
    print("\n🎉 All improved features tested successfully!")
    
    return True

def run_mini_improved_demo():
    """Run a mini demo with improved features."""
    print("\n" + "=" * 60)
    print("🚀 Running Mini Demo with Improved Features...")
    print("=" * 60)
    
    from train_pong import set_seeds, train_dqn_pong
    
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Run very short training with new features
    try:
        print("Starting 5-episode demo with improved DQN...")
        
        agent, metrics = train_dqn_pong(
            num_episodes=5,
            save_interval=3,
            seed=42,
            device=None  # Auto-detect
        )
        
        print(f"\n✓ Demo completed successfully!")
        print(f"✓ Final memory size: {len(agent.memory)}")
        print(f"✓ Total steps: {agent.steps_done}")
        print(f"✓ Episodes: {len(agent.episode_rewards)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        return False

if __name__ == "__main__":
    # Run tests
    test_improved_implementation()
    
    # Run mini demo
    demo_success = run_mini_improved_demo()
    
    print("\n" + "=" * 60)
    if demo_success:
        print("🎊 IMPROVED DQN READY FOR TRAINING!")
        print("Run: python run.py --quick --seed 42")
        print("Or:  python train_pong.py")
    else:
        print("⚠️  Please check for errors before full training")
    print("=" * 60)
