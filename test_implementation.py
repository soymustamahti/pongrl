#!/usr/bin/env python3
"""
Simple test script to verify the DQN implementation is working correctly.
This runs a very short training session and checks that all components work.
"""

import gymnasium as gym
import ale_py
import numpy as np
import torch
import os

# Register ALE environments
gym.register_envs(ale_py)

from dqn_agent import DQNAgent
from preprocessing import AtariPreprocessing
from visualization import MetricsTracker

def test_implementation():
    """Test that all components work together."""
    print("Testing DQN implementation...")
    
    # Test environment creation
    print("1. Testing environment creation...")
    env = gym.make('ALE/Pong-v5', render_mode=None)
    print(f"   ✓ Environment created successfully")
    print(f"   ✓ Action space: {env.action_space}")
    print(f"   ✓ Observation space: {env.observation_space}")
    
    # Test preprocessing
    print("2. Testing frame preprocessing...")
    preprocessor = AtariPreprocessing(frame_stack=4, frame_size=(84, 84))
    obs, _ = env.reset()
    processed_state = preprocessor.process_frame(obs)
    print(f"   ✓ Original frame shape: {obs.shape}")
    print(f"   ✓ Processed state shape: {processed_state.shape}")
    
    # Test agent creation
    print("3. Testing agent creation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(
        state_shape=(4, 84, 84),
        num_actions=env.action_space.n,
        lr=0.0001,
        device=device
    )
    print(f"   ✓ Agent created successfully on {device}")
    
    # Test action selection
    print("4. Testing action selection...")
    action = agent.select_action(processed_state, training=True)
    print(f"   ✓ Action selected: {action}")
    
    # Test environment step
    print("5. Testing environment interaction...")
    next_obs, reward, terminated, truncated, info = env.step(action)
    print(f"   ✓ Environment step successful")
    print(f"   ✓ Reward: {reward}")
    print(f"   ✓ Episode ended: {terminated or truncated}")
    
    # Test memory storage
    print("6. Testing replay memory...")
    next_processed_state = preprocessor.process_frame(next_obs)
    agent.store_transition(processed_state, action, reward, next_processed_state, terminated or truncated)
    print(f"   ✓ Transition stored in replay memory")
    print(f"   ✓ Memory size: {len(agent.memory)}")
    
    # Test metrics tracking
    print("7. Testing metrics tracking...")
    metrics = MetricsTracker(save_dir="test_results")
    metrics.add_episode_reward(-21.0)  # Typical initial Pong score
    print(f"   ✓ Metrics tracking working")
    
    env.close()
    print("\n🎉 All tests passed! DQN implementation is ready for training.")
    
    return True

def mini_training_demo():
    """Run a very short training demo (5 episodes)."""
    print("\n" + "="*50)
    print("Running mini training demo (5 episodes)...")
    print("="*50)
    
    # Initialize environment and components
    env = gym.make('ALE/Pong-v5', render_mode=None)
    preprocessor = AtariPreprocessing(frame_stack=4, frame_size=(84, 84))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(
        state_shape=(4, 84, 84),
        num_actions=env.action_space.n,
        lr=0.0001,
        device=device
    )
    
    metrics = MetricsTracker(save_dir="test_results")
    
    # Mini training loop
    for episode in range(5):
        state, _ = env.reset()
        preprocessor.reset()
        state = preprocessor.process_frame(state)
        
        episode_reward = 0
        step_count = 0
        losses = []
        
        while True:
            # Select and take action
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Process next state
            next_state = preprocessor.process_frame(next_state)
            
            # Store transition and train
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
                metrics.add_training_loss(loss, agent.steps_done)
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
            if done:
                break
        
        # Record episode metrics
        agent.episode_rewards.append(episode_reward)
        metrics.add_episode_reward(episode_reward)
        
        avg_loss = np.mean(losses) if losses else 0
        print(f"Episode {episode + 1:2d} | "
              f"Reward: {episode_reward:6.1f} | "
              f"Steps: {step_count:3d} | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Memory: {len(agent.memory):4d}")
    
    env.close()
    
    # Show final statistics
    print(f"\nDemo completed!")
    print(f"Average reward: {np.mean(agent.episode_rewards):.2f}")
    print(f"Total training steps: {agent.steps_done}")
    print(f"Memory size: {len(agent.memory)}")
    
    # Generate quick report
    if len(agent.episode_rewards) > 0:
        try:
            report_path = metrics.generate_report("mini_demo_report.pdf")
            print(f"Report generated: {report_path}")
        except Exception as e:
            print(f"Note: Could not generate PDF report ({e})")
    
    return True

if __name__ == "__main__":
    # Run tests
    test_implementation()
    
    # Run mini demo
    mini_training_demo()
    
    print("\n" + "="*60)
    print("🚀 Ready to start full training!")
    print("Run: python run.py --quick        (for 100 episodes)")
    print("Run: python train_pong.py         (for full 2000 episodes)")
    print("="*60)
