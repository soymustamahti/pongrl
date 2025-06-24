import gymnasium as gym
import ale_py
import numpy as np
import torch
import random
import os
import time
from collections import deque
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

from dqn_agent import DQNAgent
from visualization import MetricsTracker

# Register ALE environments  
gym.register_envs(ale_py)

# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train_dqn_pong(num_episodes=2000, 
                   save_interval=500,
                   model_save_path="models",
                   results_save_path="training_results",
                   seed=42,
                   device=None):
    """Train a DQN agent to play Pong.
    
    Args:
        num_episodes: Number of episodes to train for
        save_interval: Save model and generate report every N episodes
        model_save_path: Directory to save model checkpoints
        results_save_path: Directory to save training results
        seed: Random seed for reproducibility
        device: Device to use (cuda/cpu)
    """
    
    # Set seeds
    set_seeds(seed)
    
    # Create directories
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(results_save_path, exist_ok=True)
    
    # Initialize environment with proper Atari preprocessing
    env = gym.make('ALE/Pong-v5', render_mode=None, frameskip=1)
    
    # Apply Atari preprocessing wrappers
    env = AtariPreprocessing(
        env, 
        noop_max=30,           # Random no-ops at episode start
        frame_skip=4,          # Repeat each action for 4 frames
        screen_size=84,        # Resize to 84x84
        terminal_on_life_loss=True,  # Episode ends on life loss
        grayscale_obs=True,    # Convert to grayscale
        grayscale_newaxis=False,
        scale_obs=True         # Scale to [0,1]
    )
    
    # Stack 4 frames
    env = FrameStackObservation(env, stack_size=4)
    
    # Get environment info
    num_actions = env.action_space.n
    state_shape = env.observation_space.shape  # Should be (4, 84, 84)
    
    print(f"Environment: Pong with Atari preprocessing")
    print(f"Number of actions: {num_actions}")
    print(f"State shape: {state_shape}")
    print(f"Action space: {env.action_space}")
    
    # Initialize agent with improved hyperparameters
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = DQNAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        lr=0.0005,              # Increased learning rate
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=1000000,
        memory_size=100000,
        batch_size=32,
        target_update_freq=5000,  # Reduced frequency
        replay_start_size=50000,  # Warm-up period
        device=device
    )
    
    # Initialize metrics tracker
    metrics = MetricsTracker(save_dir=results_save_path)
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    total_steps = 0  # Global step counter for epsilon decay
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        # state is already preprocessed by wrappers - shape (4, 84, 84)
        
        episode_reward = 0
        episode_losses = []
        step_count = 0
        
        while True:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Clip rewards to [-1, 1] for training stability
            clipped_reward = np.clip(reward, -1, 1)
            
            # Store transition
            agent.store_transition(state, action, clipped_reward, next_state, done)
            
            # Update global step counter
            total_steps += 1
            agent.steps_done = total_steps  # Sync step counter
            
            # Train agent (only after warm-up period)
            if total_steps >= agent.replay_start_size:
                loss = agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
                    metrics.add_training_loss(loss, total_steps)
            
            # Update state and reward
            state = next_state
            episode_reward += reward  # Use original reward for tracking
            step_count += 1
            
            if done:
                break
        
        # Record episode metrics
        agent.episode_rewards.append(episode_reward)
        metrics.add_episode_reward(episode_reward)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:]) if len(agent.episode_rewards) >= 100 else np.mean(agent.episode_rewards)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            epsilon = agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * np.exp(-1. * total_steps / agent.epsilon_decay)
            
            elapsed_time = time.time() - start_time
            warmup_status = "WARMUP" if total_steps < agent.replay_start_size else "TRAINING"
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Avg Reward (100): {avg_reward:6.1f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Steps: {total_steps:7d} | "
                  f"Status: {warmup_status} | "
                  f"Time: {elapsed_time/60:.1f}m")
        
        # Evaluation and save model at intervals
        if (episode + 1) % save_interval == 0:
            # Save model
            model_filename = f"dqn_pong_episode_{episode+1}.pth"
            model_filepath = os.path.join(model_save_path, model_filename)
            agent.save_model(model_filepath)
            print(f"Model saved: {model_filepath}")
            
            # Run evaluation
            if total_steps >= agent.replay_start_size:
                from evaluation import evaluate_agent, create_evaluation_env, EpisodeRecorder
                
                eval_env = create_evaluation_env()
                recorder = EpisodeRecorder(save_dir=os.path.join(results_save_path, "recordings"))
                
                eval_results = evaluate_agent(agent, eval_env, num_episodes=5, 
                                            record_best=True, recorder=recorder)
                
                # Log evaluation results
                print(f"Evaluation at episode {episode+1}:")
                print(f"  Avg Evaluation Reward: {eval_results['avg_reward']:.2f}")
                print(f"  Best Evaluation Reward: {eval_results['best_reward']:.2f}")
                
                eval_env.close()
            
            # Generate report
            report_filename = f"training_report_episode_{episode+1}.pdf"
            metrics.generate_report(report_filename)
            
            # Save individual plots
            metrics.save_plots()
    
    # Final save and report
    final_model_path = os.path.join(model_save_path, "dqn_pong_final.pth")
    agent.save_model(final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    # Generate final report
    final_report = metrics.generate_report("final_training_report.pdf")
    print(f"Final training report: {final_report}")
    
    env.close()
    return agent, metrics

def evaluate_agent(model_path, num_episodes=10, render=True, record=True):
    """Evaluate a trained DQN agent.
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        record: Whether to record episodes
    """
    
    from evaluation import create_evaluation_env, EpisodeRecorder
    
    # Initialize environment
    if render:
        env = gym.make('ALE/Pong-v5', render_mode='human')
    else:
        env = create_evaluation_env()
    
    # Initialize agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(
        state_shape=(4, 84, 84),
        num_actions=env.action_space.n,
        device=device
    )
    
    # Load trained model
    agent.load_model(model_path)
    print(f"Loaded model from: {model_path}")
    
    # Initialize recorder
    recorder = None
    if record:
        recorder = EpisodeRecorder(save_dir="evaluation_recordings")
    
    # Evaluation
    from evaluation import evaluate_agent as eval_fn
    results = eval_fn(agent, env, num_episodes=num_episodes, 
                     record_best=record, recorder=recorder)
    
    env.close()
    return results

if __name__ == "__main__":
    # Train the agent
    print("Training DQN agent on Pong...")
    agent, metrics = train_dqn_pong(
        num_episodes=2000,
        save_interval=500,
        model_save_path="models",
        results_save_path="training_results"
    )
    
    print("\nTraining completed!")
    
    # Evaluate the final model
    print("\nEvaluating final model...")
    final_model_path = "models/dqn_pong_final.pth"
    if os.path.exists(final_model_path):
        evaluate_agent(final_model_path, num_episodes=5, render=False)
    else:
        print("Final model not found for evaluation.")
