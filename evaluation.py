import gymnasium as gym
import ale_py
import torch
import numpy as np
import cv2
import os
from datetime import datetime

# Register ALE environments
gym.register_envs(ale_py)

class EpisodeRecorder:
    """Records episodes as video files."""
    
    def __init__(self, save_dir="recordings", fps=30):
        self.save_dir = save_dir
        self.fps = fps
        self.frames = []
        self.recording = False
        os.makedirs(save_dir, exist_ok=True)
    
    def start_recording(self):
        """Start recording frames."""
        self.frames = []
        self.recording = True
    
    def add_frame(self, frame):
        """Add a frame to the recording."""
        if self.recording:
            # Convert from RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.frames.append(frame_bgr)
    
    def save_recording(self, filename=None):
        """Save the recorded frames as a video."""
        if not self.frames:
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"episode_{timestamp}.mp4"
        
        filepath = os.path.join(self.save_dir, filename)
        
        # Get frame dimensions
        height, width = self.frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, self.fps, (width, height))
        
        # Write frames
        for frame in self.frames:
            out.write(frame)
        
        out.release()
        self.recording = False
        
        print(f"Episode recorded: {filepath}")
        return filepath

def evaluate_agent(agent, env, num_episodes=5, record_best=True, recorder=None):
    """Evaluate agent performance without exploration."""
    
    episode_rewards = []
    best_reward = float('-inf')
    best_episode_frames = None
    
    print(f"Evaluating agent for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_frames = []
        
        # Start recording if this might be the best episode
        if record_best and recorder:
            recorder.start_recording()
        
        while True:
            # Get raw frame for recording before preprocessing
            if record_best:
                # Get the original frame for recording
                try:
                    raw_frame = env.unwrapped.ale.getScreenRGB()
                    episode_frames.append(raw_frame)
                    if recorder and recorder.recording:
                        recorder.add_frame(raw_frame)
                except:
                    pass
            
            # Select action (no exploration)
            action = agent.select_action(state, training=False)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Save recording if this is the best episode so far
        if record_best and episode_reward > best_reward:
            best_reward = episode_reward
            if recorder and recorder.recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"best_episode_reward_{best_reward:.1f}_{timestamp}.mp4"
                recorder.save_recording(filename)
        
        print(f"  Episode {episode + 1}: Reward = {episode_reward:.1f}")
    
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"\nEvaluation Results:")
    print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Best Reward: {max(episode_rewards):.2f}")
    print(f"  Episode Rewards: {episode_rewards}")
    
    return {
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'best_reward': max(episode_rewards),
        'all_rewards': episode_rewards
    }

def create_evaluation_env():
    """Create environment for evaluation (with rendering for recording)."""
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
    
    env = gym.make('ALE/Pong-v5', render_mode='rgb_array', frameskip=1)
    
    # Apply same preprocessing as training
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
    
    return env
