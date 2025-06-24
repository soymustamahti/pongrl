import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import os
from datetime import datetime

from dqn_model import DQN
from replay_memory import ReplayMemory
from preprocessing import AtariPreprocessing

class DQNAgent:
    """Deep Q-Network agent for playing Atari games.
    
    Implements the DQN algorithm with experience replay, target network,
    and epsilon-greedy exploration.
    """
    
    def __init__(self, 
                 state_shape=(4, 84, 84),
                 num_actions=6,
                 lr=0.0005,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay=1000000,
                 memory_size=100000,
                 batch_size=32,
                 target_update_freq=5000,
                 replay_start_size=50000,
                 double_dqn=True,
                 grad_clip=True,
                 device=None):
        """Initialize the DQN agent.
        
        Args:
            state_shape: Shape of the input state (channels, height, width)
            num_actions: Number of possible actions
            lr: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Number of steps for epsilon decay
            memory_size: Size of the replay memory
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            replay_start_size: Steps before starting training
            double_dqn: Whether to use Double DQN
            grad_clip: Whether to clip gradients
            device: Device to run the model on
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.replay_start_size = replay_start_size
        self.double_dqn = double_dqn
        self.grad_clip = grad_clip
        
        # Initialize networks
        self.q_network = DQN(state_shape[0], num_actions).to(self.device)
        self.target_network = DQN(state_shape[0], num_actions).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize replay memory
        self.memory = ReplayMemory(memory_size)
        
        # Training statistics
        self.steps_done = 0
        self.episode_rewards = []
        self.training_losses = []
        
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action
        """
        if training:
            # Calculate current epsilon
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                     np.exp(-1. * self.steps_done / self.epsilon_decay)
            
            if random.random() > epsilon:
                # Exploit: choose best action
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor)
                    return q_values.max(1)[1].item()
            else:
                # Explore: choose random action
                return random.randrange(self.num_actions)
        else:
            # Evaluation mode: always exploit
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.max(1)[1].item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in replay memory."""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step."""
        if not self.memory.can_provide_sample(self.batch_size):
            return None
        
        # Don't train until warm-up period is complete
        if self.steps_done < self.replay_start_size:
            return None
        
        # Sample batch from memory
        transitions = self.memory.sample(self.batch_size)
        
        # Convert batch to tensors
        states = torch.FloatTensor(np.array([t.state for t in transitions])).to(self.device)
        actions = torch.LongTensor([t.action for t in transitions]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in transitions]).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in transitions])).to(self.device)
        dones = torch.BoolTensor([t.done for t in transitions]).to(self.device)
        
        # Compute Q(s_t, a)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use main network to select action, target network to evaluate
                next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze()
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1)[0]
            
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save_model(self, filepath):
        """Save model weights and training state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episode_rewards': self.episode_rewards,
            'training_losses': self.training_losses
        }, filepath)
    
    def load_model(self, filepath):
        """Load model weights and training state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.episode_rewards = checkpoint['episode_rewards']
        self.training_losses = checkpoint['training_losses']
