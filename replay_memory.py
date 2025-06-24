import random
import numpy as np
from collections import deque, namedtuple

# Define transition tuple
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory:
    """Experience replay buffer for storing and sampling transitions.
    
    Stores transitions from the environment and allows random sampling
    of mini-batches for training the DQN.
    """
    
    def __init__(self, capacity=100000):
        """Initialize the replay memory.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Store a transition in the replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode ended
        """
        transition = Transition(state, action, reward, next_state, done)
        self.memory.append(transition)
        
    def sample(self, batch_size):
        """Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of sampled transitions
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def can_provide_sample(self, batch_size):
        """Check if we have enough samples for a batch."""
        return len(self.memory) >= batch_size
