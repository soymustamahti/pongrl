import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """Deep Q-Network for Atari games.
    
    A convolutional neural network that takes preprocessed game frames
    and outputs Q-values for each possible action.
    """
    
    def __init__(self, input_channels=4, num_actions=6):
        """Initialize the DQN.
        
        Args:
            input_channels: Number of stacked frames (default: 4)
            num_actions: Number of possible actions in the environment
        """
        super(DQN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of flattened features
        # For 84x84 input: (84-8)//4+1 = 20, (20-4)//2+1 = 9, (9-3)//1+1 = 7
        # So conv output is 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Q-values for each action
        """
        # Apply convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
