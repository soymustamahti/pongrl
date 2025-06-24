# Deep Q-Learning Agent for Pong

This project implements a Deep Q-Network (DQN) agent that learns to play the Atari game Pong using PyTorch and reinforcement learning.

## Features

- **DQN Implementation**: Complete DQN with experience replay and target network
- **Epsilon-Greedy Exploration**: Balanced exploration and exploitation strategy
- **Frame Preprocessing**: Converts RGB frames to grayscale, resizes, and stacks frames
- **Automatic Visualization**: Generates training plots and PDF reports
- **Model Persistence**: Save and load trained models

## Project Structure

```
Artari game/
├── requirements.txt          # Python dependencies
├── dqn_model.py             # Neural network architecture
├── replay_memory.py         # Experience replay buffer
├── preprocessing.py         # Frame preprocessing pipeline
├── dqn_agent.py            # Main DQN agent implementation
├── visualization.py        # Metrics tracking and visualization
├── train_pong.py           # Training and evaluation script
├── README.md               # This file
├── models/                 # Saved model checkpoints (created during training)
└── training_results/       # Training plots and reports (created during training)
```

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Install additional Atari ROMs (if needed):

```bash
pip install "gymnasium[accept-rom-license]"
```

## Usage

### Training

To train a new DQN agent:

```bash
python train_pong.py
```

This will:

- Train the agent for 2000 episodes by default
- Save model checkpoints every 500 episodes in `models/`
- Generate training reports every 500 episodes in `training_results/`
- Create final visualizations and PDF report after training

### Evaluation

To evaluate a trained model:

```python
from train_pong import evaluate_agent

# Evaluate the final model
evaluate_agent("models/dqn_pong_final.pth", num_episodes=10, render=True)
```

## Architecture

### DQN Model (`dqn_model.py`)

- Convolutional neural network with 3 conv layers and 2 fully connected layers
- Input: 4 stacked 84x84 grayscale frames
- Output: Q-values for each possible action

### Agent (`dqn_agent.py`)

- Implements the DQN algorithm with:
  - Experience replay buffer
  - Target network updated every 10,000 steps
  - Epsilon-greedy exploration (ε decays from 1.0 to 0.1)
  - Adam optimizer with learning rate 0.0001

### Preprocessing (`preprocessing.py`)

- Converts RGB frames to grayscale
- Resizes frames to 84x84 pixels
- Normalizes pixel values to [0, 1]
- Stacks 4 consecutive frames for temporal information

### Visualization (`visualization.py`)

- Tracks episode rewards and training losses
- Generates matplotlib plots
- Creates comprehensive PDF reports with statistics

## Training Parameters

Default hyperparameters:

- Learning rate: 0.0001
- Discount factor (γ): 0.99
- Replay buffer size: 100,000
- Batch size: 32
- Target network update frequency: 10,000 steps
- Epsilon decay: 1,000,000 steps
- Frame stack: 4 frames
- Frame size: 84x84 pixels

## Results

The training process automatically generates:

1. **Episode Rewards Plot**: Shows reward progression over episodes
2. **Training Loss Plot**: Shows learning curve of the neural network
3. **PDF Report**: Combined report with plots and statistics
4. **Model Checkpoints**: Saved every 500 episodes for resuming training

## Performance Expectations

- Initial episodes: Random performance (negative rewards)
- After ~500-1000 episodes: Agent starts learning basic game mechanics
- After ~1500+ episodes: Agent should show consistent improvement
- Well-trained agent: Should achieve positive average rewards

## Customization

You can modify training parameters by editing the values in `train_pong.py`:

```python
agent = DQNAgent(
    lr=0.0001,           # Learning rate
    gamma=0.99,          # Discount factor
    epsilon_decay=1000000,  # Exploration decay rate
    memory_size=100000,  # Replay buffer size
    batch_size=32,       # Training batch size
    # ... other parameters
)
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Gymnasium with Atari support
- OpenCV
- Matplotlib
- NumPy

## Notes

- Training can take several hours depending on hardware
- GPU acceleration is recommended for faster training
- The agent uses deterministic evaluation (no exploration) during testing
- Model checkpoints allow resuming training from any saved point
