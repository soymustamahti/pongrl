import cv2
import numpy as np
import gymnasium as gym
from collections import deque

class AtariPreprocessing:
    """Preprocessing pipeline for Atari game frames.
    
    Converts game frames to grayscale, resizes them, normalizes pixel values,
    and stacks consecutive frames to capture temporal information.
    """
    
    def __init__(self, frame_stack=4, frame_size=(84, 84)):
        """Initialize the preprocessing pipeline.
        
        Args:
            frame_stack: Number of consecutive frames to stack
            frame_size: Target size for resized frames (width, height)
        """
        self.frame_stack = frame_stack
        self.frame_size = frame_size
        self.frames = deque(maxlen=frame_stack)
        
    def reset(self):
        """Reset the frame buffer."""
        self.frames.clear()
        
    def preprocess_frame(self, frame):
        """Preprocess a single frame.
        
        Args:
            frame: Raw RGB frame from the environment
            
        Returns:
            Preprocessed grayscale frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to target size
        resized = cv2.resize(gray, self.frame_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1] range
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def process_frame(self, frame):
        """Process frame and add to stack.
        
        Args:
            frame: Raw frame from environment
            
        Returns:
            Stacked frames as numpy array
        """
        processed = self.preprocess_frame(frame)
        self.frames.append(processed)
        
        # If we don't have enough frames yet, repeat the current frame
        while len(self.frames) < self.frame_stack:
            self.frames.append(processed)
            
        return np.array(self.frames)
    
    def get_state(self):
        """Get current stacked state.
        
        Returns:
            Current state as stacked frames
        """
        return np.array(self.frames) if len(self.frames) == self.frame_stack else None
