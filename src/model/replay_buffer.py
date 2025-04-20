import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """
    Replay buffer for storing and sampling experiences.
    """
    
    def __init__(self, capacity, state_dim, action_dim):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum capacity of the buffer
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def add(self, state, action, reward, next_state, done):
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Ensure action is properly shaped
        if isinstance(action, (int, float)):
            action = np.array([action])
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
        """
        # Ensure we don't sample more than buffer size
        batch_size = min(batch_size, len(self.buffer))
        
        # Sample random experiences
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32).reshape(-1, 1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)
