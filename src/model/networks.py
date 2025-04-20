import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorNetwork(nn.Module):
    """
    Actor network for DDPG agent.
    Maps states to deterministic actions.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super(ActorNetwork, self).__init__()
        
        # Create hidden layers
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            action: Output action tensor (tanh activated to bound between -1 and 1)
        """
        x = self.layers(state)
        # Use tanh to bound actions between -1 and 1
        return torch.tanh(self.output_layer(x))
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

class CriticNetwork(nn.Module):
    """
    Critic network for DDPG agent.
    Maps state-action pairs to Q-values.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super(CriticNetwork, self).__init__()
        
        # First hidden layer processes only the state
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        
        # Second hidden layer processes both state and action
        self.fc2 = nn.Linear(hidden_dims[0] + action_dim, hidden_dims[1])
        
        # Output layer
        self.fc3 = nn.Linear(hidden_dims[1], 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, state, action):
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            action: Input action tensor
            
        Returns:
            q_value: Estimated Q-value
        """
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
