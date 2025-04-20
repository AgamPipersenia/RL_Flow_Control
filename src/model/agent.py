import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model.networks import ActorNetwork, CriticNetwork
from model.replay_buffer import ReplayBuffer
import config

class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) agent for continuous control.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], buffer_capacity=100000,
                 batch_size=64, gamma=0.99, tau=0.005, lr_actor=0.0001, lr_critic=0.001):
        """
        Initialize the DDPG agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Dimensions of hidden layers
            buffer_capacity: Capacity of replay buffer
            batch_size: Batch size for training
            gamma: Discount factor
            tau: Soft update parameter
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize actor networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Initialize critic networks
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)
        
        # Initialize exploration noise
        self.noise_scale = config.NOISE_SCALE
        self.noise_decay = config.NOISE_DECAY
        self.min_noise_scale = config.MIN_NOISE_SCALE
        
        # Training metrics
        self.actor_losses = []
        self.critic_losses = []
        self.update_count = 0
    
    def select_action(self, state, add_noise=True):
        """
        Select action based on current policy.
        
        Args:
            state: Current state
            add_noise: Whether to add exploration noise
            
        Returns:
            action: Selected action
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Set actor to evaluation mode
        self.actor.eval()
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        
        # Set actor back to training mode
        self.actor.train()
        
        # Add exploration noise if in training mode
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=self.action_dim)
            action = np.clip(action + noise, -1.0, 1.0)
            
            # Decay noise scale
            self.noise_scale = max(self.min_noise_scale, self.noise_scale * self.noise_decay)
        
        return action
    
    def update(self):
        """Update actor and critic networks."""
        # Sample batch from replay buffer
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        # Store losses
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.update_count += 1
    
    def _soft_update(self, local_model, target_model):
        """
        Soft update target network parameters.
        
        θ_target = τ*θ_local + (1-τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, filepath):
        """Save model parameters."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'update_count': self.update_count,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'noise_scale': self.noise_scale
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model parameters."""
        if not os.path.exists(filepath):
            print(f"No model found at {filepath}")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.update_count = checkpoint['update_count']
        self.actor_losses = checkpoint['actor_losses']
        self.critic_losses = checkpoint['critic_losses']
        self.noise_scale = checkpoint['noise_scale']
        print(f"Model loaded from {filepath}")
