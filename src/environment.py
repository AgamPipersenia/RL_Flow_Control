import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import gym
from gym import spaces

import lbm_simulation as lbm
import config

class FlowControlEnv(gym.Env):
    """
    Reinforcement Learning environment for active flow control of a cylinder wake.
    
    This environment wraps the LBM simulation and provides an interface for RL agents
    to control the jet strength and receive observations and rewards.
    """
    
    def __init__(self):
        super(FlowControlEnv, self).__init__()
        
        # Initialize simulation parameters
        self.npointsx = lbm.N_POINTS_X
        self.npointsy = lbm.N_POINTS_Y
        self.cylinder_x = lbm.CYLINDER_CENTER_INDEX_X
        self.cylinder_y = lbm.CYLINDER_CENTER_INDEX_Y
        self.cylinder_radius = lbm.CYLINDER_RADIUS_INDICES
        
        # RL parameters
        self.max_jet_strength = config.MAX_JET_STRENGTH
        self.action_dim = config.ACTION_DIM
        self.state_dim = config.STATE_DIM
        self.reward_weights = config.REWARD_WEIGHTS
        self.steps_per_action = config.STEPS_PER_ACTION
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Initialize history for metrics calculation
        self.vorticity_history = []
        self.drag_history = []
        self.lift_history = []
        
        # Initialize simulation state
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state."""
        # Initialize the flow field
        self.discrete_velocities = lbm.initialize_flow()
        
        # Run a few steps to establish flow
        for _ in range(config.INITIAL_STEPS):
            self.discrete_velocities = lbm.update(self.discrete_velocities, jet_strength=0.0)
        
        # Reset counters and history
        self.step_count = 0
        self.vorticity_history = []
        self.drag_history = []
        self.lift_history = []
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Normalized jet strength in range [-1, 1]
            
        Returns:
            observation: Current state observation
            reward: Reward for the current step
            done: Whether the episode is done
            info: Additional information
        """
        # Process action (convert from normalized range to actual jet strength)
        jet_strength = self._process_action(action)
        
        # Run simulation for several steps with this jet strength
        for _ in range(self.steps_per_action):
            self.discrete_velocities = lbm.update(self.discrete_velocities, jet_strength)
            self.step_count += 1
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate metrics
        density = lbm.get_density(self.discrete_velocities)
        velocities = lbm.get_macroscopic_velocities(self.discrete_velocities, density)
        drag_coef, lift_coef = lbm.calculate_forces(self.discrete_velocities)
        wake_metric = lbm.get_wake_metric(velocities)
        
        # Store history for metrics
        self.drag_history.append(drag_coef)
        self.lift_history.append(lift_coef)
        
        # Calculate reward
        reward = self._calculate_reward(drag_coef, lift_coef, wake_metric, jet_strength)
        
        # Check if episode is done
        done = self.step_count >= config.MAX_EPISODE_STEPS
        
        # Additional info
        info = {
            'drag': float(drag_coef),
            'lift': float(lift_coef),
            'wake_metric': float(wake_metric),
            'jet_strength': float(jet_strength)
        }
        
        return observation, reward, done, info
    
    def render(self, mode='human'):
        """Render the current state of the environment."""
        density = lbm.get_density(self.discrete_velocities)
        velocities = lbm.get_macroscopic_velocities(self.discrete_velocities, density)
        drag_coef, lift_coef = lbm.calculate_forces(self.discrete_velocities)
        wake_metric = lbm.get_wake_metric(velocities)
        
        info = {
            'drag': float(drag_coef),
            'lift': float(lift_coef),
            'wake_metric': float(wake_metric)
        }
        
        fig = lbm.visualize_flow(self.discrete_velocities, info=info)
        
        if mode == 'human':
            plt.draw()
            plt.pause(0.01)
            return None
        elif mode == 'rgb_array':
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return data
    
    def _get_observation(self):
        """
        Extract relevant flow features as state representation.
        
        Returns:
            observation: A vector of flow features
        """
        density = lbm.get_density(self.discrete_velocities)
        velocities = lbm.get_macroscopic_velocities(self.discrete_velocities, density)
        curl = lbm.calculate_vorticity(velocities)
        
        # Extract features around and behind the cylinder
        # 1. Velocity probes behind the cylinder
        x_probes = [
            int(self.cylinder_x + self.cylinder_radius * 1.5),
            int(self.cylinder_x + self.cylinder_radius * 3),
            int(self.cylinder_x + self.cylinder_radius * 5)
        ]
        
        y_probes = np.linspace(
            self.cylinder_y - self.cylinder_radius,
            self.cylinder_y + self.cylinder_radius,
            5
        ).astype(int)
        
        # Extract velocity and vorticity at probe points
        features = []
        for x in x_probes:
            for y in y_probes:
                # Ensure indices are within bounds
                if 0 <= x < self.npointsx and 0 <= y < self.npointsy:
                    features.append(float(velocities[x, y, 0]))  # u
                    features.append(float(velocities[x, y, 1]))  # v
                    features.append(float(curl[x, y]))           # vorticity
                else:
                    # Add default values if out of bounds
                    features.append(0.0)
                    features.append(0.0)
                    features.append(0.0)
        
        # 2. Pressure (density) around the cylinder
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for angle in angles:
            x = int(self.cylinder_x + (self.cylinder_radius + 1) * np.cos(angle))
            y = int(self.cylinder_y + (self.cylinder_radius + 1) * np.sin(angle))
            # Ensure indices are within bounds
            if 0 <= x < self.npointsx and 0 <= y < self.npointsy:
                features.append(float(density[x, y]))
            else:
                features.append(1.0)  # Default density value
        
        # 3. Global flow metrics
        drag_coef, lift_coef = lbm.calculate_forces(self.discrete_velocities)
        features.append(float(drag_coef))
        features.append(float(lift_coef))
        
        # 4. Recent history of drag and lift (if available)
        if len(self.drag_history) > 0:
            features.append(float(np.mean(self.drag_history[-5:])))
            features.append(float(np.mean(self.lift_history[-5:])))
        else:
            features.append(float(drag_coef))
            features.append(float(lift_coef))
        
        # Convert to numpy array and ensure correct length
        features = np.array(features, dtype=np.float32)
        
        # Pad or truncate to match state_dim
        if len(features) < self.state_dim:
            features = np.pad(features, (0, self.state_dim - len(features)))
        elif len(features) > self.state_dim:
            features = features[:self.state_dim]
        
        return features
    
    def _process_action(self, action):
        """Convert normalized action to actual jet strength."""
        # Ensure action is a numpy array
        if isinstance(action, (list, tuple)):
            action = np.array(action)
        elif isinstance(action, jnp.ndarray):
            action = np.array(action)
        
        # Extract the first element if it's an array with shape (1,)
        if hasattr(action, 'shape') and action.shape == (1,):
            action = action[0]
        
        # Scale from [-1, 1] to [0, max_jet_strength]
        return float(action * self.max_jet_strength)
    
    def _calculate_reward(self, drag_coef, lift_coef, wake_metric, jet_strength):
        """
        Calculate reward based on drag, lift, wake stability, and control effort.
        
        Args:
            drag_coef: Drag coefficient
            lift_coef: Lift coefficient
            wake_metric: Wake stability metric
            jet_strength: Applied jet strength
            
        Returns:
            reward: Combined reward value
        """
        # Negative drag is good (we want to minimize drag)
        drag_reward = -drag_coef * self.reward_weights['drag']
        
        # Lift oscillation reduction is good
        if len(self.lift_history) > 1:
            lift_oscillation = np.abs(lift_coef - self.lift_history[-1])
            lift_reward = -lift_oscillation * self.reward_weights['lift']
        else:
            lift_reward = 0
        
        # Wake stability reward
        wake_reward = wake_metric * self.reward_weights['wake']
        
        # Penalize excessive control effort
        control_penalty = -np.abs(jet_strength) * self.reward_weights['control']
        
        # Combine rewards
        reward = drag_reward + lift_reward + wake_reward + control_penalty
        
        return float(reward)
