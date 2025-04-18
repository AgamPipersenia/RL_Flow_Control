import gymnasium as gym
from gymnasium.spaces import Box
import jax
import jax.numpy as jnp
import numpy as np

from LBM_ import (
    get_density, get_macroscopic_velocities, get_equilibrium_discrete_velocities,
    N_POINTS_X, N_POINTS_Y, CYLINDER_CENTER_INDEX_X, CYLINDER_CENTER_INDEX_Y,
    CYLINDER_RADIUS_INDICES, MAX_HORIZONTAL_INFLOW_VELOCITY, LATTICE_WEIGHTS,
    LATTICE_VELOCITIES, RIGHT_VELOCITIES, LEFT_VELOCITIES, UP_VELOCITIES,
    DOWN_VELOCITIES, PURE_VERTICAL_VELOCITIES, OPPOSITE_LATTICE_INDICES,
    LATTICE_INDICES, REYNOLDS_NUMBER, update,
    kinematic_viscosity, relaxation_omega, velocity_profile, X, Y, obstacle_mask,
    JET_THETA, JET_WIDTH, JET_X, JET_Y, jet_mask
)

class FlowControlEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = Box(low=-0.1, high=0.1, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        
        self.kinematic_viscosity = kinematic_viscosity
        self.relaxation_omega = relaxation_omega
        self.velocity_profile = velocity_profile
        self.X = X
        self.Y = Y
        self.obstacle_mask = obstacle_mask
        self.JET_THETA = JET_THETA
        self.JET_WIDTH = JET_WIDTH
        self.JET_X = JET_X
        self.JET_Y = JET_Y
        self.jet_mask = jet_mask
        
        self.step_count = 0
        self.max_steps = 1000

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.discrete_velocities = get_equilibrium_discrete_velocities(
            self.velocity_profile, jnp.ones((N_POINTS_X, N_POINTS_Y))
        )
        return self._get_state(), {}

    def step(self, action):
        self.discrete_velocities = update(self.discrete_velocities, action[0])
        self.step_count += 1
        
        state = self._get_state()
        reward = self._compute_reward()
        done = self.step_count >= self.max_steps
        
        return state, reward, done, False, {}

    def _get_state(self):
        density = get_density(self.discrete_velocities)
        velocities = get_macroscopic_velocities(self.discrete_velocities, density)
        x_range = slice(CYLINDER_CENTER_INDEX_X-10, CYLINDER_CENTER_INDEX_X+10)
        y_pos = CYLINDER_CENTER_INDEX_Y + CYLINDER_RADIUS_INDICES + 2
        return velocities[x_range, y_pos, 0]

    def _compute_reward(self):
        density = get_density(self.discrete_velocities)
        velocities = get_macroscopic_velocities(self.discrete_velocities, density)
        d_u__d_y = jnp.gradient(velocities[..., 0], axis=1)
        drag = jnp.mean(d_u__d_y[CYLINDER_CENTER_INDEX_X, CYLINDER_CENTER_INDEX_Y:CYLINDER_CENTER_INDEX_Y + CYLINDER_RADIUS_INDICES])
        return -drag