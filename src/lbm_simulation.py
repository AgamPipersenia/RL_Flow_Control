import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cmasher as cmr
import numpy as np
from tqdm import tqdm

# Existing constants (unchanged)
N_ITERATIONS = 15_000
REYNOLDS_NUMBER = 80
N_POINTS_X = 300
N_POINTS_Y = 50
CYLINDER_CENTER_INDEX_X = N_POINTS_X // 5
CYLINDER_CENTER_INDEX_Y = N_POINTS_Y // 2
CYLINDER_RADIUS_INDICES = N_POINTS_Y // 9
MAX_HORIZONTAL_INFLOW_VELOCITY = 0.04
VISUALIZE = True
PLOT_EVERY_N_STEPS = 100
SKIP_FIRST_N_ITERATIONS = 5000
N_DISCRETE_VELOCITIES = 9
LATTICE_VELOCITIES = jnp.array([
    [ 0,  1,  0, -1,  0,  1, -1, -1,  1],
    [ 0,  0,  1,  0, -1,  1,  1, -1, -1]
])
LATTICE_INDICES = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
OPPOSITE_LATTICE_INDICES = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
LATTICE_WEIGHTS = jnp.array([
    4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36
])
RIGHT_VELOCITIES = jnp.array([1, 5, 8])
UP_VELOCITIES = jnp.array([2, 5, 6])
LEFT_VELOCITIES = jnp.array([3, 6, 7])
DOWN_VELOCITIES = jnp.array([4, 7, 8])
PURE_VERTICAL_VELOCITIES = jnp.array([0, 2, 4])
PURE_HORIZONTAL_VELOCITIES = jnp.array([0, 1, 3])

# Global variables for update
kinematic_viscosity = (
    (MAX_HORIZONTAL_INFLOW_VELOCITY * CYLINDER_RADIUS_INDICES) / REYNOLDS_NUMBER
)
relaxation_omega = 1.0 / (3.0 * kinematic_viscosity + 0.5)
X = jnp.arange(N_POINTS_X)
Y = jnp.arange(N_POINTS_Y)
X, Y = jnp.meshgrid(X, Y, indexing="ij")
obstacle_mask = (
    jnp.sqrt(
        (X - CYLINDER_CENTER_INDEX_X)**2 +
        (Y - CYLINDER_CENTER_INDEX_Y)**2
    ) < CYLINDER_RADIUS_INDICES
)
velocity_profile = jnp.zeros((N_POINTS_X, N_POINTS_Y, 2))
velocity_profile = velocity_profile.at[:, :, 0].set(MAX_HORIZONTAL_INFLOW_VELOCITY)

# Jet configuration
JET_THETA = jnp.pi / 2
JET_WIDTH = 3
JET_X = CYLINDER_CENTER_INDEX_X + CYLINDER_RADIUS_INDICES * jnp.cos(JET_THETA)
JET_Y = CYLINDER_CENTER_INDEX_Y + CYLINDER_RADIUS_INDICES * jnp.sin(JET_THETA)
jet_mask = (
    (jnp.abs(X - JET_X) < JET_WIDTH / 2) &
    (jnp.abs(Y - JET_Y) < JET_WIDTH / 2) &
    (jnp.sqrt((X - CYLINDER_CENTER_INDEX_X)**2 + (Y - CYLINDER_CENTER_INDEX_Y)**2) <= CYLINDER_RADIUS_INDICES)
)

def get_density(discrete_velocities):
    density = jnp.sum(discrete_velocities, axis=-1)
    return density

def get_macroscopic_velocities(discrete_velocities, density):
    macroscopic_velocities = jnp.einsum(
        "NMQ,dQ->NMd",
        discrete_velocities,
        LATTICE_VELOCITIES,
    ) / density[..., jnp.newaxis]
    return macroscopic_velocities

def get_equilibrium_discrete_velocities(macroscopic_velocities, density):
    projected_discrete_velocities = jnp.einsum(
        "dQ,NMd->NMQ",
        LATTICE_VELOCITIES,
        macroscopic_velocities,
    )
    macroscopic_velocity_magnitude = jnp.linalg.norm(
        macroscopic_velocities, axis=-1, ord=2
    )
    equilibrium_discrete_velocities = (
        density[..., jnp.newaxis]
        * LATTICE_WEIGHTS[jnp.newaxis, jnp.newaxis, :]
        * (
            1
            + 3 * projected_discrete_velocities
            + 9/2 * projected_discrete_velocities**2
            - 3/2 * macroscopic_velocity_magnitude[..., jnp.newaxis]**2
        )
    )
    return equilibrium_discrete_velocities

def calculate_forces(discrete_velocities):
    """Calculate drag and lift forces on the cylinder."""
    density = get_density(discrete_velocities)
    velocities = get_macroscopic_velocities(discrete_velocities, density)
    
    # Get points around the cylinder (one grid point away from surface)
    cylinder_boundary = jnp.zeros_like(obstacle_mask)
    for i in range(N_POINTS_X):
        for j in range(N_POINTS_Y):
            dist = jnp.sqrt((i - CYLINDER_CENTER_INDEX_X)**2 + (j - CYLINDER_CENTER_INDEX_Y)**2)
            cylinder_boundary = cylinder_boundary.at[i, j].set(
                (dist >= CYLINDER_RADIUS_INDICES) & 
                (dist <= CYLINDER_RADIUS_INDICES + 1)
            )
    
    # Calculate pressure around cylinder
    pressure = density / 3.0  # p = rho * cs^2, where cs^2 = 1/3 in lattice units
    
    # Calculate normal vectors at boundary points
    normals = jnp.zeros((N_POINTS_X, N_POINTS_Y, 2))
    for i in range(N_POINTS_X):
        for j in range(N_POINTS_Y):
            if cylinder_boundary[i, j]:
                nx = (i - CYLINDER_CENTER_INDEX_X) / jnp.sqrt((i - CYLINDER_CENTER_INDEX_X)**2 + (j - CYLINDER_CENTER_INDEX_Y)**2)
                ny = (j - CYLINDER_CENTER_INDEX_Y) / jnp.sqrt((i - CYLINDER_CENTER_INDEX_X)**2 + (j - CYLINDER_CENTER_INDEX_Y)**2)
                normals = normals.at[i, j, 0].set(nx)
                normals = normals.at[i, j, 1].set(ny)
    
    # Calculate forces (simplified model)
    drag = jnp.sum(pressure[cylinder_boundary] * normals[cylinder_boundary, 0])
    lift = jnp.sum(pressure[cylinder_boundary] * normals[cylinder_boundary, 1])
    
    # Calculate coefficients
    drag_coef = drag / (0.5 * MAX_HORIZONTAL_INFLOW_VELOCITY**2 * 2 * CYLINDER_RADIUS_INDICES)
    lift_coef = lift / (0.5 * MAX_HORIZONTAL_INFLOW_VELOCITY**2 * 2 * CYLINDER_RADIUS_INDICES)
    
    return drag_coef, lift_coef

def calculate_vorticity(velocities):
    """Calculate vorticity field from velocity field."""
    d_u__d_x, d_u__d_y = jnp.gradient(velocities[..., 0])
    d_v__d_x, d_v__d_y = jnp.gradient(velocities[..., 1])
    curl = (d_u__d_y - d_v__d_x)
    return curl

def get_wake_metric(velocities):
    """Calculate a metric for wake oscillation/stability."""
    # Extract a line of points behind the cylinder
    x_probe = CYLINDER_CENTER_INDEX_X + 2 * CYLINDER_RADIUS_INDICES
    y_range = jnp.arange(CYLINDER_CENTER_INDEX_Y - CYLINDER_RADIUS_INDICES, 
                         CYLINDER_CENTER_INDEX_Y + CYLINDER_RADIUS_INDICES + 1)
    
    # Calculate velocity fluctuation in the wake
    v_fluctuation = jnp.std(velocities[x_probe, y_range, 1])
    
    # Higher fluctuation means more unstable wake (negative reward)
    return -v_fluctuation

@jax.jit
def update(discrete_velocities_prev, jet_strength):
    # (1) Prescribe the outflow BC on the right boundary
    discrete_velocities_prev = discrete_velocities_prev.at[-1, :, LEFT_VELOCITIES].set(
        discrete_velocities_prev[-2, :, LEFT_VELOCITIES]
    )

    # (2) Macroscopic Velocities
    density_prev = get_density(discrete_velocities_prev)
    macroscopic_velocities_prev = get_macroscopic_velocities(
        discrete_velocities_prev, density_prev
    )

    # (3) Prescribe Inflow Dirichlet BC using Zou/He scheme
    macroscopic_velocities_prev = macroscopic_velocities_prev.at[0, 1:-1, :].set(
        velocity_profile[0, 1:-1, :]
    )
    density_prev = density_prev.at[0, :].set(
        (
            get_density(discrete_velocities_prev[0, :, PURE_VERTICAL_VELOCITIES].T)
            + 2 * get_density(discrete_velocities_prev[0, :, LEFT_VELOCITIES].T)
        ) / (1 - macroscopic_velocities_prev[0, :, 0])
    )

    # (4) Compute discrete Equilibria velocities
    equilibrium_discrete_velocities = get_equilibrium_discrete_velocities(
        macroscopic_velocities_prev, density_prev
    )

    # (3) Belongs to the Zou/He scheme
    discrete_velocities_prev = discrete_velocities_prev.at[0, :, RIGHT_VELOCITIES].set(
        equilibrium_discrete_velocities[0, :, RIGHT_VELOCITIES]
    )

    # (5) Collide according to BGK
    discrete_velocities_post_collision = (
        discrete_velocities_prev
        - relaxation_omega
        * (discrete_velocities_prev - equilibrium_discrete_velocities)
    )

    # (6) Apply jet effect using jnp.where
    jet_effect = jet_strength * LATTICE_WEIGHTS
    for vel in UP_VELOCITIES:
        discrete_velocities_post_collision = discrete_velocities_post_collision.at[:, :, vel].set(
            jnp.where(jet_mask, discrete_velocities_post_collision[:, :, vel] + jet_effect[vel], discrete_velocities_post_collision[:, :, vel])
        )
    for vel in DOWN_VELOCITIES:
        discrete_velocities_post_collision = discrete_velocities_post_collision.at[:, :, vel].set(
            jnp.where(jet_mask, discrete_velocities_post_collision[:, :, vel] - jet_effect[vel], discrete_velocities_post_collision[:, :, vel])
        )

    # (7) Bounce-Back Boundary Conditions for no-slip
    for i in range(N_DISCRETE_VELOCITIES):
        discrete_velocities_post_collision = discrete_velocities_post_collision.at[obstacle_mask, LATTICE_INDICES[i]].set(
            discrete_velocities_prev[obstacle_mask, OPPOSITE_LATTICE_INDICES[i]]
        )

    # (8) Stream alongside lattice velocities
    discrete_velocities_streamed = discrete_velocities_post_collision
    for i in range(N_DISCRETE_VELOCITIES):
        discrete_velocities_streamed = discrete_velocities_streamed.at[:, :, i].set(
            jnp.roll(
                jnp.roll(
                    discrete_velocities_post_collision[:, :, i],
                    LATTICE_VELOCITIES[0, i],
                    axis=0,
                ),
                LATTICE_VELOCITIES[1, i],
                axis=1,
            )
        )

    return discrete_velocities_streamed

def initialize_flow():
    """Initialize the flow field."""
    return get_equilibrium_discrete_velocities(
        velocity_profile, jnp.ones((N_POINTS_X, N_POINTS_Y))
    )

def visualize_flow(discrete_velocities, jet_strength=None, action=None, reward=None, info=None):
    """Visualize the flow field."""
    density = get_density(discrete_velocities)
    macroscopic_velocities = get_macroscopic_velocities(
        discrete_velocities, density
    )
    velocity_magnitude = jnp.linalg.norm(
        macroscopic_velocities, axis=-1, ord=2
    )
    curl = calculate_vorticity(macroscopic_velocities)

    plt.figure(figsize=(15, 10))
    
    # Plot velocity magnitude
    plt.subplot(2, 2, 1)
    plt.contourf(X, Y, velocity_magnitude, levels=50, cmap=cmr.amber)
    plt.colorbar().set_label("Velocity Magnitude")
    plt.gca().add_patch(plt.Circle(
        (CYLINDER_CENTER_INDEX_X, CYLINDER_CENTER_INDEX_Y),
        CYLINDER_RADIUS_INDICES, color="darkgreen"
    ))
    plt.title("Velocity Magnitude")

    # Plot vorticity
    plt.subplot(2, 2, 2)
    plt.contourf(X, Y, curl, levels=50, cmap=cmr.redshift, vmin=-0.02, vmax=0.02)
    plt.colorbar().set_label("Vorticity")
    plt.gca().add_patch(plt.Circle(
        (CYLINDER_CENTER_INDEX_X, CYLINDER_CENTER_INDEX_Y),
        CYLINDER_RADIUS_INDICES, color="darkgreen"
    ))
    plt.title("Vorticity")
    
    # Plot jet strength if provided
    if jet_strength is not None or action is not None:
        plt.subplot(2, 2, 3)
        strength = jet_strength if jet_strength is not None else action
        plt.bar(['Jet Strength'], [strength])
        plt.ylim(-1.1, 1.1)
        plt.title(f'Jet Strength: {strength:.3f}')
    
    # Plot reward and info if provided
    if reward is not None:
        plt.subplot(2, 2, 4)
        info_text = f'Reward: {reward:.3f}\n'
        if info is not None:
            for key, value in info.items():
                info_text += f'{key}: {value:.3f}\n'
        plt.text(0.5, 0.5, info_text, horizontalalignment='center', 
                 verticalalignment='center', fontsize=12, transform=plt.gca().transAxes)
        plt.axis('off')
        plt.title('Performance Metrics')
    
    plt.tight_layout()
    return plt.gcf()

def main():
    jax.config.update("jax_enable_x64", True)

    discrete_velocities_prev = initialize_flow()

    plt.style.use("dark_background")
    plt.figure(figsize=(15, 6), dpi=100)

    for iteration_index in tqdm(range(N_ITERATIONS)):
        discrete_velocities_next = update(discrete_velocities_prev, jet_strength=0.09)
        discrete_velocities_prev = discrete_velocities_next

        if iteration_index % PLOT_EVERY_N_STEPS == 0 and VISUALIZE and iteration_index > SKIP_FIRST_N_ITERATIONS:
            density = get_density(discrete_velocities_next)
            macroscopic_velocities = get_macroscopic_velocities(
                discrete_velocities_next, density
            )
            velocity_magnitude = jnp.linalg.norm(
                macroscopic_velocities, axis=-1, ord=2
            )
            curl = calculate_vorticity(macroscopic_velocities)

            plt.subplot(211)
            plt.contourf(X, Y, velocity_magnitude, levels=50, cmap=cmr.amber)
            plt.colorbar().set_label("Velocity Magnitude")
            plt.gca().add_patch(plt.Circle(
                (CYLINDER_CENTER_INDEX_X, CYLINDER_CENTER_INDEX_Y),
                CYLINDER_RADIUS_INDICES, color="darkgreen"
            ))

            plt.subplot(212)
            plt.contourf(X, Y, curl, levels=50, cmap=cmr.redshift, vmin=-0.02, vmax=0.02)
            plt.colorbar().set_label("Vorticity Magnitude")
            plt.gca().add_patch(plt.Circle(
                (CYLINDER_CENTER_INDEX_X, CYLINDER_CENTER_INDEX_Y),
                CYLINDER_RADIUS_INDICES, color="darkgreen"
            ))

            plt.draw()
            plt.pause(0.0001)
            plt.clf()
    
    if VISUALIZE:
        plt.show()

if __name__ == "__main__":
    main()
