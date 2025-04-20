import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

def plot_training_progress(rewards, drag, lift, wake_metric, actor_losses, critic_losses, save_path=None):
    """
    Plot training metrics.
    
    Args:
        rewards: List of episode rewards
        drag: List of average drag coefficients per episode
        lift: List of average lift coefficients per episode
        wake_metric: List of average wake metrics per episode
        actor_losses: List of actor losses
        critic_losses: List of critic losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 12))
    
    # Plot rewards
    plt.subplot(3, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot drag coefficient
    plt.subplot(3, 2, 2)
    plt.plot(drag)
    plt.title('Average Drag Coefficient')
    plt.xlabel('Episode')
    plt.ylabel('Drag')
    
    # Plot lift coefficient
    plt.subplot(3, 2, 3)
    plt.plot(lift)
    plt.title('Average Lift Coefficient')
    plt.xlabel('Episode')
    plt.ylabel('Lift')
    
    # Plot wake metric
    plt.subplot(3, 2, 4)
    plt.plot(wake_metric)
    plt.title('Average Wake Metric')
    plt.xlabel('Episode')
    plt.ylabel('Wake Metric')
    
    # Plot actor loss
    plt.subplot(3, 2, 5)
    if actor_losses:
        plt.plot(actor_losses)
        plt.title('Actor Loss')
        plt.xlabel('Update')
        plt.ylabel('Loss')
    
    # Plot critic loss
    plt.subplot(3, 2, 6)
    if critic_losses:
        plt.plot(critic_losses)
        plt.title('Critic Loss')
        plt.xlabel('Update')
        plt.ylabel('Loss')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def save_training_video(env, agent, filename, fps=20, max_steps=500):
    """
    Create a video of the agent's performance.
    
    Args:
        env: Environment
        agent: Trained agent
        filename: Output filename
        fps: Frames per second
        max_steps: Maximum steps to record
    """
    # Reset environment
    state = env.reset()
    frames = []
    
    # Run episode
    for step in range(max_steps):
        # Select action without noise
        action = agent.select_action(state, add_noise=False)
        
        # Take action in environment
        next_state, reward, done, info = env.step(action)
        
        # Render and capture frame
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        
        # Update state
        state = next_state
        
        if done:
            break
    
    # Save video
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    imageio.mimsave(filename, frames, fps=fps)
    print(f"Video saved to {filename}")

def save_comparison_video(results, filename, fps=20):
    """
    Create a video comparing different control strategies.
    
    Args:
        results: Dictionary of results for each strategy
        filename: Output filename
        fps: Frames per second
    """
    # Get strategies and frames
    strategies = list(results.keys())
    all_frames = [results[s]['frames'] for s in strategies]
    
    # Find minimum number of frames across all strategies
    min_frames = min(len(frames) for frames in all_frames)
    
    # Prepare frames for video
    video_frames = []
    
    # For each timestep
    for i in range(min_frames):
        # Create a figure with subplots for each strategy
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, len(strategies))
        
        for j, strategy in enumerate(strategies):
            # Get frame for this strategy
            frame = results[strategy]['frames'][i]
            
            # Plot velocity field
            ax1 = plt.subplot(gs[0, j])
            ax1.imshow(frame)
            ax1.set_title(f"{strategy.replace('_', ' ').title()}")
            ax1.axis('off')
            
            # Plot jet strength over time
            ax2 = plt.subplot(gs[1, j])
            jet_strengths = results[strategy]['jet_strengths'][:i+1]
            time_steps = range(len(jet_strengths))
            ax2.plot(time_steps, jet_strengths, 'r-')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Jet Strength')
            ax2.set_ylim(-1.1, 1.1)
            
            # Add drag information
            drag_value = results[strategy]['drags'][i]
            ax2.text(0.5, 0.9, f"Drag: {drag_value:.4f}", 
                     transform=ax2.transAxes, ha='center')
        
        plt.tight_layout()
        
        # Convert figure to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Add to video frames
        video_frames.append(image)
        
        plt.close(fig)
    
    # Save video
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    imageio.mimsave(filename, video_frames, fps=fps)
    print(f"Comparison video saved to {filename}")

def plot_comparison_results(results, save_path=None):
    """
    Plot comparison of different control strategies.
    
    Args:
        results: Dictionary of results for each strategy
        save_path: Path to save the plot
    """
    strategies = list(results.keys())
    
    plt.figure(figsize=(15, 15))
    
    # Plot drag coefficient
    plt.subplot(3, 1, 1)
    for strategy in strategies:
        drags = results[strategy]['drags']
        plt.plot(drags, label=strategy.replace('_', ' ').title())
    plt.title('Drag Coefficient Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Drag Coefficient')
    plt.legend()
    
    # Plot lift coefficient
    plt.subplot(3, 1, 2)
    for strategy in strategies:
        lifts = results[strategy]['lifts']
        plt.plot(lifts, label=strategy.replace('_', ' ').title())
    plt.title('Lift Coefficient Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Lift Coefficient')
    plt.legend()
    
    # Plot jet strength
    plt.subplot(3, 1, 3)
    for strategy in strategies:
        jet_strengths = results[strategy]['jet_strengths']
        plt.plot(jet_strengths, label=strategy.replace('_', ' ').title())
    plt.title('Jet Strength Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Jet Strength')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_flow_field(env, agent=None, save_path=None):
    """
    Visualize the flow field with or without control.
    
    Args:
        env: Environment
        agent: RL agent (optional)
        save_path: Path to save the visualization
    """
    # Reset environment
    state = env.reset()
    
    # Run for a few steps to establish flow
    for _ in range(100):
        if agent:
            action = agent.select_action(state, add_noise=False)
        else:
            action = np.array([0.0])  # No control
        
        next_state, _, _, _ = env.step(action)
        state = next_state
    
    # Render flow field
    plt.figure(figsize=(15, 10))
    
    # Get flow field data
    frame = env.render(mode='rgb_array')
    
    # Display flow field
    plt.imshow(frame)
    plt.axis('off')
    plt.title('Flow Field Visualization')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_vorticity_evolution(env, agent=None, num_steps=500, interval=10, save_path=None):
    """
    Plot the evolution of vorticity in the wake over time.
    
    Args:
        env: Environment
        agent: RL agent (optional)
        num_steps: Number of simulation steps
        interval: Interval between plots
        save_path: Path to save the visualization
    """
    # Reset environment
    state = env.reset()
    
    # Storage for vorticity data
    vorticity_data = []
    
    # Run simulation
    for step in range(num_steps):
        if agent:
            action = agent.select_action(state, add_noise=False)
        else:
            action = np.array([0.0])  # No control
        
        next_state, _, _, _ = env.step(action)
        
        # Store vorticity at specified intervals
        if step % interval == 0:
            # Extract vorticity from environment
            density = env.get_density()
            velocities = env.get_macroscopic_velocities(density)
            vorticity = env.calculate_vorticity(velocities)
            vorticity_data.append(vorticity)
        
        state = next_state
    
    # Plot vorticity evolution
    num_plots = len(vorticity_data)
    cols = 4
    rows = (num_plots + cols - 1) // cols
    
    plt.figure(figsize=(15, 3 * rows))
    
    for i, vorticity in enumerate(vorticity_data):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(vorticity, cmap='RdBu', vmin=-0.05, vmax=0.05)
        plt.colorbar(label='Vorticity')
        plt.title(f'Step {i * interval}')
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_control_effectiveness(drag_reduction, control_effort, strategies, save_path=None):
    """
    Plot the effectiveness of different control strategies.
    
    Args:
        drag_reduction: List of drag reduction percentages for each strategy
        control_effort: List of control efforts for each strategy
        strategies: List of strategy names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    plt.scatter(control_effort, drag_reduction, s=100)
    
    # Add labels for each point
    for i, strategy in enumerate(strategies):
        plt.annotate(
            strategy.replace('_', ' ').title(),
            (control_effort[i], drag_reduction[i]),
            xytext=(10, 5),
            textcoords='offset points'
        )
    
    plt.xlabel('Control Effort (Average |Jet Strength|)')
    plt.ylabel('Drag Reduction (%)')
    plt.title('Control Effectiveness Comparison')
    plt.grid(True)
    
    # Add diagonal line representing efficiency
    max_effort = max(control_effort)
    max_reduction = max(drag_reduction)
    plt.plot([0, max_effort], [0, max_effort/max_effort*max_reduction], 'k--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
