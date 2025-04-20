import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import json
from tqdm import tqdm

from environment import FlowControlEnv
from model.agent import DDPGAgent
from src.utils.visualization import save_comparison_video, plot_comparison_results
from src.utils.metrics import calculate_strouhal_number
import config

def test_agent(model_path, num_episodes=5, render=True, save_results=True):
    """
    Test a trained agent on the flow control environment.
    
    Args:
        model_path: Path to the trained model
        num_episodes: Number of test episodes
        render: Whether to render the environment
        save_results: Whether to save results
        
    Returns:
        results: Dictionary of test results
    """
    # Create environment
    env = FlowControlEnv()
    
    # Initialize agent
    agent = DDPGAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dims=config.HIDDEN_DIMS
    )
    
    # Load trained model
    agent.load(model_path)
    
    # Test metrics
    rewards = []
    drags = []
    lifts = []
    wake_metrics = []
    jet_strengths = []
    
    print(f"Testing agent from {model_path}...")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_drag = []
        episode_lift = []
        episode_wake = []
        episode_jet = []
        
        for step in range(config.MAX_EPISODE_STEPS):
            # Select action without noise
            action = agent.select_action(state, add_noise=False)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            episode_drag.append(info['drag'])
            episode_lift.append(info['lift'])
            episode_wake.append(info['wake_metric'])
            episode_jet.append(info['jet_strength'])
            
            # Render if needed
            if render and episode == 0:
                env.render()
                plt.pause(0.01)
            
            if done:
                break
        
        # Record metrics
        rewards.append(episode_reward)
        drags.append(np.mean(episode_drag))
        lifts.append(np.mean(episode_lift))
        wake_metrics.append(np.mean(episode_wake))
        jet_strengths.append(np.mean(np.abs(episode_jet)))
        
        print(f"Episode {episode+1}/{num_episodes} | " +
              f"Reward: {episode_reward:.2f} | " +
              f"Avg Drag: {np.mean(episode_drag):.4f} | " +
              f"Avg Jet: {np.mean(np.abs(episode_jet)):.4f}")
    
    # Compile results
    results = {
        'rewards': rewards,
        'drags': drags,
        'lifts': lifts,
        'wake_metrics': wake_metrics,
        'jet_strengths': jet_strengths,
        'mean_reward': np.mean(rewards),
        'mean_drag': np.mean(drags),
        'mean_lift': np.mean(lifts),
        'mean_wake': np.mean(wake_metrics),
        'mean_jet': np.mean(jet_strengths)
    }
    
    # Save results
    if save_results:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        with open(os.path.join(config.RESULTS_DIR, 'test_results.json'), 'w') as f:
            json.dump(results, f)
    
    return results

def compare_control_strategies():
    """
    Compare different control strategies:
    1. No control
    2. Constant jet
    3. Sinusoidal jet
    4. RL-controlled jet
    """
    # Create environment
    env = FlowControlEnv()
    
    # Initialize RL agent
    agent = DDPGAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dims=config.HIDDEN_DIMS
    )
    
    # Load trained model
    agent.load(os.path.join(config.MODEL_DIR, "model_best.pt"))
    
    # Test parameters
    num_steps = 1000
    strategies = ['no_control', 'constant_jet', 'sinusoidal_jet', 'rl_agent']
    
    # Results storage
    all_results = {}
    
    for strategy in strategies:
        print(f"Testing {strategy} strategy...")
        
        # Reset environment
        state = env.reset()
        
        # Storage for this strategy
        rewards = []
        drags = []
        lifts = []
        wake_metrics = []
        jet_strengths = []
        frames = []
        
        for step in tqdm(range(num_steps)):
            # Select action based on strategy
            if strategy == 'no_control':
                action = np.array([0.0])
            elif strategy == 'constant_jet':
                action = np.array([0.5])  # 50% of max jet strength
            elif strategy == 'sinusoidal_jet':
                # Sinusoidal control with period of 100 steps
                action = np.array([0.5 * np.sin(2 * np.pi * step / 100)])
            elif strategy == 'rl_agent':
                action = agent.select_action(state, add_noise=False)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Update state and metrics
            state = next_state
            rewards.append(reward)
            drags.append(info['drag'])
            lifts.append(info['lift'])
            wake_metrics.append(info['wake_metric'])
            jet_strengths.append(info['jet_strength'])
            
            # Capture frame for video
            if step % 5 == 0:  # Save every 5th frame to reduce video size
                frames.append(env.render(mode='rgb_array'))
            
            if done:
                break
        
        # Compile results for this strategy
        all_results[strategy] = {
            'rewards': rewards,
            'drags': drags,
            'lifts': lifts,
            'wake_metrics': wake_metrics,
            'jet_strengths': jet_strengths,
            'mean_reward': np.mean(rewards),
            'mean_drag': np.mean(drags),
            'std_drag': np.std(drags),
            'mean_lift': np.mean(lifts),
            'std_lift': np.std(lifts),
            'mean_wake': np.mean(wake_metrics),
            'mean_jet': np.mean(np.abs(jet_strengths)),
            'frames': frames
        }
        
        print(f"Strategy: {strategy} | " +
              f"Mean Reward: {np.mean(rewards):.2f} | " +
              f"Mean Drag: {np.mean(drags):.4f} | " +
              f"Drag Reduction: {(1 - np.mean(drags)/np.mean(all_results.get('no_control', {'drags': [1.0]})['drags'])):.2%}")
    
    # Save comparison results
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Save metrics (excluding frames)
    metrics_results = {k: {k2: v2 for k2, v2 in v.items() if k2 != 'frames'} 
                      for k, v in all_results.items()}
    
    with open(os.path.join(config.RESULTS_DIR, 'comparison_results.json'), 'w') as f:
        json.dump(metrics_results, f)
    
    # Create comparison video
    save_comparison_video(
        all_results,
        os.path.join(config.RESULTS_DIR, 'comparison_video.mp4'),
        fps=20
    )
    
    # Plot comparison results
    plot_comparison_results(
        all_results,
        os.path.join(config.RESULTS_DIR, 'comparison_plot.png')
    )
    
    return all_results

if __name__ == "__main__":
    # Test the best trained agent
    test_results = test_agent(
        model_path=os.path.join(config.MODEL_DIR, "model_best.pt"),
        num_episodes=5,
        render=True,
        save_results=True
    )
    
    # Compare different control strategies
    comparison_results = compare_control_strategies()
    
    print("\nTesting completed!")
    print(f"RL Agent Mean Reward: {test_results['mean_reward']:.2f}")
    print(f"RL Agent Mean Drag: {test_results['mean_drag']:.4f}")
    
    print("\nStrategy Comparison:")
    for strategy, results in comparison_results.items():
        if strategy == 'no_control':
            baseline_drag = results['mean_drag']
            print(f"{strategy}: Mean Drag = {results['mean_drag']:.4f} (baseline)")
        else:
            drag_reduction = (1 - results['mean_drag']/baseline_drag) * 100
            print(f"{strategy}: Mean Drag = {results['mean_drag']:.4f} " +
                  f"(Reduction: {drag_reduction:.2f}%)")
