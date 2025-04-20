import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json

from environment import FlowControlEnv
from model.agent import DDPGAgent
from utils.visualization import plot_training_progress, save_training_video
from utils.metrics import calculate_strouhal_number
from config import config

def train():
    """Train the RL agent for active flow control."""
    # Create output directories
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    
    # Create environment
    env = FlowControlEnv()
    
    # Initialize agent
    agent = DDPGAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dims=config.HIDDEN_DIMS,
        buffer_capacity=config.BUFFER_CAPACITY,
        batch_size=config.BATCH_SIZE,
        gamma=config.GAMMA,
        tau=config.TAU,
        lr_actor=config.LR_ACTOR,
        lr_critic=config.LR_CRITIC
    )
    
    # Load pretrained model if specified
    if config.LOAD_MODEL and os.path.exists(config.LOAD_MODEL_PATH):
        agent.load(config.LOAD_MODEL_PATH)
    
    # Training metrics
    rewards_history = []
    episode_lengths = []
    drag_history = []
    lift_history = []
    wake_metric_history = []
    
    # Training loop
    total_steps = 0
    best_reward = -float('inf')
    
    print("Starting training...")
    start_time = time.time()
    
    for episode in range(config.NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        episode_drag = []
        episode_lift = []
        episode_wake = []
        
        for step in range(config.MAX_EPISODE_STEPS):
            # Select action
            action = agent.select_action(state, add_noise=config.USE_NOISE)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience in replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # Update agent
            if len(agent.replay_buffer) > config.BATCH_SIZE:
                for _ in range(config.UPDATES_PER_STEP):
                    agent.update()
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            episode_drag.append(info['drag'])
            episode_lift.append(info['lift'])
            episode_wake.append(info['wake_metric'])
            total_steps += 1
            
            # Render if needed
            if config.RENDER and episode % config.RENDER_EVERY == 0:
                env.render()
            
            if done:
                break
        
        # Record metrics
        rewards_history.append(episode_reward)
        episode_lengths.append(step + 1)
        drag_history.append(np.mean(episode_drag))
        lift_history.append(np.mean(episode_lift))
        wake_metric_history.append(np.mean(episode_wake))
        
        # Print progress
        elapsed_time = time.time() - start_time
        print(f"Episode {episode+1}/{config.NUM_EPISODES} | " +
              f"Reward: {episode_reward:.2f} | " +
              f"Avg Drag: {np.mean(episode_drag):.4f} | " +
              f"Steps: {step+1} | " +
              f"Time: {elapsed_time:.2f}s")
        
        # Visualize progress
        if episode % config.PLOT_INTERVAL == 0:
            plot_training_progress(
                rewards_history, 
                drag_history, 
                lift_history, 
                wake_metric_history,
                agent.actor_losses, 
                agent.critic_losses,
                save_path=os.path.join(config.RESULTS_DIR, f"training_progress_ep{episode}.png")
            )
        
        # Save model
        if episode % config.SAVE_INTERVAL == 0:
            agent.save(os.path.join(config.MODEL_DIR, f"model_ep{episode}.pt"))
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(config.MODEL_DIR, "model_best.pt"))
        
        # Save training video
        if episode % config.VIDEO_INTERVAL == 0:
            save_training_video(
                env, agent, 
                os.path.join(config.RESULTS_DIR, f"video_ep{episode}.mp4"),
                fps=20, max_steps=200
            )
    
    # Save final model and results
    agent.save(os.path.join(config.MODEL_DIR, "model_final.pt"))
    
    # Save training history
    history = {
        'rewards': rewards_history,
        'episode_lengths': episode_lengths,
        'drag': drag_history,
        'lift': lift_history,
        'wake_metric': wake_metric_history,
        'actor_losses': agent.actor_losses,
        'critic_losses': agent.critic_losses
    }
    
    with open(os.path.join(config.RESULTS_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    print(f"Best episode reward: {best_reward:.2f}")
    
    return agent, history

if __name__ == "__main__":
    agent, history = train()
