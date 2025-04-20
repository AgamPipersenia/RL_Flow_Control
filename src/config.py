# LBM Simulation Parameters
NITERATIONS = 15000
REYNOLDSNUMBER = 80
NPOINTSX = 300
NPOINTSY = 50
CYLINDERCENTERINDEXX = NPOINTSX // 5
CYLINDERCENTERINDEXY = NPOINTSY // 2
CYLINDERRADIUSINDICES = NPOINTSY // 9
MAXHORIZONTALINFLOWVELOCITY = 0.04
VISUALIZE = True
PLOTEVERYNSTEPS = 100
SKIPFIRSTNITERATIONS = 5000

# RL Environment Parameters
STATE_DIM = 64  # Dimension of state representation
ACTION_DIM = 1  # Dimension of action space (jet strength)
MAX_JET_STRENGTH = 0.2  # Maximum jet strength
STEPS_PER_ACTION = 10  # Number of simulation steps per RL action
MAX_EPISODE_STEPS = 500  # Maximum steps per episode
INITIAL_STEPS = 1000  # Initial steps to establish flow
REWARD_WEIGHTS = {
    'drag': 1.0,
    'lift': 0.0,
    'wake': 0.2,
    'control': 0.1  # Added control penalty weight
}

# RL Agent Parameters
HIDDEN_DIMS = [256, 256]  # Hidden layer dimensions
BUFFER_CAPACITY = 100000  # Replay buffer capacity
BATCH_SIZE = 64  # Batch size for training
GAMMA = 0.99  # Discount factor
TAU = 0.005  # Soft update parameter
LR_ACTOR = 0.0001  # Learning rate for actor
LR_CRITIC = 0.001  # Learning rate for critic
USE_NOISE = True  # Whether to use exploration noise
NOISE_SCALE = 0.1  # Scale of exploration noise
NOISE_DECAY = 0.995  # Noise decay factor
MIN_NOISE_SCALE = 0.01  # Minimum noise scale

# Training Parameters
NUM_EPISODES = 1000  # Number of episodes to train
UPDATES_PER_STEP = 1  # Number of updates per step
PLOT_INTERVAL = 10  # Plot training progress every N episodes
SAVE_INTERVAL = 50  # Save model every N episodes
VIDEO_INTERVAL = 100  # Save video every N episodes
MODEL_DIR = "saved_models"  # Directory to save models
RESULTS_DIR = "results"  # Directory to save results (THIS WAS MISSING)
RENDER = False  # Whether to render during training
RENDER_EVERY = 50  # Render every N episodes
RANDOM_SEED = 42  # Random seed for reproducibility

# Testing Parameters
LOAD_MODEL = False  # Whether to load a model
LOAD_MODEL_PATH = "saved_models/model_best.pt"  # Path to load model from
