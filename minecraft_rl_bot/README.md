# Minecraft RL Bot 🎮🤖

A reinforcement learning agent designed to beat Minecraft by reaching the end credits on randomly generated world seeds, without any imitation learning.

## 🎯 Project Goals

This project implements a complete RL training framework for Minecraft with the following objectives:

- **Beat Minecraft**: Train an agent to defeat the Ender Dragon and reach the end credits
- **Generalize**: Work on randomly generated seeds and diverse biomes
- **Pure RL**: No human data or imitation learning - learn entirely from experience
- **Curriculum Learning**: Master basic skills before attempting full game runs
- **Low Resource**: Run on standard hardware without GPU requirements

## 📁 Project Structure

```
minecraft_rl_bot/
│
├── env/                    # Environment logic
│   ├── core_env.py         # Mock Minecraft world simulation
│   ├── blocks.py           # Block types and properties
│   └── actions.py          # Action definitions (discrete & continuous)
│
├── agent/                  # Neural network and policy
│   ├── model.py            # 3D CNN-based policy network
│   └── policy.py           # Policy wrapper with exploration
│
├── training/               # Training components
│   ├── train.py            # Main training loop
│   ├── reward.py           # Reward shaping logic
│   └── curriculum.py       # Curriculum learning stages
│
├── utils/                  # Utilities
│   ├── logger.py           # Training logger and visualization
│   └── helpers.py          # Helper functions
│
├── checkpoints/            # Saved models
├── config.yaml             # Hyperparameters and settings
├── run.sh                  # Shell script for training
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- NumPy (only required dependency!)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/minecraft-rl-bot.git
   cd minecraft-rl-bot
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install numpy
   ```

### Training

**Start training with default settings:**
```bash
cd minecraft_rl_bot
./run.sh train
```

Or directly with Python:
```bash
python -m minecraft_rl_bot.training.train --config config.yaml
```

**Resume from checkpoint:**
```bash
./run.sh resume checkpoints/checkpoint_100.npz
```

**Evaluate model:**
```bash
./run.sh evaluate checkpoints/best_model.npz
```

### Quick Test

Verify everything works:
```bash
./run.sh test
```

Or in Python:
```python
from minecraft_rl_bot import MinecraftEnv, Policy

# Create environment
env = MinecraftEnv(seed=42)
obs, info = env.reset()

# Create policy
policy = Policy()

# Run a step
action, log_prob, value = policy.act({
    'blocks': obs['blocks'][None, ...],
    'inventory': obs['inventory'][None, ...],
    'agent_state': obs['agent_state'][None, ...]
})
```

## 🎓 Curriculum Learning

The agent progresses through 9 curriculum stages:

| Stage | Name | Description |
|-------|------|-------------|
| 0 | Basic Survival | Don't die, move around safely |
| 1 | Resource Gathering | Mine blocks, collect items |
| 2 | Tool Crafting | Craft tools using crafting table |
| 3 | Nether Access | Build portal, enter the Nether |
| 4 | Blaze Hunting | Find fortress, collect Blaze rods |
| 5 | Ender Pearl Hunt | Collect Ender Pearls |
| 6 | End Preparation | Craft Eyes, find stronghold |
| 7 | Dragon Fight | Enter End, defeat the dragon |
| 8 | Full Game | Complete speedrun from start |

The agent automatically advances through stages as it demonstrates mastery.

## 🎁 Reward System

### Positive Rewards
- Mining valuable items (coal, iron, diamond)
- Crafting tools and items
- Surviving days
- Entering the Nether
- Entering the End
- Defeating the Ender Dragon
- Game completion

### Negative Rewards
- **Only when taking damage** (fall, mobs, lava)
- No penalty for idle time or inefficiency

This design encourages exploration while discouraging dangerous behavior.

## ⚙️ Configuration

Edit `config.yaml` to customize training:

```yaml
# Training Parameters
num_episodes: 10000           # Total episodes
max_steps_per_episode: 10000  # Max steps per episode
batch_size: 64                # Optimization batch size
gamma: 0.99                   # Discount factor
learning_rate: 0.0003         # Learning rate

# Features
use_curriculum: true          # Enable curriculum learning
continuous_actions: false     # Use discrete actions

# Random Seed
seed: 42                      # For reproducibility
```

## 🧠 Model Architecture

The policy network consists of:

1. **Block Encoder**: 3D CNN that processes the 21×21×21 block observation
   - Converts block types to features
   - 3 convolutional layers with stride for dimension reduction
   - Global average pooling to feature vector

2. **State Encoder**: MLP for auxiliary inputs
   - Processes inventory counts
   - Processes agent state (health, hunger, position)

3. **Policy Head**: Outputs action probabilities
   - Discrete: Separate head for each action type
   - Continuous: Mean and log-std for Gaussian policy

4. **Value Head**: Estimates state value for actor-critic training

## 🔌 Extending to Real Minecraft

The codebase is designed for easy integration with real Minecraft:

### Using Project Malmo

```python
from minecraft_rl_bot.env import MinecraftEnv

class MalmoEnv(MinecraftEnv):
    """Real Minecraft environment via Project Malmo."""
    
    def __init__(self, malmo_client):
        super().__init__()
        self.malmo = malmo_client
    
    def reset(self, seed=None, options=None):
        # Connect to Malmo and reset world
        self.malmo.start_mission(seed)
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Send action to Malmo
        self.malmo.send_command(self._convert_action(action))
        # Get new state
        return self._get_observation(), reward, done, truncated, info
    
    def _get_observation(self):
        # Get observation from Malmo
        world_state = self.malmo.get_world_state()
        return {
            'blocks': self._parse_blocks(world_state),
            'inventory': self._parse_inventory(world_state),
            'agent_state': self._parse_agent_state(world_state)
        }
```

### Using MineRL

```python
import minerl

class MineRLEnv(MinecraftEnv):
    """Real Minecraft environment via MineRL."""
    
    def __init__(self, env_name='MineRLObtainDiamond-v0'):
        super().__init__()
        self.minerl_env = gym.make(env_name)
    
    # Override methods to wrap MineRL...
```

## 📊 Viewing Logs

Training logs are saved to `checkpoints/logs/`. View them with:

```python
from minecraft_rl_bot.utils import TrainingLogger

# Load and display logs
import json
with open('checkpoints/logs/run_*/episodes.json') as f:
    episodes = json.load(f)
    
for ep in episodes[-10:]:
    print(f"Episode {ep['episode']}: Reward={ep['reward']:.2f}")
```

## ⚠️ Limitations

1. **Mock Environment**: The included environment is simplified. For real Minecraft performance, integrate with Malmo or MineRL.

2. **No GPU Acceleration**: Designed for CPU training. Training is slower but works on any machine.

3. **Simplified Physics**: The mock world doesn't perfectly replicate Minecraft physics.

4. **Basic Gradient Estimation**: Uses simple parameter updates. For faster convergence, integrate PyTorch with proper autograd.

## 🗺️ Roadmap

- [ ] Add PyTorch backend for GPU acceleration (optional)
- [ ] Implement PPO algorithm for stable training
- [ ] Add real Minecraft integration via MineRL
- [ ] Add multi-agent support
- [ ] Implement curiosity-driven exploration
- [ ] Add video recording of agent behavior
- [ ] Create web dashboard for training visualization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Inspired by OpenAI's work on game-playing AI
- MineRL team for Minecraft RL benchmarks
- Project Malmo team at Microsoft Research
