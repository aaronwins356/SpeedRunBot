# 🎮 Minecraft RL Bot

A reinforcement learning agent designed to beat Minecraft (reaching the end credits) using pure RL — no imitation learning, no human data.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Goals

This project implements a complete RL training framework for Minecraft with the following objectives:

- **Beat Minecraft**: Train an agent to defeat the Ender Dragon and reach the end credits
- **Generalize**: Work on randomly generated seeds and diverse biomes
- **Pure RL**: No human data or imitation learning — learn entirely from experience
- **Curriculum Learning**: Master basic skills before attempting full game runs
- **Low Resource**: Run on standard hardware without GPU requirements

## 📁 Project Structure

```
SpeedRunBot/
├── agent/                    # Neural network and policy
│   ├── model.py              # PyTorch 3D CNN-based policy network
│   └── policy.py             # Policy wrapper with exploration
├── env/                      # Environment logic
│   ├── core_env.py           # Mock Minecraft world simulation
│   ├── blocks.py             # Block types and properties
│   └── actions.py            # Action definitions (discrete & continuous)
├── training/                 # Training components
│   ├── train.py              # Main training loop with PyTorch
│   ├── reward.py             # Reward shaping logic
│   └── curriculum.py         # Curriculum learning stages
├── utils/                    # Utilities
│   ├── logger.py             # Training logger and visualization
│   └── config.py             # Configuration management
├── checkpoints/              # Saved models
├── main.py                   # Main entry point (CLI)
├── config.yaml               # Hyperparameters and settings
├── run.sh                    # Shell script for training
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- PyTorch (CPU version is sufficient)
- NumPy

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/aaronwins356/SpeedRunBot.git
   cd SpeedRunBot
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Install PyTorch CPU version (recommended for low-resource hardware)
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   
   # Install other dependencies
   pip install numpy pyyaml
   ```

### Training

**Start training with default settings:**
```bash
python main.py --train
```

Or using the shell script:
```bash
./run.sh train
```

**Start at a specific curriculum stage:**
```bash
python main.py --stage survival
python main.py --stage resource_gathering
python main.py --stage tool_crafting
```

**Resume from checkpoint:**
```bash
python main.py --resume checkpoints/best_model.pt
```

**Evaluate trained model:**
```bash
python main.py --evaluate --checkpoint checkpoints/best_model.pt
```

### Quick Test

Verify everything works:
```bash
python main.py --test
```

## 🎓 Curriculum Learning

The agent progresses through 9 curriculum stages, mastering each before moving to the next:

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

The agent automatically advances through stages as it demonstrates mastery (configurable success threshold).

## 🎁 Reward System

### Positive Rewards
- Mining valuable items (coal: +2, iron: +5, diamond: +20)
- Crafting tools and items
- Surviving days (+5 per day)
- Entering the Nether (+50)
- Entering the End (+100)
- Defeating the Ender Dragon (+500)
- Game completion (+1000)

### Negative Rewards
- **Only when taking damage** (fall, mobs, lava)
- No penalty for idle time or inefficiency

This design encourages exploration while discouraging dangerous behavior.

### Modifying Rewards

Edit `training/reward.py` to customize reward values:

```python
REWARD_VALUES = {
    'mine_diamond': 20.0,       # Increase for more diamond focus
    'enter_nether': 50.0,       # Milestone rewards
    'kill_dragon': 500.0,       # Ultimate goal
    'damage_fall': -2.0,        # Only damage is penalized
}
```

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
continuous_actions: false     # Use discrete actions (faster)

# Random Seed
seed: 42                      # For reproducibility
```

## 🧠 Model Architecture

The policy network is a PyTorch Actor-Critic model:

1. **Block Encoder**: 3D CNN that processes the 21×21×21 block observation
   - Learnable block type embeddings
   - 3 convolutional layers with batch normalization
   - Global average pooling to feature vector

2. **State Encoder**: MLP for auxiliary inputs
   - Processes inventory counts
   - Processes agent state (health, hunger, position)

3. **Policy Head**: Outputs action probabilities
   - Separate heads for movement, camera, interaction, inventory
   - Discrete actions by default (configurable for continuous)

4. **Value Head**: Estimates state value for actor-critic training

## 💻 Running on Low-End Hardware

This project is designed for CPU-only training. Tips for limited hardware:

1. **Use discrete actions** (default): Faster than continuous
   ```yaml
   continuous_actions: false
   ```

2. **Reduce batch size**:
   ```yaml
   batch_size: 32  # or even 16
   ```

3. **Limit episode length**:
   ```yaml
   max_steps_per_episode: 5000
   ```

4. **Increase logging interval** to reduce I/O:
   ```yaml
   log_interval: 50
   ```

5. **Use curriculum learning** to avoid wasting compute on impossible tasks:
   ```yaml
   use_curriculum: true
   ```

## 📦 Resuming Training

Save your progress and resume later:

```bash
# Training automatically saves checkpoints every 100 episodes
# Resume from latest:
python main.py --resume checkpoints/checkpoint_500.pt

# Or resume from best model:
python main.py --resume checkpoints/best_model.pt
```

Checkpoints save:
- Model weights
- Optimizer state
- Training progress (episode, steps, best reward)
- Curriculum stage
- Exploration epsilon

## 🗺️ Roadmap: Real Minecraft Integration

The codebase is designed for easy integration with real Minecraft:

### Using Project Malmo

```python
from env import MinecraftEnv

class MalmoEnv(MinecraftEnv):
    """Real Minecraft environment via Project Malmo."""
    
    def __init__(self, malmo_client):
        super().__init__()
        self.malmo = malmo_client
    
    def reset(self, seed=None, options=None):
        self.malmo.start_mission(seed)
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.malmo.send_command(self._convert_action(action))
        return self._get_observation(), reward, done, truncated, info
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

### Future Plans

- [ ] Add PPO algorithm for more stable training
- [ ] Add curiosity-driven exploration (ICM)
- [ ] Real Minecraft integration via MineRL
- [ ] Multi-agent support
- [ ] Video recording of agent behavior
- [ ] Web dashboard for training visualization

## 📊 Viewing Logs

Training logs are saved to `checkpoints/logs/`. View them with:

```python
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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Inspired by OpenAI's work on game-playing AI
- MineRL team for Minecraft RL benchmarks
- Project Malmo team at Microsoft Research
