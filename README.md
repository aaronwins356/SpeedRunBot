# 🎮 Minecraft RL Bot & Elytra Finder

A reinforcement learning agent designed to beat Minecraft, plus a live bot for finding Elytra in End Cities on servers like LemonCloud.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project provides two main features:

### 1. RL Training Mode
Train an AI agent to beat Minecraft using pure reinforcement learning:
- Curriculum learning through 9 progressive stages
- Actor-critic architecture with PyTorch
- Designed for low-resource hardware (CPU-friendly)

### 2. Live Elytra Finder Bot
A live bot that connects to Minecraft servers to find Elytra:
- Connects to LemonCloud or other servers
- Navigates to The End dimension
- Searches for End Cities and End Ships
- Detects and logs Elytra locations
- Supports scripted or RL-based navigation

## 📁 Project Structure

```
SpeedRunBot/
├── main.py                 # CLI: choose TRAIN or LIVE_BOT mode
├── config.yaml             # Configuration for both modes
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
│
├── agent/                  # Neural network and policy
│   ├── model.py            # PyTorch 3D CNN-based policy network
│   └── policy.py           # Policy wrapper with inference helpers
│
├── env/                    # Environment implementations
│   ├── offline_env.py      # Mock environment for training
│   └── lemoncloud_env.py   # Live environment wrapper
│
├── live_bot/               # Elytra Finder Bot components
│   ├── controller.py       # State machine and bot logic
│   ├── navigation.py       # Pathfinding and movement
│   ├── perception.py       # World observation
│   ├── end_city_scanner.py # End City/Ship detection
│   ├── inventory_manager.py# Chest and item handling
│   ├── tasks.py            # Reusable task primitives
│   └── login_flow.py       # Server connection flow
│
├── integration/            # Minecraft client integration
│   └── mc_client.py        # Client abstraction layer
│
├── training/               # Training components
│   ├── train.py            # Training loop
│   ├── reward.py           # Reward shaping
│   └── curriculum.py       # Curriculum learning
│
├── utils/                  # Utilities
│   ├── logger.py           # Training logger
│   └── config.py           # Configuration helpers
│
├── scripts/                # Runner scripts
│   ├── run_live_bot.sh     # Linux/Mac bot runner
│   └── run_live_bot.bat    # Windows bot runner
│
├── checkpoints/            # Saved models
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- PyTorch (CPU version is sufficient)
- NumPy, PyYAML

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
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (for live bot)
   ```bash
   cp .env.example .env
   # Edit .env with your Minecraft credentials
   ```

### Running

#### Training Mode

```bash
# Start training with default settings
python main.py train

# Start at a specific curriculum stage
python main.py train --stage survival

# Resume from checkpoint
python main.py train --resume checkpoints/best_model.pt

# Legacy flag-based syntax (still supported)
python main.py --train
```

#### Live Bot Mode

```bash
# Run Elytra Finder Bot
python main.py live-bot

# Dry run (simulate without sending commands)
python main.py live-bot --dry-run

# With custom settings
python main.py live-bot --host play.lemoncloud.net --max-runtime 60

# Using the runner script
./scripts/run_live_bot.sh --dry-run
```

#### Quick Test

```bash
python main.py --test
```

## ⚙️ Configuration

Edit `config.yaml` to customize behavior:

```yaml
# Mode Selection
mode: "TRAIN"  # or "LIVE_BOT"

# Training Parameters
num_episodes: 10000
learning_rate: 0.0003
use_curriculum: true

# LemonCloud Server Settings
lemoncloud:
  host: "play.lemoncloud.net"
  port: 25565
  login_commands:
    - "/survival"
  go_to_end_commands:
    - "/warp end"

# Account Settings
account:
  username: "your_username"
  password_env_var: "MC_PASSWORD"  # Read from environment

# Bot Behavior
bot_behavior:
  max_runtime_minutes: 120
  control_mode: "scripted"  # or "rl"
  search_radius_blocks: 256
```

## 🎓 Curriculum Learning (Training Mode)

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

## 🦅 Elytra Finder Bot (Live Mode)

The bot uses a state machine to:

1. **CONNECT** - Connect to the server
2. **LOGIN_FLOW** - Execute server commands to reach survival
3. **ENTER_END** - Navigate to The End dimension
4. **SEARCH_FOR_CITY** - Scan for End City structures
5. **PATH_TO_CITY** - Navigate to found city
6. **SEARCH_FOR_SHIP** - Look for End Ship
7. **PATH_TO_SHIP** - Navigate to ship
8. **OPEN_SHIP_CHEST** - Open the ship's chest
9. **CHECK_FOR_ELYTRA** - Check for Elytra item
10. **LOG_RESULT** - Log coordinates and findings
11. **MOVE_TO_NEXT_TARGET** - Continue searching

### Output

Elytra finds are logged to `elytra_finds.jsonl`:

```json
{
  "timestamp": "2024-01-15T12:30:45",
  "dimension": "the_end",
  "x": 1234,
  "y": 65,
  "z": -5678,
  "elytra_found": true
}
```

## 🧠 Model Architecture

The policy network is a PyTorch Actor-Critic model:

1. **Block Encoder**: 3D CNN that processes 21×21×21 block observations
2. **State Encoder**: MLP for inventory and agent state
3. **Policy Head**: Outputs action probabilities
4. **Value Head**: Estimates state value

## ⚠️ Safety & Ethics

**This bot is intended to be used only where automation is explicitly allowed by the server owner.**

- ✅ Use on your own private servers
- ✅ Use on servers that explicitly allow bots
- ✅ Use with permission from server owners
- ❌ Do NOT use to violate server rules
- ❌ Do NOT use for unauthorized automation
- ❌ Do NOT use to gain unfair advantages

The bot is designed to:
- Respect rate limits (no packet spam)
- Not bypass authentication
- Not evade anti-cheat systems
- Support dry-run mode for testing

## 💻 Running on Low-End Hardware

This project is designed for CPU-only training:

```yaml
# In config.yaml:
batch_size: 32
max_steps_per_episode: 5000
continuous_actions: false
log_interval: 50
```

## 📦 Minecraft Client Integration

The bot uses an abstraction layer (`integration/mc_client.py`) that can be implemented using:

- **pyCraft**: Pure Python Minecraft protocol
- **Mineflayer Bridge**: WebSocket to Node.js Mineflayer
- **RCON**: For server commands (limited)

The current implementation is a placeholder that logs actions. For actual server connectivity, implement the `MinecraftClient` class methods using your preferred library.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Inspired by OpenAI's work on game-playing AI
- MineRL team for Minecraft RL benchmarks
- Project Malmo team at Microsoft Research
