# Checkpoints Directory

This directory stores model checkpoints during training.

## Files

- `checkpoint_N.pt` - Periodic checkpoints (every N episodes)
- `best_model.pt` - Best performing model (highest reward)
- `final_model.pt` - Model at end of training

## Usage

### Save checkpoint (automatic during training)
Checkpoints are saved automatically every `save_interval` episodes (default: 100).

### Load checkpoint
```python
from training import Trainer, TrainingConfig

trainer = Trainer(TrainingConfig())
trainer.load_checkpoint('checkpoints/best_model.pt')

# Resume training
trainer.train()

# Or evaluate
results = trainer.evaluate(num_episodes=10)
```

### From command line
```bash
# Resume training
python main.py --resume checkpoints/best_model.pt

# Evaluate
python main.py --evaluate --checkpoint checkpoints/best_model.pt
```

## Checkpoint Contents

Each `.pt` file contains:
- `model_state_dict`: PyTorch model weights
- `optimizer_state_dict`: Adam optimizer state
- `episode`: Training episode number
- `total_steps`: Total environment steps
- `best_reward`: Best reward achieved
- `epsilon`: Current exploration rate
- `curriculum_state`: Curriculum learning progress
- `config`: Training configuration

## Note

- This directory is automatically created during training
- Model files (`.pt`) are excluded from git via `.gitignore`
- Logs are stored in the `logs/` subdirectory
