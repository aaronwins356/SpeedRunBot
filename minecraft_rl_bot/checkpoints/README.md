# Checkpoints Directory

This directory stores model checkpoints during training.

## Files

- `checkpoint_N.npz` - Periodic checkpoints (every N episodes)
- `checkpoint_N_state.json` - Training state for checkpoint N
- `best_model.npz` - Best performing model
- `final_model.npz` - Model at end of training

## Usage

### Save checkpoint
```python
trainer._save_checkpoint(episode=100)
```

### Load checkpoint
```python
trainer.load_checkpoint('checkpoints/best_model.npz')
```

## Note

This directory is automatically created during training.
Model files are excluded from git via `.gitignore`.
