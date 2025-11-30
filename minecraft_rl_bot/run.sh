#!/bin/bash
# Minecraft RL Bot - Training Script
# ===================================
# This script provides easy commands for training and managing the agent.
#
# Usage:
#   ./run.sh train              # Start training with default config
#   ./run.sh train --episodes 5000  # Train for specific episodes
#   ./run.sh resume checkpoint.npz  # Resume from checkpoint
#   ./run.sh evaluate           # Evaluate current best model
#   ./run.sh test               # Run quick test
#   ./run.sh clean              # Clean checkpoints and logs

set -e

# Configuration
PYTHON=${PYTHON:-python3}
CONFIG_FILE="config.yaml"
CHECKPOINT_DIR="checkpoints"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${GREEN}[Minecraft RL Bot]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[Warning]${NC} $1"
}

print_error() {
    echo -e "${RED}[Error]${NC} $1"
}

# Check Python and dependencies
check_requirements() {
    print_msg "Checking requirements..."
    
    if ! command -v $PYTHON &> /dev/null; then
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check NumPy
    if ! $PYTHON -c "import numpy" 2>/dev/null; then
        print_warn "NumPy not found. Installing..."
        $PYTHON -m pip install numpy
    fi
    
    print_msg "Requirements OK"
}

# Train the agent
train() {
    print_msg "Starting training..."
    check_requirements
    
    # Create checkpoint directory
    mkdir -p $CHECKPOINT_DIR
    
    # Run training
    $PYTHON -m minecraft_rl_bot.training.train --config $CONFIG_FILE "$@"
}

# Resume training from checkpoint
resume() {
    if [ -z "$1" ]; then
        # Look for latest checkpoint
        CHECKPOINT=$(ls -t $CHECKPOINT_DIR/checkpoint_*.npz 2>/dev/null | head -1)
        if [ -z "$CHECKPOINT" ]; then
            print_error "No checkpoint found. Use: ./run.sh train"
            exit 1
        fi
    else
        CHECKPOINT=$1
    fi
    
    print_msg "Resuming from: $CHECKPOINT"
    check_requirements
    
    $PYTHON -m minecraft_rl_bot.training.train --config $CONFIG_FILE --checkpoint "$CHECKPOINT" "${@:2}"
}

# Evaluate the model
evaluate() {
    print_msg "Evaluating model..."
    check_requirements
    
    CHECKPOINT="${1:-$CHECKPOINT_DIR/best_model.npz}"
    
    if [ ! -f "$CHECKPOINT" ]; then
        print_error "Model not found: $CHECKPOINT"
        exit 1
    fi
    
    $PYTHON -c "
from minecraft_rl_bot import MinecraftEnv, Trainer, TrainingConfig

config = TrainingConfig()
trainer = Trainer(config)
trainer.load_checkpoint('$CHECKPOINT')

results = trainer.evaluate(num_episodes=10)
print()
print('Evaluation Results:')
print(f'  Mean Reward: {results[\"mean_reward\"]:.2f} ± {results[\"std_reward\"]:.2f}')
print(f'  Mean Steps: {results[\"mean_steps\"]:.0f}')
print(f'  Completion Rate: {results[\"completion_rate\"]:.1%}')
"
}

# Run quick test
test() {
    print_msg "Running quick test..."
    check_requirements
    
    $PYTHON -c "
from minecraft_rl_bot import MinecraftEnv, Policy, DiscreteAction

# Test environment
env = MinecraftEnv(seed=42)
obs, info = env.reset()
print('Environment created successfully')
print(f'  Observation shape: blocks={obs[\"blocks\"].shape}, inventory={obs[\"inventory\"].shape}')

# Test policy
policy = Policy()
action, log_prob, value = policy.act({
    'blocks': obs['blocks'][None, ...],
    'inventory': obs['inventory'][None, ...],
    'agent_state': obs['agent_state'][None, ...]
})
print('Policy created successfully')
print(f'  Action: {action}')

# Test step
obs, reward, term, trunc, info = env.step(DiscreteAction())
print('Environment step successful')
print(f'  Reward: {reward:.2f}')

print()
print('All tests passed!')
"
}

# Clean checkpoints and logs
clean() {
    print_warn "This will delete all checkpoints and logs."
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf $CHECKPOINT_DIR
        rm -rf logs
        print_msg "Cleaned successfully"
    else
        print_msg "Cancelled"
    fi
}

# Show help
show_help() {
    echo "Minecraft RL Bot - Training Script"
    echo ""
    echo "Usage: ./run.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  train              Start training with default config"
    echo "  resume [file]      Resume training from checkpoint"
    echo "  evaluate [file]    Evaluate model performance"
    echo "  test               Run quick functionality test"
    echo "  clean              Remove checkpoints and logs"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh train"
    echo "  ./run.sh train --episodes 5000"
    echo "  ./run.sh resume checkpoints/checkpoint_100.npz"
    echo "  ./run.sh evaluate checkpoints/best_model.npz"
    echo ""
}

# Main entry point
case "${1:-help}" in
    train)
        shift
        train "$@"
        ;;
    resume)
        shift
        resume "$@"
        ;;
    evaluate|eval)
        shift
        evaluate "$@"
        ;;
    test)
        test
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
