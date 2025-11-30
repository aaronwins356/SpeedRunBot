#!/bin/bash
# Minecraft RL Bot - Training Script
# ===================================
# This script provides easy commands for training and managing the agent.
#
# Usage:
#   ./run.sh train                    # Start training with default config
#   ./run.sh train --episodes 5000    # Train for specific episodes
#   ./run.sh resume checkpoint.pt     # Resume from checkpoint
#   ./run.sh evaluate model.pt        # Evaluate model performance
#   ./run.sh test                     # Run quick test
#   ./run.sh stages                   # List curriculum stages
#   ./run.sh clean                    # Clean checkpoints and logs

set -e

# Configuration
PYTHON=${PYTHON:-python3}
CONFIG_FILE="config.yaml"
CHECKPOINT_DIR="checkpoints"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_info() {
    echo -e "${BLUE}[Info]${NC} $1"
}

# Check Python and dependencies
check_requirements() {
    print_msg "Checking requirements..."
    
    if ! command -v $PYTHON &> /dev/null; then
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check PyTorch
    if ! $PYTHON -c "import torch" 2>/dev/null; then
        print_warn "PyTorch not found. Installing CPU version..."
        $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Check NumPy
    if ! $PYTHON -c "import numpy" 2>/dev/null; then
        print_warn "NumPy not found. Installing..."
        $PYTHON -m pip install numpy
    fi
    
    print_msg "Requirements OK ✓"
}

# Train the agent
train() {
    print_msg "Starting training..."
    check_requirements
    
    # Create checkpoint directory
    mkdir -p $CHECKPOINT_DIR
    
    # Run training
    $PYTHON main.py --train --config $CONFIG_FILE "$@"
}

# Train with specific stage
train_stage() {
    STAGE=$1
    shift
    print_msg "Starting training at stage: $STAGE"
    check_requirements
    
    mkdir -p $CHECKPOINT_DIR
    
    $PYTHON main.py --stage "$STAGE" --config $CONFIG_FILE "$@"
}

# Resume training from checkpoint
resume() {
    if [ -z "$1" ]; then
        # Look for latest checkpoint
        CHECKPOINT=$(ls -t $CHECKPOINT_DIR/checkpoint_*.pt 2>/dev/null | head -1)
        if [ -z "$CHECKPOINT" ]; then
            CHECKPOINT="$CHECKPOINT_DIR/best_model.pt"
        fi
        if [ ! -f "$CHECKPOINT" ]; then
            print_error "No checkpoint found. Use: ./run.sh train"
            exit 1
        fi
    else
        CHECKPOINT=$1
    fi
    
    print_msg "Resuming from: $CHECKPOINT"
    check_requirements
    
    $PYTHON main.py --resume "$CHECKPOINT" --config $CONFIG_FILE "${@:2}"
}

# Evaluate the model
evaluate() {
    print_msg "Evaluating model..."
    check_requirements
    
    CHECKPOINT="${1:-$CHECKPOINT_DIR/best_model.pt}"
    
    if [ ! -f "$CHECKPOINT" ]; then
        print_error "Model not found: $CHECKPOINT"
        exit 1
    fi
    
    $PYTHON main.py --evaluate --checkpoint "$CHECKPOINT" "${@:2}"
}

# Run quick test
test() {
    print_msg "Running quick test..."
    check_requirements
    
    $PYTHON main.py --test
}

# List curriculum stages
stages() {
    print_msg "Curriculum stages:"
    $PYTHON main.py --list-stages
}

# Clean checkpoints and logs
clean() {
    print_warn "This will delete all checkpoints and logs."
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf $CHECKPOINT_DIR
        rm -rf logs
        print_msg "Cleaned successfully ✓"
    else
        print_msg "Cancelled"
    fi
}

# Show help
show_help() {
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           🎮 Minecraft RL Bot - Training Script              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Usage: ./run.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  train              Start training with default config"
    echo "  stage <name>       Start training at specific curriculum stage"
    echo "  resume [file]      Resume training from checkpoint"
    echo "  evaluate [file]    Evaluate model performance"
    echo "  test               Run quick functionality test"
    echo "  stages             List all curriculum stages"
    echo "  clean              Remove checkpoints and logs"
    echo "  help               Show this help message"
    echo ""
    echo "Curriculum Stages:"
    echo "  survival           Basic survival and movement"
    echo "  resource_gathering Mining and collecting items"
    echo "  tool_crafting      Crafting tools"
    echo "  nether_access      Building Nether portal"
    echo "  blaze_hunting      Finding Blaze rods"
    echo "  ender_pearl_hunt   Collecting Ender Pearls"
    echo "  end_preparation    Finding stronghold"
    echo "  dragon_fight       Defeating the dragon"
    echo "  full_game          Complete speedrun"
    echo ""
    echo "Examples:"
    echo "  ./run.sh train"
    echo "  ./run.sh stage survival"
    echo "  ./run.sh train --episodes 5000"
    echo "  ./run.sh resume checkpoints/checkpoint_100.pt"
    echo "  ./run.sh evaluate checkpoints/best_model.pt"
    echo ""
}

# Main entry point
case "${1:-help}" in
    train)
        shift
        train "$@"
        ;;
    stage)
        shift
        train_stage "$@"
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
    stages|list-stages)
        stages
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
