#!/bin/bash
# =============================================================================
# Minecraft Elytra Finder Bot - Live Bot Runner
# =============================================================================
# This script starts the live Elytra Finder Bot.
#
# Usage:
#   ./scripts/run_live_bot.sh                    # Normal run
#   ./scripts/run_live_bot.sh --dry-run          # Test without sending commands
#   ./scripts/run_live_bot.sh --help             # Show help
#
# Environment:
#   - Set MC_USERNAME and MC_PASSWORD environment variables, or
#   - Create a .env file from .env.example
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON=${PYTHON:-python3}
CONFIG_FILE="${PROJECT_DIR}/config.yaml"
VENV_DIR="${PROJECT_DIR}/venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_msg() {
    echo -e "${GREEN}[Elytra Bot]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[Warning]${NC} $1"
}

print_error() {
    echo -e "${RED}[Error]${NC} $1"
}

# Load .env file if it exists
if [ -f "${PROJECT_DIR}/.env" ]; then
    print_msg "Loading environment from .env file"
    set -a
    source "${PROJECT_DIR}/.env"
    set +a
fi

# Activate virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    print_msg "Activating virtual environment"
    source "${VENV_DIR}/bin/activate"
fi

# Check Python
if ! command -v $PYTHON &> /dev/null; then
    print_error "Python not found. Please install Python 3.8+"
    exit 1
fi

# Check dependencies
print_msg "Checking dependencies..."
if ! $PYTHON -c "import torch; import numpy; import yaml" 2>/dev/null; then
    print_warn "Some dependencies are missing. Installing..."
    pip install -r "${PROJECT_DIR}/requirements.txt"
fi

# Change to project directory
cd "$PROJECT_DIR"

# Print banner
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           🚀 Minecraft Elytra Finder Bot                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Safety reminder
echo "⚠️  SAFETY REMINDER:"
echo "This bot is intended to be used only where automation is"
echo "explicitly allowed by the server owner. Do not use this"
echo "in violation of any server's terms of service."
echo ""

# Run the bot
print_msg "Starting live bot..."
$PYTHON main.py live-bot --config "$CONFIG_FILE" "$@"
