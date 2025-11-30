#!/usr/bin/env python3
"""
main.py - Main entry point for Minecraft RL Bot.

This script provides CLI access to:
1. Train the Minecraft reinforcement learning agent
2. Run the live LemonCloud Elytra Finder Bot

Usage:
    # Training mode
    python main.py train                        Start training
    python main.py train --stage survival       Start at survival stage
    python main.py train --resume checkpoint.pt Resume from checkpoint
    
    # Live bot mode
    python main.py live-bot                     Run Elytra Finder Bot
    python main.py live-bot --dry-run           Test without sending commands
    
    # Legacy flag-based commands (still supported)
    python main.py --train
    python main.py --test
    python main.py --evaluate model.pt

Available Curriculum Stages:
    - basic_survival: Learn to survive and move around
    - resource_gathering: Mine blocks and collect items
    - tool_crafting: Craft tools using crafting systems
    - nether_access: Build and enter Nether portal
    - blaze_hunting: Find fortress and collect Blaze rods
    - ender_pearl_hunt: Collect Ender Pearls
    - end_preparation: Craft Eyes and find stronghold
    - dragon_fight: Enter End and defeat the dragon
    - full_game: Complete speedrun from start

SAFETY NOTE:
The live bot is intended to be used only where automation is explicitly
allowed by the server owner. Do not use this in violation of any
server's terms of service.
"""

import argparse
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training import Trainer, TrainingConfig, CurriculumStage, list_all_stages
from agent import Policy, PolicyConfig, ModelConfig
from env import MinecraftEnv, DiscreteAction
from utils import set_seed, load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def train(args):
    """Run training with the specified configuration."""
    print("=" * 60)
    print("🎮 Minecraft RL Bot - Training Mode")
    print("=" * 60)
    
    # Load configuration
    config_dict = load_config(args.config) if args.config else {}
    config = TrainingConfig(**config_dict) if config_dict else TrainingConfig()
    
    # Override from command line
    if args.episodes:
        config.num_episodes = args.episodes
    if args.seed:
        config.seed = args.seed
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Create trainer
    trainer = Trainer(config)
    
    # Set curriculum stage if specified
    if args.stage:
        stage_map = {s.name.lower(): s for s in CurriculumStage}
        stage_key = args.stage.lower().replace('-', '_').replace(' ', '_')
        if stage_key in stage_map:
            trainer.curriculum.set_stage(stage_map[stage_key])
            print(f"Starting at curriculum stage: {stage_map[stage_key].name}")
        else:
            print(f"Warning: Unknown stage '{args.stage}'. Using default.")
            print(f"Available stages: {list(stage_map.keys())}")
    
    # Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    results = trainer.train()
    
    # Print results
    print("\n" + "=" * 60)
    print("🏆 Training Complete!")
    print("=" * 60)
    print(f"Total Episodes: {results['total_episodes']}")
    print(f"Total Steps: {results['total_steps']}")
    print(f"Best Reward: {results['best_reward']:.2f}")
    print(f"Elapsed Time: {results['elapsed_time']:.1f} seconds")
    
    return results


def evaluate(args):
    """Evaluate a trained model."""
    print("=" * 60)
    print("🎮 Minecraft RL Bot - Evaluation Mode")
    print("=" * 60)
    
    if not args.checkpoint:
        print("Error: --checkpoint required for evaluation")
        return None
    
    # Load configuration
    config = TrainingConfig()
    
    # Create trainer
    trainer = Trainer(config)
    
    # Load checkpoint
    trainer.load_checkpoint(args.checkpoint)
    
    # Evaluate
    num_episodes = args.eval_episodes or 10
    print(f"\nEvaluating over {num_episodes} episodes...")
    
    results = trainer.evaluate(num_episodes=num_episodes)
    
    print("\n" + "=" * 60)
    print("📊 Evaluation Results")
    print("=" * 60)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Steps: {results['mean_steps']:.0f}")
    print(f"Completion Rate: {results['completion_rate']:.1%}")
    
    return results


def test(args):
    """Run a quick test to verify installation."""
    print("=" * 60)
    print("🧪 Minecraft RL Bot - Test Mode")
    print("=" * 60)
    
    print("\n1. Testing environment creation...")
    env = MinecraftEnv(seed=42)
    obs, info = env.reset()
    print(f"   ✓ Environment created")
    print(f"   ✓ Observation shape: blocks={obs['blocks'].shape}")
    
    print("\n2. Testing policy creation...")
    policy = Policy()
    print(f"   ✓ Policy created with PyTorch model")
    
    print("\n3. Testing action selection...")
    import numpy as np
    obs_batch = {
        'blocks': obs['blocks'][np.newaxis, ...],
        'inventory': obs['inventory'][np.newaxis, ...],
        'agent_state': obs['agent_state'][np.newaxis, ...]
    }
    action, log_prob, value = policy.act(obs_batch)
    print(f"   ✓ Action selected: {list(action.keys())}")
    print(f"   ✓ Value estimate: {value[0]:.4f}")
    
    print("\n4. Testing environment step...")
    action_obj = DiscreteAction(
        movement=int(action['movement'][0]),
        camera=int(action['camera'][0]),
        interaction=int(action['interaction'][0]),
        inventory=int(action['inventory'][0])
    )
    obs, reward, term, trunc, info = env.step(action_obj)
    print(f"   ✓ Step completed")
    print(f"   ✓ Reward: {reward:.4f}")
    
    print("\n5. Testing trainer creation...")
    config = TrainingConfig(num_episodes=1, max_steps_per_episode=10)
    trainer = Trainer(config)
    print(f"   ✓ Trainer created")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Installation is working correctly.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Start training: python main.py --train")
    print("  2. Or with a stage: python main.py --stage survival")
    
    return True


def list_stages(args):
    """List all curriculum stages."""
    print("=" * 60)
    print("📚 Minecraft RL Bot - Curriculum Stages")
    print("=" * 60)
    
    stages = list_all_stages()
    
    for i, stage in enumerate(stages):
        print(f"\n{i}. {stage['name']}")
        print(f"   {stage['description']}")
        print(f"   Objectives: {', '.join(stage['objectives'][:3])}")
        print(f"   Min episodes: {stage['min_episodes']}, Max: {stage['max_episodes']}")
    
    print("\n" + "=" * 60)
    print("Use --stage <name> to start at a specific stage.")
    print("Example: python main.py --stage resource_gathering")


def run_live_bot(args):
    """Run the live LemonCloud Elytra Finder Bot."""
    print("=" * 60)
    print("🚀 Minecraft Elytra Finder Bot - Live Mode")
    print("=" * 60)
    
    # Load configuration
    config_dict = load_config(args.config) if args.config else {}
    
    # Import live bot components
    from live_bot.controller import BotController, BotConfig
    
    # Build bot config from file and command line
    bot_config = BotConfig()
    
    # Load from config file if available
    if 'lemoncloud' in config_dict:
        lc = config_dict['lemoncloud']
        bot_config.host = lc.get('host', bot_config.host)
        bot_config.port = lc.get('port', bot_config.port)
        bot_config.login_commands = lc.get('login_commands', bot_config.login_commands)
        bot_config.go_to_end_commands = lc.get('go_to_end_commands', bot_config.go_to_end_commands)
    
    if 'account' in config_dict:
        acc = config_dict['account']
        bot_config.username = acc.get('username', bot_config.username)
    
    if 'bot_behavior' in config_dict:
        bb = config_dict['bot_behavior']
        bot_config.max_runtime_minutes = bb.get('max_runtime_minutes', bot_config.max_runtime_minutes)
        bot_config.search_radius_blocks = bb.get('search_radius_blocks', bot_config.search_radius_blocks)
        bot_config.control_mode = bb.get('control_mode', bot_config.control_mode)
    
    # Command line overrides
    if args.dry_run:
        bot_config.dry_run = True
    if args.host:
        bot_config.host = args.host
    if args.port:
        bot_config.port = args.port
    if args.username:
        bot_config.username = args.username
    if args.max_runtime:
        bot_config.max_runtime_minutes = args.max_runtime
    if args.control_mode:
        bot_config.control_mode = args.control_mode
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Host: {bot_config.host}:{bot_config.port}")
    print(f"  Username: {bot_config.username or '(not set)'}")
    print(f"  Control mode: {bot_config.control_mode}")
    print(f"  Max runtime: {bot_config.max_runtime_minutes} minutes")
    print(f"  Dry run: {bot_config.dry_run}")
    print()
    
    if bot_config.dry_run:
        print("⚠️  DRY RUN MODE: No actual commands will be sent to server")
        print()
    
    # Safety reminder
    print("=" * 60)
    print("⚠️  SAFETY REMINDER:")
    print("This bot is intended to be used only where automation is")
    print("explicitly allowed by the server owner. Do not use this")
    print("in violation of any server's terms of service.")
    print("=" * 60)
    print()
    
    # Create and run bot
    controller = BotController(bot_config)
    
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    finally:
        stats = controller.get_stats()
        print("\n" + "=" * 60)
        print("📊 Session Summary")
        print("=" * 60)
        print(f"Runtime: {stats['runtime_minutes']:.1f} minutes")
        print(f"Cities searched: {stats['cities_searched']}")
        print(f"Elytra found: {stats['elytra_found']}")
        print("=" * 60)


def main():
    """Main entry point with subcommand support."""
    parser = argparse.ArgumentParser(
        description="Minecraft RL Bot - Train AI or run live Elytra Finder Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training mode
  python main.py train                        Start training
  python main.py train --stage survival       Start at survival stage
  python main.py train --resume checkpoint.pt Resume from checkpoint
  
  # Live bot mode
  python main.py live-bot                     Run Elytra Finder Bot
  python main.py live-bot --dry-run           Test without sending commands
  
  # Legacy flag-based commands
  python main.py --train                      Start training
  python main.py --test                       Test installation
  python main.py --evaluate model.pt          Evaluate trained model
        """
    )
    
    # Create subparsers for train and live-bot
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train the RL agent')
    train_parser.add_argument('--config', type=str, default='config.yaml',
                             help='Path to configuration file')
    train_parser.add_argument('--stage', type=str, default=None,
                             help='Curriculum stage to start from')
    train_parser.add_argument('--resume', type=str, default=None,
                             help='Path to checkpoint to resume from')
    train_parser.add_argument('--episodes', type=int, default=None,
                             help='Number of episodes to train')
    train_parser.add_argument('--seed', type=int, default=None,
                             help='Random seed')
    train_parser.add_argument('--learning-rate', type=float, default=None,
                             help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=None,
                             help='Batch size')
    
    # Live-bot subcommand
    bot_parser = subparsers.add_parser('live-bot', help='Run the Elytra Finder Bot')
    bot_parser.add_argument('--config', type=str, default='config.yaml',
                           help='Path to configuration file')
    bot_parser.add_argument('--dry-run', action='store_true',
                           help='Simulate actions without sending commands')
    bot_parser.add_argument('--host', type=str, default=None,
                           help='Server host')
    bot_parser.add_argument('--port', type=int, default=None,
                           help='Server port')
    bot_parser.add_argument('--username', type=str, default=None,
                           help='Minecraft username')
    bot_parser.add_argument('--max-runtime', type=int, default=None,
                           help='Maximum runtime in minutes')
    bot_parser.add_argument('--control-mode', type=str, choices=['scripted', 'rl'],
                           default=None, help='Control mode')
    
    # Legacy mode arguments (for backward compatibility)
    legacy_group = parser.add_argument_group('Legacy Mode (use subcommands instead)')
    legacy_group.add_argument('--train', action='store_true',
                             help='Start training (legacy)')
    legacy_group.add_argument('--evaluate', action='store_true',
                             help='Evaluate a trained model (legacy)')
    legacy_group.add_argument('--test', action='store_true',
                             help='Run quick installation test')
    legacy_group.add_argument('--list-stages', action='store_true',
                             help='List all curriculum stages')
    legacy_group.add_argument('--config', type=str, default='config.yaml',
                             help='Path to configuration file')
    legacy_group.add_argument('--stage', type=str, default=None,
                             help='Curriculum stage to start from')
    legacy_group.add_argument('--resume', type=str, default=None,
                             help='Path to checkpoint to resume from')
    legacy_group.add_argument('--checkpoint', type=str, default=None,
                             help='Path to checkpoint for evaluation')
    legacy_group.add_argument('--episodes', type=int, default=None,
                             help='Number of episodes to train')
    legacy_group.add_argument('--seed', type=int, default=None,
                             help='Random seed')
    legacy_group.add_argument('--learning-rate', type=float, default=None,
                             help='Learning rate')
    legacy_group.add_argument('--batch-size', type=int, default=None,
                             help='Batch size')
    legacy_group.add_argument('--eval-episodes', type=int, default=10,
                             help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    # Handle subcommands
    if args.command == 'train':
        train(args)
    elif args.command == 'live-bot':
        run_live_bot(args)
    # Handle legacy flags
    elif args.test:
        test(args)
    elif args.list_stages:
        list_stages(args)
    elif args.evaluate:
        evaluate(args)
    elif args.train or args.stage or args.resume:
        train(args)
    else:
        # Default: show help
        parser.print_help()
        print("\n💡 Quick start:")
        print("  Training:  python main.py train")
        print("  Live bot:  python main.py live-bot --dry-run")


if __name__ == "__main__":
    main()
