#!/usr/bin/env python3
"""
main.py - Main entry point for Minecraft RL Bot.

This script provides CLI access to train and evaluate the Minecraft
reinforcement learning agent.

Usage:
    # Start training with default settings
    python main.py --train
    
    # Start training at a specific curriculum stage
    python main.py --stage survival
    python main.py --stage resource_gathering
    python main.py --stage tool_crafting
    
    # Resume training from checkpoint
    python main.py --resume checkpoints/best_model.pt
    
    # Evaluate a trained model
    python main.py --evaluate checkpoints/best_model.pt
    
    # Quick test to verify installation
    python main.py --test

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
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training import Trainer, TrainingConfig, CurriculumStage, list_all_stages
from agent import Policy, PolicyConfig, ModelConfig
from env import MinecraftEnv, DiscreteAction
from utils import set_seed, load_config


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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Minecraft RL Bot - Train an AI to beat Minecraft",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train                    Start training
  python main.py --stage survival           Start at survival stage
  python main.py --resume checkpoint.pt     Resume from checkpoint
  python main.py --evaluate model.pt        Evaluate trained model
  python main.py --test                     Test installation
  python main.py --list-stages              List curriculum stages
        """
    )
    
    # Mode selection
    mode_group = parser.add_argument_group('Mode')
    mode_group.add_argument('--train', action='store_true',
                           help='Start training')
    mode_group.add_argument('--evaluate', action='store_true',
                           help='Evaluate a trained model')
    mode_group.add_argument('--test', action='store_true',
                           help='Run quick installation test')
    mode_group.add_argument('--list-stages', action='store_true',
                           help='List all curriculum stages')
    
    # Training options
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument('--config', type=str, default='config.yaml',
                            help='Path to configuration file')
    train_group.add_argument('--stage', type=str, default=None,
                            help='Curriculum stage to start from')
    train_group.add_argument('--resume', type=str, default=None,
                            help='Path to checkpoint to resume from')
    train_group.add_argument('--checkpoint', type=str, default=None,
                            help='Path to checkpoint for evaluation')
    train_group.add_argument('--episodes', type=int, default=None,
                            help='Number of episodes to train')
    train_group.add_argument('--seed', type=int, default=None,
                            help='Random seed')
    train_group.add_argument('--learning-rate', type=float, default=None,
                            help='Learning rate')
    train_group.add_argument('--batch-size', type=int, default=None,
                            help='Batch size')
    
    # Evaluation options
    eval_group = parser.add_argument_group('Evaluation Options')
    eval_group.add_argument('--eval-episodes', type=int, default=10,
                           help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    # Determine mode
    if args.test:
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
        print("\n💡 Quick start: python main.py --train")


if __name__ == "__main__":
    main()
