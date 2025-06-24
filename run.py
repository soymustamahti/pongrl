#!/usr/bin/env python3
"""
Quick start script for training and evaluating a DQN agent on Pong.
This script provides an easy way to run the training with different configurations.
"""

import argparse
import os
from train_pong import train_dqn_pong, evaluate_agent

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate DQN agent on Pong')
    parser.add_argument('--mode', choices=['train', 'eval', 'both'], default='train',
                        help='Mode: train, eval, or both (default: train)')
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of training episodes (default: 2000)')
    parser.add_argument('--save-interval', type=int, default=500,
                        help='Save model every N episodes (default: 500)')
    parser.add_argument('--model-path', type=str, default='models/dqn_pong_final.pth',
                        help='Path to model for evaluation (default: models/dqn_pong_final.pth)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, default: auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during evaluation')
    parser.add_argument('--record', action='store_true',
                        help='Record episodes during evaluation')
    parser.add_argument('--quick', action='store_true',
                        help='Quick training mode (100 episodes, save every 50)')
    
    args = parser.parse_args()
    
    # Adjust parameters for quick mode
    if args.quick:
        args.episodes = 100
        args.save_interval = 50
        print("Quick training mode: 100 episodes, save every 50 episodes")
    
    if args.mode in ['train', 'both']:
        print(f"Starting training for {args.episodes} episodes...")
        print(f"Device: {args.device if args.device else 'auto'}")
        print(f"Seed: {args.seed}")
        
        agent, metrics = train_dqn_pong(
            num_episodes=args.episodes,
            save_interval=args.save_interval,
            model_save_path="models",
            results_save_path="training_results",
            seed=args.seed,
            device=args.device
        )
        print("Training completed!")
    
    if args.mode in ['eval', 'both']:
        if os.path.exists(args.model_path):
            print(f"Evaluating model: {args.model_path}")
            evaluate_agent(args.model_path, args.eval_episodes, args.render, args.record)
        else:
            print(f"Model file not found: {args.model_path}")
            print("Please train a model first or specify a valid model path.")

if __name__ == "__main__":
    main()
