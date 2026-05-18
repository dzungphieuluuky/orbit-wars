# Orbit Wars Reinforcement Learning

A reinforcement learning project that trains an intelligent agent to play **Orbit Wars** using Proximal Policy Optimization (PPO). The agent learns to control planetary fleets in a strategy-based space combat game.

## Overview

This project implements a complete RL pipeline for the Kaggle Orbit Wars competition, including:
- **Training**: PPO agent with configurable self-play and opponent selection
- **Evaluation**: Scripts to test the trained agent against various opponents
- **Visualization**: HTML replay generation for game playback
- **Tutorial**: Jupyter notebook demonstrating the full workflow

## Game Description

Orbit Wars is a turn-based strategy game where players control planets and fleets:
- Planets generate ships based on their production rate
- Players send fleets to attack enemy planets
- Victory condition: eliminate all opponent planets
- The game board is a 100×100 space with up to 48 planets

## Project Structure

```
src/
├── config.py          # Configuration dataclasses (TrainConfig, EnvConfig, etc.)
├── env.py             # Orbit Wars environment wrapper
├── game_types.py      # Game state data structures
├── features.py        # Feature encoding for neural network input
├── policy.py          # Neural network policy (actor-critic)
├── ppo.py             # PPO training algorithm
├── opponents.py       # Opponent implementations (random, self-play)
└── train.py           # Main training script

artifacts/
└── orbit_wars_ppo/    # Trained model checkpoints
    ├── ckpt_000050.pth
    ├── ckpt_000100.pth
    ├── ...
    └── ckpt_last.pth

play_vs_sniper.py      # Play one game and save HTML replay
eval_vs_sniper.py      # Evaluate agent against sniper opponent
```

## Installation

```bash
# Install dependencies
pip install torch kaggle-environments numpy pyyaml

# For Jupyter notebook
pip install jupyter
```

## Quick Start

### 1. Train the Agent

```bash
python src/train.py --config default_cfg.yaml
```

The training script will:
- Initialize a PPO agent
- Run self-play training for 3000 updates
- Save checkpoints every 50 updates
- Evaluate against a baseline opponent

### 2. Play Against the Sniper Opponent

```bash
python play_vs_sniper.py --checkpoint artifacts/orbit_wars_ppo/ckpt_001300.pth
```

This generates an HTML replay file at `artifacts/orbit_wars_ppo/replays/vs_sniper.html`

### 3. Evaluate Agent Performance

```bash
python eval_vs_sniper.py --checkpoint artifacts/orbit_wars_ppo/ckpt_001300.pth
```

Runs multiple evaluation episodes and reports win rate statistics.

### 4. Interactive Tutorial

```bash
jupyter notebook orbit_wars_reinforcement_learning_tutorial.ipynb
```

## Configuration

Edit default_cfg.yaml to customize training:

```yaml
# Training parameters
seed: 42
run_name: orbit_wars_ppo
opponent: self  # "random" or "self" for self-play
self_play_update_interval: 50
self_play_deterministic: false
alternate_player_sides: true

# Environment
env:
  candidate_count: 8         # Max action targets per planet
  ship_bucket_count: 8       # Discretization for ship counts

# Neural network
model:
  hidden_size: 128

# PPO hyperparameters
ppo:
  rollout_steps: 64
  num_envs: 2                # Parallel environments
  total_updates: 3000        # Training iterations
  epochs: 4
  minibatch_size: 256
  gamma: 0.99               # Discount factor
  clip_coef: 0.2            # PPO clipping parameter
  ent_coef: 0.01            # Entropy bonus
  vf_coef: 0.5              # Value function loss weight
  lr: 0.0003                # Learning rate
  max_grad_norm: 0.5
```

## Features

- **Self-Play Training**: Agent improves by playing against previous versions of itself
- **Flexible Opponents**: Train against random or other trained agents
- **Checkpointing**: Regular model saves with easy resumption
- **Batch Processing**: Efficient multi-environment parallel training
- **Feature Encoding**: Sophisticated state representation for neural network

## Model Architecture

The policy network (`PlanetPolicy`) uses:
- **Input features**: Self features, candidate features, global features
- **Attention mechanism**: Handles variable number of planets per game
- **Output**: Action distribution over available planets + ship count selection

## Training Results

The project includes 26 checkpoints (50 to 1300 updates) from a trained agent. Latest checkpoint is `ckpt_last.pth`.

## Args Reference

### Training Script
```bash
python src/train.py --config <path_to_config>
```

### Play Script
```bash
python play_vs_sniper.py \
  --config <config_file> \
  --checkpoint <model_path> \
  --output <replay_file> \
  --seed <seed> \
  --deterministic
```

### Evaluation Script
```bash
python eval_vs_sniper.py \
  --config <config_file> \
  --checkpoint <model_path> \
  --seed <seed>
```

## Dependencies

- **torch**: Deep learning framework
- **numpy**: Numerical computing
- **pyyaml**: Configuration file parsing
- **kaggle-environments**: Orbit Wars game environment

## License

MIT License

## References

- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [Kaggle Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)

---
