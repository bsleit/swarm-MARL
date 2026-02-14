# Multi-Agent Reinforcement Learning for Swarm Robotics SAR

A complete Multi-Agent Reinforcement Learning (MARL) system for Search and Rescue (SAR) coordination in swarm robotics. The system implements QMIX with bio-inspired pheromone-based task allocation and configurable communication models.

## Features

- **PettingZoo Environment**: 2D grid world with configurable obstacles, communication denial zones
- **Bio-inspired Baseline**: [InProgress] Pheromone-based task allocation with fixed threshold
- **QMIX Implementation**:  [InProgress] Custom implementation following EPyMARL patterns
- **Evaluation Metrics**: Coverage rate, steps to completion, communication cost

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

## Project Structure

```
swarm_marl/
├── configs/              # YAML configuration files
│   ├── default.yaml      # Default hyperparameters
│   ├── envs/             # Environment configs
│   │   ├── small.yaml    # 20x20, 5 agents
│   │   ├── medium.yaml   # 30x30, 10 agents
│   │   └── large.yaml    # 50x50, 20 agents
│   └── comms/            # Communication configs
│       ├── fixed_denial.yaml
│       ├── probabilistic.yaml
│       └── distance_zones.yaml
├── envs/                 # PettingZoo environment
│   ├── sar_env.py        # Main ParallelEnv
│   ├── grid_world.py     # Grid, obstacles, comm zones
│   ├── agent.py          # Agent state & logic
│   ├── pheromone.py      # Pheromone field mechanics
│   └── comm_model.py     # Communication models
├── models/               # Neural network architectures
│   ├── agent_network.py  # Per-agent Q-network
│   └── mixing_network.py # QMIX mixing network
├── algos/                # MARL algorithms
│   ├── qmix.py           # QMIX trainer
│   ├── replay_buffer.py  # Experience replay
│   └── epsilon.py        # Exploration schedule
├── baselines/            # Baseline policies
│   ├── random_walk.py
│   └── static_pheromone.py
├── evaluation/           # Metrics & visualizers
│   ├── metrics.py
│   └── visualizer.py
├── scripts/              # Entry points
│   ├── train.py
│   ├── evaluate.py
│   ├── compare.py
│   └── visualize.py
└── tests/                # Unit tests
    ├── test_env.py
    ├── test_pheromone.py
    ├── test_qmix.py
    └── test_baselines.py
```

## Configuration

The default configuration includes:

### Environment
- Grid sizes: 20×20, 30×30, 50×50
- Agent counts: 5, 10, 20
- Obstacle density: 15%
- Max steps: 200-1000 depending on size

### QMIX Training
- Learning rate: 5e-4
- Discount factor: 0.99
- Batch size: 32 episodes
- Replay buffer: 5000 episodes
- Target update: Every 200 episodes
- Epsilon decay: 1.0 → 0.05 over 50k steps

### Rewards
- α (coverage): 1.0
- β (step penalty): 0.01
- γ (communication cost): 0.1

## Running Tests

```bash
# Run all tests
pytest swarm_marl/tests/

# Run specific test file
pytest swarm_marl/tests/test_env.py -v
pytest swarm_marl/tests/test_qmix.py -v
```

## Key Metrics

- **C_R (Coverage Rate)**: Percentage of explorable area mapped
- **T_C (Time to Completion)**: Steps to reach 90% coverage
- **E_comm (Communication Energy)**: Total transmission count

## Citation

If you use this code in your research, please cite:

```bibtex
@software{swarm_marl_sar,
  title={MARL Swarm SAR: Multi-Agent Reinforcement Learning for Swarm Robotics Search and Rescue},
  year={2026},
  url={https://github.com/bsleit/swarm-MARL.git}
}
```
