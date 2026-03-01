"""Automated experiment runner for comparing policies."""

import os
import yaml
import argparse
import numpy as np
import torch
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm_marl.envs.sar_env import SAREnv
from swarm_marl.algos.qmix import QMIXTrainer
from swarm_marl.baselines.random_walk import RandomWalkPolicy
from swarm_marl.baselines.static_pheromone import StaticPheromonePolicy
from swarm_marl.evaluation.metrics import MetricsTracker, compare_policies
from swarm_marl.evaluation.visualizer import TrainingVisualizer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_experiment(env_config: dict, policy_type: str,
                   checkpoint_path: str = None, n_episodes: int = 100,
                   fail_pct: float = 0.0, seed: int = 42) -> List[Dict]:
    """Run experiment with given configuration.

    Args:
        env_config: Environment configuration
        policy_type: 'qmix', 'random', or 'static'
        checkpoint_path: Path to QMIX checkpoint (if policy_type='qmix')
        n_episodes: Number of episodes to run
        fail_pct: Percentage of agents to fail (for fault tolerance)
        seed: Random seed

    Returns:
        List of episode metrics
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create environment
    env = SAREnv(env_config)
    env.reset(seed=seed)

    # Create policy
    if policy_type == 'qmix':
        if not checkpoint_path:
            raise ValueError("Checkpoint required for QMIX")

        sample_obs = env.observation_spaces['agent_0'].sample()
        sample_state = env.state_space.sample()
        obs_dim = sample_obs.shape[0]
        state_dim = sample_state.shape[0]
        n_actions = env.action_spaces['agent_0'].n
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        trainer = QMIXTrainer(
            obs_dim=obs_dim,
            state_dim=state_dim,
            n_actions=n_actions,
            n_agents=env.num_agents,
            config=env_config,
            device=device
        )
        trainer.load(checkpoint_path)
        policy = trainer
    elif policy_type == 'random':
        policy = RandomWalkPolicy(
            num_agents=env.num_agents,
            action_space_size=env.action_spaces['agent_0'].n,
            seed=seed
        )
    elif policy_type == 'static':
        policy = StaticPheromonePolicy(
            num_agents=env.num_agents,
            grid_size=env.grid_size,
            tau=env_config['agent']['tau_default'],
            always_communicate=True,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    # Run episodes
    tracker = MetricsTracker()

    for ep in range(n_episodes):
        observations, infos = env.reset(seed=seed + ep)
        tracker.reset_current()

        done = False
        episode_reward = 0

        # Fault tolerance: disable agents
        disabled_agents = set()
        fail_timestep = None
        if fail_pct > 0:
            n_disable = int(env.num_agents * fail_pct)
            disabled_agents = set(np.random.choice(env.num_agents, n_disable, replace=False))
            fail_timestep = np.random.randint(10, env.max_steps - 10)

        # Initialize hidden states for QMIX
        if policy_type == 'qmix':
            hidden_states = np.zeros((env.num_agents, policy.hidden_dim))

        step = 0
        while not done:
            # Get actions
            if policy_type == 'qmix':
                # Skip disabled agents
                obs_array = np.array([observations[f'agent_{i}']
                                     if i not in disabled_agents else np.zeros(obs_dim)
                                     for i in range(env.num_agents)])
                actions, hidden_states = policy.select_actions(
                    observations=obs_array,
                    hidden_states=hidden_states,
                    epsilon=0.0
                )
                # Force disabled agents to stay
                for i in disabled_agents:
                    actions[i] = 4  # STAY
                actions_dict = {f'agent_{i}': actions[i] for i in range(env.num_agents)}
            else:
                if step == fail_timestep:
                    for i in disabled_agents:
                        env.agents_dict[f'agent_{i}'].communicated = False
                actions_dict = policy.get_actions(observations, infos)
                for i in disabled_agents:
                    actions_dict[f'agent_{i}'] = 4  # STAY

            # Step
            next_observations, rewards, terminations, truncations, infos = env.step(actions_dict)

            done = any(terminations.values()) or any(truncations.values())
            reward = rewards['agent_0']
            episode_reward += reward

            # Track metrics
            coverage = env.grid_world.get_coverage()
            comm_count = sum(1 for agent in env.agents_dict.values() if agent.communicated)
            tracker.add_step(coverage, reward, comm_count)

            observations = next_observations
            step += 1

        tracker.end_episode(any(terminations.values()))

    return tracker.get_all_metrics()


def experiment_1_baseline_comparison(config: dict, save_dir: str, n_episodes: int = 100):
    """Experiment 1: Compare QMIX vs baselines."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Baseline Comparison")
    print("="*60)

    results = {}

    # Random walk
    print("\nRunning Random Walk...")
    results['random'] = run_experiment(config, 'random', n_episodes=n_episodes)

    # Static pheromone
    print("\nRunning Static Pheromone...")
    results['static'] = run_experiment(config, 'static', n_episodes=n_episodes)

    # QMIX (if checkpoint available)
    qmix_checkpoint = os.path.join(save_dir, 'final_model.pt')
    if os.path.exists(qmix_checkpoint):
        print("\nRunning QMIX...")
        results['qmix'] = run_experiment(config, 'qmix', qmix_checkpoint, n_episodes=n_episodes)
    else:
        print(f"\nQMIX checkpoint not found at {qmix_checkpoint}, skipping QMIX evaluation")

    # Compare and visualize
    comparison = compare_policies(results)

    # Print results
    print("\n" + "-"*60)
    print("Results Summary:")
    print("-"*60)
    for policy, metrics in comparison.items():
        print(f"\n{policy.upper()}:")
        for metric, (mean, std) in metrics.items():
            print(f"  {metric:20s}: {mean:.4f} +/- {std:.4f}")

    # Save to CSV
    rows = []
    for policy, metrics in comparison.items():
        for metric, (mean, std) in metrics.items():
            rows.append({
                'policy': policy,
                'metric': metric,
                'mean': mean,
                'std': std,
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, 'experiment1_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Visualize
    visualizer = TrainingVisualizer(save_dir)
    visualizer.plot_comparison_bars(comparison,
                                      ['coverage_rate', 'total_reward', 'total_communications'],
                                      'experiment1_comparison.png')

    return results


def experiment_2_communication_denial(config: dict, save_dir: str, n_episodes: int = 50):
    """Experiment 2: Communication denial impact."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Communication Denial Impact")
    print("="*60)

    denial_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    results_by_level = {level: {} for level in denial_levels}

    for level in denial_levels:
        print(f"\nTesting with {level*100:.0f}% communication denial...")

        # Modify config for this level
        test_config = config.copy()

        # Create comm zones based on level
        grid_size = config['env']['grid_size']
        n_zones = int(level * 5)  # Scale zones with denial level

        denial_zones = []
        np.random.seed(42)
        for _ in range(n_zones):
            x1 = np.random.randint(0, grid_size - 5)
            y1 = np.random.randint(0, grid_size - 5)
            x2 = min(x1 + np.random.randint(2, 6), grid_size - 1)
            y2 = min(y1 + np.random.randint(2, 6), grid_size - 1)
            denial_zones.append([x1, y1, x2, y2])

        test_config['comm']['denial_zones'] = denial_zones
        test_config['comm']['model'] = 'fixed_denial'

        # Test policies
        results_by_level[level]['random'] = run_experiment(test_config, 'random', n_episodes=n_episodes)
        results_by_level[level]['static'] = run_experiment(test_config, 'static', n_episodes=n_episodes)

        qmix_checkpoint = os.path.join(save_dir, 'final_model.pt')
        if os.path.exists(qmix_checkpoint):
            results_by_level[level]['qmix'] = run_experiment(test_config, 'qmix', qmix_checkpoint, n_episodes=n_episodes)

    # Plot degradation curves
    visualizer = TrainingVisualizer(save_dir)

    # Collect metrics
    metrics_to_plot = ['coverage_rate', 'total_communications']
    for metric in metrics_to_plot:
        data = {}
        for policy in ['random', 'static', 'qmix']:
            values = []
            for level in denial_levels:
                if policy in results_by_level[level]:
                    metric_data = [m[metric] for m in results_by_level[level][policy]]
                    values.append(np.mean(metric_data))
                else:
                    values.append(0)
            if values and max(values) > 0:
                data[policy] = values

        if data:
            visualizer.plot_degradation_curves(
                [l*100 for l in denial_levels],
                data,
                f'experiment2_{metric}_degradation.png'
            )

    return results_by_level


def experiment_3_fault_tolerance(config: dict, save_dir: str, n_episodes: int = 50):
    """Experiment 3: Fault tolerance."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Fault Tolerance")
    print("="*60)

    fail_rates = [0.0, 0.1, 0.2, 0.3]
    results = {}

    for fail_rate in fail_rates:
        print(f"\nTesting with {fail_rate*100:.0f}% agent failure...")

        # Run without failures
        baseline = run_experiment(config, 'static', fail_pct=0.0, n_episodes=n_episodes)

        # Run with failures
        failures = run_experiment(config, 'static', fail_pct=fail_rate, n_episodes=n_episodes)

        results[fail_rate] = {
            'baseline': baseline,
            'failures': failures,
        }

        # Compute fault tolerance
        tracker = MetricsTracker()
        ft = tracker.fault_tolerance(baseline, failures)
        print(f"  Fault Tolerance: {ft:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run comparison experiments')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--env-config', type=str, default='configs/envs/medium.yaml')
    parser.add_argument('--save-dir', type=str, default='results')
    parser.add_argument('--experiments', type=str, default='1,2,3',
                       help='Comma-separated list of experiments to run')
    parser.add_argument('--n-episodes', type=int, default=100,
                       help='Number of episodes per experiment')

    args = parser.parse_args()

    # Load configs
    base_config = load_config(args.config)
    if args.env_config:
        env_config = load_config(args.env_config)
        base_config.update(env_config)

    # Parse experiments
    experiments = [int(x.strip()) for x in args.experiments.split(',')]

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Run experiments
    for exp_num in experiments:
        if exp_num == 1:
            experiment_1_baseline_comparison(base_config, args.save_dir, args.n_episodes)
        elif exp_num == 2:
            experiment_2_communication_denial(base_config, args.save_dir, args.n_episodes)
        elif exp_num == 3:
            experiment_3_fault_tolerance(base_config, args.save_dir, args.n_episodes)
        else:
            print(f"Unknown experiment: {exp_num}")

    print("\nAll experiments complete!")


if __name__ == '__main__':
    main()
