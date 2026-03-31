"""Evaluation script for trained models."""

import os
import yaml
import argparse
import numpy as np
import torch
from typing import Dict, List

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm_marl.envs.sar_env import SAREnv
from swarm_marl.algos.qmix import QMIXTrainer
from swarm_marl.baselines.random_walk import RandomWalkPolicy
from swarm_marl.baselines.static_pheromone import StaticPheromonePolicy
from swarm_marl.evaluation.metrics import MetricsTracker
from swarm_marl.evaluation.visualizer import TrainingVisualizer, GridVisualizer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts; override values take precedence."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def evaluate_policy(env: SAREnv, policy, n_episodes: int = 100,
                    render: bool = False, render_dir: str = None,
                    save_frames: bool = False) -> Dict:
    """Evaluate a policy over multiple episodes.

    Args:
        env: SAR environment
        policy: Policy to evaluate (QMIX trainer or baseline policy)
        n_episodes: Number of episodes to evaluate
        render: Whether to render episodes
        render_dir: Directory to save renderings
        save_frames: If True, save a PNG per step in per-episode subfolders
                     instead of a single animated GIF per episode.

    Returns:
        Dictionary of metrics
    """
    tracker = MetricsTracker()
    episode_frames = []

    base_seed = env.config.get('seed', 42)
    for ep in range(n_episodes):
        observations, infos = env.reset(seed=base_seed + ep)
        tracker.reset_current()

        done = False
        episode_reward = 0

        # Initialize hidden states for QMIX
        if isinstance(policy, QMIXTrainer):
            hidden_states = np.zeros((env.num_agents, policy.hidden_dim))

        while not done:
            # Get actions
            if isinstance(policy, QMIXTrainer):
                obs_array = np.array([observations[f'agent_{i}'] for i in range(env.num_agents)])
                actions, hidden_states = policy.select_actions(
                    observations=obs_array,
                    hidden_states=hidden_states,
                    epsilon=0.0  # No exploration during evaluation
                )
                actions_dict = {f'agent_{i}': actions[i] for i in range(env.num_agents)}
            else:
                actions_dict = policy.get_actions(observations, infos)

            # Step
            next_observations, rewards, terminations, truncations, infos = env.step(actions_dict)

            done = any(terminations.values()) or any(truncations.values())
            reward = rewards['agent_0']
            episode_reward += reward

            # Track metrics
            coverage = env.grid_world.get_coverage()
            comm_count = sum(1 for agent in env.agents_dict.values() if agent.communicated)
            tracker.add_step(coverage, reward, comm_count)

            # Render
            # if render and render_dir:
            #     visualizer = GridVisualizer(env.grid_size)
            #     current_step = tracker.current['steps']  # step index just recorded

            #     coverage_pct = int(env.grid_world.get_coverage() * 100)

            #     if save_frames:
            #         # Save individual PNG for this step
            #         ep_dir = os.path.join(render_dir, f'episode_{ep:03d}')
            #         os.makedirs(ep_dir, exist_ok=True)
            #         frame_path = os.path.join(
            #             ep_dir, f'step_{current_step:04d}_cov{coverage_pct:03d}.png'
            #         )
            #         frame = visualizer.render(
            #             env.grid_world,
            #             env.pheromone_field,
            #             env.agents_dict,
            #             env.comm_model,
            #             save_path=frame_path,
            #             title=f'Episode {ep} | Step {current_step} | Coverage {coverage_pct}%'
            #         )
            #     else:
            #         frame = visualizer.render(
            #             env.grid_world,
            #             env.pheromone_field,
            #             env.agents_dict,
            #             env.comm_model
            #         )
            #     episode_frames.append(frame)

            observations = next_observations

        # End episode
        metrics = tracker.end_episode(any(terminations.values()))

        # if render and render_dir and episode_frames:
        #     from PIL import Image
        #     os.makedirs(render_dir, exist_ok=True)
        #     images = [Image.fromarray(f) for f in episode_frames]
        #     images[0].save(
        #         os.path.join(render_dir, f'episode_{ep:03d}.gif'),
        #         save_all=True,
        #         append_images=images[1:],
        #         duration=200,
        #         loop=0
        #     )
        #     episode_frames = []

        # if save_frames and render_dir:
        #     ep_dir = os.path.join(render_dir, f'episode_{ep:03d}')
        #     n_frames = tracker.current['steps'] if tracker.current else len(tracker.episode_data[-1]['coverages'])
        #     outcome = 'COMPLETED' if tracker.episode_data[-1]['terminated'] else 'TIMEOUT'
        #     print(f"  Episode {ep}: saved {len(os.listdir(ep_dir)) if os.path.isdir(ep_dir) else 0} frames → {ep_dir}  [{outcome}]")

    # Aggregate metrics
    all_metrics = tracker.get_all_metrics()
    summary = tracker.aggregate_metrics()

    return {
        'all_metrics': all_metrics,
        'summary': summary,
        'raw_episodes': tracker.episode_data,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate policies on SAR environment')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--env-config', type=str, default=None,
                       help='Path to environment config')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to QMIX checkpoint')
    parser.add_argument('--policy', type=str, default='qmix',
                       choices=['qmix', 'random', 'static'],
                       help='Policy to evaluate')
    parser.add_argument('--n-episodes', type=int, default=100,
                       help='Number of episodes to evaluate')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Override max steps per episode (default: use config value)')
    parser.add_argument('--render', action='store_true', default=True,
                       help='Render episodes')
    parser.add_argument('--render-dir', type=str, default='results/renders',
                       help='Directory to save renderings')
    parser.add_argument('--save-frames', action='store_true', default=False,
                       help='Save a PNG per step in per-episode subfolders (instead of animated GIF)')
    parser.add_argument('--save-results', type=str, default='results/eval_results.yaml',
                       help='Path to save results')

    args = parser.parse_args()
    print(f"Evaluating against configuration: {args}")

    # Load config
    config = load_config(args.config)

    if args.env_config:
        env_config = load_config(args.env_config)
        config = deep_merge(config, env_config)

    # Override max_steps if specified via CLI
    if args.max_steps is not None:
        config['env']['max_steps'] = args.max_steps
        print(f"[INFO] max_steps overridden to {args.max_steps}")

    # Set seed
    seed = config.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create environment
    env = SAREnv(config)
    env.reset(seed=seed)

    # Create policy
    if args.policy == 'qmix':
        if not args.checkpoint:
            print("Error: --checkpoint required for QMIX policy")
            return

        # Create trainer
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
            config=config,
            device=device
        )

        trainer.load(args.checkpoint)
        policy = trainer
        print(f"Loaded QMIX model from {args.checkpoint}")

    elif args.policy == 'random':
        policy = RandomWalkPolicy(
            num_agents=env.num_agents,
            action_space_size=env.action_spaces['agent_0'].n,
            seed=seed
        )
        print("Using Random Walk policy")

    elif args.policy == 'static':
        policy = StaticPheromonePolicy(
            num_agents=env.num_agents,
            grid_size=env.grid_size,
            tau=config['agent']['tau_default'],
            always_communicate=True,
            seed=seed
        )
        print("Using Static Pheromone policy")

    # Evaluate
    print(f"\nEvaluating {args.policy} over {args.n_episodes} episodes...")
    results = evaluate_policy(
        env, policy, args.n_episodes,
        render=args.render, render_dir=args.render_dir,
        save_frames=args.save_frames
    )

    # Print results
    print("\n" + "="*50)
    print(f"Evaluation Results: {args.policy}")
    print("="*50)
    for metric, (mean, std) in results['summary'].items():
        print(f"{metric:20s}: {mean:.4f} +/- {std:.4f}")

    # Per-episode breakdown: steps, final coverage, and outcome
    print("\n--- Per-Episode Breakdown ---")
    print(f"{'Ep':>4}  {'Steps':>6}  {'Coverage':>9}  {'Outcome'}")
    print("-" * 38)
    for i, ep in enumerate(results['raw_episodes']):
        steps = ep['steps']
        coverage = ep['coverages'][-1] if ep['coverages'] else 0.0
        outcome = "COMPLETED" if ep['terminated'] else f"TIMEOUT (max={config['env']['max_steps']})"
        print(f"{i+1:>4}  {steps:>6}  {coverage:>8.1%}  {outcome}")

    # Save results
    if args.save_results:
        import yaml
        os.makedirs(os.path.dirname(args.save_results), exist_ok=True)

        # Convert to serializable format
        save_data = {
            'policy': args.policy,
            'n_episodes': args.n_episodes,
            'summary': {k: {'mean': float(m), 'std': float(s)}
                       for k, (m, s) in results['summary'].items()}
        }

        with open(args.save_results, 'w') as f:
            yaml.dump(save_data, f)
        print(f"\nResults saved to {args.save_results}")


if __name__ == '__main__':
    main()
