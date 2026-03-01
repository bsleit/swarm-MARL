"""Visualization script for rendering episodes."""

import os
import yaml
import argparse
import numpy as np
import torch

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm_marl.envs.sar_env import SAREnv
from swarm_marl.algos.qmix import QMIXTrainer
from swarm_marl.baselines.random_walk import RandomWalkPolicy
from swarm_marl.baselines.static_pheromone import StaticPheromonePolicy
from swarm_marl.evaluation.visualizer import GridVisualizer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def render_episode(env: SAREnv, policy, save_dir: str, episode_id: int = 0):
    """Render a single episode and save frames."""
    os.makedirs(save_dir, exist_ok=True)

    observations, infos = env.reset(seed=episode_id)

    visualizer = GridVisualizer(env.grid_size)
    frames = []

    done = False
    step = 0

    # Initialize hidden states for QMIX
    if isinstance(policy, QMIXTrainer):
        hidden_states = np.zeros((env.num_agents, policy.hidden_dim))

    while not done and step < env.max_steps:
        # Render
        frame = visualizer.render(
            env.grid_world,
            env.pheromone_field,
            env.agents_dict,
            env.comm_model,
            save_path=os.path.join(save_dir, f'frame_{step:04d}.png')
        )
        frames.append(frame)

        # Get actions
        if isinstance(policy, QMIXTrainer):
            obs_array = np.array([observations[f'agent_{i}'] for i in range(env.num_agents)])
            actions, hidden_states = policy.select_actions(
                observations=obs_array,
                hidden_states=hidden_states,
                epsilon=0.0
            )
            actions_dict = {f'agent_{i}': actions[i] for i in range(env.num_agents)}
        else:
            actions_dict = policy.get_actions(observations, infos)

        # Step
        next_observations, rewards, terminations, truncations, infos = env.step(actions_dict)

        done = any(terminations.values()) or any(truncations.values())
        observations = next_observations
        step += 1

    # Save GIF
    if frames:
        try:
            from PIL import Image
            images = [Image.fromarray(f) for f in frames]
            images[0].save(
                os.path.join(save_dir, f'episode_{episode_id}.gif'),
                save_all=True,
                append_images=images[1:],
                duration=200,
                loop=0
            )
            print(f"Saved GIF: {os.path.join(save_dir, f'episode_{episode_id}.gif')}")
        except ImportError:
            print("PIL not available, frames saved as PNGs")

    # Final frame
    visualizer.render(
        env.grid_world,
        env.pheromone_field,
        env.agents_dict,
        env.comm_model,
        save_path=os.path.join(save_dir, f'final_frame.png')
    )

    print(f"Episode complete: {step} steps")
    print(f"Final coverage: {env.grid_world.get_coverage():.3f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize SAR episodes')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--env-config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--policy', type=str, default='qmix',
                       choices=['qmix', 'random', 'static'])
    parser.add_argument('--save-dir', type=str, default='results/visualizations')
    parser.add_argument('--episode-id', type=int, default=0)

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.env_config:
        env_config = load_config(args.env_config)
        config.update(env_config)

    # Create environment
    seed = config.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = SAREnv(config)

    # Create policy
    if args.policy == 'qmix':
        if not args.checkpoint:
            print("Error: --checkpoint required for QMIX")
            return

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
    elif args.policy == 'random':
        policy = RandomWalkPolicy(
            num_agents=env.num_agents,
            action_space_size=env.action_spaces['agent_0'].n,
            seed=seed
        )
    else:  # static
        policy = StaticPheromonePolicy(
            num_agents=env.num_agents,
            grid_size=env.grid_size,
            tau=config['agent']['tau_default'],
            always_communicate=True,
            seed=seed
        )

    # Render
    render_episode(env, policy, args.save_dir, args.episode_id)


if __name__ == '__main__':
    main()
