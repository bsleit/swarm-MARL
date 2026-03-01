"""Main training script for QMIX."""

import os
import yaml
import argparse
import numpy as np
import torch
from tqdm import tqdm
import time
from typing import Optional

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm_marl.envs.sar_env import SAREnv
from swarm_marl.algos.qmix import QMIXTrainer
from swarm_marl.algos.replay_buffer import Episode
from swarm_marl.evaluation.visualizer import TrainingVisualizer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: dict, override_config: dict) -> dict:
    """Merge override config into base config."""
    merged = base_config.copy()

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                deep_update(d[k], v)
            else:
                d[k] = v

    deep_update(merged, override_config)
    return merged


def setup_environment(config: dict) -> str:
    """Setup device for training."""
    device_config = config.get('device', 'auto')

    if device_config == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"CUDA available, using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("CUDA not available, using CPU")
    else:
        device = device_config
        print(f"Using device: {device}")

    return device


def train(config: dict, device: str, save_dir: str,
          checkpoint_path: Optional[str] = None) -> None:
    """Main training loop."""

    # Set random seed
    seed = config.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create environment
    env = SAREnv(config)
    env.reset(seed=seed)

    # Get dimensions
    sample_obs = env.observation_spaces['agent_0'].sample()
    sample_state = env.state_space.sample()
    obs_dim = sample_obs.shape[0]
    state_dim = sample_state.shape[0]
    n_actions = env.action_spaces['agent_0'].n
    n_agents = env.num_agents

    print(f"Environment: {env.grid_size}x{env.grid_size}, {n_agents} agents")
    print(f"Observation dim: {obs_dim}, State dim: {state_dim}, Actions: {n_actions}")

    # Create trainer
    trainer = QMIXTrainer(
        obs_dim=obs_dim,
        state_dim=state_dim,
        n_actions=n_actions,
        n_agents=n_agents,
        config=config,
        device=device
    )

    # Load checkpoint if provided
    start_episode = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        trainer.load(checkpoint_path)
        start_episode = trainer.episode_count

    # Create results directory
    os.makedirs(save_dir, exist_ok=True)

    # Initialize visualizer
    visualizer = TrainingVisualizer(save_dir)

    # Training history
    history = {
        'episode_reward': [],
        'coverage': [],
        'loss': [],
        'epsilon': [],
    }

    # Training loop
    num_episodes = config['training']['num_episodes']
    log_interval = config.get('log_interval', 100)
    save_interval = config.get('save_interval', 1000)

    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"Results will be saved to: {save_dir}")

    for episode in tqdm(range(start_episode, num_episodes), desc="Training"):
        # Reset environment
        observations, infos = env.reset(seed=seed + episode)

        # Initialize episode
        episode_data = Episode(
            num_agents=n_agents,
            max_steps=config['env']['max_steps'],
            obs_dim=obs_dim,
            state_dim=state_dim,
            n_actions=n_actions
        )

        # Initialize hidden states
        hidden_states = np.zeros((n_agents, trainer.hidden_dim))

        # Episode loop
        done = False
        episode_reward = 0

        while not done:
            # Get current state
            state = env.get_global_state()

            # Select actions
            epsilon = trainer.get_epsilon()
            obs_array = np.array([observations[f'agent_{i}'] for i in range(n_agents)])

            actions, hidden_states = trainer.select_actions(
                observations=obs_array,
                hidden_states=hidden_states,
                epsilon=epsilon
            )

            # Convert to dict
            actions_dict = {f'agent_{i}': actions[i] for i in range(n_agents)}

            # Step environment
            next_observations, rewards, terminations, truncations, infos = env.step(actions_dict)

            # Check if done
            done = any(terminations.values()) or any(truncations.values())

            # Get reward (same for all agents)
            reward = rewards['agent_0']
            episode_reward += reward

            # Get next state
            next_state = env.get_global_state()

            # Get next observations array
            next_obs_array = np.array([next_observations.get(f'agent_{i}', np.zeros(obs_dim))
                                       for i in range(n_agents)])

            # Add to episode
            episode_data.add(
                state=state,
                observations=obs_array,
                actions=actions,
                reward=reward,
                terminated=done,
                mask=1.0
            )

            # Update for next step
            observations = next_observations
            obs_array = next_obs_array

            if done:
                break

        # Add episode to replay buffer
        trainer.replay_buffer.add(episode_data)
        trainer.increment_episode()

        # Get final coverage
        final_coverage = env.grid_world.get_coverage()

        # Train if enough episodes in buffer
        if trainer.replay_buffer.can_sample(trainer.batch_size):
            batch = trainer.replay_buffer.sample(trainer.batch_size)
            loss = trainer.train_step(batch)
            history['loss'].append(loss)

        # Log
        history['episode_reward'].append(episode_reward)
        history['coverage'].append(final_coverage)
        history['epsilon'].append(epsilon)

        if (episode + 1) % log_interval == 0:
            recent_rewards = history['episode_reward'][-log_interval:]
            recent_coverage = history['coverage'][-log_interval:]

            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Reward: {np.mean(recent_rewards):.2f} (+/- {np.std(recent_rewards):.2f})")
            print(f"  Coverage: {np.mean(recent_coverage):.3f}")
            print(f"  Epsilon: {epsilon:.3f}")
            if history['loss']:
                recent_loss = history['loss'][-log_interval:]
                print(f"  Loss: {np.mean(recent_loss):.4f}")

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_file = os.path.join(save_dir, f'checkpoint_{episode + 1}.pt')
            trainer.save(checkpoint_file)
            print(f"Checkpoint saved: {checkpoint_file}")

            # Save training curves
            visualizer.plot_training_curves(history, 'training_curves.png')

    # Final save
    final_checkpoint = os.path.join(save_dir, 'final_model.pt')
    trainer.save(final_checkpoint)
    print(f"\nTraining complete! Final model saved to: {final_checkpoint}")

    # Save training history
    visualizer.plot_training_curves(history, 'training_curves.png')

    return trainer, history


def main():
    parser = argparse.ArgumentParser(description='Train QMIX on SAR environment')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--env-config', type=str, default=None,
                       help='Path to environment config (optional)')
    parser.add_argument('--comms-config', type=str, default=None,
                       help='Path to communication config (optional)')
    parser.add_argument('--save-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu/cuda/auto)')

    args = parser.parse_args()

    # Load base config
    config = load_config(args.config)

    # Merge environment config if provided
    if args.env_config:
        env_config = load_config(args.env_config)
        config = merge_configs(config, env_config)

    # Merge communication config if provided
    if args.comms_config:
        comms_config = load_config(args.comms_config)
        config = merge_configs(config, {'comm': comms_config['comm']})

    # Override device if provided
    if args.device:
        config['device'] = args.device

    # Setup device
    device = setup_environment(config)

    # Train
    train(config, device, args.save_dir, args.checkpoint)


if __name__ == '__main__':
    main()
