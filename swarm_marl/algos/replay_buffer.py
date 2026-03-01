"""Replay buffer for QMIX."""

import numpy as np
from typing import Dict, List, Tuple
from collections import deque


class Episode:
    """Stores data for a single episode."""

    def __init__(self, num_agents: int, max_steps: int, obs_dim: int,
                 state_dim: int, n_actions: int):
        """Initialize episode buffer.

        Args:
            num_agents: Number of agents
            max_steps: Maximum episode length
            obs_dim: Observation dimension
            state_dim: Global state dimension
            n_actions: Number of actions
        """
        self.num_agents = num_agents
        self.max_steps = max_steps

        # Pre-allocate arrays
        self.states = np.zeros((max_steps, state_dim), dtype=np.float32)
        self.observations = np.zeros((max_steps, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_steps, num_agents), dtype=np.int64)
        self.rewards = np.zeros((max_steps,), dtype=np.float32)
        self.avail_actions = np.ones((max_steps, num_agents, n_actions), dtype=np.float32)
        self.terminated = np.zeros((max_steps,), dtype=np.float32)
        self.mask = np.zeros((max_steps,), dtype=np.float32)

        self.step = 0
        self.filled = False

    def add(self, state: np.ndarray, observations: np.ndarray,
            actions: np.ndarray, reward: float,
            terminated: bool, mask: float = 1.0) -> None:
        """Add a transition to the episode."""
        if self.step >= self.max_steps:
            self.filled = True
            return

        self.states[self.step] = state
        self.observations[self.step] = observations
        self.actions[self.step] = actions
        self.rewards[self.step] = reward
        self.terminated[self.step] = float(terminated)
        self.mask[self.step] = mask

        self.step += 1

    def get_data(self) -> Dict[str, np.ndarray]:
        """Get episode data."""
        length = self.step
        return {
            'states': self.states[:length],
            'observations': self.observations[:length],
            'actions': self.actions[:length],
            'rewards': self.rewards[:length],
            'avail_actions': self.avail_actions[:length],
            'terminated': self.terminated[:length],
            'mask': self.mask[:length],
            'max_seq_length': length,
        }


class EpisodeReplayBuffer:
    """Replay buffer that stores full episodes for QMIX."""

    def __init__(self, capacity: int, num_agents: int, max_steps: int,
                 obs_dim: int, state_dim: int, n_actions: int, seed: int = 42):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of episodes to store
            num_agents: Number of agents
            max_steps: Maximum episode length
            obs_dim: Observation dimension
            state_dim: Global state dimension
            n_actions: Number of actions
            seed: Random seed
        """
        self.capacity = capacity
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.buffer = deque(maxlen=capacity)
        self.rng = np.random.RandomState(seed)

    def add(self, episode: Episode) -> None:
        """Add an episode to the buffer."""
        self.buffer.append(episode)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of episodes.

        Returns:
            Dictionary containing batched episode data
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # Sample episode indices
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)

        # Get episodes
        episodes = [self.buffer[i] for i in indices]

        # Find max length for padding
        max_length = max(ep.step for ep in episodes)

        # Batch data
        batch_states = []
        batch_observations = []
        batch_actions = []
        batch_rewards = []
        batch_terminated = []
        batch_mask = []

        for ep in episodes:
            data = ep.get_data()
            length = data['max_seq_length']

            # Pad sequences to max_length
            pad_length = max_length - length

            if pad_length > 0:
                states = np.pad(data['states'], ((0, pad_length), (0, 0)), mode='constant')
                obs = np.pad(data['observations'], ((0, pad_length), (0, 0), (0, 0)), mode='constant')
                actions = np.pad(data['actions'], ((0, pad_length), (0, 0)), mode='constant')
                rewards = np.pad(data['rewards'], (0, pad_length), mode='constant')
                terminated = np.pad(data['terminated'], (0, pad_length), mode='constant')
                mask = np.concatenate([data['mask'], np.zeros(pad_length)])
            else:
                states = data['states']
                obs = data['observations']
                actions = data['actions']
                rewards = data['rewards']
                terminated = data['terminated']
                mask = data['mask']

            batch_states.append(states)
            batch_observations.append(obs)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_terminated.append(terminated)
            batch_mask.append(mask)

        return {
            'states': np.array(batch_states),
            'observations': np.array(batch_observations),
            'actions': np.array(batch_actions),
            'rewards': np.array(batch_rewards),
            'terminated': np.array(batch_terminated),
            'mask': np.array(batch_mask),
            'max_seq_length': max_length,
            'batch_size': batch_size,
        }

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer has enough episodes to sample."""
        return len(self.buffer) >= batch_size
