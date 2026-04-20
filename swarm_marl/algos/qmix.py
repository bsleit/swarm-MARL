"""QMIX trainer implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

from ..models.agent_network import AgentNetworkGroup
from ..models.mixing_network import QMIXMixingNetwork
from .replay_buffer import EpisodeReplayBuffer, Episode
from .epsilon import EpsilonScheduler


class QMIXTrainer:
    """QMIX trainer for multi-agent reinforcement learning."""

    def __init__(self, obs_dim: int, state_dim: int, n_actions: int,
                 n_agents: int, config: dict, device: str = 'cpu'):
        """Initialize QMIX trainer.

        Args:
            obs_dim: Observation dimension per agent
            state_dim: Global state dimension
            n_actions: Number of actions per agent
            n_agents: Number of agents
            config: Training configuration
            device: Device to use ('cpu' or 'cuda')
        """
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.config = config
        self.device = device

        # Extract config
        self.lr = config['training']['learning_rate']
        self.gamma = config['training']['gamma']
        self.target_update_interval = config['training']['target_update_interval']
        self.gradient_clip = config['training']['gradient_clip']
        self.batch_size = config['training']['batch_size']
        self.hidden_dim = config['network']['gru_hidden_dim']
        self.embed_dim = config['network']['mixing_embed_dim']

        # Create networks
        self.agent_network = AgentNetworkGroup(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=self.hidden_dim,
            n_agents=n_agents
        ).to(device)

        self.mixing_network = QMIXMixingNetwork(
            state_dim=state_dim,
            n_agents=n_agents,
            embed_dim=self.embed_dim
        ).to(device)

        # Target networks
        self.target_agent_network = AgentNetworkGroup(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=self.hidden_dim,
            n_agents=n_agents
        ).to(device)

        self.target_mixing_network = QMIXMixingNetwork(
            state_dim=state_dim,
            n_agents=n_agents,
            embed_dim=self.embed_dim
        ).to(device)

        # Copy weights to target networks
        self.target_agent_network.load_state_dict(self.agent_network.state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

        # Freeze target networks
        for param in self.target_agent_network.parameters():
            param.requires_grad = False
        for param in self.target_mixing_network.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.agent_network.parameters()) +
            list(self.mixing_network.parameters()),
            lr=self.lr
        )

        # Replay buffer
        self.replay_buffer = EpisodeReplayBuffer(
            capacity=config['training']['replay_buffer_size'],
            num_agents=n_agents,
            max_steps=config['env']['max_steps'],
            obs_dim=obs_dim,
            state_dim=state_dim,
            n_actions=n_actions,
            seed=config.get('seed', 42)
        )

        # Epsilon scheduler
        self.epsilon_scheduler = EpsilonScheduler(
            start=config['training']['epsilon_start'],
            end=config['training']['epsilon_end'],
            decay_steps=config['training']['epsilon_decay_steps']
        )

        # Training step counter
        self.train_step_count = 0
        self.episode_count = 0

    def select_actions(self, observations: np.ndarray, hidden_states: np.ndarray,
                       epsilon: float, available_actions: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """Select actions using epsilon-greedy.

        Args:
            observations: (n_agents, obs_dim)
            hidden_states: (n_agents, hidden_dim)
            epsilon: Exploration rate
            available_actions: (n_agents, n_actions) binary mask

        Returns:
            Tuple of (actions, new_hidden_states)
        """
        # Convert to tensors
        obs_tensor = torch.FloatTensor(observations).unsqueeze(0).to(self.device)
        hidden_tensor = torch.FloatTensor(hidden_states).unsqueeze(0).to(self.device)

        # Get Q-values
        # Need previous actions - use zeros for first step
        prev_actions = torch.zeros(1, self.n_agents, self.n_actions).to(self.device)

        with torch.no_grad():
            q_values, new_hidden = self.agent_network(obs_tensor, prev_actions, hidden_tensor)

        q_values = q_values.squeeze(0).cpu().numpy()
        new_hidden = new_hidden.squeeze(0).cpu().numpy()

        # Epsilon-greedy
        actions = np.zeros(self.n_agents, dtype=np.int64)

        for i in range(self.n_agents):
            if np.random.random() < epsilon:
                # Random action
                if available_actions is not None:
                    valid_actions = np.where(available_actions[i] > 0)[0]
                    if len(valid_actions) > 0:
                        actions[i] = np.random.choice(valid_actions)
                    else:
                        actions[i] = 0
                else:
                    actions[i] = np.random.randint(0, self.n_actions)
            else:
                # Greedy action
                if available_actions is not None:
                    # Mask unavailable actions
                    masked_q = q_values[i].copy()
                    masked_q[available_actions[i] == 0] = -np.inf
                    actions[i] = np.argmax(masked_q)
                else:
                    actions[i] = np.argmax(q_values[i])

        return actions, new_hidden

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Perform one training step.

        Args:
            batch: Batch of episode data

        Returns:
            Loss value
        """
        self.train_step_count += 1

        # Move batch to device
        states = batch['states'].to(self.device)
        observations = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        terminated = batch['terminated'].to(self.device)
        mask = batch['mask'].to(self.device)

        batch_size = states.shape[0]
        seq_length = states.shape[1]

        # Initialize hidden states
        hidden = self.agent_network.init_hidden(batch_size).to(self.device)
        target_hidden = self.target_agent_network.init_hidden(batch_size).to(self.device)

        # Reshape for batch processing
        hidden = hidden.view(batch_size, self.n_agents, self.hidden_dim)
        target_hidden = target_hidden.view(batch_size, self.n_agents, self.hidden_dim)

        # Lists to collect outputs
        q_values_list = []
        target_q_values_list = []

        # Process sequence
        for t in range(seq_length):
            # Current step
            obs_t = observations[:, t]  # (batch, n_agents, obs_dim)
            actions_t = actions[:, t]   # (batch, n_agents)

            # Previous actions (one-hot) - use zeros for first step, then actual actions
            if t == 0:
                prev_actions = torch.zeros(batch_size, self.n_agents, self.n_actions).to(self.device)
            else:
                prev_actions = F.one_hot(actions[:, t-1], num_classes=self.n_actions).float()

            # Get Q-values from agent network
            q_t, hidden = self.agent_network(obs_t, prev_actions, hidden)
            q_values_list.append(q_t)

            # Get Q-values from target network for next state
            with torch.no_grad():
                target_q_t, target_hidden = self.target_agent_network(obs_t, prev_actions, target_hidden)
                target_q_values_list.append(target_q_t)

        # Stack Q-values: (batch, seq_len, n_agents, n_actions)
        q_values = torch.stack(q_values_list, dim=1)
        target_q_values = torch.stack(target_q_values_list, dim=1)

        # Get Q-values for chosen actions
        # actions shape: (batch, seq_len, n_agents)
        # Expand for gathering
        actions_expanded = actions.unsqueeze(-1)  # (batch, seq_len, n_agents, 1)
        chosen_q = torch.gather(q_values, dim=-1, index=actions_expanded).squeeze(-1)
        # chosen_q shape: (batch, seq_len, n_agents)

        # Get max Q-values from target network for next states
        max_target_q = target_q_values.max(dim=-1)[0]  # (batch, seq_len, n_agents)

        # Mix Q-values
        q_tot_list = []
        target_q_tot_list = []

        for t in range(seq_length):
            # Current Q_tot
            q_tot_t = self.mixing_network(chosen_q[:, t], states[:, t])
            q_tot_list.append(q_tot_t)

            # Target Q_tot
            with torch.no_grad():
                target_q_tot_t = self.target_mixing_network(max_target_q[:, t], states[:, t])
                target_q_tot_list.append(target_q_tot_t)

        q_tot = torch.stack(q_tot_list, dim=1).squeeze(-1)  # (batch, seq_len)
        target_q_tot = torch.stack(target_q_tot_list, dim=1).squeeze(-1)  # (batch, seq_len)

        # Compute TD target
        # R_t + gamma * Q_target(s_{t+1}) * (1 - terminated_t)
        # Shift target q_tot for next states
        target_q_tot_shifted = torch.cat([
            target_q_tot[:, 1:],
            torch.zeros(batch_size, 1).to(self.device)
        ], dim=1)

        td_target = rewards + self.gamma * target_q_tot_shifted * (1 - terminated)

        # Compute loss
        # Mask out padded transitions
        loss = F.mse_loss(q_tot * mask, td_target.detach() * mask)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.agent_network.parameters()) +
            list(self.mixing_network.parameters()),
            self.gradient_clip
        )

        self.optimizer.step()

        # Update target networks
        if self.train_step_count % self.target_update_interval == 0:
            self.update_targets()

        return loss.item()

    def update_targets(self) -> None:
        """Update target network weights."""
        self.target_agent_network.load_state_dict(self.agent_network.state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'agent_network': self.agent_network.state_dict(),
            'mixing_network': self.mixing_network.state_dict(),
            'target_agent_network': self.target_agent_network.state_dict(),
            'target_mixing_network': self.target_mixing_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_step_count': self.train_step_count,
            'episode_count': self.episode_count,
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent_network.load_state_dict(checkpoint['agent_network'])
        self.mixing_network.load_state_dict(checkpoint['mixing_network'])
        self.target_agent_network.load_state_dict(checkpoint['target_agent_network'])
        self.target_mixing_network.load_state_dict(checkpoint['target_mixing_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_step_count = checkpoint['train_step_count']
        self.episode_count = checkpoint['episode_count']

    def get_epsilon(self) -> float:
        """Get current epsilon value."""
        return self.epsilon_scheduler.get_epsilon(self.episode_count)

    def increment_episode(self) -> None:
        """Increment episode counter."""
        self.episode_count += 1
