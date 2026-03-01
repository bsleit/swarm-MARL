"""Agent Q-network for QMIX."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentQNetwork(nn.Module):
    """Per-agent Q-value network with GRU for partial observability."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64,
                 n_agents: int = 5):
        """Initialize agent network.

        Args:
            obs_dim: Observation dimension
            n_actions: Number of actions
            hidden_dim: Hidden dimension for GRU and FC layers
            n_agents: Number of agents (for agent ID embedding)
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents

        # Input includes: observation + previous action (one-hot) + agent ID (one-hot)
        input_dim = obs_dim + n_actions + n_agents

        # Fully connected layer before GRU
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # GRU layer for temporal information
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Output layer
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize hidden state.

        Args:
            batch_size: Batch size

        Returns:
            Initial hidden state tensor
        """
        return torch.zeros(batch_size, self.hidden_dim)

    def forward(self, obs: torch.Tensor, prev_action: torch.Tensor,
                agent_id: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: Observation tensor (batch, obs_dim)
            prev_action: Previous action one-hot (batch, n_actions)
            agent_id: Agent ID one-hot (batch, n_agents)
            hidden: Previous hidden state (batch, hidden_dim)

        Returns:
            Tuple of (Q-values, new_hidden_state)
        """
        # Concatenate inputs
        x = torch.cat([obs, prev_action, agent_id], dim=-1)

        # FC layer
        x = F.relu(self.fc1(x))

        # GRU
        hidden = self.gru(x, hidden)

        # Output Q-values
        q_values = self.fc2(hidden)

        return q_values, hidden


class AgentNetworkGroup(nn.Module):
    """Group of agent networks with shared weights."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64,
                 n_agents: int = 5):
        """Initialize group of agent networks.

        Args:
            obs_dim: Observation dimension per agent
            n_actions: Number of actions per agent
            hidden_dim: Hidden dimension
            n_agents: Number of agents
        """
        super().__init__()

        self.n_agents = n_agents
        self.n_actions = n_actions

        # Single shared network for all agents
        self.agent_network = AgentQNetwork(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            n_agents=n_agents
        )

        # Agent ID one-hot encodings
        self.register_buffer(
            'agent_ids',
            torch.eye(n_agents)
        )

    def forward(self, observations: torch.Tensor,
                prev_actions: torch.Tensor,
                hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for all agents.

        Args:
            observations: (batch, n_agents, obs_dim)
            prev_actions: (batch, n_agents, n_actions) one-hot
            hidden_states: (batch, n_agents, hidden_dim)

        Returns:
            Tuple of (Q-values (batch, n_agents, n_actions), new_hidden_states)
        """
        batch_size = observations.shape[0]

        # Reshape to process all agents
        # (batch * n_agents, ...)
        obs_flat = observations.view(-1, observations.shape[-1])
        prev_actions_flat = prev_actions.view(-1, self.n_actions)
        hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])

        # Repeat agent IDs for batch
        agent_ids_batch = self.agent_ids.unsqueeze(0).expand(batch_size, -1, -1)
        agent_ids_flat = agent_ids_batch.reshape(-1, self.n_agents)

        # Forward pass
        q_values_flat, new_hidden_flat = self.agent_network(
            obs_flat, prev_actions_flat, agent_ids_flat, hidden_flat
        )

        # Reshape back
        q_values = q_values_flat.view(batch_size, self.n_agents, self.n_actions)
        new_hidden = new_hidden_flat.view(batch_size, self.n_agents, -1)

        return q_values, new_hidden

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize hidden states for all agents."""
        return self.agent_network.init_hidden(batch_size * self.n_agents)
