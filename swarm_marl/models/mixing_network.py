"""QMIX mixing network with hypernetworks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QMIXMixingNetwork(nn.Module):
    """QMIX mixing network that produces Q_tot from individual Q_i values."""

    def __init__(self, state_dim: int, n_agents: int, embed_dim: int = 32):
        """Initialize mixing network.

        Args:
            state_dim: Global state dimension
            n_agents: Number of agents
            embed_dim: Embedding dimension for hypernetworks
        """
        super().__init__()

        self.state_dim = state_dim
        self.n_agents = n_agents
        self.embed_dim = embed_dim

        # Hypernetwork for first layer weights (n_agents x embed_dim)
        # Produces weights for mixing network layer 1
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)

        # Hypernetwork for first layer bias (embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        # Hypernetwork for second layer weights (embed_dim x 1)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)

        # Hypernetwork for second layer bias (1)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to produce Q_tot.

        Args:
            q_values: Individual Q-values (batch, n_agents)
            state: Global state (batch, state_dim)

        Returns:
            Q_tot (batch, 1)
        """
        batch_size = q_values.shape[0]

        # Generate weights and biases from hypernetworks
        # w1: (batch, n_agents * embed_dim) -> (batch, n_agents, embed_dim)
        w1 = torch.abs(self.hyper_w1(state))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)

        # b1: (batch, embed_dim)
        b1 = self.hyper_b1(state)
        b1 = b1.view(-1, 1, self.embed_dim)

        # w2: (batch, embed_dim) -> (batch, embed_dim, 1)
        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(-1, self.embed_dim, 1)

        # b2: (batch, 1) -> (batch, 1, 1)
        b2 = self.hyper_b2(state)
        b2 = b2.view(-1, 1, 1)

        # First layer: Q_i (batch, n_agents) @ w1 (batch, n_agents, embed_dim)
        # -> (batch, 1, embed_dim)
        hidden = F.elu(torch.bmm(q_values.unsqueeze(1), w1) + b1)

        # Second layer: hidden (batch, 1, embed_dim) @ w2 (batch, embed_dim, 1)
        # -> (batch, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2

        # Squeeze to (batch, 1)
        q_tot = q_tot.squeeze(1)

        return q_tot

    def get_monotonicity_check(self, state: torch.Tensor) -> torch.Tensor:
        """Get weights for monotonicity check (should be non-negative).

        Returns:
            Concatenated weights from both layers
        """
        # This is useful for verifying monotonicity constraint
        w1 = torch.abs(self.hyper_w1(state))
        w2 = torch.abs(self.hyper_w2(state))
        return torch.cat([w1.flatten(), w2.flatten()])
