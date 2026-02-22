"""Unit tests for QMIX components."""

import pytest
import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm_marl.models.agent_network import AgentQNetwork, AgentNetworkGroup
from swarm_marl.models.mixing_network import QMIXMixingNetwork
from swarm_marl.algos.replay_buffer import Episode, EpisodeReplayBuffer
from swarm_marl.algos.epsilon import EpsilonScheduler


class TestAgentNetwork:
    """Tests for AgentQNetwork."""

    def test_initialization(self):
        """Test agent network initialization."""
        net = AgentQNetwork(obs_dim=50, n_actions=30, hidden_dim=64, n_agents=5)
        assert net.obs_dim == 50
        assert net.n_actions == 30
        assert net.hidden_dim == 64

    def test_forward(self):
        """Test forward pass."""
        net = AgentQNetwork(obs_dim=50, n_actions=30, hidden_dim=64, n_agents=5)

        batch_size = 4
        obs = torch.randn(batch_size, 50)
        prev_action = torch.zeros(batch_size, 30)
        prev_action[:, 0] = 1  # One-hot for action 0
        agent_id = torch.zeros(batch_size, 5)
        agent_id[:, 0] = 1  # One-hot for agent 0
        hidden = torch.zeros(batch_size, 64)

        q_values, new_hidden = net(obs, prev_action, agent_id, hidden)

        assert q_values.shape == (batch_size, 30)
        assert new_hidden.shape == (batch_size, 64)

    def test_init_hidden(self):
        """Test hidden state initialization."""
        net = AgentQNetwork(obs_dim=50, n_actions=30, hidden_dim=64)

        hidden = net.init_hidden(batch_size=4)
        assert hidden.shape == (4, 64)
        assert torch.all(hidden == 0)


class TestAgentNetworkGroup:
    """Tests for AgentNetworkGroup."""

    def test_initialization(self):
        """Test agent network group initialization."""
        group = AgentNetworkGroup(obs_dim=50, n_actions=30, hidden_dim=64, n_agents=5)
        assert group.n_agents == 5
        assert group.n_actions == 30

    def test_forward(self):
        """Test forward pass for all agents."""
        group = AgentNetworkGroup(obs_dim=50, n_actions=30, hidden_dim=64, n_agents=5)

        batch_size = 4
        observations = torch.randn(batch_size, 5, 50)
        prev_actions = torch.zeros(batch_size, 5, 30)
        hidden_states = torch.zeros(batch_size, 5, 64)

        q_values, new_hidden = group(observations, prev_actions, hidden_states)

        assert q_values.shape == (batch_size, 5, 30)
        assert new_hidden.shape == (batch_size, 5, 64)


class TestMixingNetwork:
    """Tests for QMIXMixingNetwork."""

    def test_initialization(self):
        """Test mixing network initialization."""
        mixer = QMIXMixingNetwork(state_dim=100, n_agents=5, embed_dim=32)
        assert mixer.state_dim == 100
        assert mixer.n_agents == 5
        assert mixer.embed_dim == 32

    def test_forward(self):
        """Test forward pass produces Q_tot."""
        mixer = QMIXMixingNetwork(state_dim=100, n_agents=5, embed_dim=32)

        batch_size = 4
        q_values = torch.randn(batch_size, 5)
        state = torch.randn(batch_size, 100)

        q_tot = mixer(q_values, state)

        assert q_tot.shape == (batch_size, 1)

    def test_monotonicity(self):
        """Test that monotonicity is enforced (weights are non-negative)."""
        mixer = QMIXMixingNetwork(state_dim=100, n_agents=5, embed_dim=32)

        state = torch.randn(1, 100)
        weights = mixer.get_monotonicity_check(state)

        # All weights should be non-negative (abs is applied in network)
        assert torch.all(weights >= 0)


class TestReplayBuffer:
    """Tests for EpisodeReplayBuffer."""

    def test_initialization(self):
        """Test replay buffer initialization."""
        buffer = EpisodeReplayBuffer(
            capacity=100,
            num_agents=5,
            max_steps=50,
            obs_dim=50,
            state_dim=100,
            n_actions=30
        )
        assert buffer.capacity == 100

    def test_add_and_sample(self):
        """Test adding and sampling episodes."""
        buffer = EpisodeReplayBuffer(
            capacity=10,
            num_agents=5,
            max_steps=50,
            obs_dim=50,
            state_dim=100,
            n_actions=30
        )

        # Add episodes
        for _ in range(5):
            episode = Episode(
                num_agents=5,
                max_steps=20,
                obs_dim=50,
                state_dim=100,
                n_actions=30
            )

            for t in range(10):
                episode.add(
                    state=np.random.randn(100).astype(np.float32),
                    observations=np.random.randn(5, 50).astype(np.float32),
                    actions=np.random.randint(0, 30, size=5),
                    reward=0.0,
                    terminated=False
                )

            buffer.add(episode)

        assert len(buffer) == 5

        # Sample
        batch = buffer.sample(batch_size=3)
        assert batch['batch_size'] == 3

    def test_can_sample(self):
        """Test can_sample method."""
        buffer = EpisodeReplayBuffer(
            capacity=10,
            num_agents=5,
            max_steps=50,
            obs_dim=50,
            state_dim=100,
            n_actions=30
        )

        assert buffer.can_sample(5) == False

        # Add episodes
        for _ in range(5):
            episode = Episode(5, 20, 50, 100, 30)
            episode.add(
                state=np.zeros(100),
                observations=np.zeros((5, 50)),
                actions=np.zeros(5),
                reward=0.0,
                terminated=True
            )
            buffer.add(episode)

        assert buffer.can_sample(5) == True


class TestEpsilonScheduler:
    """Tests for EpsilonScheduler."""

    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = EpsilonScheduler(start=1.0, end=0.05, decay_steps=1000)
        assert scheduler.start == 1.0
        assert scheduler.end == 0.05
        assert scheduler.decay_steps == 1000

    def test_epsilon_values(self):
        """Test epsilon values over time."""
        scheduler = EpsilonScheduler(start=1.0, end=0.05, decay_steps=1000)

        # Start
        assert scheduler.get_epsilon(0) == 1.0

        # End
        assert scheduler.get_epsilon(1000) == 0.05

        # Beyond end
        assert scheduler.get_epsilon(2000) == 0.05

        # Middle
        eps_500 = scheduler.get_epsilon(500)
        assert 0.05 < eps_500 < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
