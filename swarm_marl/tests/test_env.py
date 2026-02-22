"""Unit tests for environment components."""

import pytest
import numpy as np
import yaml
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm_marl.envs.grid_world import GridWorld, Direction, CellType
from swarm_marl.envs.pheromone import PheromoneField
from swarm_marl.envs.agent import Agent, AgentRole
from swarm_marl.envs.comm_model import (
    FixedDenialModel, ProbabilisticModel, DistanceZoneModel
)
from swarm_marl.envs.sar_env import SAREnv


class TestGridWorld:
    """Tests for GridWorld."""

    def test_initialization(self):
        """Test grid world initialization."""
        grid = GridWorld(size=20, obstacle_density=0.15, seed=42)
        assert grid.size == 20
        assert grid.grid.shape == (20, 20)

    def test_valid_move(self):
        """Test move validation."""
        grid = GridWorld(size=10, obstacle_density=0.0, seed=42)
        grid.reset()

        # Mark some cells as obstacles
        grid.grid[5, 5] = CellType.OBSTACLE

        # Test valid move
        assert grid.is_valid_move((5, 4), Direction.SOUTH) == False  # Blocked by obstacle
        assert grid.is_valid_move((0, 0), Direction.NORTH) == False  # Out of bounds
        assert grid.is_valid_move((5, 4), Direction.NORTH) == True   # Valid

    def test_coverage(self):
        """Test coverage tracking."""
        grid = GridWorld(size=10, obstacle_density=0.0, seed=42)
        grid.reset()

        initial_coverage = grid.get_coverage()
        assert initial_coverage == 0.0

        # Mark some cells as explored
        grid.mark_explored((0, 0))
        grid.mark_explored((1, 1))

        coverage = grid.get_coverage()
        assert coverage > 0.0


class TestPheromoneField:
    """Tests for PheromoneField."""

    def test_deposit(self):
        """Test pheromone deposit."""
        field = PheromoneField(size=10, max_level=10)

        field.deposit((5, 5), 'discovery', 3)
        assert field.discovery[5, 5] == 3

        # Test max level clamping
        field.deposit((5, 5), 'discovery', 10)
        assert field.discovery[5, 5] == 10

    def test_decay(self):
        """Test pheromone decay."""
        field = PheromoneField(size=10, max_level=10, decay_rate=1)

        field.deposit((5, 5), 'discovery', 5)
        assert field.discovery[5, 5] == 5

        field.decay()
        assert field.discovery[5, 5] == 4

        field.decay()
        assert field.discovery[5, 5] == 3

        # Test clamping at 0
        for _ in range(10):
            field.decay()
        assert field.discovery[5, 5] == 0


class TestAgent:
    """Tests for Agent."""

    def test_role_switch(self):
        """Test role switching logic."""
        agent = Agent(agent_id=0, position=(0, 0), tau=5)
        assert agent.role == AgentRole.EXPLORER

        # Add stimulus below threshold
        agent.add_stimulus(3.0)
        switched = agent.check_role_switch()
        assert switched == False
        assert agent.role == AgentRole.EXPLORER

        # Add more stimulus to exceed threshold
        agent.add_stimulus(3.0)
        switched = agent.check_role_switch()
        assert switched == True
        assert agent.role == AgentRole.REPORTER

        # Stimulus should be reset
        assert agent.stimulus == 0.0

    def test_tau_adjustment(self):
        """Test tau adjustment."""
        agent = Agent(agent_id=0, position=(0, 0), tau=5)
        assert agent.tau == 5

        agent.adjust_tau(-1, tau_min=1, tau_max=10)
        assert agent.tau == 4

        agent.adjust_tau(1, tau_min=1, tau_max=10)
        assert agent.tau == 5

        # Test clamping
        agent.adjust_tau(-10, tau_min=1, tau_max=10)
        assert agent.tau == 1

        agent.adjust_tau(20, tau_min=1, tau_max=10)
        assert agent.tau == 10


class TestCommunicationModels:
    """Tests for communication models."""

    def test_fixed_denial(self):
        """Test fixed denial model."""
        denial_zones = [[5, 5, 10, 10]]
        model = FixedDenialModel(denial_zones=denial_zones)

        # Inside denial zone
        assert model.can_communicate((6, 6), (7, 7)) == False

        # Outside denial zone
        assert model.can_communicate((1, 1), (2, 2)) == True

        # One in, one out
        assert model.can_communicate((6, 6), (1, 1)) == False

    def test_probabilistic(self):
        """Test probabilistic model."""
        # Create with high probability everywhere
        prob_map = np.ones((10, 10))
        model = ProbabilisticModel(grid_size=10, prob_map=prob_map, seed=42)

        # Should always communicate with probability 1.0
        assert model.can_communicate((5, 5), (6, 6)) == True

    def test_distance_zones(self):
        """Test distance + zones model."""
        denial_zones = [[5, 5, 10, 10]]
        model = DistanceZoneModel(comm_range=5.0, denial_zones=denial_zones)

        # Within range, outside zone
        assert model.can_communicate((0, 0), (2, 2)) == True

        # Within range, inside zone
        assert model.can_communicate((6, 6), (7, 7)) == False

        # Outside range
        assert model.can_communicate((0, 0), (10, 10)) == False


class TestSAREnv:
    """Tests for SAR environment."""

    def test_environment_creation(self):
        """Test environment creation."""
        config = {
            'env': {
                'grid_size': 20,
                'num_agents': 5,
                'max_steps': 200,
                'obstacle_density': 0.15,
                'coverage_threshold': 0.9,
            },
            'pheromone': {
                'max_level': 10,
                'decay_rate': 1,
                'discovery_deposit': 3,
                'return_deposit': 3,
            },
            'agent': {
                'tau_min': 1,
                'tau_max': 10,
                'tau_default': 5,
                'tau_delta': 1,
                'observation_radius': 1,
            },
            'comm': {
                'model': 'fixed_denial',
                'denial_zones': [],
            },
            'reward': {
                'alpha': 1.0,
                'beta': 0.01,
                'gamma': 0.1,
            },
            'training': {
                'num_episodes': 1000,
                'batch_size': 32,
                'replay_buffer_size': 5000,
                'learning_rate': 0.0005,
                'gamma': 0.99,
                'target_update_interval': 200,
                'gradient_clip': 10.0,
                'epsilon_start': 1.0,
                'epsilon_end': 0.05,
                'epsilon_decay_steps': 50000,
            },
            'network': {
                'agent_hidden_dim': 64,
                'gru_hidden_dim': 64,
                'mixing_embed_dim': 32,
            },
            'seed': 42,
        }

        env = SAREnv(config)
        assert env.grid_size == 20
        assert len(env.possible_agents) == 5

    def test_reset(self):
        """Test environment reset."""
        config = self._get_test_config()
        env = SAREnv(config)

        observations, infos = env.reset(seed=42)

        # Check observations
        assert len(observations) == env.num_agents
        for agent_id in env.agents:
            assert agent_id in observations
            assert observations[agent_id].shape[0] > 0

    def test_step(self):
        """Test environment step."""
        config = self._get_test_config()
        env = SAREnv(config)

        observations, infos = env.reset(seed=42)

        # Take random actions
        actions = {agent_id: env.action_spaces[agent_id].sample()
                    for agent_id in env.agents}

        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # Check outputs
        assert len(next_observations) == env.num_agents
        assert len(rewards) == env.num_agents
        assert len(terminations) == env.num_agents
        assert len(truncations) == env.num_agents

    def test_action_decoding(self):
        """Test action decoding."""
        config = self._get_test_config()
        env = SAREnv(config)

        # Test action 0: should be NORTH, HOLD, SILENT
        locomotion, tau_adjust, communicate = env._decode_action(0)
        assert locomotion == 0  # NORTH
        assert tau_adjust == 0  # Decrease
        assert communicate == 0  # Silent

        # Test action encoding verification
        # locomotion + 5 * (tau_adjust + 3 * communicate)
        # 0 + 5 * (0 + 3 * 0) = 0

        # Test action 14: locomotion=4 (STAY), tau_adjust=2 (increase), communicate=0 (silent)
        # 4 + 5 * (2 + 3 * 0) = 4 + 10 = 14
        locomotion, tau_adjust, communicate = env._decode_action(14)
        assert locomotion == 4  # STAY
        assert tau_adjust == 2  # Increase
        assert communicate == 0   # Silent

        # Test action 29: max action (4 + 5*(2+3*1) = 4 + 5*5 = 29)
        locomotion, tau_adjust, communicate = env._decode_action(29)
        assert locomotion == 4  # STAY
        assert tau_adjust == 2  # Increase
        assert communicate == 1  # Transmit

    def _get_test_config(self):
        """Get test configuration."""
        return {
            'env': {
                'grid_size': 10,
                'num_agents': 3,
                'max_steps': 50,
                'obstacle_density': 0.1,
                'coverage_threshold': 0.9,
            },
            'pheromone': {
                'max_level': 10,
                'decay_rate': 1,
                'discovery_deposit': 3,
                'return_deposit': 3,
            },
            'agent': {
                'tau_min': 1,
                'tau_max': 10,
                'tau_default': 5,
                'tau_delta': 1,
                'observation_radius': 1,
            },
            'comm': {
                'model': 'fixed_denial',
                'denial_zones': [],
            },
            'reward': {
                'alpha': 1.0,
                'beta': 0.01,
                'gamma': 0.1,
            },
            'training': {
                'num_episodes': 1000,
                'batch_size': 32,
                'replay_buffer_size': 5000,
                'learning_rate': 0.0005,
                'gamma': 0.99,
                'target_update_interval': 200,
                'gradient_clip': 10.0,
                'epsilon_start': 1.0,
                'epsilon_end': 0.05,
                'epsilon_decay_steps': 50000,
            },
            'network': {
                'agent_hidden_dim': 64,
                'gru_hidden_dim': 64,
                'mixing_embed_dim': 32,
            },
            'seed': 42,
        }


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
