import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
"""Unit tests for baseline policies."""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm_marl.baselines.random_walk import RandomWalkPolicy
from swarm_marl.baselines.static_pheromone import StaticPheromonePolicy


class TestRandomWalkPolicy:
    """Tests for RandomWalkPolicy."""

    def test_initialization(self):
        """Test policy initialization."""
        policy = RandomWalkPolicy(num_agents=5, action_space_size=30, seed=42)
        assert policy.num_agents == 5
        assert policy.action_space_size == 30

    def test_get_actions(self):
        """Test getting actions."""
        policy = RandomWalkPolicy(num_agents=5, action_space_size=30, seed=42)

        observations = {f'agent_{i}': np.random.randn(50) for i in range(5)}
        infos = {f'agent_{i}': {} for i in range(5)}

        actions = policy.get_actions(observations, infos)

        assert len(actions) == 5
        for i in range(5):
            assert f'agent_{i}' in actions
            assert 0 <= actions[f'agent_{i}'] < 30

    def test_deterministic_with_same_seed(self):
        """Test that same seed produces same actions."""
        policy1 = RandomWalkPolicy(num_agents=5, action_space_size=30, seed=42)
        policy2 = RandomWalkPolicy(num_agents=5, action_space_size=30, seed=42)

        observations = {f'agent_{i}': np.random.randn(50) for i in range(5)}
        infos = {f'agent_{i}': {} for i in range(5)}

        actions1 = policy1.get_actions(observations, infos)
        actions2 = policy2.get_actions(observations, infos)

        for i in range(5):
            assert actions1[f'agent_{i}'] == actions2[f'agent_{i}']


class TestStaticPheromonePolicy:
    """Tests for StaticPheromonePolicy."""

    def test_initialization(self):
        """Test policy initialization."""
        policy = StaticPheromonePolicy(
            num_agents=5,
            grid_size=20,
            tau=5,
            always_communicate=True,
            seed=42
        )
        assert policy.num_agents == 5
        assert policy.grid_size == 20
        assert policy.tau == 5
        assert policy.always_communicate == True

    def test_get_actions(self):
        """Test getting actions."""
        policy = StaticPheromonePolicy(
            num_agents=3,
            grid_size=20,
            tau=5,
            always_communicate=True,
            seed=42
        )

        # Create observation with valid shape
        # obs shape: flattened (5, 3, 3) + 5 scalars = 45 + 5 = 50
        obs = np.zeros(50, dtype=np.float32)
        # Set cell types to non-obstacle (values < 1.0)
        obs[:9] = 0.33  # Cell types
        obs[9:18] = 0.5  # Discovery pheromone
        obs[18:27] = 0.5  # Return pheromone

        observations = {f'agent_{i}': obs for i in range(3)}
        infos = {f'agent_{i}': {'role': 0} for i in range(3)}

        actions = policy.get_actions(observations, infos)

        assert len(actions) == 3
        for i in range(3):
            assert f'agent_{i}' in actions
            assert 0 <= actions[f'agent_{i}'] < 30

    def test_explorer_moves_toward_min_pheromone(self):
        """Test that explorer moves toward minimum pheromone."""
        policy = StaticPheromonePolicy(
            num_agents=1,
            grid_size=20,
            tau=5,
            seed=42
        )

        # Create observation with low pheromone at north
        obs = np.zeros(50, dtype=np.float32)
        obs[:9] = 0.33  # Cell types (non-obstacle)

        # Discovery pheromone: center=0.5, north=0.0 (min), south=1.0
        discovery = np.array([
            0.0, 0.3, 0.3,  # North row: low at center
            0.3, 0.5, 0.3,  # Center row
            1.0, 0.3, 0.3   # South row: high at center
        ])
        obs[9:18] = discovery

        observations = {'agent_0': obs}
        infos = {'agent_0': {'role': 0}}  # EXPLORER

        actions = policy.get_actions(observations, infos)
        action = actions['agent_0']

        # Decode action to get locomotion
        locomotion = action % 5

        # Should move north (0) toward minimum pheromone
        assert locomotion == 0

    def test_reporter_moves_toward_max_pheromone(self):
        """Test that reporter moves toward maximum return pheromone."""
        policy = StaticPheromonePolicy(
            num_agents=1,
            grid_size=20,
            tau=5,
            seed=42
        )

        # Create observation with high return pheromone at south
        obs = np.zeros(50, dtype=np.float32)
        obs[:9] = 0.33  # Cell types (non-obstacle)

        # Return pheromone: south=1.0 (max), others low
        return_pheromone = np.array([
            0.0, 0.0, 0.0,  # North row
            0.0, 0.1, 0.0,  # Center row
            0.5, 1.0, 0.5   # South row: high at center
        ])
        obs[18:27] = return_pheromone

        observations = {'agent_0': obs}
        infos = {'agent_0': {'role': 1}}  # REPORTER

        actions = policy.get_actions(observations, infos)
        action = actions['agent_0']

        # Decode action to get locomotion
        locomotion = action % 5

        # Should move south (1) toward maximum pheromone
        assert locomotion == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
