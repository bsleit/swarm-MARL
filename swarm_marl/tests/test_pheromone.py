"""Unit tests for pheromone field."""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm_marl.envs.pheromone import PheromoneField


class TestPheromoneField:
    """Tests for PheromoneField."""

    def test_initialization(self):
        """Test pheromone field initialization."""
        field = PheromoneField(size=20, max_level=10, decay_rate=1)
        assert field.size == 20
        assert field.max_level == 10
        assert field.decay_rate == 1
        assert field.discovery.shape == (20, 20)
        assert field.return_pheromone.shape == (20, 20)
        assert np.all(field.discovery == 0)
        assert np.all(field.return_pheromone == 0)

    def test_reset(self):
        """Test reset clears all pheromones."""
        field = PheromoneField(size=10, max_level=10)

        # Add some pheromones
        field.deposit((5, 5), 'discovery', 5)
        field.deposit((3, 3), 'return', 3)

        # Reset
        field.reset()

        assert np.all(field.discovery == 0)
        assert np.all(field.return_pheromone == 0)

    def test_discovery_deposit(self):
        """Test discovery pheromone deposit."""
        field = PheromoneField(size=10, max_level=10)

        field.deposit((5, 5), 'discovery', 3)
        assert field.discovery[5, 5] == 3

        # Add more
        field.deposit((5, 5), 'discovery', 4)
        assert field.discovery[5, 5] == 7

        # Test clamping
        field.deposit((5, 5), 'discovery', 10)
        assert field.discovery[5, 5] == 10  # Max level

    def test_return_deposit(self):
        """Test return pheromone deposit."""
        field = PheromoneField(size=10, max_level=10)

        field.deposit((3, 3), 'return', 5)
        assert field.return_pheromone[3, 3] == 5

    def test_out_of_bounds_deposit(self):
        """Test deposit out of bounds is handled gracefully."""
        field = PheromoneField(size=10, max_level=10)

        # Should not raise error
        field.deposit((-1, 5), 'discovery', 3)
        field.deposit((15, 15), 'discovery', 3)

    def test_decay(self):
        """Test pheromone decay."""
        field = PheromoneField(size=10, max_level=10, decay_rate=1)

        field.deposit((5, 5), 'discovery', 5)
        field.deposit((3, 3), 'return', 8)

        field.decay()

        assert field.discovery[5, 5] == 4
        assert field.return_pheromone[3, 3] == 7

        # Decay more
        for _ in range(5):
            field.decay()

        assert field.discovery[5, 5] == 0  # Clamped at 0
        assert field.return_pheromone[3, 3] == 2

    def test_get_level(self):
        """Test get_level method."""
        field = PheromoneField(size=10, max_level=10)

        field.deposit((5, 5), 'discovery', 7)
        assert field.get_level((5, 5), 'discovery') == 7
        assert field.get_level((5, 5), 'return') == 0

        # Out of bounds
        assert field.get_level((-1, 5), 'discovery') == 0
        assert field.get_level((15, 15), 'discovery') == 0

    def test_local_field(self):
        """Test get_local_field method."""
        field = PheromoneField(size=10, max_level=10)

        # Deposit at center
        field.deposit((5, 5), 'discovery', 5)
        field.deposit((5, 5), 'return', 3)

        discovery, return_pheromone = field.get_local_field((5, 5), radius=1)

        assert discovery.shape == (3, 3)
        assert return_pheromone.shape == (3, 3)
        assert discovery[1, 1] == 5  # Center
        assert return_pheromone[1, 1] == 3

    def test_local_field_edge(self):
        """Test local field at grid edge."""
        field = PheromoneField(size=10, max_level=10)

        discovery, return_pheromone = field.get_local_field((0, 0), radius=1)

        assert discovery.shape == (3, 3)
        assert return_pheromone.shape == (3, 3)
        # Edge should have zeros outside grid
        assert discovery[0, 0] == 0  # (-1, -1) outside grid

    def test_gradient(self):
        """Test get_gradient method."""
        field = PheromoneField(size=10, max_level=10)

        # Deposit at different locations
        field.deposit((6, 6), 'discovery', 10)
        field.deposit((4, 4), 'discovery', 5)

        dx, dy = field.get_gradient((5, 5), 'discovery')

        # Should point toward higher pheromone (6, 6)
        assert dx >= 0
        assert dy >= 0

    def test_find_min_pheromone(self):
        """Test find_min_pheromone_direction method."""
        field = PheromoneField(size=10, max_level=10)

        # Create a gradient: higher pheromone around center
        for x in range(3, 8):
            for y in range(3, 8):
                field.deposit((x, y), 'discovery', 5)

        # Find direction from outside
        dx, dy = field.find_min_pheromone_direction((5, 5), None)

        # Should move away from center (toward lower pheromone)
        assert isinstance(dx, int)
        assert isinstance(dy, int)
        assert -1 <= dx <= 1
        assert -1 <= dy <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
