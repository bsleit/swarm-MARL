"""Pheromone field mechanics for bio-inspired task allocation."""

import numpy as np
from typing import Tuple


class PheromoneField:
    """Manages pheromone levels on the grid."""

    def __init__(self, size: int, max_level: int = 10, decay_rate: int = 1):
        """Initialize pheromone fields.

        Args:
            size: Grid size (size x size)
            max_level: Maximum pheromone level (0-max_level)
            decay_rate: Amount to decay per step
        """
        self.size = size
        self.max_level = max_level
        self.decay_rate = decay_rate

        # Two separate pheromone grids
        self.discovery = np.zeros((size, size), dtype=np.int32)
        self.return_pheromone = np.zeros((size, size), dtype=np.int32)

    def reset(self) -> None:
        """Reset all pheromone levels to zero."""
        self.discovery.fill(0)
        self.return_pheromone.fill(0)

    def deposit(self, pos: Tuple[int, int], pheromone_type: str, amount: int) -> None:
        """Deposit pheromone at position.

        Args:
            pos: (x, y) position
            pheromone_type: 'discovery' or 'return'
            amount: Amount to deposit (will be clamped at max_level)
        """
        x, y = pos
        if not (0 <= x < self.size and 0 <= y < self.size):
            return

        if pheromone_type == 'discovery':
            self.discovery[x, y] = min(self.discovery[x, y] + amount, self.max_level)
        elif pheromone_type == 'return':
            self.return_pheromone[x, y] = min(self.return_pheromone[x, y] + amount, self.max_level)

    def decay(self) -> None:
        """Apply decay to all pheromone levels."""
        self.discovery = np.maximum(self.discovery - self.decay_rate, 0)
        self.return_pheromone = np.maximum(self.return_pheromone - self.decay_rate, 0)

    def get_level(self, pos: Tuple[int, int], pheromone_type: str) -> int:
        """Get pheromone level at position."""
        x, y = pos
        if not (0 <= x < self.size and 0 <= y < self.size):
            return 0

        if pheromone_type == 'discovery':
            return self.discovery[x, y]
        elif pheromone_type == 'return':
            return self.return_pheromone[x, y]
        return 0

    def get_local_field(self, pos: Tuple[int, int], radius: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Get local pheromone field around position.

        Returns:
            Tuple of (discovery_field, return_field), each (2*radius+1, 2*radius+1)
        """
        window_size = 2 * radius + 1
        discovery_window = np.zeros((window_size, window_size), dtype=np.int32)
        return_window = np.zeros((window_size, window_size), dtype=np.int32)

        cx, cy = pos
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = cx + dx, cy + dy
                wx, wy = dx + radius, dy + radius

                if 0 <= x < self.size and 0 <= y < self.size:
                    discovery_window[wx, wy] = self.discovery[x, y]
                    return_window[wx, wy] = self.return_pheromone[x, y]

        return discovery_window, return_window

    def get_gradient(self, pos: Tuple[int, int], pheromone_type: str) -> Tuple[int, int]:
        """Get gradient direction (dx, dy) toward higher pheromone.

        Returns direction that leads to highest pheromone level in 3x3 neighborhood.
        """
        x, y = pos
        max_val = -1
        best_dx, best_dy = 0, 0

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    val = self.get_level((nx, ny), pheromone_type)
                    if val > max_val:
                        max_val = val
                        best_dx, best_dy = dx, dy

        return best_dx, best_dy

    def find_min_pheromone_direction(self, pos: Tuple[int, int],
                                      grid_valid_check=None) -> Tuple[int, int]:
        """Find direction toward minimum discovery pheromone (for exploration).

        Args:
            pos: Current position
            grid_valid_check: Function to check if move is valid

        Returns:
            Direction (dx, dy) toward minimum pheromone
        """
        x, y = pos
        min_val = float('inf')
        best_dx, best_dy = 0, 0

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy

                # Check bounds
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue

                # Check validity if provided
                if grid_valid_check and not grid_valid_check((nx, ny)):
                    continue

                val = self.discovery[nx, ny]
                if val < min_val:
                    min_val = val
                    best_dx, best_dy = dx, dy

        return best_dx, best_dy
