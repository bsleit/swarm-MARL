"""Grid world implementation for SAR environment."""

import numpy as np
from typing import Tuple, List, Optional
from enum import IntEnum


class CellType(IntEnum):
    """Cell types in the grid world."""
    EMPTY = 0
    OBSTACLE = 1
    UNEXPLORED = 2
    EXPLORED = 3


class Direction:
    """Direction constants."""
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    STAY = 4

    DELTA = {
        NORTH: (0, -1),
        SOUTH: (0, 1),
        EAST: (1, 0),
        WEST: (-1, 0),
        STAY: (0, 0),
    }


class GridWorld:
    """2D grid world for SAR simulation."""

    def __init__(self, size: int, obstacle_density: float = 0.15, seed: Optional[int] = None):
        """Initialize the grid world.

        Args:
            size: Grid size (size x size)
            obstacle_density: Percentage of cells to fill with obstacles
            seed: Random seed for reproducibility
        """
        self.size = size
        self.obstacle_density = obstacle_density
        self.rng = np.random.RandomState(seed)

        # Initialize grid: all cells start as unexplored
        self.grid = np.full((size, size), CellType.UNEXPLORED, dtype=np.int32)

        # Track agent positions (cell -> agent_id or None)
        self.agent_positions = {}

        # Track coverage
        self.total_explorable = size * size
        self.explored_count = 0

    def reset(self, agent_positions: Optional[dict] = None) -> None:
        """Reset the grid world."""
        # Clear grid
        self.grid.fill(CellType.UNEXPLORED)

        # Generate obstacles
        self._generate_obstacles()

        # Reset agent positions
        self.agent_positions = agent_positions if agent_positions else {}

        # Reset coverage count
        self.explored_count = 0

    def _generate_obstacles(self) -> None:
        """Generate random obstacles in the grid."""
        num_obstacles = int(self.size * self.size * self.obstacle_density)

        # Get all valid positions
        all_positions = [(x, y) for x in range(self.size) for y in range(self.size)]

        # Randomly select positions for obstacles
        obstacle_positions = self.rng.choice(len(all_positions), num_obstacles, replace=False)

        for idx in obstacle_positions:
            x, y = all_positions[idx]
            self.grid[x, y] = CellType.OBSTACLE
            self.total_explorable -= 1

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not an obstacle."""
        x, y = pos
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        return self.grid[x, y] != CellType.OBSTACLE

    def is_occupied(self, pos: Tuple[int, int]) -> bool:
        """Check if position is occupied by another agent."""
        return pos in self.agent_positions and self.agent_positions[pos] is not None

    def is_valid_move(self, pos: Tuple[int, int], direction: int) -> bool:
        """Check if move is valid (in bounds, not obstacle, not occupied)."""
        dx, dy = Direction.DELTA[direction]
        new_pos = (pos[0] + dx, pos[1] + dy)

        if not self.is_valid_position(new_pos):
            return False

        if self.is_occupied(new_pos):
            return False

        return True

    def get_new_position(self, pos: Tuple[int, int], direction: int) -> Tuple[int, int]:
        """Get new position after moving in direction."""
        dx, dy = Direction.DELTA[direction]
        return (pos[0] + dx, pos[1] + dy)

    def mark_explored(self, pos: Tuple[int, int]) -> bool:
        """Mark a cell as explored. Returns True if newly explored."""
        x, y = pos
        if 0 <= x < self.size and 0 <= y < self.size:
            if self.grid[x, y] == CellType.UNEXPLORED:
                self.grid[x, y] = CellType.EXPLORED
                self.explored_count += 1
                return True
        return False

    def get_coverage(self) -> float:
        """Get current coverage rate."""
        if self.total_explorable == 0:
            return 1.0
        return self.explored_count / self.total_explorable

    def get_cell_type(self, pos: Tuple[int, int]) -> int:
        """Get cell type at position."""
        x, y = pos
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.grid[x, y]
        return CellType.OBSTACLE

    def get_observation_window(self, pos: Tuple[int, int], radius: int = 1) -> np.ndarray:
        """Get observation window around position.

        Returns a (2*radius+1, 2*radius+1) array with cell types.
        """
        window_size = 2 * radius + 1
        window = np.full((window_size, window_size), CellType.OBSTACLE, dtype=np.int32)

        cx, cy = pos
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = cx + dx, cy + dy
                wx, wy = dx + radius, dy + radius

                if 0 <= x < self.size and 0 <= y < self.size:
                    window[wx, wy] = self.grid[x, y]
                # else: remains OBSTACLE (out of bounds)

        return window

    def get_empty_positions(self, n: int = 1) -> List[Tuple[int, int]]:
        """Get n random empty (non-obstacle) positions."""
        empty_positions = []
        for x in range(self.size):
            for y in range(self.size):
                if self.grid[x, y] != CellType.OBSTACLE:
                    empty_positions.append((x, y))

        if len(empty_positions) < n:
            raise ValueError(f"Not enough empty positions. Requested {n}, have {len(empty_positions)}")

        selected_indices = self.rng.choice(len(empty_positions), n, replace=False)
        return [empty_positions[i] for i in selected_indices]

    def update_agent_position(self, agent_id: int, old_pos: Optional[Tuple[int, int]],
                               new_pos: Tuple[int, int]) -> None:
        """Update agent position tracking."""
        if old_pos is not None:
            self.agent_positions.pop(old_pos, None)
        self.agent_positions[new_pos] = agent_id
