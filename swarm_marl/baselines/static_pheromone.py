"""Static pheromone baseline policy (bio-inspired without learning)."""

import numpy as np
from typing import Dict, Tuple


class StaticPheromonePolicy:
    """Bio-inspired policy with fixed tau and pheromone following."""

    def __init__(self, num_agents: int, grid_size: int, tau: int = 5,
                 always_communicate: bool = True, seed: int = 42):
        """Initialize static pheromone policy.

        Args:
            num_agents: Number of agents
            grid_size: Grid size
            tau: Fixed threshold value
            always_communicate: Whether agents always communicate
            seed: Random seed
        """
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.tau = tau
        self.always_communicate = always_communicate
        self.rng = np.random.RandomState(seed)

        # Movement directions: N=0, S=1, E=2, W=3, STAY=4
        self.directions = {
            'N': 0, 'S': 1, 'E': 2, 'W': 3, 'STAY': 4
        }

    def get_actions(self, observations: Dict[str, np.ndarray],
                    infos: Dict[str, dict]) -> Dict[str, int]:
        """Get actions for all agents based on pheromone gradients.

        Args:
            observations: Dictionary of agent observations
            infos: Dictionary of agent info (contains role, position, etc.)

        Returns:
            Dictionary of agent_id -> action
        """
        actions = {}

        for agent_id, obs in observations.items():
            info = infos.get(agent_id, {})
            role = info.get('role', 0)  # 0=EXPLORER, 1=REPORTER

            # Extract pheromone info from observation
            # obs shape: flattened (5, 3, 3) + 5 scalars
            # Channels: cell_type, discovery_pheromone, return_pheromone, agent_presence, self_state
            window_size = 3

            # Unflatten first 45 elements (5 channels * 3 * 3)
            grid_obs = obs[:45].reshape(5, window_size, window_size)
            cell_type = grid_obs[0]
            discovery_pheromone = grid_obs[1]
            return_pheromone = grid_obs[2]

            # Get scalar features
            scalars = obs[45:]
            current_tau = int(scalars[0] * 10)  # Denormalize

            # Decide locomotion based on role
            if role == 0:  # EXPLORER
                # Move toward lowest discovery pheromone (anti-pheromone)
                locomotion = self._find_min_pheromone_direction(
                    discovery_pheromone, cell_type
                )
            else:  # REPORTER
                # Move toward highest return pheromone or center
                locomotion = self._find_max_pheromone_direction(
                    return_pheromone, cell_type
                )

            # Tau adjustment: always hold (1) since tau is fixed
            tau_adjust = 1  # Hold

            # Communication
            communicate = 1 if self.always_communicate else 0

            # Encode action: locomotion + 5 * (tau_adjust + 3 * communicate)
            action = locomotion + 5 * (tau_adjust + 3 * communicate)
            actions[agent_id] = action

        return actions

    def _find_min_pheromone_direction(self, pheromone_grid: np.ndarray,
                                       cell_type: np.ndarray) -> int:
        """Find direction toward minimum pheromone (for exploration)."""
        center = 1
        min_val = float('inf')
        best_direction = 4  # Default: STAY

        # Direction mapping: N=0, S=1, E=2, W=3
        direction_map = {
            (-1, 0): 0,  # North (row -1)
            (1, 0): 1,   # South (row +1)
            (0, 1): 2,   # East (col +1)
            (0, -1): 3,  # West (col -1)
        }

        for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            nr, nc = center + dr, center + dc

            # Check bounds
            if 0 <= nr < 3 and 0 <= nc < 3:
                # Check if not obstacle
                if cell_type[nr, nc] < 1.0:  # Not obstacle
                    val = pheromone_grid[nr, nc]
                    if val < min_val:
                        min_val = val
                        best_direction = direction_map[(dr, dc)]

        return best_direction

    def _find_max_pheromone_direction(self, pheromone_grid: np.ndarray,
                                       cell_type: np.ndarray) -> int:
        """Find direction toward maximum pheromone (for returning)."""
        center = 1
        max_val = -1
        best_direction = 4  # Default: STAY

        # Direction mapping: N=0, S=1, E=2, W=3
        direction_map = {
            (-1, 0): 0,
            (1, 0): 1,
            (0, 1): 2,
            (0, -1): 3,
        }

        for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            nr, nc = center + dr, center + dc

            if 0 <= nr < 3 and 0 <= nc < 3:
                if cell_type[nr, nc] < 1.0:
                    val = pheromone_grid[nr, nc]
                    if val > max_val:
                        max_val = val
                        best_direction = direction_map[(dr, dc)]

        # If no pheromone gradient, move randomly (but not stay)
        if max_val == 0:
            best_direction = self.rng.randint(0, 4)

        return best_direction

    def reset(self) -> None:
        """Reset policy state."""
        pass
