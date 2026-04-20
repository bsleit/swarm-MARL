"""Static pheromone baseline policy (bio-inspired without learning)."""

import numpy as np
from typing import Dict, Tuple


class StaticPheromonePolicy:
    """Bio-inspired policy with fixed tau and pheromone following."""

    # Cell-type values used in the normalized observation (from agent.py)
    _OBSTACLE = 1.0
    _UNEXPLORED = 0.66
    _EXPLORED = 0.33

    # Tolerance for float comparison of cell types
    _EPS = 0.05

    def __init__(self, num_agents: int, grid_size: int, tau: int = 5,
                 always_communicate: bool = True, novelty_threshold: int = 3,
                 seed: int = 42):
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
        self.novelty_threshold = novelty_threshold
        self.rng = np.random.RandomState(seed)

        # Movement directions: N=0, S=1, E=2, W=3, STAY=4
        self.directions = {
            'N': 0, 'S': 1, 'E': 2, 'W': 3, 'STAY': 4
        }

        # Track last direction per agent to maintain momentum
        self._last_dir: Dict[str, int] = {}

        # Opposite direction lookup for anti-reversal
        self._opposite = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4}

        # Ballistic escape state: when stuck in fully-explored areas,
        # pick a random direction and persist for several steps.
        self._stuck_count: Dict[str, int] = {}    # consecutive "no unexplored" steps
        self._escape_dir: Dict[str, int] = {}     # locked escape direction
        self._escape_steps: Dict[str, int] = {}   # remaining escape steps
        self._STUCK_THRESHOLD = 3   # steps before triggering escape
        self._ESCAPE_LENGTH = 6     # steps to persist in escape direction

        # Direction offsets for the 3x3 observation window (center=1).
        # Window is indexed [dx+r, dy+r] where the real position is
        # (cx+dx, cy+dy).  Movement deltas from Direction are:
        #   NORTH=0 → (0,-1), SOUTH=1 → (0,+1),
        #   EAST=2  → (+1,0), WEST=3  → (-1,0).
        # So window row offset dr corresponds to dx, and col offset dc
        # corresponds to dy.
        self._dir_offsets = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self._dir_map = {
            (-1, 0): 3,   # dx=-1 → WEST
            (1, 0):  2,   # dx=+1 → EAST
            (0, 1):  1,   # dy=+1 → SOUTH
            (0, -1): 0,   # dy=-1 → NORTH
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

            last_dir = self._last_dir.get(agent_id, None)

            # Check if agent is in ballistic escape mode
            if self._escape_steps.get(agent_id, 0) > 0:
                # Cancel escape if unexplored cells are visible again
                has_unexplored = any(
                    abs(cell_type[1 + dr, 1 + dc] - self._UNEXPLORED) < self._EPS
                    for dr, dc in self._dir_offsets
                    if 0 <= 1 + dr < 3 and 0 <= 1 + dc < 3
                )
                if has_unexplored:
                    self._escape_steps[agent_id] = 0
                    self._stuck_count[agent_id] = 0
                    # Fall through to normal exploration
                    if role == 0:
                        locomotion, _ = self._explore_direction(
                            cell_type, discovery_pheromone, prefer_low=True,
                            last_dir=last_dir)
                    else:
                        locomotion, _ = self._explore_direction(
                            cell_type, return_pheromone, prefer_low=False,
                            last_dir=last_dir)
                else:
                    locomotion = self._escape_step(agent_id, cell_type)
            else:
                # Both roles explore; explorers use anti-discovery-pheromone as
                # tiebreaker, reporters use pro-return-pheromone as tiebreaker.
                if role == 0:  # EXPLORER
                    locomotion, had_unexplored = self._explore_direction(
                        cell_type, discovery_pheromone, prefer_low=True,
                        last_dir=last_dir)
                else:  # REPORTER
                    locomotion, had_unexplored = self._explore_direction(
                        cell_type, return_pheromone, prefer_low=False,
                        last_dir=last_dir)

                # Track stuck state for ballistic escape
                if had_unexplored:
                    self._stuck_count[agent_id] = 0
                else:
                    self._stuck_count[agent_id] = self._stuck_count.get(agent_id, 0) + 1
                    if self._stuck_count[agent_id] >= self._STUCK_THRESHOLD:
                        # Trigger ballistic escape in a random non-obstacle direction
                        locomotion = self._start_escape(agent_id, cell_type, last_dir)

            self._last_dir[agent_id] = locomotion

            # Increase tau so agents stay as explorers longer (tau_adjust=2 → +1)
            tau_adjust = 2  # Increase

            # Communication: always-on or novelty-triggered (selective)
            if self.always_communicate:
                communicate = 1
            else:
                cells_since_last_comm = info.get('cells_since_last_comm', 0)
                communicate = 1 if cells_since_last_comm >= self.novelty_threshold else 0

            # Encode action: locomotion + 5 * (tau_adjust + 3 * communicate)
            action = locomotion + 5 * (tau_adjust + 3 * communicate)
            actions[agent_id] = action

        return actions

    def _explore_direction(self, cell_type: np.ndarray,
                           pheromone_grid: np.ndarray,
                           prefer_low: bool,
                           last_dir: int = None) -> Tuple[int, bool]:
        """Pick the best direction for exploration.

        Priority order:
        1. Unexplored, non-obstacle neighbors (primary goal)
        2. Explored, non-obstacle neighbors (fallback)
        Within each group, use the pheromone grid as tiebreaker
        (lowest if prefer_low, highest otherwise).
        When all pheromone values tie, prefer continuing in the same
        direction (momentum) and avoid reversing.

        Returns:
            (direction, had_unexplored) — direction to move and whether
            any unexplored neighbor was available.
        """
        center = 1
        unexplored = []  # (pheromone_val, direction)
        explored = []

        for dr, dc in self._dir_offsets:
            nr, nc = center + dr, center + dc
            if not (0 <= nr < 3 and 0 <= nc < 3):
                continue
            ct = cell_type[nr, nc]
            # Skip obstacles
            if ct > self._OBSTACLE - self._EPS:
                continue

            direction = self._dir_map[(dr, dc)]
            pval = pheromone_grid[nr, nc]

            if abs(ct - self._UNEXPLORED) < self._EPS:
                unexplored.append((pval, direction))
            else:
                explored.append((pval, direction))

        had_unexplored = len(unexplored) > 0

        # Pick from the best available group
        candidates = unexplored if unexplored else explored
        if not candidates:
            return 4, had_unexplored  # STAY (surrounded by obstacles)

        # Tiebreak by pheromone
        if prefer_low:
            best_val = min(c[0] for c in candidates)
        else:
            best_val = max(c[0] for c in candidates)

        best = [c[1] for c in candidates if c[0] == best_val]

        # Momentum: if last direction is among the best, prefer it
        if last_dir is not None and last_dir in best:
            return last_dir, had_unexplored

        # Anti-reversal: remove the opposite of last direction if other
        # options exist, to avoid oscillation
        if last_dir is not None and len(best) > 1:
            opp = self._opposite[last_dir]
            best = [d for d in best if d != opp] or best

        return self.rng.choice(best), had_unexplored

    def _get_passable_dirs(self, cell_type: np.ndarray) -> list:
        """Return list of directions that are not blocked by obstacles."""
        center = 1
        passable = []
        for dr, dc in self._dir_offsets:
            nr, nc = center + dr, center + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                ct = cell_type[nr, nc]
                if ct < self._OBSTACLE - self._EPS:
                    passable.append(self._dir_map[(dr, dc)])
        return passable

    def _start_escape(self, agent_id: str, cell_type: np.ndarray,
                      last_dir: int) -> int:
        """Trigger ballistic escape: pick a random passable direction
        (avoiding the opposite of last_dir) and lock it for several steps."""
        passable = self._get_passable_dirs(cell_type)
        if not passable:
            return 4  # fully surrounded by obstacles

        # Prefer directions other than the opposite of last movement
        if last_dir is not None and len(passable) > 1:
            opp = self._opposite[last_dir]
            filtered = [d for d in passable if d != opp]
            if filtered:
                passable = filtered

        escape_dir = self.rng.choice(passable)
        self._escape_dir[agent_id] = escape_dir
        self._escape_steps[agent_id] = self._ESCAPE_LENGTH
        self._stuck_count[agent_id] = 0
        return escape_dir

    def _escape_step(self, agent_id: str, cell_type: np.ndarray) -> int:
        """Continue ballistic escape. If the locked direction is blocked,
        pick a new random passable direction (bounce)."""
        self._escape_steps[agent_id] -= 1
        esc_dir = self._escape_dir[agent_id]

        # Check if locked direction is passable
        passable = self._get_passable_dirs(cell_type)
        if esc_dir in passable:
            return esc_dir

        # Bounce: pick a new direction from passable ones
        if not passable:
            self._escape_steps[agent_id] = 0
            return 4

        opp = self._opposite[esc_dir]
        filtered = [d for d in passable if d != opp]
        new_dir = self.rng.choice(filtered if filtered else passable)
        self._escape_dir[agent_id] = new_dir
        return new_dir

        return self.rng.choice(best)

    def reset(self) -> None:
        """Reset policy state."""
        pass
