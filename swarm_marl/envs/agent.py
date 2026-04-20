"""Agent state and logic for SAR environment."""

import numpy as np
from typing import Tuple, Dict, Optional, Any
from enum import IntEnum


class AgentRole(IntEnum):
    """Agent roles in the swarm."""
    EXPLORER = 0
    REPORTER = 1


class Agent:
    """Individual agent in the swarm."""

    def __init__(self, agent_id: int, position: Tuple[int, int], tau: int = 5):
        """Initialize agent.

        Args:
            agent_id: Unique agent identifier
            position: Initial (x, y) position
            tau: Task switching threshold
        """
        self.agent_id = agent_id
        self.position = position
        self.tau = tau
        self.role = AgentRole.EXPLORER

        # Accumulated stimulus for task switching
        self.stimulus = 0.0

        # Local map (what this agent has seen)
        self.local_map = None

        # Communication buffer (received messages)
        self.comm_buffer = []

        # Previous action (for observation)
        self.prev_action = 0

        # Communication status
        self.communicated = False

        # Novelty tracking for selective communication
        self.cells_since_last_comm = 0
        self.new_cells_received = 0

    def reset(self, position: Tuple[int, int], tau: int = 5) -> None:
        """Reset agent state."""
        self.position = position
        self.tau = tau
        self.role = AgentRole.EXPLORER
        self.stimulus = 0.0
        self.local_map = None
        self.comm_buffer = []
        self.prev_action = 0
        self.communicated = False
        self.cells_since_last_comm = 0
        self.new_cells_received = 0

    def update_position(self, new_position: Tuple[int, int]) -> None:
        """Update agent position."""
        self.position = new_position

    def add_stimulus(self, amount: float) -> None:
        """Add to accumulated stimulus."""
        self.stimulus += amount

    def check_role_switch(self) -> bool:
        """Check if agent should switch roles based on stimulus threshold.

        Returns:
            True if role was switched
        """
        if self.stimulus > self.tau:
            # Switch role
            if self.role == AgentRole.EXPLORER:
                self.role = AgentRole.REPORTER
            else:
                self.role = AgentRole.EXPLORER
            # Reset stimulus
            self.stimulus = 0.0
            return True
        return False

    def adjust_tau(self, adjustment: int, tau_min: int = 1, tau_max: int = 10) -> None:
        """Adjust tau value.

        Args:
            adjustment: -1 (decrease), 0 (hold), or +1 (increase)
            tau_min: Minimum tau value
            tau_max: Maximum tau value
        """
        self.tau = max(tau_min, min(tau_max, self.tau + adjustment))

    def get_observation(self,
                        grid_world,
                        pheromone_field,
                        agent_positions: Dict[Tuple[int, int], int],
                        observation_radius: int = 1,
                        partial_observability: bool = False) -> np.ndarray:
        """Get observation tensor for this agent.

        Args:
            grid_world: GridWorld instance
            pheromone_field: PheromoneField instance
            agent_positions: Dictionary mapping positions to agent IDs
            observation_radius: Radius of observation window

        Returns:
            Observation tensor of shape (5, 2*radius+1, 2*radius+1) flattened
            Channels: cell_type, discovery_pheromone, return_pheromone, agent_presence, self_state
        """
        window_size = 2 * observation_radius + 1

        # Channel 0: Cell type from agent's local knowledge (partial obs) or global grid
        if partial_observability and self.local_map is not None:
            _ws = 2 * observation_radius + 1
            cell_type = np.full((_ws, _ws), 1, dtype=np.int32)  # default: OBSTACLE (=1)
            _cx, _cy = self.position
            for _dy in range(-observation_radius, observation_radius + 1):
                for _dx in range(-observation_radius, observation_radius + 1):
                    _x, _y = _cx + _dx, _cy + _dy
                    _wx, _wy = _dx + observation_radius, _dy + observation_radius
                    if 0 <= _x < self.local_map.shape[0] and 0 <= _y < self.local_map.shape[1]:
                        cell_type[_wx, _wy] = self.local_map[_x, _y]
        else:
            cell_type = grid_world.get_observation_window(self.position, observation_radius)
        # Normalize: obstacle=1.0, unexplored=0.66, explored=0.33, empty=0.0
        cell_type_norm = np.zeros_like(cell_type, dtype=np.float32)
        cell_type_norm[cell_type == 1] = 1.0  # OBSTACLE
        cell_type_norm[cell_type == 2] = 0.66  # UNEXPLORED
        cell_type_norm[cell_type == 3] = 0.33  # EXPLORED

        # Channel 1: Discovery pheromone levels
        discovery_pheromone, return_pheromone = pheromone_field.get_local_field(
            self.position, observation_radius)
        discovery_norm = discovery_pheromone.astype(np.float32) / 10.0  # Normalize to 0-1

        # Channel 2: Return pheromone levels
        return_norm = return_pheromone.astype(np.float32) / 10.0

        # Channel 3: Other agents' presence (binary)
        agent_presence = np.zeros((window_size, window_size), dtype=np.float32)
        cx, cy = self.position
        for dy in range(-observation_radius, observation_radius + 1):
            for dx in range(-observation_radius, observation_radius + 1):
                x, y = cx + dx, cy + dy
                wx, wy = dx + observation_radius, dy + observation_radius

                if (x, y) in agent_positions and agent_positions[(x, y)] != self.agent_id:
                    agent_presence[wx, wy] = 1.0

        # Channel 4: Agent's own state (broadcast as scalars across the window)
        self_state = np.zeros((window_size, window_size), dtype=np.float32)
        # Fill with normalized agent state values
        tau_norm = self.tau / 10.0  # Assuming max tau is 10
        role_val = float(self.role) / 1.0  # 0 or 1
        comm_val = 1.0 if self.communicated else 0.0
        stimulus_norm = min(self.stimulus / 10.0, 1.0)

        # Put agent state in center cell
        center = observation_radius
        self_state[center, center] = tau_norm

        # Stack channels
        observation = np.stack([
            cell_type_norm,
            discovery_norm,
            return_norm,
            agent_presence,
            self_state
        ], axis=0)

        # Flatten to 1D vector
        # Also append scalar features at the end
        scalar_features = np.array([
            tau_norm,
            role_val,
            comm_val,
            stimulus_norm,
            self.prev_action / 29.0,  # Normalize action (0-29) to 0-1
        ], dtype=np.float32)

        flattened = observation.flatten()
        full_observation = np.concatenate([flattened, scalar_features])

        return full_observation.astype(np.float32)

    def receive_message(self, message: Dict[str, Any]) -> None:
        """Receive a message from another agent."""
        self.comm_buffer.append(message)

    def clear_comm_buffer(self) -> None:
        """Clear communication buffer."""
        self.comm_buffer = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent state to dictionary."""
        return {
            'agent_id': self.agent_id,
            'position': self.position,
            'tau': self.tau,
            'role': int(self.role),
            'stimulus': self.stimulus,
            'communicated': self.communicated,
        }
