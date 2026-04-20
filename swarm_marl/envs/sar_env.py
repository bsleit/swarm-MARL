"""Main SAR environment as a PettingZoo ParallelEnv."""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from pettingzoo import ParallelEnv
from gymnasium import spaces

from .grid_world import GridWorld, Direction, CellType
from .pheromone import PheromoneField
from .agent import Agent, AgentRole
from .comm_model import create_comm_model


class SAREnv(ParallelEnv):
    """Search and Rescue environment for multi-agent reinforcement learning."""

    metadata = {
        "name": "sar_env_v0",
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(self, config: dict, render_mode: Optional[str] = None):
        """Initialize SAR environment.

        Args:
            config: Configuration dictionary
            render_mode: Rendering mode
        """
        super().__init__()

        self.config = config
        self.render_mode = render_mode

        # Extract config values
        self.grid_size = config['env']['grid_size']
        self._num_agents = config['env']['num_agents']
        self.max_steps = config['env']['max_steps']
        self.obstacle_density = config['env']['obstacle_density']
        self.coverage_threshold = config['env']['coverage_threshold']

        # Pheromone config
        self.pheromone_max = config['pheromone']['max_level']
        self.pheromone_decay = config['pheromone']['decay_rate']
        self.discovery_deposit = config['pheromone']['discovery_deposit']
        self.return_deposit = config['pheromone']['return_deposit']

        # Agent config
        self.tau_min = config['agent']['tau_min']
        self.tau_max = config['agent']['tau_max']
        self.tau_default = config['agent']['tau_default']
        self.tau_delta = config['agent']['tau_delta']
        self.obs_radius = config['agent']['observation_radius']
        self.partial_observability = config['env'].get('partial_observability', False)

        # Reward weights
        self.alpha = config['reward']['alpha']
        self.beta = config['reward']['beta']
        self.gamma = config['reward']['gamma']

        # Initialize components
        self.grid_world = GridWorld(
            size=self.grid_size,
            obstacle_density=self.obstacle_density,
            seed=config.get('seed', 42)
        )

        self.pheromone_field = PheromoneField(
            size=self.grid_size,
            max_level=self.pheromone_max,
            decay_rate=self.pheromone_decay
        )

        self.comm_model = create_comm_model(
            config['comm'],
            self.grid_size,
            config.get('seed', 42)
        )

        # Agent objects
        self.agents = {}

        # PettingZoo required attributes - MUST set possible_agents before other initialization
        self.possible_agents = [f"agent_{i}" for i in range(config['env']['num_agents'])]
        self.agents = []

        # Episode state
        self.steps = 0
        self.prev_coverage = 0.0
        self.cumulative_reward = 0.0

        # Action space: 5 locomotion * 3 tau-adjust * 2 communicate = 30 discrete actions
        self.action_spaces = {
            agent: spaces.Discrete(30) for agent in self.possible_agents
        }

        # Observation space: 5 channels * 3x3 window + 5 scalar features = 50
        obs_dim = 5 * (2 * self.obs_radius + 1) ** 2 + 5
        self.observation_spaces = {
            agent: spaces.Box(
                low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            ) for agent in self.possible_agents
        }

        # Global state space (for QMIX)
        # Positions (n_agents * 2) + grid (size^2) + pheromones (2 * size^2) + comm links (n_agents^2)
        state_dim = (self._num_agents * 2 + self.grid_size ** 2 +
                     2 * self.grid_size ** 2 + self._num_agents ** 2)
        self.state_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment."""
        self.agents = self.possible_agents[:]

        # Reset components (propagate seed to grid_world for reproducibility)
        self.grid_world.reset(seed=seed)
        self.pheromone_field.reset()
        self.comm_model.reset()

        # Initialize agents at random positions
        agent_positions = self.grid_world.get_empty_positions(self._num_agents)
        self.agents_dict = {}

        agent_pos_map = {}
        for i, agent_id in enumerate(self.agents):
            pos = agent_positions[i]
            self.agents_dict[agent_id] = Agent(
                agent_id=i,
                position=pos,
                tau=self.tau_default
            )
            agent_pos_map[pos] = i

        self.grid_world.agent_positions = agent_pos_map

        # Reset episode state
        self.steps = 0
        self.prev_coverage = 0.0
        self.cumulative_reward = 0.0

        # Mark initial positions as explored; initialise each agent's local knowledge
        for agent in self.agents_dict.values():
            self.grid_world.mark_explored(agent.position)
            if self.partial_observability:
                # Agents know obstacle layout but start with everything else unexplored
                local_map = np.full(
                    (self.grid_size, self.grid_size), CellType.UNEXPLORED, dtype=np.int32
                )
                local_map[self.grid_world.grid == CellType.OBSTACLE] = CellType.OBSTACLE
                agent.local_map = local_map
                self._reveal_local_map(agent)
            else:
                agent.local_map = self.grid_world.grid.copy()

        # Get initial observations
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute one step in the environment."""
        self.steps += 1

        # Parse and execute actions
        comm_count = 0

        for agent_id, action in actions.items():
            if agent_id not in self.agents_dict:
                continue

            agent = self.agents_dict[agent_id]
            locomotion, tau_adjust, communicate = self._decode_action(action)

            # Execute locomotion
            if locomotion != Direction.STAY:
                if self.grid_world.is_valid_move(agent.position, locomotion):
                    old_pos = agent.position
                    new_pos = self.grid_world.get_new_position(agent.position, locomotion)
                    agent.update_position(new_pos)
                    self.grid_world.update_agent_position(
                        agent.agent_id, old_pos, new_pos
                    )

            # Adjust tau
            tau_adjustment = {0: -self.tau_delta, 1: 0, 2: self.tau_delta}[tau_adjust]
            agent.adjust_tau(tau_adjustment, self.tau_min, self.tau_max)

            # Communication
            agent.communicated = (communicate == 1)
            if agent.communicated:
                comm_count += 1

            # Store previous action for observation
            agent.prev_action = action

            # Mark current cell as explored
            newly_explored = self.grid_world.mark_explored(agent.position)
            if newly_explored:
                agent.add_stimulus(1.0)  # Add stimulus for discovery

            # Update agent's partial-observability local map
            if self.partial_observability:
                new_cells = self._reveal_local_map(agent)
                agent.cells_since_last_comm += new_cells

            # Deposit pheromone based on role
            if agent.role == AgentRole.EXPLORER:
                self.pheromone_field.deposit(
                    agent.position, 'discovery', self.discovery_deposit
                )
            else:
                self.pheromone_field.deposit(
                    agent.position, 'return', self.return_deposit
                )

            # Check role switch
            agent.check_role_switch()

        # Reset per-step received-cells counter before processing communications
        if self.partial_observability:
            for _ag in self.agents_dict.values():
                _ag.new_cells_received = 0

        # Process communications
        self._process_communications()

        # After broadcasting: reset novelty counter so agents don't re-broadcast immediately
        if self.partial_observability:
            for _ag in self.agents_dict.values():
                if _ag.communicated:
                    _ag.cells_since_last_comm = 0

        # Decay pheromones
        self.pheromone_field.decay()

        # Compute reward
        coverage = self.grid_world.get_coverage()
        delta_coverage = coverage - self.prev_coverage
        self.prev_coverage = coverage

        # Reward: alpha * delta_coverage - beta - gamma * comm_count
        step_penalty = self.beta
        comm_penalty = self.gamma * comm_count
        coverage_reward = self.alpha * delta_coverage * 100  # Scale up for meaningful signal

        reward = coverage_reward - step_penalty - comm_penalty

        # Get observations
        observations = self._get_observations()

        # Check termination
        terminations = {}
        truncations = {}
        infos = {}

        terminated = coverage >= self.coverage_threshold
        truncated = self.steps >= self.max_steps

        for agent_id in self.agents:
            terminations[agent_id] = terminated
            truncations[agent_id] = truncated
            _ag = self.agents_dict[agent_id]
            infos[agent_id] = {
                'coverage': coverage,
                'steps': self.steps,
                'role': int(_ag.role),
                'tau': _ag.tau,
                'cells_since_last_comm': _ag.cells_since_last_comm,
                'new_cells_received': _ag.new_cells_received,
            }

        # Remove terminated agents
        if terminated or truncated:
            self.agents = []

        return observations, {agent: reward for agent in observations.keys()}, terminations, truncations, infos

    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        """Decode single discrete action into locomotion, tau_adjust, communicate.

        Action space: 5 * 3 * 2 = 30
        action = locomotion + 5 * (tau_adjust + 3 * communicate)
        """
        locomotion = action % 5
        remaining = action // 5
        tau_adjust = remaining % 3
        communicate = remaining // 3
        return locomotion, tau_adjust, communicate

    def _process_communications(self) -> None:
        """Process communication between agents."""
        transmitting_agents = [
            agent for agent in self.agents_dict.values()
            if agent.communicated
        ]

        for sender in transmitting_agents:
            for receiver in self.agents_dict.values():
                if sender.agent_id != receiver.agent_id:
                    # Check if communication is possible
                    if self.comm_model.can_communicate(sender.position, receiver.position):
                        # Share local map
                        receiver.receive_message({
                            'sender': sender.agent_id,
                            'map': sender.local_map.copy() if sender.local_map is not None else None,
                            'position': sender.position,
                            'role': sender.role,
                        })
                        # Merge maps
                        if receiver.local_map is not None and sender.local_map is not None:
                            if self.partial_observability:
                                # OBSTACLE=1 < UNEXPLORED=2 < EXPLORED=3, so np.maximum
                                # would silently overwrite obstacle cells with higher values.
                                # Preserve obstacle knowledge: a cell is an obstacle if
                                # either agent's map marks it as such.
                                before_map = receiver.local_map.copy()
                                merged = np.maximum(receiver.local_map, sender.local_map)
                                obstacle_mask = (
                                    (receiver.local_map == CellType.OBSTACLE) |
                                    (sender.local_map == CellType.OBSTACLE)
                                )
                                merged[obstacle_mask] = CellType.OBSTACLE
                                receiver.local_map = merged
                                receiver.new_cells_received += int(
                                    np.sum(before_map != merged)
                                )
                            else:
                                receiver.local_map = np.maximum(
                                    receiver.local_map, sender.local_map
                                )

    def _reveal_local_map(self, agent: Agent) -> int:
        """Sync agent's local_map with global grid within obs_radius.

        Returns the number of cells newly revealed (transitioned from UNEXPLORED).
        """
        if agent.local_map is None:
            return 0
        revealed = 0
        cx, cy = agent.position
        for dy in range(-self.obs_radius, self.obs_radius + 1):
            for dx in range(-self.obs_radius, self.obs_radius + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    global_val = self.grid_world.grid[x, y]
                    local_val = agent.local_map[x, y]
                    if local_val != global_val:
                        if local_val == CellType.UNEXPLORED:
                            revealed += 1
                        agent.local_map[x, y] = global_val
        return revealed

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents."""
        observations = {}
        for agent_id, agent in self.agents_dict.items():
            if agent_id in self.agents:
                observations[agent_id] = agent.get_observation(
                    self.grid_world,
                    self.pheromone_field,
                    self.grid_world.agent_positions,
                    self.obs_radius,
                    partial_observability=self.partial_observability,
                )
        return observations

    def get_global_state(self) -> np.ndarray:
        """Get global state for centralized training.

        Returns flattened state vector containing:
        - All agent positions
        - Full grid coverage
        - Full pheromone fields
        - Communication link states
        """
        state_parts = []

        # Agent positions (normalized)
        positions = np.zeros(self._num_agents * 2, dtype=np.float32)
        for i, agent_id in enumerate(self.possible_agents):
            if agent_id in self.agents_dict:
                agent = self.agents_dict[agent_id]
                positions[i * 2] = agent.position[0] / self.grid_size
                positions[i * 2 + 1] = agent.position[1] / self.grid_size
        state_parts.append(positions)

        # Grid state (normalized)
        grid_norm = self.grid_world.grid.astype(np.float32) / 3.0
        state_parts.append(grid_norm.flatten())

        # Pheromone fields (normalized)
        discovery_norm = self.pheromone_field.discovery.astype(np.float32) / self.pheromone_max
        return_norm = self.pheromone_field.return_pheromone.astype(np.float32) / self.pheromone_max
        state_parts.append(discovery_norm.flatten())
        state_parts.append(return_norm.flatten())

        # Communication links (n_agents x n_agents binary matrix)
        comm_links = np.zeros(self._num_agents * self._num_agents, dtype=np.float32)
        for i, agent_id_i in enumerate(self.possible_agents):
            for j, agent_id_j in enumerate(self.possible_agents):
                if i != j and agent_id_i in self.agents_dict and agent_id_j in self.agents_dict:
                    agent_i = self.agents_dict[agent_id_i]
                    agent_j = self.agents_dict[agent_id_j]
                    if self.comm_model.can_communicate(agent_i.position, agent_j.position):
                        comm_links[i * self._num_agents + j] = 1.0
        state_parts.append(comm_links)

        return np.concatenate(state_parts)

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode is None:
            return None

        # Return RGB array for visualization
        # This would be implemented with matplotlib in visualizer.py
        return None

    def state(self) -> np.ndarray:
        """Get current global state."""
        return self.get_global_state()
