"""Random walk baseline policy."""

import numpy as np
from typing import Dict, Tuple


class RandomWalkPolicy:
    """Random walk baseline policy."""

    def __init__(self, num_agents: int, action_space_size: int = 30, seed: int = 42):
        """Initialize random walk policy.

        Args:
            num_agents: Number of agents
            action_space_size: Size of action space (default 30)
            seed: Random seed
        """
        self.num_agents = num_agents
        self.action_space_size = action_space_size
        self.rng = np.random.RandomState(seed)

    def get_actions(self, observations: Dict[str, np.ndarray],
                    infos: Dict[str, dict]) -> Dict[str, int]:
        """Get random actions for all agents.

        Args:
            observations: Dictionary of agent observations
            infos: Dictionary of agent info

        Returns:
            Dictionary of agent_id -> action
        """
        return {
            agent_id: self.rng.randint(0, self.action_space_size)
            for agent_id in observations.keys()
        }

    def reset(self) -> None:
        """Reset policy state."""
        pass
