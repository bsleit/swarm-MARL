"""Communication models for the SAR environment."""

import numpy as np
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod


class CommunicationModel(ABC):
    """Abstract base class for communication models."""

    @abstractmethod
    def can_communicate(self, pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> bool:
        """Check if two agents can communicate.

        Args:
            pos_a: Position of agent A
            pos_b: Position of agent B

        Returns:
            True if communication is possible
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the communication model state."""
        pass


class FixedDenialModel(CommunicationModel):
    """Fixed denial zones communication model."""

    def __init__(self, denial_zones: List[List[int]], grid_size: Optional[int] = None):
        """Initialize fixed denial model.

        Args:
            denial_zones: List of [x1, y1, x2, y2] rectangles where comm is denied
            grid_size: Size of the grid (for validation)
        """
        self.denial_zones = denial_zones
        self.grid_size = grid_size

    def _in_denial_zone(self, pos: Tuple[int, int]) -> bool:
        """Check if position is in a denial zone."""
        x, y = pos
        for zone in self.denial_zones:
            x1, y1, x2, y2 = zone
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    def can_communicate(self, pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> bool:
        """Check if agents can communicate (neither in denial zone)."""
        return not (self._in_denial_zone(pos_a) or self._in_denial_zone(pos_b))

    def reset(self) -> None:
        """No state to reset."""
        pass


class ProbabilisticModel(CommunicationModel):
    """Probabilistic per-cell communication model."""

    def __init__(self, grid_size: int, prob_map: Optional[np.ndarray] = None,
                 default_prob: float = 0.8, seed: Optional[int] = None):
        """Initialize probabilistic model.

        Args:
            grid_size: Size of the grid
            prob_map: 2D array of success probabilities per cell, or None for random
            default_prob: Default probability if generating random map
            seed: Random seed
        """
        self.grid_size = grid_size
        self.rng = np.random.RandomState(seed)

        if prob_map is not None:
            self.prob_map = prob_map
        else:
            # Generate random probability map centered around default_prob
            self.prob_map = self.rng.uniform(
                max(0, default_prob - 0.2),
                min(1.0, default_prob + 0.2),
                (grid_size, grid_size)
            )

    def can_communicate(self, pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> bool:
        """Check if agents can communicate based on cell probabilities.

        Both cells must succeed for communication to work.
        """
        # Communication succeeds if both cells are "active"
        prob_a = self.prob_map[pos_a[0], pos_a[1]]
        prob_b = self.prob_map[pos_b[0], pos_b[1]]

        # Use average probability as success chance
        success_prob = (prob_a + prob_b) / 2.0
        return self.rng.random() < success_prob

    def reset(self) -> None:
        """No state to reset."""
        pass


class DistanceZoneModel(CommunicationModel):
    """Distance-based communication with denial zones."""

    def __init__(self, comm_range: float, denial_zones: List[List[int]]):
        """Initialize distance + zones model.

        Args:
            comm_range: Maximum Euclidean distance for communication
            denial_zones: List of [x1, y1, x2, y2] rectangles where comm is denied
        """
        self.comm_range = comm_range
        self.denial_zones = denial_zones

    def _in_denial_zone(self, pos: Tuple[int, int]) -> bool:
        """Check if position is in a denial zone."""
        x, y = pos
        for zone in self.denial_zones:
            x1, y1, x2, y2 = zone
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    def _in_range(self, pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> bool:
        """Check if positions are within communication range."""
        dx = pos_a[0] - pos_b[0]
        dy = pos_a[1] - pos_b[1]
        distance = np.sqrt(dx * dx + dy * dy)
        return distance <= self.comm_range

    def can_communicate(self, pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> bool:
        """Check if agents can communicate (in range and not in denial zone)."""
        # Check distance first
        if not self._in_range(pos_a, pos_b):
            return False

        # Check denial zones
        if self._in_denial_zone(pos_a) or self._in_denial_zone(pos_b):
            return False

        return True

    def reset(self) -> None:
        """No state to reset."""
        pass


def create_comm_model(config: dict, grid_size: int,
                      seed: Optional[int] = None) -> CommunicationModel:
    """Factory function to create communication model from config.

    Args:
        config: Communication configuration dict
        grid_size: Size of the grid
        seed: Random seed

    Returns:
        CommunicationModel instance
    """
    model_type = config.get('model', 'fixed_denial')

    if model_type == 'fixed_denial':
        return FixedDenialModel(
            denial_zones=config.get('denial_zones', []),
            grid_size=grid_size
        )
    elif model_type == 'probabilistic':
        prob_map = config.get('prob_map')
        if prob_map is not None:
            prob_map = np.array(prob_map)
        return ProbabilisticModel(
            grid_size=grid_size,
            prob_map=prob_map,
            default_prob=config.get('default_prob', 0.8),
            seed=seed
        )
    elif model_type == 'distance_zones':
        return DistanceZoneModel(
            comm_range=config.get('range', 10.0),
            denial_zones=config.get('denial_zones', [])
        )
    else:
        raise ValueError(f"Unknown communication model: {model_type}")
