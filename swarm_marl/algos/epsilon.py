"""Epsilon-greedy exploration schedule."""

import numpy as np


class EpsilonScheduler:
    """Linear epsilon decay schedule."""

    def __init__(self, start: float = 1.0, end: float = 0.05,
                 decay_steps: int = 50000):
        """Initialize epsilon scheduler.

        Args:
            start: Starting epsilon value
            end: Ending epsilon value
            decay_steps: Number of steps to decay over
        """
        self.start = start
        self.end = end
        self.decay_steps = decay_steps

    def get_epsilon(self, step: int) -> float:
        """Get epsilon value for given step.

        Args:
            step: Current training step

        Returns:
            Epsilon value
        """
        if step >= self.decay_steps:
            return self.end

        # Linear decay
        progress = step / self.decay_steps
        epsilon = self.start + (self.end - self.start) * progress
        return epsilon

    def reset(self) -> None:
        """Reset scheduler (no-op)."""
        pass
