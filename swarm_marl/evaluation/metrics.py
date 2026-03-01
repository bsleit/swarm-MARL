"""Evaluation metrics for SAR experiments."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class MetricsTracker:
    """Tracks and computes evaluation metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.episode_data = []
        self.reset_current()

    def reset_current(self) -> None:
        """Reset current episode data."""
        self.current = {
            'coverages': [],
            'rewards': [],
            'communications': [],
            'steps': 0,
            'terminated': False,
        }

    def add_step(self, coverage: float, reward: float,
                 communication_count: int) -> None:
        """Add step data."""
        self.current['coverages'].append(coverage)
        self.current['rewards'].append(reward)
        self.current['communications'].append(communication_count)
        self.current['steps'] += 1

    def end_episode(self, terminated: bool) -> Dict[str, float]:
        """End current episode and return metrics."""
        self.current['terminated'] = terminated

        # Compute metrics
        metrics = self.compute_episode_metrics(self.current)
        self.episode_data.append(self.current)

        return metrics

    def compute_episode_metrics(self, episode: Dict) -> Dict[str, float]:
        """Compute metrics for a single episode."""
        final_coverage = episode['coverages'][-1] if episode['coverages'] else 0.0
        total_reward = sum(episode['rewards'])
        total_comms = sum(episode['communications'])

        return {
            'coverage_rate': final_coverage,
            'total_reward': total_reward,
            'episode_length': episode['steps'],
            'total_communications': total_comms,
            'terminated': float(episode['terminated']),
        }

    @staticmethod
    def coverage_rate(coverages: List[float], t_max: int) -> float:
        """Compute C_R: coverage rate at end of episode.

        Args:
            coverages: List of coverage values over time
            t_max: Maximum timestep

        Returns:
            Coverage rate at t_max or end of episode
        """
        if not coverages:
            return 0.0

        if len(coverages) <= t_max:
            return coverages[-1]
        else:
            return coverages[t_max]

    @staticmethod
    def time_to_completion(coverages: List[float],
                           threshold: float = 0.9) -> Optional[int]:
        """Compute T_C: first timestep where coverage >= threshold.

        Args:
            coverages: List of coverage values over time
            threshold: Coverage threshold

        Returns:
            Timestep where threshold was reached, or None if not reached
        """
        for t, coverage in enumerate(coverages):
            if coverage >= threshold:
                return t
        return None

    @staticmethod
    def fault_tolerance(episodes_baseline: List[Dict],
                        episodes_failure: List[Dict]) -> float:
        """Compute F_T: fault tolerance metric.

        Measures the relative drop in coverage when agents fail vs. baseline.

        Args:
            episodes_baseline: List of episode data without failures
            episodes_failure: List of episode data with failures

        Returns:
            Fault tolerance score (1.0 = perfect, lower = worse)
        """
        baseline_coverage = np.mean([ep['coverages'][-1] for ep in episodes_baseline])
        failure_coverage = np.mean([ep['coverages'][-1] for ep in episodes_failure])

        if baseline_coverage > 0:
            ft = 1.0 - (baseline_coverage - failure_coverage) / baseline_coverage
        else:
            ft = 0.0

        return max(0.0, ft)

    @staticmethod
    def communication_cost(communications: List[int]) -> int:
        """Compute E_comm: total number of transmit actions.

        Args:
            communications: List of communication counts per timestep

        Returns:
            Total number of communications
        """
        return sum(communications)

    def compute_summary_statistics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        """Compute mean and std for a list of metric dictionaries.

        Returns:
            Dictionary of metric_name -> (mean, std)
        """
        if not metrics_list:
            return {}

        # Collect all metric names
        metric_names = set()
        for m in metrics_list:
            metric_names.update(m.keys())

        summary = {}
        for name in metric_names:
            values = [m[name] for m in metrics_list if name in m]
            if values:
                mean = np.mean(values)
                std = np.std(values)
                summary[name] = (mean, std)

        return summary

    def get_all_metrics(self) -> List[Dict[str, float]]:
        """Get all episode metrics."""
        return [self.compute_episode_metrics(ep) for ep in self.episode_data]

    def aggregate_metrics(self) -> Dict[str, Tuple[float, float]]:
        """Aggregate all episode metrics."""
        all_metrics = self.get_all_metrics()
        return self.compute_summary_statistics(all_metrics)


def compare_policies(metrics_by_policy: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Compare metrics across multiple policies.

    Args:
        metrics_by_policy: Dictionary mapping policy name to list of metrics

    Returns:
        Dictionary mapping policy name to summary statistics
    """
    tracker = MetricsTracker()
    results = {}

    for policy_name, metrics_list in metrics_by_policy.items():
        results[policy_name] = tracker.compute_summary_statistics(metrics_list)

    return results
