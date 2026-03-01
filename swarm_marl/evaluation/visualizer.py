"""Visualizer for SAR environment and training curves."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple
import os


class GridVisualizer:
    """Visualizes the SAR grid world."""

    def __init__(self, grid_size: int, figsize: Tuple[int, int] = (10, 10)):
        """Initialize grid visualizer.

        Args:
            grid_size: Size of the grid
            figsize: Figure size
        """
        self.grid_size = grid_size
        self.figsize = figsize

        # Color mapping
        self.colors = {
            'empty': 'white',
            'obstacle': 'gray',
            'unexplored': 'lightyellow',
            'explored': 'lightgreen',
            'explorer': 'blue',
            'reporter': 'red',
            'rendezvous': 'gold',
        }

    def render(self, grid_world, pheromone_field, agents,
               comm_model, save_path: Optional[str] = None,
               title: Optional[str] = None) -> np.ndarray:
        """Render the current state of the environment.

        Args:
            grid_world: GridWorld instance
            pheromone_field: PheromoneField instance
            agents: Dictionary of Agent instances
            comm_model: CommunicationModel instance
            save_path: Path to save figure

        Returns:
            RGB array of the rendered image
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Draw grid cells
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell_type = grid_world.get_cell_type((x, y))

                if cell_type == 1:  # OBSTACLE
                    color = self.colors['obstacle']
                elif cell_type == 2:  # UNEXPLORED
                    color = self.colors['unexplored']
                elif cell_type == 3:  # EXPLORED
                    color = self.colors['explored']
                else:
                    color = self.colors['empty']

                rect = Rectangle((x, y), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)

        # Draw pheromone levels (translucent overlay)
        discovery_alpha = pheromone_field.discovery / pheromone_field.max_level * 0.3
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if discovery_alpha[x, y] > 0:
                    rect = Rectangle((x, y), 1, 1, facecolor='cyan',
                                     alpha=discovery_alpha[x, y], edgecolor='none')
                    ax.add_patch(rect)

        return_alpha = pheromone_field.return_pheromone / pheromone_field.max_level * 0.3
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if return_alpha[x, y] > 0:
                    rect = Rectangle((x, y), 1, 1, facecolor='magenta',
                                     alpha=return_alpha[x, y], edgecolor='none')
                    ax.add_patch(rect)

        # Draw agents
        from ..envs.agent import AgentRole

        for agent_id, agent in agents.items():
            x, y = agent.position

            # Determine color based on role
            if agent.role == AgentRole.EXPLORER:
                color = self.colors['explorer']
                marker = 'o'
            else:
                color = self.colors['reporter']
                marker = 's'

            # Draw agent
            circle = Circle((x + 0.5, y + 0.5), 0.3, facecolor=color,
                           edgecolor='black', linewidth=2)
            ax.add_patch(circle)

            # Add agent ID
            ax.text(x + 0.5, y + 0.5, str(agent.agent_id),
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   color='white')

            # Draw communication links
            if agent.communicated:
                for other_id, other_agent in agents.items():
                    if agent_id != other_id:
                        if comm_model.can_communicate(agent.position, other_agent.position):
                            x1, y1 = agent.position
                            x2, y2 = other_agent.position
                            ax.plot([x1 + 0.5, x2 + 0.5], [y1 + 0.5, y2 + 0.5],
                                   'g-', alpha=0.3, linewidth=1)

        # Configure plot
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=self.colors['explorer'], edgecolor='black', label='Explorer'),
            mpatches.Patch(facecolor=self.colors['reporter'], edgecolor='black', label='Reporter'),
            mpatches.Patch(facecolor=self.colors['obstacle'], edgecolor='black', label='Obstacle'),
            mpatches.Patch(facecolor=self.colors['unexplored'], edgecolor='black', label='Unexplored'),
            mpatches.Patch(facecolor=self.colors['explored'], edgecolor='black', label='Explored'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        if title:
            ax.set_title(title, fontsize=10)

        plt.tight_layout()

        # Get RGB array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')

        plt.close(fig)
        return img


class TrainingVisualizer:
    """Visualizes training curves and results."""

    def __init__(self, save_dir: str = "results"):
        """Initialize training visualizer.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_training_curves(self, history: Dict[str, List[float]],
                              save_path: str = "training_curves.png") -> None:
        """Plot training curves.

        Args:
            history: Dictionary of metric_name -> list of values
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Episode rewards
        if 'episode_reward' in history:
            axes[0, 0].plot(history['episode_reward'])
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].set_title('Episode Reward')
            axes[0, 0].grid(True)

        # Coverage
        if 'coverage' in history:
            axes[0, 1].plot(history['coverage'])
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Coverage')
            axes[0, 1].set_title('Coverage Rate')
            axes[0, 1].grid(True)

        # Loss
        if 'loss' in history:
            axes[1, 0].plot(history['loss'])
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].grid(True)

        # Epsilon
        if 'epsilon' in history:
            axes[1, 1].plot(history['epsilon'])
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Epsilon')
            axes[1, 1].set_title('Exploration Rate')
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_path), dpi=150)
        plt.close()

    def plot_comparison_bars(self, results: Dict[str, Dict[str, Tuple[float, float]]],
                              metrics: List[str],
                              save_path: str = "comparison.png") -> None:
        """Plot comparison bar chart across policies.

        Args:
            results: Dictionary of policy_name -> {metric: (mean, std)}
            metrics: List of metric names to plot
            save_path: Path to save figure
        """
        policies = list(results.keys())
        n_metrics = len(metrics)

        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            means = []
            stds = []

            for policy in policies:
                if metric in results[policy]:
                    mean, std = results[policy][metric]
                    means.append(mean)
                    stds.append(std)
                else:
                    means.append(0)
                    stds.append(0)

            x = np.arange(len(policies))
            axes[idx].bar(x, means, yerr=stds, capsize=5)
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(policies, rotation=45, ha='right')
            axes[idx].set_ylabel(metric)
            axes[idx].set_title(metric.replace('_', ' ').title())
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_path), dpi=150)
        plt.close()

    def plot_degradation_curves(self, denial_levels: List[float],
                                 metrics_by_level: Dict[str, List[float]],
                                 save_path: str = "degradation.png") -> None:
        """Plot communication denial degradation curves.

        Args:
            denial_levels: List of denial percentages
            metrics_by_level: Dictionary of metric_name -> list of values
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        for metric, values in metrics_by_level.items():
            ax.plot(denial_levels, values, marker='o', label=metric)

        ax.set_xlabel('Communication Denial (%)')
        ax.set_ylabel('Metric Value')
        ax.set_title('Performance Degradation under Communication Denial')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_path), dpi=150)
        plt.close()

    def create_episode_gif(self, frames: List[np.ndarray],
                          save_path: str = "episode.gif") -> None:
        """Create GIF from episode frames.

        Args:
            frames: List of RGB arrays
            save_path: Path to save GIF
        """
        try:
            from PIL import Image

            images = [Image.fromarray(frame) for frame in frames]
            images[0].save(
                os.path.join(self.save_dir, save_path),
                save_all=True,
                append_images=images[1:],
                duration=200,  # ms per frame
                loop=0
            )
        except ImportError:
            print("PIL not available, cannot create GIF")
