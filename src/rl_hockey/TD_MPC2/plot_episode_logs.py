"""
Plotting function for TD-MPC2 episode logs.
Reads all episode log CSV files (including checkpoints) and creates comprehensive plots.
"""

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt


def load_episode_logs_from_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load episode logs from a CSV file."""
    episode_logs = []

    if not csv_path.exists():
        return episode_logs

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episode_log = {
                "episode": int(row["episode"]),
                "reward": float(row["reward"]) if row["reward"] else 0.0,
                "shaped_reward": float(row["shaped_reward"])
                if row["shaped_reward"]
                else 0.0,
                "losses": {},
            }

            # Extract all loss columns
            for key, value in row.items():
                if key not in ["episode", "reward", "shaped_reward"] and value:
                    try:
                        episode_log["losses"][key] = float(value)
                    except ValueError:
                        pass

            episode_logs.append(episode_log)

    return episode_logs


def find_all_episode_log_files(csvs_dir: Path, run_name: str) -> List[Path]:
    """Find all episode log CSV files for a run (including checkpoints)."""
    log_files = []

    # Main episode logs file
    main_file = csvs_dir / f"{run_name}_episode_logs.csv"
    if main_file.exists():
        log_files.append(main_file)

    # Checkpoint episode logs files (pattern: {run_name}_ep{episode}_episode_logs.csv)
    pattern = f"{run_name}_ep*_episode_logs.csv"
    checkpoint_files = sorted(csvs_dir.glob(pattern))
    log_files.extend(checkpoint_files)

    return sorted(log_files, key=lambda x: x.name)


def combine_episode_logs(log_files: List[Path]) -> List[Dict[str, Any]]:
    """Combine episode logs from multiple CSV files, removing duplicates."""
    all_logs = {}

    for log_file in log_files:
        logs = load_episode_logs_from_csv(log_file)
        for log in logs:
            episode_num = log["episode"]
            # Keep the latest entry if there are duplicates
            all_logs[episode_num] = log

    # Sort by episode number
    sorted_logs = [all_logs[ep] for ep in sorted(all_logs.keys())]
    return sorted_logs


def plot_episode_logs(
    folder_path: str, window_size: int = 10, save_path: Optional[Path] = None
):
    """
    Plot episode logs including all loss types.

    Args:
        folder_path: Path to the run folder (e.g., "results/tdmpc2_runs/2026-01-18_12-24-23")
        window_size: Window size for moving average
        save_path: Optional custom path for saving plot
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")

    csvs_dir = folder / "csvs"
    plots_dir = folder / "plots"

    if not csvs_dir.exists():
        raise ValueError(f"Could not find csvs directory in {folder_path}")

    # Find all episode log CSV files to determine run name
    episode_log_files = list(csvs_dir.glob("*_episode_logs.csv"))

    if not episode_log_files:
        raise ValueError(f"No episode log files found in {csvs_dir}")

    # Extract run name from the first file (remove _episode_logs.csv or _ep*_episode_logs.csv)
    first_file = episode_log_files[0]
    filename = first_file.stem  # Get filename without extension

    # Remove checkpoint suffix if present (e.g., _ep001500_episode_logs)
    # Pattern: _ep followed by digits, then _episode_logs
    match = re.search(r"_ep\d+_episode_logs$", filename)
    if match:
        # Remove _epXXXXX_episode_logs suffix
        run_name = filename[: match.start()]
    else:
        # Just remove _episode_logs suffix
        run_name = filename.replace("_episode_logs", "")

    # Find all episode log files for this run
    log_files = find_all_episode_log_files(csvs_dir, run_name)

    if not log_files:
        print(f"Warning: No episode log files found for run {run_name}")
        return

    # Combine logs from all files
    episode_logs = combine_episode_logs(log_files)

    if not episode_logs:
        print(f"Warning: No episode logs loaded for run {run_name}")
        return

    # Extract data
    episodes = [log["episode"] for log in episode_logs]
    rewards = [log["reward"] for log in episode_logs]
    shaped_rewards = [log["shaped_reward"] for log in episode_logs]

    # Extract all loss types
    all_loss_keys = set()
    for log in episode_logs:
        all_loss_keys.update(log["losses"].keys())

    sorted_loss_keys = sorted(all_loss_keys)

    # Prepare loss data (only episodes that have training)
    loss_data = {key: [] for key in sorted_loss_keys}
    loss_episodes = {key: [] for key in sorted_loss_keys}

    for log in episode_logs:
        for loss_key in sorted_loss_keys:
            if loss_key in log["losses"]:
                loss_data[loss_key].append(log["losses"][loss_key])
                loss_episodes[loss_key].append(log["episode"])

    # Find the first episode where all losses are present (warm-up period ends)
    first_complete_episode = None
    if sorted_loss_keys:
        # Find episodes that have all loss types present
        all_episodes_sets = [
            set(loss_episodes[key]) for key in sorted_loss_keys if loss_episodes[key]
        ]
        if all_episodes_sets and len(all_episodes_sets) == len(sorted_loss_keys):
            # Find intersection of all episodes (episodes where all losses are present)
            episodes_with_all_losses = set.intersection(*all_episodes_sets)
            if episodes_with_all_losses:
                first_complete_episode = min(episodes_with_all_losses)

        # Filter ALL data (rewards, shaped rewards, and losses) to start from first_complete_episode
        if first_complete_episode is not None:
            # Filter rewards and shaped rewards
            filtered_episodes = []
            filtered_rewards = []
            filtered_shaped_rewards = []
            for ep, rew, sh_rew in zip(episodes, rewards, shaped_rewards):
                if ep >= first_complete_episode:
                    filtered_episodes.append(ep)
                    filtered_rewards.append(rew)
                    filtered_shaped_rewards.append(sh_rew)
            episodes = filtered_episodes
            rewards = filtered_rewards
            shaped_rewards = filtered_shaped_rewards

            # Filter loss data to only include episodes >= first_complete_episode
            filtered_loss_data = {}
            filtered_loss_episodes = {}
            for key in sorted_loss_keys:
                filtered_values = []
                filtered_eps = []
                for ep, val in zip(loss_episodes[key], loss_data[key]):
                    if ep >= first_complete_episode:
                        filtered_values.append(val)
                        filtered_eps.append(ep)
                # Only use filtered data if there are still values after filtering
                if filtered_values:
                    filtered_loss_data[key] = filtered_values
                    filtered_loss_episodes[key] = filtered_eps
                else:
                    # Keep original data if filtering removed everything
                    filtered_loss_data[key] = loss_data[key]
                    filtered_loss_episodes[key] = loss_episodes[key]
            loss_data = filtered_loss_data
            loss_episodes = filtered_loss_episodes

    # Create figure with subplots
    num_losses = len(sorted_loss_keys)
    if num_losses == 0:
        # Only rewards, no losses
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        axes = list(axes.flatten()) if isinstance(axes, np.ndarray) else [axes]
        plot_rewards_only = False
    else:
        # Calculate grid size for subplots: rewards + shaped rewards + all losses
        n_plots = 2 + num_losses  # rewards, shaped_rewards, and all loss types
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = list(axes.flatten()) if isinstance(axes, np.ndarray) else [axes]
        plot_rewards_only = False

    # Set main title for the entire figure
    fig.suptitle(run_name, fontsize=14, fontweight="bold", y=0.995)

    # Plot rewards
    ax_idx = 0
    moving_avg_rewards = _moving_average(rewards, window_size)
    axes[ax_idx].plot(episodes, rewards, alpha=0.3, label="Raw", color="blue")
    axes[ax_idx].plot(
        episodes,
        moving_avg_rewards,
        label=f"Moving Avg (window={window_size})",
        color="blue",
        linewidth=2,
    )
    axes[ax_idx].set_xlabel("Episode")
    axes[ax_idx].set_ylabel("Reward")
    axes[ax_idx].set_title("Reward per Episode")
    axes[ax_idx].legend()
    axes[ax_idx].grid(True, alpha=0.3)

    # Plot shaped rewards
    if not plot_rewards_only:
        ax_idx += 1
        moving_avg_shaped = _moving_average(shaped_rewards, window_size)
        axes[ax_idx].plot(
            episodes, shaped_rewards, alpha=0.3, label="Raw", color="green"
        )
        axes[ax_idx].plot(
            episodes,
            moving_avg_shaped,
            label=f"Moving Avg (window={window_size})",
            color="green",
            linewidth=2,
        )
        axes[ax_idx].set_xlabel("Episode")
        axes[ax_idx].set_ylabel("Shaped Reward")
        axes[ax_idx].set_title("Shaped Reward per Episode")
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)

    # Plot each loss type with reward overlay
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_loss_keys)))
    for i, loss_key in enumerate(sorted_loss_keys):
        if not plot_rewards_only:
            ax_idx += 1
        if ax_idx >= len(axes):
            break

        loss_values = loss_data[loss_key]
        loss_eps = loss_episodes[loss_key]

        if loss_values:
            # Plot losses on primary y-axis
            moving_avg_losses = _moving_average(loss_values, window_size)
            axes[ax_idx].plot(
                loss_eps, loss_values, alpha=0.3, label="Raw", color=colors[i]
            )
            axes[ax_idx].plot(
                loss_eps,
                moving_avg_losses,
                label=f"Moving Avg (window={window_size})",
                color=colors[i],
                linewidth=2,
            )
            axes[ax_idx].set_xlabel("Episode")
            axes[ax_idx].set_ylabel("Loss")
            axes[ax_idx].set_title(f"{loss_key} per Episode")
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(ax_idx + 1, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for main title

    # Save plot
    if save_path is None:
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = plots_dir / f"{run_name}_episode_logs.png"

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Episode logs plot saved to: {save_path}")


def _moving_average(data: List[float], window_size: int) -> List[float]:
    """Calculate moving average of data."""
    if not data:
        return []
    moving_averages = []
    for i in range(len(data)):
        window_start = max(0, i - window_size + 1)
        window = data[window_start : i + 1]
        moving_averages.append(sum(window) / len(window))
    return moving_averages


if __name__ == "__main__":
    # folder_path = "results/tdmpc2_runs/2026-01-19_22-19-45"
    # first one with right buffer
    folder_path_1 = "results/tdmpc2_runs/2026-01-20_16-45-11"
    folder_path_2 = "results/tdmpc2_runs/2026-01-20_17-48-56"  # first with right buffer and init moldes
    window_size = 20

    plot_episode_logs(folder_path_1, window_size=window_size)
    plot_episode_logs(folder_path_2, window_size=window_size)
