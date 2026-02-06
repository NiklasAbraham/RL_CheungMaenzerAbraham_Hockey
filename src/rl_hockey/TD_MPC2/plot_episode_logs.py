"""Plot TD-MPC2 episode logs from CSV (including checkpoints)."""

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
                "backprop_reward": float(row["backprop_reward"])
                if row.get("backprop_reward")
                else 0.0,
                "losses": {},
            }
            for key, value in row.items():
                if (
                    key
                    not in [
                        "episode",
                        "reward",
                        "shaped_reward",
                        "backprop_reward",
                        "total_gradient_steps",
                    ]
                    and value
                ):
                    try:
                        episode_log["losses"][key] = float(value)
                    except ValueError:
                        pass

            episode_logs.append(episode_log)

    return episode_logs


def find_all_episode_log_files(csvs_dir: Path, run_name: str) -> List[Path]:
    """Find all episode log CSV files for a run (including checkpoints)."""
    log_files = []
    main_file = csvs_dir / f"{run_name}_episode_logs.csv"
    if main_file.exists():
        log_files.append(main_file)
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
            all_logs[episode_num] = log
    sorted_logs = [all_logs[ep] for ep in sorted(all_logs.keys())]
    return sorted_logs


def plot_episode_logs(
    folder_path: str, window_size: int = 500, save_path: Optional[Path] = None
):
    """Plot episode logs (rewards, shaped reward, losses)."""
    folder = Path(folder_path)

    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")

    csvs_dir = folder / "csvs"
    plots_dir = folder / "plots"

    if not csvs_dir.exists():
        raise ValueError(f"Could not find csvs directory in {folder_path}")

    episode_log_files = list(csvs_dir.glob("*_episode_logs.csv"))

    if not episode_log_files:
        raise ValueError(f"No episode log files found in {csvs_dir}")

    first_file = episode_log_files[0]
    filename = first_file.stem
    match = re.search(r"_ep\d+_episode_logs$", filename)
    if match:
        run_name = filename[: match.start()]
    else:
        run_name = filename.replace("_episode_logs", "")
    log_files = find_all_episode_log_files(csvs_dir, run_name)

    if not log_files:
        print(f"Warning: No episode log files found for run {run_name}")
        return

    episode_logs = combine_episode_logs(log_files)

    if not episode_logs:
        print(f"Warning: No episode logs loaded for run {run_name}")
        return

    episodes = [log["episode"] for log in episode_logs]
    rewards = [log["reward"] for log in episode_logs]
    shaped_rewards = [log["shaped_reward"] for log in episode_logs]

    all_loss_keys = set()
    for log in episode_logs:
        all_loss_keys.update(log["losses"].keys())

    opponent_loss_keys = sorted(
        [k for k in all_loss_keys if "opponent" in k.lower() and "cloning" in k.lower()]
    )
    other_loss_keys = sorted([k for k in all_loss_keys if k not in opponent_loss_keys])
    sorted_loss_keys = other_loss_keys + opponent_loss_keys
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
        # Only rewards + reward distribution histogram
        n_plots = 3  # rewards, shaped_rewards, reward distribution
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = list(axes.flatten()) if isinstance(axes, np.ndarray) else [axes]
        plot_rewards_only = False
    else:
        # rewards, shaped_rewards, reward distribution histogram, and all losses
        n_plots = 3 + num_losses
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
    _add_rolling_percentile_bands(axes[ax_idx], episodes, rewards, window=500)
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
        _add_rolling_percentile_bands(
            axes[ax_idx], episodes, shaped_rewards, window=500
        )
        axes[ax_idx].set_xlabel("Episode")
        axes[ax_idx].set_ylabel("Shaped Reward")
        axes[ax_idx].set_title("Shaped Reward per Episode")
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)

    ax_idx += 1
    _plot_reward_distribution_histogram(axes[ax_idx], rewards, n_last=1000)
    # ax_idx stays so next plot (first loss or hide) uses ax_idx + 1 via loop increment

    # Plot each loss type
    colors = plt.cm.tab10(np.linspace(0, 1, len(other_loss_keys)))
    opponent_colors = plt.cm.Reds(
        np.linspace(0.4, 0.9, max(1, len(opponent_loss_keys)))
    )

    # Plot other losses
    for i, loss_key in enumerate(other_loss_keys):
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

    # Plot opponent cloning losses (combine in one plot if multiple opponents)
    if opponent_loss_keys:
        if not plot_rewards_only:
            ax_idx += 1
        if ax_idx < len(axes):
            for i, loss_key in enumerate(opponent_loss_keys):
                loss_values = loss_data[loss_key]
                loss_eps = loss_episodes[loss_key]

                if loss_values:
                    moving_avg_losses = _moving_average(loss_values, window_size)
                    # Extract opponent ID from loss key (e.g., "opponent_0_cloning_loss" -> "Opponent 0")
                    opponent_label = loss_key.replace("opponent_", "Opponent ").replace(
                        "_cloning_loss", ""
                    )

                    axes[ax_idx].plot(
                        loss_eps, loss_values, alpha=0.2, color=opponent_colors[i]
                    )
                    axes[ax_idx].plot(
                        loss_eps,
                        moving_avg_losses,
                        label=opponent_label,
                        color=opponent_colors[i],
                        linewidth=2,
                    )

            axes[ax_idx].set_xlabel("Episode")
            axes[ax_idx].set_ylabel("Cloning Loss (MSE)")
            axes[ax_idx].set_title("Opponent Cloning Losses per Episode")
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


def _rolling_percentiles(
    values: List[float], window: int, upper_p: float
) -> Tuple[List[float], List[float]]:
    """For each index i, compute upper and lower percentile over values[i-window+1:i+1]. Lower = 100 - upper."""
    if not values or window < 1:
        return [], []
    lower_p = 100.0 - upper_p
    upper_line: List[float] = []
    lower_line: List[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        w = values[start : i + 1]
        upper_line.append(float(np.percentile(w, upper_p)))
        lower_line.append(float(np.percentile(w, lower_p)))
    return upper_line, lower_line


def _add_rolling_percentile_bands(
    ax: plt.Axes,
    episodes: List[int],
    values: List[float],
    window: int = 500,
) -> None:
    """Draw rolling 90/10 percentile band over the full episode range."""
    if not episodes or not values or len(episodes) != len(values):
        return
    window = min(window, len(values))
    u90, l10 = _rolling_percentiles(values, window, 90.0)
    ax.plot(
        episodes,
        u90,
        color="gray",
        linestyle="--",
        linewidth=2.2,
        alpha=0.95,
        label=f"90/10 %ile (w={window})",
    )
    ax.plot(episodes, l10, color="gray", linestyle="--", linewidth=2.2, alpha=0.95)


def _plot_reward_distribution_histogram(
    ax: plt.Axes,
    values: List[float],
    n_last: int = 1000,
) -> None:
    """Histogram of reward distribution over the last n_last episodes (or all if fewer). Show mean, std, 10/50/90 percentiles."""
    if not values:
        return
    n = min(n_last, len(values))
    data = np.array(values[-n:], dtype=float)
    mean = float(np.mean(data))
    std = float(np.std(data))
    p10, p50, p90 = (
        float(np.percentile(data, 10)),
        float(np.percentile(data, 50)),
        float(np.percentile(data, 90)),
    )
    bins = min(50, max(10, n // 20))
    ax.hist(
        data, bins=bins, color="steelblue", alpha=0.7, edgecolor="black", density=False
    )
    ax.axvline(mean, color="green", linewidth=2, label=f"Mean = {mean:.2f}")
    ax.axvline(
        mean - std,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean \u2212 \u03c3 = {mean - std:.2f}",
    )
    ax.axvline(
        mean + std,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean + \u03c3 = {mean + std:.2f}",
    )
    ax.axvline(
        p10, color="gray", linestyle=":", linewidth=1.5, label=f"10th %ile = {p10:.2f}"
    )
    ax.axvline(p50, color="red", linewidth=1.5, label=f"50th %ile = {p50:.2f}")
    ax.axvline(
        p90, color="gray", linestyle="-.", linewidth=1.5, label=f"90th %ile = {p90:.2f}"
    )
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.set_title(f"Reward distribution (last {n} episodes)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    folder_path_1 = "results/tdmpc2_runs/2026-01-31_11-50-00"  # 94 -> 16
    folder_path_2 = "results/tdmpc2_runs/2026-01-31_11-49-50"  # 93 -> 16
    folder_path_3 = "results/tdmpc2_runs/2026-01-31_11-49-35"  # 92 -> 8
    folder_path_4 = "results/tdmpc2_runs/2026-01-31_12-37-02"  # 103 -> 8 aber opponents

    folder_path_5 = "results/tdmpc2_runs_test/2026-02-01_09-55-22"  # der ganz gute run eigentlich alles

    window_size = 250

    plot_episode_logs(folder_path_1, window_size=window_size)
    plot_episode_logs(folder_path_2, window_size=window_size)
    plot_episode_logs(folder_path_3, window_size=window_size)
    plot_episode_logs(folder_path_4, window_size=window_size)
    plot_episode_logs(folder_path_5, window_size=window_size)
