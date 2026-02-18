"""
General plotting of episode logs from training runs (agent-agnostic).

Reads CSV(s) from a run folder (csvs/), infers columns dynamically (rewards, losses,
grad_norm, etc.) so it works for SAC, TD3, TD-MPC2, or any agent. Saves plots
into the same run folder (plots/) so outputs stay with the run.
"""

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Standard columns written by train_run_refactored (may be missing in older runs)
STANDARD_COLUMNS = [
    "episode",
    "reward",
    "shaped_reward",
    "backprop_reward",
    "total_gradient_steps",
]


def load_episode_logs_from_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Load episode logs from a CSV file. Column names and count are inferred
    from the header so any agent (SAC, TD3, TD-MPC2, etc.) is supported.
    """
    episode_logs: List[Dict[str, Any]] = []

    if not csv_path.exists():
        return episode_logs

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            log: Dict[str, Any] = {"losses": {}}
            for key in fieldnames:
                value = row.get(key, "").strip()
                if key == "episode":
                    try:
                        log["episode"] = int(value) if value else 0
                    except ValueError:
                        log["episode"] = len(episode_logs)
                elif key in (
                    "reward",
                    "shaped_reward",
                    "backprop_reward",
                    "total_gradient_steps",
                ):
                    try:
                        log[key] = float(value) if value else None
                    except ValueError:
                        log[key] = None
                else:
                    if value:
                        try:
                            log["losses"][key] = float(value)
                        except ValueError:
                            pass
            episode_logs.append(log)

    return episode_logs


def find_all_episode_log_files(csvs_dir: Path, run_name: str) -> List[Path]:
    """Find main episode_logs CSV and any checkpoint episode_logs for this run."""
    log_files: List[Path] = []
    main_file = csvs_dir / f"{run_name}_episode_logs.csv"
    if main_file.exists():
        log_files.append(main_file)
    pattern = f"{run_name}_ep*_episode_logs.csv"
    for p in sorted(csvs_dir.glob(pattern)):
        if p != main_file:
            log_files.append(p)
    return sorted(log_files, key=lambda x: x.name)


def infer_run_name_from_csv_path(csv_path: Path) -> str:
    """Infer run name from filename (e.g. tdmpc2_run_..._episode_logs -> tdmpc2_run_...)."""
    stem = csv_path.stem
    match = re.search(r"_ep\d+_episode_logs$", stem)
    if match:
        return stem[: match.start()]
    if stem.endswith("_episode_logs"):
        return stem[: -len("_episode_logs")]
    return stem


def combine_episode_logs(log_files: List[Path]) -> List[Dict[str, Any]]:
    """Merge episode logs from multiple CSVs (e.g. main + checkpoints), dedupe by episode."""
    by_episode: Dict[int, Dict[str, Any]] = {}
    for path in log_files:
        for log in load_episode_logs_from_csv(path):
            ep = log.get("episode", 0)
            by_episode[ep] = log
    return [by_episode[ep] for ep in sorted(by_episode.keys())]


def _moving_average(data: List[float], window_size: int) -> List[float]:
    if not data or window_size < 1:
        return list(data) if data else []
    out: List[float] = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        window = data[start : i + 1]
        out.append(sum(window) / len(window))
    return out


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


def _collect_metric_series(
    episode_logs: List[Dict[str, Any]],
) -> Tuple[List[int], Dict[str, List[float]]]:
    """Collect episode indices and per-metric value series (only episodes where value exists)."""
    all_metric_keys: set = set()
    for log in episode_logs:
        all_metric_keys.update(log.get("losses", {}).keys())
    sorted_metrics = sorted(all_metric_keys)
    episodes_by_metric: Dict[str, List[int]] = {k: [] for k in sorted_metrics}
    values_by_metric: Dict[str, List[float]] = {k: [] for k in sorted_metrics}
    for log in episode_logs:
        ep = log.get("episode", 0)
        losses = log.get("losses", {})
        for key in sorted_metrics:
            if key in losses and losses[key] is not None:
                episodes_by_metric[key].append(ep)
                values_by_metric[key].append(losses[key])
    return episodes_by_metric, values_by_metric


def _first_complete_episode(
    episode_logs: List[Dict[str, Any]],
    metric_keys: List[str],
) -> Optional[int]:
    """First episode where all given metrics are present (e.g. after warmup)."""
    if not metric_keys:
        return None
    ep_sets = []
    for key in metric_keys:
        eps = [
            log["episode"]
            for log in episode_logs
            if log.get("losses", {}).get(key) is not None
        ]
        if not eps:
            return None
        ep_sets.append(set(eps))
    common = set.intersection(*ep_sets)
    return min(common) if common else None


def plot_episode_logs(
    folder_path: str,
    run_name: Optional[str] = None,
    window_size: int = 500,
    skip_warmup: bool = True,
    plot_flat_losses: bool = False,
) -> Optional[Path]:
    """
    Plot episode logs from a run folder. Columns are discovered from CSV(s);
    works for any agent (SAC, TD3, TD-MPC2, etc.). Saves plot in the run folder.

    Args:
        folder_path: Run directory containing csvs/ and plots/ (e.g. results/tdmpc2_runs/2026-01-25_11-40-04).
        run_name: Run name for file lookup. If None, inferred from first *_episode_logs.csv found.
        window_size: Moving average window for smoothing.
        skip_warmup: If True, trim to first episode where all loss/metric columns are present.
        plot_flat_losses: If True and {run_name}_losses.csv exists, also plot step vs loss there.

    Returns:
        Path to saved episode_logs plot, or None if no data.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    csvs_dir = folder / "csvs"
    plots_dir = folder / "plots"
    if not csvs_dir.exists():
        raise FileNotFoundError(f"No csvs directory in {folder_path}")

    episode_log_files = list(csvs_dir.glob("*_episode_logs.csv"))
    if not episode_log_files:
        raise FileNotFoundError(f"No *_episode_logs.csv files in {csvs_dir}")

    if run_name is None:
        run_name = infer_run_name_from_csv_path(episode_log_files[0])
    log_files = find_all_episode_log_files(csvs_dir, run_name)
    if not log_files:
        raise FileNotFoundError(f"No episode log files found for run {run_name}")

    episode_logs = combine_episode_logs(log_files)
    if not episode_logs:
        return None

    episodes = [log["episode"] for log in episode_logs]
    rewards = [log.get("reward") for log in episode_logs]
    shaped_rewards = [log.get("shaped_reward") for log in episode_logs]
    backprop_rewards = [log.get("backprop_reward") for log in episode_logs]

    episodes_by_metric, values_by_metric = _collect_metric_series(episode_logs)
    all_metric_keys = sorted(values_by_metric.keys())
    opponent_like = [
        k for k in all_metric_keys if "opponent" in k.lower() and "cloning" in k.lower()
    ]
    other_metrics = [k for k in all_metric_keys if k not in opponent_like]
    sorted_metric_keys = other_metrics + opponent_like

    first_complete = None
    if skip_warmup and sorted_metric_keys:
        first_complete = _first_complete_episode(episode_logs, sorted_metric_keys)
    if first_complete is not None:
        keep = [i for i, ep in enumerate(episodes) if ep >= first_complete]
        episodes = [episodes[i] for i in keep]
        rewards = [rewards[i] for i in keep]
        shaped_rewards = [shaped_rewards[i] for i in keep]
        backprop_rewards = [backprop_rewards[i] for i in keep]
        for key in sorted_metric_keys:
            eps_list = episodes_by_metric[key]
            vals_list = values_by_metric[key]
            episodes_by_metric[key] = [e for e in eps_list if e >= first_complete]
            values_by_metric[key] = [
                v for e, v in zip(eps_list, vals_list) if e >= first_complete
            ]

    n_plots = 0
    if any(r is not None for r in rewards):
        n_plots += 2  # reward series + reward distribution histogram
    if any(s is not None for s in shaped_rewards):
        n_plots += 1
    if any(b is not None for b in backprop_rewards):
        n_plots += 1
    n_plots += len(sorted_metric_keys)
    if n_plots == 0:
        return None

    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten().tolist()
    fig.suptitle(run_name, fontsize=14, fontweight="bold", y=0.995)

    ax_idx = 0

    def _safe_float_list(xs, default=0.0):
        return [(x if x is not None else default) for x in xs]

    if any(r is not None for r in rewards):
        r_vals = _safe_float_list(rewards)
        mov = _moving_average(r_vals, window_size)
        axes[ax_idx].plot(episodes, r_vals, alpha=0.3, label="Raw", color="blue")
        axes[ax_idx].plot(
            episodes, mov, label=f"MA (w={window_size})", color="blue", linewidth=2
        )
        _add_rolling_percentile_bands(axes[ax_idx], episodes, r_vals, window=500)
        axes[ax_idx].set_xlabel("Episode")
        axes[ax_idx].set_ylabel("Reward")
        axes[ax_idx].set_title("Reward per Episode")
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1
        _plot_reward_distribution_histogram(axes[ax_idx], r_vals, n_last=1000)
        ax_idx += 1

    if any(s is not None for s in shaped_rewards):
        s_vals = _safe_float_list(shaped_rewards)
        mov = _moving_average(s_vals, window_size)
        axes[ax_idx].plot(episodes, s_vals, alpha=0.3, label="Raw", color="green")
        axes[ax_idx].plot(
            episodes, mov, label=f"MA (w={window_size})", color="green", linewidth=2
        )
        _add_rolling_percentile_bands(axes[ax_idx], episodes, s_vals, window=500)
        axes[ax_idx].set_xlabel("Episode")
        axes[ax_idx].set_ylabel("Shaped Reward")
        axes[ax_idx].set_title("Shaped Reward per Episode")
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1

    if any(b is not None for b in backprop_rewards):
        b_vals = _safe_float_list(backprop_rewards)
        mov = _moving_average(b_vals, window_size)
        axes[ax_idx].plot(episodes, b_vals, alpha=0.3, label="Raw", color="orange")
        axes[ax_idx].plot(
            episodes, mov, label=f"MA (w={window_size})", color="orange", linewidth=2
        )
        _add_rolling_percentile_bands(axes[ax_idx], episodes, b_vals, window=500)
        axes[ax_idx].set_xlabel("Episode")
        axes[ax_idx].set_ylabel("Backprop Reward")
        axes[ax_idx].set_title("Backprop Reward per Episode")
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1

    colors_main = plt.cm.tab10(np.linspace(0, 1, max(1, len(other_metrics))))
    colors_opp = plt.cm.Reds(np.linspace(0.4, 0.9, max(1, len(opponent_like))))
    for i, key in enumerate(sorted_metric_keys):
        if ax_idx >= len(axes):
            break
        vals = values_by_metric.get(key, [])
        eps = episodes_by_metric.get(key, [])
        if not vals:
            ax_idx += 1
            continue
        color = (
            colors_opp[i - len(other_metrics)]
            if key in opponent_like
            else colors_main[i]
        )
        mov = _moving_average(vals, window_size)
        axes[ax_idx].plot(eps, vals, alpha=0.3, label="Raw", color=color)
        axes[ax_idx].plot(
            eps, mov, label=f"MA (w={window_size})", color=color, linewidth=2
        )
        axes[ax_idx].set_xlabel("Episode")
        axes[ax_idx].set_ylabel("Value")
        axes[ax_idx].set_title(f"{key}")
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1

    for idx in range(ax_idx, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / f"{run_name}_episode_logs.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    if plot_flat_losses:
        flat_path = _plot_flat_losses_if_present(
            csvs_dir, plots_dir, run_name, window_size
        )
        if flat_path:
            print(f"Flat losses plot saved to {flat_path}")

    print(f"Episode logs plot saved to {save_path}")
    return save_path


def _plot_flat_losses_if_present(
    csvs_dir: Path,
    plots_dir: Path,
    run_name: str,
    window_size: int,
) -> Optional[Path]:
    """If {run_name}_losses.csv exists (step, loss), plot and save in plots_dir."""
    loss_csv = csvs_dir / f"{run_name}_losses.csv"
    if not loss_csv.exists():
        return None
    steps: List[int] = []
    losses: List[float] = []
    with open(loss_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(int(row.get("step", len(steps))))
                losses.append(float(row.get("loss", 0)))
            except (ValueError, KeyError):
                continue
    if not steps or not losses:
        return None
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    mov = _moving_average(losses, window_size)
    ax.plot(steps, losses, alpha=0.3, label="Raw")
    ax.plot(steps, mov, label=f"MA (w={window_size})", linewidth=2)
    ax.set_xlabel("Gradient Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"{run_name} - Loss per Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = plots_dir / f"{run_name}_losses.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_run_folder(
    folder_path: str,
    run_name: Optional[str] = None,
    window_size: int = 500,
    skip_warmup: bool = True,
    plot_flat_losses: bool = False,
) -> Optional[Path]:
    """
    Convenience wrapper: plot episode logs for a single run folder.
    Same as plot_episode_logs with the same arguments.
    """
    return plot_episode_logs(
        folder_path,
        run_name=run_name,
        window_size=window_size,
        skip_warmup=skip_warmup,
        plot_flat_losses=plot_flat_losses,
    )


def plot_training_metrics(
    training_metrics,
    result_dir: str,
    run_name: str = "training_run",
    window_size: int = 500,
) -> Optional[Path]:
    """
    Plot training metrics from TrainingMetrics dataclass (used in train.py).
    
    Args:
        training_metrics: TrainingMetrics instance with episodes, updates, winrates
        result_dir: Directory to save plots
        run_name: Name for the run (used in plot title and filename)
        window_size: Moving average window for smoothing
    
    Returns:
        Path to saved plot, or None if no data
    """
    if not training_metrics.episodes and not training_metrics.updates:
        return None
    
    plots_dir = Path(result_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract episode data
    episodes = [i for i in range(len(training_metrics.episodes))]
    rewards = [ep["reward"] for ep in training_metrics.episodes]
    ratings = [ep["rating"] for ep in training_metrics.episodes]
    
    # Extract update metrics
    metric_names = set()
    for update in training_metrics.updates:
        metric_names.update(k for k in update.keys() if k != "step")
    metric_names = sorted(metric_names)
    
    # Organize metrics by name; normalize list values (e.g. SAC/TD3 return lists) to scalar per step
    def _to_scalar(v: Any) -> float:
        if isinstance(v, list):
            return float(np.mean(v)) if v else 0.0
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    metrics_by_name = {name: {"steps": [], "values": []} for name in metric_names}
    for update in training_metrics.updates:
        for metric_name in metric_names:
            if metric_name in update:
                metrics_by_name[metric_name]["steps"].append(update["step"])
                metrics_by_name[metric_name]["values"].append(
                    _to_scalar(update[metric_name])
                )
    
    # Separate opponent cloning losses from other metrics
    opponent_metrics = [k for k in metric_names if "opponent" in k.lower() and "cloning" in k.lower()]
    other_metrics = [k for k in metric_names if k not in opponent_metrics]
    
    # Count plots needed
    n_plots = 0
    if rewards:
        n_plots += 2  # reward series + reward distribution
    if ratings:
        n_plots += 1  # rating series
    n_plots += len(other_metrics)  # individual metric plots
    if opponent_metrics:
        n_plots += 1  # combined opponent cloning plot
    
    if n_plots == 0:
        return None
    
    # Create figure
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten().tolist()
    fig.suptitle(run_name, fontsize=14, fontweight="bold", y=0.995)
    
    ax_idx = 0
    
    # Plot rewards
    if rewards:
        mov = _moving_average(rewards, window_size)
        axes[ax_idx].plot(episodes, rewards, alpha=0.3, label="Raw", color="blue")
        axes[ax_idx].plot(
            episodes, mov, label=f"MA (w={window_size})", color="blue", linewidth=2
        )
        _add_rolling_percentile_bands(axes[ax_idx], episodes, rewards, window=min(500, len(rewards)))
        axes[ax_idx].set_xlabel("Episode")
        axes[ax_idx].set_ylabel("Reward")
        axes[ax_idx].set_title("Reward per Episode")
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1
        
        # Reward distribution histogram
        _plot_reward_distribution_histogram(axes[ax_idx], rewards, n_last=1000)
        ax_idx += 1
    
    # Plot ratings
    if ratings:
        mov = _moving_average(ratings, window_size)
        axes[ax_idx].plot(episodes, ratings, alpha=0.3, label="Raw", color="purple")
        axes[ax_idx].plot(
            episodes, mov, label=f"MA (w={window_size})", color="purple", linewidth=2
        )
        axes[ax_idx].set_xlabel("Episode")
        axes[ax_idx].set_ylabel("Rating")
        axes[ax_idx].set_title("Rating over Episodes")
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1
    
    # Plot other metrics
    colors_main = plt.cm.tab10(np.linspace(0, 1, max(1, len(other_metrics))))
    for i, metric_name in enumerate(other_metrics):
        if ax_idx >= len(axes):
            break
        
        steps = metrics_by_name[metric_name]["steps"]
        values = metrics_by_name[metric_name]["values"]
        
        if values:
            mov = _moving_average(values, window_size)
            axes[ax_idx].plot(steps, values, alpha=0.3, label="Raw", color=colors_main[i])
            axes[ax_idx].plot(
                steps, mov, label=f"MA (w={window_size})", color=colors_main[i], linewidth=2
            )
            axes[ax_idx].set_xlabel("Step")
            axes[ax_idx].set_ylabel("Value")
            axes[ax_idx].set_title(f"{metric_name}")
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1
    
    # Plot opponent cloning losses (combined)
    if opponent_metrics and ax_idx < len(axes):
        colors_opp = plt.cm.Reds(np.linspace(0.4, 0.9, max(1, len(opponent_metrics))))
        
        for i, metric_name in enumerate(opponent_metrics):
            steps = metrics_by_name[metric_name]["steps"]
            values = metrics_by_name[metric_name]["values"]
            
            if values:
                mov = _moving_average(values, window_size)
                opponent_label = metric_name.replace("opponent_", "Opponent ").replace("_cloning_loss", "")
                
                axes[ax_idx].plot(steps, values, alpha=0.2, color=colors_opp[i])
                axes[ax_idx].plot(
                    steps, mov, label=opponent_label, color=colors_opp[i], linewidth=2
                )
        
        axes[ax_idx].set_xlabel("Step")
        axes[ax_idx].set_ylabel("Cloning Loss (MSE)")
        axes[ax_idx].set_title("Opponent Cloning Losses")
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1
    
    # Hide unused subplots
    for idx in range(ax_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = plots_dir / f"{run_name}_training_metrics.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Training metrics plot saved to {save_path}")
    return save_path


if __name__ == "__main__":
    folder_path = "results/tdmpc2_runs_test/2026-01-31_19-19-57"
    run_name = None
    window_size = 80
    plot_episode_logs(
        folder_path,
        run_name=run_name,
        window_size=window_size,
        plot_flat_losses=True,
    )
