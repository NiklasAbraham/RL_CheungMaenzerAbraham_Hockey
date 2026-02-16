"""
Plot TD-MPC2 episode logs from a single CSV for the report.
Uses tueplots for publication-ready styling. Set EPISODE_LOG_CSV below.
"""

import csv

import matplotlib

matplotlib.use("Agg")
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Path to the episode log CSV (edit to your run's csvs folder and file)
EPISODE_LOG_CSV = "results/tdmpc2_runs_horizon/2026-02-13_13-49-57/csvs/TDMPC2_run_lr3e04_bs512_hencoder_dynamics_reward_termination_q_function_policy_3b5198ec_20260213_134957_episode_logs.csv"

OUTPUT_DIR = Path(__file__).resolve().parent
WINDOW_SIZE = 250


try:
    from tueplots import axes, bundles, figsizes

    plt.rcParams.update(
        bundles.neurips2024(usetex=False, rel_width=1.0, family="sans-serif")
    )
    plt.rcParams.update(axes.grid(grid_alpha=0.3))
    plt.rcParams.update(axes.spines(right=False, top=False))
    plt.rcParams["figure.constrained_layout.use"] = False
    FIG_SINGLE = figsizes.neurips2024(nrows=1, ncols=1)["figure.figsize"]
    FIG_TWO = figsizes.neurips2024(nrows=2, ncols=1)["figure.figsize"]
    USE_TUEPLOTS = True
except ImportError:
    USE_TUEPLOTS = False
    FIG_SINGLE = (6, 4)
    FIG_TWO = (6, 8)


def load_episode_logs_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Load episode logs from a single CSV file."""
    episode_logs = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episode_log = {
                "episode": int(row["episode"]),
                "reward": float(row["reward"]) if row.get("reward") else 0.0,
                "shaped_reward": float(row["shaped_reward"])
                if row.get("shaped_reward")
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


def _moving_average(data: list[float], window_size: int) -> list[float]:
    if not data:
        return []
    out = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        w = data[start : i + 1]
        out.append(sum(w) / len(w))
    return out


def _rolling_percentiles(
    values: list[float], window: int, upper_p: float
) -> tuple[list[float], list[float]]:
    if not values or window < 1:
        return [], []
    lower_p = 100.0 - upper_p
    upper_line = []
    lower_line = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        w = values[start : i + 1]
        upper_line.append(float(np.percentile(w, upper_p)))
        lower_line.append(float(np.percentile(w, lower_p)))
    return upper_line, lower_line


def _add_rolling_percentile_bands(
    ax: plt.Axes,
    episodes: list[int],
    values: list[float],
    window: int = 500,
) -> None:
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
    ax: plt.Axes, values: list[float], n_last: int = 1000
) -> None:
    if not values:
        return
    n = min(n_last, len(values))
    data = np.array(values[-n:], dtype=float)
    mean = float(np.mean(data))
    std = float(np.std(data))
    p10 = float(np.percentile(data, 10))
    p50 = float(np.percentile(data, 50))
    p90 = float(np.percentile(data, 90))
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
        label=f"Mean - sigma = {mean - std:.2f}",
    )
    ax.axvline(
        mean + std,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean + sigma = {mean + std:.2f}",
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
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)


def _filter_from_first_complete_episode(
    episode_logs: list[dict],
    sorted_loss_keys: list[str],
) -> tuple[
    list[int], list[float], list[float], dict[str, list[float]], dict[str, list[int]]
]:
    """Filter rewards and losses to start from first episode where all loss types are present."""
    episodes = [log["episode"] for log in episode_logs]
    rewards = [log["reward"] for log in episode_logs]
    shaped_rewards = [log["shaped_reward"] for log in episode_logs]
    loss_data = {k: [] for k in sorted_loss_keys}
    loss_episodes = {k: [] for k in sorted_loss_keys}
    for log in episode_logs:
        for k in sorted_loss_keys:
            if k in log["losses"]:
                loss_data[k].append(log["losses"][k])
                loss_episodes[k].append(log["episode"])

    first_complete = None
    if sorted_loss_keys and all(loss_episodes[k] for k in sorted_loss_keys):
        common = set(loss_episodes[sorted_loss_keys[0]])
        for k in sorted_loss_keys[1:]:
            common &= set(loss_episodes[k])
        if common:
            first_complete = min(common)

    if first_complete is None:
        return episodes, rewards, shaped_rewards, loss_data, loss_episodes

    filtered_episodes = [e for e in episodes if e >= first_complete]
    filtered_rewards = [r for e, r in zip(episodes, rewards) if e >= first_complete]
    filtered_shaped = [
        s for e, s in zip(episodes, shaped_rewards) if e >= first_complete
    ]
    filtered_loss_data = {}
    filtered_loss_episodes = {}
    for k in sorted_loss_keys:
        filtered_loss_data[k] = [
            v for e, v in zip(loss_episodes[k], loss_data[k]) if e >= first_complete
        ]
        filtered_loss_episodes[k] = [e for e in loss_episodes[k] if e >= first_complete]
    return (
        filtered_episodes,
        filtered_rewards,
        filtered_shaped,
        filtered_loss_data,
        filtered_loss_episodes,
    )


def plot_episode_logs(
    csv_path: Path = EPISODE_LOG_CSV,
    output_dir: Path = OUTPUT_DIR,
    window_size: int = WINDOW_SIZE,
    save_name: str = "episode_logs",
) -> None:
    """Plot reward, shaped reward, reward distribution, and losses; save to output_dir."""
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    episode_logs = load_episode_logs_from_csv(csv_path)
    if not episode_logs:
        raise FileNotFoundError(f"No rows in {csv_path}")

    all_loss_keys = set()
    for log in episode_logs:
        all_loss_keys.update(log["losses"].keys())
    opponent_loss_keys = sorted(
        k for k in all_loss_keys if "opponent" in k.lower() and "cloning" in k.lower()
    )
    skip_loss_keys = {"length", "step", "rating", "loss"}
    other_loss_keys = sorted(
        k
        for k in all_loss_keys
        if k not in opponent_loss_keys and k not in skip_loss_keys
    )
    grad_norm_keys = sorted(k for k in other_loss_keys if k.startswith("grad_norm_"))
    other_non_grad = [k for k in other_loss_keys if k not in grad_norm_keys]
    loss_keys = sorted(k for k in other_non_grad if k.endswith("_loss"))
    other_rest = [k for k in other_non_grad if k not in loss_keys]
    n_grad_panels = (len(grad_norm_keys) + 1) // 2
    n_loss_panels = (len(loss_keys) + 1) // 2
    sorted_loss_keys = other_loss_keys + opponent_loss_keys

    episodes, rewards, shaped_rewards, loss_data, loss_episodes = (
        _filter_from_first_complete_episode(episode_logs, sorted_loss_keys)
    )

    n_plots = (
        2
        + len(other_rest)
        + n_loss_panels
        + n_grad_panels
        + (1 if opponent_loss_keys else 0)
    )
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols
    if USE_TUEPLOTS:
        base = figsizes.neurips2024(nrows=n_rows, ncols=n_cols)["figure.figsize"]
        figsize = (base[0] * 1.5, base[1] * 1.35)
    else:
        figsize = (18, 5.5 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = list(np.atleast_1d(axes).flatten())

    fig.suptitle(
        "TD-MPC2 agent with the horizon 4 \n without the opponent aware dynamics",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    ax_idx = 0
    moving_avg = _moving_average(rewards, window_size)
    axes[ax_idx].plot(episodes, rewards, alpha=0.3, label="Raw", color="blue")
    axes[ax_idx].plot(
        episodes,
        moving_avg,
        label=f"Moving Avg (w={window_size})",
        color="blue",
        linewidth=2,
    )
    _add_rolling_percentile_bands(axes[ax_idx], episodes, rewards, window=window_size)
    axes[ax_idx].set_xlabel("Episode")
    axes[ax_idx].set_ylabel("Reward")
    axes[ax_idx].set_title("Reward per Episode")
    axes[ax_idx].legend()
    axes[ax_idx].grid(True, alpha=0.3)

    ax_idx += 1
    _plot_reward_distribution_histogram(axes[ax_idx], rewards, n_last=1000)

    n_other_panels = len(other_rest) + n_loss_panels + n_grad_panels
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, n_other_panels)))
    opponent_colors = plt.cm.Reds(
        np.linspace(0.4, 0.9, max(1, len(opponent_loss_keys)))
    )

    for i, loss_key in enumerate(other_rest):
        ax_idx += 1
        if ax_idx >= len(axes):
            break
        loss_vals = loss_data[loss_key]
        loss_eps = loss_episodes[loss_key]
        if loss_vals:
            mov = _moving_average(loss_vals, window_size)
            axes[ax_idx].plot(
                loss_eps, loss_vals, alpha=0.3, label="Raw", color=colors[i]
            )
            axes[ax_idx].plot(
                loss_eps,
                mov,
                label=f"Moving Avg (w={window_size})",
                color=colors[i],
                linewidth=2,
            )
        axes[ax_idx].set_xlabel("Episode")
        axes[ax_idx].set_ylabel("Loss")
        axes[ax_idx].set_title(f"{loss_key} per Episode")
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)

    for p in range(n_loss_panels):
        ax_idx += 1
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        pair = loss_keys[2 * p : 2 * p + 2]
        color_idx = len(other_rest) + p
        color_left = colors[color_idx % len(colors)]
        color_right = colors[(color_idx + 1) % max(1, len(colors))]
        for j, loss_key in enumerate(pair):
            loss_vals = loss_data[loss_key]
            loss_eps = loss_episodes[loss_key]
            if not loss_vals:
                continue
            mov = _moving_average(loss_vals, window_size)
            label = loss_key.replace("_loss", "")
            if j == 0:
                ax.plot(loss_eps, loss_vals, alpha=0.3, label=label, color=color_left)
                ax.plot(
                    loss_eps, mov, label=f"{label} (avg)", color=color_left, linewidth=2
                )
                ax.set_ylabel(label, color=color_left)
                ax.tick_params(axis="y", labelcolor=color_left)
            else:
                ax2 = ax.twinx()
                ax2.plot(loss_eps, loss_vals, alpha=0.3, label=label, color=color_right)
                ax2.plot(
                    loss_eps,
                    mov,
                    label=f"{label} (avg)",
                    color=color_right,
                    linewidth=2,
                )
                ax2.set_ylabel(label, color=color_right)
                ax2.tick_params(axis="y", labelcolor=color_right)
                ax2.legend(loc="upper right")
        ax.legend(loc="upper left")
        ax.set_xlabel("Episode")
        ax.set_title("Loss: " + " / ".join(k.replace("_loss", "") for k in pair))
        ax.grid(True, alpha=0.3)

    for p in range(n_grad_panels):
        ax_idx += 1
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        pair = grad_norm_keys[2 * p : 2 * p + 2]
        color_idx = len(other_rest) + n_loss_panels + p
        color_left = colors[color_idx % len(colors)]
        color_right = colors[(color_idx + 1) % max(1, len(colors))]
        for j, loss_key in enumerate(pair):
            loss_vals = loss_data[loss_key]
            loss_eps = loss_episodes[loss_key]
            if not loss_vals:
                continue
            mov = _moving_average(loss_vals, window_size)
            label = loss_key.replace("grad_norm_", "")
            if j == 0:
                ax.plot(loss_eps, loss_vals, alpha=0.3, label=label, color=color_left)
                ax.plot(
                    loss_eps, mov, label=f"{label} (avg)", color=color_left, linewidth=2
                )
                ax.set_ylabel(label, color=color_left)
                ax.tick_params(axis="y", labelcolor=color_left)
            else:
                ax2 = ax.twinx()
                ax2.plot(loss_eps, loss_vals, alpha=0.3, label=label, color=color_right)
                ax2.plot(
                    loss_eps,
                    mov,
                    label=f"{label} (avg)",
                    color=color_right,
                    linewidth=2,
                )
                ax2.set_ylabel(label, color=color_right)
                ax2.tick_params(axis="y", labelcolor=color_right)
                ax2.legend(loc="upper right")
        ax.legend(loc="upper left")
        ax.set_xlabel("Episode")
        ax.set_title(
            "Grad Norm: " + " / ".join(k.replace("grad_norm_", "") for k in pair)
        )
        ax.grid(True, alpha=0.3)

    if opponent_loss_keys and ax_idx + 1 <= len(axes):
        ax_idx += 1
        for i, loss_key in enumerate(opponent_loss_keys):
            loss_vals = loss_data[loss_key]
            loss_eps = loss_episodes[loss_key]
            if loss_vals:
                mov = _moving_average(loss_vals, window_size)
                label = loss_key.replace("opponent_", "Opponent ").replace(
                    "_cloning_loss", ""
                )
                axes[ax_idx].plot(
                    loss_eps, loss_vals, alpha=0.2, color=opponent_colors[i]
                )
                axes[ax_idx].plot(
                    loss_eps, mov, label=label, color=opponent_colors[i], linewidth=2
                )
        axes[ax_idx].set_xlabel("Episode")
        axes[ax_idx].set_ylabel("Cloning Loss (MSE)")
        axes[ax_idx].set_title("Opponent Cloning Losses per Episode")
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)

    for idx in range(ax_idx + 1, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{save_name}.png"
    plt.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"Episode logs plot saved to: {out_path}")


if __name__ == "__main__":
    plot_episode_logs(
        csv_path=EPISODE_LOG_CSV,
        output_dir=OUTPUT_DIR,
        window_size=WINDOW_SIZE,
    )
