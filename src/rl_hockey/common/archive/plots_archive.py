"""
Archive Performance Visualization

This script generates various plots to visualize agent performance from the archive data.
Uses tueplots (https://github.com/pnkraemer/tueplots) for publication-ready styling.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    from tueplots import axes, bundles, figsizes

    plt.rcParams.update(
        bundles.neurips2024(usetex=False, rel_width=1.0, family="sans-serif")
    )
    plt.rcParams.update(axes.grid(grid_alpha=0.3))
    plt.rcParams.update(axes.spines(right=False, top=False))
    plt.rcParams["figure.constrained_layout.use"] = False
    FIG_BAR = figsizes.neurips2024(nrows=1, ncols=1)["figure.figsize"]
    FIG_HEATMAP = figsizes.neurips2024(nrows=1.5, ncols=1.5)["figure.figsize"]
    FIG_SCATTER = figsizes.neurips2024(nrows=1, ncols=1)["figure.figsize"]
    FIG_TABLE = figsizes.neurips2024(nrows=2, ncols=1.5)["figure.figsize"]
    USE_TUEPLOTS = True
except ImportError:
    USE_TUEPLOTS = False
    FIG_BAR = (12, 6)
    FIG_HEATMAP = (12, 10)
    FIG_SCATTER = (10, 6)
    FIG_TABLE = (14, 6)

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_data(archive_root: Path = Path("archive")):
    """Load registry and match history data."""
    registry_path = archive_root / "registry.json"
    match_history_path = archive_root / "match_history.json"

    with open(registry_path, "r") as f:
        registry = json.load(f)

    with open(match_history_path, "r") as f:
        match_history = json.load(f)

    return registry, match_history


def plot_agent_ratings(registry: dict, output_path: Path = Path("plots")):
    """Create a bar plot of agent ratings with uncertainty."""
    output_path.mkdir(exist_ok=True)

    # Extract data
    agents = []
    ratings = []
    sigmas = []
    matches = []

    for agent_id, data in registry.items():
        agents.append(
            agent_id.replace("_", "\n", 1) if len(agent_id) > 15 else agent_id
        )
        ratings.append(data["rating"]["rating"])
        sigmas.append(data["rating"]["sigma"] * 3)  # 3-sigma for confidence
        matches.append(data["rating"]["matches_played"])

    # Sort by rating
    sorted_indices = np.argsort(ratings)[::-1]
    agents = [agents[i] for i in sorted_indices]
    ratings = [ratings[i] for i in sorted_indices]
    sigmas = [sigmas[i] for i in sorted_indices]
    matches = [matches[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=FIG_BAR)
    colors = [
        "#1f77b4"
        if "basic" in str(agents[i]).lower()
        else "#ff7f0e"
        if "SAC" in str(agents[i])
        else "#2ca02c"
        for i in range(len(agents))
    ]

    bars = ax.bar(
        range(len(agents)),
        ratings,
        yerr=sigmas,
        capsize=5,
        color=colors,
        alpha=0.7,
        edgecolor="black",
    )

    # Add match count on top of bars
    for i, (bar, match_count) in enumerate(zip(bars, matches)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + sigmas[i] + 1,
            f"{match_count}m",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlabel("Agent ID", fontsize=12, fontweight="bold")
    ax.set_ylabel("TrueSkill Rating", fontsize=12, fontweight="bold")
    ax.set_title(
        "Agent Performance Ratings (with 3-sigma confidence intervals)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=25, color="red", linestyle="--", alpha=0.5, label="Initial Rating")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / "agent_ratings.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_win_matrix(
    registry: dict, match_history: list, output_path: Path = Path("plots")
):
    """Create a win/loss matrix between agents."""
    output_path.mkdir(exist_ok=True)

    agent_ids = list(registry.keys())
    n_agents = len(agent_ids)

    # Initialize matrices
    win_matrix = np.zeros((n_agents, n_agents))
    match_count = np.zeros((n_agents, n_agents))

    # Create ID to index mapping
    id_to_idx = {agent_id: i for i, agent_id in enumerate(agent_ids)}

    # Process match history
    for match in match_history:
        agent1_id = match.get("agent1_id")
        agent2_id = match.get("agent2_id")
        result = match.get("result")

        # Skip if agent IDs are not in registry (could be ratings or other data)
        if (
            agent1_id not in id_to_idx
            or not isinstance(agent2_id, str)
            or agent2_id not in id_to_idx
        ):
            continue

        idx1 = id_to_idx[agent1_id]
        idx2 = id_to_idx[agent2_id]

        match_count[idx1, idx2] += 1

        if result == 1:  # agent1 wins
            win_matrix[idx1, idx2] += 1
        elif result == -1:  # agent2 wins
            win_matrix[idx2, idx1] += 1
        # result == 0 is a draw, no wins added

    # Calculate win percentages
    win_percentage = np.zeros_like(win_matrix)
    for i in range(n_agents):
        for j in range(n_agents):
            if match_count[i, j] > 0:
                win_percentage[i, j] = (win_matrix[i, j] / match_count[i, j]) * 100

    # Create shorter labels
    short_labels = [
        aid.replace("_2026", "").replace("TDMPC2", "TD")[:20] for aid in agent_ids
    ]

    fig, ax = plt.subplots(figsize=FIG_HEATMAP)
    sns.heatmap(
        win_percentage,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        xticklabels=short_labels,
        yticklabels=short_labels,
        cbar_kws={"label": "Win %"},
        vmin=0,
        vmax=100,
        ax=ax,
    )

    ax.set_title(
        "Win Percentage Matrix (Row vs Column)", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Opponent", fontsize=12, fontweight="bold")
    ax.set_ylabel("Agent", fontsize=12, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    if not USE_TUEPLOTS:
        plt.tight_layout()
    else:
        plt.subplots_adjust(left=0.15, bottom=0.2, right=0.92, top=0.92)
    plt.savefig(output_path / "win_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_rating_uncertainty(registry: dict, output_path: Path = Path("plots")):
    """Scatter plot of rating vs uncertainty (sigma) with match count."""
    output_path.mkdir(exist_ok=True)

    agents = []
    ratings = []
    sigmas = []
    matches = []
    agent_types = []

    for agent_id, data in registry.items():
        agents.append(agent_id)
        ratings.append(data["rating"]["rating"])
        sigmas.append(data["rating"]["sigma"])
        matches.append(data["rating"]["matches_played"])

        if "basic" in agent_id.lower():
            agent_types.append("baseline")
        elif "SAC" in agent_id:
            agent_types.append("SAC")
        elif "TDMPC2" in agent_id or "TD" in agent_id:
            agent_types.append("TDMPC2")
        else:
            agent_types.append("other")

    fig, ax = plt.subplots(figsize=FIG_SCATTER)

    # Create color map
    type_colors = {
        "baseline": "#1f77b4",
        "SAC": "#ff7f0e",
        "TDMPC2": "#2ca02c",
        "other": "#9467bd",
    }
    colors = [type_colors[t] for t in agent_types]

    scatter = ax.scatter(
        ratings,
        sigmas,
        s=[m * 3 for m in matches],
        c=colors,
        alpha=0.6,
        edgecolors="black",
    )

    # Add labels for each point
    for i, agent in enumerate(agents):
        label = agent.replace("_2026", "").replace("TDMPC2", "TD")[:15]
        ax.annotate(
            label,
            (ratings[i], sigmas[i]),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("TrueSkill Rating", fontsize=12, fontweight="bold")
    ax.set_ylabel("Uncertainty (Sigma)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Rating vs Uncertainty (bubble size = matches played)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Create legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color, label=label) for label, color in type_colors.items()
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path / "rating_uncertainty.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_matches_played(registry: dict, output_path: Path = Path("plots")):
    """Bar plot showing matches played per agent."""
    output_path.mkdir(exist_ok=True)

    agents = []
    matches = []

    for agent_id, data in registry.items():
        agents.append(agent_id.replace("_2026", "\n")[:25])
        matches.append(data["rating"]["matches_played"])

    # Sort by matches
    sorted_indices = np.argsort(matches)[::-1]
    agents = [agents[i] for i in sorted_indices]
    matches = [matches[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=FIG_SCATTER)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(agents)))
    bars = ax.bar(
        range(len(agents)), matches, color=colors, edgecolor="black", alpha=0.8
    )

    # Add values on bars
    for bar, match_count in zip(bars, matches):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2,
            f"{match_count}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Agent ID", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Matches Played", fontsize=12, fontweight="bold")
    ax.set_title(
        "Calibration Progress: Matches Played per Agent", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "matches_played.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_summary_table(registry: dict, output_path: Path = Path("plots")):
    """Create a summary table of all agents."""
    output_path.mkdir(exist_ok=True)

    data = []
    for agent_id, info in registry.items():
        data.append(
            {
                "Agent ID": agent_id,
                "Rating": f"{info['rating']['rating']:.2f}",
                "Mu": f"{info['rating']['mu']:.2f}",
                "Sigma": f"{info['rating']['sigma']:.2f}",
                "Matches": info["rating"]["matches_played"],
                "Step": info["step"] if info["step"] else "N/A",
                "Archived": info["archived_at"][:10],
            }
        )

    df = pd.DataFrame(data)
    df = df.sort_values("Rating", ascending=False)

    # Save as CSV
    df.to_csv(output_path / "agent_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(FIG_TABLE[0], max(FIG_TABLE[1], len(df) * 0.4)))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Color rows alternately
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")

    plt.title("Agent Performance Summary Table", fontsize=16, fontweight="bold", pad=20)
    plt.savefig(output_path / "agent_summary_table.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_all_plots(
    archive_root: Path = Path("archive"), output_path: Path = Path("plots")
):
    """Generate all visualization plots."""
    registry, match_history = load_data(archive_root)

    plot_agent_ratings(registry, output_path)
    plot_win_matrix(registry, match_history, output_path)
    plot_rating_uncertainty(registry, output_path)
    plot_matches_played(registry, output_path)
    create_summary_table(registry, output_path)

    print(f"Plots saved to {output_path.absolute()}")


if __name__ == "__main__":
    # Run from project root
    archive_root = Path(__file__).parent.parent.parent.parent.parent / "archive"
    output_path = Path(__file__).parent.parent.parent.parent.parent / "plots"

    generate_all_plots(archive_root, output_path)
