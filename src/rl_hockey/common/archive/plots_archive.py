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


def _match_agent(agent_id: str, selector: str) -> bool:
    """True if agent_id is selected by selector (exact or prefix match)."""
    if agent_id == selector:
        return True
    if agent_id.startswith(selector + "_"):
        return True
    return False


def _wrap_label(text: str, max_chars: int = 18) -> str:
    """Insert line breaks so each line is at most max_chars, breaking at spaces."""
    if len(text) <= max_chars:
        return text
    out = []
    rest = text
    while rest:
        if len(rest) <= max_chars:
            out.append(rest)
            break
        chunk = rest[: max_chars + 1]
        last_space = chunk.rfind(" ")
        if last_space == -1:
            last_space = chunk.rfind("_")
        if last_space == -1:
            last_space = max_chars
        out.append(rest[:last_space])
        rest = rest[last_space + 1 :].lstrip()
    return "\n".join(out)


def plot_agent_ratings(
    registry: dict,
    output_path: Path = Path("plots"),
    agent_selection: list[tuple[str, str]] | None = None,
    output_filename: str = "agent_ratings.png",
    show_match_count: bool = True,
):
    """Create a bar plot of agent ratings with uncertainty.

    agent_selection: If given, list of (selector, label) to include only those
        agents and use custom x-tick labels. Selector is exact agent_id or prefix
        (e.g. "0006" or "basic_weak"). Order and labels follow this list.
    output_filename: Output file name (used when saving a selected-agents plot).
    show_match_count: If True, show match count (e.g. "884m") above each bar.
    """
    output_path.mkdir(exist_ok=True)

    if agent_selection is not None:
        # Build list of (agent_id, label) in selection order; each selector picks first match
        ordered = []
        seen = set()
        for selector, label in agent_selection:
            for agent_id, data in registry.items():
                if agent_id not in seen and _match_agent(agent_id, selector):
                    ordered.append((agent_id, label, data))
                    seen.add(agent_id)
                    break
        labels = [t[1] for t in ordered]
        agents = labels
        ratings = [t[2]["rating"]["rating"] for t in ordered]
        sigmas = [t[2]["rating"]["sigma"] * 3 for t in ordered]
        matches = [t[2]["rating"]["matches_played"] for t in ordered]
    else:
        agents = []
        ratings = []
        sigmas = []
        matches = []
        for agent_id, data in registry.items():
            agents.append(
                agent_id.replace("_", "\n", 1) if len(agent_id) > 15 else agent_id
            )
            ratings.append(data["rating"]["rating"])
            sigmas.append(data["rating"]["sigma"] * 3)
            matches.append(data["rating"]["matches_played"])
        sorted_indices = np.argsort(ratings)[::-1]
        agents = [agents[i] for i in sorted_indices]
        ratings = [ratings[i] for i in sorted_indices]
        sigmas = [sigmas[i] for i in sorted_indices]
        matches = [matches[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=FIG_BAR)
    if agent_selection is not None:
        # One distinct color per bar for selected-agents plot
        distinct_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#17becf",
        ]
        colors = [distinct_colors[i % len(distinct_colors)] for i in range(len(agents))]
    else:
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
        width=0.4,
        yerr=sigmas,
        capsize=5,
        color=colors,
        alpha=0.8,
        edgecolor="black",
    )

    if show_match_count:
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

    ax.set_xlabel("Agent ID", fontsize=12)
    ax.set_ylabel("TrueSkill Rating", fontsize=12)
    if agent_selection is not None:
        title = "TrueSkill ratings of selected agents"
        title_pad = 15
    else:
        title = "Agent Performance Ratings (with 3-sigma confidence intervals)"
        title_pad = 6
    ax.set_title(title, fontsize=11, pad=title_pad)
    ax.set_xticks(range(len(agents)))
    if agent_selection is not None:
        display_labels = [_wrap_label(a) for a in agents]
        ax.set_xticklabels(display_labels, rotation=0, ha="center", fontsize=9)
    else:
        ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / output_filename, dpi=300, bbox_inches="tight")
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

    ax.set_title("Win Percentage Matrix (Row vs Column)", fontsize=11)
    ax.set_xlabel("Opponent", fontsize=12)
    ax.set_ylabel("Agent", fontsize=12)
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

    ax.set_xlabel("TrueSkill Rating", fontsize=12)
    ax.set_ylabel("Uncertainty (Sigma)", fontsize=12)
    ax.set_title(
        "Rating vs Uncertainty (bubble size = matches played)",
        fontsize=11,
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
        range(len(agents)),
        matches,
        width=0.55,
        color=colors,
        edgecolor="black",
        alpha=0.8,
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

    ax.set_xlabel("Agent ID", fontsize=12)
    ax.set_ylabel("Number of Matches Played", fontsize=12)
    ax.set_title("Calibration Progress: Matches Played per Agent", fontsize=11)
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "matches_played.png", dpi=300, bbox_inches="tight")
    plt.close()


def _place_label(n: int) -> str:
    """Return ordinal label for position n (1-based): 1st, 2nd, 3rd, 4th, ..."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _collect_horizon_series(
    registry: dict,
    agent_numbers: tuple,
    horizon_start: int,
    horizon_step: int,
):
    """Collect rating and sigma per horizon for one group of agents. Returns (ratings, sigmas, horizon_labels)."""
    ratings = []
    sigmas = []
    horizon_labels = []

    for i, num in enumerate(agent_numbers):
        prefix = f"{num:04d}_"
        agent_id, data = next((p for p in registry.items() if p[0].startswith(prefix)))
        ratings.append(data["rating"]["rating"])
        sigmas.append(data["rating"]["sigma"] * 3)
        horizon_labels.append(f"horizon {horizon_start + i * horizon_step}")

    return ratings, sigmas, horizon_labels


def plot_horizon_ratings(
    registry: dict,
    agent_numbers: tuple = (6, 7, 8, 9, 10),
    agent_numbers_b: tuple | None = (11, 12, 13, 14, 15),
    group_label: str = "no internal opponent modelling",
    group_label_b: str = "internal opponent modelling",
    horizon_start: int = 4,
    horizon_step: int = 2,
    output_path: Path = Path("plots"),
    use_place_labels: bool = False,
):
    """Bar plot of rating (mean and deviation) by horizon, optionally grouped by two agent series.

    agent_numbers: first set of archive agent numbers (e.g. 006-010).
    agent_numbers_b: second set (e.g. 011-015). If None, only the first set is plotted.
    group_label, group_label_b: legend labels for the two series.
    horizon_start, horizon_step: horizon labels (e.g. 4, 6, 8, 10, 12).
    use_place_labels: if True, 1st, 2nd, 3rd, ... are shown on top of each horizon bar.
    """
    output_path.mkdir(exist_ok=True)

    ratings_a, sigmas_a, labels_a = _collect_horizon_series(
        registry, agent_numbers, horizon_start, horizon_step
    )

    # Reference bars: weak and strong bot from registry
    ref_labels = []
    ref_ratings = []
    ref_sigmas = []
    if "basic_weak" in registry:
        ref_labels.append("weak bot")
        d = registry["basic_weak"]["rating"]
        ref_ratings.append(d["rating"])
        ref_sigmas.append(d["sigma"] * 3)
    if "basic_strong" in registry:
        ref_labels.append("strong bot")
        d = registry["basic_strong"]["rating"]
        ref_ratings.append(d["rating"])
        ref_sigmas.append(d["sigma"] * 3)

    n_ref = len(ref_labels)

    if agent_numbers_b is not None:
        ratings_b, sigmas_b, labels_b = _collect_horizon_series(
            registry, agent_numbers_b, horizon_start, horizon_step
        )
        n_horizons = len(labels_a)
        horizon_labels = list(labels_a) + ref_labels
        n_cats = n_horizons + n_ref
        x = np.arange(n_cats)
        bar_width = 0.24
        fig, ax = plt.subplots(figsize=(max(FIG_BAR[0], n_cats * 1.5), FIG_BAR[1]))
        # Horizon bars (only for first n_horizons)
        ax.bar(
            x[:n_horizons] - bar_width / 2,
            ratings_a,
            bar_width,
            yerr=sigmas_a,
            capsize=3,
            label=group_label,
            color="#2ca02c",
            alpha=0.7,
            edgecolor="black",
        )
        ax.bar(
            x[:n_horizons] + bar_width / 2,
            ratings_b,
            bar_width,
            yerr=sigmas_b,
            capsize=3,
            label=group_label_b,
            color="#1f77b4",
            alpha=0.7,
            edgecolor="black",
        )
        # Reference bars (weak / strong bot) at the end
        if n_ref > 0:
            ref_x = x[n_horizons:]
            for i in range(n_ref):
                ax.bar(
                    ref_x[i],
                    ref_ratings[i],
                    bar_width,
                    yerr=ref_sigmas[i],
                    capsize=3,
                    label=ref_labels[i],
                    color="#7f7f7f",
                    alpha=0.8,
                    edgecolor="black",
                )
        ax.set_xticks(x)
        ax.set_xticklabels(horizon_labels)
        if use_place_labels:
            # Rank by rating within each series (1st = highest)
            order_a = np.argsort(ratings_a)[::-1]
            order_b = np.argsort(ratings_b)[::-1]
            rank_a = np.empty(n_horizons, dtype=int)
            rank_b = np.empty(n_horizons, dtype=int)
            for r, idx in enumerate(order_a):
                rank_a[idx] = r + 1
            for r, idx in enumerate(order_b):
                rank_b[idx] = r + 1
            y_lo, y_hi = ax.get_ylim()
            offset = (y_hi - y_lo) * 0.02
            for i in range(n_horizons):
                ax.text(
                    x[i] - bar_width / 2,
                    ratings_a[i] + sigmas_a[i] + offset,
                    _place_label(rank_a[i]),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
                ax.text(
                    x[i] + bar_width / 2,
                    ratings_b[i] + sigmas_b[i] + offset,
                    _place_label(rank_b[i]),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
    else:
        n_horizons = len(labels_a)
        horizon_labels = list(labels_a) + ref_labels
        n_cats = n_horizons + n_ref
        x = np.arange(n_cats)
        bar_width = 0.28
        fig, ax = plt.subplots(figsize=(max(FIG_BAR[0], n_cats * 1.5), FIG_BAR[1]))
        ax.bar(
            x[:n_horizons],
            ratings_a,
            bar_width,
            yerr=sigmas_a,
            capsize=5,
            label=group_label,
            color="#2ca02c",
            alpha=0.7,
            edgecolor="black",
        )
        if n_ref > 0:
            ref_x = x[n_horizons:]
            for i in range(n_ref):
                ax.bar(
                    ref_x[i],
                    ref_ratings[i],
                    bar_width,
                    yerr=ref_sigmas[i],
                    capsize=5,
                    label=ref_labels[i],
                    color="#7f7f7f",
                    alpha=0.8,
                    edgecolor="black",
                )
        ax.set_xticks(x)
        ax.set_xticklabels(horizon_labels)
        if use_place_labels:
            order_a = np.argsort(ratings_a)[::-1]
            rank_a = np.empty(n_horizons, dtype=int)
            for r, idx in enumerate(order_a):
                rank_a[idx] = r + 1
            y_lo, y_hi = ax.get_ylim()
            offset = (y_hi - y_lo) * 0.02
            for i in range(n_horizons):
                ax.text(
                    x[i],
                    ratings_a[i] + sigmas_a[i] + offset,
                    _place_label(rank_a[i]),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    ax.set_xlabel("Horizon", fontsize=12)
    ax.set_ylabel("TrueSkill Rating", fontsize=12)
    ax.set_title("Agent ratings by horizon", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / "horizon_ratings.png", dpi=300, bbox_inches="tight")
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

    plt.title("Agent Performance Summary Table", fontsize=11, pad=20)
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

    registry, match_history = load_data(archive_root)
    plot_horizon_ratings(
        registry,
        agent_numbers=(6, 7, 8, 9, 10),
        agent_numbers_b=(11, 12, 13, 14, 15),
        group_label="no internal opponent modelling",
        group_label_b="internal opponent modelling",
        horizon_start=4,
        horizon_step=2,
        output_path=output_path,
        use_place_labels=True,
    )

    # Optional: plot only selected agents with custom labels (no match count labels)
    plot_agent_ratings(
        registry,
        output_path=output_path,
        agent_selection=[
            ("basic_weak", "weak bot"),
            ("basic_strong", "strong bot"),
            ("0001_SAC_2026-01-30_00-57-20", "SAC"),
            ("0002_TDMPC2_2026-02-13_12-39-15", "TDMPC2 internal opp and h=8"),
            ("0012_TDMPC2_2026-02-18_22-39-21", "TDMPC2 internal opp and h=6"),
        ],
        output_filename="agent_ratings_selected.png",
        show_match_count=False,
    )
