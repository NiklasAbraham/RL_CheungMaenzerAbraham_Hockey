"""
Reward sparsity analysis for the Laser Hockey environment.

This script investigates the sparse reward problem: most time steps yield rewards
near zero while the decisive +10 signal arrives only at the terminal winning step.
Without intervention this sparsity caused TD-MPC2 training to stagnate because
the value function received almost no useful learning signal.

The reward backpropagation technique (reward_backprop.py) addresses this by
injecting a discounted win bonus backwards through every step of a winning episode
in the replay buffer before any gradient updates are performed:

    r_t <- r_t + b * gamma_b^{(T-2) - t},   t = 0, ..., T-2

where b is the win bonus magnitude (b = 10), gamma_b is the backpropagation
discount factor (gamma_b = 0.82 at the start of training), and T is the episode
length.  The terminal step t = T-1 is left unchanged as it already carries +10.
During training the bonus magnitude is gradually phased out to zero so that the
agent eventually learns to optimise the true sparse signal.

This script simulates 100 random-play episodes to characterise the raw reward
distribution, then visualises the effect of the backpropagation technique.
Output figures are saved to report/figures/ for inclusion in the report.
"""

import os

import hockey.hockey_env as h_env
import matplotlib.pyplot as plt
import numpy as np

from rl_hockey.common.reward_backprop import apply_win_reward_backprop

try:
    from tueplots import axes as tue_axes
    from tueplots import bundles, figsizes

    plt.rcParams.update(
        bundles.neurips2024(usetex=False, rel_width=1.0, family="sans-serif")
    )
    plt.rcParams.update(tue_axes.grid(grid_alpha=0.3))
    plt.rcParams.update(tue_axes.spines(right=False, top=False))
    plt.rcParams["figure.constrained_layout.use"] = False
    _FIG_BACKPROP = figsizes.neurips2024(nrows=2, ncols=1)["figure.figsize"]
    _FIG_BACKPROP = (_FIG_BACKPROP[0] * 1.4, _FIG_BACKPROP[1] * 1.3)
    USE_TUEPLOTS = True
except ImportError:
    USE_TUEPLOTS = False
    _FIG_BACKPROP = (8, 8)

FIGURES_DIR = "report/figures"
NUM_GAMES = 100
MAX_STEPS = 250
WIN_REWARD_BONUS = 10.0
WIN_REWARD_DISCOUNT = 0.82


def play_random_games(num_games=NUM_GAMES, max_steps=MAX_STEPS):
    """
    Simulate episodes with uniformly random actions for both players.

    This establishes a baseline distribution of rewards prior to any learned
    policy.  The result reflects how sparse the reward signal actually is when
    the agent has no systematic strategy for scoring goals.

    Args:
        num_games: Number of episodes to simulate.
        max_steps: Maximum steps per episode before forced termination.

    Returns:
        Dictionary with keys 'all', 'player1_wins', 'player2_wins', 'draws',
        each containing a list of episode-data dictionaries.
    """
    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
    all_episodes = {"player1_wins": [], "player2_wins": [], "draws": [], "all": []}

    print(f"Simulating {num_games} random-play episodes...")
    for game_num in range(num_games):
        obs, info = env.reset()
        episode_rewards = []

        for _ in range(max_steps):
            action = np.hstack([env.action_space.sample(), env.action_space.sample()])
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
                truncated = False
            episode_rewards.append(float(reward))
            if done or truncated:
                break

        winner = info.get("winner", 0)
        episode_data = {
            "game_num": game_num,
            "rewards": episode_rewards,
            "total_reward": sum(episode_rewards),
            "steps": len(episode_rewards),
            "winner": winner,
        }
        all_episodes["all"].append(episode_data)
        if winner == 1:
            all_episodes["player1_wins"].append(episode_data)
        elif winner == -1:
            all_episodes["player2_wins"].append(episode_data)
        else:
            all_episodes["draws"].append(episode_data)

        if (game_num + 1) % 25 == 0:
            print(f"  {game_num + 1}/{num_games} episodes completed")

    env.close()
    n = len(all_episodes["all"])
    print(
        f"Results: {len(all_episodes['player1_wins'])} P1 wins, "
        f"{len(all_episodes['player2_wins'])} P2 wins, "
        f"{len(all_episodes['draws'])} draws  (total {n})"
    )
    return all_episodes


def plot_backprop_analysis(
    episodes,
    win_reward_bonus=WIN_REWARD_BONUS,
    win_reward_discount=WIN_REWARD_DISCOUNT,
    output_dir=FIGURES_DIR,
):
    """
    Three-panel visualisation of how reward backpropagation transforms the
    reward signal in winning episodes.

    Top panel: per-step reward trace for a representative winning episode
    before and after backpropagation, showing how a near-zero trajectory
    is enriched with a temporally discounted bonus.

    Bottom panel: scatter of episode total rewards before versus after
    backpropagation across all winning episodes, confirming that the
    technique shifts the total reward distribution upward and injects a
    meaningful training signal into the replay buffer.

    Args:
        episodes: Dictionary returned by play_random_games.
        win_reward_bonus: Bonus magnitude b added at the terminal step.
        win_reward_discount: Discount factor gamma_b for the bonus decay.
        output_dir: Directory in which to save the figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    win_eps = episodes["player1_wins"]
    if not win_eps:
        print("No winning episodes found; skipping backprop analysis.")
        return

    # Choose a representative episode with median episode length
    lengths = [ep["steps"] for ep in win_eps]
    median_len = float(np.median(lengths))
    rep_ep = min(win_eps, key=lambda e: abs(e["steps"] - median_len))
    rewards_raw = np.array(rep_ep["rewards"], dtype=np.float32)
    final_r, orig_r, _ = apply_win_reward_backprop(
        rewards_raw,
        winner=rep_ep["winner"],
        win_reward_bonus=win_reward_bonus,
        win_reward_discount=win_reward_discount,
    )
    steps_arr = np.arange(len(rewards_raw))

    # Collect totals across all winning episodes for the scatter panel
    orig_totals, final_totals = [], []
    for ep in win_eps:
        r = np.array(ep["rewards"], dtype=np.float32)
        fr, orr, _ = apply_win_reward_backprop(
            r,
            winner=ep["winner"],
            win_reward_bonus=win_reward_bonus,
            win_reward_discount=win_reward_discount,
        )
        orig_totals.append(float(np.sum(orr)))
        final_totals.append(float(np.sum(fr)))

    fig, axes = plt.subplots(2, 1, figsize=_FIG_BACKPROP)

    ax = axes[0]
    ax.plot(
        steps_arr,
        orig_r,
        color="#1f77b4",
        linewidth=1.5,
        alpha=0.8,
        label="Original reward",
    )
    ax.plot(
        steps_arr,
        final_r,
        color="#2ca02c",
        linewidth=2,
        alpha=0.9,
        label="After backprop",
    )
    ax.axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title(
        f"Representative winning episode: per-step reward "
        f"(episode {rep_ep['game_num']}, {rep_ep['steps']} steps)"
    )
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(
        orig_totals,
        final_totals,
        alpha=0.7,
        s=30,
        color="#9467bd",
        edgecolors="black",
        linewidths=0.4,
    )
    lo = min(min(orig_totals), min(final_totals))
    hi = max(max(orig_totals), max(final_totals))
    ax.plot(
        [lo, hi],
        [lo, hi],
        "r--",
        linewidth=1.5,
        alpha=0.6,
        label="No change ($y = x$)",
    )
    ax.set_xlabel("Original episode total reward")
    ax.set_ylabel("Episode total after backprop")
    ax.set_title(
        f"Total reward change from backpropagation "
        f"({len(win_eps)} winning episodes)"
    )
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "reward_backprop_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Backprop analysis figure saved to: {out_path}")


def print_summary(episodes):
    """Print a concise statistical summary of the simulated episodes."""
    n = len(episodes["all"])
    n_w = len(episodes["player1_wins"])
    n_l = len(episodes["player2_wins"])
    n_d = len(episodes["draws"])
    all_rewards = [r for ep in episodes["all"] for r in ep["rewards"]]

    print()
    print("REWARD SUMMARY")
    print("=" * 50)
    print(f"Total episodes    : {n}")
    print(f"Player 1 wins     : {n_w}  ({100 * n_w / n:.1f} %)")
    print(f"Player 2 wins     : {n_l}  ({100 * n_l / n:.1f} %)")
    print(f"Draws             : {n_d}  ({100 * n_d / n:.1f} %)")
    print()
    if all_rewards:
        pos = sum(1 for r in all_rewards if r > 1e-9)
        neg = sum(1 for r in all_rewards if r < -1e-9)
        zero = len(all_rewards) - pos - neg
        print(f"Total steps       : {len(all_rewards)}")
        print(f"Mean step reward  : {np.mean(all_rewards):.5f}")
        print(f"Zero rewards      : {zero}  ({100 * zero / len(all_rewards):.1f} %)")
        print(f"Positive rewards  : {pos}  ({100 * pos / len(all_rewards):.1f} %)")
        print(f"Negative rewards  : {neg}  ({100 * neg / len(all_rewards):.1f} %)")
    print()


def main():
    episodes = play_random_games()
    print_summary(episodes)
    plot_backprop_analysis(episodes)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
