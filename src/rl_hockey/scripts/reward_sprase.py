import os
import shutil

import hockey.hockey_env as h_env
import matplotlib.pyplot as plt
import numpy as np

from rl_hockey.common.reward_backprop import apply_win_reward_backprop


def play_random_games(num_games=100, max_steps=250):
    """
    Play multiple games with random actions for both players.

    Args:
        num_games: Number of games to play
        max_steps: Maximum steps per episode

    Returns:
        Dictionary containing all episode data
    """
    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)

    # Storage for all episodes
    all_episodes = {
        "player1_wins": [],  # Episodes where player 1 won
        "player2_wins": [],  # Episodes where player 2 won
        "draws": [],  # Episodes that ended in a draw
        "all": [],  # All episodes
    }

    print(f"Playing {num_games} games with random actions for both players...")
    print("=" * 60)

    for game_num in range(num_games):
        obs, info = env.reset()
        episode_rewards = []
        episode_steps = 0

        for step in range(max_steps):
            # Random actions for both players
            a1 = env.action_space.sample()
            a2 = env.action_space.sample()
            action = np.hstack([a1, a2])

            # Step environment
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
                truncated = False

            # Store reward for player 1
            episode_rewards.append(float(reward))
            episode_steps += 1

            if done or truncated:
                break

        # Get winner information
        winner = info.get("winner", 0)

        # Store episode data
        episode_data = {
            "game_num": game_num,
            "rewards": episode_rewards,
            "total_reward": sum(episode_rewards),
            "mean_reward": np.mean(episode_rewards),
            "steps": episode_steps,
            "winner": winner,
        }

        all_episodes["all"].append(episode_data)

        if winner == 1:
            all_episodes["player1_wins"].append(episode_data)
        elif winner == -1:
            all_episodes["player2_wins"].append(episode_data)
        else:
            all_episodes["draws"].append(episode_data)

        if (game_num + 1) % 10 == 0:
            print(f"Completed {game_num + 1}/{num_games} games...")

    env.close()
    print("=" * 60)
    print("All games completed!")
    print()

    return all_episodes


def analyze_rewards(episodes):
    """
    Analyze reward distributions for different episode outcomes.

    Args:
        episodes: Dictionary with episode data separated by outcome
    """
    print("REWARD DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print()

    # Collect all rewards
    all_rewards = []
    for ep in episodes["all"]:
        all_rewards.extend(ep["rewards"])

    player1_win_rewards = []
    for ep in episodes["player1_wins"]:
        player1_win_rewards.extend(ep["rewards"])

    player2_win_rewards = []
    for ep in episodes["player2_wins"]:
        player2_win_rewards.extend(ep["rewards"])

    draw_rewards = []
    for ep in episodes["draws"]:
        draw_rewards.extend(ep["rewards"])

    # Print statistics
    print("OVERALL STATISTICS (All Episodes)")
    print("-" * 60)
    print(f"Total games: {len(episodes['all'])}")
    print(
        f"Player 1 wins: {len(episodes['player1_wins'])} ({100 * len(episodes['player1_wins']) / len(episodes['all']):.1f}%)"
    )
    print(
        f"Player 2 wins: {len(episodes['player2_wins'])} ({100 * len(episodes['player2_wins']) / len(episodes['all']):.1f}%)"
    )
    print(
        f"Draws: {len(episodes['draws'])} ({100 * len(episodes['draws']) / len(episodes['all']):.1f}%)"
    )
    print()

    print("REWARD STATISTICS - ALL EPISODES")
    print("-" * 60)
    if all_rewards:
        print(f"Total steps: {len(all_rewards)}")
        print(f"Min reward: {min(all_rewards):.6f}")
        print(f"Max reward: {max(all_rewards):.6f}")
        print(f"Mean reward: {np.mean(all_rewards):.6f}")
        print(f"Median reward: {np.median(all_rewards):.6f}")
        print(f"Std reward: {np.std(all_rewards):.6f}")
        positive_count = sum(1 for r in all_rewards if r > 1e-12)
        negative_count = sum(1 for r in all_rewards if r < -1e-12)
        zero_count = sum(1 for r in all_rewards if abs(r) <= 1e-12)
        print(
            f"Positive rewards: {positive_count} ({100 * positive_count / len(all_rewards):.2f}%)"
        )
        print(
            f"Negative rewards: {negative_count} ({100 * negative_count / len(all_rewards):.2f}%)"
        )
        print(
            f"Zero rewards: {zero_count} ({100 * zero_count / len(all_rewards):.2f}%)"
        )
    print()

    print("REWARD STATISTICS - PLAYER 1 WINS")
    print("-" * 60)
    if player1_win_rewards:
        print(f"Total steps: {len(player1_win_rewards)}")
        print(f"Min reward: {min(player1_win_rewards):.6f}")
        print(f"Max reward: {max(player1_win_rewards):.6f}")
        print(f"Mean reward: {np.mean(player1_win_rewards):.6f}")
        print(f"Median reward: {np.median(player1_win_rewards):.6f}")
        print(f"Std reward: {np.std(player1_win_rewards):.6f}")
        positive_count = sum(1 for r in player1_win_rewards if r > 1e-12)
        negative_count = sum(1 for r in player1_win_rewards if r < -1e-12)
        zero_count = sum(1 for r in player1_win_rewards if abs(r) <= 1e-12)
        print(
            f"Positive rewards: {positive_count} ({100 * positive_count / len(player1_win_rewards):.2f}%)"
        )
        print(
            f"Negative rewards: {negative_count} ({100 * negative_count / len(player1_win_rewards):.2f}%)"
        )
        print(
            f"Zero rewards: {zero_count} ({100 * zero_count / len(player1_win_rewards):.2f}%)"
        )

        # Episode-level statistics
        total_rewards = [ep["total_reward"] for ep in episodes["player1_wins"]]
        print("\nEpisode total rewards:")
        print(f"  Min: {min(total_rewards):.6f}")
        print(f"  Max: {max(total_rewards):.6f}")
        print(f"  Mean: {np.mean(total_rewards):.6f}")
        print(f"  Median: {np.median(total_rewards):.6f}")
    else:
        print("No episodes where Player 1 won.")
    print()

    print("REWARD STATISTICS - PLAYER 2 WINS (Player 1's perspective)")
    print("-" * 60)
    if player2_win_rewards:
        print(f"Total steps: {len(player2_win_rewards)}")
        print(f"Min reward: {min(player2_win_rewards):.6f}")
        print(f"Max reward: {max(player2_win_rewards):.6f}")
        print(f"Mean reward: {np.mean(player2_win_rewards):.6f}")
        print(f"Median reward: {np.median(player2_win_rewards):.6f}")
        print(f"Std reward: {np.std(player2_win_rewards):.6f}")
        positive_count = sum(1 for r in player2_win_rewards if r > 1e-12)
        negative_count = sum(1 for r in player2_win_rewards if r < -1e-12)
        zero_count = sum(1 for r in player2_win_rewards if abs(r) <= 1e-12)
        print(
            f"Positive rewards: {positive_count} ({100 * positive_count / len(player2_win_rewards):.2f}%)"
        )
        print(
            f"Negative rewards: {negative_count} ({100 * negative_count / len(player2_win_rewards):.2f}%)"
        )
        print(
            f"Zero rewards: {zero_count} ({100 * zero_count / len(player2_win_rewards):.2f}%)"
        )

        # Episode-level statistics
        total_rewards = [ep["total_reward"] for ep in episodes["player2_wins"]]
        print("\nEpisode total rewards:")
        print(f"  Min: {min(total_rewards):.6f}")
        print(f"  Max: {max(total_rewards):.6f}")
        print(f"  Mean: {np.mean(total_rewards):.6f}")
        print(f"  Median: {np.median(total_rewards):.6f}")
    else:
        print("No episodes where Player 2 won.")
    print()

    print("REWARD STATISTICS - DRAWS")
    print("-" * 60)
    if draw_rewards:
        print(f"Total steps: {len(draw_rewards)}")
        print(f"Min reward: {min(draw_rewards):.6f}")
        print(f"Max reward: {max(draw_rewards):.6f}")
        print(f"Mean reward: {np.mean(draw_rewards):.6f}")
        print(f"Median reward: {np.median(draw_rewards):.6f}")
        print(f"Std reward: {np.std(draw_rewards):.6f}")
        positive_count = sum(1 for r in draw_rewards if r > 1e-12)
        negative_count = sum(1 for r in draw_rewards if r < -1e-12)
        zero_count = sum(1 for r in draw_rewards if abs(r) <= 1e-12)
        print(
            f"Positive rewards: {positive_count} ({100 * positive_count / len(draw_rewards):.2f}%)"
        )
        print(
            f"Negative rewards: {negative_count} ({100 * negative_count / len(draw_rewards):.2f}%)"
        )
        print(
            f"Zero rewards: {zero_count} ({100 * zero_count / len(draw_rewards):.2f}%)"
        )

        # Episode-level statistics
        total_rewards = [ep["total_reward"] for ep in episodes["draws"]]
        print("\nEpisode total rewards:")
        print(f"  Min: {min(total_rewards):.6f}")
        print(f"  Max: {max(total_rewards):.6f}")
        print(f"  Mean: {np.mean(total_rewards):.6f}")
        print(f"  Median: {np.median(total_rewards):.6f}")
    else:
        print("No draw episodes.")
    print()

    return {
        "all": all_rewards,
        "player1_wins": player1_win_rewards,
        "player2_wins": player2_win_rewards,
        "draws": draw_rewards,
    }


def plot_reward_distributions(reward_data, episodes):
    """
    Create comprehensive plots of reward distributions.

    Args:
        reward_data: Dictionary with reward lists by outcome
        episodes: Dictionary with episode data
    """
    import os

    plt.figure(figsize=(16, 16))

    # Plot 1: Overall reward distribution
    ax1 = plt.subplot(4, 2, 1)
    if reward_data["all"]:
        ax1.hist(
            reward_data["all"], bins=100, alpha=0.7, color="gray", edgecolor="black"
        )
        ax1.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero")
        ax1.set_xlabel("Reward")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Overall Reward Distribution (All Episodes)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Plot 2: Player 1 wins - reward distribution
    ax2 = plt.subplot(4, 2, 2)
    if reward_data["player1_wins"]:
        ax2.hist(
            reward_data["player1_wins"],
            bins=100,
            alpha=0.7,
            color="green",
            edgecolor="black",
        )
        ax2.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero")
        ax2.set_xlabel("Reward")
        ax2.set_ylabel("Frequency")
        ax2.set_title(
            f"Reward Distribution - Player 1 Wins ({len(episodes['player1_wins'])} episodes)"
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Plot 3: Player 2 wins - reward distribution (Player 1's perspective)
    ax3 = plt.subplot(4, 2, 3)
    if reward_data["player2_wins"]:
        ax3.hist(
            reward_data["player2_wins"],
            bins=100,
            alpha=0.7,
            color="red",
            edgecolor="black",
        )
        ax3.axvline(x=0, color="blue", linestyle="--", linewidth=2, label="Zero")
        ax3.set_xlabel("Reward")
        ax3.set_ylabel("Frequency")
        ax3.set_title(
            f"Reward Distribution - Player 2 Wins ({len(episodes['player2_wins'])} episodes)"
        )
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Draws - reward distribution
    ax4 = plt.subplot(4, 2, 4)
    if reward_data["draws"]:
        ax4.hist(
            reward_data["draws"], bins=100, alpha=0.7, color="orange", edgecolor="black"
        )
        ax4.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero")
        ax4.set_xlabel("Reward")
        ax4.set_ylabel("Frequency")
        ax4.set_title(
            f"Reward Distribution - Draws ({len(episodes['draws'])} episodes)"
        )
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # Plot 5: Comparison of reward distributions
    ax5 = plt.subplot(4, 2, 5)
    if reward_data["player1_wins"]:
        ax5.hist(
            reward_data["player1_wins"],
            bins=50,
            alpha=0.5,
            color="green",
            label=f"Player 1 Wins (n={len(reward_data['player1_wins'])})",
            density=True,
        )
    if reward_data["player2_wins"]:
        ax5.hist(
            reward_data["player2_wins"],
            bins=50,
            alpha=0.5,
            color="red",
            label=f"Player 2 Wins (n={len(reward_data['player2_wins'])})",
            density=True,
        )
    if reward_data["draws"]:
        ax5.hist(
            reward_data["draws"],
            bins=50,
            alpha=0.5,
            color="orange",
            label=f"Draws (n={len(reward_data['draws'])})",
            density=True,
        )
    ax5.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax5.set_xlabel("Reward")
    ax5.set_ylabel("Density")
    ax5.set_title("Normalized Reward Distribution Comparison")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Episode total rewards by outcome
    ax6 = plt.subplot(4, 2, 6)
    if episodes["player1_wins"]:
        p1_totals = [ep["total_reward"] for ep in episodes["player1_wins"]]
        ax6.hist(
            p1_totals,
            bins=30,
            alpha=0.5,
            color="green",
            label=f"Player 1 Wins (n={len(episodes['player1_wins'])})",
            density=True,
        )
    if episodes["player2_wins"]:
        p2_totals = [ep["total_reward"] for ep in episodes["player2_wins"]]
        ax6.hist(
            p2_totals,
            bins=30,
            alpha=0.5,
            color="red",
            label=f"Player 2 Wins (n={len(episodes['player2_wins'])})",
            density=True,
        )
    if episodes["draws"]:
        draw_totals = [ep["total_reward"] for ep in episodes["draws"]]
        ax6.hist(
            draw_totals,
            bins=30,
            alpha=0.5,
            color="orange",
            label=f"Draws (n={len(episodes['draws'])})",
            density=True,
        )
    ax6.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax6.set_xlabel("Episode Total Reward")
    ax6.set_ylabel("Density")
    ax6.set_title("Episode Total Reward Distribution by Outcome")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Plot 7: Player 1 wins - Episode total rewards distribution with mean
    ax7 = plt.subplot(4, 2, 7)
    if episodes["player1_wins"]:
        p1_totals = [ep["total_reward"] for ep in episodes["player1_wins"]]
        mean_total = np.mean(p1_totals)
        ax7.hist(
            p1_totals,
            bins=30,
            alpha=0.7,
            color="green",
            edgecolor="black",
        )
        ax7.axvline(
            x=mean_total,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_total:.4f}",
        )
        ax7.axvline(x=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
        ax7.set_xlabel("Episode Total Reward")
        ax7.set_ylabel("Frequency")
        ax7.set_title(
            f"Player 1 Wins - Episode Total Rewards (n={len(episodes['player1_wins'])})"
        )
        ax7.legend()
        ax7.grid(True, alpha=0.3)

    # Plot 8: Player 1 wins - Step-by-step reward mean across episodes
    ax8 = plt.subplot(4, 2, 8)
    if episodes["player1_wins"]:
        # Find maximum episode length
        max_length = max(len(ep["rewards"]) for ep in episodes["player1_wins"])

        # Pad all episodes to same length with NaN, then compute mean per step
        padded_rewards = []
        for ep in episodes["player1_wins"]:
            padded = ep["rewards"] + [np.nan] * (max_length - len(ep["rewards"]))
            padded_rewards.append(padded)

        # Compute mean reward per step (ignoring NaN)
        mean_per_step = []
        for step_idx in range(max_length):
            step_rewards = [
                padded[step_idx]
                for padded in padded_rewards
                if not np.isnan(padded[step_idx])
            ]
            if step_rewards:
                mean_per_step.append(np.mean(step_rewards))
            else:
                mean_per_step.append(np.nan)

        steps = np.arange(len(mean_per_step))
        ax8.plot(
            steps,
            mean_per_step,
            color="green",
            linewidth=2,
            label="Mean reward per step",
        )
        ax8.axhline(y=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
        ax8.set_xlabel("Step in Episode")
        ax8.set_ylabel("Mean Reward")
        ax8.set_title(
            f"Player 1 Wins - Mean Reward per Step Across Episodes (n={len(episodes['player1_wins'])})"
        )
        ax8.legend()
        ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    figures_dir = "src/rl_hockey/scripts/figures"
    os.makedirs(figures_dir, exist_ok=True)
    output_file = os.path.join(figures_dir, "reward_distribution_analysis.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {output_file}")
    plt.close()


def plot_individual_player1_win_episodes(
    episodes,
    output_dir=None,
    win_reward_bonus=0.0,
    win_reward_discount=0.98,
):
    """
    Plot and save individual reward distributions for each episode where Player 1 wins.

    Args:
        episodes: Dictionary with episode data
        output_dir: Directory to save individual episode plots (default: figures/player1_win_episodes)
        win_reward_bonus: Bonus reward to add to each step in a winning episode
        win_reward_discount: Discount factor for applying win reward bonus
    """
    import os

    if output_dir is None:
        output_dir = "src/rl_hockey/scripts/figures/player1_win_episodes"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    player1_win_episodes = episodes["player1_wins"]

    if not player1_win_episodes:
        print("No episodes where Player 1 won. Skipping individual episode plots.")
        return

    print(
        f"\nCreating individual plots for {len(player1_win_episodes)} Player 1 win episodes..."
    )

    for idx, episode in enumerate(player1_win_episodes):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        rewards = np.array(episode["rewards"], dtype=np.float32)
        game_num = episode["game_num"]
        total_reward = episode["total_reward"]
        mean_reward = episode["mean_reward"]
        steps = episode["steps"]
        winner = episode["winner"]

        # Apply backpropagation to get final rewards
        final_rewards, original_rewards, bonus_rewards = apply_win_reward_backprop(
            rewards,
            winner=winner,
            win_reward_bonus=win_reward_bonus,
            win_reward_discount=win_reward_discount,
            use_torch=False,
        )
        final_total_reward = np.sum(final_rewards)
        final_mean_reward = np.mean(final_rewards)

        # Plot 1: Reward distribution histogram
        ax1 = axes[0]
        ax1.hist(
            rewards,
            bins=min(50, len(rewards)),
            alpha=0.7,
            color="green",
            edgecolor="black",
        )
        ax1.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero")
        ax1.axvline(
            x=mean_reward,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_reward:.4f}",
        )
        ax1.set_xlabel("Reward")
        ax1.set_ylabel("Frequency")
        ax1.set_title(
            f"Player 1 Win - Episode {game_num} - Reward Distribution\n"
            f"Total Reward: {total_reward:.4f}, Mean: {mean_reward:.4f}, Steps: {steps}"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Reward over time (step-by-step)
        ax2 = axes[1]
        steps_array = np.arange(len(rewards))
        # Plot original rewards
        ax2.plot(
            steps_array,
            original_rewards,
            color="green",
            linewidth=1.5,
            alpha=0.7,
            label="Original Rewards",
        )
        # Plot final rewards after backpropagation
        ax2.plot(
            steps_array,
            final_rewards,
            color="purple",
            linewidth=2,
            linestyle="-",
            label=f"After Backprop (bonus={win_reward_bonus}, γ={win_reward_discount})",
        )
        ax2.axhline(y=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
        ax2.axhline(
            y=mean_reward,
            color="blue",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Original Mean: {mean_reward:.4f}",
        )
        ax2.axhline(
            y=final_mean_reward,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Final Mean: {final_mean_reward:.4f}",
        )
        ax2.fill_between(
            steps_array,
            original_rewards,
            0,
            alpha=0.2,
            color="green",
            where=(original_rewards >= 0),
        )
        ax2.fill_between(
            steps_array,
            original_rewards,
            0,
            alpha=0.2,
            color="red",
            where=(original_rewards < 0),
        )
        ax2.set_xlabel("Step in Episode")
        ax2.set_ylabel("Reward")
        ax2.set_title(
            f"Reward Over Time - Episode {game_num}\n"
            f"Original Total: {total_reward:.4f}, Final Total: {final_total_reward:.4f}, "
            f"Bonus Added: {np.sum(bonus_rewards):.4f}"
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(
            output_dir, f"player1_win_episode_{game_num:03d}.png"
        )
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        if (idx + 1) % 10 == 0:
            print(f"  Saved {idx + 1}/{len(player1_win_episodes)} episode plots...")

    print(f"All individual episode plots saved to: {output_dir}/")
    print(f"Total: {len(player1_win_episodes)} plots created")


def plot_individual_player2_win_episodes(episodes, output_dir=None):
    """
    Plot and save individual reward distributions for each episode where Player 2 wins (Player 1 loses).

    Args:
        episodes: Dictionary with episode data
        output_dir: Directory to save individual episode plots (default: figures/player2_win_episodes)
    """
    import os

    if output_dir is None:
        output_dir = "src/rl_hockey/scripts/figures/player2_win_episodes"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    player2_win_episodes = episodes["player2_wins"]

    if not player2_win_episodes:
        print("No episodes where Player 2 won. Skipping individual episode plots.")
        return

    print(
        f"\nCreating individual plots for {len(player2_win_episodes)} Player 2 win episodes (Player 1 losses)..."
    )

    for idx, episode in enumerate(player2_win_episodes):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        rewards = episode["rewards"]
        game_num = episode["game_num"]
        total_reward = episode["total_reward"]
        mean_reward = episode["mean_reward"]
        steps = episode["steps"]

        # Plot 1: Reward distribution histogram
        ax1 = axes[0]
        ax1.hist(
            rewards,
            bins=min(50, len(rewards)),
            alpha=0.7,
            color="red",
            edgecolor="black",
        )
        ax1.axvline(x=0, color="blue", linestyle="--", linewidth=2, label="Zero")
        ax1.axvline(
            x=mean_reward,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_reward:.4f}",
        )
        ax1.set_xlabel("Reward")
        ax1.set_ylabel("Frequency")
        ax1.set_title(
            f"Player 2 Win (Player 1 Loss) - Episode {game_num} - Reward Distribution\n"
            f"Total Reward: {total_reward:.4f}, Mean: {mean_reward:.4f}, Steps: {steps}"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Reward over time (step-by-step)
        ax2 = axes[1]
        steps_array = np.arange(len(rewards))
        ax2.plot(steps_array, rewards, color="red", linewidth=1.5, alpha=0.7)
        ax2.axhline(y=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
        ax2.axhline(
            y=mean_reward,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_reward:.4f}",
        )
        ax2.fill_between(
            steps_array,
            rewards,
            0,
            alpha=0.3,
            color="green",
            where=(np.array(rewards) >= 0),
        )
        ax2.fill_between(
            steps_array,
            rewards,
            0,
            alpha=0.3,
            color="red",
            where=(np.array(rewards) < 0),
        )
        ax2.set_xlabel("Step in Episode")
        ax2.set_ylabel("Reward")
        ax2.set_title(f"Reward Over Time - Episode {game_num}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(
            output_dir, f"player2_win_episode_{game_num:03d}.png"
        )
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        if (idx + 1) % 10 == 0:
            print(f"  Saved {idx + 1}/{len(player2_win_episodes)} episode plots...")

    print(f"All individual episode plots saved to: {output_dir}/")
    print(f"Total: {len(player2_win_episodes)} plots created")


def visualize_reward_backprop(
    episodes,
    win_reward_bonus=1.0,
    win_reward_discount=0.99,
    output_dir=None,
):
    """
    Visualize how reward backpropagation affects rewards at each time step.

    Creates plots showing original rewards, bonus rewards, and final rewards
    after backpropagation for winning episodes.

    Args:
        episodes: Dictionary with episode data, must have 'player1_wins' key
        win_reward_bonus: Bonus reward to add to each step in a winning episode
        win_reward_discount: Discount factor for applying win reward bonus
        output_dir: Directory to save plots (default: figures/reward_backprop)
    """
    import os

    if output_dir is None:
        output_dir = "src/rl_hockey/scripts/figures/reward_backprop"

    os.makedirs(output_dir, exist_ok=True)

    player1_win_episodes = episodes.get("player1_wins", [])

    if not player1_win_episodes:
        print("No episodes where Player 1 won. Skipping backpropagation visualization.")
        return

    print(
        f"\nCreating reward backpropagation visualizations for {len(player1_win_episodes)} Player 1 win episodes..."
    )

    for idx, episode in enumerate(player1_win_episodes):
        rewards = np.array(episode["rewards"], dtype=np.float32)
        game_num = episode["game_num"]
        winner = episode["winner"]

        # Apply backpropagation
        final_rewards, original_rewards, bonus_rewards = apply_win_reward_backprop(
            rewards,
            winner=winner,
            win_reward_bonus=win_reward_bonus,
            win_reward_discount=win_reward_discount,
            use_torch=False,
        )

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        steps = np.arange(len(rewards))

        # Plot 1: Original rewards
        ax1 = axes[0]
        ax1.plot(
            steps, original_rewards, color="blue", linewidth=2, label="Original Rewards"
        )
        ax1.fill_between(
            steps,
            original_rewards,
            0,
            alpha=0.3,
            color="green",
            where=(original_rewards >= 0),
        )
        ax1.fill_between(
            steps,
            original_rewards,
            0,
            alpha=0.3,
            color="red",
            where=(original_rewards < 0),
        )
        ax1.axhline(y=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
        ax1.set_xlabel("Step in Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title(f"Episode {game_num} - Original Rewards (Before Backpropagation)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Bonus rewards
        ax2 = axes[1]
        ax2.plot(
            steps, bonus_rewards, color="orange", linewidth=2, label="Bonus Rewards"
        )
        ax2.fill_between(steps, bonus_rewards, 0, alpha=0.3, color="orange")
        ax2.axhline(y=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
        ax2.set_xlabel("Step in Episode")
        ax2.set_ylabel("Bonus Reward")
        ax2.set_title(
            f"Episode {game_num} - Bonus Rewards (win_bonus={win_reward_bonus}, "
            f"discount={win_reward_discount})"
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Final rewards (original + bonus)
        ax3 = axes[2]
        ax3.plot(
            steps, final_rewards, color="green", linewidth=2, label="Final Rewards"
        )
        ax3.plot(
            steps,
            original_rewards,
            color="blue",
            linewidth=1.5,
            linestyle="--",
            alpha=0.5,
            label="Original Rewards",
        )
        ax3.fill_between(
            steps,
            final_rewards,
            0,
            alpha=0.3,
            color="green",
            where=(final_rewards >= 0),
        )
        ax3.fill_between(
            steps,
            final_rewards,
            0,
            alpha=0.3,
            color="red",
            where=(final_rewards < 0),
        )
        ax3.axhline(y=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
        ax3.set_xlabel("Step in Episode")
        ax3.set_ylabel("Reward")
        ax3.set_title(
            f"Episode {game_num} - Final Rewards (After Backpropagation)\n"
            f"Total Original: {np.sum(original_rewards):.4f}, "
            f"Total Bonus: {np.sum(bonus_rewards):.4f}, "
            f"Total Final: {np.sum(final_rewards):.4f}"
        )
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(
            output_dir, f"reward_backprop_episode_{game_num:03d}.png"
        )
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close()

        if (idx + 1) % 10 == 0:
            print(
                f"  Saved {idx + 1}/{len(player1_win_episodes)} backpropagation plots..."
            )

    # Create summary plot comparing original vs final rewards across all episodes
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Collect all rewards
    all_original_totals = []
    all_final_totals = []
    all_bonus_totals = []

    for episode in player1_win_episodes:
        rewards = np.array(episode["rewards"], dtype=np.float32)
        winner = episode["winner"]
        final_rewards, original_rewards, bonus_rewards = apply_win_reward_backprop(
            rewards,
            winner=winner,
            win_reward_bonus=win_reward_bonus,
            win_reward_discount=win_reward_discount,
            use_torch=False,
        )
        all_original_totals.append(np.sum(original_rewards))
        all_final_totals.append(np.sum(final_rewards))
        all_bonus_totals.append(np.sum(bonus_rewards))

    # Plot 1: Distribution of episode totals
    ax1 = axes[0]
    ax1.hist(
        all_original_totals,
        bins=30,
        alpha=0.5,
        color="blue",
        label=f"Original Total Rewards (mean={np.mean(all_original_totals):.4f})",
        edgecolor="black",
    )
    ax1.hist(
        all_final_totals,
        bins=30,
        alpha=0.5,
        color="green",
        label=f"Final Total Rewards (mean={np.mean(all_final_totals):.4f})",
        edgecolor="black",
    )
    ax1.axvline(x=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax1.set_xlabel("Episode Total Reward")
    ax1.set_ylabel("Frequency")
    ax1.set_title(
        f"Distribution of Episode Total Rewards\n"
        f"Before vs After Backpropagation (n={len(player1_win_episodes)} episodes)"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scatter plot showing change
    ax2 = axes[1]
    ax2.scatter(
        all_original_totals,
        all_final_totals,
        alpha=0.6,
        s=50,
        color="green",
        edgecolors="black",
        linewidths=0.5,
    )
    # Add diagonal line (y=x)
    min_val = min(min(all_original_totals), min(all_final_totals))
    max_val = max(max(all_original_totals), max(all_final_totals))
    ax2.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="No Change (y=x)",
        alpha=0.5,
    )
    ax2.set_xlabel("Original Total Reward")
    ax2.set_ylabel("Final Total Reward")
    ax2.set_title(
        f"Reward Change After Backpropagation\n"
        f"Mean Bonus Added: {np.mean(all_bonus_totals):.4f}"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    summary_file = os.path.join(output_dir, "reward_backprop_summary.png")
    plt.savefig(summary_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"All backpropagation visualizations saved to: {output_dir}/")
    print(f"Total: {len(player1_win_episodes)} individual plots + 1 summary plot")


def main():
    """Main function to run the reward analysis."""
    # Clean up old figures and directories
    figures_base_dir = "src/rl_hockey/scripts/figures"
    directories_to_remove = [
        os.path.join(figures_base_dir, "player1_win_episodes"),
        os.path.join(figures_base_dir, "player2_win_episodes"),
        os.path.join(figures_base_dir, "reward_backprop"),
    ]
    files_to_remove = [
        os.path.join(figures_base_dir, "reward_distribution_analysis.png"),
        os.path.join(figures_base_dir, "reward_backprop_summary.png"),
    ]

    print("Cleaning up old figures and directories...")
    for dir_path in directories_to_remove:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"  Removed directory: {dir_path}")

    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"  Removed file: {file_path}")

    print("Cleanup complete!\n")

    # Play 100 games with random actions
    episodes = play_random_games(num_games=100, max_steps=250)

    # Analyze rewards
    reward_data = analyze_rewards(episodes)

    # Plot distributions
    plot_reward_distributions(reward_data, episodes)

    # Plot individual Player 1 win episodes (with backpropagation visualization)
    plot_individual_player1_win_episodes(
        episodes,
        win_reward_bonus=10.0,
        win_reward_discount=0.82,
    )

    # Plot individual Player 2 win episodes (Player 1 losses)
    plot_individual_player2_win_episodes(episodes)

    # Visualize reward backpropagation for winning episodes
    visualize_reward_backprop(
        episodes,
        win_reward_bonus=10.0,
        win_reward_discount=0.82,
    )

    print("Analysis complete!")


if __name__ == "__main__":
    main()
