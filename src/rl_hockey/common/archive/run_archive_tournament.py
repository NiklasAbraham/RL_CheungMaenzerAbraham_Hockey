"""
Run archive tournament: evaluate agents against each other.
"""

import logging
from pathlib import Path

import hockey.hockey_env as h_env
import numpy as np

from rl_hockey.common.archive.archive import Archive
from rl_hockey.common.archive.matchmaker import Matchmaker
from rl_hockey.common.archive.rating_system import RatingSystem

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Config: edit these paths and parameters
CONFIG = {
    "archive_dir": Path("archive"),
    "num_games_per_match": 150,
    "max_steps": 500,
    "filter_tags": [],
    "exclude_baselines": True,
}


def play_match(env, player1, player2, max_steps=250, seed=None):
    """Play a single match between two players."""
    if seed is not None:
        np.random.seed(seed)

    obs, _ = env.reset()

    for _ in range(max_steps):
        # Player 1 action
        action1 = player1.act(obs.astype(np.float32), deterministic=True)

        # Player 2 action
        obs_player2 = env.obs_agent_two()
        action2 = player2.act(obs_player2.astype(np.float32), deterministic=True)

        # Step environment
        action = np.hstack([action1, action2])
        obs, reward, done, trunc, info = env.step(action)

        if done or trunc:
            break

    return info.get("winner", 0)


def run_match_series(
    archive,
    matchmaker,
    rating_system,
    agent1_id,
    agent2_id,
    num_games=10,
    max_steps=250,
    update_ratings=True,
):
    """Run a series of matches between two agents."""
    logger.info(f"\nMatch: {agent1_id} vs {agent2_id}")

    # Load agents
    agent1_meta = archive.get_agent_metadata(agent1_id)
    agent2_meta = archive.get_agent_metadata(agent2_id)

    if agent1_meta is None or agent2_meta is None:
        logger.error("Could not load agent metadata")
        return None

    agent1 = matchmaker.load_opponent(agent1_meta, deterministic=True)
    agent2 = matchmaker.load_opponent(agent2_meta, deterministic=True)

    # Play matches
    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)

    wins_1 = wins_2 = draws = 0
    for game in range(num_games):
        seed = np.random.randint(0, 2**31)
        result = play_match(env, agent1, agent2, max_steps=max_steps, seed=seed)

        if result == 1:
            wins_1 += 1
        elif result == -1:
            wins_2 += 1
        else:
            draws += 1

        # Update ratings after each game if requested
        if update_ratings:
            rating_system.update_ratings(agent1_id, agent2_id, result, save=False)

    env.close()

    # Save ratings once after all games
    if update_ratings:
        rating_system.save_ratings()

    # Results
    logger.info(f"  {agent1_id}: {wins_1} wins")
    logger.info(f"  {agent2_id}: {wins_2} wins")
    logger.info(f"  Draws: {draws}")

    return {
        "agent1_id": agent1_id,
        "agent2_id": agent2_id,
        "agent1_wins": wins_1,
        "agent2_wins": wins_2,
        "draws": draws,
        "num_games": num_games,
    }


def run_round_robin_tournament(
    archive_dir,
    num_games_per_match,
    max_steps,
    filter_tags,
    exclude_baselines,
):
    logger.info("=" * 70)
    logger.info("Archive Round-Robin Tournament")
    logger.info("=" * 70)

    archive = Archive(base_dir=str(archive_dir))
    rating_system = RatingSystem(archive)
    matchmaker = Matchmaker(archive, rating_system)

    # Get agents for tournament
    if filter_tags:
        agents = archive.get_agents(tags=filter_tags)
    else:
        agents = archive.get_agents()

    if exclude_baselines:
        agents = [a for a in agents if "baseline" not in a.tags]

    logger.info(f"Tournament participants: {len(agents)} agent(s)")
    for agent in agents:
        logger.info(f"  {agent.agent_id} (rating: {agent.rating.rating:.2f})")

    if len(agents) < 2:
        logger.error("Need at least 2 agents for tournament")
        return

    # Generate all match pairs
    matches = []
    for i, agent1 in enumerate(agents):
        for agent2 in agents[i + 1 :]:
            matches.append((agent1.agent_id, agent2.agent_id))

    logger.info(f"\nTotal matches: {len(matches)}")
    logger.info(f"Games per match: {num_games_per_match}")
    logger.info(f"Total games: {len(matches) * num_games_per_match}")

    # Run tournament
    results = []
    total_matches = len(matches)
    for match_idx, (agent1_id, agent2_id) in enumerate(matches, 1):
        logger.info(f"Match {match_idx}/{total_matches}")
        result = run_match_series(
            archive,
            matchmaker,
            rating_system,
            agent1_id,
            agent2_id,
            num_games=num_games_per_match,
            max_steps=max_steps,
            update_ratings=True,
        )
        if result:
            results.append(result)

    # Display final leaderboard
    logger.info("\n" + "=" * 70)
    logger.info("Final Leaderboard")
    logger.info("=" * 70)

    leaderboard = rating_system.get_leaderboard(min_matches=1)
    for rank, (agent_id, rating) in enumerate(leaderboard, 1):
        if any(agent_id == a.agent_id for a in agents):
            logger.info(
                f"{rank:2d}. {agent_id:50s} | "
                f"Rating: {rating.rating:7.2f} (μ={rating.mu:.2f}, σ={rating.sigma:.2f}) | "
                f"Record: {rating.wins}W-{rating.losses}L-{rating.draws}D"
            )

    return results


def main():
    cfg = CONFIG
    run_round_robin_tournament(
        archive_dir=str(cfg["archive_dir"]),
        num_games_per_match=cfg["num_games_per_match"],
        max_steps=cfg["max_steps"],
        filter_tags=cfg.get("filter_tags"),
        exclude_baselines=cfg.get("exclude_baselines", True),
    )


if __name__ == "__main__":
    main()
