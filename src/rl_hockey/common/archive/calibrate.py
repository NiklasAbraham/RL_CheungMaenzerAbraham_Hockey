"""
Tournament-style calibration for new agents.

Calibrates new agents by playing matches against existing agents
and updating their TrueSkill ratings.
"""

import argparse
import numpy as np
import hockey.hockey_env as h_env
from tqdm import tqdm

from rl_hockey.common.archive.archive import Archive, Rating
from rl_hockey.common.archive.rating_system import RatingSystem
from rl_hockey.common.archive.matchmaker import Matchmaker


def play_match(env: h_env.HockeyEnv, player1, player2, max_steps: int = 250) -> int:
    """Play a single match between two players."""
    obs, _ = env.reset()
    
    for _ in range(max_steps):
        action1 = player1.act(obs.astype(np.float32))
        
        obs_player2 = env.obs_agent_two()
        action2 = player2.act(obs_player2.astype(np.float32))
        
        action = np.hstack([action1, action2])
        obs, reward, done, trunc, info = env.step(action)
        
        if done or trunc:
            break

    winner = info["winner"]  # 1 if player1 wins, -1 if player2 wins, 0 if draw
    return winner


def calibrate_agent(
    archive: Archive,
    rating_system: RatingSystem,
    agent_id: str,
    num_matches: int = 50,
    baseline_ratio: float = 0.1,
    random_ratio: float = 0.1,
) -> Rating:
    """
    Calibrate a single agent by playing matches against sampled opponents.
    
    Args:
        archive: Archive instance
        rating_system: Rating system instance
        agent_id: ID of the agent to calibrate
        num_matches: Total number of matches to play
        baseline_ratio: Fraction of matches against baseline opponents
        random_ratio: Fraction of matches against random opponents
        
    Returns:
        Final rating for the agent
    """
    agent_metadata = archive.get_agent_metadata(agent_id)
    if not agent_metadata:
        raise ValueError(f"Agent {agent_id} not found in archive")
    
    matchmaker = Matchmaker(archive, rating_system)

    env = h_env.HockeyEnv()
    agent = matchmaker.load_opponent(agent_metadata, deterministic=True)
    
    print(f"\nCalibrating {agent_id} over {num_matches} matches...")
    
    wins = losses = draws = 0
    for _ in tqdm(range(num_matches), desc="Calibrating"):
        # Sample opponent with skill-based matching
        opponent, opp_id, _ = matchmaker.sample_archive_opponent(
            rating=rating_system.get_rating(agent_id).rating,
            distribution={
                "skill": 1.0 - baseline_ratio - random_ratio,
                "baseline": baseline_ratio,
                "random": random_ratio,
            }
        )
        
        result = play_match(env, agent, opponent, max_steps=250)
        
        # Update ratings
        rating_system.update_ratings(agent_id, opp_id, result, save=True)
        
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1
    
    env.close()
    
    # Remove calibration tag
    archive.remove_agent_tag(agent_id, "needs_calibration")
    
    final_rating = rating_system.get_rating(agent_id)
    
    print(f"\nCalibration Complete: {agent_id}")
    print(f"  Wins:   {wins} ({wins/num_matches*100:.1f}%)")
    print(f"  Losses: {losses} ({losses/num_matches*100:.1f}%)")
    print(f"  Draws:  {draws} ({draws/num_matches*100:.1f}%)")
    print(f"  Final Rating: {final_rating.rating:.2f} (μ={final_rating.mu:.2f}, σ={final_rating.sigma:.2f})")
    
    return final_rating


def calibrate_all_pending(
    archive_dir: str = "archive",
    num_matches: int = 50,
    baseline_ratio: float = 0.1,
    random_ratio: float = 0.1,
):
    """Calibrate all agents with 'needs_calibration' tag."""
    archive = Archive(base_dir=archive_dir)
    rating_system = RatingSystem(archive)
    
    pending_agents = archive.get_agents(tags=["needs_calibration"])
    
    if not pending_agents:
        print("No agents need calibration.")
        return
    
    print(f"\nCalibrating {len(pending_agents)} agent(s)...")
    
    for agent_metadata in pending_agents:
        calibrate_agent(
            archive,
            rating_system,
            agent_metadata.agent_id,
            num_matches=num_matches,
            baseline_ratio=baseline_ratio,
            random_ratio=random_ratio,
        )
    
    print("\nLeaderboard:")
    leaderboard = rating_system.get_leaderboard(min_matches=1)
    for rank, (agent_id, rating) in enumerate(leaderboard[:10], 1):
        print(f"  {rank:2d}. {agent_id:40s} | {rating.rating:7.2f}")


def main():
    """Command-line interface for agent calibration."""
    parser = argparse.ArgumentParser(description="Calibrate new agents")
    parser.add_argument("--archive", type=str, default="archive", help="Archive directory")
    parser.add_argument("--agent-id", type=str, help="Specific agent to calibrate")
    parser.add_argument("--matches", type=int, default=100, help="Number of matches")
    parser.add_argument("--baseline-ratio", type=float, default=0.1, help="Baseline match ratio")
    parser.add_argument("--random-ratio", type=float, default=0.1, help="Random match ratio")
    
    args = parser.parse_args()
    
    archive = Archive(base_dir=args.archive)
    rating_system = RatingSystem(archive)
    
    if args.agent_id:
        calibrate_agent(
            archive,
            rating_system,
            args.agent_id,
            num_matches=args.matches,
            baseline_ratio=args.baseline_ratio,
            random_ratio=args.random_ratio,
        )
    else:
        calibrate_all_pending(
            archive_dir=args.archive,
            num_matches=args.matches,
            baseline_ratio=args.baseline_ratio,
            random_ratio=args.random_ratio,
        )


if __name__ == "__main__":
    main()
