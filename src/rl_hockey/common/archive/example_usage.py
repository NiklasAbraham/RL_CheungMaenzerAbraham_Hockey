"""
Example usage of the agent archive and rating system.

This script demonstrates:
1. Creating an archive
2. Adding agents to archive
3. Head-to-head evaluation
4. Rating updates
5. Leaderboard generation
"""

from pathlib import Path
from rl_hockey.common.archive import AgentArchiveManager, AgentRatingSystem
from rl_hockey.common.evaluation.agent_evaluator import evaluate_head_to_head


def example_basic_archive():
    """Example 1: Basic archive usage."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Archive Usage")
    print("=" * 70)
    
    # Initialize archive
    archive = AgentArchiveManager(base_archive_dir="results/archive")
    
    print(f"\nArchive directory: {archive.base_dir}")
    print(f"Current agents in archive: {len(archive.list_agents())}")
    
    # Example: Add an agent to archive
    # Uncomment and adjust paths when you have a trained agent
    """
    agent_id = archive.add_agent(
        checkpoint_path="results/runs/2026-01-21_12-00-00/models/checkpoint_ep005000.pt",
        config_path="results/runs/2026-01-21_12-00-00/configs/config.json",
        metadata={
            "training_info": {
                "algorithm": "DDDQN",
                "episodes": 5000,
                "training_date": "2026-01-21",
            },
            "performance": {
                "win_rate_vs_weak": 0.85,
                "win_rate_vs_strong": 0.42,
                "mean_reward": 7.2,
            }
        },
        agent_name="dddqn_5k_episodes",
        source_run_dir="results/runs/2026-01-21_12-00-00"
    )
    print(f"\nAdded agent to archive: {agent_id}")
    """
    
    # List all agents
    print("\nAgents in archive:")
    for agent_meta in archive.list_agents():
        print(f"  - {agent_meta['agent_id']}")
        print(f"    Archived: {agent_meta['archived_at']}")
        if 'training_info' in agent_meta:
            training_info = agent_meta['training_info']
            print(f"    Algorithm: {training_info.get('algorithm', 'N/A')}")
            print(f"    Episodes: {training_info.get('episodes', 'N/A')}")
    
    # Get archive stats
    stats = archive.get_archive_stats()
    print(f"\nArchive statistics:")
    print(f"  Total agents: {stats['total_agents']}")


def example_rating_system():
    """Example 2: Rating system usage."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Rating System")
    print("=" * 70)
    
    # Initialize rating system
    rating_system = AgentRatingSystem(archive_dir="results/archive")
    
    # Initialize some demo agents
    print("\nInitializing agents with default ratings...")
    for i in range(1, 4):
        agent_id = f"agent_{i:04d}_demo"
        rating_system.initialize_agent(agent_id)
        rating = rating_system.get_rating(agent_id)
        print(f"  {agent_id}: μ={rating.mu:.1f}, σ={rating.sigma:.1f}, rating={rating.rating:.1f}")
    
    # Simulate some matches
    print("\nSimulating matches...")
    matches = [
        ("agent_0001_demo", "agent_0002_demo", 1),   # agent_0001 wins
        ("agent_0001_demo", "agent_0003_demo", 1),   # agent_0001 wins
        ("agent_0002_demo", "agent_0003_demo", -1),  # agent_0003 wins
        ("agent_0001_demo", "agent_0002_demo", 0),   # draw
    ]
    
    for agent1, agent2, result in matches:
        result_str = "won" if result == 1 else "lost" if result == -1 else "drew"
        print(f"  {agent1} vs {agent2}: {agent1} {result_str}")
        rating_system.update_ratings(agent1, agent2, result)
    
    # Display leaderboard
    print("\nLeaderboard after matches:")
    for rank, (agent_id, rating) in enumerate(rating_system.get_leaderboard(), 1):
        print(f"  {rank}. {agent_id}")
        print(f"     Rating: {rating.rating:.1f} (μ={rating.mu:.1f}, σ={rating.sigma:.1f})")
        print(f"     Record: {rating.wins}W-{rating.losses}L-{rating.draws}D ({rating.matches_played} matches)")
    
    # Predict win probability
    print("\nWin probability predictions:")
    prob = rating_system.predict_win_probability("agent_0001_demo", "agent_0002_demo")
    print(f"  agent_0001_demo vs agent_0002_demo: {prob:.1%}")
    
    # Overall statistics
    stats = rating_system.get_statistics()
    print(f"\nRating system statistics:")
    print(f"  Total agents: {stats['total_agents']}")
    print(f"  Total matches: {stats['total_matches']}")
    print(f"  Average rating: {stats['avg_rating']:.1f}")
    print(f"  Rating range: {stats['min_rating']:.1f} - {stats['max_rating']:.1f}")


def example_full_workflow():
    """Example 3: Full workflow with actual agents."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Full Workflow (Requires Trained Agents)")
    print("=" * 70)
    print("\nThis example requires at least 2 trained agents in the archive.")
    print("To use this:")
    print("  1. Train some agents using train_single_run.py")
    print("  2. Add them to archive using archive.add_agent()")
    print("  3. Run head-to-head evaluation")
    print("  4. Update ratings based on results")
    
    # Example workflow (commented out - uncomment when you have agents):
    """
    # Step 1: Initialize systems
    archive = AgentArchiveManager()
    rating_system = AgentRatingSystem()
    
    # Step 2: Add agents to archive
    agent1_id = archive.add_agent(
        checkpoint_path="path/to/agent1_checkpoint.pt",
        config_path="path/to/agent1_config.json",
        metadata={"training_info": {"algorithm": "DDDQN", "episodes": 5000}},
        agent_name="dddqn_5k"
    )
    
    agent2_id = archive.add_agent(
        checkpoint_path="path/to/agent2_checkpoint.pt",
        config_path="path/to/agent2_config.json",
        metadata={"training_info": {"algorithm": "DDDQN", "episodes": 10000}},
        agent_name="dddqn_10k"
    )
    
    # Step 3: Initialize ratings
    rating_system.initialize_agent(agent1_id)
    rating_system.initialize_agent(agent2_id)
    
    # Step 4: Run head-to-head evaluation
    agent1_checkpoint = archive.get_agent_checkpoint_path(agent1_id)
    agent2_checkpoint = archive.get_agent_checkpoint_path(agent2_id)
    
    results = evaluate_head_to_head(
        agent1_path=agent1_checkpoint,
        agent2_path=agent2_checkpoint,
        num_games=20,
        max_steps=250,
    )
    
    print(f"Match results: {agent1_id} vs {agent2_id}")
    print(f"  Agent 1 wins: {results['agent1_wins']}")
    print(f"  Agent 2 wins: {results['agent2_wins']}")
    print(f"  Draws: {results['draws']}")
    print(f"  Agent 1 win rate: {results['agent1_win_rate']:.1%}")
    
    # Step 5: Update ratings based on overall result
    if results['agent1_wins'] > results['agent2_wins']:
        match_result = 1
    elif results['agent1_wins'] < results['agent2_wins']:
        match_result = -1
    else:
        match_result = 0
    
    rating_system.update_ratings(agent1_id, agent2_id, match_result)
    
    # Step 6: View updated ratings
    print(f"\nUpdated ratings:")
    for agent_id, rating in rating_system.get_leaderboard():
        print(f"  {agent_id}: {rating.rating:.1f}")
    """


if __name__ == "__main__":
    # Run examples
    example_basic_archive()
    example_rating_system()
    example_full_workflow()
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
