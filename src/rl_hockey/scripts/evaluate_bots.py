"""
Evaluate a SAC agent against BasicOpponent using TrueSkill ratings.
"""

import argparse
import numpy as np
import trueskill
import hockey.hockey_env as h_env
from rl_hockey.sac import SAC


def play_match(env, player1, player2, max_steps=250, render=False):
    """
    Play a single match between two players.
    
    Returns:
        result: 1 if player1 wins, -1 if player2 wins, 0 if draw
    """
    obs, _ = env.reset()
    
    for _ in range(max_steps):
        # Get actions
        obs_player1 = obs
        action1 = player1.act(obs_player1)
        
        obs_player2 = env.obs_agent_two()
        action2 = player2.act(obs_player2)
        
        # Step environment
        full_action = np.hstack([action1, action2])
        obs, reward, done, trunc, info = env.step(full_action)
        
        if render:
            env.render()
        
        if done or trunc:
            break
    
    # Determine winner
    winner = info.get("winner", 0)
    
    if winner == 1:
        return 1  # Player 1 wins
    elif winner == -1:
        return -1  # Player 2 wins
    else:
        return 0  # Draw


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAC agent against BasicOpponent")
    parser.add_argument("checkpoint", type=str, help="Path to SAC checkpoint file")
    parser.add_argument("--opponent", type=str, choices=["weak", "strong"], default="weak",
                        help="Opponent strength (default: weak)")
    parser.add_argument("--matches", type=int, default=10, help="Number of matches to play (default: 10)")
    parser.add_argument("--render", action="store_true", help="Render matches")
    args = parser.parse_args()
    
    # Setup TrueSkill environment
    ts_env = trueskill.TrueSkill(mu=25.0, sigma=8.333, beta=4.166, tau=0.083, draw_probability=0.15)
    
    # Initialize ratings
    # Agent starts at specified rating
    agent_rating = ts_env.create_rating()
    opponent_rating = ts_env.create_rating()
    
    # Create environment
    env = h_env.HockeyEnv()
    
    # Load SAC agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] // 2
    agent = SAC(state_dim=state_dim, action_dim=action_dim)
    
    print(f"Loading agent from: {args.checkpoint}")
    agent.load(args.checkpoint)
    print("Agent loaded successfully!")
    
    # Create opponent
    opponent = h_env.BasicOpponent(weak=(args.opponent == "weak"))
    opponent_name = "Weak Bot" if args.opponent == "weak" else "Strong Bot"
    
    print("="*60)
    print(f"Evaluation: SAC Agent vs {opponent_name}")
    print("="*60)
    print(f"\nInitial Ratings:")
    print(f"  Agent:    μ={agent_rating.mu:.2f}, σ={agent_rating.sigma:.2f}, Rating={agent_rating.mu - 3*agent_rating.sigma:.2f}")
    print(f"  Opponent: μ={opponent_rating.mu:.2f}, σ={opponent_rating.sigma:.2f}, Rating={opponent_rating.mu - 3*opponent_rating.sigma:.2f}")
    
    # Play matches
    num_matches = args.matches
    results = []
    
    print(f"\nPlaying {num_matches} matches...\n")
    
    for i in range(num_matches):
        # Create a wrapper for the agent that uses the agent.act method
        class AgentWrapper:
            def __init__(self, sac_agent):
                self.agent = sac_agent
            
            def act(self, obs):
                return self.agent.act(obs.astype(np.float32))
        
        agent_wrapper = AgentWrapper(agent)
        result = play_match(env, agent_wrapper, opponent, max_steps=250, render=args.render)
        results.append(result)
        
        # Update ratings
        if result == 1:  # Agent wins
            agent_rating, opponent_rating = ts_env.rate_1vs1(agent_rating, opponent_rating)
        elif result == -1:  # Opponent wins
            opponent_rating, agent_rating = ts_env.rate_1vs1(opponent_rating, agent_rating)
        else:  # Draw
            agent_rating, opponent_rating = ts_env.rate_1vs1(agent_rating, opponent_rating, drawn=True)
        
        # Print match result
        result_str = "Agent Win" if result == 1 else ("Opponent Win" if result == -1 else "Draw")
        print(f"Match {i+1:2d}: {result_str:12s} | Agent: {agent_rating.mu - 3*agent_rating.sigma:6.2f}, Opponent: {opponent_rating.mu - 3*opponent_rating.sigma:6.2f}")
    
    env.close()
    
    # Print summary
    wins_agent = sum(1 for r in results if r == 1)
    wins_opponent = sum(1 for r in results if r == -1)
    draws = sum(1 for r in results if r == 0)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\nMatches Played: {num_matches}")
    print(f"  Agent Wins:    {wins_agent:2d} ({wins_agent/num_matches*100:.1f}%)")
    print(f"  Opponent Wins: {wins_opponent:2d} ({wins_opponent/num_matches*100:.1f}%)")
    print(f"  Draws:         {draws:2d} ({draws/num_matches*100:.1f}%)")
    
    print(f"\nFinal Ratings:")
    print(f"  Agent:    μ={agent_rating.mu:.2f}, σ={agent_rating.sigma:.2f}, Rating={agent_rating.mu - 3*agent_rating.sigma:.2f}")
    print(f"  Opponent: μ={opponent_rating.mu:.2f}, σ={opponent_rating.sigma:.2f}, Rating={opponent_rating.mu - 3*opponent_rating.sigma:.2f}")
    
    print(f"\nRating Change:")
    print(f"  Agent:    {(agent_rating.mu - 3*agent_rating.sigma):+.2f}")
    print(f"  Opponent: {(opponent_rating.mu - 3*opponent_rating.sigma):+.2f}")
    print("="*60)


if __name__ == "__main__":
    main()
