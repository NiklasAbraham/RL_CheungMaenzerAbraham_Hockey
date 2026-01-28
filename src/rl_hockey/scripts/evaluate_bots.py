"""
Simple script to evaluate weak bot vs strong bot using TrueSkill ratings.
"""

import numpy as np
import trueskill
import hockey.hockey_env as h_env


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
    # Setup TrueSkill environment
    ts_env = trueskill.TrueSkill(mu=25.0, sigma=8.333, beta=4.166, tau=0.083, draw_probability=0.15)
    
    # Initialize ratings (weak bot starts at 100 as requested)
    weak_rating = ts_env.create_rating()  # Conservative estimate = 100
    strong_rating = ts_env.create_rating()  # Default: mu=25, sigma=8.333
    
    print("="*60)
    print("Bot Evaluation: Weak Bot vs Strong Bot")
    print("="*60)
    print(f"\nInitial Ratings:")
    print(f"  Weak Bot:   μ={weak_rating.mu:.2f}, σ={weak_rating.sigma:.2f}, Rating={weak_rating.mu - 3*weak_rating.sigma:.2f}")
    print(f"  Strong Bot: μ={strong_rating.mu:.2f}, σ={strong_rating.sigma:.2f}, Rating={strong_rating.mu - 3*strong_rating.sigma:.2f}")
    
    # Create environment and bots
    env = h_env.HockeyEnv()
    weak_bot = h_env.BasicOpponent(weak=True)
    strong_bot = h_env.BasicOpponent(weak=False)
    
    # Play matches
    num_matches = 1000
    results = []
    
    print(f"\nPlaying {num_matches} matches...\n")
    
    for i in range(num_matches):
        result = play_match(env, weak_bot, strong_bot, max_steps=250)
        results.append(result)
        
        # Update ratings
        if result == 1:  # Weak wins
            weak_rating, strong_rating = ts_env.rate_1vs1(weak_rating, strong_rating)
        elif result == -1:  # Strong wins
            strong_rating, weak_rating = ts_env.rate_1vs1(strong_rating, weak_rating)
        else:  # Draw
            weak_rating, strong_rating = ts_env.rate_1vs1(weak_rating, strong_rating, drawn=True)
        
        # Print match result
        result_str = "Weak Win" if result == 1 else ("Strong Win" if result == -1 else "Draw")
        print(f"Match {i+1:2d}: {result_str:12s} | Weak: {weak_rating.mu - 3*weak_rating.sigma:6.2f}, Strong: {strong_rating.mu - 3*strong_rating.sigma:6.2f}")
    
    env.close()
    
    # Print summary
    wins_weak = sum(1 for r in results if r == 1)
    wins_strong = sum(1 for r in results if r == -1)
    draws = sum(1 for r in results if r == 0)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\nMatches Played: {num_matches}")
    print(f"  Weak Bot Wins:   {wins_weak:2d} ({wins_weak/num_matches*100:.1f}%)")
    print(f"  Strong Bot Wins: {wins_strong:2d} ({wins_strong/num_matches*100:.1f}%)")
    print(f"  Draws:           {draws:2d} ({draws/num_matches*100:.1f}%)")
    
    print(f"\nFinal Ratings:")
    print(f"  Weak Bot:   μ={weak_rating.mu:.2f}, σ={weak_rating.sigma:.2f}, Rating={weak_rating.mu - 3*weak_rating.sigma:.2f}")
    print(f"  Strong Bot: μ={strong_rating.mu:.2f}, σ={strong_rating.sigma:.2f}, Rating={strong_rating.mu - 3*strong_rating.sigma:.2f}")
    
    print(f"\nRating Change:")
    print(f"  Weak Bot:   {(weak_rating.mu - 3*weak_rating.sigma) - 100.0:+.2f}")
    print(f"  Strong Bot: {(strong_rating.mu - 3*strong_rating.sigma) - 0.0:+.2f}")
    print("="*60)


if __name__ == "__main__":
    main()
