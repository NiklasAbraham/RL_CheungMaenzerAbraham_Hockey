import json
import numpy as np
import hockey.hockey_env as h_env
from rl_hockey.REDQ.redq_td3 import REDQTD3


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
        action1 = player1.act(obs_player1, deterministic=True)
        
        # Bot opponent
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
    # Create environment
    env = h_env.HockeyEnv()

    config = json.load(open("models/redqtd3/redqtd3.json", "r"))
    model_path = "models/redqtd3/redqtd3.pt"
    
    # Load REDQTD3 agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] // 2
    agent = REDQTD3(state_dim=state_dim, action_dim=action_dim, **config)
    agent.load(model_path)

    print("Loaded REDQTD3 agent")
    print(agent.log_architecture())

    agent.actor.eval()
    for critic in agent.critic_net_list:
        critic.eval()
    
    # Create opponent
    weak_opponent = h_env.BasicOpponent(weak=True)
    strong_opponent = h_env.BasicOpponent(weak=False)
    
    # Play matches
    num_matches = 500
    weak_results = []
    strong_results = []
    
    print(f"\nPlaying {num_matches} matches...\n")
    
    for i in range(num_matches):        
        weak_result = play_match(env, agent, weak_opponent, max_steps=500, render=False)
        strong_result = play_match(env, agent, strong_opponent, max_steps=500, render=False)
        weak_results.append(weak_result)
        strong_results.append(strong_result)
        
        # # Print match result
        # weak_result_str = "Agent Win" if weak_result == 1 else ("Opponent Win" if weak_result == -1 else "Draw")
        # strong_result_str = "Agent Win" if strong_result == 1 else ("Opponent Win" if strong_result == -1 else "Draw")
        # print(f"Match {i+1:2d}: Weak Opponent: {weak_result_str:12s} | Strong Opponent: {strong_result_str:12s}")
    
    env.close()
    
    # Print summary
    weak_wins_agent = sum(1 for r in weak_results if r == 1)
    weak_wins_opponent = sum(1 for r in weak_results if r == -1)
    weak_draws = sum(1 for r in weak_results if r == 0)

    strong_wins_agent = sum(1 for r in strong_results if r == 1)
    strong_wins_opponent = sum(1 for r in strong_results if r == -1)
    strong_draws = sum(1 for r in strong_results if r == 0)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\nMatches Played: {num_matches}")
    print(f"  Agent Wins (Weak):    {weak_wins_agent:2d} ({weak_wins_agent/num_matches*100:.1f}%)")
    print(f"  Opponent Wins (Weak): {weak_wins_opponent:2d} ({weak_wins_opponent/num_matches*100:.1f}%)")
    print(f"  Draws (Weak):         {weak_draws:2d} ({weak_draws/num_matches*100:.1f}%)")
    print(f"  Agent Wins (Strong):    {strong_wins_agent:2d} ({strong_wins_agent/num_matches*100:.1f}%)")
    print(f"  Opponent Wins (Strong): {strong_wins_opponent:2d} ({strong_wins_opponent/num_matches*100:.1f}%)")
    print(f"  Draws (Strong):         {strong_draws:2d} ({strong_draws/num_matches*100:.1f}%)")


if __name__ == "__main__":
    main()
