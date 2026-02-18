import numpy as np
from typing import Tuple, Optional
import hockey.hockey_env as h_env
from rl_hockey.common.agent import Agent


def evaluate_winrate(
    agent: Agent,
    num_episodes: int = 100,
    max_steps: int = 250,
    opponent_weak: bool = True,
    deterministic: bool = True,
    verbose: bool = False,
) -> Tuple[float, float]:
    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
    opponent = h_env.BasicOpponent(weak=opponent_weak)
    
    win_count = 0
    for i in range(num_episodes):
        state, _ = env.reset()
        
        for t in range(max_steps):
            action1 = agent.act(state.astype(np.float32), deterministic=deterministic)
            action2 = opponent.act(env.obs_agent_two())
            next_state, reward, done, trunc, info = env.step(np.hstack([action1, action2]))
            state = next_state
            
            if done or trunc:
                break
        
        # Check if agent won
        if info.get('winner') == 1:
            win_count += 1
    
    env.close()
    
    winrate = win_count / num_episodes
    
    if verbose:
        print(f"Evaluation agaist {'weak' if opponent_weak else 'strong'} opponent: Winrate={winrate:.2%} ({num_episodes} episodes)")
    
    return winrate
