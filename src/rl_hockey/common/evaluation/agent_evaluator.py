import multiprocessing as mp
import numpy as np
from typing import Dict, Any, Tuple
import hockey.hockey_env as h_env

from rl_hockey.common.training.agent_factory import create_agent, get_action_space_info
from rl_hockey.common.training.curriculum_manager import AgentConfig, load_curriculum

def run_single_game(args: Tuple) -> Dict[str, Any]:
    agent_path, agent_config_dict, weak_opponent, max_steps, seed = args
    np.random.seed(seed)
    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
    state_dim, action_dim, is_discrete = get_action_space_info(env, agent_config_dict['type'])
    from rl_hockey.common.training.curriculum_manager import AgentConfig
    agent_config = AgentConfig(
        type=agent_config_dict['type'],
        hyperparameters=agent_config_dict['hyperparameters']
    )
    agent = create_agent(agent_config, state_dim, action_dim, is_discrete, {})
    agent.load(agent_path)
    if hasattr(agent, 'q_network'):
        agent.q_network.eval()
    if hasattr(agent, 'q_network_target'):
        agent.q_network_target.eval()
    if hasattr(agent, 'actor') and hasattr(agent.actor, 'eval'):
        agent.actor.eval()
    if hasattr(agent, 'critic1') and hasattr(agent.critic1, 'eval'):
        agent.critic1.eval()
    opponent = h_env.BasicOpponent(weak=weak_opponent)
    state, _ = env.reset()
    obs_agent2 = env.obs_agent_two()
    total_reward = 0
    for step in range(max_steps):
        if is_discrete:
            discrete_action = agent.act(state.astype(np.float32), deterministic=True)
            action_p1 = env.discrete_to_continous_action(discrete_action)
        else:
            action_p1 = agent.act(state.astype(np.float32), deterministic=True)
        action_p2 = opponent.act(obs_agent2)
        action = np.hstack([action_p1, action_p2])
        next_state, reward, done, trunc, info = env.step(action)
        total_reward += reward
        state = next_state
        obs_agent2 = env.obs_agent_two()
        if done or trunc:
            break
    winner = info.get('winner', 0)
    env.close()
    return {
        'winner': winner,
        'reward': total_reward,
        'steps': step + 1
    }

def evaluate_agent(
    agent_path: str,
    config_path: str = None,
    agent_config_dict: Dict[str, Any] = None,
    num_games: int = 100,
    weak_opponent: bool = True,
    max_steps: int = 250,
    num_parallel: int = None
) -> Dict[str, Any]:
    if config_path is not None:
        curriculum = load_curriculum(config_path)
        agent_config_dict = {
            'type': curriculum.agent.type,
            'hyperparameters': curriculum.agent.hyperparameters
        }
    elif agent_config_dict is None:
        raise ValueError("Either config_path or agent_config_dict must be provided")
    if num_parallel is None:
        num_parallel = min(mp.cpu_count(), num_games)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    seeds = np.random.randint(0, 2**31, size=num_games)
    args_list = [
        (agent_path, agent_config_dict, weak_opponent, max_steps, int(seed))
        for seed in seeds
    ]
    results = []
    if num_parallel > 1:
        with mp.Pool(processes=num_parallel) as pool:
            results = pool.map(run_single_game, args_list)
    else:
        results = [run_single_game(args) for args in args_list]
    wins = sum(1 for r in results if r['winner'] == 1)
    losses = sum(1 for r in results if r['winner'] == -1)
    draws = sum(1 for r in results if r['winner'] == 0)
    rewards = [r['reward'] for r in results]
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    win_rate = wins / num_games if num_games > 0 else 0.0
    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'num_games': num_games,
        'weak_opponent': weak_opponent,
        'all_rewards': rewards,
        'all_winners': [r['winner'] for r in results]
    }

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    print(evaluate_agent(agent_path="results/hyperparameter_runs/2026-01-02_13-12-27/models/run_lr1e04_bs256_h128_128_128_7703d10e_20260102_131227_ep009100.pt", config_path="configs/curriculum_simple.json", num_games=100, weak_opponent=True, max_steps=250, num_parallel=None))