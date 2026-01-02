import numpy as np
from tqdm import tqdm
import hockey.hockey_env as h_env

from rl_hockey.DDDQN import DDDQN
from rl_hockey.common import utils
from rl_hockey.common.training.run_manager import RunManager


def train_single_phase(config: dict, base_output_dir: str = "results/runs", run_name: str = None, 
                       verbose: bool = True):
    """Train a single phase with configurable environment mode and opponent."""
    run_manager = RunManager(base_output_dir=base_output_dir)
    if run_name is None:
        run_name = run_manager.generate_run_name(config)

    max_episodes = config.get('max_episodes', 3000)
    max_episode_steps = config.get('max_episode_steps', 500)
    updates_per_step = config.get('updates_per_step', 1)
    warmup_steps = config.get('warmup_steps', 400)
    reward_scale = config.get('reward_scale', 0.1)

    # Environment config
    env_mode_str = config.get('mode', 'TRAIN_SHOOTING')
    env_mode = getattr(h_env.Mode, env_mode_str)
    keep_mode = config.get('keep_mode', True)

    # Opponent config
    opponent_type = config.get('opponent', 'none')
    opponent = None
    if opponent_type == 'basic_weak':
        opponent = h_env.BasicOpponent(weak=True)
    elif opponent_type == 'basic_strong':
        opponent = h_env.BasicOpponent(weak=False)

    # Reward shaping (optional - can be None)
    reward_shaping = config.get('reward_shaping')
    if reward_shaping is None:
        N = K = CLOSENESS_START = TOUCH_START = CLOSENESS_FINAL = TOUCH_FINAL = DIRECTION_FINAL = None
    else:
        N = reward_shaping.get('N', 600)
        K = reward_shaping.get('K', 400)
        CLOSENESS_START = reward_shaping.get('CLOSENESS_START', 20.0)
        TOUCH_START = reward_shaping.get('TOUCH_START', 15.0)
        CLOSENESS_FINAL = reward_shaping.get('CLOSENESS_FINAL', 1.5)
        TOUCH_FINAL = reward_shaping.get('TOUCH_FINAL', 1.0)
        DIRECTION_FINAL = reward_shaping.get('DIRECTION_FINAL', 2.0)

    # Agent config
    hidden_dim = config.get('hidden_dim', [256, 256, 256])
    agent_config = {
        'target_update_freq': config.get('target_update_freq', 50),
        'batch_size': config.get('batch_size', 256),
        'learning_rate': config.get('learning_rate', 0.0001),
        'eps': config.get('eps', 0.3),
        'eps_min': config.get('eps_min', 0.05),
        'eps_decay': config.get('eps_decay', 0.999),
        'use_huber_loss': config.get('use_huber_loss', True),
    }

    env = h_env.HockeyEnv(mode=env_mode, keep_mode=keep_mode)
    state_dim = env.observation_space.shape[0]
    action_dim = 7 if not env.keep_mode else 8

    agent = DDDQN(state_dim, action_dim=action_dim, hidden_dim=hidden_dim, **agent_config)

    def get_reward_weights(episode_idx):
        """Get reward weights for current episode.
        
        Returns None if reward shaping is disabled, otherwise returns dict with weights.
        """
        if reward_shaping is None:
            return None
        
        if episode_idx < N:
            return {'closeness': CLOSENESS_START, 'touch': TOUCH_START, 'direction': 0.0}
        elif episode_idx < N + K:
            alpha = (episode_idx - N) / K
            return {
                'closeness': CLOSENESS_START * (1 - alpha) + CLOSENESS_FINAL * alpha,
                'touch': TOUCH_START * (1 - alpha) + TOUCH_FINAL * alpha,
                'direction': DIRECTION_FINAL * alpha,
            }
        else:
            return {'closeness': CLOSENESS_FINAL, 'touch': TOUCH_FINAL, 'direction': DIRECTION_FINAL}

    losses = []
    rewards = []
    steps = 0
    gradient_steps = 0

    pbar = tqdm(range(max_episodes), desc=run_name, disable=not verbose)

    for i in pbar:
        total_reward = 0
        total_shaped_reward = 0
        state, _ = env.reset()
        agent.on_episode_start(i)
        reward_weights = get_reward_weights(i)

        for t in range(max_episode_steps):
            discrete_action = agent.act(state.astype(np.float32))
            action_p1 = env.discrete_to_continous_action(discrete_action)
            
            if opponent:
                obs_agent2 = env.obs_agent_two()
                action_p2 = opponent.act(obs_agent2)
            else:
                action_p2 = np.zeros(len(action_p1))
            
            action = np.hstack([action_p1, action_p2])
            next_state, reward, done, trunc, info = env.step(action)

            # Apply reward shaping (if enabled)
            if reward_weights is not None:
                shaped_reward = reward
                shaped_reward += info.get('reward_closeness_to_puck', 0.0) * reward_weights['closeness']
                shaped_reward += info.get('reward_touch_puck', 0.0) * reward_weights['touch']
                shaped_reward += info.get('reward_puck_direction', 0.0) * reward_weights['direction']
            else:
                # No reward shaping - use raw reward
                shaped_reward = reward

            scaled_reward = shaped_reward * reward_scale
            discrete_action_array = np.array([discrete_action], dtype=np.float32)
            agent.store_transition((state, discrete_action_array, scaled_reward, next_state, done))

            mirrored_action = utils.mirror_discrete_action(discrete_action)
            mirrored_action_array = np.array([mirrored_action], dtype=np.float32)
            agent.store_transition((
                utils.mirror_state(state), mirrored_action_array, scaled_reward,
                utils.mirror_state(next_state), done
            ))

            state = next_state
            steps += 1
            total_reward += reward
            total_shaped_reward += shaped_reward

            if steps >= warmup_steps:
                stats = agent.train(updates_per_step)
                gradient_steps += updates_per_step
                losses.extend(stats['loss'])

            if done or trunc:
                break

        agent.on_episode_end(i)
        rewards.append(total_reward)

        if verbose:
            pbar.set_postfix({
                'reward': total_reward,
                'shaped': f'{total_shaped_reward:.1f}',
                'eps': agent.config['eps'],
            })

    run_manager.save_config(run_name, config)
    run_manager.save_rewards_csv(run_name, rewards)
    run_manager.save_losses_csv(run_name, losses)
    run_manager.save_plots(run_name, rewards, losses)

    model_path = run_manager.get_model_path(run_name)
    agent.save(str(model_path))
    env.close()

    return {
        'run_name': run_name,
        'final_reward': rewards[-1] if rewards else 0,
        'mean_reward': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards) if rewards else 0,
        'total_episodes': len(rewards),
        'total_steps': steps,
        'total_gradient_steps': gradient_steps,
    }


if __name__ == "__main__":
    config = {
        'max_episodes': 1000,
        'max_episode_steps': 500,
        'updates_per_step': 1,
        'warmup_steps': 10000,
        'reward_scale': 0.1,
        'mode': 'TRAIN_DEFENSE',
        'keep_mode': True,
        'opponent': 'basic_weak',
        'eps': 1,
        'eps_min': 0.08,
        'use_huber_loss': True,
    }
    train_single_phase(config, base_output_dir="results/runs/test")
