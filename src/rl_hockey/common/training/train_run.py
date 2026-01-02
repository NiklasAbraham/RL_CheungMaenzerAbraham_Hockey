"""
Main training function with curriculum learning support.
"""
import os
import numpy as np
from tqdm import tqdm
import hockey.hockey_env as h_env

from rl_hockey.common import utils
from rl_hockey.common.training.run_manager import RunManager
from rl_hockey.common.training.curriculum_manager import (
    load_curriculum, get_phase_for_episode, get_total_episodes, CurriculumConfig
)
from rl_hockey.common.training.agent_factory import create_agent, get_action_space_info
from rl_hockey.common.training.opponent_manager import (
    sample_opponent, get_opponent_action
)
from rl_hockey.common.training.config_validator import validate_config
from rl_hockey.common.evaluation.agent_evaluator import evaluate_agent


def train_run(
    config_path: str,
    base_output_dir: str = "results/runs",
    run_name: str = None,
    verbose: bool = True,
    eval_freq_steps: int = None,
    eval_num_games: int = 100,
    eval_weak_opponent: bool = True
):
    """Train an agent using curriculum learning."""
    # Validate config
    errors = validate_config(config_path)
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    # Load curriculum
    curriculum = load_curriculum(config_path)
    
    # Initialize run manager
    run_manager = RunManager(base_output_dir=base_output_dir)
    if run_name is None:
        # Generate run name from config
        config_dict = _curriculum_to_dict(curriculum)
        run_name = run_manager.generate_run_name(config_dict)
    
    # Get training parameters
    training_params = curriculum.training
    max_episode_steps = training_params.get('max_episode_steps', 500)
    updates_per_step = training_params.get('updates_per_step', 1)
    warmup_steps = training_params.get('warmup_steps', 400)
    reward_scale = training_params.get('reward_scale', 0.1)
    checkpoint_save_freq = training_params.get('checkpoint_save_freq', 100)
    
    # Get total episodes
    total_episodes = get_total_episodes(curriculum)
    
    # Initialize environment (will be recreated per phase)
    current_env = None
    current_phase_idx = -1
    current_opponent = None
    
    # Create initial environment to get action space info
    first_phase = curriculum.phases[0]
    env_mode = getattr(h_env.Mode, first_phase.environment.mode)
    current_env = h_env.HockeyEnv(mode=env_mode, keep_mode=first_phase.environment.keep_mode)
    state_dim, agent_action_dim, is_agent_discrete = get_action_space_info(current_env, curriculum.agent.type)
    
    # Create agent
    agent = create_agent(
        curriculum.agent,
        state_dim,
        agent_action_dim,
        is_agent_discrete,
        curriculum.hyperparameters
    )
    
    # Training state
    losses = []
    rewards = []
    phases = []  # Track phase for each episode
    steps = 0
    gradient_steps = 0
    
    # Evaluation state
    evaluation_results = []  # List of dicts with step, win_rate, mean_reward, etc.
    last_eval_step = 0
    
    # Get reward weights function
    def get_reward_weights(episode_idx, phase_config):
        """Get reward weights for current episode and phase.
        
        Returns None if reward shaping is disabled, otherwise returns dict with weights.
        """
        reward_shaping = phase_config.reward_shaping
        
        # If reward shaping is None, return None (no reward shaping)
        if reward_shaping is None:
            return None
        
        N = reward_shaping.N
        K = reward_shaping.K
        CLOSENESS_START = reward_shaping.CLOSENESS_START
        TOUCH_START = reward_shaping.TOUCH_START
        CLOSENESS_FINAL = reward_shaping.CLOSENESS_FINAL
        TOUCH_FINAL = reward_shaping.TOUCH_FINAL
        DIRECTION_FINAL = reward_shaping.DIRECTION_FINAL
        
        if episode_idx < N:
            return {
                'closeness': CLOSENESS_START,
                'touch': TOUCH_START,
                'direction': 0.0,
            }
        elif episode_idx < N + K:
            alpha = (episode_idx - N) / K
            return {
                'closeness': CLOSENESS_START * (1 - alpha) + CLOSENESS_FINAL * alpha,
                'touch': TOUCH_START * (1 - alpha) + TOUCH_FINAL * alpha,
                'direction': DIRECTION_FINAL * alpha,
            }
        else:
            return {
                'closeness': CLOSENESS_FINAL,
                'touch': TOUCH_FINAL,
                'direction': DIRECTION_FINAL,
            }
    
    pbar = tqdm(range(total_episodes), desc=run_name, disable=not verbose)
    
    for global_episode in pbar:
            # Get current phase
            phase_idx, phase_local_episode, phase_config = get_phase_for_episode(curriculum, global_episode)
            
            # Check if phase changed
            if phase_idx != current_phase_idx:
                if verbose:
                    print(f"\nTransitioning to phase {phase_idx + 1}/{len(curriculum.phases)}: {phase_config.name}")
                
                # Recreate environment if mode or keep_mode changed
                env_mode = getattr(h_env.Mode, phase_config.environment.mode)
                if current_env is None or \
                   current_env.mode != env_mode or \
                   current_env.keep_mode != phase_config.environment.keep_mode:
                    if current_env is not None:
                        current_env.close()
                    current_env = h_env.HockeyEnv(
                        mode=env_mode,
                        keep_mode=phase_config.environment.keep_mode
                    )
                
                # Create/sample opponent for this phase
                current_opponent = sample_opponent(
                    phase_config.opponent,
                    agent=agent,
                    checkpoint_dir=str(run_manager.models_dir),
                    agent_config=curriculum.agent,
                    state_dim=state_dim,
                    action_dim=agent_action_dim,
                    is_discrete=is_agent_discrete
                )
                
                # Clear replay buffer on phase transition to avoid distribution mismatch
                # Old transitions from previous phases can cause Q-value instability
                if current_phase_idx >= 0:  # Don't clear on first phase
                    if verbose:
                        print(f"  Clearing replay buffer (size: {agent.buffer.size}) to avoid distribution mismatch")
                    agent.buffer.clear()
                
                current_phase_idx = phase_idx
            
            # Reset environment
            state, _ = current_env.reset()
            
            # Get reward weights for this phase
            reward_weights = get_reward_weights(phase_local_episode, phase_config)
            
            # Episode training
            agent.on_episode_start(global_episode)
            total_reward = 0
            total_shaped_reward = 0
            
            # For self-play with current agent, create fresh copy each episode
            if phase_config.opponent.type == "self_play" and phase_config.opponent.checkpoint is None:
                current_opponent = sample_opponent(
                    phase_config.opponent,
                    agent=agent,
                    checkpoint_dir=str(run_manager.models_dir),
                    agent_config=curriculum.agent,
                    state_dim=state_dim,
                    action_dim=agent_action_dim,
                    is_discrete=is_agent_discrete
                )
            
            for t in range(max_episode_steps):
                # Get agent action
                if is_agent_discrete:
                    discrete_action = agent.act(state.astype(np.float32))
                    action_p1 = current_env.discrete_to_continous_action(discrete_action)
                else:
                    action_p1 = agent.act(state.astype(np.float32))
                
                # Get opponent action
                obs_agent2 = current_env.obs_agent_two()
                deterministic_opponent = phase_config.opponent.deterministic if phase_config.opponent.type == "self_play" else True
                action_p2 = get_opponent_action(
                    current_opponent,
                    obs_agent2,
                    deterministic=deterministic_opponent
                )
                
                # Handle discrete opponent actions (from self-play DDDQN)
                if isinstance(action_p2, (int, np.integer)):
                    action_p2 = current_env.discrete_to_continous_action(action_p2)
                
                # Combine actions
                action = np.hstack([action_p1, action_p2])
                
                # Step environment
                next_state, reward, done, trunc, info = current_env.step(action)
                
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
                
                # Store transition
                if is_agent_discrete:
                    discrete_action_array = np.array([discrete_action], dtype=np.float32)
                    agent.store_transition((state, discrete_action_array, scaled_reward, next_state, done))
                    
                    # Store mirrored transition
                    mirrored_action = utils.mirror_discrete_action(discrete_action)
                    mirrored_action_array = np.array([mirrored_action], dtype=np.float32)
                    agent.store_transition((
                        utils.mirror_state(state),
                        mirrored_action_array,
                        scaled_reward,
                        utils.mirror_state(next_state),
                        done
                    ))
                else:
                    action_array = action_p1.astype(np.float32)
                    agent.store_transition((state, action_array, scaled_reward, next_state, done))
                
                state = next_state
                steps += 1
                total_reward += reward
                total_shaped_reward += shaped_reward
                
                # Train agent
                if steps >= warmup_steps:
                    stats = agent.train(updates_per_step)
                    gradient_steps += updates_per_step
                    if isinstance(stats, dict):
                        # Collect losses from various possible keys
                        loss_list = None
                        if 'loss' in stats and stats['loss'] is not None:
                            loss_list = stats['loss']
                        elif 'critic_loss' in stats and stats['critic_loss'] is not None:
                            loss_list = stats['critic_loss']
                        elif 'actor_loss' in stats and stats['actor_loss'] is not None:
                            loss_list = stats['actor_loss']
                        
                        # Extend losses list if we got valid losses
                        if loss_list is not None:
                            if isinstance(loss_list, list):
                                losses.extend(loss_list)
                            else:
                                # Single value, convert to list
                                losses.append(loss_list)
                
                # Periodic evaluation
                if eval_freq_steps is not None and steps - last_eval_step >= eval_freq_steps:
                    if verbose:
                        print(f"\nEvaluating agent at step {steps}...")
                    
                    # Save temporary checkpoint for evaluation
                    temp_checkpoint_path = run_manager.models_dir / f"{run_name}_temp_eval.pt"
                    agent.save(str(temp_checkpoint_path))
                    
                    # Prepare agent config dict for evaluation
                    agent_config_dict = {
                        'type': curriculum.agent.type,
                        'hyperparameters': curriculum.agent.hyperparameters
                    }
                    
                    try:
                        # Evaluate agent
                        eval_results = evaluate_agent(
                            agent_path=str(temp_checkpoint_path),
                            agent_config_dict=agent_config_dict,
                            num_games=eval_num_games,
                            weak_opponent=eval_weak_opponent,
                            max_steps=max_episode_steps,
                            num_parallel=None
                        )
                        
                        # Store evaluation results
                        evaluation_results.append({
                            'step': steps,
                            'episode': global_episode,
                            'win_rate': eval_results['win_rate'],
                            'mean_reward': eval_results['mean_reward'],
                            'std_reward': eval_results['std_reward'],
                            'wins': eval_results['wins'],
                            'losses': eval_results['losses'],
                            'draws': eval_results['draws']
                        })
                        
                        # Update plots with current evaluation results
                        if evaluation_results:
                            run_manager.save_evaluation_plot(run_name, evaluation_results)
                            run_manager.save_evaluation_csv(run_name, evaluation_results)
                        
                        if verbose:
                            print(f"Evaluation results: Win rate: {eval_results['win_rate']:.2%}, "
                                  f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
                    finally:
                        # Clean up temporary checkpoint
                        if temp_checkpoint_path.exists():
                            os.remove(temp_checkpoint_path)
                    
                    last_eval_step = steps
                
                if done or trunc:
                    break
            
            agent.on_episode_end(global_episode)
            rewards.append(total_reward)
            phases.append(phase_config.name)
            
            # Save checkpoint periodically
            if (global_episode + 1) % checkpoint_save_freq == 0:
                run_manager.save_checkpoint(
                    run_name, global_episode + 1, agent,
                    phase_index=phase_idx,
                    phase_episode=phase_local_episode
                )
            
            # Update progress bar
            if verbose:
                pbar.set_postfix({
                    'reward': total_reward,
                    'shaped': f'{total_shaped_reward:.1f}',
                    'phase': phase_config.name[:10],
                    'eps': agent.config.get('eps', 'N/A') if hasattr(agent, 'config') else 'N/A',
                })
    
    # Save final results
    run_manager.save_config(run_name, _curriculum_to_dict(curriculum))
    run_manager.save_rewards_csv(run_name, rewards, phases=phases)
    run_manager.save_losses_csv(run_name, losses)
    
    # Debug output
    if verbose:
        print(f"\nTraining Summary:")
        print(f"  Total episodes: {len(rewards)}")
        print(f"  Total steps: {steps}")
        print(f"  Total gradient steps: {gradient_steps}")
        print(f"  Losses collected: {len(losses)}")
        if losses:
            print(f"  Mean loss: {sum(losses)/len(losses):.4f}")
        else:
            print(f"  Warning: No losses collected!")
            print(f"    This might mean training hasn't started (warmup_steps={warmup_steps})")
            print(f"    or the replay buffer is too small (current size: {agent.buffer.size})")
    
    run_manager.save_plots(run_name, rewards, losses)
    
    # Save final evaluation results if any evaluations were performed
    if evaluation_results:
        run_manager.save_evaluation_plot(run_name, evaluation_results)
        run_manager.save_evaluation_csv(run_name, evaluation_results)
    
    # Save final model
    model_path = run_manager.get_model_path(run_name)
    agent.save(str(model_path))
    
    current_env.close()
    
    return {
        'run_name': run_name,
        'final_reward': rewards[-1] if rewards else 0,
        'mean_reward': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards) if rewards else 0,
        'total_episodes': len(rewards),
        'total_steps': steps,
        'total_gradient_steps': gradient_steps,
        'evaluation_results': evaluation_results,
    }


def _curriculum_to_dict(curriculum: CurriculumConfig) -> dict:
    """Convert curriculum config to dictionary for saving."""
    return {
        'curriculum': {
            'phases': [
                {
                    'name': phase.name,
                    'episodes': phase.episodes,
                    'environment': {
                        'mode': phase.environment.mode,
                        'keep_mode': phase.environment.keep_mode,
                    },
                    'opponent': {
                        'type': phase.opponent.type,
                        'weight': phase.opponent.weight,
                        'checkpoint': phase.opponent.checkpoint,
                        'deterministic': phase.opponent.deterministic,
                        'opponents': phase.opponent.opponents,
                    },
                    'reward_shaping': None if phase.reward_shaping is None else {
                        'N': phase.reward_shaping.N,
                        'K': phase.reward_shaping.K,
                        'CLOSENESS_START': phase.reward_shaping.CLOSENESS_START,
                        'TOUCH_START': phase.reward_shaping.TOUCH_START,
                        'CLOSENESS_FINAL': phase.reward_shaping.CLOSENESS_FINAL,
                        'TOUCH_FINAL': phase.reward_shaping.TOUCH_FINAL,
                        'DIRECTION_FINAL': phase.reward_shaping.DIRECTION_FINAL,
                    }
                }
                for phase in curriculum.phases
            ]
        },
        'hyperparameters': curriculum.hyperparameters,
        'training': curriculum.training,
        'agent': {
            'type': curriculum.agent.type,
            'hyperparameters': curriculum.agent.hyperparameters,
        }
    }
