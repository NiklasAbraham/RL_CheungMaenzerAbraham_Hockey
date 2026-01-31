"""
Refactored training script with improved separation of concerns.
This is a cleaner version of train_run2.py with modular functions.
"""

import logging
import os
import sys
from dataclasses import dataclass
from functools import partial
from typing import Optional, Union, List, Dict, Any

import hockey.hockey_env as h_env
import numpy as np

from rl_hockey.common import utils
from rl_hockey.common.evaluation.agent_evaluator import evaluate_agent
from rl_hockey.common.evaluation.value_propagation import evaluate_episodes
from rl_hockey.common.training.agent_factory import create_agent, get_action_space_info
from rl_hockey.common.training.config_validator import validate_config
from rl_hockey.common.training.curriculum_manager import (
    CurriculumConfig,
    get_phase_for_episode,
    get_total_episodes,
    load_curriculum,
)
from rl_hockey.common.training.opponent_manager import (
    get_opponent_action,
    sample_opponent,
)
from rl_hockey.common.training.run_manager import RunManager
from rl_hockey.common.utils import (
    discrete_to_continuous_action_with_fineness,
    set_cuda_device,
    discrete_to_continuous_action_standard,
)
from rl_hockey.common.vectorized_env import (
    ThreadedVectorizedHockeyEnvOptimized,
    VectorizedHockeyEnvOptimized,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    max_episode_steps: int = 500
    updates_per_step: int = 1
    warmup_steps: int = 400
    reward_scale: float = 0.1
    checkpoint_save_freq: int = 100
    train_freq: int = 1
    
    @classmethod
    def from_curriculum(cls, curriculum: CurriculumConfig):
        """Create from curriculum config."""
        params = curriculum.training
        return cls(
            max_episode_steps=params.get("max_episode_steps", 500),
            updates_per_step=params.get("updates_per_step", 1),
            warmup_steps=params.get("warmup_steps", 400),
            reward_scale=params.get("reward_scale", 0.1),
            checkpoint_save_freq=params.get("checkpoint_save_freq", 100),
            train_freq=params.get("train_freq", 1),
        )


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    reward: float = 0.0
    shaped_reward: float = 0.0
    steps: int = 0
    losses: Dict[str, float] = None
    
    def __post_init__(self):
        if self.losses is None:
            self.losses = {}


@dataclass
class TrainingState:
    """Current state of training."""
    steps: int = 0
    gradient_steps: int = 0
    completed_episodes: int = 0
    current_phase_idx: int = -1
    current_phase_start: int = 0
    last_eval_step: int = 0
    
    # Accumulated stats
    rewards: List[float] = None
    phases: List[str] = None
    losses: List[float] = None
    episode_logs: List[Dict] = None
    evaluation_results: List[Dict] = None
    q_values: List[np.ndarray] = None  # Separate list for value propagation plotting
    
    def __post_init__(self):
        if self.rewards is None:
            self.rewards = []
        if self.phases is None:
            self.phases = []
        if self.losses is None:
            self.losses = []
        if self.episode_logs is None:
            self.episode_logs = []
        if self.evaluation_results is None:
            self.evaluation_results = []
        if self.q_values is None:
            self.q_values = []


# ============================================================================
# Reward Shaping
# ============================================================================


def calculate_reward_weights(episode_idx: int, phase_config) -> Optional[Dict[str, float]]:
    """Calculate reward shaping weights for current episode."""
    reward_shaping = phase_config.reward_shaping
    if reward_shaping is None:
        return None
    
    N = reward_shaping.N
    K = reward_shaping.K
    
    if episode_idx < N:
        return {
            "closeness": reward_shaping.CLOSENESS_START,
            "touch": reward_shaping.TOUCH_START,
            "direction": 0.0,
        }
    elif episode_idx < N + K:
        alpha = (episode_idx - N) / K
        return {
            "closeness": reward_shaping.CLOSENESS_START * (1 - alpha) + reward_shaping.CLOSENESS_FINAL * alpha,
            "touch": reward_shaping.TOUCH_START * (1 - alpha) + reward_shaping.TOUCH_FINAL * alpha,
            "direction": reward_shaping.DIRECTION_FINAL * alpha,
        }
    else:
        return {
            "closeness": reward_shaping.CLOSENESS_FINAL,
            "touch": reward_shaping.TOUCH_FINAL,
            "direction": reward_shaping.DIRECTION_FINAL,
        }


def apply_reward_shaping(reward: float, info: Dict, weights: Optional[Dict[str, float]]) -> float:
    """Apply reward shaping to environment reward."""
    if weights is None:
        return reward
    
    shaped_reward = reward
    shaped_reward += info.get("reward_closeness_to_puck", 0.0) * weights["closeness"]
    shaped_reward += info.get("reward_touch_puck", 0.0) * weights["touch"]
    shaped_reward += info.get("reward_puck_direction", 0.0) * weights["direction"]
    return shaped_reward


# ============================================================================
# Action Handling
# ============================================================================


def get_agent_action(agent, state, is_discrete: bool, action_fineness, env, keep_mode: bool):
    """Get action from agent and convert to continuous if needed."""
    if is_discrete:
        discrete_action = agent.act(state)
        if action_fineness is not None:
            action = discrete_to_continuous_action_with_fineness(
                discrete_action, fineness=action_fineness, keep_mode=keep_mode
            )
        else:
            action = env.discrete_to_continous_action(discrete_action)
        return action, discrete_action
    else:
        action = agent.act(state)
        return action, None


def convert_opponent_action(action, action_fineness, keep_mode: bool):
    """Convert opponent action to continuous if it's discrete."""
    if isinstance(action, (int, np.integer)):
        if action_fineness is not None:
            return discrete_to_continuous_action_with_fineness(
                action, fineness=action_fineness, keep_mode=keep_mode
            )
        else:
            return utils.discrete_to_continuous_action_standard(action, keep_mode=keep_mode)
    return action


# ============================================================================
# Transition Storage
# ============================================================================


def store_transition(agent, state, action, reward, next_state, done, 
                     is_discrete: bool, store_mirrored: bool = False):
    """Store transition in replay buffer."""
    if is_discrete:
        action_array = np.array([action], dtype=np.float32)
    else:
        action_array = action.astype(np.float32) if action.dtype != np.float32 else action
    
    agent.store_transition((state, action_array, reward, next_state, done))
    
    # Store mirrored transition for self-play
    if store_mirrored:
        if is_discrete:
            mirrored_action = utils.mirror_discrete_action(action)
            mirrored_action_array = np.array([mirrored_action], dtype=np.float32)
        else:
            mirrored_action_array = utils.mirror_action(action_array)
        
        agent.store_transition((
            utils.mirror_state(state),
            mirrored_action_array,
            reward,
            utils.mirror_state(next_state),
            done
        ))


# ============================================================================
# Training & Loss Tracking
# ============================================================================


def train_agent(agent, state: TrainingState, config: TrainingConfig,
                current_episode_losses: Dict[str, List[float]]) -> None:
    """Train agent and update loss tracking."""
    if state.steps < state.current_phase_start + config.warmup_steps:
        return
    
    if state.steps % config.train_freq != 0:
        return
    
    stats = agent.train(config.updates_per_step)
    state.gradient_steps += config.updates_per_step
    
    if not isinstance(stats, dict):
        return
    
    # Track all losses
    for loss_key, loss_value in stats.items():
        if loss_value is None:
            continue
        if "loss" not in loss_key.lower() and "grad_norm" not in loss_key.lower():
            continue
        
        # Initialize tracking if needed
        if loss_key not in current_episode_losses:
            current_episode_losses[loss_key] = []
        
        # Add values
        if isinstance(loss_value, list):
            current_episode_losses[loss_key].extend(loss_value)
            state.losses.extend(loss_value)  # Legacy backward compatibility
        else:
            current_episode_losses[loss_key].append(loss_value)
            state.losses.append(loss_value)


# ============================================================================
# Evaluation
# ============================================================================


def run_evaluation(agent, curriculum: CurriculumConfig, run_manager: RunManager,
                   run_name: str, state: TrainingState, eval_num_games: int,
                   eval_weak_opponent: bool, max_episode_steps: int,
                   device, verbose: bool) -> None:
    """Run agent evaluation and save results."""
    if verbose:
        print(f"\nEvaluating agent at step {state.steps}...")
    
    # Save temporary checkpoint
    temp_checkpoint_path = run_manager.models_dir / f"{run_name}_temp_eval.pt"
    agent.save(str(temp_checkpoint_path))
    
    agent_config_dict = {
        "type": curriculum.agent.type,
        "hyperparameters": curriculum.agent.hyperparameters,
    }
    
    try:
        # Run evaluation
        eval_results = evaluate_agent(
            agent_path=str(temp_checkpoint_path),
            agent_config_dict=agent_config_dict,
            num_games=eval_num_games,
            weak_opponent=eval_weak_opponent,
            max_steps=max_episode_steps,
            num_parallel=None,
            device=device,
        )
        
        # Get Q-values and store separately for value propagation plot
        q_values = evaluate_episodes(agent)
        state.q_values.append(q_values)
        
        # Store results
        state.evaluation_results.append({
            "step": state.steps,
            "episode": state.completed_episodes,
            "win_rate": eval_results["win_rate"],
            "mean_reward": eval_results["mean_reward"],
            "std_reward": eval_results["std_reward"],
            "wins": eval_results["wins"],
            "losses": eval_results["losses"],
            "draws": eval_results["draws"],
        })
        
        # Save plots and CSVs
        run_manager.save_evaluation_plot(run_name, state.evaluation_results)
        run_manager.save_evaluation_csv(run_name, state.evaluation_results)
        run_manager.save_value_propagation_plot(run_name, state.q_values)
        
        if verbose:
            print(
                f"Evaluation: Win rate: {eval_results['win_rate']:.2%}, "
                f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}"
            )
    finally:
        if temp_checkpoint_path.exists():
            os.remove(temp_checkpoint_path)
    
    state.last_eval_step = state.steps


# ============================================================================
# Episode Completion
# ============================================================================


def complete_episode(episode_stats: EpisodeStats, state: TrainingState,
                     phase_config, current_episode_losses: Dict[str, List[float]],
                     log_freq_episodes: int, verbose: bool) -> None:
    """Handle episode completion: logging and stats tracking."""
    state.rewards.append(episode_stats.reward)
    state.phases.append(phase_config.name)
    state.completed_episodes += 1
    
    # Calculate average losses for this episode
    avg_losses = {}
    for loss_key, loss_values in current_episode_losses.items():
        if loss_values:
            avg_losses[loss_key] = sum(loss_values) / len(loss_values)
    
    # Store episode log
    state.episode_logs.append({
        "episode": state.completed_episodes,
        "reward": episode_stats.reward,
        "shaped_reward": episode_stats.shaped_reward,
        "steps": episode_stats.steps,
        "losses": avg_losses,
    })
    
    # Log to console (respecting log frequency)
    if verbose and (state.completed_episodes % log_freq_episodes == 0 or state.completed_episodes == 1):
        loss_parts = [
            f"Episode {state.completed_episodes}: reward={episode_stats.reward:.2f}, "
            f"shaped_reward={episode_stats.shaped_reward:.2f}, "
            f"steps={episode_stats.steps}",
            f"opponent={phase_config.opponent.type}",
        ]
        
        if avg_losses:
            for loss_key in sorted(avg_losses.keys()):
                loss_parts.append(f"{loss_key}={avg_losses[loss_key]:.6f}")
        else:
            loss_parts.append("(no training)")
        
        logger.info(" | ".join(loss_parts))


# ============================================================================
# Phase Management
# ============================================================================


def handle_phase_transition(curriculum: CurriculumConfig, agent, state: TrainingState,
                            current_opponent, run_manager: RunManager,
                            state_dim: int, action_dim: int, is_discrete: bool,
                            phase_idx: int, phase_config, verbose: bool):
    """Handle transition to a new curriculum phase."""
    if phase_idx == state.current_phase_idx:
        return current_opponent, None
    
    if verbose:
        print(f"\nTransitioning to phase {phase_idx + 1}/{len(curriculum.phases)}: {phase_config.name}")
    
    # Clear replay buffer
    if state.current_phase_idx >= 0:
        if verbose:
            print(f"  Clearing replay buffer (size: {agent.buffer.size})")
        agent.buffer.clear()
    
    # Sample new opponent
    new_opponent = sample_opponent(
        phase_config.opponent,
        agent=agent,
        checkpoint_dir=str(run_manager.models_dir),
        agent_config=curriculum.agent,
        state_dim=state_dim,
        action_dim=action_dim,
        is_discrete=is_discrete,
    )
    
    state.current_phase_idx = phase_idx
    state.current_phase_start = state.steps
    
    # Create new environment
    mode_str = phase_config.environment.get_mode_for_episode(0)
    env_mode = getattr(h_env.Mode, mode_str)
    new_env = h_env.HockeyEnv(mode=env_mode, keep_mode=phase_config.environment.keep_mode)
    
    return new_opponent, new_env


# ============================================================================
# Main Training Loop (Single Environment)
# ============================================================================


def run_training_episode(env, agent, opponent, phase_config, phase_local_episode: int,
                        config: TrainingConfig, state: TrainingState, is_discrete: bool,
                        action_fineness, current_episode_losses: Dict[str, List[float]]) -> EpisodeStats:
    """Run a single training episode."""
    # Reset environment
    obs, _ = env.reset()
    obs = obs.astype(np.float32, copy=False) if obs.dtype != np.float32 else obs
    
    # Initialize episode
    episode_stats = EpisodeStats()
    reward_weights = calculate_reward_weights(phase_local_episode, phase_config)
    agent.on_episode_start(state.completed_episodes)
    
    # Determine if opponent is stochastic
    deterministic_opponent = (
        phase_config.opponent.deterministic 
        if phase_config.opponent.type == "self_play" 
        else True
    )
    
    # Episode loop
    for t in range(config.max_episode_steps):
        # Get actions
        action_p1, discrete_action = get_agent_action(
            agent, obs, is_discrete, action_fineness, env, phase_config.environment.keep_mode
        )
        
        obs_p2 = env.obs_agent_two()
        action_p2 = get_opponent_action(opponent, obs_p2, deterministic=deterministic_opponent)
        action_p2 = convert_opponent_action(action_p2, action_fineness, phase_config.environment.keep_mode)
        
        # Step environment
        full_action = np.hstack([action_p1, action_p2])
        next_obs, reward, done, trunc, info = env.step(full_action)
        next_obs = next_obs.astype(np.float32, copy=False) if next_obs.dtype != np.float32 else next_obs
        
        # Apply reward shaping
        shaped_reward = apply_reward_shaping(reward, info, reward_weights)
        scaled_reward = shaped_reward * config.reward_scale
        
        # Store transition
        store_action = discrete_action if is_discrete else action_p1
        store_mirrored = opponent is None or phase_config.opponent.type == "none"
        store_transition(agent, obs, store_action, scaled_reward, next_obs, done,
                        is_discrete, store_mirrored)
        
        # Update stats
        obs = next_obs
        state.steps += 1
        episode_stats.reward += reward
        episode_stats.shaped_reward += shaped_reward
        episode_stats.steps += 1
        
        # Train agent
        train_agent(agent, state, config, current_episode_losses)
        
        if done or trunc:
            break
    
    agent.on_episode_end(state.completed_episodes)
    return episode_stats


# ============================================================================
# Main Training Function
# ============================================================================


def train_run(
    config_path: str,
    base_output_dir: str = "results/runs",
    run_name: str = None,
    verbose: bool = True,
    eval_freq_steps: int = None,
    eval_num_games: int = 100,
    eval_weak_opponent: bool = True,
    device: Optional[Union[str, int]] = None,
    checkpoint_path: Optional[str] = None,
    num_envs: int = 1,
    log_freq_episodes: int = 10,
):
    """
    Train an agent with curriculum learning.
    
    Args:
        config_path: Path to curriculum configuration JSON file
        base_output_dir: Base directory for saving results
        run_name: Name for this run (auto-generated if None)
        verbose: Whether to print progress information
        eval_freq_steps: Frequency of evaluation in steps
        eval_num_games: Number of games for evaluation
        eval_weak_opponent: Use weak (True) or strong (False) opponent for eval
        device: CUDA device ('cpu', 'cuda', 'cuda:0', etc.)
        checkpoint_path: Path to checkpoint to continue training from
        num_envs: Number of parallel environments (>1 uses vectorized training)
        log_freq_episodes: Log episode info every N episodes (default: 10)
    """
    # Use vectorized version if num_envs > 1
    if num_envs > 1:
        from multiprocessing import current_process
        current_proc = current_process()
        is_daemon = getattr(current_proc, "daemon", False)
        proc_name = getattr(current_proc, "name", "")
        is_pool_worker = "PoolWorker" in proc_name or is_daemon
        
        if verbose:
            env_type = "threaded" if is_pool_worker else "multiprocess"
            print(f"Using {env_type} vectorized environments with {num_envs} parallel instances")
        
        return train_run_vectorized(
            config_path, base_output_dir, run_name, verbose, eval_freq_steps,
            eval_num_games, eval_weak_opponent, device, checkpoint_path, num_envs,
            log_freq_episodes, use_threading=is_pool_worker
        )
    
    # Setup
    set_cuda_device(device)
    errors = validate_config(config_path)
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    curriculum = load_curriculum(config_path)
    config = TrainingConfig.from_curriculum(curriculum)
    
    # Initialize run
    run_manager = RunManager(base_output_dir=base_output_dir)
    if run_name is None:
        config_dict = curriculum_to_dict(curriculum)
        run_name = run_manager.generate_run_name(config_dict)
    
    # Get action space info
    first_phase = curriculum.phases[0]
    mode_str = first_phase.environment.get_mode_for_episode(0)
    env_mode = getattr(h_env.Mode, mode_str)
    action_fineness = curriculum.agent.hyperparameters.get("action_fineness", None)
    
    temp_env = h_env.HockeyEnv(mode=env_mode, keep_mode=first_phase.environment.keep_mode)
    state_dim, action_dim, is_discrete = get_action_space_info(
        temp_env, curriculum.agent.type, fineness=action_fineness
    )
    temp_env.close()
    
    # Create agent
    agent = create_agent(
        curriculum.agent, state_dim, action_dim, curriculum.hyperparameters, device=device
    )
    
    if verbose:
        print("\n" + agent.log_architecture() + "\n")
    
    if checkpoint_path:
        if verbose:
            print(f"Loading checkpoint: {checkpoint_path}")
        agent.load(checkpoint_path)
    
    # Training state
    state = TrainingState()
    total_episodes = get_total_episodes(curriculum)
    
    # Evaluate initial Q-values before training (episode 0)
    if verbose:
        print("Evaluating initial Q-values before training...")

    initial_q_values = evaluate_episodes(agent)
    state.q_values.append(initial_q_values)
    
    # Environment and opponent (initialized on first phase transition)
    env = None
    opponent = None
    
    # Training loop
    for global_episode in range(total_episodes):
        # Get current phase
        phase_idx, phase_local_episode, phase_config = get_phase_for_episode(
            curriculum, global_episode
        )
        
        # Handle phase transitions
        opponent, new_env = handle_phase_transition(
            curriculum, agent, state, opponent, run_manager,
            state_dim, action_dim, is_discrete, phase_idx, phase_config, verbose
        )
        if new_env is not None:
            if env is not None:
                env.close()
            env = new_env
        
        # Sample opponent for self-play episodes
        if phase_config.opponent.type == "self_play" and phase_config.opponent.checkpoint is None:
            opponent = sample_opponent(
                phase_config.opponent, agent=agent, checkpoint_dir=str(run_manager.models_dir),
                agent_config=curriculum.agent, state_dim=state_dim, action_dim=action_dim,
                is_discrete=is_discrete
            )
        
        # Track losses for this episode
        current_episode_losses = {}
        
        # Run episode
        episode_stats = run_training_episode(
            env, agent, opponent, phase_config, phase_local_episode,
            config, state, is_discrete, action_fineness, current_episode_losses
        )
        
        # Complete episode (logging, stats)
        complete_episode(episode_stats, state, phase_config, current_episode_losses, log_freq_episodes, verbose)
        
        # Evaluation
        if eval_freq_steps and state.steps - state.last_eval_step >= eval_freq_steps:
            run_evaluation(
                agent, curriculum, run_manager, run_name, state, eval_num_games,
                eval_weak_opponent, config.max_episode_steps, device, verbose
            )
        
        # Checkpoint saving
        if (state.completed_episodes) % config.checkpoint_save_freq == 0:
            run_manager.save_checkpoint(
                run_name, state.completed_episodes, agent,
                phase_index=phase_idx, phase_episode=phase_local_episode,
                episode_logs=state.episode_logs
            )
    
    # Cleanup and save results
    if env:
        env.close()
    
    save_final_results(run_manager, run_name, curriculum, state, agent, verbose)
    
    return {
        "run_name": run_name,
        "final_reward": state.rewards[-1] if state.rewards else 0,
        "mean_reward": np.mean(state.rewards[-100:]) if len(state.rewards) >= 100 else np.mean(state.rewards) if state.rewards else 0,
        "total_episodes": state.completed_episodes,
        "total_steps": state.steps,
        "total_gradient_steps": state.gradient_steps,
        "evaluation_results": state.evaluation_results,
    }


# ============================================================================
# Vectorized Training - Batched Actions Across Multiple Environments
# ============================================================================


@dataclass
class VectorizedEpisodeStats:
    """Statistics for multiple parallel episodes."""
    rewards: List[float]  # One per environment
    shaped_rewards: List[float]  # One per environment
    steps: List[int]  # One per environment
    
    @classmethod
    def create(cls, num_envs: int):
        return cls(
            rewards=[0.0] * num_envs,
            shaped_rewards=[0.0] * num_envs,
            steps=[0] * num_envs,
        )
    
    def reset_env(self, env_idx: int):
        """Reset stats for a specific environment."""
        self.rewards[env_idx] = 0.0
        self.shaped_rewards[env_idx] = 0.0
        self.steps[env_idx] = 0


def _make_hockey_env(mode, keep_mode):
    """Module-level factory function for creating HockeyEnv instances.
    
    This function must be at module level to be picklable for multiprocessing.
    """
    return h_env.HockeyEnv(mode=mode, keep_mode=keep_mode)


def get_batched_agent_actions(agent, states, is_discrete: bool, action_fineness,
                              keep_mode: bool) -> tuple:
    """Get actions for all environments in parallel (batched)."""
    if is_discrete:
        discrete_actions = agent.act_batch(states)
        if action_fineness is not None:
            actions = np.array([
                discrete_to_continuous_action_with_fineness(
                    da, fineness=action_fineness, keep_mode=keep_mode
                )
                for da in discrete_actions
            ])
        else:
            actions = np.array([
                utils.discrete_to_continuous_action_standard(da, keep_mode=keep_mode)
                for da in discrete_actions
            ])
        return actions, discrete_actions
    else:
        actions = agent.act_batch(states)
        return actions, None


def get_batched_opponent_actions(opponents, obs_agent2, num_envs: int,
                                phase_config, action_fineness, keep_mode: bool):
    """Get opponent actions for all environments."""
    deterministic = (
        phase_config.opponent.deterministic
        if phase_config.opponent.type == "self_play"
        else True
    )
    
    actions_p2 = np.array([
        get_opponent_action(opponents[i], obs_agent2[i], deterministic=deterministic)
        for i in range(num_envs)
    ])
    
    # Convert discrete opponent actions to continuous
    for i in range(num_envs):
        if isinstance(actions_p2[i], (int, np.integer)):
            if action_fineness is not None:
                actions_p2[i] = discrete_to_continuous_action_with_fineness(
                    actions_p2[i], fineness=action_fineness, keep_mode=keep_mode
                )
            else:
                actions_p2[i] = utils.discrete_to_continuous_action_standard(
                    actions_p2[i], keep_mode=keep_mode
                )
    
    return actions_p2


def store_vectorized_transitions(agent, states, actions_or_discrete, rewards,
                                 next_states, dones, num_envs: int, is_discrete: bool):
    """Store transitions for all environments in replay buffer."""
    for i in range(num_envs):
        if is_discrete:
            action_array = np.array([actions_or_discrete[i]], dtype=np.float32)
        else:
            action = actions_or_discrete[i]
            action_array = action.astype(np.float32) if action.dtype != np.float32 else action
        
        agent.store_transition((
            states[i], action_array, rewards[i], next_states[i], dones[i]
        ))


def handle_vectorized_episode_completion(
    env_idx: int,
    vec_episode_stats: VectorizedEpisodeStats,
    state: TrainingState,
    phase_config,
    current_episode_losses: Dict[str, List[float]],
    log_freq_episodes: int,
    verbose: bool,
):
    """Handle completion of a single episode in vectorized training."""
    # Create episode stats from vectorized stats
    episode_stats = EpisodeStats(
        reward=vec_episode_stats.rewards[env_idx],
        shaped_reward=vec_episode_stats.shaped_rewards[env_idx],
        steps=vec_episode_stats.steps[env_idx],
    )
    
    # Use the same completion logic as single-env
    complete_episode(episode_stats, state, phase_config, current_episode_losses, log_freq_episodes, verbose)
    
    # Reset this environment's stats
    vec_episode_stats.reset_env(env_idx)


def recreate_vectorized_env(num_envs: int, phase_config, use_threading: bool):
    """Create a new vectorized environment for a phase."""
    mode_str = phase_config.environment.get_mode_for_episode(0)
    env_mode = getattr(h_env.Mode, mode_str)
    
    VecEnvClass = (
        ThreadedVectorizedHockeyEnvOptimized
        if use_threading
        else VectorizedHockeyEnvOptimized
    )
    
    return VecEnvClass(
        num_envs=num_envs,
        env_fn=partial(
            _make_hockey_env,
            mode=env_mode,
            keep_mode=phase_config.environment.keep_mode,
        ),
    )


def initialize_vectorized_opponents(num_envs: int, phase_config, agent,
                                    run_manager: RunManager, curriculum: CurriculumConfig,
                                    state_dim: int, action_dim: int, is_discrete: bool):
    """Initialize opponents for all parallel environments."""
    opponents = []
    for i in range(num_envs):
        opponent = sample_opponent(
            phase_config.opponent,
            agent=agent,
            checkpoint_dir=str(run_manager.models_dir),
            agent_config=curriculum.agent,
            state_dim=state_dim,
            action_dim=action_dim,
            is_discrete=is_discrete,
        )
        opponents.append(opponent)
    return opponents


def check_vectorized_phase_transition(
    env_idx: int,
    curriculum: CurriculumConfig,
    state: TrainingState,
    total_episodes: int,
    num_envs: int,
    vec_env,
    opponents: List,
    agent,
    run_manager: RunManager,
    state_dim: int,
    action_dim: int,
    is_discrete: bool,
    use_threading: bool,
    verbose: bool,
) -> tuple:
    """Check and handle phase transition after episode completion.
    
    Returns: (new_phase_config, new_reward_weights, new_vec_env, new_opponents)
    """
    if state.completed_episodes >= total_episodes:
        return None, None, vec_env, opponents
    
    new_phase_idx, new_phase_local_episode, new_phase_config = get_phase_for_episode(
        curriculum, state.completed_episodes
    )
    
    # If same phase, just update reward weights
    if new_phase_idx == state.current_phase_idx:
        reward_weights = calculate_reward_weights(new_phase_local_episode, new_phase_config)
        return new_phase_config, reward_weights, vec_env, opponents
    
    # Phase transition - recreate everything
    if verbose:
        print(
            f"\nTransitioning to phase {new_phase_idx + 1}/{len(curriculum.phases)}: "
            f"{new_phase_config.name}"
        )
        print(f"  Clearing replay buffer (size: {agent.buffer.size})")
    
    agent.buffer.clear()
    
    # Recreate vectorized environment
    vec_env.close()
    new_vec_env = recreate_vectorized_env(num_envs, new_phase_config, use_threading)
    
    # Reinitialize opponents
    new_opponents = initialize_vectorized_opponents(
        num_envs, new_phase_config, agent, run_manager, curriculum,
        state_dim, action_dim, is_discrete
    )
    
    state.current_phase_idx = new_phase_idx
    state.current_phase_start = state.steps
    
    reward_weights = calculate_reward_weights(new_phase_local_episode, new_phase_config)
    
    return new_phase_config, reward_weights, new_vec_env, new_opponents


def process_vectorized_step(
    states,
    vec_env,
    agent,
    opponents,
    phase_config,
    reward_weights,
    vec_episode_stats: VectorizedEpisodeStats,
    state: TrainingState,
    config: TrainingConfig,
    current_episode_losses: Dict[str, List[float]],
    is_discrete: bool,
    action_fineness,
    num_envs: int,
    curriculum: CurriculumConfig,
    run_manager: RunManager,
    run_name: str,
    state_dim: int,
    action_dim: int,
    log_freq_episodes: int,
    use_threading: bool,
    verbose: bool,
) -> tuple:
    """Process one step for all vectorized environments.
    
    Returns: (next_states, phase_config, reward_weights, vec_env, opponents, current_episode_losses)
    """
    # Get batched actions
    actions_p1, discrete_actions = get_batched_agent_actions(
        agent, states, is_discrete, action_fineness, phase_config.environment.keep_mode
    )
    
    obs_agent2 = vec_env.obs_agent_two()
    actions_p2 = get_batched_opponent_actions(
        opponents, obs_agent2, num_envs, phase_config,
        action_fineness, phase_config.environment.keep_mode
    )
    
    # Step all environments
    full_actions = np.hstack([actions_p1, actions_p2])
    next_states, env_rewards, dones, truncs, infos = vec_env.step(full_actions)
    next_states = next_states.astype(np.float32, copy=False) if next_states.dtype != np.float32 else next_states
    
    # Count all environment steps
    state.steps += num_envs
    
    # Process each environment's result
    scaled_rewards = []
    for i in range(num_envs):
        # Apply reward shaping
        shaped_reward = apply_reward_shaping(env_rewards[i], infos[i], reward_weights)
        scaled_reward = shaped_reward * config.reward_scale
        scaled_rewards.append(scaled_reward)
        
        # Update episode stats
        vec_episode_stats.rewards[i] += env_rewards[i]
        vec_episode_stats.shaped_rewards[i] += shaped_reward
        vec_episode_stats.steps[i] += 1
        
        # Check for episode completion
        if dones[i] or truncs[i] or vec_episode_stats.steps[i] >= config.max_episode_steps:
            # Handle episode completion
            handle_vectorized_episode_completion(
                i, vec_episode_stats, state, phase_config,
                current_episode_losses, log_freq_episodes, verbose
            )
            
            # Reset episode losses for next episode
            current_episode_losses = {}
            
            # Initialize next episode
            agent.on_episode_start(state.completed_episodes)
            
            # Check for phase transition
            total_episodes = get_total_episodes(curriculum)
            phase_config, reward_weights, vec_env, opponents = check_vectorized_phase_transition(
                i, curriculum, state, total_episodes, num_envs, vec_env, opponents,
                agent, run_manager, state_dim, action_dim, is_discrete, use_threading, verbose
            )
            
            # Save checkpoint if needed
            if state.completed_episodes % config.checkpoint_save_freq == 0:
                phase_idx, phase_local_episode, _ = get_phase_for_episode(curriculum, state.completed_episodes - 1)
                run_manager.save_checkpoint(
                    run_name, state.completed_episodes, agent,
                    phase_index=phase_idx, phase_episode=phase_local_episode,
                    episode_logs=state.episode_logs
                )
    
    # Store transitions
    actions_to_store = discrete_actions if is_discrete else actions_p1
    store_vectorized_transitions(
        agent, states, actions_to_store, scaled_rewards,
        next_states, dones, num_envs, is_discrete
    )
    
    return next_states, phase_config, reward_weights, vec_env, opponents, current_episode_losses


def train_run_vectorized(
    config_path: str,
    base_output_dir: str = "results/runs",
    run_name: str = None,
    verbose: bool = True,
    eval_freq_steps: int = None,
    eval_num_games: int = 100,
    eval_weak_opponent: bool = True,
    device: Optional[Union[str, int]] = None,
    checkpoint_path: Optional[str] = None,
    num_envs: int = 4,
    log_freq_episodes: int = 10,
    use_threading: bool = False,
):
    """
    Train with vectorized environments (multiple environments in parallel).
    
    This provides significant speedup (1.4-2.4x) by:
    - Batching actions across environments
    - Batching GPU operations
    - Parallelizing environment steps
    
    Args:
        config_path: Path to curriculum configuration JSON file
        base_output_dir: Base directory for saving results
        run_name: Name for this run (auto-generated if None)
        verbose: Whether to print progress information
        eval_freq_steps: Frequency of evaluation in steps
        eval_num_games: Number of games for evaluation
        eval_weak_opponent: Use weak (True) or strong (False) opponent for eval
        device: CUDA device ('cpu', 'cuda', 'cuda:0', etc.)
        checkpoint_path: Path to checkpoint to continue training from
        num_envs: Number of parallel environments (4-8 recommended)
        use_threading: Use threaded (True) vs multiprocess (False) vectorization
    """
    # Setup
    set_cuda_device(device)
    errors = validate_config(config_path)
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    curriculum = load_curriculum(config_path)
    config = TrainingConfig.from_curriculum(curriculum)
    
    # Initialize run
    run_manager = RunManager(base_output_dir=base_output_dir)
    if run_name is None:
        config_dict = curriculum_to_dict(curriculum)
        run_name = run_manager.generate_run_name(config_dict) + f"_vec{num_envs}"
    
    # Get action space info from temporary environment
    first_phase = curriculum.phases[0]
    mode_str = first_phase.environment.get_mode_for_episode(0)
    env_mode = getattr(h_env.Mode, mode_str)
    action_fineness = curriculum.agent.hyperparameters.get("action_fineness", None)
    
    temp_env = h_env.HockeyEnv(mode=env_mode, keep_mode=first_phase.environment.keep_mode)
    state_dim, action_dim, is_discrete = get_action_space_info(
        temp_env, curriculum.agent.type, fineness=action_fineness
    )
    temp_env.close()
    
    # Create agent
    agent = create_agent(
        curriculum.agent, state_dim, action_dim, curriculum.hyperparameters, device=device
    )
    
    if verbose:
        print("\n" + agent.log_architecture() + "\n")
    
    if checkpoint_path:
        if verbose:
            print(f"Loading checkpoint: {checkpoint_path}")
        agent.load(checkpoint_path)
    
    # Training state
    state = TrainingState()
    total_episodes = get_total_episodes(curriculum)
    
    # Evaluate initial Q-values before training (episode 0)
    if verbose:
        print("Evaluating initial Q-values before training...")

    initial_q_values = evaluate_episodes(agent)
    state.q_values.append(initial_q_values)

    # Initialize first phase
    phase_idx, phase_local_episode, phase_config = get_phase_for_episode(curriculum, 0)
    state.current_phase_idx = phase_idx
    
    # Create vectorized environment
    vec_env = recreate_vectorized_env(num_envs, phase_config, use_threading)
    
    # Initialize opponents for all environments
    opponents = initialize_vectorized_opponents(
        num_envs, phase_config, agent, run_manager, curriculum,
        state_dim, action_dim, is_discrete
    )
    
    # Reset all environments
    states = vec_env.reset()
    states = states.astype(np.float32, copy=False) if states.dtype != np.float32 else states
    
    # Initialize episode tracking for each environment
    vec_episode_stats = VectorizedEpisodeStats.create(num_envs)
    current_episode_losses = {}
    
    # Initialize agent and reward weights
    agent.on_episode_start(0)
    reward_weights = calculate_reward_weights(0, phase_config)
    
    # Main vectorized training loop
    while state.completed_episodes < total_episodes:
        # Process one step for all environments
        states, phase_config, reward_weights, vec_env, opponents, current_episode_losses = process_vectorized_step(
            states, vec_env, agent, opponents, phase_config, reward_weights,
            vec_episode_stats, state, config, current_episode_losses,
            is_discrete, action_fineness, num_envs, curriculum, run_manager, run_name,
            state_dim, action_dim, log_freq_episodes, use_threading, verbose
        )
        
        # Train agent
        train_agent(agent, state, config, current_episode_losses)
        
        # Evaluation
        if eval_freq_steps and state.steps - state.last_eval_step >= eval_freq_steps:
            run_evaluation(
                agent, curriculum, run_manager, run_name, state, eval_num_games,
                eval_weak_opponent, config.max_episode_steps, device, verbose
            )
    
    # Cleanup and save results
    vec_env.close()
    save_final_results(run_manager, run_name, curriculum, state, agent, verbose)
    
    if verbose:
        print(f"\nVectorized training completed with {num_envs} parallel environments")
        print(f"  Average steps per episode: {state.steps / len(state.rewards) if state.rewards else 0:.1f}")
    
    return {
        "run_name": run_name,
        "final_reward": state.rewards[-1] if state.rewards else 0,
        "mean_reward": np.mean(state.rewards[-100:]) if len(state.rewards) >= 100 else np.mean(state.rewards) if state.rewards else 0,
        "total_episodes": state.completed_episodes,
        "total_steps": state.steps,
        "total_gradient_steps": state.gradient_steps,
        "evaluation_results": state.evaluation_results,
    }


# ============================================================================
# Utilities
# ============================================================================


def curriculum_to_dict(curriculum: CurriculumConfig) -> dict:
    """Convert curriculum config to dictionary for saving."""
    return {
        "curriculum": {
            "phases": [
                {
                    "name": phase.name,
                    "episodes": phase.episodes,
                    "environment": {
                        "mode": phase.environment.mode,
                        "keep_mode": phase.environment.keep_mode,
                    },
                    "opponent": {
                        "type": phase.opponent.type,
                        "weight": phase.opponent.weight,
                        "checkpoint": phase.opponent.checkpoint,
                        "deterministic": phase.opponent.deterministic,
                        "opponents": phase.opponent.opponents,
                    },
                    "reward_shaping": None if phase.reward_shaping is None else {
                        "N": phase.reward_shaping.N,
                        "K": phase.reward_shaping.K,
                        "CLOSENESS_START": phase.reward_shaping.CLOSENESS_START,
                        "TOUCH_START": phase.reward_shaping.TOUCH_START,
                        "CLOSENESS_FINAL": phase.reward_shaping.CLOSENESS_FINAL,
                        "TOUCH_FINAL": phase.reward_shaping.TOUCH_FINAL,
                        "DIRECTION_FINAL": phase.reward_shaping.DIRECTION_FINAL,
                    },
                }
                for phase in curriculum.phases
            ]
        },
        "hyperparameters": curriculum.hyperparameters,
        "training": curriculum.training,
        "agent": {
            "type": curriculum.agent.type,
            "hyperparameters": curriculum.agent.hyperparameters,
        },
    }


def save_final_results(run_manager: RunManager, run_name: str, curriculum: CurriculumConfig,
                      state: TrainingState, agent, verbose: bool) -> None:
    """Save all final training results."""
    # Save configs and data
    run_manager.save_config(run_name, curriculum_to_dict(curriculum))
    run_manager.save_rewards_csv(run_name, state.rewards, phases=state.phases)
    run_manager.save_losses_csv(run_name, state.losses)
    
    if state.episode_logs:
        run_manager.save_episode_logs_csv(run_name, state.episode_logs)
    
    # Save plots
    run_manager.save_plots(run_name, state.rewards, state.losses)
    
    if state.evaluation_results:
        run_manager.save_evaluation_plot(run_name, state.evaluation_results)
        run_manager.save_evaluation_csv(run_name, state.evaluation_results)
    
    # Save final model
    model_path = run_manager.get_model_path(run_name)
    agent.save(str(model_path))
    
    # Print summary
    if verbose:
        print("\nTraining Summary:")
        print(f"  Total episodes: {state.completed_episodes}")
        print(f"  Total steps: {state.steps}")
        print(f"  Total gradient steps: {state.gradient_steps}")
        print(f"  Losses collected: {len(state.losses)}")
        if state.losses:
            print(f"  Mean loss: {sum(state.losses) / len(state.losses):.4f}")
