import csv
import logging
import os
import random
import sys
from functools import partial
from typing import Dict, List, Optional, Union

import hockey.hockey_env as h_env
import numpy as np

from rl_hockey.common import utils
from rl_hockey.common.evaluation.agent_evaluator import evaluate_agent
from rl_hockey.common.reward_backprop import apply_win_reward_backprop
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
    get_resource_usage,
    set_cuda_device,
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


def _save_random_episodes_to_csv(
    agent, run_manager, run_name, num_episodes=10, save_episode=None
):
    """Save random episodes from the buffer to a CSV file.

    Args:
        agent: The agent with a buffer attribute
        run_manager: RunManager instance for getting output directory
        run_name: Name of the current run
        num_episodes: Number of random episodes to sample (default: 10)
        save_episode: Episode number when this save occurred (for tracking)
    """
    try:
        buffer = agent.buffer

        # Check if buffer has episodes
        if not hasattr(buffer, "num_eps") or buffer.num_eps == 0:
            logger.warning("Buffer is empty, cannot save episodes to CSV")
            return

        # Check if buffer has enough episodes
        available_episodes = buffer.num_eps
        if available_episodes < num_episodes:
            logger.warning(
                f"Buffer only has {available_episodes} episodes, "
                f"but requested {num_episodes}. Saving {available_episodes} episodes instead."
            )
            num_episodes = available_episodes

        # Get access to episodes - check buffer type
        if hasattr(buffer, "_episodes"):
            # TDMPC2ReplayBuffer
            all_episodes = buffer._episodes
        elif hasattr(buffer, "buffer") and hasattr(buffer.buffer, "_storage"):
            # Regular ReplayBuffer - we need to handle differently
            logger.warning(
                "Regular ReplayBuffer detected, episode sampling may not work as expected"
            )
            return
        else:
            logger.warning(f"Unknown buffer type: {type(buffer)}, cannot save episodes")
            return

        # Sample random episodes
        selected_indices = random.sample(range(len(all_episodes)), num_episodes)
        selected_episodes = [all_episodes[i] for i in selected_indices]

        # Prepare CSV data
        csv_data = []

        for ep_idx, episode in enumerate(selected_episodes):
            # Convert tensors to numpy if needed
            obs = episode["obs"]
            action = episode["action"]
            reward = episode["reward"]
            terminated = episode["terminated"]
            winner = episode.get("winner", None)

            # Handle torch tensors
            if hasattr(obs, "cpu"):
                obs = obs.cpu().numpy()
            if hasattr(action, "cpu"):
                action = action.cpu().numpy()
            if hasattr(reward, "cpu"):
                reward = reward.cpu().numpy()
            if hasattr(terminated, "cpu"):
                terminated = terminated.cpu().numpy()

            # Ensure arrays are numpy
            obs = np.asarray(obs)
            action = np.asarray(action)
            reward = np.asarray(reward).flatten()
            terminated = np.asarray(terminated).flatten()

            # Convert winner to int if it's a tensor
            if winner is not None:
                if hasattr(winner, "cpu"):
                    winner = winner.cpu().item()
                elif hasattr(winner, "item"):
                    winner = winner.item()
                winner = int(winner) if winner is not None else None

            # Get original reward if available
            reward_original = episode.get("reward_original", reward)
            if hasattr(reward_original, "cpu"):
                reward_original = reward_original.cpu().numpy()
            reward_original = np.asarray(reward_original).flatten()

            # obs shape: (T+1, obs_dim), action/reward/terminated: (T, ...)
            T = len(action)

            for step in range(T):
                row = {
                    "save_episode": save_episode if save_episode is not None else 0,
                    "episode_id": ep_idx,
                    "step": step,
                }

                # Add observation features (current state)
                obs_flat = obs[step].flatten()
                for i, val in enumerate(obs_flat):
                    row[f"obs_{i}"] = val

                # Add action features
                action_flat = action[step].flatten()
                for i, val in enumerate(action_flat):
                    row[f"action_{i}"] = val

                # Add both original and backprop rewards
                row["reward_original"] = (
                    reward_original[step]
                    if reward_original.ndim == 1
                    else reward_original[step, 0]
                )
                row["reward_backprop"] = (
                    reward[step] if reward.ndim == 1 else reward[step, 0]
                )

                # Add terminated/done flag
                row["terminated"] = (
                    terminated[step] if terminated.ndim == 1 else terminated[step, 0]
                )

                # Add winner (same for all steps in the episode)
                row["winner"] = winner if winner is not None else ""

                csv_data.append(row)

        # Write to CSV - save to a new file each time (not append)
        # Include save_episode in filename to make each file unique
        save_episode_str = f"ep{save_episode}" if save_episode is not None else "ep0"
        csv_path = (
            run_manager.csvs_dir
            / f"{run_name}_buffer_episodes_sample_{save_episode_str}.csv"
        )

        if csv_data:
            fieldnames = [
                "save_episode",
                "episode_id",
                "step",
                "reward_original",
                "reward_backprop",
                "terminated",
                "winner",
            ]
            # Add obs and action fieldnames
            first_row = csv_data[0]
            for key in first_row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

            # Always write new file (not append)
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)

            logger.info(
                f"Saved {num_episodes} random episodes to {csv_path} (save_episode={save_episode})"
            )

    except Exception as e:
        logger.warning(f"Failed to save episodes to CSV: {e}")


def _make_hockey_env(mode, keep_mode):
    """Module-level factory function for creating HockeyEnv instances.

    This function must be at module level to be picklable for multiprocessing.
    """
    return h_env.HockeyEnv(mode=mode, keep_mode=keep_mode)


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
    run_manager: Optional[RunManager] = None,
    start_episode: int = 0,
    initial_episode_logs: Optional[List[Dict]] = None,
):
    """
    Train an agent with curriculum learning.

    Args:
        config_path: Path to curriculum configuration JSON file
        base_output_dir: Base directory for saving results
        run_name: Name for this run (if None, generated automatically)
        verbose: Whether to print progress information
        eval_freq_steps: Frequency of evaluation in steps (if None, uses config default)
        eval_num_games: Number of games to run for evaluation
        eval_weak_opponent: Whether to use weak (True) or strong (False) BasicOpponent for evaluation
        device: CUDA device to use (None = auto-detect, 'cpu' = CPU, 'cuda' = cuda:0, 'cuda:0' = first GPU, 'cuda:1' = second GPU, etc.). Can also be an integer (0, 1, etc.) for device ID.
        checkpoint_path: Optional path to a checkpoint file to load and continue training from
        num_envs: Number of parallel environments (1 = single env, 4-8 recommended for speedup)
        run_manager: Optional RunManager instance to reuse (if None, creates a new one)
        start_episode: Episode number to start training from (for resuming from checkpoint, default 0)
        initial_episode_logs: Optional list of episode logs to preload (for resuming from checkpoint)
    """
    # Route to vectorized training if num_envs > 1
    # Check if we're in a multiprocessing Pool worker to decide which implementation to use
    if num_envs > 1:
        from multiprocessing import current_process

        current_proc = current_process()
        is_daemon = getattr(current_proc, "daemon", False)
        proc_name = getattr(current_proc, "name", "")
        is_pool_worker = "PoolWorker" in proc_name or is_daemon

        if num_envs > 1:
            if is_pool_worker:
                if verbose:
                    print(
                        f"Using threaded vectorized environments with {num_envs} parallel instances (Pool worker mode)"
                    )
            else:
                if verbose:
                    print(
                        f"Using multiprocess vectorized environments with {num_envs} parallel instances"
                    )
            return _train_run_vectorized(
                config_path,
                base_output_dir,
                run_name,
                verbose,
                eval_freq_steps,
                eval_num_games,
                eval_weak_opponent,
                device,
                checkpoint_path,
                num_envs,
                use_threading=is_pool_worker,
                run_manager=run_manager,
                start_episode=start_episode,
                initial_episode_logs=initial_episode_logs,
            )
    # Set CUDA device if specified
    set_cuda_device(device)

    errors = validate_config(config_path)
    if errors:
        raise ValueError(
            "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    curriculum = load_curriculum(config_path)

    if run_manager is None:
        run_manager = RunManager(base_output_dir=base_output_dir)
    if run_name is None:
        config_dict = _curriculum_to_dict(curriculum)
        run_name = run_manager.generate_run_name(config_dict)

    training_params = curriculum.training
    max_episode_steps = training_params.get("max_episode_steps", 500)
    updates_per_step = training_params.get("updates_per_step", 1)
    warmup_steps = training_params.get("warmup_steps", 400)
    reward_scale = training_params.get("reward_scale", 0.1)
    checkpoint_save_freq = training_params.get("checkpoint_save_freq", 100)
    train_freq = training_params.get("train_freq", 1)

    total_episodes = get_total_episodes(curriculum)

    current_env = None
    current_phase_idx = -1
    current_opponent = None
    current_phase_start = 0

    # none is the default action space of 8 for DDDQN
    action_fineness = curriculum.agent.hyperparameters.get("action_fineness", None)

    first_phase = curriculum.phases[0]
    initial_mode_str = first_phase.environment.get_mode_for_episode(0)
    env_mode = getattr(h_env.Mode, initial_mode_str)
    current_env = h_env.HockeyEnv(
        mode=env_mode, keep_mode=first_phase.environment.keep_mode
    )
    state_dim, agent_action_dim, is_agent_discrete = get_action_space_info(
        current_env, curriculum.agent.type, fineness=action_fineness
    )

    agent = create_agent(
        curriculum.agent,
        state_dim,
        agent_action_dim,
        curriculum.hyperparameters,
        device=device,
    )

    # Log the agent network architecture
    if verbose:
        print("\n" + agent.log_architecture() + "\n")

    # Log backprop parameters if available
    if (
        verbose
        and hasattr(agent, "buffer")
        and hasattr(agent.buffer, "win_reward_bonus")
        and hasattr(agent.buffer, "win_reward_discount")
    ):
        print("REWARD BACKPROPAGATION PARAMETERS:")
        print(f"  win_reward_bonus: {agent.buffer.win_reward_bonus}")
        print(f"  win_reward_discount: {agent.buffer.win_reward_discount}")
        print("")

    if checkpoint_path is not None:
        if verbose:
            print(f"Loading agent from checkpoint: {checkpoint_path}")
        agent.load(checkpoint_path)
        if verbose:
            print("Checkpoint loaded successfully")

    losses = []
    all_losses_dict = {}  # Dictionary to store all loss types: {loss_type: [values]}
    rewards = []
    phases = []
    episode_logs = []  # List to store episode logs with all loss types
    steps = 0
    gradient_steps = 0
    evaluation_results = []
    last_eval_step = 0
    resource_logs = []
    resource_log_freq = training_params.get(
        "resource_log_freq", 10
    )  # Log every N steps
    last_resource_log_step = 0

    # Track resource usage per episode for averaging
    episode_resource_window = training_params.get("resource_avg_window_episodes", 10)
    episode_resource_samples = []  # List of dicts, one per episode
    current_episode_cpu_samples = []  # CPU samples during current episode
    current_episode_gpu_samples = []  # GPU samples during current episode

    def get_reward_weights(episode_idx, phase_config):
        reward_shaping = phase_config.reward_shaping

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
                "closeness": CLOSENESS_START,
                "touch": TOUCH_START,
                "direction": 0.0,
            }
        elif episode_idx < N + K:
            alpha = (episode_idx - N) / K
            return {
                "closeness": CLOSENESS_START * (1 - alpha) + CLOSENESS_FINAL * alpha,
                "touch": TOUCH_START * (1 - alpha) + TOUCH_FINAL * alpha,
                "direction": DIRECTION_FINAL * alpha,
            }
        else:
            return {
                "closeness": CLOSENESS_FINAL,
                "touch": TOUCH_FINAL,
                "direction": DIRECTION_FINAL,
            }

    for global_episode in range(start_episode, total_episodes):
        phase_idx, phase_local_episode, phase_config = get_phase_for_episode(
            curriculum, global_episode
        )

        if phase_idx != current_phase_idx:
            if verbose:
                print(
                    f"\nTransitioning to phase {phase_idx + 1}/{len(curriculum.phases)}: {phase_config.name}"
                )

            current_opponent = sample_opponent(
                phase_config.opponent,
                agent=agent,
                checkpoint_dir=str(run_manager.models_dir),
                agent_config=curriculum.agent,
                state_dim=state_dim,
                action_dim=agent_action_dim,
                is_discrete=is_agent_discrete,
            )

            if current_phase_idx >= 0:
                if verbose:
                    print(
                        f"  Clearing replay buffer (size: {agent.buffer.size}) to avoid distribution mismatch"
                    )
                agent.buffer.clear()

            current_phase_idx = phase_idx
            current_phase_start = steps

        sampled_mode_str = phase_config.environment.get_mode_for_episode(
            phase_local_episode
        )
        env_mode = getattr(h_env.Mode, sampled_mode_str)

        if (
            current_env is None
            or current_env.mode != env_mode
            or current_env.keep_mode != phase_config.environment.keep_mode
        ):
            if current_env is not None:
                current_env.close()
            current_env = h_env.HockeyEnv(
                mode=env_mode, keep_mode=phase_config.environment.keep_mode
            )

        state, _ = current_env.reset()
        # Ensure state is float32 to avoid repeated conversions
        if state.dtype != np.float32:
            state = state.astype(np.float32, copy=False)

        reward_weights = get_reward_weights(phase_local_episode, phase_config)

        agent.on_episode_start(global_episode)
        total_reward = 0
        total_shaped_reward = 0
        # Track scaled rewards for backprop calculation
        episode_scaled_rewards = []
        episode_winner = None
        # Reset episode resource tracking
        current_episode_cpu_samples = []
        current_episode_gpu_samples = []
        # Track losses for this episode
        episode_losses = {}

        if (
            phase_config.opponent.type == "self_play"
            and phase_config.opponent.checkpoint is None
        ):
            current_opponent = sample_opponent(
                phase_config.opponent,
                agent=agent,
                checkpoint_dir=str(run_manager.models_dir),
                agent_config=curriculum.agent,
                state_dim=state_dim,
                action_dim=agent_action_dim,
                is_discrete=is_agent_discrete,
            )

        for t in range(max_episode_steps):
            if is_agent_discrete:
                discrete_actions = agent.act_batch(state[None, :])
                discrete_action = discrete_actions[0]
                if action_fineness is not None:
                    action_p1 = discrete_to_continuous_action_with_fineness(
                        discrete_action,
                        fineness=action_fineness,
                        keep_mode=phase_config.environment.keep_mode,
                    )
                else:
                    action_p1 = current_env.discrete_to_continous_action(
                        discrete_action
                    )
            else:
                t0s = np.array([t == 0], dtype=bool)
                actions_p1 = agent.act_batch(state[None, :], t0s=t0s)
                action_p1 = actions_p1[0]

            obs_agent2 = current_env.obs_agent_two()
            deterministic_opponent = (
                phase_config.opponent.deterministic
                if phase_config.opponent.type == "self_play"
                else True
            )
            action_p2 = get_opponent_action(
                current_opponent, obs_agent2, deterministic=deterministic_opponent
            )

            if isinstance(action_p2, (int, np.integer)):
                if action_fineness is not None:
                    action_p2 = discrete_to_continuous_action_with_fineness(
                        action_p2,
                        fineness=action_fineness,
                        keep_mode=phase_config.environment.keep_mode,
                    )
                else:
                    action_p2 = current_env.discrete_to_continous_action(action_p2)

            action = np.hstack([action_p1, action_p2])

            next_state, reward, done, trunc, info = current_env.step(action)
            # Ensure next_state is float32 to avoid repeated conversions
            if next_state.dtype != np.float32:
                next_state = next_state.astype(np.float32, copy=False)

            if reward_weights is not None:
                shaped_reward = reward
                shaped_reward += (
                    info.get("reward_closeness_to_puck", 0.0)
                    * reward_weights["closeness"]
                )
                shaped_reward += (
                    info.get("reward_touch_puck", 0.0) * reward_weights["touch"]
                )
                shaped_reward += (
                    info.get("reward_puck_direction", 0.0) * reward_weights["direction"]
                )
            else:
                shaped_reward = reward

            scaled_reward = shaped_reward * reward_scale

            # Get winner information if episode is done (only pass when done=True, not trunc)
            # Winner is 1 for agent win, -1 for agent loss
            winner = None
            if done:
                winner = info.get("winner", 0)

            if is_agent_discrete:
                # Pre-allocate array once and reuse (optimization)
                discrete_action_array = np.array([discrete_action], dtype=np.float32)
                agent.store_transition(
                    (state, discrete_action_array, scaled_reward, next_state, done),
                    winner=winner,
                )

                if current_opponent is None or phase_config.opponent.type == "none":
                    if action_fineness is not None:
                        mirrored_action = utils.mirror_discrete_action(
                            discrete_action,
                            fineness=action_fineness,
                            keep_mode=phase_config.environment.keep_mode,
                        )
                    else:
                        mirrored_action = utils.mirror_discrete_action(discrete_action)
                    mirrored_action_array = np.array(
                        [mirrored_action], dtype=np.float32
                    )
                    agent.store_transition(
                        (
                            utils.mirror_state(state),
                            mirrored_action_array,
                            scaled_reward,
                            utils.mirror_state(next_state),
                            done,
                        ),
                        winner=winner,
                    )
            else:
                # Ensure action is float32, avoid copy if already correct type
                if action_p1.dtype != np.float32:
                    action_array = action_p1.astype(np.float32, copy=False)
                else:
                    action_array = action_p1
                agent.store_transition(
                    (state, action_array, scaled_reward, next_state, done),
                    winner=winner,
                )

                if current_opponent is None or phase_config.opponent.type == "none":
                    mirrored_action_array = utils.mirror_action(action_array)
                    agent.store_transition(
                        (
                            utils.mirror_state(state),
                            mirrored_action_array,
                            scaled_reward,
                            utils.mirror_state(next_state),
                            done,
                        ),
                        winner=winner,
                    )

            state = next_state
            steps += 1
            total_reward += reward
            total_shaped_reward += shaped_reward

            # Log resource usage at intervals
            if (
                resource_log_freq > 0
                and steps - last_resource_log_step >= resource_log_freq
            ):
                try:
                    resource_usage = get_resource_usage()
                    resource_usage["step"] = steps
                    resource_usage["episode"] = global_episode
                    resource_logs.append(resource_usage)
                    last_resource_log_step = steps

                    # Collect samples for episode averages
                    cpu_percent = resource_usage.get("cpu_percent", 0)
                    gpu_utilization = resource_usage.get("gpu_utilization", 0)
                    if cpu_percent is not None and cpu_percent > 0:
                        current_episode_cpu_samples.append(cpu_percent)
                    if gpu_utilization is not None and gpu_utilization > 0:
                        current_episode_gpu_samples.append(gpu_utilization)
                except Exception as e:
                    if verbose:
                        logger.warning(f"Failed to log resource usage: {e}")

            if steps >= current_phase_start + warmup_steps and steps % train_freq == 0:
                stats = agent.train(updates_per_step)
                gradient_steps += updates_per_step
                if isinstance(stats, dict):
                    # Track all loss types and gradient norms from stats
                    for loss_key, loss_value in stats.items():
                        if loss_value is not None and (
                            "loss" in loss_key.lower()
                            or "grad_norm" in loss_key.lower()
                        ):
                            if loss_key not in all_losses_dict:
                                all_losses_dict[loss_key] = []
                            if loss_key not in episode_losses:
                                episode_losses[loss_key] = []

                            if isinstance(loss_value, list):
                                all_losses_dict[loss_key].extend(loss_value)
                                episode_losses[loss_key].extend(loss_value)
                            else:
                                all_losses_dict[loss_key].append(loss_value)
                                episode_losses[loss_key].append(loss_value)

                    # Keep backward compatibility: extract single loss for old loss tracking
                    loss_list = None
                    if "loss" in stats and stats["loss"] is not None:
                        loss_list = stats["loss"]
                    elif "critic_loss" in stats and stats["critic_loss"] is not None:
                        loss_list = stats["critic_loss"]
                    elif "actor_loss" in stats and stats["actor_loss"] is not None:
                        loss_list = stats["actor_loss"]

                    if loss_list is not None:
                        if isinstance(loss_list, list):
                            losses.extend(loss_list)
                        else:
                            losses.append(loss_list)

            if (
                eval_freq_steps is not None
                and steps - last_eval_step >= eval_freq_steps
            ):
                if verbose:
                    print(f"\nEvaluating agent at step {steps}...")

                temp_checkpoint_path = (
                    run_manager.models_dir / f"{run_name}_temp_eval.pt"
                )
                agent.save(str(temp_checkpoint_path))

                agent_config_dict = {
                    "type": curriculum.agent.type,
                    "hyperparameters": curriculum.agent.hyperparameters,
                }

                try:
                    # Convert to absolute path to ensure it works even after cwd changes
                    abs_checkpoint_path = os.path.abspath(str(temp_checkpoint_path))
                    eval_results = evaluate_agent(
                        agent_path=abs_checkpoint_path,
                        agent_config_dict=agent_config_dict,
                        num_games=eval_num_games,
                        weak_opponent=eval_weak_opponent,
                        max_steps=max_episode_steps,
                        num_parallel=None,
                        device=device,
                    )

                    evaluation_results.append(
                        {
                            "step": steps,
                            "episode": global_episode,
                            "win_rate": eval_results["win_rate"],
                            "mean_reward": eval_results["mean_reward"],
                            "std_reward": eval_results["std_reward"],
                            "wins": eval_results["wins"],
                            "losses": eval_results["losses"],
                            "draws": eval_results["draws"],
                        }
                    )

                    if evaluation_results:
                        run_manager.save_evaluation_plot(run_name, evaluation_results)
                        run_manager.save_evaluation_csv(run_name, evaluation_results)

                    if verbose:
                        print(
                            f"Evaluation results: Win rate: {eval_results['win_rate']:.2%}, "
                            f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}"
                        )
                finally:
                    if temp_checkpoint_path.exists():
                        os.remove(temp_checkpoint_path)

                last_eval_step = steps

            if done or trunc:
                break

        agent.on_episode_end(global_episode)

        # Save 10 random episodes from buffer to CSV every 10 episodes
        if steps >= warmup_steps and (global_episode + 1) % 10 == 0:
            if hasattr(agent, "buffer") and hasattr(agent.buffer, "num_eps"):
                if agent.buffer.num_eps >= 10:
                    _save_random_episodes_to_csv(
                        agent,
                        run_manager,
                        run_name,
                        num_episodes=10,
                        save_episode=global_episode + 1,
                    )

        rewards.append(total_reward)
        phases.append(phase_config.name)

        # Calculate episode averages for resource usage
        episode_cpu_avg = (
            sum(current_episode_cpu_samples) / len(current_episode_cpu_samples)
            if current_episode_cpu_samples
            else 0.0
        )
        episode_gpu_avg = (
            sum(current_episode_gpu_samples) / len(current_episode_gpu_samples)
            if current_episode_gpu_samples
            else 0.0
        )
        episode_resource_samples.append(
            {
                "episode": global_episode,
                "cpu_percent": episode_cpu_avg,
                "gpu_utilization": episode_gpu_avg,
            }
        )

        # Calculate and print averages over last n episodes (only once every episode_resource_window episodes)
        if (
            len(episode_resource_samples) >= episode_resource_window
            and (global_episode + 1) % episode_resource_window == 0
        ):
            recent_samples = episode_resource_samples[-episode_resource_window:]
            avg_cpu = sum(s["cpu_percent"] for s in recent_samples) / len(
                recent_samples
            )
            avg_gpu = sum(s["gpu_utilization"] for s in recent_samples) / len(
                recent_samples
            )

            if verbose:
                logger.info(
                    f"Resource Usage (last {episode_resource_window} episodes): "
                    f"CPU: {avg_cpu:.1f}%, GPU: {avg_gpu:.1f}%"
                )

        if (global_episode + 1) % checkpoint_save_freq == 0:
            run_manager.save_checkpoint(
                run_name,
                global_episode + 1,
                agent,
                phase_index=phase_idx,
                phase_episode=phase_local_episode,
                episode_logs=episode_logs,
            )

        # Log all losses at end of episode
        # Calculate average losses for this episode
        avg_losses = {}
        if episode_losses:
            for loss_key, loss_values in episode_losses.items():
                if loss_values:
                    avg_losses[loss_key] = sum(loss_values) / len(loss_values)

        # Calculate backprop reward sum if buffer supports it
        total_backprop_reward = (
            total_shaped_reward  # Default to shaped reward if no backprop
        )
        if (
            hasattr(agent, "buffer")
            and hasattr(agent.buffer, "win_reward_bonus")
            and hasattr(agent.buffer, "win_reward_discount")
            and episode_winner is not None
            and len(episode_scaled_rewards) > 0
        ):
            # Apply backprop to calculate the backprop reward sum
            scaled_rewards_array = np.array(episode_scaled_rewards, dtype=np.float32)
            backprop_rewards, _, _ = apply_win_reward_backprop(
                scaled_rewards_array,
                winner=episode_winner,
                win_reward_bonus=agent.buffer.win_reward_bonus,
                win_reward_discount=agent.buffer.win_reward_discount,
                use_torch=False,
            )
            total_backprop_reward = float(np.sum(backprop_rewards))

        # Store episode log (always, even if no losses)
        episode_logs.append(
            {
                "episode": global_episode + 1,
                "reward": total_reward,  # Original reward
                "shaped_reward": total_shaped_reward,  # Shaped reward (before backprop)
                "backprop_reward": total_backprop_reward,  # Reward after backprop
                "total_gradient_steps": gradient_steps,
                "losses": avg_losses,
            }
        )

        # Log episode information (always log, even if no losses)
        # Get environment mode and opponent type for logging
        sampled_mode_str = phase_config.environment.get_mode_for_episode(
            phase_local_episode
        )
        opponent_type = phase_config.opponent.type

        loss_info_parts = [
            f"Episode {global_episode + 1}: reward={total_reward:.2f}, shaped_reward={total_shaped_reward:.2f}, backprop_reward={total_backprop_reward:.2f}",
            f"env={sampled_mode_str}",
            f"opponent={opponent_type}",
        ]
        if avg_losses:
            for loss_key in sorted(avg_losses.keys()):
                loss_info_parts.append(f"{loss_key}={avg_losses[loss_key]:.6f}")
        else:
            loss_info_parts.append("(no training in this episode)")
        logger.info(" | ".join(loss_info_parts))

    run_manager.save_config(run_name, _curriculum_to_dict(curriculum))
    run_manager.save_rewards_csv(run_name, rewards, phases=phases)
    run_manager.save_losses_csv(run_name, losses)
    if episode_logs:
        run_manager.save_episode_logs_csv(run_name, episode_logs)
    if resource_logs:
        run_manager.save_resources_csv(run_name, resource_logs)

    if verbose:
        print("\nTraining Summary:")
        print(f"  Total episodes: {len(rewards)}")
        print(f"  Total steps: {steps}")
        print(f"  Total gradient steps: {gradient_steps}")
        print(f"  Losses collected: {len(losses)}")
        if losses:
            print(f"  Mean loss: {sum(losses) / len(losses):.4f}")
        else:
            print("  Warning: No losses collected!")
            print(
                f"    This might mean training hasn't started (warmup_steps={warmup_steps})"
            )
            print(
                f"    or the replay buffer is too small (current size: {agent.buffer.size})"
            )

    run_manager.save_plots(run_name, rewards, losses)

    if evaluation_results:
        run_manager.save_evaluation_plot(run_name, evaluation_results)
        run_manager.save_evaluation_csv(run_name, evaluation_results)

    model_path = run_manager.get_model_path(run_name)
    agent.save(str(model_path))

    current_env.close()

    return {
        "run_name": run_name,
        "final_reward": rewards[-1] if rewards else 0,
        "mean_reward": np.mean(rewards[-100:])
        if len(rewards) >= 100
        else np.mean(rewards)
        if rewards
        else 0,
        "total_episodes": len(rewards),
        "total_steps": steps,
        "total_gradient_steps": gradient_steps,
        "evaluation_results": evaluation_results,
    }


def _train_run_vectorized(
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
    use_threading: bool = False,
    run_manager: Optional[RunManager] = None,
    start_episode: int = 0,
    initial_episode_logs: Optional[List[Dict]] = None,
):
    """
    Train with vectorized environments (multiple environments in parallel).
    This provides 1.4-2.4x speedup by batching GPU operations.

    Args:
        use_threading: If True, use threaded vectorized env (for Pool workers).
                      If False, use multiprocess vectorized env (better performance).
        run_manager: Optional RunManager instance to reuse (if None, creates a new one)
        start_episode: Episode number to start training from (for resuming from checkpoint, default 0)
    """
    set_cuda_device(device)

    errors = validate_config(config_path)
    if errors:
        raise ValueError(
            "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    curriculum = load_curriculum(config_path)

    if run_manager is None:
        run_manager = RunManager(base_output_dir=base_output_dir)
    if run_name is None:
        config_dict = _curriculum_to_dict(curriculum)
        run_name = run_manager.generate_run_name(config_dict) + f"_vec{num_envs}"

    training_params = curriculum.training
    max_episode_steps = training_params.get("max_episode_steps", 500)
    updates_per_step = training_params.get("updates_per_step", 1)
    warmup_steps = training_params.get("warmup_steps", 400)
    reward_scale = training_params.get("reward_scale", 0.1)
    checkpoint_save_freq = training_params.get("checkpoint_save_freq", 100)
    train_freq = training_params.get("train_freq", 1)

    total_episodes = get_total_episodes(curriculum)

    current_vec_env = None
    current_phase_idx = -1
    current_opponents = [None] * num_envs
    current_phase_start = 0

    action_fineness = curriculum.agent.hyperparameters.get("action_fineness", None)

    # Create temporary environment to get dimensions
    first_phase = curriculum.phases[0]
    initial_mode_str = first_phase.environment.get_mode_for_episode(0)
    env_mode = getattr(h_env.Mode, initial_mode_str)
    temp_env = h_env.HockeyEnv(
        mode=env_mode, keep_mode=first_phase.environment.keep_mode
    )
    state_dim, agent_action_dim, is_agent_discrete = get_action_space_info(
        temp_env, curriculum.agent.type, fineness=action_fineness
    )
    temp_env.close()

    # Create agent
    agent = create_agent(
        curriculum.agent,
        state_dim,
        agent_action_dim,
        curriculum.hyperparameters,
        device=device,
    )

    # Log the agent network architecture
    if verbose:
        print("\n" + agent.log_architecture() + "\n")

    # Log backprop parameters if available
    if (
        verbose
        and hasattr(agent, "buffer")
        and hasattr(agent.buffer, "win_reward_bonus")
        and hasattr(agent.buffer, "win_reward_discount")
    ):
        print("REWARD BACKPROPAGATION PARAMETERS:")
        print(f"  win_reward_bonus: {agent.buffer.win_reward_bonus}")
        print(f"  win_reward_discount: {agent.buffer.win_reward_discount}")
        print("")

    # Enable fast mode for TDMPC2 if specified in training config
    if hasattr(agent, "set_fast_mode") and training_params.get("use_fast_mode", False):
        agent.set_fast_mode(True)
        if verbose:
            print(
                "Fast mode enabled for TDMPC2 (using policy network instead of full MPC)"
            )

    if checkpoint_path is not None:
        if verbose:
            print(f"Loading agent from checkpoint: {checkpoint_path}")
        agent.load(checkpoint_path)
        if verbose:
            print("Checkpoint loaded successfully")

    losses = []
    all_losses_dict = {}  # Dictionary to store all loss types: {loss_type: [values]}
    rewards = []
    phases = []
    # Initialize with old episode logs if resuming
    episode_logs = initial_episode_logs.copy() if initial_episode_logs else []
    steps = 0
    gradient_steps = 0
    evaluation_results = []
    last_eval_step = 0
    resource_logs = []
    resource_log_freq = training_params.get(
        "resource_log_freq", 200
    )  # Log every N steps
    last_resource_log_step = 0

    # Track resource usage per episode for averaging
    episode_resource_window = training_params.get("resource_avg_window_episodes", 10)
    episode_resource_samples = []  # List of dicts, one per episode
    current_episode_cpu_samples = []  # CPU samples during current episode
    current_episode_gpu_samples = []  # GPU samples during current episode

    # Track episode state for each parallel environment
    episode_rewards = [0.0] * num_envs
    episode_shaped_rewards = [0.0] * num_envs
    episode_scaled_rewards = [
        [] for _ in range(num_envs)
    ]  # Track scaled rewards per step for backprop
    episode_winners = [None] * num_envs  # Track winner for each environment
    episode_steps = [0] * num_envs
    completed_episodes = start_episode
    # Track losses for current episode (across all training steps)
    current_episode_losses = {}

    def get_reward_weights(episode_idx, phase_config):
        reward_shaping = phase_config.reward_shaping

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
                "closeness": CLOSENESS_START,
                "touch": TOUCH_START,
                "direction": 0.0,
            }
        elif episode_idx < N + K:
            alpha = (episode_idx - N) / K
            return {
                "closeness": CLOSENESS_START * (1 - alpha) + CLOSENESS_FINAL * alpha,
                "touch": TOUCH_START * (1 - alpha) + TOUCH_FINAL * alpha,
                "direction": DIRECTION_FINAL * alpha,
            }
        else:
            return {
                "closeness": CLOSENESS_FINAL,
                "touch": TOUCH_FINAL,
                "direction": DIRECTION_FINAL,
            }

    # Initialize first phase
    phase_idx, phase_local_episode, phase_config = get_phase_for_episode(curriculum, 0)
    current_phase_idx = phase_idx

    # Create vectorized environment
    sampled_mode_str = phase_config.environment.get_mode_for_episode(0)
    env_mode = getattr(h_env.Mode, sampled_mode_str)

    # Choose appropriate vectorized env class based on whether we're in a Pool worker
    VecEnvClass = (
        ThreadedVectorizedHockeyEnvOptimized
        if use_threading
        else VectorizedHockeyEnvOptimized
    )

    current_vec_env = VecEnvClass(
        num_envs=num_envs,
        env_fn=partial(
            _make_hockey_env,
            mode=env_mode,
            keep_mode=phase_config.environment.keep_mode,
        ),
    )

    # Initialize opponents for each environment
    for i in range(num_envs):
        current_opponents[i] = sample_opponent(
            phase_config.opponent,
            agent=agent,
            checkpoint_dir=str(run_manager.models_dir),
            agent_config=curriculum.agent,
            state_dim=state_dim,
            action_dim=agent_action_dim,
            is_discrete=is_agent_discrete,
        )

    # Reset all environments
    states = current_vec_env.reset()
    if states.dtype != np.float32:
        states = states.astype(np.float32, copy=False)

    t0s = np.ones(num_envs, dtype=bool)

    agent.on_episode_start(0)
    reward_weights = get_reward_weights(0, phase_config)

    while completed_episodes < total_episodes:
        # Get actions for all environments (batched!)
        if is_agent_discrete:
            discrete_actions = agent.act_batch(states, t0s=t0s)
            if action_fineness is not None:
                actions_p1 = np.array(
                    [
                        discrete_to_continuous_action_with_fineness(
                            da,
                            fineness=action_fineness,
                            keep_mode=phase_config.environment.keep_mode,
                        )
                        for da in discrete_actions
                    ]
                )
            else:
                # Convert standard discrete actions to continuous
                actions_p1 = np.array(
                    [
                        utils.discrete_to_continuous_action_standard(
                            da, keep_mode=phase_config.environment.keep_mode
                        )
                        for da in discrete_actions
                    ]
                )
        else:
            actions_p1 = agent.act_batch(states, t0s=t0s)

        # Get opponent actions for each environment
        obs_agent2 = current_vec_env.obs_agent_two()
        actions_p2 = np.array(
            [
                get_opponent_action(
                    current_opponents[i],
                    obs_agent2[i],
                    deterministic=phase_config.opponent.deterministic
                    if phase_config.opponent.type == "self_play"
                    else True,
                )
                for i in range(num_envs)
            ]
        )

        # Convert discrete actions if needed
        for i in range(num_envs):
            if isinstance(actions_p2[i], (int, np.integer)):
                if action_fineness is not None:
                    actions_p2[i] = discrete_to_continuous_action_with_fineness(
                        actions_p2[i],
                        fineness=action_fineness,
                        keep_mode=phase_config.environment.keep_mode,
                    )
                else:
                    actions_p2[i] = utils.discrete_to_continuous_action_standard(
                        actions_p2[i], keep_mode=phase_config.environment.keep_mode
                    )

        # Combine actions
        full_actions = np.hstack([actions_p1, actions_p2])

        # Step all environments
        next_states, env_rewards, dones, truncs, infos = current_vec_env.step(
            full_actions
        )
        t0s = dones | truncs
        if next_states.dtype != np.float32:
            next_states = next_states.astype(np.float32, copy=False)

        # Count steps: each iteration steps ALL environments, so add num_envs
        steps += num_envs

        # Process each environment
        for i in range(num_envs):
            reward = env_rewards[i]
            done = dones[i]
            trunc = truncs[i]
            info = infos[i]

            # Apply reward shaping
            if reward_weights is not None:
                shaped_reward = reward
                shaped_reward += (
                    info.get("reward_closeness_to_puck", 0.0)
                    * reward_weights["closeness"]
                )
                shaped_reward += (
                    info.get("reward_touch_puck", 0.0) * reward_weights["touch"]
                )
                shaped_reward += (
                    info.get("reward_puck_direction", 0.0) * reward_weights["direction"]
                )
            else:
                shaped_reward = reward

            scaled_reward = shaped_reward * reward_scale

            # Track scaled rewards for backprop calculation
            episode_scaled_rewards[i].append(scaled_reward)

            # Get winner information if episode is done (only pass when done=True, not trunc)
            # Winner is 1 for agent win, -1 for agent loss
            winner = None
            if done:
                winner = info.get("winner", 0)
                episode_winners[i] = winner

            # Store transition
            if is_agent_discrete:
                discrete_action_array = np.array(
                    [discrete_actions[i]], dtype=np.float32
                )
                agent.store_transition(
                    (
                        states[i],
                        discrete_action_array,
                        scaled_reward,
                        next_states[i],
                        done,
                    ),
                    winner=winner,
                )
            else:
                action_array = (
                    actions_p1[i].astype(np.float32, copy=False)
                    if actions_p1[i].dtype != np.float32
                    else actions_p1[i]
                )
                agent.store_transition(
                    (states[i], action_array, scaled_reward, next_states[i], done),
                    winner=winner,
                )

            # Track episode stats
            episode_rewards[i] += reward
            episode_shaped_rewards[i] += shaped_reward
            episode_steps[i] += 1

            # Handle episode completion
            if done or trunc or episode_steps[i] >= max_episode_steps:
                rewards.append(episode_rewards[i])
                phases.append(phase_config.name)
                completed_episodes += 1

                # Calculate episode averages for resource usage (using recent samples)
                episode_cpu_avg = (
                    sum(current_episode_cpu_samples[-20:])
                    / len(current_episode_cpu_samples[-20:])
                    if current_episode_cpu_samples
                    else 0.0
                )
                episode_gpu_avg = (
                    sum(current_episode_gpu_samples[-20:])
                    / len(current_episode_gpu_samples[-20:])
                    if current_episode_gpu_samples
                    else 0.0
                )
                episode_resource_samples.append(
                    {
                        "episode": completed_episodes,
                        "cpu_percent": episode_cpu_avg,
                        "gpu_utilization": episode_gpu_avg,
                    }
                )

                # Calculate and print averages over last n episodes (only once every episode_resource_window episodes)
                if (
                    len(episode_resource_samples) >= episode_resource_window
                    and completed_episodes % episode_resource_window == 0
                ):
                    recent_samples = episode_resource_samples[-episode_resource_window:]
                    avg_cpu = sum(s["cpu_percent"] for s in recent_samples) / len(
                        recent_samples
                    )
                    avg_gpu = sum(s["gpu_utilization"] for s in recent_samples) / len(
                        recent_samples
                    )

                    if verbose:
                        logger.info(
                            f"Resource Usage (last {episode_resource_window} episodes): "
                            f"CPU: {avg_cpu:.1f}%, GPU: {avg_gpu:.1f}%"
                        )

                # Trim resource sample lists to prevent unbounded growth
                # Keep only last 100 samples (enough for ~5 episodes at 20 samples/episode)
                if len(current_episode_cpu_samples) > 100:
                    current_episode_cpu_samples = current_episode_cpu_samples[-100:]
                if len(current_episode_gpu_samples) > 100:
                    current_episode_gpu_samples = current_episode_gpu_samples[-100:]

                # Save episode rewards before resetting (for logging)
                final_episode_reward = episode_rewards[i]
                final_episode_shaped_reward = episode_shaped_rewards[i]

                # Calculate backprop reward sum if buffer supports it
                final_episode_backprop_reward = final_episode_shaped_reward  # Default to shaped reward if no backprop
                if (
                    hasattr(agent, "buffer")
                    and hasattr(agent.buffer, "win_reward_bonus")
                    and hasattr(agent.buffer, "win_reward_discount")
                    and episode_winners[i] is not None
                    and len(episode_scaled_rewards[i]) > 0
                ):
                    # Apply backprop to calculate the backprop reward sum
                    scaled_rewards_array = np.array(
                        episode_scaled_rewards[i], dtype=np.float32
                    )
                    backprop_rewards, _, _ = apply_win_reward_backprop(
                        scaled_rewards_array,
                        winner=episode_winners[i],
                        win_reward_bonus=agent.buffer.win_reward_bonus,
                        win_reward_discount=agent.buffer.win_reward_discount,
                        use_torch=False,
                    )
                    final_episode_backprop_reward = float(np.sum(backprop_rewards))

                # Reset episode tracking for this environment
                episode_rewards[i] = 0.0
                episode_shaped_rewards[i] = 0.0
                episode_scaled_rewards[i] = []
                episode_winners[i] = None
                episode_steps[i] = 0

                # IMPORTANT: We need to manually reset this environment!
                # The next_states[i] is already the reset state from the vectorized env
                # But we need to initialize the next episode
                agent.on_episode_start(completed_episodes)

                # Save 10 random episodes from buffer to CSV every 10 episodes
                if steps >= warmup_steps and completed_episodes % 10 == 0:
                    if hasattr(agent, "buffer") and hasattr(agent.buffer, "num_eps"):
                        if agent.buffer.num_eps >= 10:
                            _save_random_episodes_to_csv(
                                agent,
                                run_manager,
                                run_name,
                                num_episodes=10,
                                save_episode=completed_episodes,
                            )

                # Log all losses at end of episode
                # Calculate average losses for this episode
                avg_losses = {}
                if current_episode_losses:
                    for loss_key, loss_values in current_episode_losses.items():
                        if loss_values:
                            avg_losses[loss_key] = sum(loss_values) / len(loss_values)

                # Store episode log (always, even if no losses)
                episode_logs.append(
                    {
                        "episode": completed_episodes,
                        "reward": final_episode_reward,  # Original reward
                        "shaped_reward": final_episode_shaped_reward,  # Shaped reward (before backprop)
                        "backprop_reward": final_episode_backprop_reward,  # Reward after backprop
                        "total_gradient_steps": gradient_steps,
                        "losses": avg_losses,
                    }
                )

                # Log episode information (always log, even if no losses)
                # Get environment mode and opponent type for logging
                # Compute phase_local_episode for current episode
                _, current_phase_local_episode, current_phase_config = (
                    get_phase_for_episode(curriculum, completed_episodes)
                )
                sampled_mode_str = (
                    current_phase_config.environment.get_mode_for_episode(
                        current_phase_local_episode
                    )
                )
                opponent_type = current_phase_config.opponent.type

                loss_info_parts = [
                    f"Episode {completed_episodes}: reward={final_episode_reward:.2f}, shaped_reward={final_episode_shaped_reward:.2f}, backprop_reward={final_episode_backprop_reward:.2f}",
                    f"env={sampled_mode_str}",
                    f"opponent={opponent_type}",
                ]
                if avg_losses:
                    for loss_key in sorted(avg_losses.keys()):
                        loss_info_parts.append(f"{loss_key}={avg_losses[loss_key]:.6f}")
                else:
                    loss_info_parts.append("(no training in this episode)")
                logger.info(" | ".join(loss_info_parts))

                # Reset episode losses after logging (for next episode)
                current_episode_losses = {}

                # Check if we need to change phase
                if completed_episodes < total_episodes:
                    new_phase_idx, new_phase_local_episode, new_phase_config = (
                        get_phase_for_episode(curriculum, completed_episodes)
                    )

                    if new_phase_idx != current_phase_idx:
                        if verbose:
                            print(
                                f"\nTransitioning to phase {new_phase_idx + 1}/{len(curriculum.phases)}: {new_phase_config.name}"
                            )

                        # Clear buffer for phase transition
                        if verbose:
                            print(
                                f"  Clearing replay buffer (size: {agent.buffer.size})"
                            )
                        agent.buffer.clear()

                        # Recreate environments with new phase settings
                        current_vec_env.close()

                        sampled_mode_str = (
                            new_phase_config.environment.get_mode_for_episode(
                                new_phase_local_episode
                            )
                        )
                        env_mode = getattr(h_env.Mode, sampled_mode_str)

                        current_vec_env = VecEnvClass(
                            num_envs=num_envs,
                            env_fn=partial(
                                _make_hockey_env,
                                mode=env_mode,
                                keep_mode=new_phase_config.environment.keep_mode,
                            ),
                        )

                        # Update opponents
                        for j in range(num_envs):
                            current_opponents[j] = sample_opponent(
                                new_phase_config.opponent,
                                agent=agent,
                                checkpoint_dir=str(run_manager.models_dir),
                                agent_config=curriculum.agent,
                                state_dim=state_dim,
                                action_dim=agent_action_dim,
                                is_discrete=is_agent_discrete,
                            )

                        current_phase_idx = new_phase_idx
                        current_phase_start = steps
                        phase_config = new_phase_config

                    # Update reward weights for the next episode (must be done for every episode,
                    # not just phase transitions, since reward shaping depends on phase_local_episode)
                    reward_weights = get_reward_weights(
                        new_phase_local_episode, new_phase_config
                    )

                # Note: individual environment resets are handled automatically by vectorized env

        # Update states
        states = next_states

        # Log resource usage at intervals
        if (
            resource_log_freq > 0
            and steps - last_resource_log_step >= resource_log_freq
        ):
            try:
                resource_usage = get_resource_usage()
                resource_usage["step"] = steps
                resource_usage["episode"] = completed_episodes
                resource_logs.append(resource_usage)
                last_resource_log_step = steps

                # Collect samples for episode averages
                cpu_percent = resource_usage.get("cpu_percent", 0)
                gpu_utilization = resource_usage.get("gpu_utilization", 0)
                if cpu_percent is not None and cpu_percent > 0:
                    current_episode_cpu_samples.append(cpu_percent)
                if gpu_utilization is not None and gpu_utilization > 0:
                    current_episode_gpu_samples.append(gpu_utilization)
            except Exception as e:
                if verbose:
                    logger.warning(f"Failed to log resource usage: {e}")

        # Train agent
        if steps >= current_phase_start + warmup_steps and steps % train_freq == 0:
            stats = agent.train(updates_per_step)
            gradient_steps += updates_per_step
            if isinstance(stats, dict):
                # Track all loss types and gradient norms from stats
                for loss_key, loss_value in stats.items():
                    if loss_value is not None and (
                        "loss" in loss_key.lower() or "grad_norm" in loss_key.lower()
                    ):
                        if loss_key not in all_losses_dict:
                            all_losses_dict[loss_key] = []
                        if loss_key not in current_episode_losses:
                            current_episode_losses[loss_key] = []

                        if isinstance(loss_value, list):
                            all_losses_dict[loss_key].extend(loss_value)
                            current_episode_losses[loss_key].extend(loss_value)
                        else:
                            all_losses_dict[loss_key].append(loss_value)
                            current_episode_losses[loss_key].append(loss_value)

                # Keep backward compatibility: extract single loss for old loss tracking
                loss_list = None
                if "loss" in stats and stats["loss"] is not None:
                    loss_list = stats["loss"]
                elif "critic_loss" in stats and stats["critic_loss"] is not None:
                    loss_list = stats["critic_loss"]
                elif "actor_loss" in stats and stats["actor_loss"] is not None:
                    loss_list = stats["actor_loss"]

                if loss_list is not None:
                    if isinstance(loss_list, list):
                        losses.extend(loss_list)
                    else:
                        losses.append(loss_list)

        # Evaluation
        if eval_freq_steps is not None and steps - last_eval_step >= eval_freq_steps:
            if verbose:
                print(f"\nEvaluating agent at step {steps}...")

            temp_checkpoint_path = run_manager.models_dir / f"{run_name}_temp_eval.pt"
            agent.save(str(temp_checkpoint_path))

            agent_config_dict = {
                "type": curriculum.agent.type,
                "hyperparameters": curriculum.agent.hyperparameters,
            }

            try:
                # Convert to absolute path to ensure it works even after cwd changes
                abs_checkpoint_path = os.path.abspath(str(temp_checkpoint_path))
                eval_results = evaluate_agent(
                    agent_path=abs_checkpoint_path,
                    agent_config_dict=agent_config_dict,
                    num_games=eval_num_games,
                    weak_opponent=eval_weak_opponent,
                    max_steps=max_episode_steps,
                    num_parallel=None,
                    device=device,
                )

                evaluation_results.append(
                    {
                        "step": steps,
                        "episode": completed_episodes,
                        "win_rate": eval_results["win_rate"],
                        "mean_reward": eval_results["mean_reward"],
                        "std_reward": eval_results["std_reward"],
                        "wins": eval_results["wins"],
                        "losses": eval_results["losses"],
                        "draws": eval_results["draws"],
                    }
                )

                if evaluation_results:
                    run_manager.save_evaluation_plot(run_name, evaluation_results)
                    run_manager.save_evaluation_csv(run_name, evaluation_results)

                if verbose:
                    print(
                        f"Evaluation results: Win rate: {eval_results['win_rate']:.2%}, "
                        f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}"
                    )
            finally:
                if temp_checkpoint_path.exists():
                    os.remove(temp_checkpoint_path)

            last_eval_step = steps

        # Checkpoint saving
        if completed_episodes > 0 and completed_episodes % checkpoint_save_freq == 0:
            run_manager.save_checkpoint(
                run_name,
                completed_episodes,
                agent,
                phase_index=phase_idx,
                phase_episode=phase_local_episode,
                episode_logs=episode_logs,
            )

    # Save final results
    run_manager.save_config(run_name, _curriculum_to_dict(curriculum))
    run_manager.save_rewards_csv(run_name, rewards, phases=phases)
    run_manager.save_losses_csv(run_name, losses)
    if episode_logs:
        run_manager.save_episode_logs_csv(run_name, episode_logs)
    if resource_logs:
        run_manager.save_resources_csv(run_name, resource_logs)

    if verbose:
        print("\nTraining Summary:")
        print(f"  Total episodes: {len(rewards)}")
        print(f"  Total steps: {steps}")
        print(f"  Total gradient steps: {gradient_steps}")
        print(f"  Losses collected: {len(losses)}")
        print(f"  Vectorized environments: {num_envs}")
        print(
            f"  Average steps per episode: {steps / len(rewards) if rewards else 0:.1f}"
        )
        print(
            f"  Average episode length per env: {steps / num_envs / len(rewards) if rewards else 0:.1f}"
        )
        if losses:
            print(f"  Mean loss: {sum(losses) / len(losses):.4f}")

    run_manager.save_plots(run_name, rewards, losses)

    if evaluation_results:
        run_manager.save_evaluation_plot(run_name, evaluation_results)
        run_manager.save_evaluation_csv(run_name, evaluation_results)

    model_path = run_manager.get_model_path(run_name)
    agent.save(str(model_path))

    current_vec_env.close()

    return {
        "run_name": run_name,
        "final_reward": rewards[-1] if rewards else 0,
        "mean_reward": np.mean(rewards[-100:])
        if len(rewards) >= 100
        else np.mean(rewards)
        if rewards
        else 0,
        "total_episodes": len(rewards),
        "total_steps": steps,
        "total_gradient_steps": gradient_steps,
        "evaluation_results": evaluation_results,
    }


def _curriculum_to_dict(curriculum: CurriculumConfig) -> dict:
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
                    "reward_shaping": None
                    if phase.reward_shaping is None
                    else {
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
