"""
Minimal training script for debugging.
No curriculum, just SAC vs weak opponent with basic logging.
"""

import csv
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hockey.hockey_env as h_env
import matplotlib.pyplot as plt
import numpy as np

from rl_hockey.common.agent import Agent
from rl_hockey.common.archive import Archive, Matchmaker, Rating, RatingSystem
from rl_hockey.common.archive.matchmaker import Opponent
from rl_hockey.common.evaluation.value_propagation import (
    evaluate_episodes,
    plot_value_heatmap,
    plot_values_line,
    sample_trajectories,
)
from rl_hockey.common.evaluation.winrate_evaluator import evaluate_winrate
from rl_hockey.common.reward_backprop import apply_win_reward_backprop
from rl_hockey.common.training.agent_factory import create_agent
from rl_hockey.common.training.curriculum_manager import (
    CurriculumConfig,
    PhaseConfig,
    get_phase_for_episode,
    get_total_episodes,
    load_curriculum,
)
from rl_hockey.common.training.opponent_manager import (
    sample_opponent,
)
from rl_hockey.common.training.plot_episode_logs import plot_training_metrics
from rl_hockey.common.training.run_manager import RunManager
from rl_hockey.common.vectorized_env import VectorizedHockeyEnvOptimized

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Holds the state of training."""

    step: int = 0
    episode: int = 0
    gradient_steps: int = 0
    last_eval_step: int = 0
    last_checkpoint_step: int = 0
    phase: PhaseConfig = None
    phase_index: int = 0
    last_warmup_reset_step: int = 0
    rating: Rating = Rating()


@dataclass
class TrainingMetrics:
    """Flexible container for all training data, indexed by global steps."""

    episodes: List[Dict[str, Any]] = field(default_factory=list)
    updates: List[Dict[str, Any]] = field(default_factory=list)
    winrates: List[Dict[str, Any]] = field(default_factory=list)

    def add_episode(
        self,
        step: int,
        rating: float,
        reward: float,
        length: int,
        phase: Optional[str] = None,
    ):
        """Records an episode finish at a specific global step."""
        episode_data = {
            "step": step,
            "rating": rating,
            "reward": reward,
            "length": length,
        }
        if phase:
            episode_data["phase"] = phase
        self.episodes.append(episode_data)

    def add_update(self, step: int, **metrics):
        """Records training metrics at a specific global step."""
        self.updates.append({"step": step, **metrics})

    def add_winrate(self, step: int, winrate: float):
        """Records winrate evaluation at a specific global step."""
        self.winrates.append(
            {
                "step": step,
                "winrate": winrate,
            }
        )


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""

    reward: float = 0.0
    shaped_reward: float = 0.0
    length: int = 0
    step_rewards: List[float] = field(default_factory=list)

    def reset(self):
        """Reset episode metrics."""
        self.reward = 0.0
        self.shaped_reward = 0.0
        self.length = 0
        self.step_rewards = []


def _make_hockey_env(mode, keep_mode):
    """Factory function for creating HockeyEnv instances (must be at module level for pickling)."""
    return h_env.HockeyEnv(mode=mode, keep_mode=keep_mode)


def _create_opponents(
    num_envs: int,
    phase: PhaseConfig,
    state_dim: int,
    action_dim: int,
    rating: Optional[float] = None,
    matchmaker: Optional[Matchmaker] = None,
) -> List:
    """Create opponents based on curriculum phase or default to weak opponents."""
    if phase:
        return [
            sample_opponent(
                phase.opponent,
                state_dim=state_dim,
                action_dim=action_dim,
                is_discrete=False,
                # rating=rating,
                # matchmaker=matchmaker,
            )
            for _ in range(num_envs)
        ]
    return [h_env.BasicOpponent(weak=True) for _ in range(num_envs)]


def _to_scalar_loss(val: Any) -> float:
    """Convert metric value to scalar for CSV (handle lists from SAC/TD3)."""
    if val is None:
        return float("nan")
    if isinstance(val, (list, tuple)):
        return float(np.mean(val)) if val else float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _format_losses_for_log(metrics: Optional[Dict[str, Any]]) -> List[str]:
    """Format training metrics (losses, grad norms) for episode log line. Works with SAC, TD3, TDMPC2."""
    if not metrics:
        return []
    parts = []
    for key in sorted(metrics.keys()):
        key_lower = key.lower()
        if "loss" not in key_lower and "grad_norm" not in key_lower:
            continue
        val = metrics[key]
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            if not val:
                continue
            try:
                scalar = sum(float(x) for x in val) / len(val)
            except (TypeError, ValueError):
                continue
        else:
            try:
                scalar = float(val)
            except (TypeError, ValueError):
                continue
        parts.append(f"{key}={scalar:.4f}")
    return parts


# ============================================================================
# Reward Shaping
# ============================================================================


def calculate_reward_weights(
    episode_idx: int, phase_config
) -> Optional[Dict[str, float]]:
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
            "closeness": reward_shaping.CLOSENESS_START * (1 - alpha)
            + reward_shaping.CLOSENESS_FINAL * alpha,
            "touch": reward_shaping.TOUCH_START * (1 - alpha)
            + reward_shaping.TOUCH_FINAL * alpha,
            "direction": reward_shaping.DIRECTION_FINAL * alpha,
        }
    else:
        return {
            "closeness": reward_shaping.CLOSENESS_FINAL,
            "touch": reward_shaping.TOUCH_FINAL,
            "direction": reward_shaping.DIRECTION_FINAL,
        }


def apply_reward_shaping(
    reward: float, info: Dict, weights: Optional[Dict[str, float]]
) -> float:
    """Apply reward shaping to environment reward."""
    if weights is None:
        return reward

    shaped_reward = reward
    shaped_reward += info.get("reward_closeness_to_puck", 0.0) * weights["closeness"]
    shaped_reward += info.get("reward_touch_puck", 0.0) * weights["touch"]
    shaped_reward += info.get("reward_puck_direction", 0.0) * weights["direction"]
    return shaped_reward


# ============================================================================
# Reward Bonus (for TDMPC2)
# ============================================================================


def get_reward_bonus_values(phase_local_episode: int, phase_config) -> Optional[tuple]:
    """
    Phase-local reward bonus (win_bonus, win_discount) from phase.reward_bonus.
    Uses N, K, WIN_BONUS_START/FINAL, WIN_DISCOUNT_START/FINAL (linear interpolation over K).
    Returns (win_reward_bonus, win_reward_discount) or None if phase has no reward_bonus.
    """
    reward_bonus = getattr(phase_config, "reward_bonus", None)
    if reward_bonus is None:
        return None
    N = reward_bonus.N
    K = reward_bonus.K
    if phase_local_episode < N:
        return (reward_bonus.WIN_BONUS_START, reward_bonus.WIN_DISCOUNT_START)
    elif phase_local_episode < N + K:
        alpha = (phase_local_episode - N) / K
        win_bonus = (
            reward_bonus.WIN_BONUS_START * (1 - alpha)
            + reward_bonus.WIN_BONUS_FINAL * alpha
        )
        win_discount = (
            reward_bonus.WIN_DISCOUNT_START * (1 - alpha)
            + reward_bonus.WIN_DISCOUNT_FINAL * alpha
        )
        return (win_bonus, win_discount)
    else:
        return (reward_bonus.WIN_BONUS_FINAL, reward_bonus.WIN_DISCOUNT_FINAL)


def update_buffer_reward_bonus(
    agent, win_reward_bonus: float, win_reward_discount: float
) -> None:
    """
    Write reward bonus values into agent.buffer if it supports them.
    No-op for agents whose buffer has no win_reward_bonus/win_reward_discount (e.g. SAC, TD3).
    """
    buffer = getattr(agent, "buffer", None)
    if buffer is None:
        return
    if hasattr(buffer, "win_reward_bonus"):
        buffer.win_reward_bonus = win_reward_bonus
    if hasattr(buffer, "win_reward_discount"):
        buffer.win_reward_discount = win_reward_discount


def _compute_backprop_reward(
    agent: Optional[Agent],
    episode_rewards: Optional[List[float]],
    winner: int,
    fallback_shaped_reward: float,
) -> float:
    """Compute backprop_reward when buffer has win_reward_bonus/discount; else use shaped_reward."""
    if (
        agent is None
        or not hasattr(agent, "buffer")
        or agent.buffer is None
        or not hasattr(agent.buffer, "win_reward_bonus")
        or not hasattr(agent.buffer, "win_reward_discount")
        or episode_rewards is None
        or len(episode_rewards) == 0
    ):
        return fallback_shaped_reward
    buf = agent.buffer
    rewards_out, _, _ = apply_win_reward_backprop(
        np.array(episode_rewards, dtype=np.float32),
        winner,
        win_reward_bonus=buf.win_reward_bonus,
        win_reward_discount=buf.win_reward_discount,
    )
    return float(np.sum(rewards_out))


def _save_csvs_and_plots(
    episode_logs: List[Dict[str, Any]],
    training_metrics: TrainingMetrics,
    csvs_dir: str,
    result_dir: str,
    run_name: str,
    verbose: bool = True,
    window_size: int = 250,
) -> None:
    """Write episode logs, losses, winrates CSVs and training metrics plots."""
    if episode_logs:
        fixed_cols = {
            "episode",
            "step",
            "reward",
            "shaped_reward",
            "backprop_reward",
            "total_gradient_steps",
            "rating",
            "length",
            "phase",
        }
        loss_keys = sorted(
            set().union(*(ep.keys() for ep in episode_logs)) - fixed_cols
        )
        fixed_cols_list = [
            "episode",
            "step",
            "reward",
            "shaped_reward",
            "backprop_reward",
            "total_gradient_steps",
            "rating",
            "length",
            "phase",
        ]
        episode_csv_path = os.path.join(csvs_dir, f"{run_name}_episode_logs.csv")
        with open(episode_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fixed_cols_list + loss_keys,
                extrasaction="ignore",
            )
            writer.writeheader()
            for ep in episode_logs:
                row = {k: ep.get(k, "") for k in fixed_cols_list}
                row.update({k: ep.get(k, "") for k in loss_keys})
                writer.writerow(row)
        if verbose:
            logger.info("Episode logs saved to %s", episode_csv_path)

    if training_metrics.updates:
        all_keys = sorted(
            set().union(*(u.keys() for u in training_metrics.updates)) - {"step"}
        )
        losses_csv_path = os.path.join(csvs_dir, f"{run_name}_losses.csv")
        with open(losses_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["step"] + all_keys,
                extrasaction="ignore",
            )
            writer.writeheader()
            for u in training_metrics.updates:
                row = {"step": u["step"]}
                for k in all_keys:
                    val = u.get(k)
                    row[k] = _to_scalar_loss(val) if val is not None else ""
                writer.writerow(row)
        if verbose:
            logger.info("Losses saved to %s", losses_csv_path)

    if training_metrics.winrates:
        winrate_csv_path = os.path.join(csvs_dir, f"{run_name}_winrates.csv")
        with open(winrate_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "winrate"])
            for wr in training_metrics.winrates:
                writer.writerow([wr["step"], wr["winrate"]])
        if verbose:
            logger.info("Winrates saved to %s", winrate_csv_path)

    if training_metrics.episodes or training_metrics.updates:
        plot_training_metrics(
            training_metrics,
            result_dir=result_dir,
            run_name=run_name,
            window_size=window_size,
        )


def _get_opponent_actions(
    opponents: List[Opponent], obs_opponent: np.ndarray, num_envs: int
) -> np.ndarray:
    def get_action(opponent, obs):
        if opponent is None:
            return np.zeros(4)
        elif isinstance(opponent, h_env.BasicOpponent):
            return opponent.act(obs)
        elif isinstance(opponent, Agent):
            return opponent.act(obs.astype(np.float32))

    return np.array(
        [get_action(opponents[i][0], obs_opponent[i]) for i in range(num_envs)]
    )


def _get_agent_actions_with_t0(
    agent: Agent, states: np.ndarray, num_envs: int, t0_flags: np.ndarray
) -> np.ndarray:
    """Get batched agent actions with t0 flags for agents that need episode-start signals (e.g. TDMPC2).
    
    Args:
        agent: The agent to get actions from
        states: (num_envs, state_dim) observations
        num_envs: Number of parallel environments
        t0_flags: (num_envs,) bool array indicating which envs are at episode start
        
    Returns:
        actions: (num_envs, action_dim) continuous actions
    """
    # Check if agent supports batched actions with t0
    if hasattr(agent, "act_batch"):
        # Try to pass t0s parameter (TDMPC2 supports this)
        try:
            return agent.act_batch(states.astype(np.float32), t0s=t0_flags)
        except TypeError:
            # Agent doesn't support t0s parameter (SAC/TD3)
            return agent.act_batch(states.astype(np.float32))
    else:
        # Fallback: call act() for each environment individually
        actions = []
        for i in range(num_envs):
            try:
                action = agent.act(states[i].astype(np.float32), t0=t0_flags[i])
            except TypeError:
                action = agent.act(states[i].astype(np.float32))
            actions.append(action)
        return np.array(actions)


def _switch_phase(
    curriculum: CurriculumConfig,
    training_state: TrainingState,
    state_dim: int,
    action_dim: int,
    num_envs: int,
    env: Optional[VectorizedHockeyEnvOptimized] = None,
    agent: Optional[Agent] = None,
    matchmaker: Optional[Matchmaker] = None,
    verbose: bool = True,
) -> Tuple[VectorizedHockeyEnvOptimized, np.ndarray, List]:
    """Check and handle curriculum phase transitions. Returns updated env, states, and opponents."""
    new_phase_index, _, new_phase = get_phase_for_episode(
        curriculum, training_state.episode
    )

    training_state.phase_index = new_phase_index
    training_state.phase = new_phase

    if verbose:
        logger.info(
            "Starting phase %d/%d: %s",
            new_phase_index + 1,
            len(curriculum.phases),
            new_phase.name,
        )

    # Recreate environment
    if env:
        env.close()
    mode_str = new_phase.environment.get_mode_for_episode(0)
    env_mode = getattr(h_env.Mode, mode_str)
    env = VectorizedHockeyEnvOptimized(
        num_envs=num_envs,
        env_fn=partial(
            _make_hockey_env, mode=env_mode, keep_mode=new_phase.environment.keep_mode
        ),
    )
    states = env.reset()

    # Create opponents
    opponents = [
        matchmaker.get_opponent(new_phase.opponent, training_state.rating.rating)
        for _ in range(num_envs)
    ]

    # Clear agent buffer if specified
    if agent and new_phase.clear_buffer:
        training_state.last_warmup_reset_step = training_state.step
        buffer_size = agent.buffer.size
        agent.buffer.clear()
        if verbose:
            logger.info("Cleared agent replay buffer (size was %d)", buffer_size)

    return env, states, opponents


def _get_resume_state(resume_from: str) -> Tuple[str, str, Dict[str, Any]]:
    """Infer run_name, latest checkpoint path and metadata from an existing run directory.

    Returns:
        (run_name, checkpoint_path, metadata). metadata may contain episode, phase_index, step, etc.
    """
    base = Path(resume_from)
    configs_dir = base / "configs"
    models_dir = base / "models"
    if not configs_dir.is_dir():
        raise FileNotFoundError(
            f"Resume directory has no configs/: {resume_from}"
        )
    if not models_dir.is_dir():
        raise FileNotFoundError(
            f"Resume directory has no models/: {resume_from}"
        )
    config_files = list(configs_dir.glob("*.json"))
    if not config_files:
        raise FileNotFoundError(
            f"No config JSON found in {configs_dir}"
        )
    run_name = config_files[0].stem

    pt_files = list(models_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(
            f"No .pt checkpoints found in {models_dir}"
        )
    ep_re = re.compile(r"_ep(\d+)\.pt$")

    def episode_from_path(p: Path) -> int:
        m = ep_re.search(p.name)
        return int(m.group(1)) if m else 0

    pt_files.sort(key=episode_from_path)
    latest_pt = pt_files[-1]
    checkpoint_path = str(latest_pt)
    episode = episode_from_path(latest_pt)

    metadata = {"episode": episode}
    metadata_path = latest_pt.parent / (latest_pt.stem + "_metadata.json")
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata.update(json.load(f))

    return run_name, checkpoint_path, metadata


def train_vectorized(
    config_path: str,
    verbose: bool = True,
    result_dir: str = "./results/minimal_runs",
    archive_dir: str = "./archive",
    num_envs: int = 4,
    resume_from: Optional[str] = None,
):
    """
    Minimal training with multiple parallel environments.

    Args:
        config_path: Path to curriculum config file
        verbose: Print episode info
        result_dir: Directory to save results (ignored if resume_from is set)
        archive_dir: Directory for agent archive
        num_envs: Number of parallel environments
        resume_from: If set, path to an existing run directory to continue training from
            (e.g. results/tdmpc2_runs/2026-02-01_09-55-22). Uses latest checkpoint in models/.
    """

    # Load curriculum and determine episodes
    curriculum = load_curriculum(config_path)
    total_episodes = get_total_episodes(curriculum)

    if verbose:
        logger.info(
            "Curriculum: %d phases, %d episodes", len(curriculum.phases), total_episodes
        )

    is_resume = resume_from is not None
    if is_resume:
        run_manager = RunManager(existing_run_dir=resume_from)
        run_name, checkpoint_path, resume_metadata = _get_resume_state(resume_from)
        if verbose:
            logger.info(
                "Resuming from %s: run_name=%s, checkpoint=%s, episode=%s",
                resume_from,
                run_name,
                checkpoint_path,
                resume_metadata.get("episode"),
            )
    else:
        run_manager = RunManager(base_output_dir=result_dir)
        config_dict = {
            "agent": {
                "type": curriculum.agent.type,
                "hyperparameters": curriculum.agent.hyperparameters,
            },
            "hyperparameters": curriculum.hyperparameters,
        }
        run_name = run_manager.generate_run_name(config_dict)
        run_manager.save_config(run_name, config_dict)
        shutil.copy(config_path, os.path.join(str(run_manager.base_output_dir), "config.json"))
        if verbose:
            logger.info("Parameters (config):")
            logger.info("\n%s", json.dumps(config_dict, indent=2))

    result_dir = str(run_manager.base_output_dir)
    csvs_dir = str(run_manager.csvs_dir)
    plots_dir = str(run_manager.plots_dir)
    models_dir = str(run_manager.models_dir)

    # Setup archive
    archive = Archive(base_dir=archive_dir)
    matchmaker = Matchmaker(archive)
    rating_system = RatingSystem(archive)

    training_state = TrainingState()
    if is_resume:
        training_state.episode = resume_metadata.get("episode", 0)
        training_state.phase_index = resume_metadata.get("phase_index")
        if training_state.phase_index is not None:
            training_state.phase = curriculum.phases[training_state.phase_index]
        step = resume_metadata.get("step")
        if step is None:
            step = training_state.episode * 100
        training_state.step = step
        training_state.last_checkpoint_step = step
        training_state.last_eval_step = step
        if "gradient_steps" in resume_metadata:
            training_state.gradient_steps = resume_metadata["gradient_steps"]

    # Create agent
    temp_env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0] // 2
    temp_env.close()

    agent = create_agent(
        curriculum.agent,
        state_dim=state_dim,
        action_dim=action_dim,
        common_hyperparams=curriculum.hyperparameters,
        deterministic=False,
    )

    if is_resume:
        agent.load(checkpoint_path)
        if verbose:
            logger.info("Loaded agent from %s", checkpoint_path)

    logger.info("Agent network architecture:")
    logger.info("\n%s\n", agent.log_architecture())

    # Setup initial phase
    env, states, opponents = _switch_phase(
        curriculum,
        training_state,
        state_dim=state_dim,
        action_dim=action_dim,
        num_envs=num_envs,
        env=None,
        agent=agent,
        matchmaker=matchmaker,
        verbose=verbose,
    )
    switch = False

    # Calculate initial reward weights based on phase config (for reward shaping)
    _, phase_local_episode, _ = get_phase_for_episode(curriculum, training_state.episode)
    reward_weights = calculate_reward_weights(phase_local_episode, training_state.phase)
    
    # Initialize buffer reward bonus if agent supports it (TDMPC2)
    bonus_values = get_reward_bonus_values(phase_local_episode, training_state.phase)
    if bonus_values is not None:
        update_buffer_reward_bonus(agent, bonus_values[0], bonus_values[1])

    # Setup metrics
    training_metrics = TrainingMetrics()
    episode_logs: List[Dict[str, Any]] = []
    episode_metrics = [EpisodeMetrics() for _ in range(num_envs)]
    last_train_metrics: Dict[str, Any] = {}
    
    # Track t0 (episode start) flags for each environment
    t0_flags = np.ones(num_envs, dtype=bool)  # All envs start at episode beginning
    
    # Call on_episode_start for initial episodes
    if hasattr(agent, "on_episode_start"):
        agent.on_episode_start(training_state.episode)

    # Log initial buffer state
    if verbose and hasattr(agent, 'buffer'):
        logger.info("Initial buffer: size=%d, horizon=%d, batch_size=%d", 
                    agent.buffer.size, 
                    getattr(agent.buffer, 'horizon', 0),
                    getattr(agent.buffer, 'batch_size', 0))

    while training_state.episode < total_episodes:
        # Get batched agent actions (with t0 flags for TDMPC2)
        actions = _get_agent_actions_with_t0(agent, states, num_envs, t0_flags)
        
        # Reset t0 flags after first step of each episode
        t0_flags = np.zeros(num_envs, dtype=bool)

        # Get opponent actions
        obs_opponent = env.obs_agent_two()
        if hasattr(agent, "collect_opponent_demonstrations"):
            for i in range(num_envs):
                agent.collect_opponent_demonstrations(obs_opponent[i])
        actions_opponent = _get_opponent_actions(opponents, obs_opponent, num_envs)

        # Step all environments
        full_actions = np.hstack([actions, actions_opponent])
        next_states, rewards, dones, truncs, infos = env.step(full_actions)

        for i in range(num_envs):
            # Apply reward shaping based on curriculum phase config
            shaped_reward = apply_reward_shaping(rewards[i], infos[i], reward_weights)
            scaled_reward = shaped_reward * curriculum.training.reward_scale

            # Extract winner for TDMPC2 buffer (needed for reward backpropagation)
            winner = infos[i].get("winner", 0)

            # Done flag logic:
            # - TDMPC2 (episode-based): needs done=True for any ending to flush episode
            # - SAC/TD3 (transition-based): needs done=True only for terminal states
            #   (not truncations, for proper TD target computation)
            # Check if this is an episode-based buffer (TDMPC2)
            is_episode_buffer = hasattr(agent.buffer, "sample_sequences")

            if is_episode_buffer:
                # TDMPC2: flush episode on any ending
                done = dones[i] or truncs[i]
            else:
                # SAC/TD3: only mark terminal states as done, not truncations
                done = dones[i]

            # For terminal transitions, use the terminal observation (from info)
            # instead of next_states which is now the reset observation
            if dones[i] or truncs[i]:
                next_state_for_buffer = infos[i].get('terminal_obs', next_states[i])
                # Log warning if terminal_obs was missing (indicates vectorized_env issue)
                if 'terminal_obs' not in infos[i] and training_state.episode < 3:
                    logger.warning("terminal_obs not found in info for env %d - using next_states (may be reset obs)", i)
            else:
                next_state_for_buffer = next_states[i]

            agent.store_transition(
                (states[i], actions[i], scaled_reward, next_state_for_buffer, done),
                winner=winner,
                env_id=i,  # Pass env_id for proper per-environment episode tracking
            )

            episode_metrics[i].reward += rewards[i]
            episode_metrics[i].shaped_reward += scaled_reward
            episode_metrics[i].step_rewards.append(scaled_reward)
            episode_metrics[i].length += 1

            # Handle episode completion
            if dones[i] or truncs[i]:
                # Call on_episode_end callback before completing episode
                if hasattr(agent, "on_episode_end"):
                    agent.on_episode_end(training_state.episode)
                
                # Update metrics
                training_state.episode += 1
                
                # Log first episode completion for debugging
                if training_state.episode == 1 and verbose:
                    logger.info("First episode complete: length=%d, buffer_size=%d, done=%s, trunc=%s, winner=%d",
                               episode_metrics[i].length, agent.buffer.size, dones[i], truncs[i], winner)

                training_state.rating = rating_system.estimate_rating(
                    training_state.rating, opponents[i][1], infos[i]["winner"]
                )
                backprop_reward = _compute_backprop_reward(
                    agent,
                    episode_metrics[i].step_rewards,
                    winner,
                    episode_metrics[i].shaped_reward,
                )

                env_mode_str = training_state.phase.environment.get_mode_for_episode(0)
                opponent_type = getattr(
                    training_state.phase.opponent, "type", training_state.phase.name
                )

                if verbose:
                    log_parts = [
                        f"Episode {training_state.episode}: reward={episode_metrics[i].reward:.2f}",
                        f"shaped_reward={episode_metrics[i].shaped_reward:.2f}",
                        f"backprop_reward={backprop_reward:.2f}",
                        f"steps={episode_metrics[i].length}",
                        f"phase={training_state.phase.name}",
                        f"rating={training_state.rating.rating:.2f}",
                        f"buffer={agent.buffer.size}",
                        f"env={env_mode_str}",
                        f"opponent={opponent_type}",
                    ]
                    log_parts.extend(_format_losses_for_log(last_train_metrics))
                    logger.info(" | ".join(log_parts))

                training_metrics.add_episode(
                    step=training_state.step,
                    rating=training_state.rating.rating,
                    reward=episode_metrics[i].reward,
                    length=episode_metrics[i].length,
                    phase=training_state.phase.name,
                )
                episode_row: Dict[str, Any] = {
                    "episode": training_state.episode,
                    "step": training_state.step,
                    "reward": episode_metrics[i].reward,
                    "shaped_reward": episode_metrics[i].shaped_reward,
                    "backprop_reward": backprop_reward,
                    "total_gradient_steps": training_state.gradient_steps,
                    "rating": training_state.rating.rating,
                    "length": episode_metrics[i].length,
                    "phase": training_state.phase.name,
                }
                for k, v in (last_train_metrics or {}).items():
                    episode_row[k] = _to_scalar_loss(v)
                episode_logs.append(episode_row)
                episode_metrics[i].reset()
                
                # Mark this environment as starting a new episode (for t0 flag)
                t0_flags[i] = True
                
                # Call on_episode_start callback for next episode
                if hasattr(agent, "on_episode_start"):
                    agent.on_episode_start(training_state.episode)

                # Check for phase transition
                new_phase_index, new_phase_local_episode, _ = get_phase_for_episode(
                    curriculum, training_state.episode
                )
                if new_phase_index != training_state.phase_index:
                    switch = True
                else:
                    # Update reward weights and bonus for new episode (within same phase)
                    reward_weights = calculate_reward_weights(new_phase_local_episode, training_state.phase)
                    bonus_values = get_reward_bonus_values(new_phase_local_episode, training_state.phase)
                    if bonus_values is not None:
                        update_buffer_reward_bonus(agent, bonus_values[0], bonus_values[1])

                # Sample new opponent
                opponent, opponent_rating = matchmaker.get_opponent(
                    training_state.phase.opponent, training_state.rating.rating
                )
                opponents[i] = (opponent, opponent_rating)

        # Step or switch phase
        if switch:
            switch = False

            env, new_states, opponents = _switch_phase(
                curriculum,
                training_state,
                state_dim=state_dim,
                action_dim=action_dim,
                num_envs=num_envs,
                env=env,
                agent=agent,
                matchmaker=matchmaker,
                verbose=verbose,
            )
            states = new_states
            # After phase switch, all environments start fresh episodes
            t0_flags = np.ones(num_envs, dtype=bool)
            
            # Update reward weights and bonus for new phase
            _, phase_local_episode, _ = get_phase_for_episode(curriculum, training_state.episode)
            reward_weights = calculate_reward_weights(phase_local_episode, training_state.phase)
            bonus_values = get_reward_bonus_values(phase_local_episode, training_state.phase)
            if bonus_values is not None:
                update_buffer_reward_bonus(agent, bonus_values[0], bonus_values[1])
        else:
            states = next_states

        # Train (respect warmup and train_freq settings)
        warmup_passed = (
            training_state.step - curriculum.training.warmup_steps
            >= training_state.last_warmup_reset_step
        )
        train_freq_ok = training_state.step % curriculum.training.train_freq == 0
        
        if warmup_passed and train_freq_ok:
            metrics = agent.train(steps=curriculum.training.updates_per_step)
            training_state.gradient_steps += curriculum.training.updates_per_step
            if metrics:
                last_train_metrics = metrics
                training_metrics.add_update(step=training_state.step, **metrics)
            
            # Log first training step for debugging
            if training_state.gradient_steps == curriculum.training.updates_per_step and verbose:
                logger.info("First training at step %d, buffer_size=%d, metrics=%s",
                           training_state.step, agent.buffer.size, 
                           {k: f"{v:.4f}" if isinstance(v, float) else v 
                            for k, v in (metrics or {}).items() if 'loss' in k.lower()})

        training_state.step += num_envs

        # Save model checkpoint, CSVs and plots at same frequency
        # TODO when to add to archive?
        if (
            training_state.step - curriculum.training.checkpoint_frequency
            >= training_state.last_checkpoint_step
        ):
            training_state.last_checkpoint_step = training_state.step
            agent.save(
                os.path.join(
                    models_dir, f"{run_name}_ep{training_state.episode:06d}.pt"
                )
            )
            _save_csvs_and_plots(
                episode_logs=episode_logs,
                training_metrics=training_metrics,
                csvs_dir=csvs_dir,
                result_dir=result_dir,
                run_name=run_name,
                verbose=verbose,
                window_size=250,
            )

        # Run evaluation
        # TODO outsource
        if (
            training_state.step - curriculum.training.eval_frequency
            >= training_state.last_eval_step
        ):
            training_state.last_eval_step = training_state.step
            if verbose:
                logger.info("Evaluating agent at step %d...", training_state.step)
            winrate = evaluate_winrate(agent, opponent_weak=False, verbose=verbose)
            training_metrics.add_winrate(step=training_state.step, winrate=winrate)
            if verbose:
                logger.info("Evaluation: Win rate: %.2f%%", winrate * 100.0)

    # Cleanup
    env.close()
    agent.save(
        os.path.join(models_dir, f"{run_name}_ep{training_state.episode:06d}.pt")
    )

    if verbose:
        logger.info("Training complete. Total steps: %d", training_state.step)

    # Final save of CSV files and training plots
    _save_csvs_and_plots(
        episode_logs=episode_logs,
        training_metrics=training_metrics,
        csvs_dir=csvs_dir,
        result_dir=result_dir,
        run_name=run_name,
        verbose=verbose,
        window_size=100,
    )

    # Plot value propagation
    opponent = h_env.BasicOpponent(weak=False)

    model_files = sorted([f for f in os.listdir(models_dir) if f.endswith(".pt")])

    agent.load(os.path.join(models_dir, model_files[-1]))
    trajectories = sample_trajectories(agent, opponent)

    all_means = []
    all_variances = []
    for model_file in model_files:
        agent.load(os.path.join(models_dir, model_file))
        means, variances = evaluate_episodes(agent, trajectories)
        all_means.append(means)
        all_variances.append(variances)

    plot_value_heatmap(
        all_means, path=os.path.join(plots_dir, "value_propagation_heatmap.png")
    )
    means = [all_means[0], all_means[-1]]
    variances = [all_variances[0], all_variances[-1]]
    labels = ["Untrained", "Trained"]
    plot_values_line(
        means,
        all_variances=variances,
        path=os.path.join(plots_dir, "value_propagation_line.png"),
        labels=labels,
    )

    # Plot winrates
    if training_metrics.winrates:
        steps = [wr["step"] for wr in training_metrics.winrates]
        winrates = [wr["winrate"] for wr in training_metrics.winrates]

        plt.figure(figsize=(12, 6))
        plt.plot(
            steps, winrates, marker="o", linewidth=2, markersize=6, label="Winrate"
        )
        plt.xlabel("Training Steps")
        plt.ylabel("Winrate")
        plt.title("Winrate over Training")
        plt.ylim(0, 1.0)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "winrates.png"))
        plt.close()


if __name__ == "__main__":
    train_vectorized(
        config_path="./configs/curriculum_sac_selfplay.json",
        result_dir="./results/sac_runs",
        num_envs=16,
    )
