"""
Minimal training script for debugging.
No curriculum, just SAC vs weak opponent with basic logging.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from functools import partial
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import hockey.hockey_env as h_env
from rl_hockey.common.agent import Agent
from rl_hockey.common.archive import Archive, Matchmaker, RatingSystem, Rating
from rl_hockey.common.archive.matchmaker import Opponent
from rl_hockey.sac import SAC
from rl_hockey.common.vectorized_env import VectorizedHockeyEnvOptimized
from rl_hockey.common.training.curriculum_manager import (
    load_curriculum,
    get_phase_for_episode,
    get_total_episodes,
    CurriculumConfig,
    PhaseConfig,
)
from rl_hockey.common.training.opponent_manager import (
    sample_opponent,
    get_opponent_action,
)
from rl_hockey.common.evaluation.value_propagation import (
    evaluate_episodes,
    plot_value_heatmap,
    plot_values_line,
    sample_trajectories,
)
from rl_hockey.common.evaluation.winrate_evaluator import evaluate_winrate
from datetime import datetime
import shutil


@dataclass
class TrainingState:
    """Holds the state of training."""
    step: int = 0
    episode: int = 0
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

    def add_episode(self, step: int, rating: float, reward: float, length: int, phase: Optional[str] = None):
        """Records an episode finish at a specific global step."""
        episode_data = {"step": step, "rating": rating, "reward": reward, "length": length}
        if phase:
            episode_data["phase"] = phase
        self.episodes.append(episode_data)

    def add_update(self, step: int, **metrics):
        """Records training metrics at a specific global step."""
        self.updates.append({
            "step": step,
            **metrics
        })
    
    def add_winrate(self, step: int, winrate: float):
        """Records winrate evaluation at a specific global step."""
        self.winrates.append({
            "step": step,
            "winrate": winrate,
        })


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    reward: float = 0.0
    length: int = 0
    
    def reset(self):
        """Reset episode metrics."""
        self.reward = 0.0
        self.length = 0


def _make_hockey_env(mode, keep_mode):
    """Factory function for creating HockeyEnv instances (must be at module level for pickling)."""
    return h_env.HockeyEnv(mode=mode, keep_mode=keep_mode)


def _create_opponents(num_envs: int, phase: PhaseConfig, state_dim: int, action_dim: int, rating: Optional[float] = None, matchmaker: Optional[Matchmaker] = None) -> List:
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


def _get_opponent_actions(opponents: List[Opponent], obs_opponent: np.ndarray, num_envs: int) -> np.ndarray:
    def get_action(opponent, obs):
        if opponent is None:
            return np.zeros(4)
        elif isinstance(opponent, h_env.BasicOpponent):
            return opponent.act(obs)
        elif isinstance(opponent, Agent):
            return opponent.act(obs.astype(np.float32))

    return np.array([get_action(opponents[i][0], obs_opponent[i]) for i in range(num_envs)])


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
    new_phase_index, _, new_phase = get_phase_for_episode(curriculum, training_state.episode)

    training_state.phase_index = new_phase_index
    training_state.phase = new_phase
    
    if verbose:
        print(f"\n  -> Starting phase {new_phase_index + 1}/{len(curriculum.phases)}: {new_phase.name}")
    
    # Recreate environment
    if env:
        env.close()
    mode_str = new_phase.environment.get_mode_for_episode(0)
    env_mode = getattr(h_env.Mode, mode_str)
    env = VectorizedHockeyEnvOptimized(
        num_envs=num_envs,
        env_fn=partial(_make_hockey_env, mode=env_mode, keep_mode=new_phase.environment.keep_mode),
    )
    states = env.reset()

    # Create opponents
    opponents = [matchmaker.get_opponent(new_phase.opponent, training_state.rating.rating) for _ in range(num_envs)]

    # Clear agent buffer if specified
    if agent and new_phase.clear_buffer:
        training_state.last_warmup_reset_step = training_state.step

        agent.buffer.clear()
        if verbose:
            print("  -> Cleared agent replay buffer")

    return env, states, opponents


def train_vectorized(
    config_path: str,
    verbose: bool = True,
    result_dir: str = "./results/minimal_runs",
    archive_dir: str = "./archive",
    num_envs: int = 4,
):
    """
    Minimal training with multiple parallel environments.
    
    Args:
        config_path: Path to curriculum config file
        verbose: Print episode info
        result_dir: Directory to save results
        archive_dir: Directory for agent archive
        num_envs: Number of parallel environments
    """
    # TODO support for loading existing training state

    # Load curriculum and determine episodes
    curriculum = load_curriculum(config_path)
    total_episodes = get_total_episodes(curriculum)

    if verbose:
        print(f"Curriculum: {len(curriculum.phases)} phases, {total_episodes} episodes")

    # Create result directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = os.path.join(result_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)

    shutil.copy(config_path, os.path.join(result_dir, "config.json"))

    # Setup archive
    archive = Archive(base_dir=archive_dir)
    matchmaker = Matchmaker(archive)
    rating_system = RatingSystem(archive)
    
    training_state = TrainingState()

    # Determine state & action dimensions
    temp_env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0] // 2
    temp_env.close()
    
    # TODO agent loading
    agent = SAC(state_dim=state_dim, action_dim=action_dim, noise="pink")
    agent.load("results/minimal_runs/8/models/final.pt")

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
    
    # TODO Initialize evaluator
    # evaluator = None
    # if eval_frequency > 0:
    #     evaluator = Evaluator(
    #         env_fn=partial(_make_hockey_env, mode=h_env.Mode.NORMAL, keep_mode=True),
    #         num_eval_episodes=num_eval_episodes,
    #         max_steps=max_steps,
    #         eval_frequency=eval_frequency,
    #     )

    # Setup metrics
    training_metrics = TrainingMetrics()
    episode_metrics = [EpisodeMetrics() for _ in range(num_envs)]

    pbar = tqdm.tqdm(total=total_episodes, unit="ep")
    while training_state.episode < total_episodes:
        # Get batched agent actions
        actions = agent.act_batch(states.astype(np.float32))
        
        # Get opponent actions
        obs_opponent = env.obs_agent_two()
        actions_opponent = _get_opponent_actions(opponents, obs_opponent, num_envs)
        
        # Step all environments
        full_actions = np.hstack([actions, actions_opponent])
        next_states, rewards, dones, truncs, infos = env.step(full_actions)
        
        for i in range(num_envs):
            reward = rewards[i] * curriculum.training.reward_scale
            done = dones[i] and infos[i]["winner"] == 0  # handle truncation (env always returns done)
            agent.store_transition((states[i], actions[i], reward, next_states[i], done))
            
            episode_metrics[i].reward += rewards[i]
            episode_metrics[i].length += 1
            
            # Handle episode completion
            if dones[i] or truncs[i]:
                # Update metrics
                training_state.episode += 1

                training_state.rating = rating_system.estimate_rating(training_state.rating, opponents[i][1], infos[i]["winner"])

                pbar.update(1)
                pbar.set_postfix({
                    "rating": f"{training_state.rating.rating:.2f}",
                    "reward": f"{episode_metrics[i].reward:.2f}",
                    "length": episode_metrics[i].length,
                    "buffer": agent.buffer.size,
                    "phase": training_state.phase.name,
                })

                training_metrics.add_episode(
                    step=training_state.step,
                    rating=training_state.rating.rating,
                    reward=episode_metrics[i].reward,
                    length=episode_metrics[i].length,
                    phase=training_state.phase.name,
                )
                episode_metrics[i].reset()

                # Check for phase transition
                new_phase_index, _, _ = get_phase_for_episode(curriculum, training_state.episode)
                if new_phase_index != training_state.phase_index:
                    switch = True

                # Sample new opponent
                opponent, opponent_rating = matchmaker.get_opponent(training_state.phase.opponent, training_state.rating.rating)
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
        else:
            states = next_states

        # Train
        if training_state.step - curriculum.training.warmup_steps > training_state.last_warmup_reset_step:
            metrics = agent.train(steps=curriculum.training.updates_per_step)
            training_metrics.add_update(step=training_state.step, **metrics)
        
        training_state.step += num_envs

        # Save model checkpoint
        if training_state.step - curriculum.training.checkpoint_frequency >= training_state.last_checkpoint_step:
            training_state.last_checkpoint_step = training_state.step
            agent.save(f"{result_dir}/models/ep_{training_state.episode}.pt")

        # Run evaluation
        # TODO outsource
        if training_state.step - curriculum.training.eval_frequency >= training_state.last_eval_step:
            training_state.last_eval_step = training_state.step

            winrate = evaluate_winrate(agent, opponent_weak=False, verbose=verbose)
            training_metrics.add_winrate(step=training_state.step, winrate=winrate)
    
    # Cleanup
    pbar.close()
    env.close()
    agent.save(f"{result_dir}/models/final.pt")
    
    if verbose:
        print(f"\nTraining complete. Total steps: {training_state.step}")
    
    # Plotting
    # TODO outsource
    os.makedirs(f"{result_dir}/plots", exist_ok=True)

    if training_metrics.episodes:
        episode_rewards = [ep["reward"] for ep in training_metrics.episodes]
        plt.figure(figsize=(12, 6))
        plt.plot(episode_rewards, label="Episode Reward", alpha=0.6)
        # Add rolling average
        if len(episode_rewards) > 100:
            window = 100
            rolling_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(episode_rewards)), rolling_avg, label=f"Rolling Avg ({window})", linewidth=2)
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Episode Rewards")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f"{result_dir}/plots/rewards.png")
        plt.close()
    
    if training_metrics.updates:
        # Extract unique metric names (excluding 'step')
        metric_names = set()
        for update in training_metrics.updates:
            metric_names.update(k for k in update.keys() if k != "step")
        metric_names = sorted(metric_names)
        
        if metric_names:
            num_metrics = len(metric_names)
            fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics))
            if num_metrics == 1:
                axes = [axes]
            
            for idx, metric_name in enumerate(metric_names):
                steps = [u["step"] for u in training_metrics.updates if metric_name in u]
                values = [u[metric_name] for u in training_metrics.updates if metric_name in u]
                
                axes[idx].plot(steps, values, label=metric_name, alpha=0.7)
                axes[idx].set_xlabel("Steps")
                axes[idx].set_ylabel(metric_name)
                axes[idx].set_title(metric_name)
                axes[idx].legend()
                axes[idx].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{result_dir}/plots/training_metrics.png")
            plt.close()    

    # Plot value propagation
    opponent = h_env.BasicOpponent(weak=False)
    
    models_dir = f"{result_dir}/models/"
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

    plot_value_heatmap(all_means, path=f"{result_dir}/plots/value_propagation_heatmap.png")
    means = [all_means[0], all_means[-1]]
    variances = [all_variances[0], all_variances[-1]]
    labels = ["Untrained", "Trained"]
    plot_values_line(means, all_variances=variances, path=f"{result_dir}/plots/value_propagation_line.png", labels=labels)
    
    # Plot winrates
    if training_metrics.winrates:
        steps = [wr["step"] for wr in training_metrics.winrates]
        winrates = [wr["winrate"] for wr in training_metrics.winrates]
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, winrates, marker='o', linewidth=2, markersize=6, label="Winrate")
        plt.xlabel("Training Steps")
        plt.ylabel("Winrate")
        plt.title("Winrate over Training")
        plt.ylim(0, 1.0)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(f"{result_dir}/plots/winrates.png")
        plt.close()

if __name__ == "__main__":
    train_vectorized(
        config_path="./configs/curriculum_sac_selfplay.json",
        result_dir="./results/sac_runs",
        num_envs=16,
    )
