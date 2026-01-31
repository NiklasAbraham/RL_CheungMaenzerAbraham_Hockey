"""Train a decoy policy to mimic agent behavior."""

import json
import logging
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import hockey.hockey_env as h_env
import numpy as np

from rl_hockey.Decoy_Policy.decoy_policy import DecoyPolicy

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Project root: parent of src/ (script lives in src/rl_hockey/scripts/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

CONFIG = {
    "target": {
        "type": "basic_strong",
        "path": None,
    },
    "opponents": [
        {"type": "basic_weak", "weight": 0.3},
        {"type": "basic_strong", "weight": 0.2},
        {
            "type": "TDMPC2",
            "weight": 0.25,
            "path": "results/reference_bots/TDMPC2/TDMPC2_run_lr3e04_bs512_hencoder_dynamics_reward_termination_q_function_policy_cfce4de1_20260123_210009_ep009200.pt",
        },
        {
            "type": "SAC",
            "weight": 0.25,
            "path": "results/reference_bots/SAC/run_lr1e03_bs256_h128_128_128_4c1f51eb_20260111_140638_vec24.pt",
        },
    ],
    "decoy": {
        "hidden_layers": [256, 256],
        "learning_rate": 3e-4,
        "buffer_max_size": 100_000,
    },
    "training": {
        "num_episodes": 20_000,
        "max_episode_steps": 500,
        "train_steps_per_update": 10,
        "save_frequency": 250,
        "log_interval": 1,
        "warmup_episodes": 10,
    },
    "output": {
        "run_name": None,
        "base_dir": "results/decoy_policies",
    },
}


def _resolve_path(path, project_root):
    """Resolve relative path against project root; return path unchanged if absolute."""
    if path is None:
        return path
    path = str(path)
    if os.path.isabs(path):
        return path
    if project_root is None:
        return path
    return os.path.normpath(os.path.join(project_root, path))


def _resolve_config_paths(config, project_root):
    """Resolve relative paths in target and opponents against project_root (in place)."""
    target = config.get("target", {})
    if target.get("path"):
        target["path"] = _resolve_path(target["path"], project_root)
        logger.info(f"Resolved target path: {target['path']}")
    for opp in config.get("opponents", []):
        if opp.get("path"):
            opp["path"] = _resolve_path(opp["path"], project_root)
            logger.info(f"Resolved opponent path: {opp['path']}")


def _load_agent_by_type(agent_type, path, env, is_target=True):
    """Load an agent from checkpoint by type (TDMPC2, SAC, TD3). Path must be absolute or cwd-relative."""
    state_dim = env.observation_space.shape[0]
    action_dim = 4 if env.keep_mode else 3
    label = "target" if is_target else "opponent"

    if agent_type.upper() == "TDMPC2":
        import torch

        from rl_hockey.TD_MPC2.tdmpc2 import TDMPC2

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        latent_dim = checkpoint.get("latent_dim") or config.get("latent_dim", 512)
        hidden_dim = checkpoint.get("hidden_dim") or config.get("hidden_dim")
        num_q = checkpoint.get("num_q") or config.get("num_q", 5)
        horizon = checkpoint.get("horizon") or config.get("horizon", 5)
        gamma = checkpoint.get("gamma") or config.get("gamma", 0.99)
        logger.info(
            f"Creating {label} TDMPC2 with latent_dim={latent_dim}, hidden_dim={hidden_dim}"
        )
        agent = TDMPC2(
            obs_dim=state_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_q=num_q,
            horizon=horizon,
            gamma=gamma,
            opponent_simulation_enabled=False,
        )
        agent.load(path)
        logger.info(f"{agent_type} {label} loaded from {path}")
        return agent
    if agent_type.upper() == "SAC":
        from rl_hockey.sac.sac import SAC

        agent = SAC(state_dim=state_dim, action_dim=action_dim)
        agent.load(path)
        logger.info(f"{agent_type} {label} loaded from {path}")
        return agent
    if agent_type.upper() == "TD3":
        from rl_hockey.td3.td3 import TD3

        agent = TD3(state_dim=state_dim, action_dim=action_dim)
        agent.load(path)
        logger.info(f"{agent_type} {label} loaded from {path}")
        return agent
    raise ValueError(f"Unknown agent type for loading: {agent_type}")


def load_target_agent(target_config, env):
    """
    Load or create the target agent to mimic.

    Args:
        target_config: Dictionary with 'type' and optional 'path'
        env: Hockey environment instance

    Returns:
        Target agent or BasicOpponent instance
    """
    target_type = target_config["type"]
    target_path = target_config.get("path")

    if target_type == "basic_weak":
        logger.info("Using BasicOpponent (weak) as target")
        return h_env.BasicOpponent(weak=True)
    if target_type == "basic_strong":
        logger.info("Using BasicOpponent (strong) as target")
        return h_env.BasicOpponent(weak=False)
    if target_path:
        logger.info(f"Loading target agent from {target_path}")
        return _load_agent_by_type(target_type, target_path, env, is_target=True)
    raise ValueError(f"Invalid target configuration: {target_config}")


def load_opponent(opponent_config, env):
    """
    Load or create an opponent agent.

    Args:
        opponent_config: Dictionary with 'type' and optional 'path'
        env: Hockey environment instance

    Returns:
        Opponent agent or BasicOpponent instance
    """
    opponent_type = opponent_config["type"]
    opponent_path = opponent_config.get("path")

    if opponent_type == "basic_weak":
        return h_env.BasicOpponent(weak=True)
    if opponent_type == "basic_strong":
        return h_env.BasicOpponent(weak=False)
    if opponent_path:
        logger.info(f"Loading opponent from {opponent_path}")
        return _load_agent_by_type(opponent_type, opponent_path, env, is_target=False)
    raise ValueError(f"Invalid opponent configuration: {opponent_config}")


def build_opponent_cache(opponents_config, env):
    """
    Load each opponent once and return a list of (agent, weight) for sampling.

    Args:
        opponents_config: List of opponent configurations with weights
        env: Hockey environment instance

    Returns:
        List of (agent, weight) tuples
    """
    cache = []
    for opponent_config in opponents_config:
        agent = load_opponent(opponent_config, env)
        weight = opponent_config["weight"]
        cache.append((agent, weight))
    logger.info(f"Built opponent cache with {len(cache)} opponents")
    return cache


def sample_opponent_from_cache(cached_opponents):
    """
    Sample an opponent from a pre-built cache by weight.

    Args:
        cached_opponents: List of (agent, weight) from build_opponent_cache

    Returns:
        Sampled opponent agent
    """
    total_weight = sum(w for _, w in cached_opponents)
    rand_val = np.random.uniform(0, total_weight)
    cumulative_weight = 0
    for agent, weight in cached_opponents:
        cumulative_weight += weight
        if rand_val <= cumulative_weight:
            return agent
    return cached_opponents[-1][0]


def collect_episode_transitions(env, target_agent, opponent, max_steps=500):
    """
    Collect transitions from one episode where target agent plays.

    Args:
        env: Hockey environment
        target_agent: Agent to mimic
        opponent: Opponent to play against
        max_steps: Maximum steps per episode

    Returns:
        List of transitions (state, action, reward, next_state, done)
    """
    transitions = []

    obs, info = env.reset()
    done = False
    step = 0

    while not done and step < max_steps:
        obs_f32 = np.asarray(obs, dtype=np.float32)
        obs_opp_f32 = np.asarray(env.obs_agent_two(), dtype=np.float32)
        if isinstance(target_agent, h_env.BasicOpponent):
            action = target_agent.act(obs)
        else:
            action = target_agent.act(obs_f32, deterministic=False)

        if isinstance(opponent, h_env.BasicOpponent):
            opponent_action = opponent.act(env.obs_agent_two())
        else:
            opponent_action = opponent.act(obs_opp_f32, deterministic=False)

        next_obs, reward, done, truncated, info = env.step(
            np.hstack([action, opponent_action])
        )

        transitions.append((obs, action, reward, next_obs, done or truncated))

        obs = next_obs
        step += 1

    return transitions


def evaluate_decoy_policy(
    decoy_policy, target_agent, env, num_episodes=10, max_steps=500
):
    """
    Evaluate how well the decoy policy mimics the target agent.

    Args:
        decoy_policy: Trained decoy policy
        target_agent: Original agent to compare against
        env: Hockey environment
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode

    Returns:
        Dictionary with evaluation metrics
    """
    action_errors = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0

        episode_errors = []

        while not done and step < max_steps:
            obs_f32 = np.asarray(obs, dtype=np.float32)
            if isinstance(target_agent, h_env.BasicOpponent):
                target_action = target_agent.act(obs)
            else:
                target_action = target_agent.act(obs_f32, deterministic=True)

            decoy_action = decoy_policy.act(obs_f32, deterministic=True)

            action_error = np.mean((target_action - decoy_action) ** 2)
            episode_errors.append(action_error)

            opponent_action = np.zeros(4 if env.keep_mode else 3)
            next_obs, reward, done, truncated, info = env.step(
                np.hstack([target_action, opponent_action])
            )

            obs = next_obs
            step += 1

        action_errors.extend(episode_errors)

    return {
        "mean_action_error": np.mean(action_errors),
        "std_action_error": np.std(action_errors),
        "median_action_error": np.median(action_errors),
    }


def train_decoy_policy(
    target_config=None,
    opponents_config=None,
    decoy_config=None,
    training_config=None,
    output_config=None,
):
    """
    Train a decoy policy to mimic target agent behavior.
    Uses CONFIG at top of file; override by passing explicit config dicts.

    Args:
        target_config: Target agent configuration (default: CONFIG["target"])
        opponents_config: List of opponent configurations (default: CONFIG["opponents"])
        decoy_config: Decoy policy hyperparameters (default: CONFIG["decoy"])
        training_config: Training loop configuration (default: CONFIG["training"])
        output_config: Output directory configuration (default: CONFIG["output"])
    """
    if target_config is None:
        target_config = deepcopy(CONFIG["target"])
    if opponents_config is None:
        opponents_config = deepcopy(CONFIG["opponents"])
    if decoy_config is None:
        decoy_config = deepcopy(CONFIG["decoy"])
    if training_config is None:
        training_config = deepcopy(CONFIG["training"])
    if output_config is None:
        output_config = deepcopy(CONFIG["output"])

    config = {
        "target": target_config,
        "opponents": opponents_config,
        "decoy": decoy_config,
        "training": training_config,
        "output": output_config,
    }
    _resolve_config_paths(config, str(_PROJECT_ROOT))

    target_config = config["target"]
    opponents_config = config["opponents"]
    decoy_config = config["decoy"]
    training_config = config["training"]
    output_config = config["output"]

    if output_config.get("run_name") is None:
        output_config["run_name"] = (
            f"decoy_policy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    logger.info("=" * 80)
    logger.info("DECOY POLICY TRAINING")
    logger.info("=" * 80)
    logger.info(f"Target: {target_config['type']}")
    logger.info(f"Opponents: {[opp['type'] for opp in opponents_config]}")
    logger.info(f"Training Episodes: {training_config['num_episodes']}")
    logger.info("=" * 80)

    run_dir = Path(output_config["base_dir"]) / output_config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(
            {
                "target": target_config,
                "opponents": opponents_config,
                "decoy": decoy_config,
                "training": training_config,
                "output": output_config,
            },
            f,
            indent=2,
        )

    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)

    state_dim = env.observation_space.shape[0]
    action_dim = 4 if env.keep_mode else 3

    target_agent = load_target_agent(target_config, env)

    decoy_policy = DecoyPolicy(
        obs_dim=state_dim,
        action_dim=action_dim,
        hidden_layers=decoy_config["hidden_layers"],
        learning_rate=decoy_config["learning_rate"],
        buffer_max_size=decoy_config["buffer_max_size"],
    )

    logger.info(decoy_policy.log_architecture())

    cached_opponents = build_opponent_cache(opponents_config, env)

    training_metrics = {
        "episode": [],
        "loss": [],
        "mse_loss": [],
        "buffer_size": [],
        "mean_action_error": [],
    }

    logger.info("Starting training...")
    num_episodes = training_config["num_episodes"]
    log_interval = training_config.get("log_interval", max(1, num_episodes // 20))
    last_train_metrics = None

    for episode in range(num_episodes):
        opponent = sample_opponent_from_cache(cached_opponents)

        transitions = collect_episode_transitions(
            env, target_agent, opponent, max_steps=training_config["max_episode_steps"]
        )

        for transition in transitions:
            decoy_policy.store_transition(transition)

        train_metrics = None
        if episode >= training_config["warmup_episodes"]:
            train_metrics = decoy_policy.train(
                steps=training_config["train_steps_per_update"]
            )
            last_train_metrics = train_metrics
            training_metrics["episode"].append(episode)
            training_metrics["loss"].append(train_metrics["loss"])
            training_metrics["mse_loss"].append(train_metrics["mse_loss"])
            training_metrics["buffer_size"].append(decoy_policy.buffer.size)

        if (episode + 1) % log_interval == 0 or episode == 0:
            eval_metrics = evaluate_decoy_policy(
                decoy_policy,
                target_agent,
                env,
                num_episodes=5,
                max_steps=training_config["max_episode_steps"],
            )
            training_metrics["mean_action_error"].append(
                eval_metrics["mean_action_error"]
            )
            loss_str = (
                f"{last_train_metrics['loss']:.4f}" if last_train_metrics else "N/A"
            )
            mse_str = (
                f"{last_train_metrics['mse_loss']:.4f}" if last_train_metrics else "N/A"
            )
            logger.info(
                "Episode %d/%d Loss=%s MSE Loss=%s Action Error=%.4f Buffer Size=%d",
                episode + 1,
                num_episodes,
                loss_str,
                mse_str,
                eval_metrics["mean_action_error"],
                decoy_policy.buffer.size,
            )

        if (episode + 1) % training_config["save_frequency"] == 0:
            save_path = models_dir / f"decoy_policy_ep{episode + 1:06d}.pt"
            decoy_policy.save(str(save_path))
            logger.info(f"Saved checkpoint: {save_path}")

    final_save_path = models_dir / "decoy_policy_final.pt"
    decoy_policy.save(str(final_save_path))
    logger.info(f"\nTraining complete! Final model saved to: {final_save_path}")

    with open(run_dir / "training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=2)

    logger.info(f"\nAll results saved to: {run_dir}")

    return decoy_policy, training_metrics


if __name__ == "__main__":
    train_decoy_policy()
