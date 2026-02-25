"""
Opponent Manager for creating and managing opponents during training.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import hockey.hockey_env as h_env
import numpy as np
import torch

from rl_hockey.common.agent import Agent
from rl_hockey.common.training.agent_factory import create_agent
from rl_hockey.common.training.curriculum_manager import AgentConfig, OpponentConfig


def create_opponent(
    config: OpponentConfig,
    agent: Optional[Agent] = None,
    checkpoint_dir: Optional[str] = None,
    agent_config: Optional[AgentConfig] = None,
    state_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    is_discrete: Optional[bool] = None,
    rating: Optional[float] = None,
) -> Union[Agent, h_env.BasicOpponent, None]:
    """Create an opponent based on configuration."""
    if config.type == "none":
        return None

    elif config.type == "basic_weak":
        return h_env.BasicOpponent(weak=True)

    elif config.type == "basic_strong":
        return h_env.BasicOpponent(weak=False)

    elif config.type == "self_play":
        return create_self_play_opponent(
            agent=agent,
            checkpoint=config.checkpoint,
            checkpoint_dir=checkpoint_dir,
            agent_config=agent_config,
            state_dim=state_dim,
            action_dim=action_dim,
            is_discrete=is_discrete,
            deterministic=config.deterministic,
            opponent_agent_type=config.agent_type,
        )

    elif config.type == "weighted_mixture":
        # Return the config itself, sampling happens per episode
        return config

    else:
        raise ValueError(f"Unknown opponent type: {config.type}")


def sample_opponent(
    opponent_config: OpponentConfig,
    agent: Optional[Agent] = None,
    checkpoint_dir: Optional[str] = None,
    agent_config: Optional[AgentConfig] = None,
    state_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    is_discrete: Optional[bool] = None,
    rating: Optional[float] = None,
) -> Union[Agent, h_env.BasicOpponent, None]:
    """Sample an opponent from a weighted mixture configuration."""
    if opponent_config.type != "weighted_mixture":
        return create_opponent(
            opponent_config,
            agent,
            checkpoint_dir,
            agent_config,
            state_dim,
            action_dim,
            is_discrete,
        )

    # Normalize weights
    weights = [opp.get("weight", 1.0) for opp in opponent_config.opponents]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Sample based on weights
    sampled_idx = np.random.choice(len(opponent_config.opponents), p=normalized_weights)
    sampled_opponent_dict = opponent_config.opponents[sampled_idx]

    # Create OpponentConfig for sampled opponent
    sampled_config = OpponentConfig(
        type=sampled_opponent_dict["type"],
        weight=sampled_opponent_dict.get("weight", 1.0),
        checkpoint=sampled_opponent_dict.get("checkpoint"),
        deterministic=sampled_opponent_dict.get("deterministic", True),
        opponents=None,
        agent_type=sampled_opponent_dict.get("agent_type"),
    )

    return create_opponent(
        sampled_config,
        agent,
        checkpoint_dir,
        agent_config,
        state_dim,
        action_dim,
        is_discrete,
        rating,
    )


def get_opponent_action(
    opponent: Union[Agent, h_env.BasicOpponent, None],
    obs: np.ndarray,
    deterministic: bool = True,
) -> np.ndarray:
    """Get action from opponent."""
    if opponent is None:
        return np.zeros(6)

    elif isinstance(opponent, h_env.BasicOpponent):
        return opponent.act(obs)

    elif isinstance(opponent, Agent):
        action = opponent.act(obs.astype(np.float32), deterministic=deterministic)
        return action


def create_self_play_opponent(
    agent: Optional[Agent],
    checkpoint: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    agent_config: Optional[AgentConfig] = None,
    state_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    is_discrete: Optional[bool] = None,
    deterministic: bool = True,
    opponent_agent_type: Optional[str] = None,
) -> Agent:
    """Create a self-play opponent from checkpoint or by copying the current agent.

    Args:
        agent: Current training agent (for copy-based self-play)
        checkpoint: Path to checkpoint file or "latest"
        checkpoint_dir: Directory containing checkpoints (for "latest")
        agent_config: Configuration for creating the agent
        state_dim: Observation space dimension
        action_dim: Action space dimension
        is_discrete: Whether the action space is discrete
        deterministic: Whether the opponent should act deterministically
        opponent_agent_type: Specific agent type for loading checkpoint (SAC, TD3, TDMPC2, etc.)
                            If None, uses agent_config.type
    """
    if checkpoint is None:
        # Copy current agent for self-play
        opponent_agent = copy.deepcopy(agent)

        if hasattr(opponent_agent, "q_network"):
            opponent_agent.q_network.eval()
        if hasattr(opponent_agent, "q_network_target"):
            opponent_agent.q_network_target.eval()
        if hasattr(opponent_agent, "actor") and hasattr(opponent_agent.actor, "eval"):
            opponent_agent.actor.eval()
        if hasattr(opponent_agent, "critic1") and hasattr(
            opponent_agent.critic1, "eval"
        ):
            opponent_agent.critic1.eval()
        return opponent_agent

    elif checkpoint == "latest":
        checkpoint_path = _find_latest_checkpoint(checkpoint_dir)
        return load_agent_checkpoint(
            checkpoint_path,
            agent_config,
            state_dim,
            action_dim,
            is_discrete,
            opponent_agent_type=opponent_agent_type,
        )

    else:
        return load_agent_checkpoint(
            checkpoint,
            agent_config,
            state_dim,
            action_dim,
            is_discrete,
            opponent_agent_type=opponent_agent_type,
        )


def load_agent_checkpoint(
    checkpoint_path: str,
    agent_config: AgentConfig,
    state_dim: int,
    action_dim: int,
    is_discrete: bool,
    opponent_agent_type: Optional[str] = None,
) -> Agent:
    """Load an agent from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        agent_config: Configuration of the training agent
        state_dim: Observation space dimension
        action_dim: Action space dimension
        is_discrete: Whether the action space is discrete
        opponent_agent_type: Specific agent type to load (SAC, TD3, TDMPC2, DECOYPOLICY)
                            If provided, overrides agent_config.type
    """
    # Create a new AgentConfig with the opponent's agent type if specified
    if opponent_agent_type is not None:
        # Create opponent-specific config
        hyperparameters = _get_default_hyperparameters(opponent_agent_type)
        if opponent_agent_type.upper() == "TDMPC2":
            checkpoint_hp = _get_tdmpc2_hyperparameters_from_checkpoint(checkpoint_path)
            hyperparameters.update(checkpoint_hp)
        opponent_config = AgentConfig(
            type=opponent_agent_type,
            hyperparameters=hyperparameters,
            checkpoint_path=None,
        )
    else:
        # Use the training agent's config
        opponent_config = agent_config

    # For TDMPC2, build with checkpoint's actual architecture so state_dict loads without shape mismatch.
    # Prefer dimensions inferred from state_dict over stored obs_dim/action_dim, since stored values
    # can disagree with the saved model (e.g. stored action_dim=4 but dynamics was built with 8).
    build_state_dim = state_dim
    build_action_dim = action_dim
    common_hyperparams = {}
    if opponent_config.type == "TDMPC2":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        latent = ckpt.get("latent_dim") or (ckpt.get("config") or {}).get(
            "latent_dim", 512
        )
        inf_obs, inf_act, is_opponent_dynamics = (
            _infer_tdmpc2_obs_action_dims_from_checkpoint(ckpt, latent)
        )
        if inf_obs is not None:
            build_state_dim = int(inf_obs)
        elif ckpt.get("obs_dim") is not None:
            build_state_dim = int(ckpt["obs_dim"])
        if inf_act is not None:
            build_action_dim = int(inf_act)
        elif ckpt.get("action_dim") is not None:
            build_action_dim = int(ckpt["action_dim"])
        if ckpt.get("opponent_simulation_enabled", False) or is_opponent_dynamics:
            common_hyperparams["opponent_simulation"] = {
                "enabled": True,
                "opponent_agents": [],
            }
        del ckpt

    # Create agent with minimal hyperparameters (will be loaded from checkpoint)
    agent = create_agent(
        opponent_config, build_state_dim, build_action_dim, common_hyperparams
    )

    # Load checkpoint
    agent.load(checkpoint_path)

    logger = logging.getLogger(__name__)
    logger.info(
        "Loaded opponent: %s from %s",
        opponent_config.type,
        checkpoint_path,
    )

    # Set to evaluation mode
    if hasattr(agent, "q_network"):
        agent.q_network.eval()
    if hasattr(agent, "q_network_target"):
        agent.q_network_target.eval()
    if hasattr(agent, "actor") and hasattr(agent.actor, "eval"):
        agent.actor.eval()
    if hasattr(agent, "critic1") and hasattr(agent.critic1, "eval"):
        agent.critic1.eval()

    return agent


def _infer_tdmpc2_obs_action_dims_from_checkpoint(
    checkpoint: Dict[str, Any], latent_dim: int
) -> Tuple[Optional[int], Optional[int], bool]:
    """Infer obs_dim, action_dim and whether dynamics is opponent-type from checkpoint state_dict.

    Encoder first layer: (hidden[0], obs_dim). Dynamics first layer:
    - DynamicsSimple: (hidden[0], latent_dim + action_dim)
    - DynamicsOpponent: (hidden[0], latent_dim + action_dim + action_opponent_dim)
    Handles torch.compile keys like _orig_mod.net.0.weight and any other prefix.
    Returns (obs_dim, action_dim, is_opponent_dynamics).
    """
    obs_dim, action_dim = None, None
    enc = checkpoint.get("encoder") or {}
    dyn = checkpoint.get("dynamics") or {}

    def first_layer_input_shape(state_dict: Dict[str, Any]) -> Optional[int]:
        for k, v in state_dict.items():
            if k.endswith("net.0.weight") and hasattr(v, "shape") and len(v.shape) == 2:
                return int(v.shape[1])
        return None

    enc_in = first_layer_input_shape(enc)
    if enc_in is not None:
        obs_dim = enc_in
    dyn_in = first_layer_input_shape(dyn)
    is_opponent_dynamics = False
    if dyn_in is not None and dyn_in >= latent_dim:
        delta = dyn_in - latent_dim
        if delta in (6, 16):
            action_dim = delta // 2
            is_opponent_dynamics = True
        else:
            action_dim = delta
    return obs_dim, action_dim, is_opponent_dynamics


def _get_tdmpc2_hyperparameters_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Read TDMPC2 architecture and config from a checkpoint so the agent can be built to match.

    Includes obs_dim and action_dim so the loaded model has the same input shapes as the
    checkpoint (avoids dynamics/encoder shape mismatch when env changed between runs).
    If the checkpoint does not store obs_dim/action_dim (older format), infers them from state_dict.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    latent_dim = checkpoint.get("latent_dim") or config.get("latent_dim", 512)
    out = {
        "latent_dim": latent_dim,
        "hidden_dim": checkpoint.get("hidden_dim") or config.get("hidden_dim"),
        "num_q": checkpoint.get("num_q") or config.get("num_q", 5),
        "horizon": checkpoint.get("horizon") or config.get("horizon", 5),
        "gamma": checkpoint.get("gamma") or config.get("gamma", 0.99),
        "num_samples": config.get("num_samples", 512),
        "num_iterations": config.get("num_iterations", 6),
        "temperature": config.get("temperature", 0.5),
    }
    if checkpoint.get("obs_dim") is not None:
        out["obs_dim"] = checkpoint["obs_dim"]
    if checkpoint.get("action_dim") is not None:
        out["action_dim"] = checkpoint["action_dim"]
    if "obs_dim" not in out or "action_dim" not in out:
        inferred_obs, inferred_action, _ = (
            _infer_tdmpc2_obs_action_dims_from_checkpoint(checkpoint, latent_dim)
        )
        if "obs_dim" not in out and inferred_obs is not None:
            out["obs_dim"] = inferred_obs
        if "action_dim" not in out and inferred_action is not None:
            out["action_dim"] = inferred_action
    return out


def _get_default_hyperparameters(agent_type: str) -> Dict[str, Any]:
    """Get default hyperparameters for different agent types."""
    defaults = {
        "SAC": {
            "discount": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "learn_alpha": False,
        },
        "TD3": {
            "learning_rate": 3e-4,
            "max_action": 1.0,
            "discount": 0.99,
            "tau": 0.005,
            "expl_noise": 0.1,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_freq": 2,
        },
        "TDMPC2": {
            "latent_dim": 512,
            "num_q": 5,
            "gamma": 0.99,
            "horizon": 5,
            "num_samples": 512,
            "num_iterations": 6,
            "temperature": 0.5,
        },
        "DECOYPOLICY": {
            "hidden_layers": [256, 256],
            "buffer_max_size": 100_000,
        },
    }
    return defaults.get(agent_type, {})


def _find_latest_checkpoint(checkpoint_dir: str) -> str:
    checkpoint_path = Path(checkpoint_dir)
    checkpoints = list(checkpoint_path.glob("*.pt"))
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(checkpoints[0])
