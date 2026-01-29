"""
Opponent Manager for creating and managing opponents during training.
"""
import numpy as np
import copy
from pathlib import Path
from typing import Optional, Dict, Any, Union

import hockey.hockey_env as h_env
from rl_hockey.common.agent import Agent
from rl_hockey.common.training.curriculum_manager import OpponentConfig, AgentConfig
from rl_hockey.common.training.agent_factory import create_agent


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
            deterministic=config.deterministic
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
            opponent_config, agent, checkpoint_dir, agent_config,
            state_dim, action_dim, is_discrete
        )
    
    # Normalize weights
    weights = [opp.get('weight', 1.0) for opp in opponent_config.opponents]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    # Sample based on weights
    sampled_idx = np.random.choice(len(opponent_config.opponents), p=normalized_weights)
    sampled_opponent_dict = opponent_config.opponents[sampled_idx]
    
    # Create OpponentConfig for sampled opponent
    sampled_config = OpponentConfig(
        type=sampled_opponent_dict['type'],
        weight=sampled_opponent_dict.get('weight', 1.0),
        checkpoint=sampled_opponent_dict.get('checkpoint'),
        deterministic=sampled_opponent_dict.get('deterministic', True),
        opponents=None
    )
    
    return create_opponent(
        sampled_config, agent, checkpoint_dir, agent_config,
        state_dim, action_dim, is_discrete, rating
    )


def get_opponent_action(
    opponent: Union[Agent, h_env.BasicOpponent, None],
    obs: np.ndarray,
    deterministic: bool = True
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
    deterministic: bool = True
) -> Agent:
    if checkpoint is None:
        opponent_agent = copy.deepcopy(agent)

        if hasattr(opponent_agent, 'q_network'):
            opponent_agent.q_network.eval()
        if hasattr(opponent_agent, 'q_network_target'):
            opponent_agent.q_network_target.eval()
        if hasattr(opponent_agent, 'actor') and hasattr(opponent_agent.actor, 'eval'):
            opponent_agent.actor.eval()
        if hasattr(opponent_agent, 'critic1') and hasattr(opponent_agent.critic1, 'eval'):
            opponent_agent.critic1.eval()
        return opponent_agent
    
    elif checkpoint == "latest":
        checkpoint_path = _find_latest_checkpoint(checkpoint_dir)
        return load_agent_checkpoint(
            checkpoint_path, agent_config, state_dim, action_dim, is_discrete
        )
    
    else:
        return load_agent_checkpoint(
            checkpoint, agent_config, state_dim, action_dim, is_discrete
        )


def load_agent_checkpoint(
    checkpoint_path: str,
    agent_config: AgentConfig,
    state_dim: int,
    action_dim: int,
    is_discrete: bool
) -> Agent:
    from rl_hockey.common.training.agent_factory import create_agent
    
    agent = create_agent(
        agent_config, state_dim, action_dim, {}
    )
    
    agent.load(checkpoint_path)
    
    if hasattr(agent, 'q_network'):
        agent.q_network.eval()
    if hasattr(agent, 'q_network_target'):
        agent.q_network_target.eval()
    if hasattr(agent, 'actor') and hasattr(agent.actor, 'eval'):
        agent.actor.eval()
    if hasattr(agent, 'critic1') and hasattr(agent.critic1, 'eval'):
        agent.critic1.eval()
    
    return agent


def _find_latest_checkpoint(checkpoint_dir: str) -> str:
    checkpoint_path = Path(checkpoint_dir)
    checkpoints = list(checkpoint_path.glob("*.pt"))
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(checkpoints[0])
