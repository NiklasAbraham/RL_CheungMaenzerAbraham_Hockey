import hockey.hockey_env as h_env
from typing import Tuple

from rl_hockey.DDDQN import DDDQN
from rl_hockey.sac.sac import SAC
from rl_hockey.td3.td3 import TD3
from rl_hockey.common.agent import Agent
from rl_hockey.common.training.curriculum_manager import AgentConfig


def get_action_space_info(
    env: h_env.HockeyEnv, agent_type: str = "DDDQN"
) -> Tuple[int, int, bool]:
    """Get action space information from environment."""
    state_dim = env.observation_space.shape[0]

    if agent_type == "DDDQN":
        # Discrete action space: 7 or 8 actions depending on keep_mode
        action_dim = 7 if not env.keep_mode else 8
        is_discrete = True
    elif agent_type == "SAC":
        # Continuous action space: 6 dimensions (x, y forces, torque)
        action_dim = 6
        is_discrete = False
    elif agent_type == "TD3":
        action_dim = 7 if not env.keep_mode else 8
        is_discrete = False
    else:
        raise NotImplementedError(
            f"{agent_type} is not implemented. Choose from DDDQN, SAC, TD3. Danke!"
        )

    return state_dim, action_dim, is_discrete


def create_agent(
    agent_config: AgentConfig,
    state_dim: int,
    action_dim: int,
    is_discrete: bool,
    common_hyperparams: dict,
) -> Agent:
    """Create an agent based on algorithm type and configuration."""
    # Merge common hyperparameters with algorithm-specific ones
    agent_hyperparams = agent_config.hyperparameters.copy()
    agent_hyperparams.update(
        {
            "learning_rate": common_hyperparams.get("learning_rate", 1e-4),
            "batch_size": common_hyperparams.get("batch_size", 256),
        }
    )

    if agent_config.type == "DDDQN":
        return DDDQN(state_dim=state_dim, action_dim=action_dim, **agent_hyperparams)
    elif agent_config.type == "SAC":
        return SAC(state_dim=state_dim, action_dim=action_dim, **agent_hyperparams)

    elif agent_config.type == "TD3":
        return TD3(state_dim=state_dim, action_dim=action_dim, **agent_hyperparams)
    elif agent_config.type == "TDMPC2":
        raise NotImplementedError("TDMPC2 is not yet implemented")
    else:
        raise ValueError(f"Unknown agent type: {agent_config.type}")
