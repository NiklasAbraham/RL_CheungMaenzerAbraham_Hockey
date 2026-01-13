from typing import Optional, Tuple

import hockey.hockey_env as h_env

from rl_hockey.td3.td3_reference import TD3
from rl_hockey.common.agent import Agent
from rl_hockey.common.training.curriculum_manager import AgentConfig
from rl_hockey.common.utils import get_discrete_action_dim
from rl_hockey.DDDQN import DDDQN, DDQN_PER
from rl_hockey.sac.sac import SAC


def get_action_space_info(
    env: h_env.HockeyEnv, agent_type: str = "DDDQN", fineness: Optional[int] = None
) -> Tuple[int, int, bool]:
    state_dim = env.observation_space.shape[0]

    if agent_type == "DDDQN" or agent_type == "DDQN_PER":
        if fineness is not None:
            action_dim = get_discrete_action_dim(
                fineness=fineness, keep_mode=env.keep_mode
            )
        else:
            action_dim = 7 if not env.keep_mode else 8
        is_discrete = True
    else:
        # continuous action space with 3 or 4 dimensions depending on keep_mode
        action_dim = 3 if not env.keep_mode else 4
        is_discrete = False

    return state_dim, action_dim, is_discrete


def create_agent(
    agent_config: AgentConfig, state_dim: int, action_dim: int, common_hyperparams: dict
) -> Agent:
    agent_hyperparams = agent_config.hyperparameters.copy()
    agent_hyperparams.update(
        {
            "learning_rate": common_hyperparams.get("learning_rate", 1e-4),
            "batch_size": common_hyperparams.get("batch_size", 256),
        }
    )

    if agent_config.type == "DDDQN":
        hidden_dim = agent_hyperparams.pop("hidden_dim", [256, 256])
        return DDDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            **agent_hyperparams,
        )
    elif agent_config.type == "DDQN_PER":
        hidden_dim = agent_hyperparams.pop("hidden_dim", [256, 256])
        use_per = agent_hyperparams.pop("use_per", True)
        return DDQN_PER(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            use_per=use_per,
            **agent_hyperparams,
        )
    elif agent_config.type == "SAC":
        return SAC(state_dim=state_dim, action_dim=action_dim, **agent_hyperparams)

    elif agent_config.type == "TD3":
        return TD3(state_dim=state_dim, action_dim=action_dim, **agent_hyperparams)
    elif agent_config.type == "TDMPC2":
        raise NotImplementedError("TDMPC2 is not yet implemented")
    else:
        raise ValueError(f"Unknown agent type: {agent_config.type}")
