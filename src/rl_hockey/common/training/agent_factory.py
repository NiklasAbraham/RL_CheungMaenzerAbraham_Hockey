from typing import Optional, Tuple

import hockey.hockey_env as h_env

from rl_hockey.common.agent import Agent
from rl_hockey.common.training.curriculum_manager import AgentConfig
from rl_hockey.common.utils import get_discrete_action_dim
from rl_hockey.DDDQN import DDDQN, DDQN_PER
from rl_hockey.sac.sac import SAC
from rl_hockey.TD_MPC2.tdmpc2 import TDMPC2
from rl_hockey.TD_MPC2_repo.tdmpc2_repo_wrapper import TDMPC2RepoWrapper


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
    elif agent_type == "TDMPC2" or agent_type == "TDMPC2_REPO":
        # TDMPC2 uses continuous actions, but also needs discretized action space for hockey
        action_dim = 4 if env.keep_mode else 3
        is_discrete = False
    else:
        # continuous action space with 3 or 4 dimensions depending on keep_mode
        action_dim = 3 if not env.keep_mode else 4
        is_discrete = False

    return state_dim, action_dim, is_discrete


def create_agent(
    agent_config: AgentConfig,
    state_dim: int,
    action_dim: int,
    common_hyperparams: dict,
    device: str = None,
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
    elif agent_config.type == "TDMPC2":
        # TD-MPC2 specific parameters
        latent_dim = agent_hyperparams.pop("latent_dim", 512)
        hidden_dim = agent_hyperparams.pop("hidden_dim", None)
        num_q = agent_hyperparams.pop("num_q", 5)
        horizon = agent_hyperparams.pop("horizon", 5)
        num_samples = agent_hyperparams.pop("num_samples", 512)
        num_iterations = agent_hyperparams.pop("num_iterations", 6)
        temperature = agent_hyperparams.pop("temperature", 0.5)
        gamma = agent_hyperparams.pop("gamma", 0.99)
        capacity = agent_hyperparams.pop("capacity", 1000000)
        simnorm_temperature = agent_hyperparams.pop("simnorm_temperature", 1.0)
        log_std_min = agent_hyperparams.pop("log_std_min", -10.0)
        log_std_max = agent_hyperparams.pop("log_std_max", 2.0)
        lambda_coef = agent_hyperparams.pop("lambda_coef", 0.95)
        vmin = agent_hyperparams.pop("vmin", -10.0)
        vmax = agent_hyperparams.pop("vmax", 10.0)
        n_step = agent_hyperparams.pop("n_step", 1)

        # Use provided device or default to CPU/CUDA
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

        return TDMPC2(
            obs_dim=state_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_q=num_q,
            lr=agent_hyperparams.get("learning_rate", 3e-4),
            gamma=gamma,
            horizon=horizon,
            num_samples=num_samples,
            num_iterations=num_iterations,
            temperature=temperature,
            capacity=capacity,
            batch_size=agent_hyperparams.get("batch_size", 256),
            device=device,
            simnorm_temperature=simnorm_temperature,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            lambda_coef=lambda_coef,
            vmin=vmin,
            vmax=vmax,
            n_step=n_step,
        )
    elif agent_config.type == "TDMPC2_REPO":
        # TD-MPC2 reference repo wrapper specific parameters
        latent_dim = agent_hyperparams.pop("latent_dim", 512)
        hidden_dim = agent_hyperparams.pop("hidden_dim", None)
        num_q = agent_hyperparams.pop("num_q", 5)
        horizon = agent_hyperparams.pop("horizon", 5)
        num_samples = agent_hyperparams.pop("num_samples", 512)
        num_iterations = agent_hyperparams.pop("num_iterations", 6)
        num_elites = agent_hyperparams.pop("num_elites", 64)
        num_pi_trajs = agent_hyperparams.pop("num_pi_trajs", 24)
        temperature = agent_hyperparams.pop("temperature", 0.5)
        gamma = agent_hyperparams.pop("gamma", 0.99)
        capacity = agent_hyperparams.pop("capacity", 1000000)
        log_std_min = agent_hyperparams.pop("log_std_min", -10.0)
        log_std_max = agent_hyperparams.pop("log_std_max", 2.0)
        vmin = agent_hyperparams.pop("vmin", -10.0)
        vmax = agent_hyperparams.pop("vmax", 10.0)
        tau = agent_hyperparams.pop("tau", 0.01)
        grad_clip_norm = agent_hyperparams.pop("grad_clip_norm", 20.0)
        consistency_coef = agent_hyperparams.pop("consistency_coef", 20.0)
        reward_coef = agent_hyperparams.pop("reward_coef", 0.1)
        value_coef = agent_hyperparams.pop("value_coef", 0.1)
        termination_coef = agent_hyperparams.pop("termination_coef", 1.0)
        rho = agent_hyperparams.pop("rho", 0.5)
        entropy_coef = agent_hyperparams.pop("entropy_coef", 1e-4)
        min_std = agent_hyperparams.pop("min_std", 0.05)
        max_std = agent_hyperparams.pop("max_std", 2.0)
        discount_denom = agent_hyperparams.pop("discount_denom", 5.0)
        discount_min = agent_hyperparams.pop("discount_min", 0.95)
        discount_max = agent_hyperparams.pop("discount_max", 0.995)
        episodic = agent_hyperparams.pop("episodic", False)
        mpc = agent_hyperparams.pop("mpc", True)
        compile_model = agent_hyperparams.pop("compile", True)
        episode_length = agent_hyperparams.pop("episode_length", 500)
        enc_lr_scale = agent_hyperparams.pop("enc_lr_scale", 0.3)
        seed = agent_hyperparams.pop("seed", 1)

        # Use provided device or default to CPU/CUDA
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

        return TDMPC2RepoWrapper(
            obs_dim=state_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_q=num_q,
            lr=agent_hyperparams.get("learning_rate", 3e-4),
            enc_lr_scale=enc_lr_scale,
            gamma=gamma,
            horizon=horizon,
            num_samples=num_samples,
            num_iterations=num_iterations,
            num_elites=num_elites,
            num_pi_trajs=num_pi_trajs,
            temperature=temperature,
            capacity=capacity,
            batch_size=agent_hyperparams.get("batch_size", 256),
            device=device,
            num_bins=101,
            vmin=vmin,
            vmax=vmax,
            tau=tau,
            grad_clip_norm=grad_clip_norm,
            consistency_coef=consistency_coef,
            reward_coef=reward_coef,
            value_coef=value_coef,
            termination_coef=termination_coef,
            rho=rho,
            entropy_coef=entropy_coef,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            min_std=min_std,
            max_std=max_std,
            discount_denom=discount_denom,
            discount_min=discount_min,
            discount_max=discount_max,
            episodic=episodic,
            mpc=mpc,
            compile=compile_model,
            episode_length=episode_length,
            seed=seed,
            **agent_hyperparams,
        )
    elif agent_config.type == "TD3":
        raise NotImplementedError("TD3 is not yet implemented")
    else:
        raise ValueError(f"Unknown agent type: {agent_config.type}")
