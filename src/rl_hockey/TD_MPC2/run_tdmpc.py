import os

import hockey.hockey_env as h_env
import numpy as np
import torch
from tqdm import tqdm

from rl_hockey.common.evaluation.agent_evaluator import find_config_from_model_path
from rl_hockey.common.training.agent_factory import create_agent, get_action_space_info
from rl_hockey.common.training.curriculum_manager import AgentConfig, load_curriculum
from rl_hockey.TD_MPC2 import TDMPC2


def main(
    checkpoint_path: str,
    config_path: str = None,
    N: int = 100,
    weak_opponent: bool = False,
    render: bool = False,
):
    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config_path is None:
        config_path = find_config_from_model_path(checkpoint_path)

    if config_path is not None and os.path.exists(config_path):
        curriculum = load_curriculum(config_path)
        agent_config = AgentConfig(
            type=curriculum.agent.type,
            hyperparameters=curriculum.agent.hyperparameters,
        )
        state_dim, action_dim, _ = get_action_space_info(env, agent_config.type)
        agent = create_agent(
            agent_config,
            state_dim,
            action_dim,
            {},
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device,
            inference_mode=True,
        )
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        agent = TDMPC2(
            obs_dim=ckpt["obs_dim"],
            action_dim=ckpt["action_dim"],
            latent_dim=ckpt["latent_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_q=ckpt["num_q"],
            num_bins=ckpt.get("num_bins", 101),
            vmin=ckpt.get("vmin", -10.0),
            vmax=ckpt.get("vmax", 10.0),
            opponent_simulation_enabled=ckpt.get("opponent_simulation_enabled", False),
            opponent_cloning_frequency=ckpt.get("opponent_cloning_frequency", 5000),
            opponent_cloning_steps=ckpt.get("opponent_cloning_steps", 20),
            opponent_cloning_samples=ckpt.get("opponent_cloning_samples", 512),
            opponent_agents=ckpt.get("opponent_agents", []),
            device=device,
            inference_mode=True,
        )
        agent.load(checkpoint_path, inference_mode=True)

    if hasattr(agent, "encoder"):
        agent.encoder.eval()
    if hasattr(agent, "dynamics"):
        agent.dynamics.eval()
    if hasattr(agent, "reward"):
        agent.reward.eval()
    if hasattr(agent, "termination"):
        agent.termination.eval()
    if hasattr(agent, "q_ensemble"):
        agent.q_ensemble.eval()
    if hasattr(agent, "policy"):
        agent.policy.eval()
    if hasattr(agent, "target_q_ensemble"):
        agent.target_q_ensemble.eval()

    opponent = h_env.BasicOpponent(weak=weak_opponent)

    win_count = 0
    total_reward = 0

    for _ in tqdm(range(N)):
        state, _ = env.reset()
        for t in range(250):
            if render:
                env.render(mode="human")

            action1 = agent.act(state.astype(np.float32), deterministic=True)
            action2 = opponent.act(env.obs_agent_two())
            next_state, reward, done, trunc, info = env.step(
                np.hstack([action1, action2])
            )
            state = next_state
            total_reward += reward
            if done or trunc:
                break

        if info["winner"] == 1:
            win_count += 1

    print(f"Average Reward over {N} episodes: {total_reward / N}")
    print(f"Win Rate over {N} episodes: {win_count / N}")


if __name__ == "__main__":
    main(
        checkpoint_path="models/tdmpc2/models/TDMPC2_run_lr3e04_bs512_hencoder_dynamics_reward_termination_q_function_policy_add21d6e_20260201_095522_ep025646.pt",
        N=100,
        weak_opponent=False,
        render=False,
    )
