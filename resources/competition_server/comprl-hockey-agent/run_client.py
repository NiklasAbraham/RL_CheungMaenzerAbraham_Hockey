from __future__ import annotations

import argparse
import re
import sys
import uuid
from pathlib import Path

import hockey.hockey_env as h_env
import numpy as np


from comprl.client import Agent, launch_client


def _run_folder_and_name_from_model_path(path: Path) -> tuple[Path | None, str | None]:
    """If path is in a run folder (e.g. .../models/Name_ep012345.pt), return (run_folder, run_name). Else (None, None)."""
    if path.parent.name != "models":
        return None, None
    run_folder = path.parent.parent
    stem = path.stem
    match = re.match(r"^(.+)_ep\d+$", stem)
    run_name = match.group(1) if match else stem
    return run_folder, run_name


def _load_agent_from_config(
    config_path: Path, checkpoint_path: Path, repo_root: Path
):
    """Load agent using curriculum config + checkpoint (e.g. from run folder configs/). Opponent paths are resolved relative to repo_root so they work when config lives inside a run folder."""
    sys.path.insert(0, str(repo_root / "src"))
    from rl_hockey.common.training.curriculum_manager import load_curriculum
    from rl_hockey.common.training.agent_factory import create_agent

    env = h_env.HockeyEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] // 2
    env.close()

    config = load_curriculum(str(config_path))
    opponent_sim = config.agent.hyperparameters.get("opponent_simulation") or {}
    for opp in opponent_sim.get("opponent_agents") or []:
        if "path" in opp and not Path(opp["path"]).is_absolute():
            opp["path"] = str((repo_root / opp["path"]).resolve())

    agent = create_agent(
        agent_config=config.agent,
        state_dim=state_dim,
        action_dim=action_dim,
        common_hyperparams=config.hyperparameters,
        checkpoint_path=str(checkpoint_path),
        deterministic=True,
        config_path=None,
    )
    if hasattr(agent, "eval"):
        agent.eval()
    elif hasattr(agent, "encoder"):
        agent.encoder.eval()
        if hasattr(agent, "dynamics"):
            agent.dynamics.eval()
        if hasattr(agent, "policy"):
            agent.policy.eval()
    return agent


def _load_agent_from_checkpoint_only(path: Path, repo_root: Path, state_dim: int, action_dim: int):
    """Load agent by inferring architecture from checkpoint (no config file)."""
    sys.path.insert(0, str(repo_root / "src"))
    import torch

    from rl_hockey.TD_MPC2.tdmpc2 import TDMPC2
    from rl_hockey.sac.sac import SAC
    from rl_hockey.td3.td3 import TD3

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if "encoder" in checkpoint and "dynamics" in checkpoint:
        config = checkpoint.get("config", {})
        latent_dim = checkpoint.get("latent_dim") or config.get("latent_dim", 512)
        hidden_dim = checkpoint.get("hidden_dim") or config.get("hidden_dim")
        num_q = checkpoint.get("num_q") or config.get("num_q", 5)
        horizon = checkpoint.get("horizon") or config.get("horizon", 5)
        gamma = checkpoint.get("gamma") or config.get("gamma", 0.99)
        opponent_simulation_enabled = checkpoint.get(
            "opponent_simulation_enabled", False
        )
        agent = TDMPC2(
            obs_dim=state_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_q=num_q,
            horizon=horizon,
            gamma=gamma,
            device="cpu",
            opponent_simulation_enabled=opponent_simulation_enabled,
        )
        agent.load(str(path))
    elif "actor" in checkpoint and "critic" in checkpoint:
        if "log_alpha" in checkpoint or "alpha" in checkpoint:
            agent = SAC(state_dim=state_dim, action_dim=action_dim)
        else:
            agent = TD3(state_dim=state_dim, action_dim=action_dim)
        agent.load(str(path))
    else:
        raise ValueError(
            f"Unknown checkpoint format at {path}. "
            "Expected TDMPC2 (encoder, dynamics) or SAC/TD3 (actor, critic)."
        )
    if hasattr(agent, "eval"):
        agent.eval()
    elif hasattr(agent, "encoder"):
        agent.encoder.eval()
        if hasattr(agent, "dynamics"):
            agent.dynamics.eval()
        if hasattr(agent, "policy"):
            agent.policy.eval()
    return agent


def _load_agent_from_path(model_path: str, repo_root: Path):
    """Load an agent from a checkpoint path. Uses config in the run folder if present (e.g. .../configs/<run_name>.json or .../config.json), otherwise infers from checkpoint."""
    path = Path(model_path)
    if not path.is_absolute():
        path = (repo_root / model_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {path}")

    env = h_env.HockeyEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] // 2
    env.close()

    run_folder, run_name = _run_folder_and_name_from_model_path(path)
    config_path = None
    if run_folder is not None and run_name is not None:
        config_in_configs = run_folder / "configs" / f"{run_name}.json"
        config_at_root = run_folder / "config.json"
        if config_in_configs.exists():
            config_path = config_in_configs
        elif config_at_root.exists():
            config_path = config_at_root

    if config_path is not None:
        return _load_agent_from_config(config_path, path, repo_root)
    return _load_agent_from_checkpoint_only(path, repo_root, state_dim, action_dim)


class ModelAgentWrapper(Agent):
    """Wraps an rl_hockey model (loaded from checkpoint path) for the comprl client."""

    def __init__(self, inner_agent, t0: bool = True) -> None:
        super().__init__()
        self.inner = inner_agent
        self.t0 = t0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.games = 0

    def get_step(self, observation: list[float]) -> list[float]:
        obs = np.array(observation, dtype=np.float32)
        action = self.inner.act(obs, deterministic=True, t0=self.t0)
        self.t0 = False
        if hasattr(action, "tolist"):
            return action.tolist()
        return list(action)

    def on_start_game(self, game_id) -> None:
        self.t0 = True
        try:
            uid = uuid.UUID(int=int.from_bytes(game_id))
            print(f"[game] started id={uid}", flush=True)
        except Exception:
            print("[game] started", flush=True)

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        self.t0 = True
        self.games += 1
        my_score = float(stats[0]) if stats else 0
        opp_score = float(stats[1]) if len(stats) > 1 else 0
        if result:
            self.wins += 1
            outcome = "WON"
        elif my_score == opp_score:
            self.draws += 1
            outcome = "DRAW"
        else:
            self.losses += 1
            outcome = "LOST"
        print(
            f"[game] {outcome} score {my_score:.0f}-{opp_score:.0f} "
            f"(total W/D/L: {self.wins}/{self.draws}/{self.losses} in {self.games} games)",
            flush=True,
        )


# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to checkpoint file (.pt), e.g. results/self_play/.../model.pt or archive/agents/.../checkpoint.pt",
    )
    args = parser.parse_args(agent_args)

    repo_root = Path(__file__).resolve().parents[3]
    inner = _load_agent_from_path(args.model_path, repo_root)
    return ModelAgentWrapper(inner)


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
