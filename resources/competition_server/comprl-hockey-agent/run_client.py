from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import hockey.hockey_env as h_env
import numpy as np


from comprl.client import Agent, launch_client


def _load_archive_agent(archive_id: str, repo_root: Path):
    """Load an agent from the project archive by id (e.g. '0010')."""
    sys.path.insert(0, str(repo_root / "src"))
    from rl_hockey.common.archive import Archive
    from rl_hockey.common.training.curriculum_manager import load_curriculum
    from rl_hockey.common.training.agent_factory import create_agent

    archive_dir = repo_root / "archive"
    archive = Archive(base_dir=str(archive_dir))
    metadata = archive.get_agent_metadata(archive_id)
    if metadata is None:
        candidates = [
            a for a in archive.get_agents()
            if a.agent_id == archive_id or a.agent_id.startswith(archive_id + "_")
        ]
        if not candidates:
            raise ValueError(
                f"No archive agent found for id '{archive_id}'. "
                "Check archive/registry.json for valid agent ids."
            )
        if len(candidates) > 1:
            candidates.sort(key=lambda a: a.archived_at, reverse=True)
        metadata = candidates[0]

    config_path = repo_root / metadata.config_path if metadata.config_path else None
    checkpoint_path = repo_root / metadata.checkpoint_path if metadata.checkpoint_path else None
    if not config_path or not config_path.exists():
        raise FileNotFoundError(f"Archive agent config not found: {config_path}")
    if not checkpoint_path or not checkpoint_path.exists():
        raise FileNotFoundError(f"Archive agent checkpoint not found: {checkpoint_path}")

    env = h_env.HockeyEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] // 2
    env.close()

    config = load_curriculum(str(config_path))
    agent = create_agent(
        agent_config=config.agent,
        state_dim=state_dim,
        action_dim=action_dim,
        common_hyperparams=config.hyperparameters,
        checkpoint_path=str(checkpoint_path),
        deterministic=True,
    )
    return agent


class ArchiveAgentWrapper(Agent):
    """Wraps an rl_hockey archive agent for the comprl client."""

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


class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, weak: bool) -> None:
        super().__init__()

        self.hockey_agent = h_env.BasicOpponent(weak=weak)

    def get_step(self, observation: list[float]) -> list[float]:
        # NOTE: If your agent is using discrete actions (0-7), you can use
        # HockeyEnv.discrete_to_continous_action to convert the action:
        #
        # from hockey.hockey_env import HockeyEnv
        # env = HockeyEnv()
        # continuous_action = env.discrete_to_continous_action(discrete_action)

        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["weak", "strong", "random", "archive"],
        default="weak",
        help="Which agent to use. Use 'archive' to run an agent from the project archive.",
    )
    parser.add_argument(
        "--archive-id",
        type=str,
        default=None,
        help="Archive agent id (e.g. 0010). Required when --agent=archive. "
        "Matches agent_id or prefix (e.g. 0010 matches 0010_TDMPC2_...).",
    )
    args = parser.parse_args(agent_args)

    agent: Agent
    if args.agent == "weak":
        agent = HockeyAgent(weak=True)
    elif args.agent == "strong":
        agent = HockeyAgent(weak=False)
    elif args.agent == "random":
        agent = RandomAgent()
    elif args.agent == "archive":
        if not args.archive_id:
            raise ValueError("--archive-id is required when --agent=archive (e.g. --archive-id=0010)")
        repo_root = Path(__file__).resolve().parents[3]
        inner = _load_archive_agent(args.archive_id, repo_root)
        agent = ArchiveAgentWrapper(inner)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
