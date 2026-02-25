from __future__ import annotations

import argparse
import uuid

import hockey.hockey_env as h_env
import numpy as np


from comprl.client import Agent, launch_client


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

class CustomAgent(Agent):
    def __init__(self, redq_agent):
        super().__init__()
        self.agent = redq_agent

    def get_step(self, obv: list[float]) -> list[float]:
        """
        Requests the agent's action based on the current observation.

        Args:
            obv (list[float]): The current observation.

        Returns:
            list[float]: The agent's action.
        """
        return self.agent.act(obv, deterministic=True).tolist()
    
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
    from rl_hockey.REDQ.redq_td3 import REDQTD3
    import json

    config = json.load(open("results/redq_td3/2026-02-22_19-32-16/config.json", "r"))
    custom_agent = REDQTD3(state_dim=18, action_dim=4, **config)
    custom_agent.load("results/redq_td3/2026-02-22_19-32-16/models/REDQTD3_run_lr3e04_bs1024_h128_128_128_92b58ff8_20260222_193216_ep029164.pt")

    agent = CustomAgent(custom_agent)

    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
