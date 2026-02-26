import numpy as np
import hockey.hockey_env as h_env
import json

from rl_hockey.sac import SAC

def main(
        checkpoint_path: str,
        config_path: str,
        N: int = 100,
        weak_opponent: bool = False,
        render: bool = False
):
    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
    o_space = env.observation_space
    ac_space = env.action_space

    config = json.load(open(config_path, "r"))
    agent = SAC(o_space.shape[0], action_dim=ac_space.shape[0] // 2, **config)
    agent.load(checkpoint_path)

    opponent = h_env.BasicOpponent(weak=weak_opponent)

    win_count = 0
    total_reward = 0

    for _ in range(N):
        state, _ = env.reset()

        episode = []
        for t in range(250):
            if render:
                env.render(mode="human")

            done = False
            action1 = agent.act(state.astype(np.float32), deterministic=True)
            action2 = opponent.act(env.obs_agent_two())

            (next_state, reward, done, trunc, info)  = env.step(np.hstack([action1, action2]))
            episode.append((state, next_state, reward, action1, action2))

            state = next_state

            total_reward += reward

            if done or trunc:
                break
        
        if info['winner'] == 1:
            win_count += 1

    print(f"Average Reward over {N} episodes: {total_reward / N}")
    print(f"Win Rate over {N} episodes: {win_count / N}")


if __name__ == "__main__":
    main(
        checkpoint_path="models/sac/sac.pt",
        config_path="models/sac/sac.json",
        N=100,
        weak_opponent=False,
        render=False
    )
