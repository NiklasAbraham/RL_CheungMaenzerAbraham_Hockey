import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import hockey.hockey_env as h_env
import datetime
import random
import os

from rl_hockey.td3 import TD3


def evaluate_policy(agent, easy=True, num_eval_rounds=10):
    eval_env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
    if easy:
        opponent = h_env.BasicOpponent(weak=True)
    else:
        opponent = h_env.BasicOpponent(weak=False)

    mean_reward = 0.0
    winners = []
    print("-" * 20, "Evaluation", "-" * 20)

    for _ in range(num_eval_rounds):
        (state, _), done = eval_env.reset(), False
        state2 = eval_env.obs_agent_two()
        while not done:
            action = agent.act(state, deterministic=True)
            action2 = opponent.act(np.array(state2))
            state, reward, done, trunc, info = eval_env.step(
                np.hstack([action, action2])
            )
            state2 = eval_env.obs_agent_two()
            mean_reward += reward

        winners.append(info["winner"])

    mean_reward /= num_eval_rounds
    winrate = winners.count(1) / num_eval_rounds
    print(f"Overall Scores, {num_eval_rounds} games:")
    print(f"Mean reward: {mean_reward:.3f}")
    print(f"Win rate: {winrate:.3f}")
    eval_env.close()

    return winrate, mean_reward


def moving_average(data, window_size):
    moving_averages = []
    for i in range(len(data)):
        window_start = max(0, i - window_size + 1)
        window = data[window_start : i + 1]
        moving_averages.append(sum(window) / len(window))

    return moving_averages


if __name__ == "__main__":

    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"/home/stud389/RL_CheungMaenzerAbraham_Hockey/results/td3/{run_name}"
    os.makedirs(
        run_dir,
        exist_ok=True,
    )
    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] // 2
    max_action = float(env.action_space.high.min())

    max_episodes = 15000
    updates_per_step = 1
    warmup_steps = 50000
    max_episode_steps = 200

    new_config = {
        "critic_lr": 1e-5,
        "actor_lr": 1e-5,
        "critic_dim": 256,
        "actor_dim": 256,
        "actor_n_layers": 2,
        "critic_n_layers": 2,
        "batch_size": 256,
        "discount": 0.999,
        "action_min": float(-max_action),
        "action_max": float(max_action),
        "policy_update_delay": 2,
        "tau": 0.01,
        "noise_type": "normal",
        "exploration_noise": 0.15,
        "policy_noise": 0.1,
        "noise_clip": 0.5,
        "target_network_update_steps": 1,
        "verbose": True,
        "prioritized_replay": False,
    }

    # agent = SAC(o_space.shape[0], action_dim=ac_space.shape[0], noise='pink', max_episode_steps=max_episode_steps)
    agent = TD3(state_dim, action_dim=action_dim, **new_config)

    critic_losses = []
    actor_losses = []
    rewards = []
    steps = 0
    gradient_steps = 0
    evaluation_mean_rewards_strong = []
    evaluation_winrates_strong = []

    evaluation_mean_rewards_weak = []
    evaluation_winrates_weak = []

    # Evaluate untrained policy
    print("Evaluating untrained policy")
    weak_opponent_winrate, weak_opponentevaluation_mean_reward = evaluate_policy(
        agent, easy=True, num_eval_rounds=10
    )
    strong_opponent_winrate, strong_evaluation_mean_reward = evaluate_policy(
        agent, easy=False, num_eval_rounds=10
    )

    strong_opponent = h_env.BasicOpponent(weak=False)
    weak_opponent = h_env.BasicOpponent(weak=True)
    opponent_pool = [strong_opponent, weak_opponent]

    pbar = tqdm(range(max_episodes), desc="TRAIN BABY TRAIN")
    for i in pbar:
        total_reward = 0
        state, info = env.reset()
        player2_state = env.obs_agent_two()
        done = False

        agent.on_episode_start(i)

        episode_opponent = random.choice(opponent_pool)

        for t in range(max_episode_steps):
            if steps < warmup_steps:
                # Use Strong Opponent to generate transitions

                # Player 1 (warm up player, the one we are trying to learn from, aka the agent state)
                action = env.action_space.sample()[:action_dim]

                # Player 2, uses state 2
                action2 = strong_opponent.act(np.array(player2_state))
            else:
                action = agent.act(state)  # Agent's action

                # Player 2 action
                action2 = episode_opponent.act(np.array(player2_state))

            (next_state, reward, done, trunc, _) = env.step(
                np.hstack([action, action2])
            )
            agent.store_transition((state, action, reward, next_state, done))

            state = next_state
            player2_state = env.obs_agent_two()  # Update player 2 state

            steps += 1
            total_reward += reward

            if steps >= warmup_steps:
                stats = agent.train(updates_per_step)

                gradient_steps += updates_per_step
                critic_losses.extend(stats["critic_loss"])
                actor_losses.extend(stats["actor_loss"])

            if done or trunc:
                break

        agent.on_episode_end(i)
        rewards.append(total_reward)

        if (i + 1) % 100 == 0 and steps >= warmup_steps:
            weak_opponent_winrate, weak_opponent_evaluation_mean_reward = (
                evaluate_policy(agent, easy=True, num_eval_rounds=10)
            )
            evaluation_mean_rewards_weak.append(weak_opponent_evaluation_mean_reward)
            evaluation_winrates_weak.append(weak_opponent_winrate)

            strong_winrate, strong_evaluation_mean_reward = evaluate_policy(
                agent, easy=False, num_eval_rounds=10
            )
            evaluation_mean_rewards_strong.append(strong_evaluation_mean_reward)
            evaluation_winrates_strong.append(strong_winrate)
            agent.save(os.path.join(run_dir, "model.pt"))

        pbar.set_postfix(
            {
                "total_reward": total_reward,
                "episode_length": t,
            }
        )

    agent.save(os.path.join(run_dir, "model.pt"))

    fig, ax = plt.subplots()
    ax.plot(moving_average(rewards, 10))
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Total Reward")
    ax.set_title("Total Reward per Episode")
    fig.savefig(os.path.join(run_dir, "reward_plot.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(moving_average(critic_losses, 100))
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Critic Loss")
    ax.set_title("Critic Loss over Time")
    fig.savefig(os.path.join(run_dir, "critic_loss_plot.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(moving_average(actor_losses, 100))
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Actor Loss")
    ax.set_title("Actor Loss over Time")
    fig.savefig(os.path.join(run_dir, "actor_loss_plot.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    episode_indices = np.arange(1, len(evaluation_mean_rewards_weak) + 1) * 100
    ax.plot(episode_indices, evaluation_mean_rewards_weak, label="Weak")
    ax.plot(episode_indices, evaluation_mean_rewards_strong, label="Strong")
    ax.set_xlabel("Evaluation Number")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Evaluation Mean Reward over Time")
    ax.legend()
    fig.savefig(os.path.join(run_dir, "evaluation_mean_reward_plot.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    episode_indices = np.arange(1, len(evaluation_winrates_weak) + 1) * 100
    ax.plot(episode_indices, evaluation_winrates_weak, label="Weak")
    ax.plot(episode_indices, evaluation_winrates_strong, label="Strong")
    ax.set_xlabel("Evaluation Number")
    ax.set_ylabel("Win Rate")
    ax.set_title("Evaluation Win Rate over Time")
    ax.legend()
    fig.savefig(os.path.join(run_dir, "evaluation_winrate_plot.png"))
    plt.close(fig)
