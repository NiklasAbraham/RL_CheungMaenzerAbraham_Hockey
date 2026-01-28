import os
import numpy as np
import matplotlib.pyplot as plt


def evaluate_episodes(agent):
    episodes_path = "src/rl_hockey/common/evaluation/episodes.npy"
    episodes = np.load(episodes_path, allow_pickle=True)

    q_values = np.zeros(len(episodes[0]))
    for transactions in episodes:
        for i, state in enumerate(transactions):
            q = agent.evaluate(state.astype(np.float32))
            q_values[i] += q
    
    q_values /= len(episodes)
    return q_values


def plot_values(q_values, labels, path="q_values_comparison.png"):
    plt.figure(figsize=(10, 6))
    for qv, label in zip(q_values, labels):
        plt.plot(qv, marker='o', label=label)
    plt.title("Value Propagation over Episodes")
    plt.xlabel("Step")
    plt.ylabel("Average Q-Value")
    plt.grid(True)
    plt.legend()
    plt.savefig(path)
    # plt.show()


def plot_value_heatmap(q_values, path="q_values_heatmap.png"):    
    q_values = np.array(q_values)
    q_values = np.flipud(q_values.T)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(q_values, aspect='auto', cmap='plasma', origin='upper')
    plt.colorbar(label='Average Q-Value')
    plt.title("Value Propagation over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Steps from goal state")
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.savefig(path)
    
    # plt.show()


if __name__ == "__main__":
    # episodes = np.load("src/rl_hockey/common/evaluation/episodes.npy", allow_pickle=True)

    # from rl_hockey.sac import SAC

    # agent_new = SAC(len(episodes[0][0]), action_dim=4, noise='pink', max_episode_steps=500)

    # agent_trained = SAC(len(episodes[0][0]), action_dim=4, noise='pink', max_episode_steps=500)
    # agent_trained.load("results/hyperparameter_runs/2026-01-11_14-06-38/models/run_lr1e03_bs256_h128_128_128_4c1f51eb_20260111_140638_vec24.pt")

    # q_values_new = evaluate_episodes(agent_new, episodes)
    # q_values_trained = evaluate_episodes(agent_trained, episodes)

    # q_values_new = np.array(q_values_new)
    # q_values_trained = np.array(q_values_trained)

    # np.save("src/rl_hockey/common/evaluation/q_values_new.npy", q_values_new)
    # np.save("src/rl_hockey/common/evaluation/q_values_trained.npy", q_values_trained)

    # plot_values([q_values_new, q_values_trained], labels=["Untrained SAC", "Trained SAC"])

    q_values_new = np.load("src/rl_hockey/common/evaluation/q_values_new.npy", allow_pickle=True)
    q_values_trained = np.load("src/rl_hockey/common/evaluation/q_values_trained.npy", allow_pickle=True)

    plot_value_heatmap([q_values_new,q_values_new,q_values_new,q_values_new, q_values_trained])