import os
import numpy as np
import random
import matplotlib.pyplot as plt
import tqdm
from hockey import hockey_env as h_env

from rl_hockey.sac import SAC


def evaluate_episodes(agent, episodes=None):
    if episodes is None:
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
    env = h_env.HockeyEnv()
    o_space = env.observation_space
    ac_space = env.action_space

    opponent = h_env.BasicOpponent(weak=False)
    
    untrained_agent = SAC(o_space.shape[0], action_dim=ac_space.shape[0] // 2, noise='pink', max_episode_steps=250)
    
    trained_agent = SAC(o_space.shape[0], action_dim=ac_space.shape[0] // 2, noise='pink', max_episode_steps=250)
    trained_agent.load("minimal_runs/7/models/final.pt")
    
    # Collect episodes
    all_episodes = []
    for episode in tqdm.tqdm(range(500)):
        obs, info = env.reset()
        episode_states = []
        done = False
        
        while not done:
            episode_states.append(obs)

            action1 = trained_agent.act(obs.astype(np.float32))
            action2 = opponent.act(env.obs_agent_two())
            action = np.hstack([action1, action2])
            obs, reward, done, trunc, info = env.step(action)
            done = done or trunc
        
        all_episodes.append(episode_states)
    
    # Filter episodes
    filtered_episodes = [ep for ep in all_episodes if len(ep) >= 50]
    filtered_episodes = [ep[-50:] for ep in filtered_episodes]
    sampled_episodes = random.sample(filtered_episodes, min(100, len(filtered_episodes)))
    
    # Evaluate both agents
    untrained_q_values = evaluate_episodes(untrained_agent, sampled_episodes)
    trained_q_values = evaluate_episodes(trained_agent, sampled_episodes)
    
    # Create comparison plot
    plot_values(
        [untrained_q_values, trained_q_values],
        ["Untrained Agent", "Trained Agent"],
        path="q_values_comparison.png"
    )
    
    print("Evaluation complete. Plot saved as q_values_comparison.png")
