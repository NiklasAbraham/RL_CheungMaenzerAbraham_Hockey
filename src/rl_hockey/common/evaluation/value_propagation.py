import os
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import ticker
import tqdm
from hockey import hockey_env as h_env

from rl_hockey.sac import SAC


def sample_trajectories(agent, opponent=None, num_trajectories=100, min_length=50) -> np.ndarray:
    env = h_env.HockeyEnv()

    if opponent is None:
        opponent = h_env.BasicOpponent(weak=False)
    
    trajectories = []

    pbar = tqdm.tqdm(range(num_trajectories))
    while len(trajectories) < num_trajectories:
        obs, _ = env.reset()
        trajectory = []
        done = False
        
        while not done:
            trajectory.append(obs)

            action1 = agent.act(obs.astype(np.float32))
            action2 = opponent.act(env.obs_agent_two())
            action = np.hstack([action1, action2])
            obs, reward, done, trunc, info = env.step(action)
            done = done or trunc
            
        
        if len(trajectory) >= min_length:
            trajectories.append(trajectory)
            pbar.update(1)
    
    trajectories = [t[-min_length:] for t in trajectories]
    
    return np.array(trajectories)


def evaluate_episodes(agent, trajectories: np.ndarray = None):
    if trajectories is None:
        trajectories = sample_trajectories(agent)

    q_values = np.zeros(len(trajectories[0]))
    q_values_sq = np.zeros(len(trajectories[0]))
    
    for trajectory in trajectories:
        for i, state in enumerate(trajectory):
            q = agent.evaluate(state.astype(np.float32))
            q_values[i] += q
            q_values_sq[i] += q ** 2
    
    n = len(trajectories)
    means = q_values / n
    variances = (q_values_sq / n) - (means ** 2)
    
    return means, variances


def plot_values_line(all_means, path, all_variances=None, labels=None, interval=500_000):
    if all_variances is None:
        all_variances = [np.zeros_like(means) for means in all_means]

    if labels is None:
        labels = [f"Step {(i+1)*interval}" for i in range(len(all_means))]

    plt.figure(figsize=(10, 6))
    for means, variances, label in zip(all_means, all_variances, labels):
        plt.plot(means, marker='o', label=label)
        plt.fill_between(range(len(means)), means - np.sqrt(variances), means + np.sqrt(variances), alpha=0.2)
    plt.title("Value Propagation")
    plt.xlabel("Steps from goal state")
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{len(means)-int(x)}"))
    plt.ylabel("Average Q-Value")
    plt.grid(True)
    plt.legend()
    plt.savefig(path)


def plot_value_heatmap(all_values, path, interval=500_000):    
    all_values = np.array(all_values)
    all_values = np.flipud(all_values.T)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(all_values, aspect='auto', cmap='plasma', origin='upper')
    plt.colorbar(label='Average Q-Value')
    plt.title("Value Propagation over Episodes")
    plt.xlabel(f"Training Steps ({interval:.0e})")
    plt.ylabel("Steps from goal state")
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.savefig(path)


if __name__ == "__main__":
    env = h_env.HockeyEnv()
    o_space = env.observation_space
    ac_space = env.action_space

    opponent = h_env.BasicOpponent(weak=False)
    
    agent = SAC(o_space.shape[0], action_dim=ac_space.shape[0] // 2, noise='pink', max_episode_steps=250)
    
    models_dir = "results/minimal_runs/8/models/"
    model_files = sorted([f for f in os.listdir(models_dir) if f.endswith(".pt")])

    agent.load(os.path.join(models_dir, model_files[-1]))
    trajectories = sample_trajectories(agent, opponent)

    all_means = []
    all_variances = []
    for model_file in model_files:
        agent.load(os.path.join(models_dir, model_file))
        means, variances = evaluate_episodes(agent, trajectories)
        all_means.append(means)
        all_variances.append(variances)

    np.save("value_propagation_means.npy", np.array(all_means))
    np.save("value_propagation_variances.npy",  np.array(all_variances))
    
    plot_values_line(all_means, all_variances=all_variances, path="value_propagation_line.png", labels=model_files)
    plot_value_heatmap(all_means, path="value_propagation_heatmap.png")
