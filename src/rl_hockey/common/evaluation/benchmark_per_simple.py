"""
Simple benchmark to compare training step speed between DDDQN and DDQN_PER.
Measures only the training step time, not full episode execution.
"""

import time
import numpy as np
import torch
import hockey.hockey_env as h_env

from rl_hockey.DDDQN import DDDQN, DDQN_PER
from rl_hockey.common.utils import get_discrete_action_dim


def fill_buffer(agent, num_transitions=10000):
    """Fill the agent's buffer with random transitions."""
    env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING, keep_mode=True)
    state, _ = env.reset()
    state_dim = env.observation_space.shape[0]
    action_dim = 7 if not env.keep_mode else 8
    
    from rl_hockey.common.utils import discrete_to_continuous_action_standard
    
    for _ in range(num_transitions):
        discrete_action = np.random.randint(0, action_dim)
        # Convert to continuous action for player 1
        action_p1 = discrete_to_continuous_action_standard(discrete_action, keep_mode=env.keep_mode)
        # Player 2 does nothing
        action_p2 = np.zeros(4 if env.keep_mode else 3, dtype=np.float32)
        # Concatenate actions
        full_action = np.hstack([action_p1, action_p2])
        
        next_state, reward, done, truncated, _ = env.step(full_action)
        
        agent.store_transition((state, np.array([discrete_action]), reward, next_state, done))
        
        if done or truncated:
            state, _ = env.reset()
        else:
            state = next_state
    
    env.close()


def benchmark_training_steps(agent, num_steps=1000, batch_size=256):
    """Benchmark training step execution time."""
    times = []
    
    for _ in range(num_steps):
        start = time.time()
        agent.train(steps=1)
        elapsed = time.time() - start
        times.append(elapsed)
    
    return times


def main():
    """Run benchmark comparison."""
    print("PER vs Standard DDDQN Training Step Benchmark")
    print("="*60)
    
    # Create environment to get dimensions
    env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING, keep_mode=True)
    state_dim = env.observation_space.shape[0]
    action_dim = 7 if not env.keep_mode else 8
    env.close()
    
    # Configuration
    num_transitions = 10000
    num_training_steps = 500
    batch_size = 256
    
    print(f"\nBuffer size: {num_transitions} transitions")
    print(f"Training steps: {num_training_steps}")
    print(f"Batch size: {batch_size}")
    
    # Create standard DDDQN
    print(f"\n{'='*60}")
    print("Setting up Standard DDDQN...")
    agent_standard = DDDQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=[256, 256],
        batch_size=batch_size
    )
    fill_buffer(agent_standard, num_transitions)
    print("Buffer filled")
    
    # Benchmark standard DDDQN
    print("\nBenchmarking Standard DDDQN training steps...")
    std_times = benchmark_training_steps(agent_standard, num_training_steps, batch_size)
    std_avg = np.mean(std_times)
    std_std = np.std(std_times)
    print(f"Average: {std_avg*1000:.2f} ms ± {std_std*1000:.2f} ms per step")
    print(f"Total: {sum(std_times):.2f} seconds")
    
    # Create DDQN_PER
    print(f"\n{'='*60}")
    print("Setting up DDQN_PER...")
    agent_per = DDQN_PER(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=[256, 256],
        use_per=True,
        batch_size=batch_size,
        per_alpha=0.6,
        per_beta=0.4
    )
    fill_buffer(agent_per, num_transitions)
    print("Buffer filled")
    
    # Benchmark DDQN_PER
    print("\nBenchmarking DDQN_PER training steps...")
    per_times = benchmark_training_steps(agent_per, num_training_steps, batch_size)
    per_avg = np.mean(per_times)
    per_std = np.std(per_times)
    print(f"Average: {per_avg*1000:.2f} ms ± {per_std*1000:.2f} ms per step")
    print(f"Total: {sum(per_times):.2f} seconds")
    
    # Compare results
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    
    slowdown = per_avg / std_avg
    overhead = ((per_avg - std_avg) / std_avg) * 100
    
    print(f"\nStandard DDDQN:  {std_avg*1000:.2f} ms ± {std_std*1000:.2f} ms per step")
    print(f"DDQN with PER:   {per_avg*1000:.2f} ms ± {per_std*1000:.2f} ms per step")
    print(f"\nPER is {slowdown:.2f}x SLOWER")
    print(f"PER overhead: {overhead:+.1f}%")
    
    # Save results
    results = {
        "standard_ddqn": {
            "avg_time_ms": std_avg * 1000,
            "std_time_ms": std_std * 1000,
            "total_time_sec": sum(std_times)
        },
        "ddqn_per": {
            "avg_time_ms": per_avg * 1000,
            "std_time_ms": per_std * 1000,
            "total_time_sec": sum(per_times)
        },
        "slowdown_factor": slowdown,
        "overhead_percent": overhead,
        "num_training_steps": num_training_steps,
        "batch_size": batch_size,
        "buffer_size": num_transitions
    }
    
    import json
    from pathlib import Path
    results_path = "results/benchmark_per_results.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
