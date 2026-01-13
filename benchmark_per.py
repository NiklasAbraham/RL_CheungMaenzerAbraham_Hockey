"""
Benchmark script to compare training speed between DDDQN and DDQN_PER.
"""

import json
import time
from pathlib import Path

from rl_hockey.common.training.train_single_run import train_single_run


def benchmark_agent(config_path, agent_name, num_runs=1):
    """Benchmark a single agent configuration."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking: {agent_name}")
    print(f"{'=' * 60}")

    times = []

    for run_idx in range(num_runs):
        print(f"\nRun {run_idx + 1}/{num_runs}")

        start_time = time.time()

        try:
            train_single_run(
                config_path=config_path,
                base_output_dir="results/benchmark",
                run_name=f"{agent_name}_run_{run_idx}",
                verbose=False,
                num_envs=1,
                device="cpu",
            )

            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"  Completed in {elapsed:.2f} seconds")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    if times:
        avg_time = sum(times) / len(times)
        print(f"\nAverage time: {avg_time:.2f} seconds")
        print(f"Times: {times}")
        return avg_time, times
    else:
        return None, None


def main():
    """Run benchmark comparison."""
    print("PER vs Standard DDDQN Benchmark")
    print("=" * 60)

    # Config files
    config_standard = "configs/curriculum_test.json"
    config_per = "configs/curriculum_test_per.json"

    # Verify configs exist
    if not Path(config_standard).exists():
        print(f"Error: {config_standard} not found")
        return

    if not Path(config_per).exists():
        print(f"Error: {config_per} not found")
        return

    num_runs = 1  # Number of runs per agent

    # Benchmark standard DDDQN
    std_avg, std_times = benchmark_agent(config_standard, "DDDQN_Standard", num_runs)

    # Benchmark DDQN_PER
    per_avg, per_times = benchmark_agent(config_per, "DDQN_PER", num_runs)

    # Compare results
    print(f"\n{'=' * 60}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 60}")

    if std_avg and per_avg:
        print(f"\nStandard DDDQN:  {std_avg:.2f} seconds")
        print(f"DDQN with PER:   {per_avg:.2f} seconds")

        speedup = std_avg / per_avg
        slowdown = per_avg / std_avg

        if speedup > 1:
            print(f"\nPER is {speedup:.2f}x FASTER")
        else:
            print(f"\nPER is {slowdown:.2f}x SLOWER")

        overhead = ((per_avg - std_avg) / std_avg) * 100
        print(f"PER overhead: {overhead:+.1f}%")

        # Save results
        results = {
            "standard_ddqn": {"average_time": std_avg, "times": std_times},
            "ddqn_per": {"average_time": per_avg, "times": per_times},
            "overhead_percent": overhead,
            "speedup_factor": speedup if speedup > 1 else slowdown,
        }

        results_path = "results/benchmark_results.json"
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_path}")
    else:
        print("\nBenchmark incomplete - some runs failed")


if __name__ == "__main__":
    main()
