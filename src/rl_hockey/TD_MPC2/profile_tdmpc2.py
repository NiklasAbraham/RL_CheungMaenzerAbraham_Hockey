"""
Profile TD-MPC2 agent to identify performance bottlenecks.
Uses torch.profiler to measure time spent in different operations.
"""

import argparse
import gc
import json
import os

import numpy as np
import torch

from rl_hockey.common.training.agent_factory import create_agent
from rl_hockey.common.training.curriculum_manager import AgentConfig


def load_agent_from_config(config_path: str, device: str = "cuda"):
    """Load TD-MPC2 agent from config file."""
    with open(config_path, "r") as f:
        config = json.load(f)

    agent_config_dict = config.get("agent", {})
    state_dim = 18  # Standard hockey env observation dimension
    action_dim = 8  # Standard hockey env action dimension

    agent_config = AgentConfig(
        type=agent_config_dict["type"],
        hyperparameters=agent_config_dict.get("hyperparameters", {}),
    )

    agent = create_agent(
        agent_config,
        state_dim,
        action_dim,
        {},
        device=device,
    )

    if hasattr(agent, "encoder"):
        agent.encoder.eval()
    if hasattr(agent, "dynamics"):
        agent.dynamics.eval()
    if hasattr(agent, "reward"):
        agent.reward.eval()
    if hasattr(agent, "q_ensemble"):
        agent.q_ensemble.eval()
    if hasattr(agent, "policy"):
        agent.policy.eval()
    if hasattr(agent, "target_q_ensemble"):
        agent.target_q_ensemble.eval()

    return agent


def profile_single_action(
    agent, obs_dim: int, num_warmup: int = 10, num_iterations: int = 100
):
    """Profile single action selection."""
    print("\n" + "=" * 80)
    print("PROFILING: Single Action Selection (act)")
    print("=" * 80)

    device = agent.device
    obs = torch.randn(obs_dim).to(device)

    for _ in range(num_warmup):
        _ = agent.act(obs.cpu().numpy())

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        if device != "cpu"
        else [torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,  # Disabled to save memory
        profile_memory=False,  # Disabled to save memory
        with_stack=False,  # Disabled to save memory
    ) as prof:
        with torch.profiler.record_function("act_total"):
            for _ in range(num_iterations):
                _ = agent.act(obs.cpu().numpy())

    print(f"\nProfiled {num_iterations} action selections")
    print("\nTop time-consuming operations:")
    table_str = prof.key_averages().table(
        sort_by="cuda_time_total" if device != "cpu" else "cpu_time_total",
        row_limit=30,
    )
    print(table_str)

    return table_str


def profile_batch_action(
    agent,
    obs_dim: int,
    batch_size: int = 4,
    num_warmup: int = 10,
    num_iterations: int = 50,
):
    """Profile batch action selection."""
    print("\n" + "=" * 80)
    print(f"PROFILING: Batch Action Selection (act_batch, batch_size={batch_size})")
    print("=" * 80)

    device = agent.device
    obs_batch = torch.randn(batch_size, obs_dim).to(device)

    for _ in range(num_warmup):
        _ = agent.act_batch(obs_batch.cpu().numpy())

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        if device != "cpu"
        else [torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,  # Disabled to save memory
        profile_memory=False,  # Disabled to save memory
        with_stack=False,  # Disabled to save memory
    ) as prof:
        with torch.profiler.record_function("act_batch_total"):
            for _ in range(num_iterations):
                _ = agent.act_batch(obs_batch.cpu().numpy())

    print(f"\nProfiled {num_iterations} batch action selections")
    print("\nTop time-consuming operations:")
    table_str = prof.key_averages().table(
        sort_by="cuda_time_total" if device != "cpu" else "cpu_time_total",
        row_limit=30,
    )
    print(table_str)

    return table_str


def profile_planning_step(
    agent, obs_dim: int, num_warmup: int = 10, num_iterations: int = 100
):
    """Profile a single planning step in detail."""
    print("\n" + "=" * 80)
    print("PROFILING: Planning Step Details")
    print("=" * 80)

    device = agent.device
    obs = torch.randn(obs_dim).to(device)

    z = agent.encoder(obs.unsqueeze(0))
    for _ in range(num_warmup):
        _ = agent.planner.plan(z.squeeze(0), return_mean=True)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        if device != "cpu"
        else [torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,  # Disabled to save memory
        profile_memory=False,  # Disabled to save memory
        with_stack=False,  # Disabled to save memory
    ) as prof:
        with torch.profiler.record_function("planning_total"):
            for _ in range(num_iterations):
                z = agent.encoder(obs.unsqueeze(0)).squeeze(0)
                with torch.profiler.record_function("planner_plan"):
                    _ = agent.planner.plan(z, return_mean=True)

    print(f"\nProfiled {num_iterations} planning steps")
    print("\nTop time-consuming operations in planning:")
    table_str = prof.key_averages().table(
        sort_by="cuda_time_total" if device != "cpu" else "cpu_time_total",
        row_limit=30,
    )
    print(table_str)

    return table_str


def profile_model_forward_passes(
    agent, obs_dim: int, num_samples: int, horizon: int, num_iterations: int = 50
):
    """Profile individual model forward passes."""
    print("\n" + "=" * 80)
    print("PROFILING: Individual Model Forward Passes")
    print("=" * 80)

    device = agent.device
    batch_size = num_samples
    latent_dim = agent.latent_dim
    action_dim = agent.action_dim

    obs = torch.randn(obs_dim).to(device)
    latent = torch.randn(latent_dim).to(device)
    action = torch.randn(action_dim).to(device)
    latents_batch = torch.randn(batch_size, latent_dim).to(device)
    actions_batch = torch.randn(batch_size, action_dim).to(device)

    for _ in range(10):
        _ = agent.encoder(obs.unsqueeze(0))
        _ = agent.dynamics(latent.unsqueeze(0), action.unsqueeze(0))
        _ = agent.reward(latent.unsqueeze(0), action.unsqueeze(0))
        _ = agent.q_ensemble.min(latent.unsqueeze(0), action.unsqueeze(0))
        _ = agent.dynamics(latents_batch, actions_batch)

    results = []

    print("\nProfiling encoder...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        if device != "cpu"
        else [torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,  # Disabled to save memory
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(num_iterations):
            with torch.profiler.record_function("encoder"):
                _ = agent.encoder(obs.unsqueeze(0))

    print("\nEncoder forward pass:")
    table_str = prof.key_averages().table(
        sort_by="cuda_time_total" if device != "cpu" else "cpu_time_total",
        row_limit=10,
    )
    print(table_str)
    results.append(("encoder", table_str))
    del prof
    if device != "cpu":
        torch.cuda.empty_cache()
        gc.collect()

    print("\nProfiling dynamics (single)...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        if device != "cpu"
        else [torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,  # Disabled to save memory
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(num_iterations):
            with torch.profiler.record_function("dynamics_single"):
                _ = agent.dynamics(latent.unsqueeze(0), action.unsqueeze(0))

    print("\nDynamics forward pass (single):")
    table_str = prof.key_averages().table(
        sort_by="cuda_time_total" if device != "cpu" else "cpu_time_total",
        row_limit=10,
    )
    print(table_str)
    results.append(("dynamics_single", table_str))
    del prof
    if device != "cpu":
        torch.cuda.empty_cache()
        gc.collect()

    print("\nProfiling dynamics (batched)...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        if device != "cpu"
        else [torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,  # Disabled to save memory
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(num_iterations):
            with torch.profiler.record_function("dynamics_batch"):
                _ = agent.dynamics(latents_batch, actions_batch)

    print("\nDynamics forward pass (batch):")
    table_str = prof.key_averages().table(
        sort_by="cuda_time_total" if device != "cpu" else "cpu_time_total",
        row_limit=10,
    )
    print(table_str)
    results.append(("dynamics_batch", table_str))
    del prof
    if device != "cpu":
        torch.cuda.empty_cache()
        gc.collect()

    print("\nProfiling reward (batched)...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        if device != "cpu"
        else [torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,  # Disabled to save memory
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(num_iterations):
            with torch.profiler.record_function("reward_batch"):
                _ = agent.reward(latents_batch, actions_batch)

    print("\nReward forward pass (batch):")
    table_str = prof.key_averages().table(
        sort_by="cuda_time_total" if device != "cpu" else "cpu_time_total",
        row_limit=10,
    )
    print(table_str)
    results.append(("reward_batch", table_str))
    del prof
    if device != "cpu":
        torch.cuda.empty_cache()
        gc.collect()

    print("\nProfiling Q-ensemble (batched)...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        if device != "cpu"
        else [torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,  # Disabled to save memory
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(num_iterations):
            with torch.profiler.record_function("q_ensemble_batch"):
                _ = agent.q_ensemble.min(latents_batch, actions_batch)

    print("\nQ-ensemble forward pass (batch):")
    table_str = prof.key_averages().table(
        sort_by="cuda_time_total" if device != "cpu" else "cpu_time_total",
        row_limit=10,
    )
    print(table_str)
    results.append(("q_ensemble_batch", table_str))
    del prof
    if device != "cpu":
        torch.cuda.empty_cache()
        gc.collect()

    return results


def profile_training_step(
    agent,
    obs_dim: int,
    batch_size: int = 256,
    num_warmup: int = 5,
    num_iterations: int = 50,
):
    """Profile training step including forward and backward passes."""
    print("\n" + "=" * 80)
    print(f"PROFILING: Training Step (batch_size={batch_size})")
    print("=" * 80)

    device = agent.device

    if hasattr(agent, "encoder"):
        agent.encoder.train()
    if hasattr(agent, "dynamics"):
        agent.dynamics.train()
    if hasattr(agent, "reward"):
        agent.reward.train()
    if hasattr(agent, "q_ensemble"):
        agent.q_ensemble.train()
    if hasattr(agent, "policy"):
        agent.policy.train()

    action_dim = agent.action_dim
    horizon = agent.horizon if hasattr(agent, "horizon") else 5
    episodes_needed = max(batch_size // 4, 10)  # At least 10 episodes
    transitions_per_episode = horizon + 5
    if hasattr(agent.buffer, "clear"):
        agent.buffer.clear()
    
    for ep in range(episodes_needed):
        for step in range(transitions_per_episode):
            obs = torch.randn(obs_dim).cpu().numpy()
            action = torch.randn(action_dim).cpu().numpy()
            reward = np.random.randn()
            next_obs = torch.randn(obs_dim).cpu().numpy()
            done = (step == transitions_per_episode - 1)
            agent.buffer.store((obs, action, reward, next_obs, done))

    for _ in range(num_warmup):
        _ = agent.train(steps=1)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        if device != "cpu"
        else [torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,  # Disabled to save memory
        profile_memory=False,  # Disabled to save memory
        with_stack=False,  # Disabled to save memory
    ) as prof:
        with torch.profiler.record_function("training_total"):
            for _ in range(num_iterations):
                with torch.profiler.record_function("train_step"):
                    _ = agent.train(steps=1)

    print(f"\nProfiled {num_iterations} training steps")
    print("\nTop time-consuming operations:")
    table_str = prof.key_averages().table(
        sort_by="cuda_time_total" if device != "cpu" else "cpu_time_total",
        row_limit=30,
    )
    print(table_str)

    print("\n" + "=" * 80)
    print("CPU-GPU Transfer Operations (memcpy, MemcpyHtoD, etc.):")
    print("=" * 80)
    all_ops = prof.key_averages()
    transfer_ops = [
        op
        for op in all_ops
        if any(
            keyword in str(op.key)
            for keyword in [
                "memcpy",
                "Memcpy",
                "copy",
                "to",
                "from_numpy",
                "FloatTensor",
            ]
        )
    ]
    if transfer_ops:
        print("\nTransfer-related operations (sorted by CPU time):")
        transfer_ops_sorted = sorted(
            transfer_ops, key=lambda x: x.cpu_time_total, reverse=True
        )
        for op in transfer_ops_sorted[:20]:  # Show top 20 transfer operations
            cpu_time = op.cpu_time_total / 1000.0  # Convert to ms
            cuda_time = (
                op.cuda_time_total / 1000.0 if hasattr(op, "cuda_time_total") else 0.0
            )
            print(
                f"  {op.key}: CPU={cpu_time:.3f}ms, CUDA={cuda_time:.3f}ms, Calls={op.count}"
            )
    else:
        print("\nNo explicit transfer operations found in top operations.")
        print("Transfer overhead may be included in tensor creation operations.")
        print("Look for operations with high CPU time but low CUDA time.")

    print("\n" + "=" * 80)
    print("Memory Usage:")
    print("=" * 80)
    print(
        prof.key_averages().table(
            sort_by="cuda_memory_usage" if device != "cpu" else "cpu_memory_usage",
            row_limit=20,
        )
    )

    if hasattr(agent, "encoder"):
        agent.encoder.eval()
    if hasattr(agent, "dynamics"):
        agent.dynamics.eval()
    if hasattr(agent, "reward"):
        agent.reward.eval()
    if hasattr(agent, "q_ensemble"):
        agent.q_ensemble.eval()
    if hasattr(agent, "policy"):
        agent.policy.eval()

    return table_str


def export_trace(prof, output_path: str):
    """Export profiling trace for visualization in Chrome tracing."""
    prof.export_chrome_trace(output_path)
    print(f"\nTrace exported to: {output_path}")
    print("Open in Chrome: chrome://tracing")


def main(
    config_path: str = "configs/curriculum_tdmpc2.json",
    device: str = None,
    output_dir: str = "results/profiling",
    num_iterations: int = 100,
    export_traces: bool = False,
    gpu_id: int = None,
):
    """Main profiling function."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device != "cpu" and torch.cuda.is_available():
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            device = f"cuda:{gpu_id}"
        else:
            device = "cuda"

        torch.cuda.empty_cache()

        current_device = torch.cuda.current_device()
        print(f"Using device: {device}")
        print(f"GPU {current_device}: {torch.cuda.get_device_name(current_device)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(current_device).total_memory / 1e9:.1f} GB"
        )

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(current_device) / 1e9
            reserved = torch.cuda.memory_reserved(current_device) / 1e9
            print(
                f"GPU Memory Status: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved"
            )
    else:
        print(f"Using device: {device}")

    print(f"\nLoading agent from config: {config_path}")
    agent = load_agent_from_config(config_path, device=device)

    obs_dim = agent.obs_dim
    num_samples = agent.num_samples
    horizon = agent.horizon

    print("\nAgent configuration:")
    print(f"  obs_dim: {obs_dim}")
    print(f"  action_dim: {agent.action_dim}")
    print(f"  latent_dim: {agent.latent_dim}")
    print(f"  horizon: {horizon}")
    print(f"  num_samples: {num_samples}")
    print(f"  num_iterations: {agent.num_iterations}")

    try:
        os.makedirs(output_dir, exist_ok=True)
        test_file = os.path.join(output_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"Output directory created/verified: {os.path.abspath(output_dir)}")
    except Exception as e:
        print(f"ERROR: Cannot create or write to output directory: {output_dir}")
        print(f"Error: {e}")
        raise

    print("\n" + "=" * 80)
    print("STARTING PROFILING")
    print("=" * 80)

    profiling_results = []

    def clear_gpu_cache():
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

    def print_memory_status():
        if device != "cpu" and torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(current_device) / 1e9
            reserved = torch.cuda.memory_reserved(current_device) / 1e9
            total = torch.cuda.get_device_properties(current_device).total_memory / 1e9
            free = total - reserved
            print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {free:.2f} GB free")

    try:
        clear_gpu_cache()
        print_memory_status()
        print("\n[1/5] Profiling single action selection...")
        table1 = profile_single_action(
            agent, obs_dim, num_iterations=num_iterations
        )
        profiling_results.append(("SINGLE ACTION SELECTION", table1))
        clear_gpu_cache()
        print_memory_status()
    except Exception as e:
        print(f"\nWARNING: Failed to profile single action: {e}")
        profiling_results.append(("SINGLE ACTION SELECTION", f"ERROR: {str(e)}"))
        clear_gpu_cache()

    try:
        print("\n[2/5] Profiling batch action selection...")
        table2 = profile_batch_action(
            agent, obs_dim, batch_size=4, num_iterations=num_iterations // 2
        )
        profiling_results.append(("BATCH ACTION SELECTION", table2))
        clear_gpu_cache()
        print_memory_status()
    except Exception as e:
        print(f"\nWARNING: Failed to profile batch action: {e}")
        profiling_results.append(("BATCH ACTION SELECTION", f"ERROR: {str(e)}"))
        clear_gpu_cache()

    try:
        print("\n[3/5] Profiling planning step...")
        table3 = profile_planning_step(
            agent, obs_dim, num_iterations=num_iterations
        )
        profiling_results.append(("PLANNING STEP", table3))
        clear_gpu_cache()
        print_memory_status()
    except Exception as e:
        print(f"\nWARNING: Failed to profile planning step: {e}")
        profiling_results.append(("PLANNING STEP", f"ERROR: {str(e)}"))
        clear_gpu_cache()

    try:
        print("\n[4/5] Profiling model forward passes...")
        model_results = profile_model_forward_passes(
            agent, obs_dim, num_samples, horizon, num_iterations=min(num_iterations, 30)
        )
        for name, table in model_results:
            profiling_results.append((f"MODEL FORWARD PASS: {name.upper()}", table))
        clear_gpu_cache()
        print_memory_status()
    except Exception as e:
        print(f"\nWARNING: Failed to profile model forward passes: {e}")
        profiling_results.append(("MODEL FORWARD PASSES", f"ERROR: {str(e)}"))
        clear_gpu_cache()

    try:
        print("\n[5/5] Profiling training step...")
        table4 = profile_training_step(
            agent,
            obs_dim,
            batch_size=agent.config.get("batch_size", 256) if hasattr(agent, "config") else 256,
            num_iterations=num_iterations // 2,  # Fewer iterations for training
        )
        profiling_results.append(("TRAINING STEP", table4))
        clear_gpu_cache()
        print_memory_status()
    except Exception as e:
        print(f"\nWARNING: Failed to profile training step: {e}")
        profiling_results.append(("TRAINING STEP", f"ERROR: {str(e)}"))
        clear_gpu_cache()

    summary_path = os.path.join(output_dir, "profiling_summary.txt")
    summary_path_abs = os.path.abspath(summary_path)
    print(f"\nSaving profiling summary to: {summary_path_abs}")
    try:
        with open(summary_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("TD-MPC2 PROFILING SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Device: {device}\n")
            if device != "cpu" and torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                f.write(
                    f"GPU {current_device}: {torch.cuda.get_device_name(current_device)}\n"
                )
                f.write(
                    f"GPU Memory: {torch.cuda.get_device_properties(current_device).total_memory / 1e9:.1f} GB\n"
                )
            f.write(f"Config: {config_path}\n")
            f.write(f"Iterations: {num_iterations}\n")
            f.write("\nAgent configuration:\n")
            f.write(f"  obs_dim: {obs_dim}\n")
            f.write(f"  action_dim: {agent.action_dim}\n")
            f.write(f"  latent_dim: {agent.latent_dim}\n")
            f.write(f"  horizon: {horizon}\n")
            f.write(f"  num_samples: {num_samples}\n")
            f.write(f"  num_iterations: {agent.num_iterations}\n")
            f.write("\n" + "=" * 80 + "\n\n")

            for section_name, table in profiling_results:
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"{section_name}\n")
                f.write("=" * 80 + "\n\n")
                f.write(table)
                f.write("\n\n")

            f.write("=" * 80 + "\n")
            f.write("PROFILING COMPLETE\n")
            f.write("=" * 80 + "\n")

        print("\n" + "=" * 80)
        print("PROFILING COMPLETE")
        print("=" * 80)
        print("\nSummary:")
        print("  - Inference profiling: single_action, batch_action, planning")
        print("  - Training profiling: training_step (forward + backward + optimizer)")
        print(f"\nProfiling summary saved to: {summary_path_abs}")
        print(f"File size: {os.path.getsize(summary_path)} bytes")
    except Exception as e:
        print(f"\nERROR: Failed to save profiling summary to {summary_path_abs}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        try:
            error_report_path = os.path.join(output_dir, "profiling_error.txt")
            with open(error_report_path, "w") as f:
                f.write(f"Profiling failed with error:\n{str(e)}\n\n")
                f.write("Traceback:\n")
                traceback.print_exc(file=f)
            print(f"Error report saved to: {os.path.abspath(error_report_path)}")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile TD-MPC2 agent performance")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/curriculum_tdmpc2.json",
        help="Path to curriculum config JSON file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detects if not specified.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/profiling",
        help="Directory to save profiling traces",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of iterations to profile",
    )
    parser.add_argument(
        "--export-traces",
        action="store_true",
        help="Export Chrome tracing files for visualization",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="GPU ID to use (0, 1, 2, etc.). Uses default GPU if not specified.",
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        device=args.device,
        output_dir=args.output_dir,
        num_iterations=args.num_iterations,
        export_traces=args.export_traces,
        gpu_id=args.gpu_id,
    )
