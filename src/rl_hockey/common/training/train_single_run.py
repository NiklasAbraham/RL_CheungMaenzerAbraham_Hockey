"""
Single training run with curriculum learning support.
Entry point: run this file to start training (e.g. python src/rl_hockey/common/training/train_single_run.py).

Uses JSON config file path; config_path is passed through to train_run and create_agent
so that relative opponent paths in the config are resolved against the project root.
Supports vectorized environments for faster training.
"""

import logging
import sys
from typing import Optional, Union

# Configure logging (ensure it's configured before importing train_run)
# Unbuffer stdout for immediate output in batch jobs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,  # Force reconfiguration if already configured
)

from rl_hockey.common.training.curriculum_manager import load_curriculum
from rl_hockey.common.training.run_manager import RunManager
from rl_hockey.common.training.train_run import _curriculum_to_dict, train_run


def train_single_run(
    config_path: str,
    base_output_dir: str = "results/runs",
    run_name: str = None,
    verbose: bool = True,
    eval_freq_steps: int = 100_000,
    eval_num_games: int = 100,
    eval_weak_opponent: bool = True,
    device: Optional[Union[str, int]] = None,
    checkpoint_path: Optional[str] = None,
    num_envs: int = 4,
):
    """
    Train a single run with optional vectorized environments.

    Args:
        config_path: Path to curriculum config JSON
        base_output_dir: Directory for saving results
        run_name: Name for this run
        verbose: Print progress
        eval_freq_steps: Evaluation frequency
        eval_num_games: Number of evaluation games
        eval_weak_opponent: Use weak opponent for eval
        device: CUDA device
        checkpoint_path: Continue from checkpoint
        num_envs: Number of parallel environments (1 = no vectorization, 4-8 recommended)
                    4 cores: use 2, 8 cores: use 4, 12+ cores: use 8 (max recommended)
    """
    # Load curriculum to generate run_name and save config
    curriculum = load_curriculum(config_path)
    config_dict = _curriculum_to_dict(curriculum)

    # Create RunManager to set up directory structure
    run_manager = RunManager(base_output_dir=base_output_dir)

    # Generate run_name if not provided
    if run_name is None:
        run_name = run_manager.generate_run_name(config_dict)

    # Save config file before starting training
    if verbose:
        logging.info(f"Saving config file for run: {run_name}")
    run_manager.save_config(run_name, config_dict)
    if verbose:
        logging.info(
            f"Config saved to: {run_manager.get_run_directories(run_name)['config']}"
        )

    return train_run(
        config_path,
        base_output_dir,
        run_name,
        verbose,
        eval_freq_steps=eval_freq_steps,
        eval_num_games=eval_num_games,
        eval_weak_opponent=eval_weak_opponent,
        device=device,
        checkpoint_path=checkpoint_path,
        num_envs=num_envs,
        run_manager=run_manager,
    )


if __name__ == "__main__":
    import os

    import torch

    # Device diagnostics (help debug "skipping cudagraphs due to cpu device" etc.)
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print(f"[Device] CUDA_VISIBLE_DEVICES={cuda_vis}")
    print(f"[Device] torch.cuda.is_available()={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Device] torch.cuda.device_count()={torch.cuda.device_count()}")

    # Enable TF32 for better performance on Ampere+ GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    path_to_config = "configs/curriculum_tdmpc2_opponent_simulation.json"

    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"[Device] Using GPU {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        print(
            "[Device] Using CPU (no GPU). TD-MPC2 repo expects GPU; use sbatch on a GPU partition."
        )

    # Get num_envs from environment variable if set, otherwise use default
    num_envs = int(
        os.environ.get("NUM_ENVS", "1")
    )  # Default to 4 for parallel environments

    train_single_run(
        path_to_config,
        base_output_dir="results/tdmpc2_runs",
        device=device,
        num_envs=num_envs,
    )

    # nohup python -u src/rl_hockey/common/training/train_single_run.py > results/tdmpc2_runs/train_single_run.log 2>&1 &
