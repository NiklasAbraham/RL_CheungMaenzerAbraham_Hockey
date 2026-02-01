"""
Single training run with curriculum learning support - REFACTORED VERSION.
Uses the refactored train_run implementation with cleaner, modular code.
Can use either JSON config file or dict config.
Supports vectorized environments for faster training.
"""

import logging
import sys
from pathlib import Path
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

from rl_hockey.common.training.run_manager import RunManager
from rl_hockey.common.training.train_run_refactored import train_run


def _resume_from_checkpoint(
    checkpoint_path: str,
    existing_run_dir: Optional[str],
) -> tuple:
    """Infer run_name, start_episode, and initial_episode_logs from checkpoint path."""
    cp = Path(checkpoint_path).resolve()
    if not cp.exists():
        return None, 0, None
    stem = cp.stem
    if "_ep" in stem:
        run_name = stem.rsplit("_ep", 1)[0]
    else:
        run_name = stem
    start_episode = 0
    initial_episode_logs = None
    metadata_path = cp.parent / f"{stem}_metadata.json"
    if metadata_path.exists():
        import json

        with open(metadata_path) as f:
            meta = json.load(f)
        start_episode = int(meta.get("episode", 0)) + 1
    if existing_run_dir is not None:
        csvs_dir = Path(existing_run_dir) / "csvs"
        episode_logs_path = csvs_dir / f"{stem}_episode_logs.csv"
        if episode_logs_path.exists():
            initial_episode_logs = RunManager.load_episode_logs_csv(episode_logs_path)
    return run_name, start_episode, initial_episode_logs


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
    existing_run_dir: Optional[str] = None,
    num_envs: int = 1,
    log_freq_episodes: int = 1,
    enable_initial_q_value_propagation: bool = False,
):
    """
    Train a single run with optional vectorized environments.

    Args:
        config_path: Path to curriculum config JSON
        base_output_dir: Directory for saving results (ignored if existing_run_dir is set)
        run_name: Name for this run (when resuming, inferred from checkpoint if not set)
        verbose: Print progress
        eval_freq_steps: Evaluation frequency
        eval_num_games: Number of evaluation games
        eval_weak_opponent: Use weak opponent for eval
        device: CUDA device
        checkpoint_path: Path to .pt checkpoint to continue from (e.g. .../models/run_name_ep000800.pt)
        existing_run_dir: When resuming, path to the existing run folder (e.g. results/tdmpc2_runs_test/2026-02-01_09-55-22).
            Results and further checkpoints are written into this folder.
        num_envs: Number of parallel environments (1 = no vectorization, 4-8 recommended)
                    4 cores: use 2, 8 cores: use 4, 12+ cores: use 8 (max recommended)
        log_freq_episodes: Logging frequency in episodes
        enable_initial_q_value_propagation: If True, evaluate and store initial Q-values before training
    """
    run_manager = None
    start_episode = 0
    initial_episode_logs = None
    if checkpoint_path:
        inferred_name, start_episode, initial_episode_logs = _resume_from_checkpoint(
            checkpoint_path, existing_run_dir
        )
        if run_name is None:
            run_name = inferred_name
    if existing_run_dir is not None:
        run_manager = RunManager(existing_run_dir=existing_run_dir)
        if run_name is None:
            configs_dir = Path(existing_run_dir) / "configs"
            if configs_dir.exists():
                jsons = list(configs_dir.glob("*.json"))
                if jsons:
                    run_name = jsons[0].stem
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
        log_freq_episodes=log_freq_episodes,
        enable_initial_q_value_propagation=enable_initial_q_value_propagation,
        run_manager=run_manager,
        start_episode=start_episode,
        initial_episode_logs=initial_episode_logs,
    )


if __name__ == "__main__":
    import torch

    # Enable TF32 for better performance on Ampere+ GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # Resolve config path relative to project root (parent of src/)
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parents[
        3
    ]  # training -> common -> rl_hockey -> src -> project
    path_to_config = (
        PROJECT_ROOT / "configs" / "curriculum_tdmpc2_mixed_opponents.json"
    ).resolve()

    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"CUDA available: Using GPU {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        print("CUDA not available: Using CPU")

    # Get num_envs from environment variable if set, otherwise use default
    import os

    num_envs = int(os.environ.get("NUM_ENVS", "1"))  # Default to 24 envs if not set

    # train_single_run(
    #    path_to_config,
    #    base_output_dir="results/tdmpc2_runs_test",
    #    device=device,
    #    num_envs=num_envs,
    # )

    # Resume from last checkpoint (use same config and num_envs as original run):
    train_single_run(
        path_to_config,
        base_output_dir="results/tdmpc2_runs_test",
        device=device,
        num_envs=num_envs,
        checkpoint_path="results/tdmpc2_runs_test/2026-02-01_09-55-22/models/TDMPC2_run_lr3e04_bs512_hencoder_dynamics_reward_termination_q_function_policy_add21d6e_20260201_095522_ep000800.pt",
        existing_run_dir="results/tdmpc2_runs_test/2026-02-01_09-55-22",
    )

    # nohup python -u src/rl_hockey/common/training/train_single_run_refactored.py > results/sac_runs/train_single_run_refactored.log 2>&1 &
