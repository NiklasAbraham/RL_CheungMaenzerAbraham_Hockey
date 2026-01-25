"""
Continue training from an existing run directory.
Loads the latest checkpoint and continues training with the original configuration.
"""

import csv
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

# Imports after logging configuration to ensure logging is set up before modules use it
# ruff: noqa: E402
from rl_hockey.common.training.curriculum_manager import load_curriculum
from rl_hockey.common.training.run_manager import RunManager
from rl_hockey.common.training.train_run import _curriculum_to_dict, train_run
from rl_hockey.common.utils import set_cuda_device


def load_episode_logs_from_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load episode logs from a CSV file."""
    episode_logs = []

    if not csv_path.exists():
        return episode_logs

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episode_log = {
                "episode": int(row["episode"]),
                "reward": float(row["reward"]) if row["reward"] else 0.0,
                "shaped_reward": float(row["shaped_reward"])
                if row["shaped_reward"]
                else 0.0,
                "backprop_reward": float(row["backprop_reward"])
                if row.get("backprop_reward")
                else 0.0,
                "losses": {},
            }

            # Extract all loss columns
            for key, value in row.items():
                if (
                    key
                    not in [
                        "episode",
                        "reward",
                        "shaped_reward",
                        "backprop_reward",
                        "total_gradient_steps",
                    ]
                    and value
                ):
                    try:
                        episode_log["losses"][key] = float(value)
                    except ValueError:
                        pass

            episode_logs.append(episode_log)

    return episode_logs


def find_all_episode_log_files(csvs_dir: Path, run_name: str) -> List[Path]:
    """Find all episode log CSV files for a run (including checkpoints)."""
    log_files = []

    # Main episode logs file
    main_file = csvs_dir / f"{run_name}_episode_logs.csv"
    if main_file.exists():
        log_files.append(main_file)

    # Checkpoint episode logs files (pattern: {run_name}_ep{episode}_episode_logs.csv)
    pattern = f"{run_name}_ep*_episode_logs.csv"
    checkpoint_files = sorted(csvs_dir.glob(pattern))
    log_files.extend(checkpoint_files)

    return sorted(log_files, key=lambda x: x.name)


def combine_episode_logs(log_files: List[Path]) -> List[Dict[str, Any]]:
    """Combine episode logs from multiple CSV files, removing duplicates."""
    all_logs = {}

    for log_file in log_files:
        logs = load_episode_logs_from_csv(log_file)
        for log in logs:
            episode_num = log["episode"]
            # Keep the latest entry if there are duplicates
            all_logs[episode_num] = log

    # Sort by episode number
    sorted_logs = [all_logs[ep] for ep in sorted(all_logs.keys())]
    return sorted_logs


def find_latest_checkpoint(models_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint in the models directory, excluding temp evaluation checkpoints."""
    if not models_dir.exists():
        return None

    checkpoints = list(models_dir.glob("*.pt"))
    if not checkpoints:
        return None

    # Filter out temp evaluation checkpoints
    checkpoints = [c for c in checkpoints if "temp_eval" not in c.name]

    if not checkpoints:
        return None

    # Sort by modification time, most recent first
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def train_continue_run(
    base_path: str,
    base_output_dir: str = None,
    run_name: str = None,
    verbose: bool = True,
    eval_freq_steps: int = 100_000,
    eval_num_games: int = 100,
    eval_weak_opponent: bool = True,
    device: Optional[Union[str, int]] = None,
    checkpoint_episode: Optional[int] = None,
    num_envs: int = 4,
):
    """
    Resume training from the last saved checkpoint in a run directory.
    Uses the original config from the run directory and continues training.

    Args:
        base_path: Path to the existing run directory (e.g., "results/tdmpc2_runs/2026-01-21_19-12-44")
        base_output_dir: Directory for saving results (if None, uses parent of base_path)
        run_name: Name for this run (if None, extracted from config file)
        verbose: Print progress
        eval_freq_steps: Evaluation frequency
        eval_num_games: Number of evaluation games
        eval_weak_opponent: Use weak opponent for eval
        device: CUDA device
        checkpoint_episode: Specific checkpoint episode to load (if None, loads latest)
        num_envs: Number of parallel environments (1 = no vectorization, 4-8 recommended)
    """
    set_cuda_device(device)

    base_path = Path(base_path)
    if not base_path.exists():
        raise ValueError(f"Base path does not exist: {base_path}")

    # Find config file
    configs_dir = base_path / "configs"
    if not configs_dir.exists():
        raise ValueError(f"Configs directory not found: {configs_dir}")

    config_files = list(configs_dir.glob("*.json"))
    if not config_files:
        raise ValueError(f"No config file found in {configs_dir}")

    # Use the first config file (should be only one per run)
    config_file = config_files[0]
    if verbose:
        logging.info(f"Found config file: {config_file}")

    # Extract run_name from config filename if not provided
    if run_name is None:
        run_name = config_file.stem
        if verbose:
            logging.info(f"Extracted run name: {run_name}")

    # Find checkpoint
    models_dir = base_path / "models"
    if checkpoint_episode is not None:
        checkpoint_path = models_dir / f"{run_name}_ep{checkpoint_episode:06d}.pt"
        if not checkpoint_path.exists():
            raise ValueError(f"Specified checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_path = find_latest_checkpoint(models_dir)
        if checkpoint_path is None:
            raise ValueError(f"No checkpoint found in {models_dir}")

    if verbose:
        logging.info(f"Found checkpoint: {checkpoint_path}")

    # Load metadata if available
    start_episode = 0
    metadata_path = checkpoint_path.parent / f"{checkpoint_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        start_episode = metadata.get("episode", 0)
        if verbose:
            logging.info(
                f"Checkpoint metadata: episode={start_episode}, "
                f"phase_index={metadata.get('phase_index')}, "
                f"phase_episode={metadata.get('phase_episode')}"
            )
            logging.info(f"Resuming training from episode {start_episode}")

    # Determine base_output_dir
    if base_output_dir is None:
        # Use parent directory of base_path (e.g., if base_path is results/tdmpc2_runs/2026-01-21_19-12-44,
        # base_output_dir should be results/tdmpc2_runs)
        base_output_dir = str(base_path.parent)

    # Load curriculum to verify it's valid
    curriculum = load_curriculum(str(config_file))
    config_dict = _curriculum_to_dict(curriculum)

    # Create RunManager to set up directory structure
    run_manager = RunManager(base_output_dir=base_output_dir)

    # Save config file (this will create a new run directory, but that's okay for resuming)
    if verbose:
        logging.info(f"Saving config file for resumed run: {run_name}")
    run_manager.save_config(run_name, config_dict)

    # Load old episode logs from CSV files
    old_csvs_dir = base_path / "csvs"
    old_episode_logs = []
    if old_csvs_dir.exists():
        log_files = find_all_episode_log_files(old_csvs_dir, run_name)
        if log_files:
            old_episode_logs = combine_episode_logs(log_files)
            # Filter to only include episodes up to start_episode
            old_episode_logs = [
                log for log in old_episode_logs if log["episode"] <= start_episode
            ]
            if verbose:
                logging.info(
                    f"Loaded {len(old_episode_logs)} old episode logs (up to episode {start_episode})"
                )

            # Copy old CSV files to new run directory
            new_csvs_dir = run_manager.csvs_dir
            for log_file in log_files:
                try:
                    shutil.copy2(log_file, new_csvs_dir / log_file.name)
                    if verbose:
                        logging.info(f"Copied {log_file.name} to new run directory")
                except Exception as e:
                    if verbose:
                        logging.warning(f"Could not copy {log_file.name}: {e}")

    # Resume training with the checkpoint
    if verbose:
        logging.info(f"Resuming training from checkpoint: {checkpoint_path}")
        if old_episode_logs:
            logging.info(f"Loaded {len(old_episode_logs)} old episode logs into memory")
        logging.info(
            "Note: Buffer will be empty and will fill up as training continues"
        )

    return train_run(
        str(config_file),
        base_output_dir,
        run_name,
        verbose,
        eval_freq_steps=eval_freq_steps,
        eval_num_games=eval_num_games,
        eval_weak_opponent=eval_weak_opponent,
        device=device,
        checkpoint_path=str(checkpoint_path),
        num_envs=num_envs,
        run_manager=run_manager,
        start_episode=start_episode,
        initial_episode_logs=old_episode_logs if old_episode_logs else None,
    )


if __name__ == "__main__":
    import torch

    # Enable TF32 for better performance on Ampere+ GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"CUDA available: Using GPU {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = "cpu"
        print("CUDA not available: Using CPU")

    # Get num_envs from environment variable if set, otherwise use default
    num_envs = int(os.environ.get("NUM_ENVS", "1"))

    # Get base_path from environment variable or use default
    base_path = os.environ.get("RESUME_PATH", "results/tdmpc2_runs/2026-01-21_16-15-43")

    print(f"Resuming training from: {base_path}")
    train_continue_run(
        base_path=base_path,
        base_output_dir="results/tdmpc2_runs",
        device=device,
        num_envs=num_envs,
        verbose=True,
    )

    # Usage examples:
    # python -u src/rl_hockey/common/training/train_continue_run.py
    # RESUME_PATH=results/tdmpc2_runs/2026-01-21_19-12-44 python -u src/rl_hockey/common/training/train_continue_run.py
    # nohup python -u src/rl_hockey/common/training/train_continue_run.py > results/tdmpc2_runs/train_continue_run.log 2>&1 &
