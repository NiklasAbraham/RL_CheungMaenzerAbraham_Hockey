"""
Continue training from an existing run directory.
Loads the latest checkpoint and continues training with the original configuration.
"""

import json
import logging
import os
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

# Imports after logging configuration to ensure logging is set up before modules use it
# ruff: noqa: E402
from rl_hockey.common.training.curriculum_manager import load_curriculum
from rl_hockey.common.training.run_manager import RunManager
from rl_hockey.common.training.train_run import _curriculum_to_dict, train_run
from rl_hockey.common.utils import set_cuda_device


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

    # Resume training with the checkpoint
    if verbose:
        logging.info(f"Resuming training from checkpoint: {checkpoint_path}")
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
    num_envs = int(os.environ.get("NUM_ENVS", "4"))

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
