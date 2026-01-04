"""
Continue training from an existing run directory.
Loads the latest checkpoint and continues training with a simple configuration.
"""
import json
import os
from pathlib import Path
from typing import Optional, Union
from rl_hockey.common.training.train_run import train_run
from rl_hockey.common.utils import set_cuda_device
import tempfile


def find_latest_checkpoint(models_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint in the models directory."""
    if not models_dir.exists():
        return None
    
    checkpoints = list(models_dir.glob("*.pt"))
    if not checkpoints:
        return None
    
    # Filter out temp evaluation checkpoints
    checkpoints = [c for c in checkpoints if "temp_eval" not in c.name]
    
    # Sort by modification time, most recent first
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def get_run_name_from_dir(run_dir: Path) -> Optional[str]:
    """Extract the run name from the directory structure."""
    configs_dir = run_dir / "configs"
    if not configs_dir.exists():
        return None
    
    # Find config files (should be only one per run directory)
    config_files = list(configs_dir.glob("*.json"))
    if not config_files:
        return None
    
    # Get run name from config file (remove .json extension)
    config_file = config_files[0]
    run_name = config_file.stem
    return run_name


def create_continuation_config(
    episodes: int = 12000,
    mixture_weights: dict = None,
    opponent_type: str = "basic_weak",
    learning_rate: float = 0.0001,
    batch_size: int = 256,
    max_episode_steps: int = 500,
    updates_per_step: int = 1,
    warmup_steps: int = 20000,
    reward_scale: float = 0.5,
    checkpoint_save_freq: int = 100,
    agent_type: str = "DDDQN",
    agent_hyperparameters: dict = None
) -> dict:
    """Create a simple continuation config with one mixed training phase."""
    if mixture_weights is None:
        mixture_weights = {
            "TRAIN_SHOOTING": 0.1,
            "TRAIN_DEFENSE": 0.1,
            "NORMAL": 0.8
        }
    
    if agent_hyperparameters is None:
        agent_hyperparameters = {}
    
    mixture_list = [
        {"mode": mode, "weight": weight}
        for mode, weight in mixture_weights.items()
    ]
    
    config = {
        "curriculum": {
            "phases": [
                {
                    "name": "continuation_mixed_training",
                    "episodes": episodes,
                    "environment": {
                        "mixture": mixture_list,
                        "keep_mode": True
                    },
                    "opponent": {
                        "type": opponent_type,
                        "weight": 1.0
                    },
                    "reward_shaping": None
                }
            ]
        },
        "hyperparameters": {
            "learning_rate": learning_rate,
            "batch_size": batch_size
        },
        "training": {
            "max_episode_steps": max_episode_steps,
            "updates_per_step": updates_per_step,
            "warmup_steps": warmup_steps,
            "reward_scale": reward_scale,
            "checkpoint_save_freq": checkpoint_save_freq
        },
        "agent": {
            "type": agent_type,
            "hyperparameters": agent_hyperparameters
        }
    }
    
    return config


def train_continue_run(
    existing_run_dir: str,
    continuation_config_path: Optional[str] = None,
    base_output_dir: str = "results/runs",
    run_name: str = None,
    verbose: bool = True,
    eval_freq_steps: int = 100000,
    eval_num_games: int = 200,
    eval_weak_opponent: bool = True,
    device: Optional[Union[str, int]] = None,
    continuation_episodes: int = 12000,
    checkpoint_episode: Optional[int] = None,
    num_envs: int = 1
):
    """
    Continue training from an existing run directory.
    
    Args:
        existing_run_dir: Path to the existing run directory (e.g., "results/hyperparameter_runs/2026-01-03_09-43-53")
        continuation_config_path: Optional path to a JSON config file for continuation.
                                  If None, uses default simple continuation config.
        base_output_dir: Base directory for saving continuation results
        run_name: Name for the continuation run (if None, generated automatically)
        verbose: Whether to print progress information
        eval_freq_steps: Frequency of evaluation in steps
        eval_num_games: Number of games to run for evaluation
        eval_weak_opponent: Whether to use weak (True) or strong (False) BasicOpponent for evaluation
        device: CUDA device to use
        continuation_episodes: Number of episodes for continuation (used if continuation_config_path is None)
        checkpoint_episode: Specific checkpoint episode to load (if None, loads latest)
        num_envs: Number of parallel environments (1 = single env, 4-8 recommended for speedup)
    """
    set_cuda_device(device)
    
    existing_run_dir = Path(existing_run_dir)
    if not existing_run_dir.exists():
        raise ValueError(f"Existing run directory does not exist: {existing_run_dir}")
    
    # Get run name from existing directory
    run_name_old = get_run_name_from_dir(existing_run_dir)
    if run_name_old is None:
        raise ValueError(f"Could not find config file in {existing_run_dir}")
    
    if verbose:
        print(f"Found existing run: {run_name_old}")
    
    # Load existing config to get agent parameters
    configs_dir = existing_run_dir / "configs"
    config_file = configs_dir / f"{run_name_old}.json"
    if not config_file.exists():
        raise ValueError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        existing_config = json.load(f)
    
    # Find latest checkpoint
    models_dir = existing_run_dir / "models"
    if checkpoint_episode is not None:
        checkpoint_path = models_dir / f"{run_name_old}_ep{checkpoint_episode:06d}.pt"
        if not checkpoint_path.exists():
            raise ValueError(f"Specified checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_path = find_latest_checkpoint(models_dir)
        if checkpoint_path is None:
            raise ValueError(f"No checkpoint found in {models_dir}")
    
    if verbose:
        print(f"Loading checkpoint: {checkpoint_path}")
    
    # Create continuation config
    if continuation_config_path is not None:
        if not os.path.exists(continuation_config_path):
            raise ValueError(f"Continuation config file not found: {continuation_config_path}")
        
        from rl_hockey.common.training.config_validator import validate_config
        errors = validate_config(continuation_config_path)
        if errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
        
        temp_config_path = continuation_config_path
        use_temp = False
    else:
        # Use default continuation config with parameters from existing config
        agent_hyperparams = existing_config.get('agent', {}).get('hyperparameters', {})
        common_hyperparams = existing_config.get('hyperparameters', {})
        training_params = existing_config.get('training', {})
        
        continuation_config_dict = create_continuation_config(
            episodes=continuation_episodes,
            learning_rate=common_hyperparams.get('learning_rate', 0.0001),
            batch_size=common_hyperparams.get('batch_size', 256),
            max_episode_steps=training_params.get('max_episode_steps', 500),
            updates_per_step=training_params.get('updates_per_step', 1),
            warmup_steps=training_params.get('warmup_steps', 20000),
            reward_scale=training_params.get('reward_scale', 0.5),
            checkpoint_save_freq=training_params.get('checkpoint_save_freq', 100),
            agent_type=existing_config.get('agent', {}).get('type', 'DDDQN'),
            agent_hyperparameters=agent_hyperparams
        )
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(continuation_config_dict, f, indent=2)
            temp_config_path = f.name
            use_temp = True
    
    try:
        # Call train_run with the checkpoint path
        return train_run(
            config_path=temp_config_path,
            base_output_dir=base_output_dir,
            run_name=run_name,
            verbose=verbose,
            eval_freq_steps=eval_freq_steps,
            eval_num_games=eval_num_games,
            eval_weak_opponent=eval_weak_opponent,
            device=device,
            checkpoint_path=str(checkpoint_path),
            num_envs=num_envs
        )
    finally:
        if use_temp and os.path.exists(temp_config_path):
            os.remove(temp_config_path)


if __name__ == "__main__":
    existing_run_dir = "results/hyperparameter_runs/2026-01-03_13-13-37"
    
    train_continue_run(
        existing_run_dir,
        continuation_config_path=None,  # Uses default continuation config
        base_output_dir="results/runs",
        device="cuda:1",
        continuation_episodes=12000,
        verbose=True,
        num_envs=60  # Use 60 parallel environments for speedup
    )
