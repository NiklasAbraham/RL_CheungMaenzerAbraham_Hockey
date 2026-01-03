"""
Hyperparameter tuning script for curriculum learning.
Samples N configurations from the grid, trains them, evaluates against BasicOpponent,
and plots performance against different parameters.
"""
import multiprocessing as mp
from itertools import product
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import torch
import json
import random
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rl_hockey.common.training.run_manager import RunManager
from rl_hockey.common.training.train_run import train_run
from rl_hockey.common.training.curriculum_manager import load_curriculum, CurriculumConfig
from rl_hockey.common.evaluation.agent_evaluator import evaluate_agent


def load_base_config(config_path: str) -> Dict[str, Any]:
    """Load base configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def merge_configs(curriculum_config: Dict[str, Any], hyperparameter_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge curriculum config (constant) with hyperparameter config (grid)."""
    # Start with curriculum config as base (this preserves curriculum and training)
    merged = json.loads(json.dumps(curriculum_config))  # Deep copy
    
    # Override hyperparameters from hyperparameter_config (these contain lists for grid search)
    if 'hyperparameters' in hyperparameter_config:
        merged['hyperparameters'] = hyperparameter_config['hyperparameters']
    
    # Merge agent config: use hyperparameter_config for hyperparameters (grid), but keep type from curriculum if not specified
    if 'agent' in hyperparameter_config:
        if 'agent' not in merged:
            merged['agent'] = {}
        if 'hyperparameters' in hyperparameter_config['agent']:
            merged['agent']['hyperparameters'] = hyperparameter_config['agent']['hyperparameters']
        # Agent type: prefer hyperparameter_config, fallback to curriculum_config
        if 'type' in hyperparameter_config['agent']:
            merged['agent']['type'] = hyperparameter_config['agent']['type']
        elif 'type' not in merged.get('agent', {}):
            # Keep type from curriculum if not in hyperparameter config
            pass
    
    # Override training parameters from hyperparameter_config if present (optional)
    if 'training' in hyperparameter_config:
        merged['training'] = hyperparameter_config['training']
    
    # IMPORTANT: Curriculum must always come from curriculum_config, never from hyperparameter_config
    # (This is already preserved since we start with curriculum_config)
    
    return merged


def create_hyperparameter_grid(curriculum_config: Dict[str, Any], hyperparameter_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate hyperparameter grid by merging curriculum (constant) with hyperparameter ranges."""
    # Extract hyperparameter ranges from hyperparameter_config
    hyperparams = hyperparameter_config.get('hyperparameters', {})
    
    # Common hyperparameters (can be lists for grid search)
    learning_rates = hyperparams.get('learning_rate', [1e-4])
    if not isinstance(learning_rates, list):
        learning_rates = [learning_rates]
    
    batch_sizes = hyperparams.get('batch_size', [256])
    if not isinstance(batch_sizes, list):
        batch_sizes = [batch_sizes]
    
    # Algorithm-specific hyperparameters (can be lists for grid search)
    agent_hyperparams = hyperparameter_config.get('agent', {}).get('hyperparameters', {})
    
    # Agent type: prefer hyperparameter_config, fallback to curriculum_config
    agent_type = (hyperparameter_config.get('agent', {}).get('type') or 
                  curriculum_config.get('agent', {}).get('type'))
    
    if agent_type is None:
        raise ValueError("agent.type must be specified in either hyperparameter_config or curriculum_config")
    
    configs = []
    
    if agent_type == "DDDQN":
        # DDDQN specific hyperparameters
        hidden_dims = agent_hyperparams.get('hidden_dim', [[256, 256, 256]])
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]
        # Check if first element is a list (nested list) or a single value
        if len(hidden_dims) > 0 and not isinstance(hidden_dims[0], list):
            hidden_dims = [hidden_dims]
        
        target_update_freqs = agent_hyperparams.get('target_update_freq', [50])
        if not isinstance(target_update_freqs, list):
            target_update_freqs = [target_update_freqs]
        
        eps_decays = agent_hyperparams.get('eps_decay', [0.999])
        if not isinstance(eps_decays, list):
            eps_decays = [eps_decays]
        
        eps_values = agent_hyperparams.get('eps', [1.0])
        if not isinstance(eps_values, list):
            eps_values = [eps_values]
        
        eps_mins = agent_hyperparams.get('eps_min', [0.05])
        if not isinstance(eps_mins, list):
            eps_mins = [eps_mins]
        
        # Get other agent hyperparameters that should be constant (not in grid)
        agent_constants = {}
        for key, value in agent_hyperparams.items():
            if key not in ['hidden_dim', 'target_update_freq', 'eps_decay', 'eps', 'eps_min']:
                agent_constants[key] = value
        
        # Generate all combinations for DDDQN
        for lr, bs, hd, tuf, ed, eps, eps_min in product(learning_rates, batch_sizes, hidden_dims, target_update_freqs, eps_decays, eps_values, eps_mins):
            # Start with curriculum config to ensure curriculum is always from curriculum_config
            config = json.loads(json.dumps(curriculum_config))  # Deep copy
            
            # Update common hyperparameters with single values (not lists)
            if 'hyperparameters' not in config:
                config['hyperparameters'] = {}
            config['hyperparameters']['learning_rate'] = lr
            config['hyperparameters']['batch_size'] = bs
            
            # Update agent-specific hyperparameters with single values
            if 'agent' not in config:
                config['agent'] = {}
            if 'hyperparameters' not in config['agent']:
                config['agent']['hyperparameters'] = {}
            config['agent']['hyperparameters']['hidden_dim'] = hd
            config['agent']['hyperparameters']['target_update_freq'] = tuf
            config['agent']['hyperparameters']['eps_decay'] = ed
            config['agent']['hyperparameters']['eps'] = eps
            config['agent']['hyperparameters']['eps_min'] = eps_min
            
            # Set agent type
            config['agent']['type'] = agent_type
            
            # Add constant agent hyperparameters from hyperparameter_config
            for key, value in agent_constants.items():
                config['agent']['hyperparameters'][key] = value
            
            # Override training parameters from hyperparameter_config if present
            if 'training' in hyperparameter_config:
                config['training'] = hyperparameter_config['training']
            
            configs.append(config)
    
    elif agent_type == "SAC":
        # SAC specific hyperparameters
        taus = agent_hyperparams.get('tau', [0.005])
        if not isinstance(taus, list):
            taus = [taus]
        
        alphas = agent_hyperparams.get('alpha', [0.2])
        if not isinstance(alphas, list):
            alphas = [alphas]
        
        learn_alphas = agent_hyperparams.get('learn_alpha', [True])
        if not isinstance(learn_alphas, list):
            learn_alphas = [learn_alphas]
        
        noise_types = agent_hyperparams.get('noise', ['normal'])
        if not isinstance(noise_types, list):
            noise_types = [noise_types]
        
        # Get other agent hyperparameters that should be constant (not in grid)
        agent_constants = {}
        for key, value in agent_hyperparams.items():
            if key not in ['tau', 'alpha', 'learn_alpha', 'noise']:
                agent_constants[key] = value
        
        # Generate all combinations for SAC
        for lr, bs, tau, alpha, learn_alpha, noise in product(learning_rates, batch_sizes, taus, alphas, learn_alphas, noise_types):
            # Start with curriculum config to ensure curriculum is always from curriculum_config
            config = json.loads(json.dumps(curriculum_config))  # Deep copy
            
            # Update common hyperparameters with single values (not lists)
            if 'hyperparameters' not in config:
                config['hyperparameters'] = {}
            config['hyperparameters']['learning_rate'] = lr
            config['hyperparameters']['batch_size'] = bs
            
            # Update agent-specific hyperparameters with single values
            if 'agent' not in config:
                config['agent'] = {}
            if 'hyperparameters' not in config['agent']:
                config['agent']['hyperparameters'] = {}
            config['agent']['hyperparameters']['tau'] = tau
            config['agent']['hyperparameters']['alpha'] = alpha
            config['agent']['hyperparameters']['learn_alpha'] = learn_alpha
            config['agent']['hyperparameters']['noise'] = noise
            
            # Set agent type
            config['agent']['type'] = agent_type
            
            # Add constant agent hyperparameters from hyperparameter_config
            for key, value in agent_constants.items():
                config['agent']['hyperparameters'][key] = value
            
            # Override training parameters from hyperparameter_config if present
            if 'training' in hyperparameter_config:
                config['training'] = hyperparameter_config['training']
            
            configs.append(config)
    
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}. Supported types are 'DDDQN' and 'SAC'")
    
    return configs




def sample_configurations(all_configs: List[Dict[str, Any]], n: int, random_seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Sample N configurations from the full grid."""
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    if n >= len(all_configs):
        return all_configs
    
    return random.sample(all_configs, n)


def run_single_config(args):
    """Wrapper function for running a single configuration (for multiprocessing)."""
    config_dict, run_output_dir, run_name, device = args
    
    # Set CUDA device if specified and available
    if device is not None:
        if isinstance(device, str):
            if device == 'cpu':
                pass  # Will use CPU (torch will detect this)
            elif device == 'cuda':
                # Use default CUDA device (cuda:0)
                if torch.cuda.is_available():
                    torch.cuda.set_device(0)
                    torch.cuda.empty_cache()
            elif device.startswith('cuda:'):
                device_id = int(device.split(':')[1])
                if torch.cuda.is_available():
                    if device_id >= torch.cuda.device_count():
                        raise ValueError(f"CUDA device {device_id} not available. Only {torch.cuda.device_count()} device(s) available.")
                    torch.cuda.set_device(device_id)
                    torch.cuda.empty_cache()
                else:
                    raise ValueError(f"CUDA not available, but device '{device}' was requested")
            else:
                raise ValueError(f"Invalid device string: {device}. Use 'cpu', 'cuda', or 'cuda:N'")
        elif isinstance(device, int):
            # Allow integer device ID for convenience
            if torch.cuda.is_available():
                if device >= torch.cuda.device_count():
                    raise ValueError(f"CUDA device {device} not available. Only {torch.cuda.device_count()} device(s) available.")
                torch.cuda.set_device(device)
                torch.cuda.empty_cache()
            else:
                raise ValueError(f"CUDA not available, but device {device} was requested")
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    import tempfile
    import os
    from pathlib import Path
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_dict, f, indent=2)
        temp_config_path = f.name
    
    # Create run-specific directory
    run_dir = Path(run_output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Pass the run directory to train_run (it will create a timestamped subdirectory inside)
    result = train_run(temp_config_path, str(run_dir), run_name=run_name, verbose=False)
    
    # Find the actual model path (train_run creates its own timestamped directory inside run_dir)
    # Look for model in run_dir / timestamp / models
    model_path = None
    if run_dir.exists():
        timestamped_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
        
        # Search in all subdirectories for the model
        for subdir in sorted(timestamped_dirs, key=lambda x: x.stat().st_mtime, reverse=True):
            models_dir = subdir / "models"
            if models_dir.exists():
                # Try exact match first
                candidate = models_dir / f"{run_name}.pt"
                if candidate.exists():
                    model_path = candidate
                    break
                # Try pattern match
                model_files = list(models_dir.glob(f"{run_name}*.pt"))
                if model_files:
                    # Get the most recent one (likely the final model)
                    model_path = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                    break
    
    if model_path and model_path.exists():
        result['model_path'] = str(model_path)
    
    os.unlink(temp_config_path)
    return result


def plot_hyperparameter_analysis(results: List[Dict[str, Any]], output_dir: Path):
    """Plot performance against different hyperparameters."""
    # Extract parameter values and performance metrics
    learning_rates = []
    batch_sizes = []
    hidden_dims = []
    target_update_freqs = []
    eps_decays = []
    eps_values = []
    eps_mins = []
    win_rates = []
    mean_rewards = []
    
    for result in results:
        config = result['config']
        learning_rates.append(config['hyperparameters']['learning_rate'])
        batch_sizes.append(config['hyperparameters']['batch_size'])
        
        agent_hp = config['agent']['hyperparameters']
        hidden_dims.append(str(agent_hp.get('hidden_dim', [256, 256, 256])))
        target_update_freqs.append(agent_hp.get('target_update_freq', 50))
        eps_decays.append(agent_hp.get('eps_decay', 0.999))
        eps_values.append(agent_hp.get('eps', 1.0))
        eps_mins.append(agent_hp.get('eps_min', 0.05))
        
        eval_result = result.get('evaluation', {})
        win_rates.append(eval_result.get('win_rate', 0.0))
        mean_rewards.append(eval_result.get('mean_reward', 0.0))
    
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Plot 1: Learning Rate vs Win Rate (assuming other params constant)
    unique_lrs = sorted(set(learning_rates))
    if len(unique_lrs) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        lr_win_rates = {lr: [] for lr in unique_lrs}
        for lr, wr in zip(learning_rates, win_rates):
            lr_win_rates[lr].append(wr)
        
        lr_means = [np.mean(lr_win_rates[lr]) for lr in unique_lrs]
        lr_stds = [np.std(lr_win_rates[lr]) for lr in unique_lrs]
        
        x_pos = np.arange(len(unique_lrs))
        ax.bar(x_pos, lr_means, yerr=lr_stds, capsize=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{lr:.0e}" if lr < 0.01 else f"{lr}" for lr in unique_lrs], rotation=45, ha='right')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate vs Learning Rate (other parameters constant)')
        ax.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(plots_dir / "win_rate_vs_learning_rate.png")
        plt.close()
    
    # Plot 2: Batch Size vs Win Rate
    unique_bs = sorted(set(batch_sizes))
    if len(unique_bs) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        bs_win_rates = {bs: [] for bs in unique_bs}
        for bs, wr in zip(batch_sizes, win_rates):
            bs_win_rates[bs].append(wr)
        
        bs_means = [np.mean(bs_win_rates[bs]) for bs in unique_bs]
        bs_stds = [np.std(bs_win_rates[bs]) for bs in unique_bs]
        
        x_pos = np.arange(len(unique_bs))
        ax.bar(x_pos, bs_means, yerr=bs_stds, capsize=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(bs) for bs in unique_bs], rotation=45, ha='right')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate vs Batch Size (other parameters constant)')
        ax.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(plots_dir / "win_rate_vs_batch_size.png")
        plt.close()
    
    # Plot 3: Target Update Freq vs Win Rate
    unique_tuf = sorted(set(target_update_freqs))
    if len(unique_tuf) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        tuf_win_rates = {tuf: [] for tuf in unique_tuf}
        for tuf, wr in zip(target_update_freqs, win_rates):
            tuf_win_rates[tuf].append(wr)
        
        tuf_means = [np.mean(tuf_win_rates[tuf]) for tuf in unique_tuf]
        tuf_stds = [np.std(tuf_win_rates[tuf]) for tuf in unique_tuf]
        
        x_pos = np.arange(len(unique_tuf))
        ax.bar(x_pos, tuf_means, yerr=tuf_stds, capsize=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(tuf) for tuf in unique_tuf], rotation=45, ha='right')
        ax.set_xlabel('Target Update Frequency')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate vs Target Update Frequency (other parameters constant)')
        ax.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(plots_dir / "win_rate_vs_target_update_freq.png")
        plt.close()
    
    # Plot 4: Epsilon Decay vs Win Rate
    unique_ed = sorted(set(eps_decays))
    if len(unique_ed) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ed_win_rates = {ed: [] for ed in unique_ed}
        for ed, wr in zip(eps_decays, win_rates):
            ed_win_rates[ed].append(wr)
        
        ed_means = [np.mean(ed_win_rates[ed]) for ed in unique_ed]
        ed_stds = [np.std(ed_win_rates[ed]) for ed in unique_ed]
        
        x_pos = np.arange(len(unique_ed))
        ax.bar(x_pos, ed_means, yerr=ed_stds, capsize=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{ed:.4f}" for ed in unique_ed], rotation=45, ha='right')
        ax.set_xlabel('Epsilon Decay')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate vs Epsilon Decay (other parameters constant)')
        ax.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(plots_dir / "win_rate_vs_eps_decay.png")
        plt.close()
    
    # Plot 5: Epsilon (eps) vs Win Rate
    unique_eps = sorted(set(eps_values))
    if len(unique_eps) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        eps_win_rates = {eps: [] for eps in unique_eps}
        for eps, wr in zip(eps_values, win_rates):
            eps_win_rates[eps].append(wr)
        
        eps_means = [np.mean(eps_win_rates[eps]) for eps in unique_eps]
        eps_stds = [np.std(eps_win_rates[eps]) for eps in unique_eps]
        
        x_pos = np.arange(len(unique_eps))
        ax.bar(x_pos, eps_means, yerr=eps_stds, capsize=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{eps:.2f}" for eps in unique_eps], rotation=45, ha='right')
        ax.set_xlabel('Epsilon (eps)')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate vs Epsilon (eps) (other parameters constant)')
        ax.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(plots_dir / "win_rate_vs_eps.png")
        plt.close()
    
    # Plot 6: Epsilon Min (eps_min) vs Win Rate
    unique_eps_min = sorted(set(eps_mins))
    if len(unique_eps_min) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        eps_min_win_rates = {eps_min: [] for eps_min in unique_eps_min}
        for eps_min, wr in zip(eps_mins, win_rates):
            eps_min_win_rates[eps_min].append(wr)
        
        eps_min_means = [np.mean(eps_min_win_rates[eps_min]) for eps_min in unique_eps_min]
        eps_min_stds = [np.std(eps_min_win_rates[eps_min]) for eps_min in unique_eps_min]
        
        x_pos = np.arange(len(unique_eps_min))
        ax.bar(x_pos, eps_min_means, yerr=eps_min_stds, capsize=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{eps_min:.2f}" for eps_min in unique_eps_min], rotation=45, ha='right')
        ax.set_xlabel('Epsilon Min (eps_min)')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate vs Epsilon Min (eps_min) (other parameters constant)')
        ax.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(plots_dir / "win_rate_vs_eps_min.png")
        plt.close()
    
    # Plot 7: Hidden Dim Architecture comparison
    unique_hd = sorted(set(hidden_dims))
    if len(unique_hd) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        hd_win_rates = {hd: [] for hd in unique_hd}
        for hd, wr in zip(hidden_dims, win_rates):
            hd_win_rates[hd].append(wr)
        
        hd_means = [np.mean(hd_win_rates[hd]) for hd in unique_hd]
        hd_stds = [np.std(hd_win_rates[hd]) for hd in unique_hd]
        
        x_pos = np.arange(len(unique_hd))
        ax.bar(x_pos, hd_means, yerr=hd_stds, capsize=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(unique_hd, rotation=45, ha='right')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate vs Hidden Layer Architecture (other parameters constant)')
        ax.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(plots_dir / "win_rate_vs_hidden_dim.png")
        plt.close()
    
    print(f"Plots saved to: {plots_dir}")


def main(
    curriculum_config_path: str,
    hyperparameter_config_path: str,
    n_samples: int = 10,
    num_parallel: int = None,
    output_dir: str = "results/hyperparameter_runs",
    eval_num_games: int = 100,
    eval_weak_opponent: bool = True,
    random_seed: Optional[int] = None,
    device: Optional[Union[str, int]] = None,
    devices: Optional[List[Union[str, int]]] = None
):
    """
    Main function to run hyperparameter testing with curriculum learning.
    
    Args:
        curriculum_config_path: Path to curriculum configuration JSON file (constant, not varied)
        hyperparameter_config_path: Path to hyperparameter configuration JSON file (contains lists for grid search)
        n_samples: Number of configurations to sample from the grid
        num_parallel: Number of parallel workers (default: auto-detect, max 4 for GPU)
        output_dir: Base output directory for results
        eval_num_games: Number of games to run for evaluation
        eval_weak_opponent: Whether to use weak (True) or strong (False) BasicOpponent for evaluation
        random_seed: Random seed for sampling configurations
        device: CUDA device to use (None = auto-detect, 'cpu' = CPU, 'cuda' = cuda:0, 'cuda:0' = first GPU, 'cuda:1' = second GPU, etc.). Can also be an integer (0, 1, etc.) for device ID. If devices is specified, this is ignored.
        devices: List of CUDA devices to use for distributing workers (e.g., ['cuda:1', 'cuda:2'] or [1, 2]). Workers will be distributed across these devices in round-robin fashion. If None, uses device parameter instead.
    """
    # Load curriculum config (constant)
    curriculum_config = load_base_config(curriculum_config_path)
    print(f"Loaded curriculum config from: {curriculum_config_path}")
    
    # Load hyperparameter config (contains grid search ranges)
    hyperparameter_config = load_base_config(hyperparameter_config_path)
    print(f"Loaded hyperparameter config from: {hyperparameter_config_path}")
    
    # Generate full hyperparameter grid
    all_configs = create_hyperparameter_grid(curriculum_config, hyperparameter_config)
    print(f"Total possible configurations: {len(all_configs)}")
    
    # Sample N configurations
    configs = sample_configurations(all_configs, n_samples, random_seed=random_seed)
    print(f"Sampling {len(configs)} configurations to test")
    
    # Create a single timestamped directory for all hyperparameter tuning runs
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    main_output_dir = Path(output_dir) / current_datetime
    main_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create run manager for generating run names and saving results
    run_manager = RunManager(base_output_dir=str(main_output_dir))
    
    # Generate run names for all configs (use index to ensure uniqueness)
    run_names = [run_manager.generate_run_name(config, index=i) for i, config in enumerate(configs)]
    
    # Determine device assignment for each worker
    # If devices list is provided, use it; otherwise use single device
    if devices is not None:
        # Validate devices
        if not isinstance(devices, list) or len(devices) == 0:
            raise ValueError("devices must be a non-empty list")
        # Assign devices in round-robin fashion
        device_assignments = [devices[i % len(devices)] for i in range(len(configs))]
    elif device is not None:
        # All workers use the same device
        device_assignments = [device] * len(configs)
    else:
        # Auto-detect: all workers use None (auto-detect)
        device_assignments = [None] * len(configs)
    
    # Prepare arguments for multiprocessing
    # Each run will create a subdirectory: main_output_dir / run_name / timestamp / ...
    args_list = [(config, str(main_output_dir), run_name, dev) for config, run_name, dev in zip(configs, run_names, device_assignments)]
    
    # Determine number of parallel workers
    if num_parallel is None:
        # Limit to 4 for GPU memory, or use CPU count if no GPU
        if torch.cuda.is_available():
            num_parallel = min(4, len(configs), mp.cpu_count())
        else:
            num_parallel = min(mp.cpu_count(), len(configs))
    
    print(f"Running {len(configs)} configurations with {num_parallel} parallel workers")
    if devices is not None:
        print(f"Using devices: {devices} (distributed across {len(devices)} device(s))")
        # Show device distribution
        device_counts = {}
        for dev in device_assignments:
            device_counts[dev] = device_counts.get(dev, 0) + 1
        print(f"Device distribution: {device_counts}")
    elif device is not None:
        print(f"Using device: {device}")
    elif torch.cuda.is_available():
        print(f"Using device: cuda (auto-detected, default: cuda:0)")
    else:
        print(f"Using device: cpu")
    print(f"Results will be saved to: {main_output_dir}")
    
    # Train all configurations
    training_results = []
    if num_parallel > 1:
        with mp.Pool(processes=num_parallel) as pool:
            for i, result in enumerate(pool.imap_unordered(run_single_config, args_list), 1):
                training_results.append(result)
                print(f"[{i}/{len(configs)}] Training '{result['run_name']}' completed. "
                      f"Mean reward: {result.get('mean_reward', 0):.2f}")
    else:
        for i, args in enumerate(args_list, 1):
            result = run_single_config(args)
            training_results.append(result)
            print(f"[{i}/{len(configs)}] Training '{result['run_name']}' completed. "
                  f"Mean reward: {result.get('mean_reward', 0):.2f}")
    
    # Match training results with configs
    result_dict = {r['run_name']: r for r in training_results}
    
    # Evaluate each trained agent
    print(f"\nEvaluating {len(training_results)} agents against BasicOpponent...")
    evaluation_results = []
    
    for i, (config, run_name) in enumerate(zip(configs, run_names), 1):
        if run_name not in result_dict:
            print(f"Warning: No training result found for {run_name}, skipping evaluation")
            continue
        
        # Get model path from training result (set in run_single_config)
        training_result = result_dict[run_name]
        model_path_str = training_result.get('model_path')
        
        if model_path_str is None:
            # Fallback: try to find model in the run-specific directory
            run_dir = main_output_dir / run_name
            if run_dir.exists():
                # Search in timestamped subdirectories
                timestamped_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
                for subdir in sorted(timestamped_dirs, key=lambda x: x.stat().st_mtime, reverse=True):
                    models_dir = subdir / "models"
                    if models_dir.exists():
                        model_files = list(models_dir.glob(f"{run_name}*.pt"))
                        if model_files:
                            model_path_str = str(sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0])
                            break
            
            if model_path_str is None:
                print(f"Warning: Model file not found for {run_name}, skipping evaluation")
                continue
        
        model_path = Path(model_path_str)
        if not model_path.exists():
            print(f"Warning: Model file not found at {model_path}, skipping evaluation")
            continue
        
        print(f"[{i}/{len(configs)}] Evaluating '{run_name}'...")
        
        # Create agent config dict
        agent_config_dict = {
            'type': config['agent']['type'],
            'hyperparameters': config['agent']['hyperparameters']
        }
        
        try:
            eval_result = evaluate_agent(
                agent_path=str(model_path),
                agent_config_dict=agent_config_dict,
                num_games=eval_num_games,
                weak_opponent=eval_weak_opponent,
                num_parallel=None  # Use auto-detection
            )
            
            evaluation_results.append({
                'run_name': run_name,
                'config': config,
                'training': result_dict[run_name],
                'evaluation': eval_result
            })
            
            print(f"  Win rate: {eval_result['win_rate']:.2%}, "
                  f"Mean reward: {eval_result['mean_reward']:.2f} "
                  f"({eval_result['wins']}W/{eval_result['losses']}L/{eval_result['draws']}D)")
        except Exception as e:
            print(f"  Error during evaluation: {e}")
            continue
    
    # Sort by win rate
    evaluation_results.sort(key=lambda x: x['evaluation'].get('win_rate', 0), reverse=True)
    
    # Print top results
    print(f"\n{'='*60}")
    print(f"Top 5 configurations (by win rate):")
    print(f"{'='*60}")
    for i, result in enumerate(evaluation_results[:5], 1):
        eval_res = result['evaluation']
        print(f"{i}. {result['run_name']}")
        print(f"   Win rate: {eval_res['win_rate']:.2%} "
              f"({eval_res['wins']}W/{eval_res['losses']}L/{eval_res['draws']}D)")
        print(f"   Mean reward: {eval_res['mean_reward']:.2f} ± {eval_res['std_reward']:.2f}")
    
    # Save results to JSON
    results_file = main_output_dir / "evaluation_results.json"
    results_to_save = []
    for result in evaluation_results:
        results_to_save.append({
            'run_name': result['run_name'],
            'config': result['config'],
            'training': result['training'],
            'evaluation': {
                'win_rate': result['evaluation']['win_rate'],
                'wins': result['evaluation']['wins'],
                'losses': result['evaluation']['losses'],
                'draws': result['evaluation']['draws'],
                'mean_reward': result['evaluation']['mean_reward'],
                'std_reward': result['evaluation']['std_reward'],
                'num_games': result['evaluation']['num_games']
            }
        })
    
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    # Create plots
    print(f"\nGenerating hyperparameter analysis plots...")
    plot_hyperparameter_analysis(evaluation_results, main_output_dir)
    
    print(f"\nAll results saved to: {main_output_dir}")


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # -----------------------------
    # Set your variables here
    # -----------------------------
    # Example settings - change as needed

    curriculum_config_path = "configs/curriculum_base.json"  # Path to curriculum configuration JSON file (constant)
    hyperparameter_config_path = "configs/hyperparameter_base.json"  # Path to hyperparameter configuration JSON file (grid search)
    n_samples = 100  # Number of configurations to sample from the grid
    num_parallel = None  # Number of parallel workers (None = auto-detect, max 4 for GPU)
    output_dir = "results/hyperparameter_runs"  # Base output directory for results
    eval_num_games = 200  # Number of games to run for evaluation
    use_weak_opponent = True  # Use weak BasicOpponent for evaluation (set to False to use strong)
    random_seed = None  # Random seed for sampling configurations
    device = None  # CUDA device to use (None = auto-detect, 'cpu' = CPU, 'cuda' = cuda:0, 'cuda:0' = first GPU, 'cuda:1' = second GPU, etc.)
    devices = ["cuda:0", "cuda:1", "cuda:2"]  # List of devices to distribute workers across (e.g., ["cuda:1", "cuda:2"]). If set, device parameter is ignored.

    # If you want to use strong opponent set this to False
    eval_weak_opponent = use_weak_opponent

    # Call main with manual variables
    main(
        curriculum_config_path=curriculum_config_path,
        hyperparameter_config_path=hyperparameter_config_path,
        n_samples=n_samples,
        num_parallel=num_parallel,
        output_dir=output_dir,
        eval_num_games=eval_num_games,
        eval_weak_opponent=eval_weak_opponent,
        random_seed=random_seed,
        device=device,
        devices=devices
    )

    # nohup python src/rl_hockey/common/training/hyperparameter_tuning.py > hyperparameter_tuning.log 2>&1 &
