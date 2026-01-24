import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import hashlib
import numpy as np


class RunManager:
    """Manages file organization for hyperparameter runs."""
    
    def __init__(self, base_output_dir: str = "results/hyperparameter_runs"):
        """Initialize the run manager."""
        # Add current date and time to the output directory
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_output_dir = Path(base_output_dir) / current_datetime
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.configs_dir = self.base_output_dir / "configs"
        self.plots_dir = self.base_output_dir / "plots"
        self.csvs_dir = self.base_output_dir / "csvs"
        self.models_dir = self.base_output_dir / "models"
        
        for dir_path in [self.configs_dir, self.plots_dir, self.csvs_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def generate_run_name(self, config: Dict[str, Any], index: int = None) -> str:
        """Generate a unique run name based on configuration."""
        # Extract agent type and hyperparameters
        agent_type = config.get('agent', {}).get('type', 'unknown')
        agent_hyperparams = config.get('agent', {}).get('hyperparameters', {})
        common_hyperparams = config.get('hyperparameters', {})
        
        # Create a hash from key hyperparameters for uniqueness
        key_params = {
            'hidden_dim': agent_hyperparams.get('hidden_dim', [128, 128, 128]),
            'learning_rate': common_hyperparams.get('learning_rate', 0.0001),
            'batch_size': common_hyperparams.get('batch_size', 256),
            'target_update_freq': agent_hyperparams.get('target_update_freq', 50),
            'eps_decay': agent_hyperparams.get('eps_decay', 0.999),
        }
        
        # Create a short hash
        config_str = json.dumps(key_params, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Create descriptive name
        lr_str = f"lr{key_params['learning_rate']:.0e}".replace('-', '')
        batch_str = f"bs{key_params['batch_size']}"
        hidden_str = f"h{'_'.join(map(str, key_params['hidden_dim']))}"
        
        # Use index if provided, otherwise use timestamp
        if index is not None:
            run_name = f"{agent_type}_run_{lr_str}_{batch_str}_{hidden_str}_{config_hash}_{index:04d}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{agent_type}_run_{lr_str}_{batch_str}_{hidden_str}_{config_hash}_{timestamp}"
        
        return run_name
    
    def get_run_directories(self, run_name: str) -> Dict[str, Path]:
        """Get all directory paths for a specific run."""
        return {
            'config': self.configs_dir / f"{run_name}.json",
            'plot_rewards': self.plots_dir / f"{run_name}_rewards.png",
            'plot_losses': self.plots_dir / f"{run_name}_losses.png",
            'plot_evaluation': self.plots_dir / f"{run_name}_evaluation.png",
            'csv_rewards': self.csvs_dir / f"{run_name}_rewards.csv",
            'csv_losses': self.csvs_dir / f"{run_name}_losses.csv",
            'csv_evaluation': self.csvs_dir / f"{run_name}_evaluation.csv",
            'csv_resources': self.csvs_dir / f"{run_name}_resources.csv",
            'csv_episode_logs': self.csvs_dir / f"{run_name}_episode_logs.csv",
            'model': self.models_dir / f"{run_name}.pt",
        }
    
    def save_config(self, run_name: str, config: Dict[str, Any]):
        """Save configuration to JSON file."""
        paths = self.get_run_directories(run_name)
        with open(paths['config'], 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def save_rewards_csv(self, run_name: str, rewards: List[float], phases: List[str] = None):
        """Save rewards to CSV file."""
        paths = self.get_run_directories(run_name)
        with open(paths['csv_rewards'], 'w', newline='') as f:
            writer = csv.writer(f)
            if phases:
                writer.writerow(['episode', 'reward', 'phase'])
                for i, reward in enumerate(rewards):
                    phase = phases[i] if i < len(phases) else phases[-1]
                    writer.writerow([i, reward, phase])
            else:
                writer.writerow(['episode', 'reward'])
                for i, reward in enumerate(rewards):
                    writer.writerow([i, reward])
    
    def save_losses_csv(self, run_name: str, losses: List[float]):
        """Save losses to CSV file."""
        paths = self.get_run_directories(run_name)
        with open(paths['csv_losses'], 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'loss'])
            for i, loss in enumerate(losses):
                writer.writerow([i, loss])
    
    def save_plots(self, run_name: str, rewards: List[float], losses: List[float], 
                   reward_window: int = 10, loss_window: int = 100):
        """Save reward and loss plots."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        paths = self.get_run_directories(run_name)
        
        # Plot rewards
        if rewards:
            moving_avg_rewards = self._moving_average(rewards, reward_window)
            plt.figure(figsize=(10, 6))
            plt.plot(rewards, alpha=0.3, label='Raw')
            plt.plot(moving_avg_rewards, label=f'Moving Avg (window={reward_window})')
            plt.xlabel('Episodes')
            plt.ylabel('Total Reward')
            plt.title(f'Reward per Episode - {run_name}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(paths['plot_rewards'])
            plt.close()
        
        # Plot losses
        if losses:
            moving_avg_losses = self._moving_average(losses, loss_window)
            plt.figure(figsize=(10, 6))
            plt.plot(losses, alpha=0.3, label='Raw')
            plt.plot(moving_avg_losses, label=f'Moving Avg (window={loss_window})')
            plt.xlabel('Training Steps')
            plt.ylabel('Q-Loss')
            plt.title(f'Q-Loss over Time - {run_name}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(paths['plot_losses'])
            plt.close()
        else:
            print(f"Warning: No losses collected for {run_name}. Loss plot will not be generated.")
            print(f"  This usually means training hasn't started yet (warmup_steps not reached) or agent.train() returned no losses.")
    
    def get_model_path(self, run_name: str) -> Path:
        """Get model save path for a run."""
        return self.get_run_directories(run_name)['model']
    
    def save_checkpoint(self, run_name: str, episode: int, agent: Any, phase_index: int = None, phase_episode: int = None, episode_logs: List[Dict[str, Any]] = None):
        """Save a checkpoint for the current training state."""
        checkpoint_name = f"{run_name}_ep{episode:06d}"
        checkpoint_path = self.models_dir / f"{checkpoint_name}.pt"
        agent.save(str(checkpoint_path))
        
        # Also save checkpoint metadata for resumption
        metadata_path = self.models_dir / f"{checkpoint_name}_metadata.json"
        metadata = {
            'episode': episode,
            'phase_index': phase_index,
            'phase_episode': phase_episode,
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save episode logs CSV with checkpoint
        if episode_logs:
            checkpoint_csv_path = self.csvs_dir / f"{checkpoint_name}_episode_logs.csv"
            self.save_episode_logs_csv(checkpoint_name, episode_logs, checkpoint_csv_path)
    
    def get_checkpoint_path(self, run_name: str, episode: int) -> Path:
        """Get checkpoint path for a specific episode."""
        checkpoint_name = f"{run_name}_ep{episode:06d}"
        return self.models_dir / f"{checkpoint_name}.pt"
    
    def save_evaluation_csv(self, run_name: str, evaluation_results: List[Dict[str, Any]]):
        """Save evaluation results to CSV file."""
        paths = self.get_run_directories(run_name)
        with open(paths['csv_evaluation'], 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'episode', 'win_rate', 'mean_reward', 'std_reward', 'wins', 'losses', 'draws'])
            for result in evaluation_results:
                writer.writerow([
                    result['step'],
                    result['episode'],
                    result['win_rate'],
                    result['mean_reward'],
                    result['std_reward'],
                    result['wins'],
                    result['losses'],
                    result['draws']
                ])
    
    def save_evaluation_plot(self, run_name: str, evaluation_results: List[Dict[str, Any]]):
        """Save evaluation metrics plot."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        paths = self.get_run_directories(run_name)
        
        if not evaluation_results:
            return
        
        steps = [r['step'] for r in evaluation_results]
        win_rates = [r['win_rate'] for r in evaluation_results]
        mean_rewards = [r['mean_reward'] for r in evaluation_results]
        std_rewards = [r['std_reward'] for r in evaluation_results]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot win rate
        ax1.plot(steps, win_rates, 'o-', label='Win Rate', linewidth=2, markersize=6)
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% (Random)')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Win Rate')
        ax1.set_title(f'Win Rate vs Base Opponent - {run_name}')
        ax1.set_ylim([0, 1])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot mean reward with error bars
        ax2.errorbar(steps, mean_rewards, yerr=std_rewards, fmt='o-', 
                    label='Mean Reward ± Std', linewidth=2, markersize=6, capsize=4)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Mean Reward')
        ax2.set_title(f'Mean Reward vs Base Opponent - {run_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(paths['plot_evaluation'])
        plt.close()

    def save_value_propagation_plot(self, run_name: str, evaluation_results: List[Dict[str, Any]]):
        """Save value propagation plot."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt

        q_values = [result['q_values'] for result in evaluation_results if 'q_values' in result]
        if not q_values:
            return
        
        q_values = np.array(q_values)
        q_values = np.flipud(q_values.T)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(q_values, aspect='auto', cmap='plasma', origin='upper')
        plt.colorbar(label='Average Q-Value')
        plt.title(f"Value Propagation over Episodes - {run_name}")
        plt.xlabel("Episodes")
        plt.ylabel("Steps from goal state")
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.savefig(self.plots_dir / f"{run_name}_value_propagation.png")
        plt.close()
    
    def save_resources_csv(self, run_name: str, resource_logs: List[Dict[str, Any]]):
        """Save resource usage logs to CSV file."""
        if not resource_logs:
            return
        
        paths = self.get_run_directories(run_name)
        
        # Get all unique keys from all resource logs
        all_keys = set()
        for log in resource_logs:
            all_keys.update(log.keys())
        
        # Sort keys for consistent column order
        key_order = [
            'step', 'episode', 'timestamp',
            'cpu_percent', 'cpu_cores', 'load_avg_1min', 'load_avg_5min', 'load_avg_15min',
            'memory_used', 'memory_total', 'memory_percent', 'memory_available',
            'gpu_available', 'gpu_device', 'gpu_utilization', 'gpu_memory_allocated',
            'gpu_memory_reserved', 'gpu_memory_total', 'gpu_memory_percent', 'gpu_temperature'
        ]
        
        # Order keys: known keys first, then any others
        ordered_keys = [k for k in key_order if k in all_keys]
        remaining_keys = sorted([k for k in all_keys if k not in key_order])
        csv_keys = ordered_keys + remaining_keys
        
        with open(paths['csv_resources'], 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_keys)
            
            for log in resource_logs:
                row = [log.get(key, '') for key in csv_keys]
                writer.writerow(row)
    
    def save_episode_logs_csv(self, run_name: str, episode_logs: List[Dict[str, Any]], custom_path: Path = None):
        """Save episode logs with all loss types to CSV file."""
        if not episode_logs:
            return
        
        # Use custom path if provided (for checkpoints), otherwise use default
        if custom_path is not None:
            csv_path = custom_path
        else:
            paths = self.get_run_directories(run_name)
            csv_path = paths['csv_episode_logs']
        
        # Collect all possible loss keys from all episodes
        all_loss_keys = set()
        for log in episode_logs:
            if 'losses' in log and isinstance(log['losses'], dict):
                all_loss_keys.update(log['losses'].keys())
        
        # Sort loss keys for consistent column order
        sorted_loss_keys = sorted(all_loss_keys)
        
        # Define column order: episode, reward, shaped_reward first, then all loss types
        csv_keys = ['episode', 'reward', 'shaped_reward'] + sorted_loss_keys
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_keys)
            
            for log in episode_logs:
                row = [
                    log.get('episode', ''),
                    log.get('reward', ''),
                    log.get('shaped_reward', ''),
                ]
                # Add loss values in sorted order
                losses_dict = log.get('losses', {})
                for loss_key in sorted_loss_keys:
                    loss_value = losses_dict.get(loss_key, '')
                    # If it's a list, compute average; otherwise use value directly
                    if isinstance(loss_value, list):
                        loss_value = sum(loss_value) / len(loss_value) if loss_value else ''
                    row.append(loss_value)
                writer.writerow(row)
    
    @staticmethod
    def _moving_average(data: List[float], window_size: int) -> List[float]:
        """Calculate moving average of data."""
        moving_averages = []
        for i in range(len(data)):
            window_start = max(0, i - window_size + 1)
            window = data[window_start:i + 1]
            moving_averages.append(sum(window) / len(window))
        return moving_averages
