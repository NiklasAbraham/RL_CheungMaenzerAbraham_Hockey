import json
from pathlib import Path
from typing import Dict, Any, List
import os


VALID_ENV_MODES = ["NORMAL", "TRAIN_SHOOTING", "TRAIN_DEFENSE"]
VALID_AGENT_TYPES = ["DDDQN", "SAC", "TD3", "TDMPC2"]
VALID_OPPONENT_TYPES = ["none", "basic_weak", "basic_strong", "self_play", "weighted_mixture"]


def validate_config(config_path: str) -> List[str]:
    """
    Validate a curriculum configuration file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {str(e)}"]
    except FileNotFoundError:
        return [f"Config file not found: {config_path}"]
    
    # Validate required top-level fields
    required_fields = ['curriculum', 'agent', 'training']
    for field in required_fields:
        if field not in config_dict:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return errors
    
    # Validate curriculum
    errors.extend(_validate_curriculum(config_dict.get('curriculum', {})))
    
    # Validate agent
    errors.extend(_validate_agent(config_dict.get('agent', {})))
    
    # Validate training parameters
    errors.extend(_validate_training(config_dict.get('training', {})))
    
    # Validate hyperparameters
    errors.extend(_validate_hyperparameters(config_dict.get('hyperparameters', {})))
    
    return errors


def _validate_curriculum(curriculum: Dict[str, Any]) -> List[str]:
    """Validate curriculum configuration."""
    errors = []
    
    if 'phases' not in curriculum:
        errors.append("curriculum.phases is required")
        return errors
    
    phases = curriculum.get('phases', [])
    if not isinstance(phases, list) or len(phases) == 0:
        errors.append("curriculum.phases must be a non-empty list")
        return errors
    
    for i, phase in enumerate(phases):
        phase_errors = _validate_phase(phase, i)
        errors.extend(phase_errors)
    
    return errors


def _validate_phase(phase: Dict[str, Any], phase_idx: int) -> List[str]:
    """Validate a single phase configuration."""
    errors = []
    prefix = f"curriculum.phases[{phase_idx}]"
    
    # Required fields (reward_shaping is optional)
    required = ['name', 'episodes', 'environment', 'opponent']
    for field in required:
        if field not in phase:
            errors.append(f"{prefix}.{field} is required")
    
    if errors:
        return errors
    
    # Validate episodes
    episodes = phase.get('episodes')
    if not isinstance(episodes, int) or episodes <= 0:
        errors.append(f"{prefix}.episodes must be a positive integer")
    
    # Validate environment
    env = phase.get('environment', {})
    
    # Check if using mixture or single mode
    if 'mixture' in env:
        # Validate mixture
        mixture = env['mixture']
        if not isinstance(mixture, list) or len(mixture) == 0:
            errors.append(f"{prefix}.environment.mixture must be a non-empty list")
        else:
            total_weight = 0.0
            for i, mix_item in enumerate(mixture):
                if not isinstance(mix_item, dict):
                    errors.append(f"{prefix}.environment.mixture[{i}] must be a dictionary")
                else:
                    if 'mode' not in mix_item:
                        errors.append(f"{prefix}.environment.mixture[{i}].mode is required")
                    elif mix_item['mode'] not in VALID_ENV_MODES:
                        errors.append(f"{prefix}.environment.mixture[{i}].mode must be one of {VALID_ENV_MODES}")
                    
                    weight = mix_item.get('weight', 1.0)
                    if not isinstance(weight, (int, float)) or weight < 0:
                        errors.append(f"{prefix}.environment.mixture[{i}].weight must be a non-negative number")
                    else:
                        total_weight += weight
            
            # Weights don't need to sum to 1.0 (will be normalized), but should be positive
            if total_weight == 0:
                errors.append(f"{prefix}.environment.mixture weights sum to zero (at least one weight must be positive)")
    elif 'mode' in env:
        # Single mode (backward compatible)
        if env['mode'] not in VALID_ENV_MODES:
            errors.append(f"{prefix}.environment.mode must be one of {VALID_ENV_MODES}")
    else:
        errors.append(f"{prefix}.environment must have either 'mode' or 'mixture'")
    
    if 'keep_mode' in env and not isinstance(env['keep_mode'], bool):
        errors.append(f"{prefix}.environment.keep_mode must be a boolean")
    
    # Validate opponent
    opponent = phase.get('opponent', {})
    opponent_errors = _validate_opponent(opponent, f"{prefix}.opponent")
    errors.extend(opponent_errors)
    
    # Validate reward shaping (optional - can be None)
    reward_shaping = phase.get('reward_shaping')
    if reward_shaping is not None:
        reward_errors = _validate_reward_shaping(reward_shaping, f"{prefix}.reward_shaping")
        errors.extend(reward_errors)
    
    return errors


def _validate_opponent(opponent: Dict[str, Any], prefix: str) -> List[str]:
    """Validate opponent configuration."""
    errors = []
    
    if 'type' not in opponent:
        errors.append(f"{prefix}.type is required")
        return errors
    
    opp_type = opponent['type']
    if opp_type not in VALID_OPPONENT_TYPES:
        errors.append(f"{prefix}.type must be one of {VALID_OPPONENT_TYPES}")
        return errors
    
    # Validate self-play specific fields
    if opp_type == "self_play":
        if 'checkpoint' in opponent:
            checkpoint = opponent['checkpoint']
            if checkpoint is not None and checkpoint != "latest":
                # Check if path exists (if it's a string path)
                if isinstance(checkpoint, str) and not checkpoint.startswith("results/"):
                    checkpoint_path = Path(checkpoint)
                    if not checkpoint_path.exists():
                        errors.append(f"{prefix}.checkpoint path does not exist: {checkpoint}")
        
        if 'deterministic' in opponent and not isinstance(opponent['deterministic'], bool):
            errors.append(f"{prefix}.deterministic must be a boolean")
    
    # Validate weighted_mixture
    elif opp_type == "weighted_mixture":
        if 'opponents' not in opponent:
            errors.append(f"{prefix}.opponents is required for weighted_mixture")
        else:
            opponents = opponent['opponents']
            if not isinstance(opponents, list) or len(opponents) == 0:
                errors.append(f"{prefix}.opponents must be a non-empty list")
            else:
                total_weight = 0.0
                for i, opp in enumerate(opponents):
                    if 'type' not in opp:
                        errors.append(f"{prefix}.opponents[{i}].type is required")
                    elif opp['type'] not in VALID_OPPONENT_TYPES:
                        errors.append(f"{prefix}.opponents[{i}].type must be one of {VALID_OPPONENT_TYPES}")
                    
                    weight = opp.get('weight', 1.0)
                    if not isinstance(weight, (int, float)) or weight < 0:
                        errors.append(f"{prefix}.opponents[{i}].weight must be a non-negative number")
                    else:
                        total_weight += weight
                
                # Check if weights sum to approximately 1.0 (allow some tolerance)
                if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
                    errors.append(f"{prefix}.opponents weights should sum to approximately 1.0 (current: {total_weight})")
    
    return errors


def _validate_reward_shaping(reward_shaping: Dict[str, Any], prefix: str) -> List[str]:
    """Validate reward shaping configuration."""
    errors = []
    
    # If None, skip validation (shouldn't happen, but be safe)
    if reward_shaping is None:
        return errors
    
    required_fields = ['N', 'K', 'CLOSENESS_START', 'TOUCH_START', 
                       'CLOSENESS_FINAL', 'TOUCH_FINAL', 'DIRECTION_FINAL']
    
    for field in required_fields:
        if field not in reward_shaping:
            errors.append(f"{prefix}.{field} is required")
        else:
            value = reward_shaping[field]
            if field in ['N', 'K']:
                if not isinstance(value, int) or value < 0:
                    errors.append(f"{prefix}.{field} must be a non-negative integer")
            else:
                if not isinstance(value, (int, float)):
                    errors.append(f"{prefix}.{field} must be a number")
    
    return errors


def _validate_agent(agent: Dict[str, Any]) -> List[str]:
    """Validate agent configuration."""
    errors = []
    
    if 'type' not in agent:
        errors.append("agent.type is required")
        return errors
    
    agent_type = agent['type']
    if agent_type not in VALID_AGENT_TYPES:
        errors.append(f"agent.type must be one of {VALID_AGENT_TYPES}")
        return errors
    
    # Validate algorithm-specific hyperparameters
    hyperparams = agent.get('hyperparameters', {})
    if agent_type == "DDDQN":
        # DDDQN specific validations
        if 'hidden_dim' in hyperparams:
            hidden_dim = hyperparams['hidden_dim']
            if not isinstance(hidden_dim, list) or len(hidden_dim) == 0:
                errors.append("agent.hyperparameters.hidden_dim must be a non-empty list")
            elif not all(isinstance(x, int) and x > 0 for x in hidden_dim):
                errors.append("agent.hyperparameters.hidden_dim must contain positive integers")
    
    elif agent_type == "SAC":
        # SAC specific validations
        if 'tau' in hyperparams and (not isinstance(hyperparams['tau'], (int, float)) or hyperparams['tau'] <= 0):
            errors.append("agent.hyperparameters.tau must be a positive number")
        
        if 'alpha' in hyperparams and (not isinstance(hyperparams['alpha'], (int, float)) or hyperparams['alpha'] < 0):
            errors.append("agent.hyperparameters.alpha must be a non-negative number")
    
    elif agent_type == "TD3":
        # TD3 specific validations (if implemented)
        if 'tau' in hyperparams and (not isinstance(hyperparams['tau'], (int, float)) or hyperparams['tau'] <= 0):
            errors.append("agent.hyperparameters.tau must be a positive number")
    
    elif agent_type == "TDMPC2":
        # TDMPC2 specific validations
        if 'latent_dim' in hyperparams:
            latent_dim = hyperparams['latent_dim']
            if not isinstance(latent_dim, int) or latent_dim <= 0:
                errors.append("agent.hyperparameters.latent_dim must be a positive integer")
        
        if 'hidden_dim' in hyperparams:
            hidden_dim = hyperparams['hidden_dim']
            if not isinstance(hidden_dim, dict):
                errors.append("agent.hyperparameters.hidden_dim must be a dict with network-specific hidden dimensions")
            else:
                # Dict format (per-network hidden dimensions)
                valid_network_types = ["encoder", "dynamics", "reward", "termination", "q_function", "policy"]
                for network_type, network_hidden_dim in hidden_dim.items():
                    if network_type not in valid_network_types:
                        errors.append(f"agent.hyperparameters.hidden_dim has unknown network type: {network_type}. Valid types: {valid_network_types}")
                    elif not isinstance(network_hidden_dim, list) or len(network_hidden_dim) == 0:
                        errors.append(f"agent.hyperparameters.hidden_dim.{network_type} must be a non-empty list")
                    elif not all(isinstance(x, int) and x > 0 for x in network_hidden_dim):
                        errors.append(f"agent.hyperparameters.hidden_dim.{network_type} must contain positive integers")
        
        if 'num_q' in hyperparams:
            num_q = hyperparams['num_q']
            if not isinstance(num_q, int) or num_q <= 0:
                errors.append("agent.hyperparameters.num_q must be a positive integer")
        
        if 'horizon' in hyperparams:
            horizon = hyperparams['horizon']
            if not isinstance(horizon, int) or horizon <= 0:
                errors.append("agent.hyperparameters.horizon must be a positive integer")
        
        if 'num_samples' in hyperparams:
            num_samples = hyperparams['num_samples']
            if not isinstance(num_samples, int) or num_samples <= 0:
                errors.append("agent.hyperparameters.num_samples must be a positive integer")
        
        if 'num_iterations' in hyperparams:
            num_iterations = hyperparams['num_iterations']
            if not isinstance(num_iterations, int) or num_iterations <= 0:
                errors.append("agent.hyperparameters.num_iterations must be a positive integer")
        
        if 'temperature' in hyperparams:
            temperature = hyperparams['temperature']
            if not isinstance(temperature, (int, float)) or temperature <= 0:
                errors.append("agent.hyperparameters.temperature must be a positive number")
        
        if 'gamma' in hyperparams:
            gamma = hyperparams['gamma']
            if not isinstance(gamma, (int, float)) or gamma <= 0 or gamma > 1:
                errors.append("agent.hyperparameters.gamma must be a number between 0 and 1")

        if 'n_step' in hyperparams:
            n_step = hyperparams['n_step']
            if not isinstance(n_step, int) or n_step <= 0:
                errors.append("agent.hyperparameters.n_step must be a positive integer")

        if 'simnorm_temperature' in hyperparams:
            simnorm_temperature = hyperparams['simnorm_temperature']
            if not isinstance(simnorm_temperature, (int, float)) or simnorm_temperature <= 0:
                errors.append("agent.hyperparameters.simnorm_temperature must be a positive number")

        if 'log_std_min' in hyperparams or 'log_std_max' in hyperparams:
            log_std_min = hyperparams.get('log_std_min', None)
            log_std_max = hyperparams.get('log_std_max', None)
            if log_std_min is not None and not isinstance(log_std_min, (int, float)):
                errors.append("agent.hyperparameters.log_std_min must be a number")
            if log_std_max is not None and not isinstance(log_std_max, (int, float)):
                errors.append("agent.hyperparameters.log_std_max must be a number")
            if log_std_min is not None and log_std_max is not None and log_std_min >= log_std_max:
                errors.append("agent.hyperparameters.log_std_min must be < log_std_max")

        if 'lambda_coef' in hyperparams:
            lambda_coef = hyperparams['lambda_coef']
            if not isinstance(lambda_coef, (int, float)) or lambda_coef <= 0 or lambda_coef > 1:
                errors.append("agent.hyperparameters.lambda_coef must be in (0, 1]")

        if 'policy_alpha' in hyperparams:
            policy_alpha = hyperparams['policy_alpha']
            if not isinstance(policy_alpha, (int, float)) or policy_alpha < 0:
                errors.append("agent.hyperparameters.policy_alpha must be non-negative")

        if 'policy_beta' in hyperparams:
            policy_beta = hyperparams['policy_beta']
            if not isinstance(policy_beta, (int, float)) or policy_beta < 0:
                errors.append("agent.hyperparameters.policy_beta must be non-negative")
    
    return errors


def _validate_training(training: Dict[str, Any]) -> List[str]:
    """Validate training parameters."""
    errors = []
    
    # Validate checkpoint_save_freq
    if 'checkpoint_save_freq' in training:
        freq = training['checkpoint_save_freq']
        if not isinstance(freq, int) or freq <= 0:
            errors.append("training.checkpoint_save_freq must be a positive integer")
    
    # Validate other training parameters (optional, but check types if present)
    numeric_fields = ['max_episode_steps', 'warmup_steps', 'updates_per_step']
    for field in numeric_fields:
        if field in training:
            value = training[field]
            if not isinstance(value, int) or value <= 0:
                errors.append(f"training.{field} must be a positive integer")
    
    if 'reward_scale' in training:
        value = training['reward_scale']
        if not isinstance(value, (int, float)) or value <= 0:
            errors.append("training.reward_scale must be a positive number")
    
    return errors


def _validate_hyperparameters(hyperparams: Dict[str, Any]) -> List[str]:
    """Validate hyperparameters."""
    errors = []
    
    # Validate learning_rate
    if 'learning_rate' in hyperparams:
        lr = hyperparams['learning_rate']
        if isinstance(lr, list):
            if not all(isinstance(x, (int, float)) and x > 0 for x in lr):
                errors.append("hyperparameters.learning_rate must contain positive numbers")
        elif not isinstance(lr, (int, float)) or lr <= 0:
            errors.append("hyperparameters.learning_rate must be a positive number")
    
    # Validate batch_size
    if 'batch_size' in hyperparams:
        bs = hyperparams['batch_size']
        if isinstance(bs, list):
            if not all(isinstance(x, int) and x > 0 for x in bs):
                errors.append("hyperparameters.batch_size must contain positive integers")
        elif not isinstance(bs, int) or bs <= 0:
            errors.append("hyperparameters.batch_size must be a positive integer")
    
    return errors
