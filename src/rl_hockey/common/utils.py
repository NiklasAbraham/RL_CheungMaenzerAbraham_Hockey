import numpy as np
import torch
import psutil
import time
from typing import Optional, Union, Dict, Any


def mirror_state(state):
    # Use view to avoid copy if possible, but we need a copy for safety
    # Optimized: only copy if not already float32 or if we need to modify
    if state.dtype != np.float32:
        mirrored_state = state.astype(np.float32, copy=True)
    else:
        mirrored_state = state.copy()

    # mirrored dimensions: y positions, y velocities, angles, angular velocities
    indices = [1, 2, 4, 5, 7, 8, 10, 11, 13, 15]
    for idx in indices:
        mirrored_state[idx] *= -1

    return mirrored_state


def mirror_action(action):
    # Optimized: ensure float32 and avoid unnecessary copies
    if action.dtype != np.float32:
        mirrored_action = action.astype(np.float32, copy=True)
    else:
        mirrored_action = action.copy()
    
    # Mirror y force and torque only
    mirrored_action[1] *= -1  # y force
    mirrored_action[2] *= -1  # torque
    
    return mirrored_action


def discrete_to_continuous_action_standard(discrete_action, keep_mode=True):
    """
    Converts standard discrete actions (7 or 8 actions) to continuous actions.
    This matches the environment's discrete_to_continous_action method.
    
    Action mapping:
    - 0: do nothing
    - 1: -1 in x
    - 2: 1 in x
    - 3: -1 in y
    - 4: 1 in y
    - 5: -1 in angle
    - 6: 1 in angle
    - 7: shoot (if keep_mode)
    """
    if keep_mode:
        action = np.zeros(4, dtype=np.float32)
    else:
        action = np.zeros(3, dtype=np.float32)
    
    if discrete_action == 0:
        return action
    elif discrete_action == 1:
        action[0] = -1.0
    elif discrete_action == 2:
        action[0] = 1.0
    elif discrete_action == 3:
        action[1] = -1.0
    elif discrete_action == 4:
        action[1] = 1.0
    elif discrete_action == 5:
        action[2] = -1.0
    elif discrete_action == 6:
        action[2] = 1.0
    elif discrete_action == 7 and keep_mode:
        action[3] = 1.0
    else:
        raise ValueError(f"Invalid discrete_action {discrete_action} for keep_mode={keep_mode}")
    
    return action


def discrete_to_continuous_action_with_fineness(discrete_action, fineness=3, keep_mode=True):
    """
    Converts a discrete action index into a continuous action vector based on the specified fineness. Supports movement and shooting, with action intensities determined by the fineness parameter.
    """
    if fineness < 3:
        raise ValueError(f"fineness must be >= 3, got {fineness}")
    
    action = np.zeros(4, dtype=np.float32)
    
    actions_per_movement = fineness - 1
    num_movement_actions = 6 * actions_per_movement
    
    if discrete_action == 0:
        return action
    
    if keep_mode and discrete_action == 1 + num_movement_actions:
        action[3] = 1.0
        return action
    
    if discrete_action < 0 or discrete_action > 1 + num_movement_actions + (1 if keep_mode else 0):
        raise ValueError(f"Invalid discrete_action {discrete_action} for fineness={fineness}, keep_mode={keep_mode}")
    
    action_idx = discrete_action - 1
    movement_type = action_idx // actions_per_movement
    intensity_level = (action_idx % actions_per_movement) + 1
    intensity = intensity_level / actions_per_movement
    
    if movement_type == 0:
        action[0] = -intensity
    elif movement_type == 1:
        action[0] = intensity
    elif movement_type == 2:
        action[1] = -intensity
    elif movement_type == 3:
        action[1] = intensity
    elif movement_type == 4:
        action[2] = -intensity
    elif movement_type == 5:
        action[2] = intensity
    
    return action


def get_discrete_action_dim(fineness=3, keep_mode=True):
    if fineness < 3:
        raise ValueError(f"fineness must be >= 3, got {fineness}")
    
    actions_per_movement = fineness - 1
    num_movement_actions = 6 * actions_per_movement
    total_actions = 1 + num_movement_actions + (1 if keep_mode else 0)
    return total_actions


def mirror_discrete_action(discrete_action, fineness=None, keep_mode=True):

    if fineness is None:
        mirror_map = {
            0: 0,  # nothing -> nothing
            1: 1,  # left -> left
            2: 2,  # right -> right
            3: 4,  # down -> up
            4: 3,  # up -> down
            5: 6,  # CCW -> CW
            6: 5,  # CW -> CCW
            7: 7,  # shoot -> shoot
        }
        return mirror_map.get(discrete_action, discrete_action)
    
    if fineness < 3:
        raise ValueError(f"fineness must be >= 3, got {fineness}")
    
    actions_per_movement = fineness - 1
    num_movement_actions = 6 * actions_per_movement
    
    if discrete_action == 0:
        return 0
    
    if keep_mode and discrete_action == 1 + num_movement_actions:
        return discrete_action
    
    if discrete_action < 0 or discrete_action > 1 + num_movement_actions + (1 if keep_mode else 0):
        raise ValueError(f"Invalid discrete_action {discrete_action} for fineness={fineness}, keep_mode={keep_mode}")
    
    action_idx = discrete_action - 1
    movement_type = action_idx // actions_per_movement
    intensity_level = action_idx % actions_per_movement
    
    mirror_map = {
        0: 1,  # left -> right
        1: 0,  # right -> left
        2: 3,  # down -> up
        3: 2,  # up -> down
        4: 5,  # CCW -> CW
        5: 4,  # CW -> CCW
    }
    
    mirrored_movement_type = mirror_map[movement_type]
    mirrored_action_idx = mirrored_movement_type * actions_per_movement + intensity_level
    return 1 + mirrored_action_idx


def set_cuda_device(device: Optional[Union[str, int]]):
    """Set CUDA device if specified and available."""
    if device is not None:
        if isinstance(device, str):
            if device == 'cpu':
                pass
            elif device == 'cuda':
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
            if torch.cuda.is_available():
                if device >= torch.cuda.device_count():
                    raise ValueError(f"CUDA device {device} not available. Only {torch.cuda.device_count()} device(s) available.")
                torch.cuda.set_device(device)
                torch.cuda.empty_cache()
            else:
                raise ValueError(f"CUDA not available, but device {device} was requested")
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_resource_usage() -> Dict[str, Any]:
    """
    Collect GPU and CPU usage metrics.
    
    Returns:
        Dictionary with resource usage metrics including:
        - gpu_utilization: GPU utilization percentage (0-100)
        - gpu_memory_used: GPU memory used in MB
        - gpu_memory_total: Total GPU memory in MB
        - gpu_memory_percent: GPU memory usage percentage
        - gpu_temperature: GPU temperature in Celsius
        - cpu_percent: CPU usage percentage (average across all cores)
        - cpu_per_core: CPU usage per core (list of percentages)
        - memory_used: System memory used in MB
        - memory_total: Total system memory in MB
        - memory_percent: System memory usage percentage
        - load_avg: System load average (1, 5, 15 min)
    """
    metrics = {
        'timestamp': time.time(),
    }
    
    # CPU metrics
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
        
        metrics['cpu_percent'] = cpu_percent
        metrics['cpu_per_core'] = cpu_per_core
        metrics['cpu_cores'] = len(cpu_per_core)
        metrics['load_avg_1min'] = load_avg[0] if len(load_avg) > 0 else 0
        metrics['load_avg_5min'] = load_avg[1] if len(load_avg) > 1 else 0
        metrics['load_avg_15min'] = load_avg[2] if len(load_avg) > 2 else 0
    except Exception as e:
        metrics['cpu_percent'] = 0
        metrics['cpu_per_core'] = []
        metrics['cpu_cores'] = 0
        metrics['load_avg_1min'] = 0
        metrics['load_avg_5min'] = 0
        metrics['load_avg_15min'] = 0
    
    # Memory metrics
    try:
        mem = psutil.virtual_memory()
        metrics['memory_used'] = mem.used / (1024 * 1024)  # MB
        metrics['memory_total'] = mem.total / (1024 * 1024)  # MB
        metrics['memory_percent'] = mem.percent
        metrics['memory_available'] = mem.available / (1024 * 1024)  # MB
    except Exception as e:
        metrics['memory_used'] = 0
        metrics['memory_total'] = 0
        metrics['memory_percent'] = 0
        metrics['memory_available'] = 0
    
    # GPU metrics
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            
            # GPU utilization and memory
            memory_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)  # MB
            memory_reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)  # MB
            memory_total = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)  # MB
            
            # Try to get utilization via nvidia-smi (if available)
            gpu_util = None
            gpu_temp = None
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(', ')
                    if len(parts) >= 2:
                        gpu_util = float(parts[0].strip())
                        gpu_temp = float(parts[1].strip())
            except Exception:
                pass
            
            metrics['gpu_available'] = True
            metrics['gpu_device'] = device
            metrics['gpu_memory_allocated'] = memory_allocated
            metrics['gpu_memory_reserved'] = memory_reserved
            metrics['gpu_memory_total'] = memory_total
            metrics['gpu_memory_percent'] = (memory_reserved / memory_total) * 100 if memory_total > 0 else 0
            metrics['gpu_utilization'] = gpu_util if gpu_util is not None else 0
            metrics['gpu_temperature'] = gpu_temp if gpu_temp is not None else 0
        else:
            metrics['gpu_available'] = False
            metrics['gpu_device'] = None
            metrics['gpu_memory_allocated'] = 0
            metrics['gpu_memory_reserved'] = 0
            metrics['gpu_memory_total'] = 0
            metrics['gpu_memory_percent'] = 0
            metrics['gpu_utilization'] = 0
            metrics['gpu_temperature'] = 0
    except Exception as e:
        metrics['gpu_available'] = False
        metrics['gpu_device'] = None
        metrics['gpu_memory_allocated'] = 0
        metrics['gpu_memory_reserved'] = 0
        metrics['gpu_memory_total'] = 0
        metrics['gpu_memory_percent'] = 0
        metrics['gpu_utilization'] = 0
        metrics['gpu_temperature'] = 0
    
    return metrics

