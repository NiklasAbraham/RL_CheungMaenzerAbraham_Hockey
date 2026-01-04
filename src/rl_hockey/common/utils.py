import numpy as np
import torch
from typing import Optional, Union


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

