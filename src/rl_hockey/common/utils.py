import numpy as np


def mirror_state(state):
    mirrored_state = state.copy()

    # mirrored dimensions: y positions, y velocities, angles, angular velocities
    indices = [1, 2, 4, 5, 7, 8, 10, 11, 13, 15]
    for idx in indices:
        mirrored_state[idx] *= -1

    return mirrored_state


def mirror_action(action):
    mirrored_action = action.copy()

    # mirrored dimensions: y forces, torques
    indices = [1, 2, 5, 6]
    for idx in indices:
        mirrored_action[idx] *= -1

    return mirrored_action


def mirror_discrete_action(discrete_action):
    """
    Discrete actions:
    0: Do nothing
    1: Move left (x-)
    2: Move right (x+)
    3: Move down (y-)
    4: Move up (y+)
    5: Rotate CCW
    6: Rotate CW
    7: Shoot
    """
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
    return mirror_map[discrete_action]

