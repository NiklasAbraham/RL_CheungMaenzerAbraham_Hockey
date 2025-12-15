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
