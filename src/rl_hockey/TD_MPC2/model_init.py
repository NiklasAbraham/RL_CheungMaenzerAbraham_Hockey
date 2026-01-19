# Weight initialization utilities for TD-MPC2
# Based on original TD-MPC2 implementation

import torch.nn as nn


def weight_init(m):
    """
    Custom weight initialization for TD-MPC2.
    Uses truncated normal initialization for Linear layers.

    Args:
        m: PyTorch module to initialize
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def zero_init_output_layer(module):
    """
    Zero-initialize the output layer of a network.
    Used for reward and Q-function networks to start with conservative estimates.

    Args:
        module: PyTorch module whose last linear layer should be zero-initialized
    """
    # Find the last Linear layer in the module
    last_linear = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last_linear = m

    if last_linear is not None:
        nn.init.constant_(last_linear.weight, 0)
        if last_linear.bias is not None:
            nn.init.constant_(last_linear.bias, 0)
