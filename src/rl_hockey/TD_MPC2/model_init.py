# Weight initialization utilities

import torch.nn as nn


def weight_init(m):
    """Weight initialization for Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def zero_init_output_layer(module):
    """Zero-initialize the output layer of a network."""
    last_linear = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last_linear = m

    if last_linear is not None:
        nn.init.constant_(last_linear.weight, 0)
        if last_linear.bias is not None:
            nn.init.constant_(last_linear.bias, 0)
