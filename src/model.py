# model.py
import numpy as np
from .bp import BPNetwork
from .dfa import DFANetwork


def make_layer_sizes(
    input_dim: int = 784,
    width: int = 200,
    depth: int = 3,
    output_dim: int = 10,
):
    """
    Build layer_sizes like [input_dim, width, width, ..., width, output_dim].
    """
    hidden = [width] * depth
    return [input_dim] + hidden + [output_dim]


def create_networks(
    width: int = 200,
    depth: int = 3,
    lr_bp: float = 0.005,
    lr_dfa: float = 0.01,
    seed: int = 0,
    feedback_scale: float = 0.1,
    input_dim: int = 784,
    output_dim: int = 10,
):
    """
    Convenience factory to create matching BP and DFA networks.

    Returns:
        bp_net, dfa_net, layer_sizes
    """
    layer_sizes = make_layer_sizes(input_dim, width, depth, output_dim)
    bp_net = BPNetwork(layer_sizes, learning_rate=lr_bp, seed=seed)
    dfa_net = DFANetwork(
        layer_sizes,
        learning_rate=lr_dfa,
        seed=seed,
        feedback_scale=feedback_scale,
    )
    return bp_net, dfa_net, layer_sizes
