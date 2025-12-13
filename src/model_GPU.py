# model_GPU.py
import torch

from .bp_GPU import BPNetwork
from .dfa_GPU import DFANetwork


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
    device: str | None = None,
):
    """
    Convenience factory to create matching BP and DFA networks on a given device.

    Args:
        device: e.g. "cuda", "cuda:0", "cuda:1", or "cpu".
                If None, uses "cuda" if available, else "cpu".

    Returns:
        bp_net, dfa_net, layer_sizes
    """
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    layer_sizes = make_layer_sizes(input_dim, width, depth, output_dim)

    bp_net = BPNetwork(
        layer_sizes,
        learning_rate=lr_bp,
        seed=seed,
        device=device_obj,
    )
    dfa_net = DFANetwork(
        layer_sizes,
        learning_rate=lr_dfa,
        seed=seed,
        feedback_scale=feedback_scale,
        device=device_obj,
    )

    return bp_net, dfa_net, layer_sizes
