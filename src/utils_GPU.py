# utils_GPU.py
import numpy as np
import random
import torch


def sigmoid(z):
    """
    Numerically stable sigmoid that works for both NumPy arrays and torch.Tensors.

    For torch.Tensors, we just use torch.sigmoid.
    For NumPy arrays, we clamp to avoid overflow.
    """
    if isinstance(z, torch.Tensor):
        return torch.sigmoid(z)
    # NumPy path
    z = np.clip(z, -250, 250)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime_from_output(y_hat):
    """
    Derivative of sigmoid using *post-activation* y_hat.

    Works for both NumPy arrays and torch.Tensors.
    """
    # same formula in both backends
    return y_hat * (1 - y_hat)


def tanh(z):
    """
    tanh activation for both NumPy arrays and torch.Tensors.
    """
    if isinstance(z, torch.Tensor):
        return torch.tanh(z)
    return np.tanh(z)


def tanh_prime_from_output(h):
    """
    Derivative of tanh using *post-activation* h = tanh(a).

    Works for both NumPy arrays and torch.Tensors.
    """
    return 1 - h**2


EPS = 1e-7  # good for float32

def binary_cross_entropy(y_true, y_pred):
    if isinstance(y_pred, torch.Tensor):
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.as_tensor(
                y_true, dtype=torch.float32, device=y_pred.device
            )

        y_pred = torch.clamp(y_pred, EPS, 1.0 - EPS)

        loss = -(y_true * torch.log(y_pred) +
                 (1 - y_true) * torch.log(1 - y_pred))
        return loss.mean()

    # NumPy path
    y_pred = np.clip(y_pred, EPS, 1.0 - EPS)
    return float(
        -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )
    )


def xavier_init(fan_in, fan_out, seed=None):
    """
    Xavier (Glorot) initialization for tanh/sigmoid networks.

    Returns a NumPy array of shape (fan_out, fan_in), as in your original code.
    We keep this NumPy-based and convert to torch inside the model constructors.
    """
    if seed is not None:
        np.random.seed(seed)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_out, fan_in))


def set_seed(seed: int = 0):
    """
    Set random seed for reproducibility across:
      - Python's random
      - NumPy
      - PyTorch (CPU and CUDA)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
