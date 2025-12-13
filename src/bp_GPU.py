# bp_GPU.py
import numpy as np
import torch
from torch import nn

from .utils_GPU import (
    tanh,
    tanh_prime_from_output,
    sigmoid,
    binary_cross_entropy,
    xavier_init,
)


class BPNetwork(nn.Module):
    """
    Fully-connected MLP trained with standard backpropagation,
    implemented manually but using torch.Tensors so it can run on GPU.

    The public API is unchanged:
      - forward(x): returns network output
      - train_step(x, y): one SGD/RMSprop step and returns BCE loss

    x and y can still be NumPy arrays; they are converted to torch.Tensors
    on the specified device internally.
    """

    def __init__(self, layer_sizes, learning_rate=0.01, seed=None, device=None):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1  # number of weight layers
        self.lr = learning_rate
        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Optional: preserve old behavior where each network could reseed NumPy.
        if seed is not None:
            np.random.seed(seed)

        # Forward weights and biases as registered parameters (so .to(device) works)
        W_params = []
        b_params = []
        for i in range(self.L):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            # Initialize with Xavier in NumPy, then convert to torch on device
            w_np = xavier_init(fan_in, fan_out)
            w_t = torch.from_numpy(w_np).to(self.device).float()
            W_params.append(nn.Parameter(w_t))

            b_t = torch.zeros(fan_out, device=self.device, dtype=torch.float32)
            b_params.append(nn.Parameter(b_t))

        # ParameterList lets us iterate like a list but still have proper registration
        self.W = nn.ParameterList(W_params)
        self.b = nn.ParameterList(b_params)

        # RMSprop statistics (non-trainable tensors)
        self.v = [torch.ones_like(W) * 0.01 for W in self.W]  # for weights
        self.vb = [torch.ones_like(b) * 0.01 for b in self.b]  # for biases
        self.epsilon = 1e-8
        self.decay = 0.95

        # Placeholders for activations (used in manual backward)
        self.h = []
        self.a = []

    # --------------------
    # Helper: cast to tensor on correct device
    # --------------------
    def _to_tensor(self, x):
        """
        Convert input (NumPy array or torch.Tensor) to a float32 tensor on self.device.
        """
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    # --------------------
    # Forward pass
    # --------------------
    def forward(self, x):
        """
        x: (B, input_dim) or (input_dim,)

        Returns:
            h_L: (B, output_dim) torch.Tensor with values in (0, 1) (sigmoid output).
        """
        x = self._to_tensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # ensure a batch dimension

        h = x
        self.h = [h]   # h[0] = input
        self.a = [None]  # placeholder for consistent indexing

        for l in range(1, self.L + 1):
            W = self.W[l - 1]  # (size_l, size_{l-1})
            b = self.b[l - 1]  # (size_l,)

            # a = h @ W^T + b : (B, size_l)
            a = h @ W.t() + b

            # Hidden layers: tanh, output layer: sigmoid
            if l < self.L:
                h = tanh(a)
            else:
                h = sigmoid(a)

            self.a.append(a)
            self.h.append(h)

        return self.h[-1]

    # --------------------
    # Manual backward pass (BP)
    # --------------------
    def backward(self, y_true):
        """
        Manual backpropagation.

        y_true: (B, output_dim) or (output_dim,)

        Returns:
            grads_W: list of gradients for each W[l], same shape as W[l]
            grads_b: list of gradients for each b[l], same shape as b[l]
        """
        y_true = self._to_tensor(y_true)
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(0)

        y_hat = self.h[-1]  # (B, n_L)
        B = y_hat.shape[0]  # batch size

        grads_W = [None] * self.L
        grads_b = [None] * self.L
        deltas = [None] * (self.L + 1)  # index by layer

        # Output layer delta: sigmoid + BCE => delta^L = y_hat - y_true
        deltas[-1] = y_hat - y_true  # (B, n_L)

        # Gradient for last weight and bias: W_{L}
        grads_W[-1] = (deltas[-1].t() @ self.h[-2]) / B  # (n_L, n_{L-1})
        grads_b[-1] = deltas[-1].mean(dim=0)             # (n_L,)

        # Backpropagate to hidden layers: l = L-1, ..., 1
        for l in range(self.L - 1, 0, -1):
            # h[l] is tanh output of layer l
            f_prime = tanh_prime_from_output(self.h[l])  # (B, n_l)

            # W[l] connects layer l -> l+1, has shape (n_{l+1}, n_l)
            deltas[l] = (deltas[l + 1] @ self.W[l]) * f_prime  # (B, n_l)

            # Gradients for W_{l}, b_{l}
            grads_W[l - 1] = (deltas[l].t() @ self.h[l - 1]) / B  # (n_l, n_{l-1})
            grads_b[l - 1] = deltas[l].mean(dim=0)                # (n_l,)

        return grads_W, grads_b

    # --------------------
    # RMSprop update
    # --------------------
    def update(self, grads_W, grads_b):
        """
        Apply RMSprop updates to all weights and biases using the given gradients.
        """
        with torch.no_grad():
            for l in range(self.L):
                # RMSprop for W
                self.v[l] = self.decay * self.v[l] + (1 - self.decay) * (grads_W[l] ** 2)
                self.W[l] -= self.lr * grads_W[l] / (self.v[l].sqrt() + self.epsilon)

                # RMSprop for b
                self.vb[l] = self.decay * self.vb[l] + (1 - self.decay) * (grads_b[l] ** 2)
                self.b[l] -= self.lr * grads_b[l] / (self.vb[l].sqrt() + self.epsilon)

    # --------------------
    # One training step
    # --------------------
    def train_step(self, x, y):
        """
        One training step:
          - forward pass
          - manual backward (BP)
          - RMSprop update
          - return BCE loss (torch scalar)
        """
        with torch.no_grad():
            y_hat = self.forward(x)
            grads_W, grads_b = self.backward(y)
            self.update(grads_W, grads_b)
            loss = binary_cross_entropy(y, y_hat)
        return loss
