# dfa_GPU.py
import numpy as np
import torch
from torch import nn

from .utils_GPU import (
    tanh,
    tanh_prime_from_output,
    sigmoid,
    binary_cross_entropy,
)


class DFANetwork(nn.Module):
    """
    Fully-connected MLP trained with Direct Feedback Alignment (DFA).

    Forward path is the same MLP as BPNetwork; only the backward rule differs.

    Uses:
      delta_L = y_hat - y_true         (same as BP)
      delta_l = (B^l e) ⊙ phi'(a^l)    for hidden layers,
    where B^l is a fixed random feedback matrix from output to layer l.

    All computations are done with torch.Tensors so it runs on GPU.
    """

    def __init__(
        self,
        layer_sizes,
        learning_rate=0.01,
        seed=None,
        feedback_scale=0.1,
        device=None,
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1  # number of weight layers
        self.lr = learning_rate
        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        if seed is not None:
            np.random.seed(seed)

        # Forward weights and biases
        # Here you were using zero init for DFA, so we preserve that behavior.
        W_params = []
        b_params = []
        for i in range(self.L):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            w_t = torch.zeros((fan_out, fan_in), device=self.device, dtype=torch.float32)
            W_params.append(nn.Parameter(w_t))

            b_t = torch.zeros(fan_out, device=self.device, dtype=torch.float32)
            b_params.append(nn.Parameter(b_t))

        self.W = nn.ParameterList(W_params)
        self.b = nn.ParameterList(b_params)

        # Feedback matrices B^l (fixed, non-trainable)
        output_dim = layer_sizes[-1]
        self.B = []
        rng = np.random.RandomState(seed if seed is not None else None)
        for i in range(1, self.L):
            hidden_dim = layer_sizes[i]
            # Same scaling as your original code: feedback_scale / sqrt(output_dim)
            scale = feedback_scale / np.sqrt(output_dim)
            B_np = rng.uniform(-scale, scale, (hidden_dim, output_dim)).astype(np.float32)
            B_t = torch.from_numpy(B_np).to(self.device)
            self.B.append(B_t)  # shape (hidden_dim, output_dim)

        # RMSprop stats
        self.v = [torch.ones_like(W) * 0.01 for W in self.W]
        self.vb = [torch.ones_like(b) * 0.01 for b in self.b]
        self.epsilon = 1e-8
        self.decay = 0.95

        # Activations and pre-activations
        self.h = []
        self.a = []

    def _to_tensor(self, x):
        """
        Convert input (NumPy array or torch.Tensor) to float32 tensor on self.device.
        """
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    # --------------------
    # Forward pass
    # --------------------
    def forward(self, x):
        """
        x: shape (B, input_dim) or (input_dim,)

        Returns:
            h_L: (B, output_dim) torch.Tensor (sigmoid output)
        """
        x = self._to_tensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        h = x
        self.h = [h]   # h[0] = input
        self.a = [None]

        for l in range(1, self.L + 1):
            W = self.W[l - 1]
            b = self.b[l - 1]
            a = h @ W.t() + b  # (B, size_l)

            if l < self.L:
                h = tanh(a)
            else:
                h = sigmoid(a)

            self.a.append(a)
            self.h.append(h)

        return self.h[-1]

    # --------------------
    # Manual DFA backward
    # --------------------
    def backward(self, y_true):
        """
        y_true: shape (B, output_dim) or (output_dim,)

        Returns:
            grads_W, grads_b: lists of gradients for each layer.
        """
        y_true = self._to_tensor(y_true)
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(0)

        y_hat = self.h[-1]             # (B, out_dim)
        batch_size = y_hat.shape[0]

        e = y_hat - y_true             # (B, out_dim)

        grads_W = [None] * self.L
        grads_b = [None] * self.L

        # Output layer gradient: same as BP (sigmoid + BCE)
        grads_W[-1] = (e.t() @ self.h[-2]) / batch_size   # (out_dim, last_hidden_dim)
        grads_b[-1] = e.mean(dim=0)                       # (out_dim,)

        # Hidden layers: direct random feedback from output error
        for l in range(1, self.L):  # layers 1..L-1
            f_prime = tanh_prime_from_output(self.h[l])   # (B, hidden_dim)
            # Feedback: delta_l[b] = B^{l} e[b]
            # self.B[l-1]: (hidden_dim, out_dim)
            delta_l = (e @ self.B[l - 1].t()) * f_prime   # (B, hidden_dim)
            grads_W[l - 1] = (delta_l.t() @ self.h[l - 1]) / batch_size
            grads_b[l - 1] = delta_l.mean(dim=0)

        return grads_W, grads_b

    # --------------------
    # RMSprop update
    # --------------------
    def update(self, grads_W, grads_b):
        """
        Apply RMSprop updates to all layers.
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
        One training step with DFA:
          - forward
          - DFA backward
          - RMSprop update
          - return BCE loss
        """
        with torch.no_grad():
            y_hat = self.forward(x)
            grads_W, grads_b = self.backward(y)
            self.update(grads_W, grads_b)
            loss = binary_cross_entropy(y, y_hat)
        return loss
