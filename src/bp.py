# bp.py
import numpy as np
from .utils import tanh, tanh_prime_from_output, sigmoid, binary_cross_entropy, xavier_init

class BPNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, seed=None):

        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        self.lr = learning_rate

        # Forward weights and biases
        self.W = []
        for i in range(self.L):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i+1]
            self.W.append(xavier_init(fan_in, fan_out))   # (fan_out, fan_in)
        self.b = [np.zeros(layer_sizes[i+1]) for i in range(self.L)]

        # RMSprop stats
        self.m = [np.zeros_like(W) for W in self.W]      # kept for symmetry
        self.v = [np.ones_like(W) * 0.01 for W in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.ones_like(b) * 0.01 for b in self.b]
        self.epsilon = 1e-8
        self.decay = 0.95

    def forward(self, x):
        """
        x: (B, input_dim) or (input_dim,)
        """
        x = np.atleast_2d(x)
        self.h = [x]
        self.a = [None]

        for l in range(1, self.L + 1):
            W = self.W[l-1]
            b = self.b[l-1]
            a = self.h[l-1] @ W.T + b  # (B, size_l)
            if l < self.L:
                h = tanh(a)
            else:
                h = sigmoid(a)
            self.a.append(a)
            self.h.append(h)

        return self.h[-1]  # (B, output_dim)

    def backward(self, y_true):
        """
        y_true: (B, output_dim) or (output_dim,)
        """
        y_true = np.atleast_2d(y_true)
        y_hat = self.h[-1]              # (B, n_L)
        B = y_hat.shape[0]

        grads_W = [None] * self.L
        grads_b = [None] * self.L

        # deltas indexed by *layer index* l (0..L)
        deltas = [None] * (self.L + 1)

        # Output layer: sigmoid + BCE
        deltas[-1] = y_hat - y_true     # delta^L, shape (B, n_L)

        # Grad for last weight W_{L} ≡ self.W[L-1]
        grads_W[-1] = (deltas[-1].T @ self.h[-2]) / B
        grads_b[-1] = deltas[-1].mean(axis=0)

        # Hidden layers: l = L-1, ..., 1
        for l in range(self.L - 1, 0, -1):
            # h[l] is tanh output of layer l
            f_prime = tanh_prime_from_output(self.h[l])            # (B, n_l)

            # W[l] connects layer l → l+1, has shape (n_{l+1}, n_l)
            deltas[l] = (deltas[l+1] @ self.W[l]) * f_prime        # (B, n_l)

            # Grad for W_l ≡ self.W[l-1]
            grads_W[l-1] = (deltas[l].T @ self.h[l-1]) / B
            grads_b[l-1] = deltas[l].mean(axis=0)

        return grads_W, grads_b


    def update(self, grads_W, grads_b):
        for l in range(self.L):
            # RMSprop for W
            self.v[l] = self.decay * self.v[l] + (1 - self.decay) * np.square(grads_W[l])
            self.W[l] -= self.lr * grads_W[l] / (np.sqrt(self.v[l]) + self.epsilon)
            # RMSprop for b
            self.vb[l] = self.decay * self.vb[l] + (1 - self.decay) * np.square(grads_b[l])
            self.b[l] -= self.lr * grads_b[l] / (np.sqrt(self.vb[l]) + self.epsilon)

    def train_step(self, x, y):
        y_hat = self.forward(x)
        grads_W, grads_b = self.backward(y)
        self.update(grads_W, grads_b)
        return binary_cross_entropy(y, y_hat)
