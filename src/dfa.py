# dfa.py
import numpy as np
from .utils import tanh, tanh_prime_from_output, sigmoid, binary_cross_entropy, xavier_init

class DFANetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, seed=None, feedback_scale=0.1):

        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1  # number of weight layers
        self.lr = learning_rate

        # Forward weights and biases
        self.W = [np.zeros((layer_sizes[i+1], layer_sizes[i]), dtype=np.float32) for i in range(self.L)]
        """
        for i in range(self.L):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i+1]
            self.W.append(xavier_init(fan_in, fan_out))  # (fan_out, fan_in)
        """
        self.b = [np.zeros(layer_sizes[i+1]) for i in range(self.L)]  # (fan_out,)

        # Feedback matrices B for DFA
        output_dim = layer_sizes[-1]
        self.B = []
        for i in range(1, self.L):
            hidden_dim = layer_sizes[i]
            scale = feedback_scale / np.sqrt(output_dim)
            self.B.append(
                np.random.uniform(-scale, scale, (hidden_dim, output_dim))
            )  # (hidden_dim, output_dim)

        # RMSprop stats
        self.m = [np.zeros_like(W) for W in self.W]      # not used but kept for consistency
        self.v = [np.ones_like(W) * 0.01 for W in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.ones_like(b) * 0.01 for b in self.b]
        self.epsilon = 1e-8
        self.decay = 0.95

    def forward(self, x):
        """
        x: shape (B, input_dim) or (input_dim,)
        """
        x = np.atleast_2d(x)  # ensure shape (B, in_dim)
        self.h = [x]          # h[0] = input, shape (B, size_0)
        self.a = [None]

        for l in range(1, self.L + 1):
            W = self.W[l-1]         # (size_l, size_{l-1})
            b = self.b[l-1]         # (size_l,)
            a = self.h[l-1] @ W.T + b   # (B, size_l)
            if l < self.L:
                h = tanh(a)
            else:
                h = sigmoid(a)
            self.a.append(a)
            self.h.append(h)

        return self.h[-1]  # shape (B, output_dim)

    def backward(self, y_true):
        """
        y_true: shape (B, output_dim) or (output_dim,)
        Computes gradients averaged over batch.
        """
        y_true = np.atleast_2d(y_true)
        y_hat = self.h[-1]          # (B, out_dim)
        batch_size = y_hat.shape[0]

        e = y_hat - y_true          # (B, out_dim)

        grads_W = [None] * self.L
        grads_b = [None] * self.L

        # Output layer (same as BP: delta = y_hat - y_true for sigmoid + BCE)
        # W_L: (out_dim, last_hidden_dim)
        grads_W[-1] = (e.T @ self.h[-2]) / batch_size      # (out_dim, last_hidden_dim)
        grads_b[-1] = e.mean(axis=0)                       # (out_dim,)

        # Hidden layers: direct random feedback (DFA)
        # Using same output error e for all hidden layers
        for l in range(1, self.L):  # l = 1 .. L-1, corresponding to layer_sizes[l]
            f_prime = tanh_prime_from_output(self.h[l])    # (B, hidden_dim)
            # Feedback: for each example, delta_l[b] = B[l-1] @ e[b]
            delta_l = (e @ self.B[l-1].T) * f_prime        # (B, hidden_dim)
            grads_W[l-1] = (delta_l.T @ self.h[l-1]) / batch_size  # (hidden_dim, prev_dim)
            grads_b[l-1] = delta_l.mean(axis=0)                    # (hidden_dim,)

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
        """
        x: (B, input_dim) or (input_dim,)
        y: (B, output_dim) or (output_dim,)
        """
        y_hat = self.forward(x)
        grads_W, grads_b = self.backward(y)
        self.update(grads_W, grads_b)
        # assumes binary_cross_entropy can handle batched tensors and returns mean loss
        return binary_cross_entropy(y, y_hat)
