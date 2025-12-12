# dfa.py
import numpy as np
from .utils import tanh, tanh_prime_from_output, sigmoid, binary_cross_entropy, xavier_init

class DFANetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, seed=42, feedback_scale=0.1):
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1 # number of hidden layers
        self.lr = learning_rate

        # Forward weights and biases
        self.W = []
        self.b = []
        for i in range(self.L): # initilization of random weights using Xavier. Biases are zero at init
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i+1]
            self.W.append(xavier_init(fan_in, fan_out, seed + i if seed else None))
            self.b.append(np.zeros(fan_out))

        # Fixed random feedback matrices: B[l] maps output error → hidden layer l pre-activation
        output_dim = layer_sizes[-1]
        self.B = [] # B_i for the matrix from output to the ith hidden layer
        np.random.seed(seed + 999)
        for i in range(1, self.L):  # one per hidden layer
            hidden_dim = layer_sizes[i]
            B = np.random.normal(0, feedback_scale, size=(hidden_dim, output_dim)) # feedback_scale controls how strong the Bs are
            self.B.append(B)

    def forward(self, x):
        self.h = [x]
        self.a = [None]

        for l in range(1, self.L + 1):
            a = self.W[l-1] @ self.h[l-1] + self.b[l-1] # a = Wx+b
            if l < self.L + 1:
                h = tanh(a)
            else:
                h = sigmoid(a)
            self.a.append(a)
            self.h.append(h)

        return self.h[-1] # returns output

    def backward(self, y_true):
        y_hat = self.h[-1] # output of model forward pass
        e = y_hat - y_true                                 # output error = delta in bp.py

        grads_W = [None] * self.L
        grads_b = [None] * self.L

        # Output layer (same as BP)
        grads_W[-1] = np.outer(e, self.h[-2]) # last weight changes by e * h_l^T
        grads_b[-1] = e.copy() # last bias changes by error

        # Hidden layers: direct random feedback
        for l in range(1, self.L):  # l = 1 to L-1
            f_prime = tanh_prime_from_output(self.h[l]) # derivative so f'(a)
            delta_l = (self.B[l-1] @ e) * f_prime          # direct feedback so delta * a_i = (B_i * e) \odot f'(a_i)
            grads_W[l-1] = np.outer(delta_l, self.h[l-1])  # gradient of W = delta * a * h^T
            grads_b[l-1] = delta_l.copy()

        return grads_W, grads_b

    def update(self, grads_W, grads_b):
        for l in range(self.L):
            self.W[l] -= self.lr * grads_W[l] # W <- W - (learning rate) * gradient
            self.b[l] -= self.lr * grads_b[l]

    def train_step(self, x, y): # one iteration (so one forward and one backward pass)
        y_hat = self.forward(x)
        grads_W, grads_b = self.backward(y)
        self.update(grads_W, grads_b)
        return binary_cross_entropy(y, y_hat)