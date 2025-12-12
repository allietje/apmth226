# dfa.py
import numpy as np
from utils import tanh, tanh_prime_from_output, sigmoid, binary_cross_entropy, xavier_init

class DFANetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, seed=42, feedback_scale=0.1):
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1 # number of hidden layers
        self.lr = learning_rate

        # Forward weights and biases
        self.W = []
        for i in range(self.L):
            fan_in = layer_sizes[i]      # inputs to this layer
            fan_out = layer_sizes[i+1]   # outputs from this layer (neurons)
            self.W.append(xavier_init(fan_in, fan_out))  # shape: (fan_out, fan_in)
        self.b = [np.zeros(layer_sizes[i+1]) for i in range(self.L)]

        # For feedback B (DFA only): uniform scale 1/sqrt(f_out)
        output_dim = layer_sizes[-1]
        self.B = []
        for i in range(1, self.L):
            hidden_dim = layer_sizes[i]
            scale = feedback_scale / np.sqrt(output_dim)  
            self.B.append(np.random.uniform(-scale, scale, (hidden_dim, output_dim)))
            
        # for RMSprop
        self.m = [np.zeros_like(W) for W in self.W]  # momentum-like for RMSprop
        self.v = [np.ones_like(W) * 0.01 for W in self.W]  # variance
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.ones_like(b) * 0.01 for b in self.b]
        self.epsilon = 1e-8
        self.decay = 0.95  # beta1 for RMSprop

    def forward(self, x):
        self.h = [x]
        self.a = [None]

        for l in range(1, self.L + 1):
            a = self.W[l-1] @ self.h[l-1] + self.b[l-1] # a = Wx+b
            if l < self.L:
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
            # RMSprop for W
            self.v[l] = self.decay * self.v[l] + (1 - self.decay) * np.square(grads_W[l])
            self.W[l] -= self.lr * grads_W[l] / (np.sqrt(self.v[l]) + self.epsilon)
            # Same for b
            self.vb[l] = self.decay * self.vb[l] + (1 - self.decay) * np.square(grads_b[l])
            self.b[l] -= self.lr * grads_b[l] / (np.sqrt(self.vb[l]) + self.epsilon)

    def train_step(self, x, y): # one iteration (so one forward and one backward pass)
        y_hat = self.forward(x)
        grads_W, grads_b = self.backward(y)
        self.update(grads_W, grads_b)
        return binary_cross_entropy(y, y_hat)