# bp.py
import numpy as np
from utils import tanh, tanh_prime_from_output, sigmoid, binary_cross_entropy, xavier_init

class BPNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, seed=42):
        """
        layer_sizes: e.g. [784, 400, 400, 400, 10]
        """
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        self.lr = learning_rate 

        # Forward weights and biases, initiate zeroes
        self.W = []
        for i in range(self.L):
            fan_in = layer_sizes[i]      # inputs to this layer
            fan_out = layer_sizes[i+1]   # outputs from this layer (neurons)
            self.W.append(xavier_init(fan_in, fan_out))  # shape: (fan_out, fan_in)
        self.b = [np.zeros(layer_sizes[i+1]) for i in range(self.L)]
        
        # for RMSprop
        self.m = [np.zeros_like(W) for W in self.W]  # momentum-like for RMSprop
        self.v = [np.ones_like(W) * 0.01 for W in self.W]  # variance
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.ones_like(b) * 0.01 for b in self.b]
        self.epsilon = 1e-8
        self.decay = 0.95  # beta1 for RMSprop

    def forward(self, x): # forward pass
        """Performs a forward pass through the neural network.

        Computes the activations for all layers, applying tanh activation
        for hidden layers and sigmoid activation for the output layer.
        Stores intermediate activations and pre-activations in self.h and self.a
        for use in backpropagation.

        Args:
            x (np.ndarray): Input vector of shape (input_dim,). The input features
                           to be propagated through the network.

        Returns:
            np.ndarray: Output vector of shape (output_dim,). The network's predictions
                       after applying sigmoid activation to the final layer.
        """
        self.h = [x]           # activations: h[0] = input x
        self.a = [None]        # pre-activations (a[0] unused)

        for l in range(1, self.L + 1): # loops over all hidden layers
            a = self.W[l-1] @ self.h[l-1] + self.b[l-1] # a = Wx + b
            if l < self.L:  # hidden layers
                h = tanh(a)
            else:               # output layer
                h = sigmoid(a)
            self.a.append(a)
            self.h.append(h)

        return self.h[-1]

    def backward(self, y_true):
        """Performs backpropagation to compute gradients of the loss with respect to weights and biases.

        Computes the error at the output layer and propagates it backward through all layers,
        calculating gradients for weights and biases at each layer. Uses the chain rule and
        tanh derivative for hidden layers. Assumes forward() has been called first to populate
        self.h with activations.

        Args:
            y_true (np.ndarray): True target values of shape (output_dim,). The ground truth
                                labels for the input that was passed through forward().

        Returns:
            tuple: A tuple containing:
                - grads_W (list of np.ndarray): List of weight gradients for each layer.
                  Each element has shape matching the corresponding weight matrix.
                - grads_b (list of np.ndarray): List of bias gradients for each layer.
                  Each element has shape (layer_output_dim,).
        """
        y_hat = self.h[-1] # output of the forward pass
        delta = y_hat - y_true                      # (output_dim,)

        grads_W = []
        grads_b = []

        for l in range(self.L, 0, -1): # backpropagating through all the layers
            
            # after a backward step, calculate gradients
            # Bias gradient: just the error (mean over batch later if needed)
            grad_b = delta.copy() # ∂J/∂b = delta * a_y \cdot 1 = delta * a_y which we call delta
            # Weight gradient: outer product
            grad_W = np.outer(delta, self.h[l-1]) # ∂J/∂W = delta * a_y * h^T

            grads_W.insert(0, grad_W) # adds new gradient at start of list since we are going in reverse so last is first
            grads_b.insert(0, grad_b)

            if l == 1: # done with backpropagation, don't do more
                break

            # Propagate delta backward -> backward step
            f_prime = tanh_prime_from_output(self.h[l-1]) # derivate of tanh is 1- tanh ^2, so here just 1- h(x)
            delta = (self.W[l-1].T @ delta) * f_prime # delta * a_i = (W_i+1^T* delta * a_i+1) \odot f'(a_i)

        return grads_W, grads_b # full list of gradients of weights and biases of each neuron.

    def update(self, grads_W, grads_b): # updates the weights and biases using the gradients
        for l in range(self.L):
            # RMSprop for W
            self.v[l] = self.decay * self.v[l] + (1 - self.decay) * np.square(grads_W[l])
            self.W[l] -= self.lr * grads_W[l] / (np.sqrt(self.v[l]) + self.epsilon)
            # Same for b
            self.vb[l] = self.decay * self.vb[l] + (1 - self.decay) * np.square(grads_b[l])
            self.b[l] -= self.lr * grads_b[l] / (np.sqrt(self.vb[l]) + self.epsilon)

    def train_step(self, x, y): # one training pass (forward + backward + update)
        y_hat = self.forward(x) # output of forward pass
        grads_W, grads_b = self.backward(y) # gradients after backward pass
        self.update(grads_W, grads_b) # update weights and biases
        return binary_cross_entropy(y, y_hat) # return the loss for eval