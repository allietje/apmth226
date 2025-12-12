# utils.py
import numpy as np

def sigmoid(z): # literally just 1/(1-e^(-z)) aka logistic function
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

def sigmoid_prime_from_output(y_hat): # derivative of sigmoid function using post-activation, not pre-activation
    return y_hat * (1 - y_hat)

def tanh(z): # tanh activation function
    return np.tanh(z)

def tanh_prime_from_output(h): # derivative of tanh function using post-activations, not pre-activations
    return 1 - h**2

def binary_cross_entropy(y_true, y_pred): # our loss function
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def xavier_init(fan_in, fan_out, seed=None): # best random initialization of weights for sigmoid and tanh
    if seed is not None:
        np.random.seed(seed)
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_out, fan_in))