# test.py
import numpy as np
from apmth226 import BPNetwork, DFANetwork   # from __init__.py 

sizes = [4, 32, 32, 32, 1]

bp_net  = BPNetwork(sizes, learning_rate=0.05, seed=0) # initializes the BP and DFA networks
dfa_net = DFANetwork(sizes, learning_rate=0.05, seed=0, feedback_scale=0.1)

x = np.random.randn(4)
y = np.array([1.0])

# actually trains
print("BP  loss:", bp_net.train_step(x, y))
print("DFA loss:", dfa_net.train_step(x, y))
print("BP  bias norm:", np.linalg.norm(bp_net.b[0]))
print("DFA bias norm:", np.linalg.norm(dfa_net.b[0]))