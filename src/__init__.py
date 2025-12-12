# __init__.py
# turns all of src into ONE package

"""
Pure NumPy implementations of Backpropagation and Direct Feedback Alignment
with arbitrary depth/width and bias terms.
"""

from .bp import BPNetwork
from .dfa import DFANetwork
from .utils import binary_cross_entropy

__version__ = "0.1.0"
__author__ = "You"

__all__ = [
    "BPNetwork",
    "DFANetwork",
    "binary_cross_entropy",
]