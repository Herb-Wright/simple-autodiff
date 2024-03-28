"""
# Simple Autodiff

Package for automatic differentiation in python.

Built on top of numpy. it supports many common operations
between tensors such as addition, multiplication, exponentiation,
summing over dimensions, etc.
"""

from .tensor import Tensor
from .ops import sum, mean
