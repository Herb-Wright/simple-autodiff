import sys, os
sys.path.append(os.path.dirname(__file__))
from tensor import Tensor
from diff import get_sum_diff_func
from typing import Union, Iterable
import numpy as np

def sum(tensor: Tensor, dim: Union[None, int, Iterable[int]] = None) -> Tensor:
	out = Tensor(np.sum(tensor.value, axis=dim))
	out.children.append(tensor)
	out.gradient_func = get_sum_diff_func(dim)
	return out

def max(tensor: Tensor, dim: Union[None, int, Iterable[int]] = None) -> Tensor:
	raise NotImplementedError

def min(tensor: Tensor, dim: Union[None, int, Iterable[int]] = None) -> Tensor:
	raise NotImplementedError

def maximum(tensor1: Tensor, tensor2: Tensor) -> Tensor:
	raise NotImplementedError

def minimum(tensor1: Tensor, tensor2: Tensor) -> Tensor:
	raise NotImplementedError

def mean(tensor: Tensor, dim: Union[None, int, Iterable[int]] = None) -> Tensor:
	if dim is None:
		N = len(tensor.value.reshape(-1))
	elif type(dim) is int:
		N = tensor.value.shape[dim]
	else:
		N = 0
		for d in dim:
			N += tensor.value.shape[d]
	return sum(tensor, dim) / N