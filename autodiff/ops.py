'''
This module contains Different differentiable operations on autodiff Tensors
'''
import sys, os
sys.path.append(os.path.dirname(__file__))
from tensor import Tensor
from diff import get_sum_diff_func, get_max_diff_func, maximum_diff
from typing import Union, Iterable
import numpy as np

def sum(tensor: Tensor, dim: Union[None, int, Iterable[int]] = None) -> Tensor:
	out = Tensor(np.sum(tensor.value, axis=dim))
	out.children.append(tensor)
	out.gradient_func = get_sum_diff_func(dim)
	return out

def max(tensor: Tensor, dim: Union[None, int, Iterable[int]] = None) -> Tensor:
	out = Tensor(np.max(tensor.value, axis=dim))
	out.children.append(tensor)
	out.gradient_func = get_max_diff_func(dim)
	return out

def min(tensor: Tensor, dim: Union[None, int, Iterable[int]] = None) -> Tensor:
	return -1 * max(-1 * tensor, dim=dim)

def maximum(tensor1: Tensor, tensor2: Tensor) -> Tensor:
	if type(tensor1) is not Tensor:
		tensor1 = Tensor(tensor1)
	if type(tensor2) is not Tensor:
		tensor2 = Tensor(tensor2)
	out = Tensor(np.maximum(tensor1.value, tensor2.value))
	out.children.append(tensor1)
	out.children.append(tensor2)
	out.gradient_func = maximum_diff
	return out

def minimum(tensor1: Tensor, tensor2: Tensor) -> Tensor:
	return -1 * maximum(-1 * tensor1, -1 * tensor2)

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

def relu(tensor: Tensor) -> Tensor:
	return maximum(tensor, Tensor(0))

def sqrt(tensor: Tensor) -> Tensor:
	return tensor ** (1 / 2)

