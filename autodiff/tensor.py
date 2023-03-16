'''
This module contains the Tensor class implementation.
'''

from __future__ import annotations
import sys, os
sys.path.append(os.path.dirname(__file__))
from diff import mat_mul_diff, mat_elem_mul_diff, mat_add_diff, get_unsqueeze_diff, mat_elem_div_diff, power_rule_diff
from diff import exponential_rule_diff, squeeze_diff, broadcast_diff
import numpy as np
from numpy import ndarray
from typing import Union, Iterable, List
from numpy.typing import ArrayLike

class Tensor:
	'''Tensor class for autodiff'''
	def __init__(self, array: ArrayLike):
		self.value: ndarray = np.array(array)
		self.gradient_func: Union[callable, None] = None
		self.gradient: Union[ndarray, None] = None
		self.children: List[Union[Tensor, ndarray]] = []

	def differentiate(self: Tensor):
		'''
		differentiates the computation graph with respect to
		the tensor. gradients are stored in the `.gradient`
		property

		**Note.** to differentiate a tensor, it must be singular.
		'''
		# checks
		assert len(self.value.reshape(-1)) == 1, 'cannot differentiate a multi-dimensional vector (use a scalar)'
		assert len(self.children) > 0, 'no children to propogate gradient to'
		# zero grads
		self._zero_gradients()
		# set self gradient and propogate
		self._propogate_diff(np.ones_like(self.value))
		
	def _propogate_diff(self: Tensor, self_grad: ndarray):
		self.gradient = self.gradient + self_grad
		if len(self.children) == 0:
			return
		children_values = [(
			child.value if type(child) is Tensor else child
		) for child in self.children]
		child_grads = self.gradient_func(self_grad, *children_values)
		for i, child in enumerate(self.children):
			if type(child) is Tensor:
				child._propogate_diff(child_grads[i])

	def _zero_gradients(self):
		self.gradient = np.zeros_like(self.value)
		for child_tensor in self.children:
			if type(child_tensor) is Tensor:
				child_tensor._zero_gradients()

	def __add__(left: Tensor, right: Union[Tensor, float, ndarray]) -> Tensor:
		if type(right) is not Tensor:
			right = np.array(right)
			out = Tensor(left.value + right)
		else:
			out = Tensor(left.value + right.value)
		out.children.append(left)
		out.children.append(right)
		out.gradient_func = mat_add_diff
		return out

	def __radd__(right: Tensor, left: Union[Tensor, float, ndarray]) -> Tensor:
		if type(left) is not Tensor:
			left = np.array(right)
			out = Tensor(left + right.value)
		else:
			out = Tensor(left.value + right.value)
		out.children.append(left)
		out.children.append(right)
		out.gradient_func = mat_add_diff
		return out

	def __sub__(left: Tensor, right: Union[Tensor, float, ndarray]) -> Tensor:
		n_right = -right
		return left + n_right

	def __rsub__(right, left: Union[Tensor, float, ndarray]) -> Tensor:
		n_right = -right
		return left + n_right

	def __mul__(left: Tensor, right: Union[Tensor, float, ndarray]) -> Tensor:
		if type(right) is not Tensor:
			right = np.array(right)
			out = Tensor(left.value * right)
		else:
			out = Tensor(left.value * right.value)
		out.children.append(left)
		out.children.append(right)
		out.gradient_func = mat_elem_mul_diff
		return out

	def __rmul__(right: Tensor, left: Union[Tensor, float, ndarray]) -> Tensor:
		if type(left) is not Tensor:
			left = np.array(left) if type(left) is ndarray else np.array([left])
			out = Tensor(left * right.value)
		else:
			out = Tensor(left.value * right.value)
		out.children.append(left)
		out.children.append(right)
		out.gradient_func = mat_elem_mul_diff
		return out

	def __truediv__(left: Tensor, right: Union[Tensor, ndarray, float]) -> Tensor:
		if type(right) is not Tensor:
			right = np.array(right) if type(right) is ndarray else np.array([right])
			out = Tensor(left.value / right)
		else:
			out = Tensor(left.value / right.value)
		out.children.append(left)
		out.children.append(right)
		out.gradient_func = mat_elem_div_diff
		return out

	def __rtruediv__(right: Tensor, left: Union[Tensor, ndarray, float]) -> Tensor:
		if type(left) is not Tensor:
			left = np.array(left) if type(left) is ndarray else np.array([left])
			out = Tensor(left / right.value)
		else:
			out = Tensor(left.value / right.value)
		out.children.append(left)
		out.children.append(right)
		out.gradient_func = mat_elem_div_diff
		return out

	def __matmul__(left: Tensor, right: Union[Tensor, ndarray]) -> Tensor:
		if type(right) is not Tensor:
			out = Tensor(left.value @ right)
		else:
			out = Tensor(left.value @ right.value)
		out.children.append(left)
		out.children.append(right)
		out.gradient_func = mat_mul_diff
		return out

	def __pow__(left: Tensor, right: float) -> Tensor:
		out = Tensor(left.value ** right)
		out.children.append(left)
		out.children.append(right)
		out.gradient_func = power_rule_diff
		return out

	def __rpow__(right: Tensor, left: float) -> Tensor:
		out = Tensor(left ** right.value)
		out.children.append(left)
		out.children.append(right)
		out.gradient_func = exponential_rule_diff
		return out

	def __str__(self: Tensor):
		return str(f'Tensor({np.array_str(self.value, precision=8)})')
	
	def __repr__(self: Tensor):
		return str(f'Tensor({np.array_str(self.value, precision=8)})')

	def __neg__(self: Tensor):
		return (-1) * self

	def unsqueeze(self: Tensor, dim: int) -> Tensor:
		shape = list(self.value.shape)
		shape.insert(dim, 1)
		out = Tensor(self.value.reshape(shape))
		out.children.append(self)
		out.gradient_func = get_unsqueeze_diff(dim)
		return out
	
	def squeeze(self: Tensor) -> Tensor:
		out = Tensor(self.value.squeeze())
		out.children.append(self)
		out.gradient_func = squeeze_diff
		return out
		

	def broadcast_to(self:Tensor, shape: Iterable[int]) -> Tensor:
		out = Tensor(np.broadcast_to(self.value, shape))
		out.children.append(self)
		out.gradient_func = broadcast_diff
		return out

	def leafify(self: Tensor) -> None:
		self.children = []
		self.gradient_func = None





