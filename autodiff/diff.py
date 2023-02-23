'''
This module contains different functions for computing backwards derivatives.
'''

from numpy import ndarray
import numpy as np
from typing import Iterable, Tuple, Union, Callable


def _sum_to_shape(x: ndarray, shape: Iterable[int]) -> ndarray:
	len_diff = len(x.shape) - len(shape)
	if len_diff > 0:
		x = np.sum(x, axis=tuple(range(len_diff)))
	for i, dim in enumerate(shape):
		if dim == 1:
			x = np.sum(x, axis=i, keepdims=True)
	return x


def power_rule_diff(df_y, x: ndarray, p: ndarray) -> Tuple[ndarray, None]:
	df_x = df_y * p * (x ** (p - 1))
	return df_x, None

def mat_mul_diff(df_y: ndarray, l: ndarray, r: ndarray) -> Tuple[ndarray, ndarray]:
	'''
	The derivative of matrix multiplication. If	`y = l @ r`,	
	then `df_l = df_y @ r.T` and `df_r = l @ df_r`.

	args:
	- `df_y`: shape (..., N, P)
	- `l`: shape (..., N, M)
	- `r`: shape (..., M, P)

	returns:
	- `df_l`: shape (..., N, M)
	- `df_r`: shape (..., M, P)
	'''
	# transpose our matrices
	if len(l.shape) > 1:
		l = np.swapaxes(l, -1, -2)
	if len(r.shape) > 1:
		r = np.swapaxes(r, -1, -2)
	# calculate our derivatives
	if len(df_y.reshape(-1)) == 1:
		df_l = df_y * r
		df_r = l * df_y
	elif len(df_y.shape) == 1:
		if len(l.shape) == 1:
			df_r = np.outer(l, df_y)
			df_l = df_y @ r
		else:
			df_l = np.outer(df_y, r)
			df_r = l @ df_y
	else:
		df_l = df_y @ r
		df_r = l @ df_y
	# return the reshaped values
	df_l = _sum_to_shape(df_l, l.shape)
	df_r = _sum_to_shape(df_r, r.shape)
	return df_l, df_r

def mat_add_diff(df_y: ndarray, l: ndarray, r: ndarray) -> Tuple[ndarray, ndarray]:
	'''
	args: `df_y`, `l`, `r` (same shape)

	returns: `df_l`,  `df_r` (same shape)
	'''
	# return the reshaped values
	df_l = _sum_to_shape(df_y, l.shape)
	df_r = _sum_to_shape(df_y, r.shape)
	return df_l, df_r

def mat_elem_mul_diff(df_y: ndarray, l: ndarray, r: ndarray) -> Tuple[ndarray, ndarray]:
	'''
	args: `df_y`, `l`, `r` (same shape)

	returns: `df_l`,  `df_r` (same shape)
	'''
	df_l = df_y * r
	df_r = df_y * l
	# return the reshaped values
	df_l = _sum_to_shape(df_l, l.shape)
	df_r = _sum_to_shape(df_r, r.shape)
	return df_l, df_r

def mat_elem_div_diff(df_y: ndarray, l: ndarray, r: ndarray) -> Tuple[ndarray, ndarray]:
	'''
	args: `df_y`, `l`, `r` (same shape)

	returns: `df_l`,  `df_r` (same shape)
	'''
	df_l = df_y / r
	df_r = - df_y * l / (r ** 2)
	# return the reshaped values
	df_l = _sum_to_shape(df_l, l.shape)
	df_r = _sum_to_shape(df_r, r.shape)
	return df_l, df_r


def exponential_rule_diff(df_y: ndarray, b: ndarray, x: ndarray) -> Tuple[None, ndarray]:
	df_x = df_y * np.log(b) * b ** x
	return None, df_x

def get_sum_diff_func(dim: Union[None, int, Iterable[int]]) -> Callable[[ndarray, ndarray], Tuple[ndarray, None]]:
	def sum_diff(df_y: ndarray, x: ndarray) -> Tuple[ndarray, None]:
		if dim is None:
			return df_y * np.ones_like(x), None
		df_y_expanded = np.expand_dims(df_y, axis=dim)
		df_x = df_y_expanded * np.ones_like(x)
		return df_x, None

	return sum_diff

def broadcast_diff(df_y: ndarray, x: ndarray) -> Tuple[ndarray, None]:
	return _sum_to_shape(df_y, x.shape), None

def get_unsqueeze_diff(dim: Union[None, int, Iterable[int]]) -> Callable[[ndarray, ndarray], Tuple[ndarray, None]]:
	def unsqueeze_diff(df_y: ndarray, x: ndarray) -> Tuple[ndarray, None]:
		df_x = np.sum(df_y, axis=dim)
		return df_x, None

	return unsqueeze_diff

def squeeze_diff(df_y: ndarray, x: ndarray) -> Tuple[ndarray, None]:
	return df_y.reshape(x.shape), None


def get_max_diff_func(dim: Union[None, int, Iterable[int]]):
	def max_diff(df_y: ndarray, x: ndarray) -> Tuple[ndarray, None]:
		bool_arr = x == np.max(x, axis=dim, keepdims=True)
		if dim is None:
			df_y = df_y * np.ones_like(x)
		if type(dim) is int:
			shape = list(df_y.shape)
			shape.insert(dim, 1)
			df_y = df_y.reshape(shape)
		if type(dim) is list:
			shape = list(df_y.shape)
			for d in sorted(dim):
				shape.insert(d, 1)
			df_y = df_y.reshape(shape)
		df_x = df_y * bool_arr / np.sum(bool_arr, axis=dim, keepdims=True)
		return df_x, None

	return max_diff


def maximum_diff(df_y: ndarray, l: ndarray, r: ndarray) -> Tuple[ndarray, ndarray]:
	df_l = df_y * (l >= r)
	df_r = df_y * (r > l)
	df_l = _sum_to_shape(df_l, l.shape)
	df_r = _sum_to_shape(df_r, r.shape)
	return df_l, df_r

