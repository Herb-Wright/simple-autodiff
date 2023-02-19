from numpy import ndarray
import numpy as np
from typing import Iterable, Tuple, Union, Callable





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
	else:
		df_l = df_y @ r
		df_r = l @ df_y
	# return the reshaped values
	return df_l, df_r

def mat_add_diff(df_y: ndarray, l: ndarray, r: ndarray) -> Tuple[ndarray, ndarray]:
	'''
	args: `df_y`, `l`, `r` (same shape)

	returns: `df_l`,  `df_r` (same shape)
	'''
	return df_y, df_y

def mat_elem_mul_diff(df_y: ndarray, l: ndarray, r: ndarray) -> Tuple[ndarray, ndarray]:
	'''
	args: `df_y`, `l`, `r` (same shape)

	returns: `df_l`,  `df_r` (same shape)
	'''
	df_l = df_y * r
	df_r = df_y * l
	return df_l, df_r

def mat_elem_div_diff(df_y: ndarray, l: ndarray, r: ndarray) -> Tuple[ndarray, ndarray]:
	'''
	args: `df_y`, `l`, `r` (same shape)

	returns: `df_l`,  `df_r` (same shape)
	'''
	df_l = df_y / r
	df_r = df_y * l / (r ** 2)
	return df_l, df_r

def get_sum_diff_func(dim: Union[None, int, Iterable[int]]) -> Callable[[ndarray, ndarray], Tuple[ndarray, None]]:
	def sum_diff(df_y: ndarray, x: ndarray) -> Tuple[ndarray, None]:
		if dim is None:
			return df_y * np.ones_like(x)
		df_y_expanded = np.expand_dims(df_y, axis=dim)
		df_x = df_y_expanded * np.ones_like(x)
		return df_x, None

	return sum_diff

def broadcast_diff(df_y: ndarray, x: ndarray) -> Tuple[ndarray, None]:
	missing_dims = [i for i, d in enumerate(df_y.shape) if x.shape[i] != d]
	df_y = np.sum(df_y, axis=missing_dims)
	return df_y





