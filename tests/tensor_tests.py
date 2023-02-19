import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from autodiff import Tensor, ops

def test_mat_mul():
	print('[TEST] test_mat_mult')
	df_a = Tensor(np.arange(6).reshape((2, 3)))
	b = Tensor(np.arange(8).reshape((2, 4)))
	c = Tensor(np.arange(12).reshape((4, 3)))
	gt_df_b = np.array([
		[  5,  14,  23,  32],
		[ 14,  50,  86, 122],
	])
	gt_df_c = np.array([
		[12., 16., 20.],
		[15., 21., 27.],
		[18., 26., 34.],
		[21., 31., 41.]
	])

	a = b @ c
	f = ops.sum(df_a * a)
	f.differentiate()
	df_b = b.gradient
	df_c = c.gradient

	assert np.all(df_b == gt_df_b), 'left grad was not correct'
	assert np.all(df_c == gt_df_c), 'right grad was not correct'

	print('passed.')

def test_scalar_mul_simple():
	print('[TEST] test_scalar_mul_simple')
	x_np = np.random.randn(2, 2)
	x = Tensor(x_np)
	y = ops.sum(2 * x)
	y.differentiate()
	assert np.all(x.gradient == (2 * np.ones_like(x_np)))
	print('passed.')

def test_negation():
	print('[TEST] test_negation')
	x_np = np.random.randn(2, 3, 4)
	x = Tensor(x_np)
	y = ops.sum(-x)
	y.differentiate()
	assert np.all(x.gradient == (- np.ones_like(x_np)))
	z = -y
	z.differentiate()
	assert np.all(x.gradient == np.ones_like(x_np))
	print('passed.')


if __name__ == '__main__':
	test_mat_mul()
	test_scalar_mul_simple()
	test_negation()
	print('all tests passed!')


	