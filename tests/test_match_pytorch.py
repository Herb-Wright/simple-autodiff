import torch
import numpy as np

from simple_autodiff import Tensor, ops


EPSILON = 1e-6

def test_mat_mul_simple():
	print("[TEST] test_mat_mult_simple")
	
	b_np= np.random.randn(2, 4)
	c_np = np.random.randn(4, 3)
	df_a_np = np.random.randn(2, 3)
	
	b = Tensor(b_np)
	c = Tensor(c_np)
	df_a = Tensor(df_a_np)
	
	b_t = torch.tensor(b_np, requires_grad=True)
	c_t = torch.tensor(c_np, requires_grad=True)
	df_a_t = torch.tensor(df_a_np, requires_grad=True)
	
	# in autodiff
	a = b @ c
	f = ops.sum(df_a * a)
	f.differentiate()
	
	# in pytorch
	a_t = b_t @ c_t
	f_t = (a_t * df_a_t).sum()
	f_t.backward()

	assert np.all(np.linalg.norm(b.gradient - b_t.grad.numpy()) < EPSILON), "left grad was not correct"
	assert np.all(np.linalg.norm(c.gradient - c_t.grad.numpy()) < EPSILON), "right grad was not correct"
	
	print("passed.")

def test_mat_mul_complex():
	print("[TEST] test_mat_mult_complex")

	D1, D2, D3, D4, D5 = 5, 6, 7, 8, 9
	c_np = np.random.randn(D1, D2, D4, D5)
	d_np = np.random.randn(D1, D2, D3, D1)
	e_np = np.random.randn(D1, D4)
	df_a_np = np.random.randn(D1, D2, D3, D5)
	
	c = Tensor(c_np)
	d = Tensor(d_np)
	e = Tensor(e_np)
	df_a = Tensor(df_a_np)
	
	c_t = torch.tensor(c_np, requires_grad=True)
	d_t = torch.tensor(d_np, requires_grad=True)
	e_t = torch.tensor(e_np, requires_grad=True)
	df_a_t = torch.tensor(df_a_np, requires_grad=True)
	
	# in autodiff
	b = d @ e
	a = b @ c
	f = ops.sum(df_a * a)
	f.differentiate()
	
	# in pytorch
	b_t = d_t @ e_t
	a_t = b_t @ c_t
	f_t = (a_t * df_a_t).sum()
	f_t.backward()

	assert np.all(np.linalg.norm(c.gradient - c_t.grad.numpy()) < EPSILON), "right grad was not correct"
	assert np.all(np.linalg.norm(d.gradient - d_t.grad.numpy()) < EPSILON), "left grad was not correct"
	assert np.all(np.linalg.norm(e.gradient - e_t.grad.numpy()) < EPSILON), "left grad was not correct"
	
	print("passed.")

def test_dot_product_simple():
	print("[TEST] test_dot_product_simple")

	b_np= np.random.randn(6)
	c_np = np.random.randn(6)
	df_a_np = np.random.randn(1)
	
	b = Tensor(b_np)
	c = Tensor(c_np)
	df_a = Tensor(df_a_np)
	
	b_t = torch.tensor(b_np, requires_grad=True)
	c_t = torch.tensor(c_np, requires_grad=True)
	df_a_t = torch.tensor(df_a_np, requires_grad=True)
	
	# in autodiff
	a = b @ c
	f = ops.sum(df_a * a)
	f.differentiate()
	
	# in pytorch
	a_t = b_t @ c_t
	f_t = (a_t * df_a_t).sum()
	f_t.backward()

	assert np.all(np.linalg.norm(b.gradient - b_t.grad.numpy()) < EPSILON), "left grad was not correct"
	assert np.all(np.linalg.norm(c.gradient - c_t.grad.numpy()) < EPSILON), "right grad was not correct"
	
	print("passed.")

def test_addition_subtraction():
	print("[TEST] test_addition_subtraction")

	a_np = np.random.randn(3, 4, 5)
	b_np = np.random.randn(3, 4, 5)
	c_np = np.random.randn(3, 4, 5)
	d_np = np.random.randn(3, 4, 5)
	a = Tensor(a_np)
	b = Tensor(b_np)
	c = Tensor(c_np)
	d = Tensor(d_np)
	a_t = torch.tensor(a_np, requires_grad = True)
	b_t = torch.tensor(b_np, requires_grad = True)
	c_t = torch.tensor(c_np, requires_grad = True)
	d_t = torch.tensor(d_np, requires_grad = True)
	
	# autodiff
	out = ops.sum(a + b - c - d)
	out.differentiate()

	# pytorch
	out = (a_t + b_t - c_t - d_t).sum()
	out.backward()

	assert np.all(np.linalg.norm(a.gradient - a_t.grad.numpy()) < EPSILON), "one of the grads did not match (a)"
	assert np.all(np.linalg.norm(b.gradient - b_t.grad.numpy()) < EPSILON), "one of the grads did not match (b)"
	assert np.all(np.linalg.norm(c.gradient - c_t.grad.numpy()) < EPSILON), "one of the grads did not match (c)"
	assert np.all(np.linalg.norm(d.gradient - d_t.grad.numpy()) < EPSILON), "one of the grads did not match (d)"
	print("passed.")

def test_scalar_mul_simple():
	print("[TEST] test_scalar_mul_simple")
	# setup
	x_np = np.random.randn(3, 4, 5)
	scalar = 17.5
	x = Tensor(x_np)
	x_t = torch.tensor(x_np, requires_grad = True)
	# autodiff
	y = ops.sum(scalar * x)
	y.differentiate()
	# pytorch
	y_t = (scalar * x_t).sum()
	y_t.backward()
	# assert
	assert np.all(np.linalg.norm(x.gradient - x_t.grad.numpy()) < EPSILON), "the gradient did not match"
	print("passed.")


def test_division_simple():
	print("[TEST] test_division_simple")
	# setup
	a_np = np.random.randn(3, 4, 5)
	b_np = np.random.randn(3, 4, 5)
	scalar = 17.5
	a = Tensor(a_np)
	b = Tensor(b_np)
	a_t = torch.tensor(a_np, requires_grad = True)
	b_t = torch.tensor(b_np, requires_grad = True)
	# autodiff
	y = ops.sum(a / b)
	y.differentiate()
	# pytorch
	y_t = (a_t / b_t).sum()
	y_t.backward()
	# assert
	assert np.all(np.linalg.norm(a.gradient - a_t.grad.numpy()) < EPSILON), "the gradient did not match: y1-a"
	assert np.all(np.linalg.norm(b.gradient - b_t.grad.numpy()) < EPSILON), "the gradient did not match: y1-b"
	# autodiff
	y = ops.sum(scalar / a)
	y.differentiate()
	# pytorch
	a_t.grad.zero_()
	y_t = (scalar / a_t).sum()
	y_t.backward()
	# assert
	assert np.all(np.linalg.norm(a.gradient - a_t.grad.numpy()) < EPSILON), "the gradient did not match: y2-a"
	# autodiff
	y = ops.sum(b / scalar)
	y.differentiate()
	# pytorch
	b_t.grad.zero_()
	y_t = (b_t / scalar).sum()
	y_t.backward()
	# assert
	assert np.all(np.linalg.norm(b.gradient - b_t.grad.numpy()) < EPSILON), "the gradient did not match: y3-b"
	print("passed.")

def test_exponentiation_simple():
	print("[TEST] test_exponentiation_simple")
	# setup
	a_np = np.random.randn(2, 3)
	scalar = 4
	a = Tensor(a_np)
	a_t = torch.tensor(a_np, requires_grad = True)
	# autodiff
	y = ops.sum(a ** scalar)
	y.differentiate()
	# pytorch
	y_t = (a_t ** scalar).sum()
	y_t.backward()
	# assert
	assert np.all(np.linalg.norm(a.gradient - a_t.grad.numpy()) < EPSILON), "the gradient did not match: Tensor ** float"
	# autodiff
	y = ops.sum(scalar ** a)
	y.differentiate()
	# pytorch
	a_t.grad.zero_()
	y_t = (scalar ** a_t).sum()
	y_t.backward()
	# assert
	assert np.all(np.linalg.norm(a.gradient - a_t.grad.numpy()) < EPSILON), "the gradient did not match: float ** Tensor"
	print("passed.")


def test_max_simple():
	print("[TEST] test_max_simple")
	x_np = np.random.randn(4, 5, 6, 7)
	x = Tensor(x_np)
	x_t = torch.tensor(x_np, requires_grad=True)
	dim = 2
	# autodiff
	m = ops.max(x, dim=dim)
	y = ops.sum(m)
	y.differentiate()
	# pytorch
	y_t = x_t.amax(dim=dim).sum()
	y_t.backward()
	# assert
	assert np.all(np.linalg.norm(x.gradient - x_t.grad.numpy()) < EPSILON), "the gradient did not match"
	print("passed.")




if __name__ == "__main__":
	test_mat_mul_simple()
	test_mat_mul_complex()
	test_dot_product_simple()
	test_addition_subtraction()
	test_scalar_mul_simple()
	test_division_simple()
	test_exponentiation_simple()
	test_max_simple()
	print("All tests pass!")

