"""
This file contains an example of using autodiff to perform gradient descent for
linear regression.
"""
from typing import Tuple

import numpy as np
from tqdm import trange
from time import sleep

import simple_autodiff as ad
from simple_autodiff import Tensor


# constants
M = Tensor(np.array([0.4, 0.1, 0.2, -0.3]))
B, N, D, SIGMA = -0.2, 2000, 4, 0.1

EPOCHS = 1000
STEP_SIZE = 0.1

def init_params(dim: int = D) -> Tuple[Tensor, Tensor]:
	"""initalizes w, b to 0s"""
	w = Tensor(np.zeros((dim,)))
	b = Tensor(np.zeros((1,)))
	return w, b

def generate_fake_data() -> Tuple[Tensor, Tensor]:
	"""generates data as a linear equation with gaussian noise"""
	X = Tensor(np.random.random((N, D)))  # shape (N, D)
	y = X @ M + B + np.random.randn(N) * 0.1  #  shape (N,)
	y.leafify()
	return X, y


def get_prediction(w: Tensor, b: Tensor, X: Tensor) -> Tensor:
	"""get predictions given w, b, X"""
	return X @ w + b

def get_loss(pred: Tensor, y: Tensor) -> Tensor:
	"""return the loss (MSE) given predictions and y values"""
	return ad.mean((pred - y) ** 2)


if __name__ == "__main__":
	X, y = generate_fake_data()  # make our data
	w, b = init_params()  # init our weights to 0s

	for i in (bar := trange(EPOCHS)):  # loop through the different epochs
		pred = get_prediction(w, b, X)
		loss = get_loss(pred, y)  # calculate the loss
		loss.differentiate()
		w -= STEP_SIZE * w.gradient  # gradient descent steps
		b -= STEP_SIZE * b.gradient
		w.leafify()  # make sure we can garbage collect old w, b
		b.leafify()
		bar.set_description(f"loss: {loss.value}")
		sleep(0.01)  # artificially sleep or else it finishes very fast.

	print(f"found w={w.value} and b={b.value}!")  # print out our results
	print(f"(actual W={M.value} and B=[{B}])")

