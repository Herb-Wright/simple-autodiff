import numpy as np
from simple_autodiff.diff import mat_mul_diff


def test_mat_mul():
	print("[TEST] test_mat_mult")
	df_a = np.arange(6).reshape((2, 3)).astype(float)
	b = np.arange(8).reshape((2, 4)).astype(float)
	c = np.arange(12).reshape((4, 3)).astype(float)
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
	df_b, df_c = mat_mul_diff(df_a, b, c)
	assert np.all(df_b == gt_df_b), "left grad was not correct"
	assert np.all(df_c == gt_df_c), "right grad was not correct"
	print("passed.")


if __name__ == "__main__":
	test_mat_mul()




