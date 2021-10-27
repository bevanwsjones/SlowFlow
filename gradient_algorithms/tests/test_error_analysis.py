import unittest
from gradient_algorithms import error_analysis as ea
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_relative_error(self):
        x, x_true = -0.3502, -0.4162
        error = ea.relative_error(x, x_true)
        self.assertEqual(round(error, 4), 0.1586)  # add assertion here
    def test_abs_error(self):
        x, x_true = -0.3502, -0.4162
        error = ea.abs_error(x, x_true)
        self.assertEqual(round(error, 3), 0.066)
    def test_error_var(self):
        x, x_true = -0.3502, 0
        error = ea.abs_error(x, x_true)
        self.assertEqual(round(error, 4), 0.3502)  # add assertion here
    def test_array_relative_error(self):
        x, x_true = np.array([[-0.3502, -0.3502]]), np.array([[-0.4162, -0.4162]])
        error = ea.array_relative_error(x, x_true)
        self.assertEqual(round(error[0][1], 4), 0.1586)  # add assertion here
        self.assertEqual(round(error[0][0], 4), 0.1586)  # add assertion here
    def test_norm_one(self):
        error = np.array([[1.0, 1.5, 2.0, 3.0, 0.5, 2.0, 4.0]])
        norm_one = ea.L_norm_one(error)
        self.assertEqual(norm_one, 14)
    def test_norm_two(self):
        error = np.array([[1.0, 1.5, 2.0, 3.0, 0.5, 2.0, 4.0]])
        norm_two = ea.L_norm_two(error)
        self.assertEqual(round(norm_two, 4), 6.0415)
    def test_norm_inf(self):
        error = np.array([[1.0, 1.5, 2.0, 3.0, 0.5, 2.0, 4.0]])
        norm_inf = ea.L_norm_inf(error)
        self.assertEqual(norm_inf, 4.0)
    def test_res_norm_test(self):
        error_table = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 3.0]])
        print(error_table[:,0])
        error_result = ea.res_error(error_table)
        print(error_result)
        self.assertEqual(error_result[0], 5**0.5)
        self.assertEqual(error_result[1], 20 ** 0.5)
        self.assertEqual(error_result[2], 18 ** 0.5)
    def test_inf_norm_vol(self):
        error = np.array([2, 5, 10, 1, 2])
        vol_table = np.array([0.1, 0.1, 0.1, 10, 10])
        val = ea.L_norm_inf(error, vol_table)
        self.assertEqual(val, 2)
