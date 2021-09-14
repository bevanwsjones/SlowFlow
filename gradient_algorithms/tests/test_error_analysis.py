import unittest
from gradient_algorithms import error_analysis as ea
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_relative_error(self):
        x, x_true = -0.3502, -0.4162
        error = ea.relative_error(x, x_true)
        self.assertEqual(round(error, 4), 0.1586)  # add assertion here
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