import unittest
from gradient_algorithms import LSMethod as LS
import numpy as np

class MyTestCase(unittest.TestCase):
    # def test_something(self):
    #     grad_phi = LS.ind_cell_LS()
    #     self.assertEqual(grad_phi[0], 4)  # add assertion here
    #     self.assertEqual(grad_phi[1], 6)  # add assertion here
    def test_ls_cell(self):
        dist = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        cells_phi_neighbour = np.array([18, 20, 10, 8])
        cell_phi_centre = 13
        grad_phi = LS.inv_cell(dist, cells_phi_neighbour, cell_phi_centre)
        self.assertEqual(grad_phi[0], 4)  # add assertion here
        self.assertEqual(grad_phi[1], 6)  # add assertion here
    def test_connectivity(self):
        i_cell = 2
        face_cell_connect = np.array([5, 2])
        check = LS.cell_face_neighbour(i_cell, face_cell_connect)
        self.assertEqual(check, 5)
