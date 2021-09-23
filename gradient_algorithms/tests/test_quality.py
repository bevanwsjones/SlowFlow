import unittest as ut
import numpy as np
import gradient_algorithms.gridquality as gq

class test_nonorthogonality_test(ut.TestCase):
    def test_nonorthogonality(self):
        cell_centroid, neighbour_centroid, face_unit_normal = np.array([1,1]), np.array([3,1]), np.array([1,0])
        zeta = gq.nonorthogonality(cell_centroid, neighbour_centroid, face_unit_normal)
        self.assertEqual(zeta, 0.0)                         # test for orthogonality
        cell_centroid, neighbour_centroid, face_unit_normal = np.array([1,1]), np.array([3,2]), np.array([1,0])
        zeta = gq.nonorthogonality(cell_centroid, neighbour_centroid, face_unit_normal)
        self.assertEqual(round(zeta, 3), 0.464)             # test for nonorthoganlity

class test_closest_face_point(ut.TestCase):
    def test_closest_point(self):
        cell_centroid, neighbour_centroid, face_centroid = np.array([2, 2]), np.array([5, 2]), np.array([4, 6])
        close_point = gq.close_point(cell_centroid, neighbour_centroid, face_centroid)
        print(close_point)
        self.assertEqual(close_point[0], 4)
        self.assertEqual(close_point[1], 2)

class test_unevenness_test(ut.TestCase):
    def test_unevenness(self):
        cell_centroid, neighbour_centroid, face_centroid = np.array([1, 1]), np.array([3, 1]), np.array([2, 1])
        unevenness = gq.unevenness(cell_centroid, neighbour_centroid, face_centroid)
        self.assertEqual(unevenness, 0)
        cell_centroid, neighbour_centroid, face_centroid = np.array([2, 2]), np.array([5, 2]), np.array([4, 6])
        unevenness = gq.unevenness(cell_centroid, neighbour_centroid, face_centroid)
        self.assertEqual(unevenness, 1/6)

class test_skewness_test(ut.TestCase):
    def test_skewness(self):
        cell_centroid, neighbour_centroid, face_centroid = np.array([1, 1]), np.array([3, 1]), np.array([2, 1])
        skewness = gq.skewness(cell_centroid, neighbour_centroid, face_centroid)
        self.assertEqual(skewness, 0)
        cell_centroid, neighbour_centroid, face_centroid = np.array([2, 2]), np.array([5, 2]), np.array([4, 6])
        skewness = gq.skewness(cell_centroid, neighbour_centroid, face_centroid)
        self.assertEqual(skewness, 4/3)

class test_cell_quality(ut.TestCase):
    def test_cell_nonorthogonality(self):
        cell_centroid = np.array([[2, 2], [5, 2], [1, 10], [-2, 4], [1, -4]])
        face_centroid = np.array([[4, 6], [3, 7], [1, 2], [1, 1]])
        face_normals = np.array([[1, 0], [-0.5, 1], [-1, 2], [-1, -1]])
        fc_connectivity = np.array([[0 , 1], [0 , 2], [0, 3], [0, 4]])
        nonortho = gq.cell_nonorthogonality(cell_centroid, face_centroid, face_normals, fc_connectivity)
        self.assertEqual(nonortho[0], 0.2193)
        self.assertEqual(nonortho[1], 0.3108)
    def test_cell_uneven(self):
        cell_centroid = np.array([[2, 2], [5, 2], [1, 10], [-2, 4], [1, -4]])
        face_centroid = np.array([[4, 6], [3, 7], [1, 2], [1, 1]])
        fc_connectivity = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
        uneven = gq.cell_unevenness(cell_centroid, face_centroid, fc_connectivity)
        self.assertEqual(uneven[0], 0.4421)
        self.assertEqual(uneven[1], 1.3333)
    def test_cell_skew(self):
        cell_centroid = np.array([[2, 2], [5, 2], [1, 10], [-2, 4], [1, -4]])
        face_centroid = np.array([[4, 6], [3, 7], [1, 2], [1, 1]])
        fc_connectivity = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
        skew = gq.cell_skewness(cell_centroid, face_centroid, fc_connectivity)
        self.assertEqual(skew[0], 0.4421)
        self.assertEqual(skew[1], 1.3333)


