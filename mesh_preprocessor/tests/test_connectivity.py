# ----------------------------------------------------------------------------------------------------------------------
#  This file is part of the SlowFlow distribution  (https://github.com/bevanwsjones/SlowFlow).
#  Copyright (c) 2020 Bevan Walter Stewart Jones.
#
#  This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation, version 3.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
#  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with this program. If not, see
#  <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------------------------------------------------
# filename: test_connectivity.py
# description: Tests the functions in connectivity .py
# ----------------------------------------------------------------------------------------------------------------------

import connectivity as ct
import numpy as np
import unittest as ut
from mesh import cell as cl


# ----------------------------------------------------------------------------------------------------------------------
# Vertex Connectivity
# ----------------------------------------------------------------------------------------------------------------------

class VertexConnectivityTest(ut.TestCase):

    def test_connect_vertices_to_cells(self):

        cell_vertex_connectivity = np.array([[0, 1, 2, 3], [2, 1, 4, 5], [5, 4, 6, 7]])
        vertex_cell_connectivity = ct.connect_vertices_to_cells(cell_vertex_connectivity, 8)

        # check lenghths
        self.assertEqual(8, len(vertex_cell_connectivity))
        self.assertEqual(1, len(vertex_cell_connectivity[0]))
        self.assertEqual(2, len(vertex_cell_connectivity[1]))
        self.assertEqual(2, len(vertex_cell_connectivity[2]))
        self.assertEqual(1, len(vertex_cell_connectivity[3]))
        self.assertEqual(2, len(vertex_cell_connectivity[4]))
        self.assertEqual(2, len(vertex_cell_connectivity[5]))
        self.assertEqual(1, len(vertex_cell_connectivity[6]))
        self.assertEqual(1, len(vertex_cell_connectivity[7]))

        # check values (must be ascending for each vertex)
        self.assertTrue(np.array_equal([0], vertex_cell_connectivity[0]))
        self.assertTrue(np.array_equal([0, 1], vertex_cell_connectivity[1]))
        self.assertTrue(np.array_equal([0, 1], vertex_cell_connectivity[2]))
        self.assertTrue(np.array_equal([0], vertex_cell_connectivity[3]))
        self.assertTrue(np.array_equal([1, 2], vertex_cell_connectivity[4]))
        self.assertTrue(np.array_equal([1, 2], vertex_cell_connectivity[5]))
        self.assertTrue(np.array_equal([2], vertex_cell_connectivity[6]))
        self.assertTrue(np.array_equal([2], vertex_cell_connectivity[7]))

    def test_connect_vertices_to_vertices(self):
        cell_vertex_connectivity = np.array([[0, 1, 2, 3], [2, 1, 4, 5], [5, 4, 6, 7]])
        vertex_vertex_connectivity = ct.connect_vertices_to_vertices(cell_vertex_connectivity, 8)

        # check lenghths
        self.assertEqual(8, len(vertex_vertex_connectivity))
        self.assertEqual(2, len(vertex_vertex_connectivity[0]))
        self.assertEqual(3, len(vertex_vertex_connectivity[1]))
        self.assertEqual(3, len(vertex_vertex_connectivity[2]))
        self.assertEqual(2, len(vertex_vertex_connectivity[3]))
        self.assertEqual(3, len(vertex_vertex_connectivity[4]))
        self.assertEqual(3, len(vertex_vertex_connectivity[5]))
        self.assertEqual(2, len(vertex_vertex_connectivity[6]))
        self.assertEqual(2, len(vertex_vertex_connectivity[7]))

        # check values (must be ascending for each vertex)
        self.assertTrue(np.array_equal([1, 3], vertex_vertex_connectivity[0]))
        self.assertTrue(np.array_equal([0, 2, 4], vertex_vertex_connectivity[1]))
        self.assertTrue(np.array_equal([1, 3, 5], vertex_vertex_connectivity[2]))
        self.assertTrue(np.array_equal([0, 2], vertex_vertex_connectivity[3]))
        self.assertTrue(np.array_equal([1, 5, 6], vertex_vertex_connectivity[4]))
        self.assertTrue(np.array_equal([2, 4, 7], vertex_vertex_connectivity[5]))
        self.assertTrue(np.array_equal([4, 7], vertex_vertex_connectivity[6]))
        self.assertTrue(np.array_equal([5, 6], vertex_vertex_connectivity[7]))


# ----------------------------------------------------------------------------------------------------------------------
# Face Connectivity
# ----------------------------------------------------------------------------------------------------------------------

class FaceConnectivityTest(ut.TestCase):

    def test_compute_number_of_faces(self):
        return True

    def test_connect_faces_to_vertex(self):
        cell_vertex_connectivity = np.array([[0, 1, 2], [2, 1, 3], [3, 1, 4], [3, 4, 5]])
        face_vertex_connectivity = ct.connect_faces_to_vertex(cell_vertex_connectivity)

        # Check lengths
        self.assertEqual(9, len(face_vertex_connectivity))
        self.assertEqual(2, len(face_vertex_connectivity[0]))
        self.assertEqual(2, len(face_vertex_connectivity[1]))
        self.assertEqual(2, len(face_vertex_connectivity[2]))
        self.assertEqual(2, len(face_vertex_connectivity[3]))
        self.assertEqual(2, len(face_vertex_connectivity[4]))
        self.assertEqual(2, len(face_vertex_connectivity[5]))
        self.assertEqual(2, len(face_vertex_connectivity[6]))
        self.assertEqual(2, len(face_vertex_connectivity[7]))
        self.assertEqual(2, len(face_vertex_connectivity[8]))

        # check values (must be ascending for each vertex) - boundary faces must be first.
        self.assertTrue(np.array_equal([0, 1], face_vertex_connectivity[0]))
        self.assertTrue(np.array_equal([0, 2], face_vertex_connectivity[1]))
        self.assertTrue(np.array_equal([2, 3], face_vertex_connectivity[2]))
        self.assertTrue(np.array_equal([1, 4], face_vertex_connectivity[3]))
        self.assertTrue(np.array_equal([3, 5], face_vertex_connectivity[4]))
        self.assertTrue(np.array_equal([4, 5], face_vertex_connectivity[5]))

        self.assertTrue(np.array_equal([1, 2], face_vertex_connectivity[6]))
        self.assertTrue(np.array_equal([1, 3], face_vertex_connectivity[7]))
        self.assertTrue(np.array_equal([3, 4], face_vertex_connectivity[8]))

    def test_connect_faces_to_cells(self):
        vertex_cell_connectivity = [np.array([0]), np.array([0, 1, 2]), np.array([0, 1]), np.array([1, 2, 3]),
                                    np.array([2, 3]), np.array([3])]
        face_vertex_connectivity = np.array([[0, 1], [0, 2], [2, 3], [1, 4], [3, 5], [4, 5], [1, 2], [1, 3], [3, 4]])
        face_cell_connectivity = ct.connect_faces_to_cells(vertex_cell_connectivity, face_vertex_connectivity)

        # Check lengths
        self.assertEqual(9, len(face_cell_connectivity))
        self.assertEqual(2, len(face_cell_connectivity[0]))
        self.assertEqual(2, len(face_cell_connectivity[1]))
        self.assertEqual(2, len(face_cell_connectivity[2]))
        self.assertEqual(2, len(face_cell_connectivity[3]))
        self.assertEqual(2, len(face_cell_connectivity[4]))
        self.assertEqual(2, len(face_cell_connectivity[5]))
        self.assertEqual(2, len(face_cell_connectivity[6]))
        self.assertEqual(2, len(face_cell_connectivity[7]))
        self.assertEqual(2, len(face_cell_connectivity[8]))

        # check values (must be ascending for each vertex) - boundary faces must be first.
        self.assertTrue(np.array_equal([0, -1], face_cell_connectivity[0]))
        self.assertTrue(np.array_equal([0, -1], face_cell_connectivity[1]))
        self.assertTrue(np.array_equal([1, -1], face_cell_connectivity[2]))
        self.assertTrue(np.array_equal([2, -1], face_cell_connectivity[3]))
        self.assertTrue(np.array_equal([3, -1], face_cell_connectivity[4]))
        self.assertTrue(np.array_equal([3, -1], face_cell_connectivity[5]))
        self.assertTrue(np.array_equal([0, 1], face_cell_connectivity[6]))
        self.assertTrue(np.array_equal([1, 2], face_cell_connectivity[7]))
        self.assertTrue(np.array_equal([2, 3], face_cell_connectivity[8]))

    def test_determine_face_boundary_status(self):

        face_cell_connectivity = np.array([[0, -1], [0, 1], [1, 2], [2, 3], [3, 4], [4, -1]])
        face_boundary_status = ct.determine_face_boundary_status(face_cell_connectivity)

        # check lenghths
        self.assertEqual(6, len(face_boundary_status))

        # check values
        self.assertTrue(np.array_equal([True, False, False, False, False, True], face_boundary_status))


# ----------------------------------------------------------------------------------------------------------------------
# Cell connectivity
# ----------------------------------------------------------------------------------------------------------------------

class CellConnectivityTest(ut.TestCase):

    def test_connect_cells_to_faces(self):
        face_cell_connectivity = np.array([[0, -1], [0, -1], [1, -1], [2, -1], [3, -1], [3, -1], [0, 1], [1, 2],
                                           [2, 3]])
        cell_face_connectivity = ct.connect_cells_to_faces(face_cell_connectivity, 4, cl.CellType.triangle)

        # Check lengths
        self.assertEqual(4, len(cell_face_connectivity))
        self.assertEqual(3, len(cell_face_connectivity[0]))
        self.assertEqual(3, len(cell_face_connectivity[1]))
        self.assertEqual(3, len(cell_face_connectivity[2]))
        self.assertEqual(3, len(cell_face_connectivity[3]))

        # check values (must be ascending for each vertex) - boundary faces must be first.
        self.assertTrue(np.array_equal([0, 1, 6], cell_face_connectivity[0]))
        self.assertTrue(np.array_equal([2, 6, 7], cell_face_connectivity[1]))
        self.assertTrue(np.array_equal([3, 7, 8], cell_face_connectivity[2]))
        self.assertTrue(np.array_equal([4, 5, 8], cell_face_connectivity[3]))
