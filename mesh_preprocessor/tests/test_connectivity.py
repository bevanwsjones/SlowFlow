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

    def test_determine_face_boundary_status(self):

        face_cell_connectivity = np.array([[0, -1], [0, 1], [1, 2], [2, 3], [3, 4], [4, -1]])
        face_boundary_status = ct.determine_face_boundary_status(face_cell_connectivity)

        # check lenghths
        self.assertEqual(6, len(face_boundary_status))

        # check values (must be ascending for each vertex)
        self.assertTrue(np.array_equal([True, False, False, False, False, True], face_boundary_status))


# ----------------------------------------------------------------------------------------------------------------------
# Cell connectivity
# ----------------------------------------------------------------------------------------------------------------------
