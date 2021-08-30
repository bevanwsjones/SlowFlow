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
from mesh import cell as cl, face as ft, vertex as vt


class VertexConnectivityTest(ut.TestCase):

    def test_connect_vertices_to_cells(self):

        cell_vertex_connectivity = np.array(((0, 1), (1, 2)), dtype=int)
        vertex_table = vt.VertexTable(3)
        ct.connect_vertices_to_cells(cell_vertex_connectivity, vertex_table)

        self.assertEqual(len(vertex_table.connected_cell[0]), 1)
        self.assertEqual(len(vertex_table.connected_cell[1]), 2)
        self.assertEqual(len(vertex_table.connected_cell[2]), 1)

        self.assertEqual(vertex_table.connected_cell[0][0], 0)
        self.assertEqual(vertex_table.connected_cell[1][0], 0)
        self.assertEqual(vertex_table.connected_cell[1][1], 1)
        self.assertEqual(vertex_table.connected_cell[2][0], 1)

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
        print(vertex_vertex_connectivity[0])
        self.assertTrue(np.array_equal([1, 3], vertex_vertex_connectivity[0]))
        self.assertTrue(np.array_equal([0, 2, 4], vertex_vertex_connectivity[1]))
        self.assertTrue(np.array_equal([1, 3, 5], vertex_vertex_connectivity[2]))
        self.assertTrue(np.array_equal([0, 2], vertex_vertex_connectivity[3]))
        self.assertTrue(np.array_equal([1, 5, 6], vertex_vertex_connectivity[4]))
        self.assertTrue(np.array_equal([2, 4, 7], vertex_vertex_connectivity[5]))
        self.assertTrue(np.array_equal([4, 7], vertex_vertex_connectivity[6]))
        self.assertTrue(np.array_equal([5, 6], vertex_vertex_connectivity[7]))







