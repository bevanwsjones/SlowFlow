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


class CartesianMeshTest(ut.TestCase):

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
