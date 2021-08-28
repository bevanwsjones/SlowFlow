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
# filename: test_finite_volume.py
# description: Tests functions and methods of finite_volume.py
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import unittest as ut
import finite_volume as fv
from mesh import cell as cl


# ----------------------------------------------------------------------------------------------------------------------
# Cell Central Co-Ordinate
# ----------------------------------------------------------------------------------------------------------------------

class CoordinateCentreTest(ut.TestCase):

    def test_calculate_edge_center(self):
        vertex_coordinates = np.array([[0.5, 0.5], [1.5, 1.0], [2.5, 2.0]])
        cell_vertex_connectivity = np.array([[0, 1], [1, 2]])
        cell_centre = fv.calculate_edge_centroid(cell_vertex_connectivity, vertex_coordinates)

        # Check lengths
        self.assertEqual(2, len(cell_centre))
        self.assertEqual(2, len(cell_centre[0]))
        self.assertEqual(2, len(cell_centre[1]))

        # check value
        self.assertEqual(1.0, cell_centre[0][0])
        self.assertEqual(0.75, cell_centre[0][1])
        self.assertEqual(2.0, cell_centre[1][0])
        self.assertEqual(1.5, cell_centre[1][1])

    def test_calculate_triangle_center(self):
        vertex_coordinates = np.array([[0.5, 0.5], [1.0, 0.5], [1.0, 1.5], [1.5, 1.5]])
        cell_vertex_connectivity = np.array([[0, 1, 2], [2, 1, 3]])
        cell_centre = fv.calculate_triangle_centroid(cell_vertex_connectivity, vertex_coordinates)

        # Check lengths
        self.assertEqual(2, len(cell_centre))
        self.assertEqual(2, len(cell_centre[0]))
        self.assertEqual(2, len(cell_centre[1]))

        # check value
        base_x = 0.5
        length = 0.5
        base_y = 0.5
        height = 1.0
        self.assertAlmostEqual(base_x + (2.0 / 3.0) * length, cell_centre[0][0])
        self.assertAlmostEqual(base_y + (1.0 / 3.0) * height, cell_centre[0][1])
        self.assertAlmostEqual(base_x + length + (1.0 / 3.0) * length, cell_centre[1][0])
        self.assertAlmostEqual(base_y + (2.0 / 3.0) * height, cell_centre[1][1])

    def test_calculate_quadrilateral_center_regular(self):
        vertex_coordinates = np.array([[0.5, 0.5], [1.0, 0.5], [1.0, 1.5], [0.5, 1.5], [2.0, 1.0], [2.0, 2.0]])
        cell_vertex_connectivity = np.array([[0, 1, 2, 3], [3, 2, 4, 5]])
        cell_centre = fv.calculate_quadrilateral_centroid(cell_vertex_connectivity, vertex_coordinates)

        # Check lengths
        self.assertEqual(2, len(cell_centre))
        self.assertEqual(2, len(cell_centre[0]))
        self.assertEqual(2, len(cell_centre[1]))

        # check value
        base0 = [0.5, 0.5]
        base1 = [1.0, 0.5]
        length0 = [0.5, 1.0]
        length1 = [1.0, 1.0]
        print(cell_centre)
        self.assertAlmostEqual(base0[0] + (1.0 / 2.0) * length0[0], cell_centre[0][0])
        self.assertAlmostEqual(base0[1] + (1.0 / 2.0) * length0[1], cell_centre[0][1])
        self.assertAlmostEqual(base1[0] + (1.0 / 2.0) * length1[0], cell_centre[1][0])
        self.assertAlmostEqual(base1[1] + (1.0 / 2.0) * length1[1], cell_centre[1][1])

    def test_calculate_quadrilateral_center_irregular(self):
        vertex_coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [1.5, 1.5], [0.5, 1.0]])
        cell_vertex_connectivity = np.array([[0, 1, 2, 3]])

        # computes geometrically weighted centroid.
        centre = np.sum(vertex_coordinates, axis=0) / 4.0
        area0 = np.abs(np.cross((vertex_coordinates[0] - centre), (vertex_coordinates[1] - centre)))
        area1 = np.abs(np.cross((vertex_coordinates[1] - centre), (vertex_coordinates[2] - centre)))
        area2 = np.abs(np.cross((vertex_coordinates[2] - centre), (vertex_coordinates[3] - centre)))
        area3 = np.abs(np.cross((vertex_coordinates[3] - centre), (vertex_coordinates[0] - centre)))
        centre0 = np.sum((vertex_coordinates[0], vertex_coordinates[1], centre), axis=0) / 3.0
        centre1 = np.sum((vertex_coordinates[1], vertex_coordinates[2], centre), axis=0) / 3.0
        centre2 = np.sum((vertex_coordinates[2], vertex_coordinates[3], centre), axis=0) / 3.0
        centre3 = np.sum((vertex_coordinates[3], vertex_coordinates[0], centre), axis=0) / 3.0
        centre = (centre0 * area0 + centre1 * area1 + centre2 * area2 + centre3 * area3)/(area0 + area1 + area2 + area3)
        cell_centre = fv.calculate_quadrilateral_centroid(cell_vertex_connectivity, vertex_coordinates)

        # check lengths
        self.assertEqual(1, len(cell_centre))
        self.assertEqual(2, len(cell_centre[0]))

        # check value
        self.assertAlmostEqual(centre[0], cell_centre[0][0])
        self.assertAlmostEqual(centre[1], cell_centre[0][1])

    def test_calculate_hexagon_center(self):
        # TODO
        return

    def test_calculate_cell_center(self):
        # Check the routing to edge
        vertex_coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [1.5, 1.5], [0.5, 1.0]])
        cell_vertex_connectivity = np.array([[0, 1, 2, 3]])
        self.assertTrue(np.allclose(fv.calculate_edge_centroid(cell_vertex_connectivity, vertex_coordinates),
                                    fv.calculate_cell_centroid(cl.CellType.edge, cell_vertex_connectivity,
                                                               vertex_coordinates)))

        # Check the routing to triangle
        vertex_coordinates = np.array([[0.5, 0.5], [1.0, 0.5], [1.0, 1.5], [1.5, 1.5]])
        cell_vertex_connectivity = np.array([[0, 1, 2], [2, 1, 3]])
        self.assertTrue(np.allclose(fv.calculate_triangle_centroid(cell_vertex_connectivity, vertex_coordinates),
                                    fv.calculate_cell_centroid(cl.CellType.triangle, cell_vertex_connectivity,
                                                               vertex_coordinates)))

        # Check the routing to quadrilateral
        vertex_coordinates = np.array([[0.5, 0.5], [1.0, 0.5], [1.0, 1.5], [0.5, 1.5], [2.0, 1.0], [2.0, 2.0]])
        cell_vertex_connectivity = np.array([[0, 1, 2, 3], [3, 2, 4, 5]])
        self.assertTrue(np.allclose(fv.calculate_quadrilateral_centroid(cell_vertex_connectivity, vertex_coordinates),
                                    fv.calculate_cell_centroid(cl.CellType.quadrilateral, cell_vertex_connectivity,
                                                               vertex_coordinates)))
