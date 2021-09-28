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
# filename: test_mesh_generator.py
# description: Contains unit tests for the mesh_entities generator.
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import unittest as ut
from mesh_entities import cell as cl
from mesh_generation import mesh_generator as mg


# ----------------------------------------------------------------------------------------------------------------------
# Standard Mesh Transformation Functions
# ----------------------------------------------------------------------------------------------------------------------

class StandardMeshTransformationFunctionsTest(ut.TestCase):

    def test_structured(self):
        n = [3, 3]
        x_0 = [-1.0, -1.0]
        ds = [2.0, 2.0]

        # demonstrate that the co-ordinates are unchanged.
        self.assertTrue(np.array_equal(np.array([1.0, 1.0]), mg.structured(n, x_0, ds, np.array([1.0, 1.0]))))
        self.assertTrue(np.array_equal(np.array([0.0, -1.0]), mg.structured(n, x_0, ds, np.array([0.0, -1.0]))))
        self.assertTrue(np.array_equal(np.array([1.0, 2.0]), mg.structured(n, x_0, ds, np.array([1.0, 2.0]))))

    def test_stretch(self):
        ratio = np.array([0.5, 0.25])
        n = [2, 2]
        x_0 = np.array([-1.0, -1.0])
        ds = np.array([2.0, 2.0])

        # demonstrate that corners don't change but the centre vertex moves appropriately.
        self.assertTrue(np.allclose(np.array([1.0/3.0, 0.6]), mg.stretch(ratio, n, x_0, ds, np.array([0.0, 0.0]))))
        self.assertTrue(np.allclose(np.array([-1.0, -1.0]), mg.stretch(ratio, n, x_0, ds, np.array([-1.0, -1.0]))))
        self.assertTrue(np.allclose(np.array([-1.0, 1.0]), mg.stretch(ratio, n, x_0, ds, np.array([-1.0, 1.0]))))
        self.assertTrue(np.allclose(np.array([1.0, -1.0]), mg.stretch(ratio, n, x_0, ds, np.array([1.0, -1.0]))))
        self.assertTrue(np.allclose(np.array([1.0, 1.0]), mg.stretch(ratio, n, x_0, ds, np.array([1.0, 1.0]))))

    def test_parallelogram_normalised(self):
        gradient = np.array([0.5, 0.25])
        n = [2, 2]
        x_0 = np.array([-1.0, -2.0])
        ds = np.array([2.0, 2.0])

        # demonstrate start and end point recovery and a random mid-point.
        self.assertTrue(np.allclose(x_0, mg.parallelogram(True, gradient, n, x_0, ds, x_0)))
        self.assertTrue(np.allclose(x_0 + ds, mg.parallelogram(True, gradient, n, x_0, ds, x_0 + ds)))
        self.assertTrue(np.allclose([0.0, -1.0], mg.parallelogram(True, gradient, n, x_0, ds, x_0 + 0.5*ds)))

    def test_parallelogram_unnormalised(self):
        gradient = np.array([0.5, 0.25])
        n = [2, 2]
        x_0 = np.array([1.0, 2.0])
        ds = np.array([2.0, 2.0])

        # Demonstrate start and gradient recovery.
        self.assertTrue(np.allclose(x_0, mg.parallelogram(False, gradient, n, x_0, ds, x_0)))
        self.assertTrue(np.allclose([2.25, 3.5], mg.parallelogram(False, gradient, n, x_0, ds, x_0 + 0.5*ds)))
        x_point = mg.parallelogram(False, gradient, n, x_0, ds, x_0 + np.array([0.5 * ds[0], 0.0]))
        self.assertAlmostEqual(gradient[0], (x_point[1] - x_0[1])/(x_point[0] - x_0[0]))
        y_point = mg.parallelogram(False, gradient, n, x_0, ds, x_0 + np.array([0.0, 0.5 * ds[1]]))
        self.assertAlmostEqual(gradient[1], (y_point[0] - x_0[0])/(y_point[1] - x_0[1]))


# ----------------------------------------------------------------------------------------------------------------------
# 2D Mesh Generation - Structured Meshes
# ----------------------------------------------------------------------------------------------------------------------

class Setup2dCartesianMeshTest(ut.TestCase):

    def test_equispaced_unit_mesh(self):
        [vertices, cells, cell_type] = mg.setup_2d_cartesian_mesh([4, 4])

        self.assertEqual(cell_type, cl.CellType.quadrilateral)

        self.assertEqual(16, len(cells))
        for cell_vertex in cells:
            self.assertEqual(4, len(cell_vertex))

        self.assertTrue(np.array_equal([0, 1, 6, 5], cells[0]))
        self.assertTrue(np.array_equal([1, 2, 7, 6], cells[1]))
        self.assertTrue(np.array_equal([2, 3, 8, 7], cells[2]))
        self.assertTrue(np.array_equal([3, 4, 9, 8], cells[3]))
        self.assertTrue(np.array_equal([5, 6, 11, 10], cells[4]))
        self.assertTrue(np.array_equal([6, 7, 12, 11], cells[5]))
        self.assertTrue(np.array_equal([7, 8, 13, 12], cells[6]))
        self.assertTrue(np.array_equal([8, 9, 14, 13], cells[7]))
        self.assertTrue(np.array_equal([10, 11, 16, 15], cells[8]))
        self.assertTrue(np.array_equal([11, 12, 17, 16], cells[9]))
        self.assertTrue(np.array_equal([12, 13, 18, 17], cells[10]))
        self.assertTrue(np.array_equal([13, 14, 19, 18], cells[11]))
        self.assertTrue(np.array_equal([15, 16, 21, 20], cells[12]))
        self.assertTrue(np.array_equal([16, 17, 22, 21], cells[13]))
        self.assertTrue(np.array_equal([17, 18, 23, 22], cells[14]))
        self.assertTrue(np.array_equal([18, 19, 24, 23], cells[15]))

        i_vertex = 0
        for i_y in range(5):
            for i_x in range(5):
                self.assertTrue(np.array_equal([0.25*i_x, 0.25*i_y], vertices[i_vertex]))
                i_vertex += 1
