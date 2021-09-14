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
# description: Contains unit tests for the mesh generator.
# ----------------------------------------------------------------------------------------------------------------------

from mesh_generation import mesh_generator as mg
import unittest as ut
import mesh.cell as cl


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
        [verticies, cells, cell_type] = mg.setup_2d_cartesian_mesh([2, 4], _ratio=[1.5, 1.0])

        print(verticies)
        for c_type in cell_type:
            self.assertEqual(c_type, cl.CellType.quadrilateral)
