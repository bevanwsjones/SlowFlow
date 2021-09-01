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
# Cell Geometry
# ----------------------------------------------------------------------------------------------------------------------
# Cell Centroids
# ----------------------------------------------------------------------------------------------------------------------

class CellCentroidTest(ut.TestCase):

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
        centre = (centre0 * area0 + centre1 * area1 + centre2 * area2 + centre3 * area3) / (
                    area0 + area1 + area2 + area3)
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


# ----------------------------------------------------------------------------------------------------------------------
# Cell Volume
# ----------------------------------------------------------------------------------------------------------------------

class CellVolumeTest(ut.TestCase):

    def test_calculate_edge_volume(self):

        vertex_coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
        cell_vertex_connectivity = np.array([[0, 1], [1, 2]])

        cell_volume = fv.calculate_edge_volume(cell_vertex_connectivity, vertex_coordinates)

        # Check lengths
        self.assertEqual(2, len(cell_volume))

        # Check values
        self.assertAlmostEqual(1.0, cell_volume[0])
        self.assertAlmostEqual(2.0, cell_volume[1])

    def test_calculate_2D_cell_volume_triangle(self):
        vertex_coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        cell_centroids = np.array([[2.0/3.0, 1.0/3.0]])
        face_normals = np.array([[0.0, -1.0], [1.0, 0.0], [-0.5*np.sqrt(2.0), 0.5*np.sqrt(2.0)]])
        face_areas = np.array([1.0, 1.0, np.sqrt(2.0)])
        cell_face_connectivity = np.array([[0, 1, 2]])
        face_cell_connectivity = np.array([[0, -1], [0, -1], [0, -1]])
        face_vertex_connectivity = np.array([[0, 1], [1, 2], [2, 0]])

        cell_volume = fv.calculate_2d_cell_volume(cell_face_connectivity, face_cell_connectivity,
                                                  face_vertex_connectivity, cell_centroids, face_normals, face_areas,
                                                  vertex_coordinates)

        # Check lengths
        self.assertEqual(1, len(cell_volume))

        # Check values
        self.assertAlmostEqual(0.5*1.0*1.0, cell_volume[0])

    def test_calculate_cell_volume_quadrilateral(self):

        vertex_coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]])
        cell_centroids = np.array([[0.5, 0.5], [1.5, 0.5]])
        face_normals = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0],
                                 [0.0, 1.0]])
        face_areas = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        cell_face_connectivity = np.array([[0, 1, 2, 3], [1, 4, 5, 6]])
        face_cell_connectivity = np.array([[0, -1], [0, 1], [0, -1], [0, -1], [1, -1], [1, -1], [1, -1]])
        face_vertex_connectivity = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [1, 4], [4, 5], [5, 2]])

        cell_volume = fv.calculate_2d_cell_volume(cell_face_connectivity, face_cell_connectivity,
                                                  face_vertex_connectivity, cell_centroids, face_normals, face_areas,
                                                  vertex_coordinates)

        # Check lengths
        self.assertEqual(2, len(cell_volume))

        # Check values
        self.assertAlmostEqual(1.0, cell_volume[0])
        self.assertAlmostEqual(1.0, cell_volume[1])


# ----------------------------------------------------------------------------------------------------------------------
# Face Geometry
# ----------------------------------------------------------------------------------------------------------------------

class FaceGeometryTest(ut.TestCase):

    def test_calculate_face_cell_cell_length(self):
        vertex_coordinates = np.array([[0.0, 0.0], [0.0, 1.0], [4.5, 3.0], [4.5, 4.0]])
        cell_centroids = np.array([[0.5, 0.5], [1.5, 1.5], [3.5, 3.5]])
        face_cell_connectivity = np.array([[0, -1], [2, -1], [0, 1], [1, 2]])
        face_vertex_connectivity = np.array([[0, 1], [2, 3], [-1, -1], [-1, -1]])

        face_length = fv.calculate_face_cell_cell_length(2, face_cell_connectivity, face_vertex_connectivity,
                                                         cell_centroids, vertex_coordinates)

        # Check lengths
        self.assertEqual(4, len(face_length))

        # Check values
        self.assertAlmostEqual(0.5, face_length[0])
        self.assertAlmostEqual(1.0, face_length[1])
        self.assertAlmostEqual(np.sqrt(2.0*(1.5 - 0.5) ** 2.0), face_length[2])
        self.assertAlmostEqual(np.sqrt(2.0*(3.5 - 1.5) ** 2.0), face_length[3])

    def test_calculate_face_cell_cell_unit_vector(self):
        vertex_coordinates = np.array([[0.0, 0.0], [0.0, 1.0], [4.5, 3.0], [4.5, 4.0]])
        cell_centroids = np.array([[0.5, 0.5], [1.5, 1.5], [3.5, 3.5]])
        face_cell_connectivity = np.array([[0, -1], [2, -1], [0, 1], [1, 2]])
        face_vertex_connectivity = np.array([[0, 1], [2, 3], [-1, -1], [-1, -1]])

        face_tangent = fv.calculate_face_cell_cell_unit_vector(2, face_cell_connectivity, face_vertex_connectivity,
                                                               cell_centroids, vertex_coordinates)

        #Check lengths
        self.assertEqual(4, len(face_tangent))
        self.assertEqual(2, len(face_tangent[0]))
        self.assertEqual(2, len(face_tangent[1]))
        self.assertEqual(2, len(face_tangent[2]))
        self.assertEqual(2, len(face_tangent[3]))

        # Check values
        self.assertAlmostEqual(-1.0, face_tangent[0][0])
        self.assertAlmostEqual(0.0, face_tangent[0][1])
        self.assertAlmostEqual(1.0, face_tangent[1][0])
        self.assertAlmostEqual(0.0, face_tangent[1][1])
        self.assertAlmostEqual(np.sqrt(2.0)/2.0, face_tangent[2][0])
        self.assertAlmostEqual(np.sqrt(2.0)/2.0, face_tangent[2][1])
        self.assertAlmostEqual(np.sqrt(2.0)/2.0, face_tangent[3][0])
        self.assertAlmostEqual(np.sqrt(2.0)/2.0, face_tangent[3][1])

    def test_calculate_face_area_1D(self):
        vertex_coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [2, 0]])
        face_vertex_connectivity = np.array([[0, 1], [1, 2]])
        face_area = fv.calculate_face_area(cl.CellType.edge, face_vertex_connectivity, vertex_coordinates)

        # Check lengths
        self.assertEqual(2, len(face_area))

        # Check values
        self.assertAlmostEqual(1.0, face_area[0])
        self.assertAlmostEqual(1.0, face_area[1])

    def test_calculate_face_area_2D(self):
        vertex_coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
        face_vertex_connectivity = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        face_area = fv.calculate_face_area(cl.CellType.quadrilateral, face_vertex_connectivity, vertex_coordinates)

        # Check lengths
        self.assertEqual(4, len(face_area))

        # Check values
        self.assertAlmostEqual(1.0, face_area[0])
        self.assertAlmostEqual(np.sqrt((2.0 - 1.0) ** 2.0 + (0.0 - 1.0) ** 2.0), face_area[1])
        self.assertAlmostEqual(2.0, face_area[2])
        self.assertAlmostEqual(1.0, face_area[3])

    def test_calculate_face_normal(self):
        vertex_coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
        face_vertex_connectivity = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        face_normals = fv.calculate_face_normal(cl.CellType.quadrilateral, face_vertex_connectivity, vertex_coordinates)

        # Check lengths
        self.assertEqual(4, len(face_normals))
        self.assertEqual(2, len(face_normals[0]))
        self.assertEqual(2, len(face_normals[1]))
        self.assertEqual(2, len(face_normals[2]))
        self.assertEqual(2, len(face_normals[3]))

        # Check values
        self.assertAlmostEqual(0.0, face_normals[0][0])
        self.assertAlmostEqual(-1.0, face_normals[0][1])
        self.assertAlmostEqual(np.sqrt(2.0) / 2.0, face_normals[1][0])
        self.assertAlmostEqual(-np.sqrt(2.0) / 2.0, face_normals[1][1])
        self.assertAlmostEqual(0.0, face_normals[2][0])
        self.assertAlmostEqual(1.0, face_normals[2][1])
        self.assertAlmostEqual(-1.0, face_normals[3][0])
        self.assertAlmostEqual(0.0, face_normals[3][1])
