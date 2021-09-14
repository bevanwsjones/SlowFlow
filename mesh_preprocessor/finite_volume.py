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
# filename: finite_volume.py
# description: Contains functions for computing the the geometric properties of a finite volume mesh.
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from mesh import cell as cl


# ----------------------------------------------------------------------------------------------------------------------
# Cell Geometry
# ----------------------------------------------------------------------------------------------------------------------
# Cell Central Co-Ordinate
# ----------------------------------------------------------------------------------------------------------------------

def calculate_edge_centroid(_cell_vertex_connectivity, _vertex_coordinates):
    """
    Computes the centroid of an edge cell, half way done.

    :param _cell_vertex_connectivity: Cell-vertex connectivity table, of the form [i_cell][list of vertices].
    :type _cell_vertex_connectivity: numpy.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :type _vertex_coordinates: numpy.array
    :return: List of the cell centroids of the form [i_cell][x, y coordinates].
    :type: numpy.array
    """""

    max_cell = len(_cell_vertex_connectivity)
    return (_vertex_coordinates[_cell_vertex_connectivity[0:max_cell, 0]]
            + _vertex_coordinates[_cell_vertex_connectivity[0:max_cell, 1]]) / 2.0


def calculate_triangle_centroid(_cell_vertex_connectivity, _vertex_coordinates):
    """
    Computes the centroid of an triangle, third of the sum of the co-ordinates.

    :param _cell_vertex_connectivity: Cell-vertex connectivity table, of the form [i_cell][list of vertices].
    :type _cell_vertex_connectivity: numpy.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :type _vertex_coordinates: numpy.array
    :return: List of the cell centroids of the form [i_cell][x, y coordinates].
    :type: numpy.array
    """""

    max_cell = len(_cell_vertex_connectivity)
    return (_vertex_coordinates[_cell_vertex_connectivity[0:max_cell, 0]]
            + _vertex_coordinates[_cell_vertex_connectivity[0:max_cell, 1]]
            + _vertex_coordinates[_cell_vertex_connectivity[0:max_cell, 2]]) / 3.0


def calculate_quadrilateral_centroid(_cell_vertex_connectivity, _vertex_coordinates):
    """
    Computes the centroid of a non-intersecting quadrilateral.

    The method for computing the centroid is as follows a set of four triangles are formed. Each triangle is formed by
    connecting three consecutive vertices in the quadrilateral. The centroids of these triangles form the co-ordinate
     set, [[x_0, y_0],[x_1, y_1],[x_2, y_2],[x_3, y_3]]. The centroid is then the intersection of the lines given by
     [X_4] = [X_2] + lambda_0 ([X_0] - [X_2]) | lambda_0 in [0, 1],
     [X_5] = [x_3] + lambda_1 ([X_1] - [X_3]) | lambda_1 in [0, 1],
     and where X denotes [x, y]. The intersection of these two lines is given by:

          || x_0 y_0 |  x_0 - x_2 |
          || x_2 y_2 |            |
          || x_1 y_1 |  x_1 - x_3 |
    x_c = || x_3 y_3 |            | ,
          -------------------------
          | x_0 - x_2   y_0 - y_2 |
          | x_1 - x_3   y_1 - y_3 |

          || x_0 y_0 |  y_0 - y_2 |
          || x_2 y_2 |            |
          || x_1 y_1 |  y_1 - y_3 |
    y_c = || x_3 y_3 |            | .
          -------------------------
          | x_0 - x_2   y_0 - y_2 |
          | x_1 - x_3   y_1 - y_3 |

    For further reading see https://mathworld.wolfram.com/Line-LineIntersection.html

    :param _cell_vertex_connectivity: Cell-vertex connectivity table, of the form [i_cell][list of vertices].
    :type _cell_vertex_connectivity: numpy.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :type _vertex_coordinates: numpy.array
    :return: List of the cell centroids of the form [i_cell][x, y coordinates].
    :type: numpy.array
    """

    triangle_centre_0 = calculate_triangle_centroid(_cell_vertex_connectivity[:, [0, 1, 2]], _vertex_coordinates)
    triangle_centre_1 = calculate_triangle_centroid(_cell_vertex_connectivity[:, [1, 2, 3]], _vertex_coordinates)
    triangle_centre_2 = calculate_triangle_centroid(_cell_vertex_connectivity[:, [2, 3, 0]], _vertex_coordinates)
    triangle_centre_3 = calculate_triangle_centroid(_cell_vertex_connectivity[:, [3, 0, 1]], _vertex_coordinates)

    numerator_det_02 = np.linalg.det(np.stack((triangle_centre_0[:, ], triangle_centre_2[:, ]), axis=1))
    numerator_det_13 = np.linalg.det(np.stack((triangle_centre_1[:, ], triangle_centre_3[:, ]), axis=1))
    numerator_x_02 = triangle_centre_0[:, 0] - triangle_centre_2[:, 0]
    numerator_x_13 = triangle_centre_1[:, 0] - triangle_centre_3[:, 0]
    numerator_y_02 = triangle_centre_0[:, 1] - triangle_centre_2[:, 1]
    numerator_y_13 = triangle_centre_1[:, 1] - triangle_centre_3[:, 1]
    numerator_x = np.linalg.det(np.stack([np.column_stack([numerator_det_02[:, ], numerator_x_02[:, ]]),
                                          np.column_stack([numerator_det_13[:, ], numerator_x_13[:, ]])], axis=1))
    numerator_y = np.linalg.det(np.stack([np.column_stack([numerator_det_02[:, ], numerator_y_02[:, ]]),
                                          np.column_stack([numerator_det_13[:, ], numerator_y_13[:, ]])], axis=1))
    denominator = np.linalg.det(np.stack(((triangle_centre_0[:] - triangle_centre_2[:]),
                                          (triangle_centre_1[:] - triangle_centre_3[:])), axis=1))
    return np.stack([numerator_x[:]/denominator[:], numerator_y[:]/denominator[:]], axis=1)


def calculate_hexagon_centroid(_cell_vertex_connectivity, _vertex_coordinates):
    """
    Computes the centroid of a non-intersecting hexagon. TODO

    :param _cell_vertex_connectivity: Cell-vertex connectivity table, of the form [i_cell][list of vertices].
    :type _cell_vertex_connectivity: numpy.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :type _vertex_coordinates: numpy.array
    :return: List of the cell centroids of the form [i_cell][x, y coordinates].
    :type: numpy.array
    """

    raise RuntimeError("Still a todo")


def calculate_cell_centroid(_cell_type, _cell_vertex_connectivity, _vertex_coordinates):
    """
    Computes the centroid of a cell based on the type of cell, edge, triangle, etc.

    :param _cell_type: The type of cells described by the _cell_vertex_connectivity.
    :type _cell_type: mesh.cell.CellType
    :param _cell_vertex_connectivity: Cell-vertex connectivity table, of the form [i_cell][list of vertices].
    :type _cell_vertex_connectivity: numpy.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :type _vertex_coordinates: numpy.array
    :return: List of the cell centroids of the form [i_cell][x, y coordinates].
    :type: numpy.array
    """

    if _cell_type.value == cl.CellType.edge.value:
        return calculate_edge_centroid(_cell_vertex_connectivity, _vertex_coordinates)
    elif _cell_type.value == cl.CellType.triangle.value:
        return calculate_triangle_centroid(_cell_vertex_connectivity, _vertex_coordinates)
    elif _cell_type.value == cl.CellType.quadrilateral.value:
        return calculate_quadrilateral_centroid(_cell_vertex_connectivity, _vertex_coordinates)
    elif _cell_type.value == cl.CellType.hexagon.value:
        return calculate_hexagon_centroid(_cell_vertex_connectivity, _vertex_coordinates)
    else:
        raise RuntimeError("Unsupported cell type, cannot compute centroid.")


# ----------------------------------------------------------------------------------------------------------------------
# Cell Volume
# ----------------------------------------------------------------------------------------------------------------------

def calculate_edge_volume(_cell_vertex_connectivity, _vertex_coordinates):
    """
    Computes the volume for edge cells, the distance between the vertices.

    :param _cell_vertex_connectivity: Cell-vertex connectivity table, of the form [i_cell][list of vertices].
    :type _cell_vertex_connectivity: numpy.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :type _vertex_coordinates: numpy.array
    :return: List of the cell volumes of the form [i_cell][volume].
    :type: numpy.array
    """
    return np.linalg.norm(_vertex_coordinates[_cell_vertex_connectivity[:, 0]]
                          - _vertex_coordinates[_cell_vertex_connectivity[:, 1]], axis=1)


def calculate_2d_cell_volume(_cell_face_connectivity, _face_cell_connectivity, _face_vertex_connectivity,
                             _cell_centroids, _face_normals, _face_areas, _vertex_coordinates):
    """
    Computes the volume for non-edge cells using the divergence theorem,

    |omega_i| = sum_(f in partial omega)  0.5 x_f . |partial omega_f| n_f,

    where |omega_i| is the cell volume, |partial omega_f| is the face area, n_f is the outward pointing face normal, and
    x_f is the vector from the cell centroid to the face centroid.

    :param _cell_face_connectivity: Cell-face connectivity table, of the form [i_cell][list of face].
    :type _cell_face_connectivity: numpy.array
    :param _face_cell_connectivity: Face-cell connectivity table, of the form [i_face][list of cells].
    :type _face_cell_connectivity: numpy.array
    :param _face_vertex_connectivity: Face-vertex connectivity table, of the form [i_face][list of vertices].
    :type _face_vertex_connectivity: numpy.array
    :param _cell_centroids: Co-ordinates for all cell centroids in the mesh, of the form [i_cell][x, y coordinates]
    :type _cell_centroids: numpy.array
    :param _face_normals: The normal vector for each face in the mesh, of the form [i_face][x, y coordinates]
    :type _face_normals: numpy.array
    :param _face_areas: The area for each face in the mesh, of the form [i_face_area]
    :type _face_areas: numpy.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :type _vertex_coordinates: numpy.array
    :return: List of the cell volumes of the form [i_cell][volume].
    :type: numpy.array
    """

    volume = np.zeros(shape=(len(_cell_face_connectivity),), dtype=float)
    for i_cell, cell_faces in enumerate(_cell_face_connectivity):
        for face in cell_faces:
            face_vector = (0.5 * (_vertex_coordinates[_face_vertex_connectivity[face][0]]
                                  + _vertex_coordinates[_face_vertex_connectivity[face][1]])
                           - _cell_centroids[i_cell])
            volume[i_cell] += ((0.5 if i_cell == _face_cell_connectivity[face][0] else -0.5)
                               * np.dot(face_vector, _face_normals[face] * _face_areas[face]))
    return volume


# ----------------------------------------------------------------------------------------------------------------------
# Face Geometry
# ----------------------------------------------------------------------------------------------------------------------

def calculate_face_centroid(_face_vertex_connectivity, _vertex_coordinates):
    """
    Computes the face centroid, which is the mid-point between the two attached face vertices.

    :param _face_vertex_connectivity: Face-vertex connectivity table, of the form [i_face][list of vertices].
    :type _face_vertex_connectivity: numpy.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :type _vertex_coordinates: numpy.array
    :return: List of the face centroids.
    :type: numpy.array
    """

    return 0.5*(_vertex_coordinates[_face_vertex_connectivity[:, 0]]
                + _vertex_coordinates[_face_vertex_connectivity[:, 1]])


def calculate_face_cell_cell_length(_number_boundary_face, _face_cell_connectivity, _face_vertex_connectivity,
                                    _cell_centroid, _vertex_coordinates):
    """
    Computes the length between two cell centroids for each face. If the face is a boundary computes the length between
    the cell centroid and the face centre. Assumes all boundary faces are at the front of the connectivity tables.

    :param _number_boundary_face: number of boundary faces in the mesh.
    :type _number_boundary_face: int
    :param _face_cell_connectivity: Face-cell connectivity table, of the form [i_face][list of cells].
    :type _face_cell_connectivity: numpy.array
    :param _face_vertex_connectivity: Face-vertex connectivity table, of the form [i_face][list of vertices].
    :type _face_vertex_connectivity: numpy.array
    :param _cell_centroid: Co-ordinates for all cell centroids in the mesh, of the form [i_cell][x, y coordinates]
    :type _cell_centroid: numpy.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :type _vertex_coordinates: numpy.array
    :return: List of the face normals of the form [i_face][cell centroid to cell centroid length].
    :type: numpy.array
    """

    return np.concatenate((
        np.linalg.norm(_cell_centroid[_face_cell_connectivity[:_number_boundary_face, 0]]
                       - 0.5 * (_vertex_coordinates[_face_vertex_connectivity[:_number_boundary_face, 1]]
                                + _vertex_coordinates[_face_vertex_connectivity[:_number_boundary_face, 0]]), axis=1),
        np.linalg.norm(_cell_centroid[_face_cell_connectivity[_number_boundary_face:, 1]]
                       - _cell_centroid[_face_cell_connectivity[_number_boundary_face:, 0]], axis=1)))


def calculate_face_cell_cell_unit_vector(_number_boundary_face, _face_cell_connectivity, _face_vertex_connectivity,
                                         _cell_centroid, _vertex_coordinates):
    """
    Computes the cell centroid to cell centroid unit vectors. The vector points from the local cell index 0 to the local
    cell index 1. If the face is a boundary the vector points from the cell centre to the face centre.

    :param _number_boundary_face: number of boundary faces in the mesh.
    :type _number_boundary_face: int
    :param _face_cell_connectivity: Face-cell connectivity table, of the form [i_face][list of cells].
    :type _face_cell_connectivity: numpy.array
    :param _face_vertex_connectivity: Face-vertex connectivity table, of the form [i_face][list of vertices].
    :type _face_vertex_connectivity: numpy.array
    :param _cell_centroid: Co-ordinates for all cell centroids in the mesh, of the form [i_cell][x, y coordinates]
    :type _cell_centroid: numpy.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :type _vertex_coordinates: numpy.array
    :return: List of the face cell-cell unit vectors of the form [i_face][cell centroid to cell centroid unit vector].
    :type: numpy.array
    """

    return np.array([vector / np.linalg.norm(vector) for vector in
                     np.concatenate((0.5 * (_vertex_coordinates[_face_vertex_connectivity[:_number_boundary_face, 1]]
                                            + _vertex_coordinates[_face_vertex_connectivity[:_number_boundary_face, 0]])
                                     - _cell_centroid[_face_cell_connectivity[:_number_boundary_face, 0]],
                                     _cell_centroid[_face_cell_connectivity[_number_boundary_face:, 1]]
                                     - _cell_centroid[_face_cell_connectivity[_number_boundary_face:, 0]]))])


def calculate_face_area(_cell_type, _face_vertex_connectivity, _vertex_coordinates):
    """
    Constructs face areas, computed as the unity in 1D (edge elements) and as the distance between the two face vertices
    in 2D.

    :param _cell_type: The type of cells described by the _cell_vertex_connectivity.
    :type _cell_type: mesh.cell.CellType
    :param _face_vertex_connectivity: Face-vertex connectivity table, of the form [i_face][list of vertices].
    :type _face_vertex_connectivity: numpy.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :type _vertex_coordinates: numpy.array
    :return: List of the face normals of the form [i_face][x, y normal].
    :type: numpy.array
    """

    if _cell_type == cl.CellType.edge:
        return np.ones(len(_face_vertex_connectivity))
    else:
        return np.linalg.norm(_vertex_coordinates[_face_vertex_connectivity[:, 1]]
                              - _vertex_coordinates[_face_vertex_connectivity[:, 0]], axis=1)


def calculate_face_normal(_cell_type, _face_vertex_connectivity, _vertex_coordinates, _face_cell_cell_unit_vector):
    """
    Constructs face normals, are orientated assuming that face are constructed by walking anti-clockwise around cells.

    :param _cell_type: The type of cells described by the _cell_vertex_connectivity.
    :type _cell_type: mesh.cell.CellType
    :param _face_vertex_connectivity: Face-vertex connectivity table, of the form [i_face][list of vertices].
    :type _face_vertex_connectivity: numpy.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :type _vertex_coordinates: numpy.array
    :param _face_cell_cell_unit_vector: Cell centroid to centroid unit vector [i_face][x, y unit vector].
    :type _face_cell_cell_unit_vector: numpy.array
    :return: List of the face normals of the form [i_face][x, y normal].
    :type: numpy.array
    """

    if _cell_type == cl.CellType.edge:
        raise RuntimeError("still to do")
    else:
        face_normal = (
            np.array([vector / np.linalg.norm(vector)
                      for vector in np.cross(_vertex_coordinates[_face_vertex_connectivity[:, 1]]
                                             - _vertex_coordinates[_face_vertex_connectivity[:, 0]],
                                             np.array((0, 0, 1), dtype=float))[:, 0:2]])
        )

        for i_face, normal in enumerate(face_normal):
            if np.dot(normal, _face_cell_cell_unit_vector[i_face]) < 0.0:
                normal *= -1.0
        return face_normal
