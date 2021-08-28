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
# Cell Central Co-Ordinate
# ----------------------------------------------------------------------------------------------------------------------

def calculate_edge_centroid(_cell_vertex_connectivity, _vertex_coordinates):
    """
    Computes the centroid of an edge cell, half way done.
    
    :param _cell_vertex_connectivity: Cell-vertex connectivity table, of the form [i_cell][list of vertices].
    :param _cell_vertex_connectivity: np.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :param _vertex_coordinates: np.array
    :return: List of the cell centroids of the form [i_cell][x, y coordinates].
    :type: np.array
    """""

    max_cell = len(_cell_vertex_connectivity)
    return (_vertex_coordinates[_cell_vertex_connectivity[0:max_cell, 0]]
            + _vertex_coordinates[_cell_vertex_connectivity[0:max_cell, 1]]) / 2.0


def calculate_triangle_centroid(_cell_vertex_connectivity, _vertex_coordinates):
    """
    Computes the centroid of an triangle, third of the sum of the co-ordinates.

    :param _cell_vertex_connectivity: Cell-vertex connectivity table, of the form [i_cell][list of vertices].
    :param _cell_vertex_connectivity: np.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :param _vertex_coordinates: np.array
    :return: List of the cell centroids of the form [i_cell][x, y coordinates].
    :type: np.array
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
     [X_4] = [X_2] + lambda_0 ([X_0] - [X_2]) | lambda_0 \in [0, 1],
     [X_5] = [x_3] + lambda_1 ([X_1] - [X_3]) | lambda_1 \in [0, 1],
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
    :param _cell_vertex_connectivity: np.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :param _vertex_coordinates: np.array
    :return: List of the cell centroids of the form [i_cell][x, y coordinates].
    :type: np.array
    """

    triangle_centre_0 = calculate_triangle_centroid(_cell_vertex_connectivity[:, [0, 1, 2]], _vertex_coordinates)
    triangle_centre_1 = calculate_triangle_centroid(_cell_vertex_connectivity[:, [1, 2, 3]], _vertex_coordinates)
    triangle_centre_2 = calculate_triangle_centroid(_cell_vertex_connectivity[:, [2, 3, 0]], _vertex_coordinates)
    triangle_centre_3 = calculate_triangle_centroid(_cell_vertex_connectivity[:, [3, 0, 1]], _vertex_coordinates)

    return np.stack((np.linalg.det(np.stack((
        np.column_stack((np.linalg.det(np.stack((triangle_centre_0[:, ], triangle_centre_2[:, ]), axis=1)[:]),
                         (triangle_centre_0[:, 0] - triangle_centre_2[:, 0])))[:],
        np.column_stack((np.linalg.det(np.stack((triangle_centre_1[:, ], triangle_centre_3[:, ]), axis=1)[:]),
                         (triangle_centre_1[:, 0] - triangle_centre_3[:, 0])))[:]), axis=1)),
                     np.linalg.det(np.stack((np.column_stack((np.linalg.det(
                         np.stack((triangle_centre_0[:, ], triangle_centre_2[:, ]), axis=1)[:]),
                                                              (triangle_centre_0[:, 1] - triangle_centre_1[:, 1])))[:],
                                             np.column_stack((np.linalg.det(
                                                 np.stack((triangle_centre_1[:, ], triangle_centre_3[:, ]), axis=1)[:]),
                                                              (triangle_centre_1[:, 1] - triangle_centre_3[:, 1])))[:]),
                                            axis=1))), axis=1)[:] \
           / np.linalg.det(
        np.stack(((triangle_centre_0 - triangle_centre_2)[:], (triangle_centre_1 - triangle_centre_3)[:]), axis=1))[:]


def calculate_hexagon_centroid(_cell_vertex_connectivity, _vertex_coordinates):
    """
    Computes the centroid of a non-intersecting hexagon. TODO

    :param _cell_vertex_connectivity: Cell-vertex connectivity table, of the form [i_cell][list of vertices].
    :param _cell_vertex_connectivity: np.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :param _vertex_coordinates: np.array
    :return: List of the cell centroids of the form [i_cell][x, y coordinates].
    :type: np.array
    """

    raise RuntimeError("Still a todo")


def calculate_cell_centroid(_cell_type, _cell_vertex_connectivity, _vertex_coordinates):
    """
    Computes the centroid of a cell based on the type of cell, edge, triangle, etc.

    :param _cell_type: The type of cells described by the _cell_vertex_connectivity.
    :type _cell_type: mesh.cell.CellType
    :param _cell_vertex_connectivity: Cell-vertex connectivity table, of the form [i_cell][list of vertices].
    :param _cell_vertex_connectivity: np.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :param _vertex_coordinates: np.array
    :return: List of the cell centroids of the form [i_cell][x, y coordinates].
    :type: np.array
    """

    if _cell_type == cl.CellType.edge:
        return calculate_edge_centroid(_cell_vertex_connectivity, _vertex_coordinates)
    elif _cell_type == cl.CellType.triangle:
        return calculate_triangle_centroid(_cell_vertex_connectivity, _vertex_coordinates)
    elif _cell_type == cl.CellType.quadrilateral:
        return calculate_quadrilateral_centroid(_cell_vertex_connectivity, _vertex_coordinates)
    elif _cell_type == cl.CellType.hexagon:
        return calculate_hexagon_centroid(_cell_vertex_connectivity, _vertex_coordinates)


# ----------------------------------------------------------------------------------------------------------------------
# Cell Volume
# ----------------------------------------------------------------------------------------------------------------------

def calculate_edge_volume(_cell_table, _vertex_table):
    max_cell = _cell_table.max_cell
    _cell_table.volume[0:max_cell] = (_vertex_table.coordinate[_cell_table.connected_vertex[0:max_cell, 0]]
                                      - _vertex_table.coordinate[_cell_table.connected_vertex[0:max_cell, 1]])


def calculate_triangle_volume():
    raise RuntimeError("Still a todo")


def calculate_quadrilateral_volume():
    raise RuntimeError("Still a todo")


def calculate_hexagon_volume():
    raise RuntimeError("Still a todo")


def calculate_cell_volume(cell_table, vertex_table):
    raise RuntimeError("Still a todo")


# ----------------------------------------------------------------------------------------------------------------------
# Face Geometry
# ----------------------------------------------------------------------------------------------------------------------

def calculate_face_cell_cell_length(_number_boundary_face, _face_cell_connectivity, _face_vertex_connectivity,
                                    _cell_centroid, _vertex_coordinates):
    """
    Computes the length between two cell centroids for each face. If the face is a boundary computes the length between
    the cell centroid and the face centre. Assumes all boundary faces are at the front of the connectivity tables.

    :param _number_boundary_face: number of boundary faces in the mesh.
    :type _number_boundary_face: int
    :param _face_cell_connectivity: Face-cell connectivity table, of the form [i_face][list of cells].
    :type _face_cell_connectivity: np.array
    :param _face_vertex_connectivity: Face-vertex connectivity table, of the form [i_face][list of vertices].
    :type _face_vertex_connectivity: np.array
    :param _cell_centroid: Co-ordinates for all cell centroids in the mesh, of the form [i_cell][x, y coordinates]
    :param _cell_centroid: np.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :param _vertex_coordinates: np.array
    :return: List of the face normals of the form [i_face][cell centroid to cell centroid length].
    :type: np.array
    """

    return np.concatenate((
        np.linalg.norm(_cell_centroid[_face_cell_connectivity[:_number_boundary_face, 0]]
                       - 0.5 * (_vertex_coordinates[_face_vertex_connectivity[:_number_boundary_face, 1]]
                                + _vertex_coordinates[_face_vertex_connectivity[:_number_boundary_face, 0]]), axis=1),
        np.linalg.norm(_cell_centroid[_face_cell_connectivity[_number_boundary_face:, 1]]
                       - _cell_centroid[_face_cell_connectivity[_number_boundary_face:, 0]], axis=1)))


def calculate_face_cell_cell_tangent(_number_boundary_face, _face_cell_connectivity, _face_vertex_connectivity,
                                     _cell_centroid, _vertex_coordinates):
    """


    :param _number_boundary_face: number of boundary faces in the mesh.
    :type _number_boundary_face: int
    :param _face_cell_connectivity: Face-cell connectivity table, of the form [i_face][list of cells].
    :type _face_cell_connectivity: np.array
    :param _face_vertex_connectivity: Face-vertex connectivity table, of the form [i_face][list of vertices].
    :type _face_vertex_connectivity: np.array
    :param _cell_centroid: Co-ordinates for all cell centroids in the mesh, of the form [i_cell][x, y coordinates]
    :param _cell_centroid: np.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :param _vertex_coordinates: np.array
    :return: List of the face normals of the form [i_face][cell centroid to cell centroid length].
    :type: np.array
    """

    return np.array([vector / np.linalg.norm(vector) for vector in
                     np.concatenate((0.5 * (_vertex_coordinates[_face_vertex_connectivity[:_number_boundary_face, 1]]
                                            + _vertex_coordinates[_face_vertex_connectivity[:_number_boundary_face, 0]])
                                     - _cell_centroid[_face_cell_connectivity[:_number_boundary_face, 0]],
                                     _cell_centroid[_face_cell_connectivity[_number_boundary_face:, 1]]
                                     - _cell_centroid[_face_cell_connectivity[_number_boundary_face:, 0]]))])


def calculate_face_area(_cell_type, _face_vertex_connectivity, _vertex_coordinates):
    """
    Constructs face areas, computed as the unity in 1D (edge elements) and as the distance between the two faec vertices
    in 2D.

    :param _cell_type: The type of cells described by the _cell_vertex_connectivity.
    :type _cell_type: mesh.cell.CellType
    :param _face_vertex_connectivity: Face-vertex connectivity table, of the form [i_face][list of vertices].
    :type _face_vertex_connectivity: np.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :param _vertex_coordinates: np.array
    :return: List of the face normals of the form [i_face][x, y normal].
    :type: np.array
    """

    if _cell_type == cl.CellType.edge:
        return np.ones(len(_face_vertex_connectivity))
    else:
        return np.linalg.norm(_vertex_coordinates[_face_vertex_connectivity[:, 1]]
                              - _vertex_coordinates[_face_vertex_connectivity[:, 0]], axis=1)


def calculate_face_normal(_cell_type, _face_vertex_connectivity, _vertex_coordinates):
    """
    Constructs face normals, are orientated assuming that face are constructed by walking anti-clockwise around cells.

    :param _cell_type: The type of cells described by the _cell_vertex_connectivity.
    :type _cell_type: mesh.cell.CellType
    :param _face_vertex_connectivity: Face-vertex connectivity table, of the form [i_face][list of vertices].
    :type _face_vertex_connectivity: np.array
    :param _vertex_coordinates: Co-ordinates for all vertices in the mesh, of the form [i_vertex][x, y coordinates]
    :param _vertex_coordinates: np.array
    :return: List of the face normals of the form [i_face][x, y normal].
    :type: np.array
    """

    if _cell_type == cl.CellType.edge:
        raise RuntimeError("still to do")
    else:
        return np.array([vector / np.linalg.norm(vector)
                         for vector in np.cross(_vertex_coordinates[_face_vertex_connectivity[:, 1]]
                                                - _vertex_coordinates[_face_vertex_connectivity[:, 0]],
                                                np.array((0, 0, 1), dtype=float))[:, 0:2]])


# ----------------------------------------------------------------------------------------------------------------------
# finite volume setup
# ----------------------------------------------------------------------------------------------------------------------

def setup_finite_volume_geometry(_mesh):
    return
    # calculate_cell_center(_mesh.cell_table, _mesh.vertex_table)
    #
    # calculate_face_cell_cell_length(_mesh.face_table, _mesh.cell_table.coordinate)
    # calculate_face_cell_cell_tangent(_mesh.face_table, _mesh.cell_table.coordinate)
    # calculate_face_area(_mesh.face_table, _mesh.vertex_table.vertex_coordinate)
    # calculate_face_normal(_mesh.face_table, _mesh.vertex_table.vertex_coordinate)
    #
    # calculate_cell_volume(_mesh.cell_table, _mesh.vertex_table)
