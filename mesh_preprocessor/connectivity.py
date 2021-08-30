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
# filename: connectivity.py
# description: Creates the connectivity for simple meshes.
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Vertex connectivity
# ----------------------------------------------------------------------------------------------------------------------

def connect_vertices_to_cells(_cell_vertex_connectivity, _number_of_vertices):
    """
    Using the cell-vertex connectivity the vertex-cell connectivity is created. Once all connected cells have been found
    each row is sorted ascending.

    :param _cell_vertex_connectivity: Cell-vertex connectivity table, of the form [i_cell][list of vertices].
    :type _cell_vertex_connectivity: np.array
    :param _number_of_vertices: Number of vertices in the mesh.
    :type _number_of_vertices: int
    :return: The vertex-vertex connectivity table of the form [i_vertex][ascending list of vertex indices]
    :type: list
    """

    vertex_cell_connectivity = [np.empty(shape=(0,), dtype=int) for _ in range(_number_of_vertices)]
    for i_cell, cv_connectivity in enumerate(_cell_vertex_connectivity):
        for i_cv, i_vertex in enumerate(cv_connectivity):
            vertex_cell_connectivity[i_vertex] = np.append(vertex_cell_connectivity[i_vertex], i_cell)

    for cell_connectivity in vertex_cell_connectivity:
        np.sort(cell_connectivity)

    return vertex_cell_connectivity


def connect_vertices_to_vertices(_cell_vertex_connectivity, _number_of_vertices):
    """
    Using the cell vertex connectivity the direct neighbours of each vertex are determined. Once all connected vertices
    have been found each row is sorted ascending.

    :param _cell_vertex_connectivity: Cell-vertex connectivity table, of the form [i_cell][list of vertices].
    :type _cell_vertex_connectivity: np.array
    :param _number_of_vertices: Number of vertices in the mesh.
    :type _number_of_vertices: int
    :return: The vertex-vertex connectivity table of the form [i_vertex][ascending list of vertex indices]
    :type: list
    """

    vertex_vertex_connectivity = [np.empty(shape=(0,), dtype=int) for _ in range(_number_of_vertices)]
    for i_cell, cv_connectivity in enumerate(_cell_vertex_connectivity):
        for i_cv, i_vertex in enumerate(cv_connectivity):

            i_backward = i_cv - 1  # -1 wraps around
            if cv_connectivity[i_backward] not in vertex_vertex_connectivity[i_vertex]:
                vertex_vertex_connectivity[i_vertex] = \
                    np.append(vertex_vertex_connectivity[i_vertex], cv_connectivity[i_backward])

            i_forward = i_cv + 1 if i_cv + 1 < len(cv_connectivity) else 0
            if cv_connectivity[i_forward] not in vertex_vertex_connectivity[i_vertex]:
                vertex_vertex_connectivity[i_vertex] = \
                   np.append(vertex_vertex_connectivity[i_vertex], cv_connectivity[i_forward])

    for vertex_connectivity in vertex_vertex_connectivity:
        vertex_connectivity.sort()

    return vertex_vertex_connectivity


# ----------------------------------------------------------------------------------------------------------------------
# Face connectivity
# ----------------------------------------------------------------------------------------------------------------------


def compute_number_of_faces():
    return -1


def connect_faces_to_vertex(_cell_vertex_connectivity, _face_table):
    """
    Using the cell-vertex connectivity the face-vertex connectivity is created. It is assumed that the local cell-vertex
    connectivity is ordered in an anti-clockwise fashion. Using this assumption a 'new' face is found when the local
    'forward' vertex index is higher than the 'backward' vertex index.

    :param _cell_vertex_connectivity: Cell vertex connectivity, data should be [i_cell][i_connected_vertex]
    :type _cell_vertex_connectivity: np.array
    :param _face_table: The vertex table for which the vertex cell connectivity is to be built.
    :type _face_table: face.FaceTable
    """

    i_face = 0

    for i_cell, cv_connectivity in enumerate(_cell_vertex_connectivity):
        for i_cv, i_vertex in enumerate(cv_connectivity):
            if i_vertex > cv_connectivity[i_vertex - 1]:
                _face_table.connected_vertex[i_face][0] = cv_connectivity[i_vertex - 1]
                _face_table.connected_vertex[i_face][1] = i_vertex
                i_face += 1


def connect_faces_to_cells(_vertex_cell_connectivity, _vertex_table, _face_table):
    """
    todo

    :param _vertex_cell_connectivity: Vertex cell connectivity, data should be [i_vertex][i_connected_cell]
    :type _vertex_cell_connectivity: np.array
    :param _vertex_table: Vertex cell connectivity, data should be [i_vertex][i_connected_cell]
    :type _vertex_table: np.array
    :param _face_table: The vertex table for which the vertex cell connectivity is to be built.
    :type _face_table: face.FaceTable
    """

    for i_fv, fv_connectivity in _face_table.connected_vertex:
        vertex_0_cells = _vertex_table.connected_cell[fv_connectivity[0]]
        vertex_1_cells = _vertex_table.connected_cell[fv_connectivity[1]]

        for cell in vertex_0_cells:
            if len(np.where(vertex_1_cells, cell)) != 0:
                _face_table.connected_cell[i_fv][0 if _face_table.connected_cell[i_fv][0] != -1 else 1] = cell

    for fc in _face_table.connected_cell:
        np.sort(fc)


def determine_face_boundary_status(_face_cell_connectivity):
    """
    Determines the boundary status for each face, boundary faces are considered to be faces not connected to two cells.

    :param _face_cell_connectivity: Face-cell connectivity table, of the form [i_cell][list of vertices].
    :type _face_cell_connectivity: np.array
    """
    
    return np.array((_face_cell_connectivity[:, 1] == -1))


# ----------------------------------------------------------------------------------------------------------------------
# Cell connectivity
# ----------------------------------------------------------------------------------------------------------------------

def connect_cells_to_faces(_face_cell_connectivity):
    """
    todo

    :param _vertex_cell_connectivity: Vertex cell connectivity, data should be [i_vertex][i_connected_cell]
    :type _vertex_cell_connectivity: np.array
    :param _vertex_table: Vertex cell connectivity, data should be [i_vertex][i_connected_cell]
    :type _vertex_table: np.array
    :param _face_table: The vertex table for which the vertex cell connectivity is to be built.
    :type _face_table: face.FaceTable
    """


# ----------------------------------------------------------------------------------------------------------------------
# Mesh Connectivity-Preprocessing Simplex Mesh
# ----------------------------------------------------------------------------------------------------------------------

def connect_1D_mesh(mesh):
    return

    # sort boundaries to found of face table.

    # todo sort for boundaries

# TODO sorting:
# 1.
#
#
#
