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
from mesh import cell as cl


# ----------------------------------------------------------------------------------------------------------------------
# Vertex connectivity
# ----------------------------------------------------------------------------------------------------------------------

def connect_vertices_to_cells(_cell_vertex_connectivity, _number_of_vertices):
    """
    Using the cell-vertex connectivity the vertex-cell connectivity is created. Once all connected cells have been found
    each row is sorted ascending.

    :param _cell_vertex_connectivity: Cell-vertex connectivity table, of the form [i_cell][list of vertices].
    :type _cell_vertex_connectivity: numpy.array
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
    :type _cell_vertex_connectivity: numpy.array
    :param _number_of_vertices: Number of vertices in the mesh.
    :type _number_of_vertices: int
    :return: The vertex-vertex connectivity table of the form [i_vertex][ascending list of vertices]
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

def compute_number_of_faces(_vertex_cell_connectivity, _cell_vertex_connectivity):
    """
    Computes the number of faces and boundary faces in the mesh, creates a temporary face_vertex_connectivity table and
    counts duplicates to determine the numbers.

    :param _vertex_cell_connectivity: Vertex-cell connectivity table, of the form [i_vertex][list of cells].
    :type _vertex_cell_connectivity: numpy.array
    :param _cell_vertex_connectivity: Cell-vertex connectivity table, of the form [i_cell][list of vertices].
    :type _cell_vertex_connectivity: numpy.array
    :return: [number of faces, number of boundary faces]
    :type: [int, int]
    """

    face_vertex_connectivity = np.empty(shape=[0, 2], dtype=int)
    for cv_connectivity in _cell_vertex_connectivity:
        for i_cv, i_vertex in enumerate(cv_connectivity):
            face_vertex_connectivity = np.append(face_vertex_connectivity,
                                                 [[i_vertex, cv_connectivity[i_cv - 1]]], axis=0)
            face_vertex_connectivity[-1].sort()

    number_of_boundary_faces = len(np.unique(face_vertex_connectivity))
    number_of_faces = int((len(face_vertex_connectivity) - number_of_boundary_faces)/2 + number_of_boundary_faces)
    return [number_of_faces, number_of_boundary_faces]


def connect_faces_to_vertex(_cell_vertex_connectivity):
    """
    Using the cell-vertex connectivity the face-vertex connectivity is created. The table is returned such that faces
    are sorted so that boundary faces are first.

    Note: It is important to used this function before other faces connectivities are built, since it generates the
          order correctly in regard to having boundary faces first.

    :param _cell_vertex_connectivity: Cell-vertex connectivity table, of the form [i_cell][list of vertices].
    :type _cell_vertex_connectivity: numpy.array
    :return Face-vertex connectivity table, of the form [i_face][ascending list of vertices].
    :type numpy.array

    todo: There are multiple sorts and memory allocations, may need some optimising in the future.
    """

    face_vertex_connectivity = np.empty(shape=[0, 2], dtype=int)

    for cv_connectivity in _cell_vertex_connectivity:
        for i_cv, i_vertex in enumerate(cv_connectivity):
            face_vertex_connectivity = np.append(face_vertex_connectivity,
                                                 [[i_vertex, cv_connectivity[i_cv - 1]]], axis=0)
            face_vertex_connectivity[-1].sort()

    # Each row is already sorted, so we now go to sort the entire list by column 0, the column 1,
    # Then delete all duplicates (both of them), but record them, they are the internal faces. The remaining list are
    # boundaries then we re-append the interior faces back onto the face_vertex_connectivity.
    # I not claiming this efficient.
    face_vertex_connectivity = face_vertex_connectivity[face_vertex_connectivity[:, 0].argsort()]
    face_vertex_connectivity = face_vertex_connectivity[face_vertex_connectivity[:, 1].argsort(kind='mergesort')]
    delete_indices = np.empty(shape=[0], dtype=int)
    interior_faces = np.empty(shape=[0, 2], dtype=int)
    for iface, vertex_connectivity in enumerate(face_vertex_connectivity):
        if np.array_equal(face_vertex_connectivity[iface - 1], vertex_connectivity):
            delete_indices = np.append(delete_indices, iface)
            delete_indices = np.append(delete_indices, iface - 1)
            interior_faces = np.append(interior_faces, [vertex_connectivity], axis=0)
    face_vertex_connectivity = np.delete(face_vertex_connectivity, delete_indices, axis=0)
    face_vertex_connectivity = np.append(face_vertex_connectivity, interior_faces, axis=0)

    return face_vertex_connectivity


def connect_faces_to_cells(_vertex_cell_connectivity, _face_vertex_connectivity):
    """
    Using the face-vertex connectivity the common cells for the vertices attached to each face can be found. These are
    the connected cells to each face. For boundary cell only 1 cell is present, as such the second index in the face-
    cell connectivity table is -1 for boundary faces.

    :param _vertex_cell_connectivity: Vertex-cell connectivity table, of the form [i_vertex][list of cells].
    :type _vertex_cell_connectivity: numpy.array
    :param _face_vertex_connectivity: Face-vertex connectivity table, of the form [i_face][list of vertices].
    :type _face_vertex_connectivity: numpy.array
    :return Face-cell connectivity table, of the form [i_face][ascending list of cells].
    :type numpy.array
    """

    face_cell_connectivity = -1*np.ones(shape=(len(_face_vertex_connectivity), 2), dtype=int)
    for i_fv, face_connectivity in enumerate(_face_vertex_connectivity):
        i_cell = np.intersect1d(_vertex_cell_connectivity[face_connectivity[0]],
                                _vertex_cell_connectivity[face_connectivity[1]])

        if len(i_cell) == 1:
            face_cell_connectivity[i_fv][0] = i_cell[0]
        elif len(i_cell) == 2:
            face_cell_connectivity[i_fv] = i_cell
        else:
            raise RuntimeError("Vertex " + str(face_connectivity[0]) + " and " + str(face_connectivity[1])
                               + " do not share a face.")

    for face_connectivity in face_cell_connectivity:
        if face_connectivity[1] != -1:
            face_connectivity.sort()

    return face_cell_connectivity


def determine_face_boundary_status(_face_cell_connectivity):
    """
    Determines the boundary status for each face, boundary faces are considered to be faces not connected to two cells.

    :param _face_cell_connectivity: Face-cell connectivity table, of the form [i_cell][list of vertices].
    :type _face_cell_connectivity: numpy.array
    :return List of booleans, true if the face is boundary else false.
    :type numpy.array
    """

    return np.array((_face_cell_connectivity[:, 1] == -1))


# ----------------------------------------------------------------------------------------------------------------------
# Cell connectivity
# ----------------------------------------------------------------------------------------------------------------------

def connect_cells_to_faces(_face_cell_connectivity, _number_of_cells, _cell_type):
    """
    Using the face-cell connectivity the cell face connectivity is constructed.

    :param _face_cell_connectivity: Face-cell connectivity table, of the form [i_face][ascending list of cells].
    :type _face_cell_connectivity: numpy.array
    :param _number_of_cells: number of cells in the mesh.
    :type _number_of_cells: integer
    :param _cell_type: The type of cell in the mesh, edge, triangle, etc
    :type _cell_type: mesh.cell.CellType
    :return: Face-cell connectivity table, of the form [i_cell][list of faces].
    :type: numpy.array

    todo: I am sure again there is a better way to do this.
    """
    cell_face_connectivity = -1*np.ones(shape=[_number_of_cells, cl.number_of_vertex_face(_cell_type)], dtype=int)
    for i_face, connected_cells in enumerate(_face_cell_connectivity):
        for icv, i_cell_face in enumerate(cell_face_connectivity[connected_cells[0]]):
            if i_cell_face == -1:
                cell_face_connectivity[connected_cells[0]][icv] = i_face
                break
        if connected_cells[1] != -1:
            for icv, i_cell_face in enumerate(cell_face_connectivity[connected_cells[1]]):
                if i_cell_face == -1:
                    cell_face_connectivity[connected_cells[1]][icv] = i_face
                    break

    return cell_face_connectivity

def determine_cell_boundary_status(_face_cell_connectivity, _face_boundry_status):
    """
    todo
    :param _face_cell_connectivity:
    :param _face_boundry_status:
    :return:
    """
    return -1