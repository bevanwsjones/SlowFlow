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
from mesh import cell as ct, face as ft, vertex as vt


# ----------------------------------------------------------------------------------------------------------------------
# Vertex connectivity
# ----------------------------------------------------------------------------------------------------------------------

def connect_vertices_to_cells(_cell_vertex_connectivity, _vertex_table):
    """
    Using the cell-vertex connectivity the vertex-cell connectivity is created.

    :param _cell_vertex_connectivity: Cell vertex connectivity, data should be [i_cell][i_connected_vertex]
    :type _cell_vertex_connectivity: np.array
    :param _vertex_table: The vertex table for which the vertex cell connectivity is to be built.
    :type _vertex_table: vertex.VertexTable
    """

    for i_cell, cv_connectivity in enumerate(_cell_vertex_connectivity):
        for i_cv, i_vertex in enumerate(cv_connectivity):
            _vertex_table.connected_cell[i_vertex] = np.append(_vertex_table.connected_cell[i_vertex], i_cell)

    for vc in _vertex_table.connected_cell:
        vc = np.sort(vc)


def connect_vertices_to_vertices(_cell_vertex_connectivity, _vertex_table):
    """
    todo

    :param _cell_vertex_connectivity: Cell vertex connectivity, data should be [i_cell][i_connected_vertex]
    :type _cell_vertex_connectivity: np.array
    :param _vertex_table: The vertex table for which the vertex cell connectivity is to be built.
    :type _vertex_table: vertex.VertexTable
    """

    for i_cell, cv_connectivity in enumerate(_cell_vertex_connectivity):
        for i_cv, i_vertex in enumerate(cv_connectivity):

            if len(np.where(_vertex_table.connected_vertex[i_vertex], cv_connectivity[i_cv - 1])) == 0:
                _vertex_table.connected_cell[i_vertex] = \
                    np.append(_vertex_table.connected_cell[i_vertex],  cv_connectivity[i_cv - 1])

            if len(np.where(_vertex_table.connected_vertex[i_vertex], cv_connectivity[i_cv + 1])) == 0:
                _vertex_table.connected_cell[i_vertex] = \
                    np.append(_vertex_table.connected_cell[i_vertex],  cv_connectivity[i_cv + 1])

    for vv in _vertex_table.connected_vertex:
        vv = np.sort(vv)

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
        fc = np.sort(fc)


def determine_face_boundary_status(_face_table):
    """
    todo

    :param _face_table: The vertex table for which the vertex cell connectivity is to be built.
    :type _face_table: face.FaceTable
    """

    for i_face in range(_face_table.connected_cell):
        _face_table.boundary[i_face] = _face_table.connected_cell[i_face][1] == -1

# ----------------------------------------------------------------------------------------------------------------------
# Cell connectivity
# ----------------------------------------------------------------------------------------------------------------------

def connect_cells_to_faces(_face_cell_connectivity, _cell_table):
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
# 1D Mesh Connectivity-Preprocessing Simplex Mesh
# ----------------------------------------------------------------------------------------------------------------------

def connect_1D_mesh(mesh):
    return

    # todo sort for boundaries




# ----------------------------------------------------------------------------------------------------------------------
# 2D Mesh Connectivity-Preprocessing Simplex Mesh
# ----------------------------------------------------------------------------------------------------------------------


def find_face_connected_cells(icell, vertex0_connected_cells, vertex1_connected_cells):
    """
    Determines the cell connected to icell given the list of cells connected to two vertices of one of the faces of
    icell. By looking for common cells in vertex0's and vertex1's connectivity the cell connected to icell can be found.
    The return value is two integers indicating the two cells connected through the common face. If no connection is
    found it is assumed that this cell face is a boundary cell face, in which case both values in the returned array are
    icell.

    :param icell: The cell index.
    :param vertex0_connected_cells: The list of cells connected to vertex 0 of the face of the cell.
    :param vertex1_connected_cells: The list of cells connected to vertex 1 of the face the cell.
    :return: [icell, iconnected cell] if internal face else [icell, icell].
    """

    cells_connected_to_face = np.empty(shape=2, dtype=int)
    is_connection_found = False

    for icell_vertex0 in vertex0_connected_cells:
        if icell_vertex0 != icell:
            connected_cell = np.where(vertex1_connected_cells == icell_vertex0)[0]

            # If the returned array is not of zero size we have found a connection.
            if len(connected_cell) != 0:
                if len(connected_cell) != 1:
                    raise ValueError("More than one face found when connecting cells.")

                cells_connected_to_face = np.array([icell, vertex1_connected_cells[connected_cell[0]]])
                is_connection_found = True

        # This is a boundary face - mark by connecting the cell to itself.
        if icell_vertex0 == vertex0_connected_cells[-1] and not is_connection_found:
            cells_connected_to_face = np.array([icell, icell])

    return cells_connected_to_face


def create_face_cell_vertex_connectivity(cells):
    """
    Given a list of cells computes the face-cell and face-vertex connectivity tables.

    :param cells: List of cells, each row is should be a list of the vertex indices.
    :return: face-cell connectivity table, face-vertex connectivity table.
    """

    face_cell_connectivity = []
    face_vertex_connectivity = []
    for icell, cell in enumerate(cells):

        for ivert in range(len(cell)):
            vert0_cell_connection = np.where(cells == cell[ivert])[0]
            vert1_cell_connection = np.where(cells == cell[(ivert + 1) % len(cell)])[0]
            face_cell_connection = find_face_connected_cells(icell, vert0_cell_connection, vert1_cell_connection)
            if face_cell_connection[0] <= face_cell_connection[1]:
                face_cell_connectivity.append(face_cell_connection)
                face_vertex_connectivity.append(np.array([cell[ivert], cell[(ivert + 1) % len(cell)]]))
    face_cell_connectivity = np.asarray(face_cell_connectivity)
    return face_cell_connectivity, face_vertex_connectivity


def add_ghost_cell(cell_table, face_table, vertex_table):
    """
    Adds ghost cells to the cell table and updates the connectivity tables for the cell, face, and vertex tables. The
    ghost cells are appended to the end of the cell table.

    :param cell_table: Cell table onto which the ghost cells will be added and connectivity updated.
    :param face_table: Face table which will be connected to the added ghost cells.
    :param vertex_table: Vertex table which will be connected to the added ghost cells.
    """

    unique, counts = np.unique(face_table.boundary, return_counts=True)
    ghost_node_start = cell_table.max_cell
    if unique[1] == True:
        cell_table.add_ghost_cells(counts[1])
    else:
        raise ValueError("No boundary faces found.")

    # loop over faces and link up the new ghost nodes
    for iface in range(face_table.number_of_face):
        if face_table.boundary[iface]:
            ivertex0 = face_table.connected_vertex[iface][0]
            ivertex1 = face_table.connected_vertex[iface][1]
            face_table.connected_cell[iface][1] = ghost_node_start
            cell_table.connected_face[ghost_node_start][0] = iface
            cell_table.connected_vertex[ghost_node_start][0] = ivertex0
            cell_table.connected_vertex[ghost_node_start][1] = ivertex1
            vertex_table.connected_cell[ivertex0] = np.concatenate(
                (vertex_table.connected_cell[ivertex0], np.array([ghost_node_start])))
            vertex_table.connected_cell[ivertex1] = np.concatenate(
                (vertex_table.connected_cell[ivertex1], np.array([ghost_node_start])))
            ghost_node_start += 1


def setup_connectivity(cells, verticies):
    """
    Sizes the data containers to describe a cell-centred Finite-Volume mesh and creates connectivity structures
    describing the local neighbourhood of a given mesh entity (cell, face, vertex). Additionally ghost nodes are added
    and boundary status resolved.

    :return: The constructed cell, face, and vertex tables.
    """

    face_cell_connectivity, face_vertex_connectivity = create_face_cell_vertex_connectivity(cells)

    # Size tables
    vertex_table = vt.VertexTable(len(verticies))
    cell_table = ct.CellTable(len(cells))
    face_table = ft.FaceTable(len(face_cell_connectivity))

    # Construct cell face connectivity.
    for iface in range(face_table.max_face):
        face_table.connected_cell[iface][0] = face_cell_connectivity[iface][0]
        face_table.connected_cell[iface][1] = face_cell_connectivity[iface][1]
        face_table.connected_vertex[iface][0] = face_vertex_connectivity[iface][0]
        face_table.connected_vertex[iface][1] = face_vertex_connectivity[iface][1]
        face_table.boundary[iface] = face_table.connected_cell[iface][0] == face_table.connected_cell[iface][1]

        inode0 = face_table.connected_cell[iface][0]
        inode1 = face_table.connected_cell[iface][1]

        cell_table.connected_face[inode0][np.where(cell_table.connected_face[inode0] == -1)[0][0]] = iface
        if not face_table.boundary[iface]:
            cell_table.connected_face[inode1][np.where(cell_table.connected_face[inode1] == -1)[0][0]] = iface
        else:
            cell_table.boundary[face_table.connected_cell[iface][0]] = True

    # Build the vertex_cell connectivity and copy the cell_vertex connectivity
    temp = [[] for _ in range(len(verticies))]
    for icell, cell in enumerate(cells):
        for ivertex in cell:
            temp[ivertex].append(icell)
        cell_table.connected_vertex[icell] = np.asarray(cell)

    # Copy data to the vertex table.
    for ivertex, vertex in enumerate(verticies):
        vertex_table.coordinate[ivertex] = np.array([vertex])
        vertex_table.connected_cell[ivertex] = np.asarray(temp[ivertex])

    # Add the ghost cells.
    add_ghost_cell(cell_table, face_table, vertex_table)

    print("WARNING: no cell first/last has been set up yet.")
    return cell_table, face_table, vertex_table
