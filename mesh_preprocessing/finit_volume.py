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
# description: todo
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np

def setup_ghost_cell_geometry(cell_table, face_table, vertex_table):

    # loop over faces and link up the new ghost nodes
    for iface in range(face_table.max_face):
        if face_table.boundary[iface]:
            ivertex0 = face_table.connected_vertex[iface][0]
            ivertex1 = face_table.connected_vertex[iface][1]
            vect_cell_vertex = vertex_table.coordinate[ivertex0] \
                               - cell_table.coordinate[face_table.connected_cell[iface, 0]]
            face_vector = vertex_table.coordinate[ivertex1] - vertex_table.coordinate[ivertex0]
            face_vector /= np.linalg.norm(face_vector)
            vector_to_face = vect_cell_vertex - np.dot(vect_cell_vertex, face_vector) * face_vector

            cell_table.coordinate[face_table.connected_cell[iface, 1]] = \
                cell_table.coordinate[face_table.connected_cell[iface, 0]] + 2.0 * vector_to_face

def setup_finite_volume_geometry(cell_table, face_table, vertex_table):

    # Create cell geometry
    max_cell = cell_table.max_cell

    cell_table.coordinate[0:max_cell] = (vertex_table.coordinate[cell_table.connected_vertex[0:max_cell, 0]]
                                         + vertex_table.coordinate[cell_table.connected_vertex[0:max_cell, 1]]
                                         + vertex_table.coordinate[cell_table.connected_vertex[0:max_cell, 2]]) / 3.0
    cell_table.volume[0:max_cell] =\
        0.5*np.cross(vertex_table.coordinate[cell_table.connected_vertex[0:max_cell, 1]]
                     - vertex_table.coordinate[cell_table.connected_vertex[0:max_cell, 0]],
                     vertex_table.coordinate[cell_table.connected_vertex[0:max_cell, 2]]
                     - vertex_table.coordinate[cell_table.connected_vertex[0:max_cell, 0]])
    setup_ghost_cell_geometry(cell_table, face_table, vertex_table)

    # Create Face Geometry
    face_table.length[:] = np.linalg.norm(cell_table.coordinate[face_table.connected_cell[:, 1]]
                                          - cell_table.coordinate[face_table.connected_cell[:, 0]], axis=1)
    face_table.tangent[:] = (cell_table.coordinate[face_table.connected_cell[:, 1]]
                             - cell_table.coordinate[face_table.connected_cell[:, 0]])/face_table.length[:, None]
    face_normal = np.delete(np.cross(np.concatenate((vertex_table.coordinate[face_table.connected_vertex[:, 1]],
                                                     np.zeros(shape=(face_table.max_face, 1))), axis=1)
                                     - np.concatenate((vertex_table.coordinate[face_table.connected_vertex[:, 0]],
                                                       np.zeros(shape=(face_table.max_face, 1))), axis=1),
                                     np.full((face_table.max_face, 3), [0.0, 0.0, 1.0])), 2, axis=1)
    face_normal = face_normal[:]/np.linalg.norm(face_normal[:], axis=1)[:, None]
    face_table.coefficient =\
        np.linalg.norm(vertex_table.coordinate[face_table.connected_vertex[:, 1]]
                       - vertex_table.coordinate[face_table.connected_vertex[:, 0]], axis=1)[:, None]*face_normal[:]
