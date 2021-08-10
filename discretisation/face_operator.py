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
# filename: face_operator.py
# description: todo
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import limiter as lm


# ----------------------------------------------------------------------------------------------------------------------
# Face Reconstruction Operators
# ----------------------------------------------------------------------------------------------------------------------


def muscl(edge_tangent, edge_length, phi_0, phi_1, gradient_phi_0, gradient_phi_1):
    """
    Computes the upwind MUSCL scheme provided a
    :param edge_tangent:
    :param edge_length:
    :param phi_0:
    :param phi_1:
    :param gradient_phi_0:
    :param gradient_phi_1:
    :return:
    """

    delta_phi = phi_1 - phi_0
    edge_vector = edge_length * edge_tangent
    delta_ph0 = 2 * np.dot(gradient_phi_0[:], edge_vector) - delta_phi
    delta_ph1 = 2 * np.dot(gradient_phi_1[:], edge_vector) - delta_phi
    return np.array(
        [phi_0 + 0.5 * lm.van_albada(delta_phi / delta_ph0) * delta_ph0 if np.abs(delta_ph0) > 1.0e-11 else phi_0,
         phi_1 - 0.5 * lm.van_albada(delta_phi / delta_ph1) * delta_ph1 if np.abs(delta_ph1) > 1.0e-11 else phi_1])


def arithmetic_mean(phi_0, phi_1):
    return 0.5 * (phi_0 + phi_1)


def construct_gauss_green_coefficient_matrix(cell_table, face_table, vertex_table):
    """
    Constructs the coefficient matrix for computing gauss green vertex based approach.

    nabla phi_i = 1/V_i phi_ij phi_jk 0.5 c_k phi_ki'/|r_ki'| nA_j
    where
    phi - scalar variable
    n - face normal
    A - face area
    c_k = (sum_i' 1/|r_ki'|)^-1
    r - displacement vector
    i - cell (i' connected cell to vertex k. i' also includes ghost cells)
    j - face
    k - vertex

    Note: Einstein notation.

    :param cell_table:
    :param face_table:
    :param vertex_table:
    :return:
    """
    guass_green_coefficient_matrix =\
        np.zeros(shape=((cell_table.max_cell)*2, cell_table.max_cell + cell_table.max_ghost_cell), dtype=float)

    # Pre-compute c_k
    vertex_cell_coefficient = np.zeros(shape=vertex_table.max_vertex, dtype=float)
    for ivertex in range(vertex_table.max_vertex):
        for icell in vertex_table.connected_cell[ivertex]:
            vertex_cell_coefficient[ivertex] += 1.0/np.linalg.norm(vertex_table.coordinate[ivertex] - cell_table.coordinate[icell])
        vertex_cell_coefficient[ivertex] = 1.0/vertex_cell_coefficient[ivertex]

    # Construct Coefficient Matrix.
    for iface in range(face_table.max_face):
        cell0 = face_table.connected_cell[iface][0]
        cell1 = face_table.connected_cell[iface][1]
        for ivertex in face_table.connected_vertex[iface]:
            for istencil_cell in vertex_table.connected_cell[ivertex]:
                distance_interpolation_coefficient = vertex_cell_coefficient[ivertex]*(1.0/np.linalg.norm(vertex_table.coordinate[ivertex] - cell_table.coordinate[istencil_cell]))

                # Rows only exist for real cells, so ignore 'ghost rows'
                if cell0 < cell_table.max_cell:
                    cell_coef = 0.5*distance_interpolation_coefficient/cell_table.volume[cell0]
                    guass_green_coefficient_matrix[2*cell0][istencil_cell] += cell_coef*face_table.coefficient[iface][0]
                    guass_green_coefficient_matrix[2*cell0 + 1][istencil_cell] += cell_coef*face_table.coefficient[iface][1]
                if cell1 < cell_table.max_cell:
                    cell_coef = 0.5 * distance_interpolation_coefficient / cell_table.volume[cell1]
                    guass_green_coefficient_matrix[2*cell1][istencil_cell] -= cell_coef*face_table.coefficient[iface][0]
                    guass_green_coefficient_matrix[2*cell1 + 1][istencil_cell] -= cell_coef*face_table.coefficient[iface][1]

    return guass_green_coefficient_matrix














