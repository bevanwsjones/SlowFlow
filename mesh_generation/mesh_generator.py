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
# filename: mesh_generator.py
# description: todo
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from compressible_solver import face as fc
from compressible_solver import node as nd
from compressible_solver import vertex as vt
import meshio
import dmsh
import optimesh

# ----------------------------------------------------------------------------------------------------------------------
# 1D Mesh Generation
# ----------------------------------------------------------------------------------------------------------------------


def setup_1d_unit_mesh(max_cell):
    """

    :param max_cell:
    :return:
    """

    # Size tables
    vertex_table = vt.VertexTable((max_cell + 1)*2)
    face_table = fc.FaceTable(max_cell + 1)
    cell_table = nd.CellTable(max_cell)
    cell_table.add_ghost_cells(2)
    cell_table.connected_face = np.delete(cell_table.connected_face, 2, 1)
    delta_x = 1.0/float(max_cell)

    # Setup vertex connectivity and co-ordinates
    for ivertex in range(int(vertex_table.max_vertex/2)):
        ivertex0 = 2 * ivertex
        ivertex1 = 2 * ivertex + 1
        vertex_table.coordinate[ivertex0] = np.array([float(ivertex) * delta_x, 0.0])
        vertex_table.coordinate[ivertex1] = np.array([float(ivertex) * delta_x, 1.0])

        if ivertex == 0:
            # index to the index of the first cell and first ghost cell
            vertex_table.connected_cell[ivertex0] = np.array([0, cell_table.max_cell])
            vertex_table.connected_cell[ivertex1] = np.array([0, cell_table.max_cell])
        elif ivertex == (int(vertex_table.max_vertex/2) - 1):
            # connect to the index of the last cell and last ghost cell
            vertex_table.connected_cell[ivertex0] = np.array([cell_table.max_cell - 1, cell_table.max_cell + 1])
            vertex_table.connected_cell[ivertex1] = np.array([cell_table.max_cell - 1, cell_table.max_cell + 1])
        else:
            vertex_table.connected_cell[ivertex0] = np.array([ivertex - 1, ivertex])
            vertex_table.connected_cell[ivertex1] = np.array([ivertex - 1, ivertex])

    # Setup face connectivity and coefficients
    for iface in range(face_table.max_face):
        face_table.connected_vertex[iface] = np.array([iface * 2, iface * 2 + 1])
        face_table.length[iface] = delta_x

        if iface == 0:
            face_table.boundary[iface] = True
            face_table.connected_cell[iface] = np.array([0, cell_table.max_cell])  # index to the first ghost cell
            face_table.coefficient[iface] = np.array([-1.0, 0.0])
        elif iface == (face_table.max_face - 1):
            face_table.boundary[iface] = True
            face_table.connected_cell[iface] = np.array([cell_table.max_cell - 1, cell_table.max_cell + 1])  # index to last ghost cell
            face_table.coefficient[iface] = np.array([1.0, 0.0])
        else:
            face_table.boundary[iface] = False
            face_table.connected_cell[iface] = np.array([iface - 1, iface])
            face_table.coefficient[iface] = np.array([1.0, 0.0])

    # Setup ghost cell connectivity
    for icell in range(cell_table.max_cell):
        cell_table.connected_face[icell] = np.array([icell, icell + 1])
        cell_table.volume[icell] = delta_x
        cell_table.coordinate[icell] = np.array([float(icell * 2 + 1) * 0.5 * delta_x, 0.5])

    # Hard code values for ghost cells.
    cell_table.connected_face[cell_table.max_cell] = np.array([0, -1])
    cell_table.volume[cell_table.max_cell] = delta_x
    cell_table.coordinate[cell_table.max_cell] = np.array([-0.5 * delta_x, 0.5])
    cell_table.connected_face[cell_table.max_cell + 1] = np.array([face_table.max_face - 1, -1])
    cell_table.volume[cell_table.max_cell + 1] = delta_x
    cell_table.coordinate[cell_table.max_cell + 1] = np.array([1 + 0.5 * delta_x, 0.5])

    return cell_table, face_table, vertex_table

# ----------------------------------------------------------------------------------------------------------------------
# 2D Mesh Generation - Simplex Mesh
# ----------------------------------------------------------------------------------------------------------------------

def setup_2D_simplex_mesh():

    geo = dmsh.Polygon([[0.0, 0.0], [1.0, 0.0], [1.0 + np.cos(np.pi/3.0), np.sin(np.pi/3.0)],
                        [1.0, 2.0*np.sin(np.pi/3.0)], [0.0, 2.0*np.sin(np.pi/3.0)], [0.0 - np.cos(np.pi/3.0), np.sin(np.pi/3.0)]])
    verticies, cells = dmsh.generate(geo, 0.5)
    dmsh.helpers.show(verticies, cells, geo)  # Put on to print mesh

    return  verticies, cells

# ----------------------------------------------------------------------------------------------------------------------
# 2D Mesh Generation - Cartesian Mesh
# ----------------------------------------------------------------------------------------------------------------------

def setup_2D_cartesian_mesh(domain_size, number_cells):

    cells = np.zeros([number_cells[0]*number_cells[1], 4], dtype=int)
    vertex = np.zeros([(number_cells[0] + 1)*(number_cells[1] + 1), 2], dtype=float)

    cell_size = [(domain_size[1][0] - domain_size[0][0])/number_cells[0],
                 (domain_size[1][1] - domain_size[0][1])/number_cells[1]]

    ivert = 0
    print(domain_size[0])
    print([cell_size[0], cell_size[1]])
    dX = np.array((cell_size[0], cell_size[1]))
    print((domain_size[0] + dX))
    for irow in range(number_cells[1] + 1):
        for icolu in range(number_cells[0] + 1):
            vertex[ivert] = np.array((domain_size[0] + [cell_size[0]*icolu, cell_size[1]*irow]))
            ivert += 1

    print(vertex)
















