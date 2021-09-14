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
# filename: main.py
# description: Program entry point, hello world.
# ----------------------------------------------------------------------------------------------------------------------

from mesh_generation import mesh_generator as mg
from mesh_preprocessor import preprocessor as pp

from gradient_algorithms import GreenGauss as gg

[vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh([3, 3], [0.5, 1.5], [3, 3])
cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)

number_of_cells, start_co_ordinate, domain_size = [3, 3], [0.5, 1.5], [3, 3]
test = gg.GreenGauss_neighbourcells(number_of_cells, start_co_ordinate, domain_size)
face = gg.facecenter_coords(number_of_cells, start_co_ordinate, domain_size)
#print(test)
#print(face)

test2 = gg.GreenGauss_2D(number_of_cells, start_co_ordinate, domain_size)
#print(test2)
#print(cell_centre_mesh.face_table.normal)

# [vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_1d_mesh(5, 0.0, 1.0)
# print(cell_type)
# print("face-cell connectivity table \n", cell_centre_mesh.face_table.connected_cell)
# print("cell-face connectivity table \n", cell_centre_mesh.cell_table.connected_face)