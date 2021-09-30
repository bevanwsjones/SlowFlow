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
from post_processor import graph as gr
from gradient_algorithms import LSMethod as ls
from gradient_algorithms import NewGG as gg
import numpy as np
from gradient_algorithms import error_analysis as ea

[vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh([3, 3])
cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
# gr.plot_field(cell_centre_mesh, gr.generate_random_field(4, cell_centre_mesh.cell_table.max_cell,
#                                                          [True, False, True, False]))

# print(ls.cell_ls(cell_centre_mesh))
# print(gg.GreenGauss(cell_centre_mesh, status = 0))
# error = ea.cells_error_analysis(cell_centre_mesh, 1)
# true_func = ea.cell_true_function(cell_centre_mesh)
# print(true_func)
# print(error)

cc_length = np.array(cell_centre_mesh.face_table.cc_length)
print(np.around(cc_length[:, None], 3))
cc_unit_vector = cell_centre_mesh.face_table.cc_unit_vector
print(np.around(cc_unit_vector, 3))
max_face = cell_centre_mesh.face_table.max_face
matrix = cc_length[:, None] * cc_unit_vector
print(np.around(matrix, 3))

#ls.ind_cell_LS()