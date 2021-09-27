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

import numpy as np
from mesh_generation import mesh_generator as mg
from mesh_preprocessor import preprocessor as pp
from post_processor import graph as gr

[vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh([10, 10])
cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
phi_0 = np.random.rand(cell_centre_mesh.cell_table.max_cell, 2)
phi_1 = np.random.rand(cell_centre_mesh.cell_table.max_cell)
phi_2 = np.random.rand(cell_centre_mesh.cell_table.max_cell)
phi_3 = np.random.rand(cell_centre_mesh.cell_table.max_cell)
gr.plot_field(cell_centre_mesh, [phi_0, phi_1, phi_2, phi_3], gr.FieldPlotMetaData)
