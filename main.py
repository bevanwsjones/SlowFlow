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
from gradient_algorithms import NewGG
from gradient_algorithms import LSMethod as ls
from gradient_algorithms import error_analysis as ea
from gradient_algorithms import gridquality as gq
import functools as ft

number_of_cells, start_co_ordinate, domain_size = [3, 1], [0, 0], [1, 1]

# [vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size)
# s_factor = 0.8
# vertex_coordinates = mg.skew_strech(s_factor, number_of_cells, vertex_coordinates)
# cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
# gr.plot_field(cell_centre_mesh, gr.generate_random_field(1, cell_centre_mesh.cell_table.max_cell, [False]))


# quality_metrics = gq.cells_grid_quality(cell_centre_mesh)
# avg_quality = gq.grid_average_quality(quality_metrics, cell_centre_mesh)
# print(avg_quality)


# [vertex_coordinates, cell_vertex_connectivity, cell_type] = \
#     mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size,  ft.partial(mg.parallelogram, True, [0.0, 0.5]))
[vertex_coordinates, cell_vertex_connectivity, cell_type] = \
 mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size,  ft.partial(mg.stretch, [0.9, 0.9]))
# [vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size)
# vertex_coordinates = mg.skew_strech(0.15, number_of_cells, vertex_coordinates)
cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)


# NewGG.node_GreenGauss(cell_centre_mesh, phi_function = 0)

# field = NewGG.cell_boundary_face_phi_dphi_calculation(cell_centre_mesh, _phi_function = 0)
# print(field[2])
gr.plot_field(cell_centre_mesh, gr.generate_random_field(1, cell_centre_mesh.cell_table.max_cell, [False]))
# gr.plot_field(cell_centre_mesh, field[0], [False])

# quality_metrics = gq.cells_grid_quality(cell_centre_mesh)
# avg_quality = gq.grid_average_quality(quality_metrics, cell_centre_mesh.cell_table.volume)
# print(avg_quality)



# ls.cell_ls(cell_centre_mesh)
#
# error = ea.cells_error_analysis(cell_centre_mesh, met = 2)
# tot_cell = cell_centre_mesh.cell_table.max_cell
# vol_table = cell_centre_mesh.cell_table.volume
# bound_error, int_error, bound_size, int_size, ext_vol_table, int_vol_table = ea.seperate_int_ext(cell_centre_mesh, error, vol_table)
# # process the boundary error
# norm_one_bound, norm_rms_bound, norm_inf_bound = ea.error_package(bound_error, tot_cell, ext_vol_table)
# norm_one_int, norm_rms_int, norm_inf_int = ea.error_package(int_error, tot_cell, int_vol_table)
# print("Internal cells \n", norm_rms_int)
# print("External cells \n", norm_rms_bound)

# dist = np.array([[0, -0.333],[0, 0.333], [0.333, 0],[-0.333, 0]])
# cells_phi_neighbour = np.array([1.4656, 1.1521, 1.6175,  1.0435])
# cell_phi_centre = 1.3570
# print(ls.inv_cell(dist, cells_phi_neighbour, cell_phi_centre))
