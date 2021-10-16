from mesh_preprocessor import preprocessor as pp
from mesh_generation import mesh_generator as mg
import numpy as np
from gradient_algorithms import gridquality as gq
from gradient_algorithms import NewGG
from gradient_algorithms import error_analysis as ea
from gradient_algorithms import LSMethod as ls
from gradient_algorithms import results_processor as rp
import functools as ft

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------- FACES & GRID QUALITY -----------------------------------------------------------------
cell_centroid, neighbour_centroid, face_unit_normal, face_centroid = np.array([2,2]), np.array([1,10]), \
                                                                      np.array([-0.5,1]), np.array([3,7])
# x = gq.nonorthogonality(cell_centroid, neighbour_centroid, face_unit_normal)
# y = gq.unevenness(cell_centroid, neighbour_centroid, face_centroid)
# z = gq.skewness(cell_centroid, neighbour_centroid, face_centroid)
# print("Nonorthogonality", x)
# print("Unevenness", y)
# print("Skewness", z)
# cell_centroid = np.array([[2, 2], [5, 2], [1, 10], [-2, 4], [1, -4]])
# face_centroid = np.array([[4, 6], [3, 7], [1, 2], [1, 1]])
# face_normals = np.array([[1, 0], [-0.5, 1], [-1, 2], [-1, -1]])
# fc_connectivity = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
# l = gq.face_nonorthogonality(cell_centroid, face_normals, fc_connectivity)
# k = gq.faces_grid_quality(cell_centroid, face_normals, face_centroid, fc_connectivity)
# print(k)

# ---------------------------------------------------------------------------------------------------------------------
# ------------------------------- Mesh Test ---------------------------------------------------------------------------
number_of_cells, start_co_ordinate, domain_size = [10, 10], [0.0, 0.0], [1.0, 1.0]
#[vertex_coordinates, cell_vertex_connectivity, cell_type] = \
#      mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size,  ft.partial(mg.stretch, [0.8, 0.8]))
# [vertex_coordinates, cell_vertex_connectivity, cell_type] = \
#    mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size,  ft.partial(mg.parallelogram, False, [0.9, 0.9]))
#[vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size)
[vertex_coordinates, cell_vertex_connectivity, cell_type] = \
     mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size)
vertex_coordinates = mg.skew_strech(0.5, number_of_cells, vertex_coordinates)
cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
# print(cell_centre_mesh.cell_table.centroid)
# q = NewGG.GreenGauss(cell_centre_mesh, status = 1, phi_function = 0)
# print(q)
# h = ls.cell_ls(cell_centre_mesh, phi_function = 0)
# print(h)
quality_metrics = gq.cells_grid_quality(cell_centre_mesh)
vol_table = cell_centre_mesh.cell_table.volume
bound_qual, int_qual, bound_size, int_cell, ext_vol_table, int_vol_table = gq.seperate_int_ext(cell_centre_mesh, quality_metrics, vol_table)
print(int_qual[:, 1])
print(np.average(int_qual[:, 0]))


#print(quality_metrics)
#print(gq.grid_average_quality(quality_metrics, cell_centre_mesh))
# error = ea.cells_error_analysis(cell_centre_mesh)
# new_error = ea.seperate_int_ext(cell_centre_mesh, error)
#print(cell_centre_mesh.face_table.centroid)
#modGG.cell_phi_function(cell_centre_mesh)
# # l = gq.single_cell_grid_quality()
# k = gq.cells_grid_quality(cell_centre_mesh)
# print(k)
# h_1 = modGG.GreenGauss(cell_centre_mesh)
# print(h_1)
# h_2 = ea.cell_true_function(cell_centre_mesh)
# ---------------------------------------- ERROR ANALYSIS TEST CODE ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# vol_table = cell_centre_mesh.cell_table.volume
# error = ea.cells_error_analysis(cell_centre_mesh)
# ext_error, int_error, bound_cell_size, ext_vol_table, int_vol_table = ea.seperate_int_ext(cell_centre_mesh, error, vol_table)
# tot_cell = cell_centre_mesh.cell_table.max_cell
# int_cell_size = tot_cell - bound_cell_size
#
# print("Boundary Cells Analysis:", bound_cell_size, "external cells")
# g = ea.error_package(ext_error, bound_cell_size, ext_vol_table)
#
# print("Internal Cells Analysis:" ,int_cell_size, "internal cells")
# h = ea.error_package(int_error, int_cell_size, int_vol_table)


# norm_one = ea.grid_norm_one(error)
# norm_two = ea.grid_norm_two(error)
# norm_inf = ea.grid_norm_inf(error)
# print("Norm one", norm_one)
# print("Norm two", norm_two)
# print("Norm inf", norm_inf)
# norm_one_avg = ea.grid_avg_norm(cell_centre_mesh, norm_one)
# norm_two_avg = ea.grid_avg_norm(cell_centre_mesh, norm_two)
# print("Each Cell on average has a", norm_one_avg,"L1 norm")
# print("Each Cell on average has a", norm_two_avg,"L2 norm")
# # --------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------ALGORITHM PRINTING TEST-------------------------

# cell_centroids = cell_centre_mesh.cell_table.centroid
# cell_phi_function = NewGG.cell_phi_function(cell_centre_mesh)
# vertex_cell_connect = cell_centre_mesh.vertex_table.connected_cell
# max_vertex = cell_centre_mesh.vertex_table.max_vertex
# vertex_field = NewGG.vertex_phi(vertex_coordinates, cell_centroids, cell_phi_function, vertex_cell_connect, max_vertex)
# print(NewGG.node_GreenGauss(cell_centre_mesh))
# h_2 = ea.cell_true_function(cell_centre_mesh)
# print(h_2)
# print(vertex_field)
# print(cell_phi_function)

# print(cell_centre_mesh.cell_table.connected_face)
# ls.cell_ls(cell_centre_mesh)
# i_cell = 2
# face_cell_connect = [0, 2]
# print(ls.cell_face_neighbour(i_cell, face_cell_connect))