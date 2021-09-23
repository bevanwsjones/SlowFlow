from gradient_algorithms import gridquality as gq
from gradient_algorithms import NewGG
from mesh_preprocessor import preprocessor as pp
from mesh_generation import mesh_generator as mg
import numpy as np
from gradient_algorithms import error_analysis as ea
from matplotlib import pyplot as plt
import functools as ft

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------- FACES & GRID QUALITY -----------------------------------------------------------------
cell_centroid, neighbour_centroid, face_unit_normal, face_centroid = np.array([2,2]), np.array([1,10]), \
                                                                      np.array([-0.5,1]), np.array([3,7])

# number_of_cells, start_co_ordinate, domain_size = [9, 9], [0.0, 0.0], [1.0, 1.0]
# [vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size, ft.partial(mg.stretch, [3.0, 3.0]))
# cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
# centroids = cell_centre_mesh.cell_table.centroid
# x, y = centroids.T
# plt.scatter(x, y)
# plt.show()




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

# h = gq.single_cell_grid_quality()
# print(h)
# print("here")
# ---------------------------------------------------------------------------------------------------------------------
# ------------------------------- Mesh Test ---------------------------------------------------------------------------
def error_plotter(int_error_array, bound_error_array, h):
    plt.scatter(h, int_error_array[:, 0, 0])
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    return -1



cells_matrix = np.array([[3, 3], [9, 9], [12, 12], [30, 30]])
def cartesian_error(cells_matrix):
    matrix_size = len(cells_matrix)
    size_store = np.empty(shape=(matrix_size, ))
    bound_error_array = np.empty(shape=(matrix_size, 3, 2))
    int_error_array =  np.empty(shape=(matrix_size, 3, 2))
    for i, i_matrix in enumerate(cells_matrix):
        size_store[i] = i_matrix[0]*i_matrix[1]
        print("For a", i_matrix, "grid:...")
        number_of_cells, start_co_ordinate, domain_size = i_matrix, [0.0, 0.0], [1.0, 1.0]
        [vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size)
        cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
        error = ea.cells_error_analysis(cell_centre_mesh)
        vol_table = cell_centre_mesh.cell_table.volume
        bound_error, int_error, bound_size, int_size, ext_vol_table, int_vol_table = ea.seperate_int_ext(cell_centre_mesh, error, vol_table)
        print("Boundary Cells Analysis:", bound_size, "external cells")
        norm_one_bound, norm_two_bound, norm_inf_bound = ea.error_package(bound_error, bound_size, ext_vol_table)
        #print(np.shape(norm_one_bound.T))
        bound_error_array[i][0], bound_error_array[i][1], bound_error_array[i][2] = norm_one_bound.T, norm_two_bound.T, norm_inf_bound.T
        print("Internal Cells Analysis:" ,int_size, "internal cells")
        norm_one_int, norm_two_int, norm_inf_int = ea.error_package(int_error, int_size, int_vol_table)
        int_error_array[i][0], int_error_array[i][1], int_error_array[i][2] = norm_one_int.T, norm_two_int.T, norm_inf_int.T
       #print(bound_error_array, int_error_array)
    h = size_store**(-0.5)
    error_plotter(int_error_array, bound_error_array, h)



g = cartesian_error(cells_matrix)




# ---------------------------------------- ERROR ANALYSIS TEST CODE ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# boundary_table = cell_centre_mesh.cell_table.boundary
# print(boundary_table)
# new_boundary_table = cell_centre_mesh.face_table.boundary
# print(new_boundary_table)
# for i, i_cell in enumerate(boundary_table):
#     print(i_cell)

# for i, i_cell in enumerate(k):
#     print(i, i_cell)
# print(NewGG.cell_phi_function(cell_centre_mesh))
# print(NewGG.boundary_phi_function(cell_centre_mesh))