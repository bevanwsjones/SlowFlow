from gradient_algorithms import gridquality as gq
from gradient_algorithms import NewGG
from mesh_preprocessor import preprocessor as pp
from mesh_generation import mesh_generator as mg
import numpy as np
from gradient_algorithms import error_analysis as ea
from matplotlib import pyplot as plt
import functools as ft
from gradient_algorithms import errorplotter as ep


# ---------------------------------------------------------------------------------------------------------------------
# ------------------------------- CARTESIAN GRID PLOTTER & RESULTS PROCESSOR ------------------------------------------
# Error Plotter for Cartesian Grid Test
def cartesian_error(cells_matrix):
    matrix_size = len(cells_matrix)
    size_store = np.empty(shape=(matrix_size, ))
    bound_error_array = np.empty(shape=(matrix_size, 3, 2))
    int_error_array =  np.empty(shape=(matrix_size, 3, 2))
    for i, i_matrix in enumerate(cells_matrix):
        size_store[i] = i_matrix[0]*i_matrix[1]
        print("For a", i_matrix, "grid:...")
        # setup the mesh, find the error of the mesh using a GG algorithm
        number_of_cells, start_co_ordinate, domain_size = i_matrix, [0.0, 0.0], [1.0, 1.0]
        [vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size)
        cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
        error = ea.cells_error_analysis(cell_centre_mesh)
        vol_table = cell_centre_mesh.cell_table.volume
        # seperate internal and external error cells - function
        bound_error, int_error, bound_size, int_size, ext_vol_table, int_vol_table = ea.seperate_int_ext(cell_centre_mesh, error, vol_table)
        print("Boundary Cells Analysis:", bound_size, "external cells")
        # process the boundary error
        norm_one_bound, norm_two_bound, norm_inf_bound = ea.error_package(bound_error, bound_size, ext_vol_table)
        bound_error_array[i][0], bound_error_array[i][1], bound_error_array[i][2] = norm_one_bound.T, norm_two_bound.T, norm_inf_bound.T
        print("Internal Cells Analysis:",int_size, "internal cells")
        # process the internal error
        norm_one_int, norm_two_int, norm_inf_int = ea.error_package(int_error, int_size, int_vol_table)
        int_error_array[i][0], int_error_array[i][1], int_error_array[i][2] = norm_one_int.T, norm_two_int.T, norm_inf_int.T
    h = size_store**(-0.5)
    ea.error_plotter(int_error_array, bound_error_array, h)


#cells_matrix = np.array([[3, 3], [9, 9], [12, 12], [30, 30]])
#g = cartesian_error(cells_matrix)


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------- FACES & GRID QUALITY -----------------------------------------------------------------
#cell_centroid, neighbour_centroid, face_unit_normal, face_centroid = np.array([2,2]), np.array([1,10]), \
#                                                                       np.array([-0.5,1]), np.array([3,7])
#
# number_of_cells, start_co_ordinate, domain_size = [15, 15], [0.0, 0.0], [1.0, 1.0]
# # [vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size, ft.partial(mg.stretch, [0.5, 0.5]))
# [vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size, ft.partial(mg.parallelogram, True, [0.0, 0.99]))
# cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
# quality_metrics = gq.cells_grid_quality(cell_centre_mesh)
# centroids = cell_centre_mesh.cell_table.centroid
# face_centroid = cell_centre_mesh.face_table.centroid
#cell_vol = cell_centre_mesh.cell_table.volume
#print(cell_vol)

#print("centroids", centroids)
#print("face", face_centroid)
# x1, y1 = centroids.T
# x2, y2 = face_centroid.T
# plt.scatter(x1, y1)
# plt.scatter(x2, y2)
# plt.show()
#
#
# e = gq.faces_grid_quality(cell_centre_mesh)
# #print("faces_grid_quality", e)
# f = gq.cells_grid_quality(cell_centre_mesh)
# #print("cell_grid_quality", f)
# g = gq.grid_average_quality(quality_metrics, cell_centre_mesh)
# print("grid average quality", np.round(g, 3))

def quality_error(cells_matrix, quality_matrix, grid_quality):
    matrix_size = len(cells_matrix)
    quality_size = len(quality_matrix)
    size_store = np.empty(shape=(matrix_size, ))
    bound_error_array = np.empty(shape=(quality_size, 3, 2))
    int_error_array =  np.empty(shape=(quality_size, 3, 2))
    for i, i_matrix in enumerate(cells_matrix):
        size_store[i] = i_matrix[0]*i_matrix[1]
        quality_array = np.empty(shape=(1, quality_size))

        # setup the mesh, find the error of the mesh using a GG algorithm
        for j, j_metric in enumerate(quality_matrix):
            number_of_cells, start_co_ordinate, domain_size = i_matrix, [0.0, 0.0], [1.0, 1.0]
            if grid_quality == 0:
                [vertex_coordinates, cell_vertex_connectivity, cell_type] = \
                    mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size, ft.partial(mg.parallelogram, False, j_metric))
            elif grid_quality == 1:
                [vertex_coordinates, cell_vertex_connectivity, cell_type] = \
                    mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size, ft.partial(mg.stretch, j_metric))
            cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)

            # error analysis
            error = ea.cells_error_analysis(cell_centre_mesh)
            vol_table = cell_centre_mesh.cell_table.volume

            # seperate internal and external error cells - function
            bound_error, int_error, bound_size, int_size, ext_vol_table, int_vol_table = ea.seperate_int_ext(cell_centre_mesh, error, vol_table)
            # print("Boundary Cells Analysis:", bound_size, "external cells")
            # process the boundary cells error
            norm_one_bound, norm_two_bound, norm_inf_bound = ea.error_package(bound_error, bound_size, ext_vol_table)
            bound_error_array[j][0], bound_error_array[j][1], bound_error_array[j][2] = norm_one_bound.T, norm_two_bound.T, norm_inf_bound.T
            # print("Internal Cells Analysis:",int_size, "internal cells")
            # process the internal cells error
            norm_one_int, norm_two_int, norm_inf_int = ea.error_package(int_error, int_size, int_vol_table)
            int_error_array[j][0], int_error_array[j][1], int_error_array[j][2] = norm_one_int.T, norm_two_int.T, norm_inf_int.T

            quality_metrics = gq.cells_grid_quality(cell_centre_mesh)
            avg_quality = gq.grid_average_quality(quality_metrics, cell_centre_mesh)
            quality_array[0][j] = avg_quality[0][grid_quality]
        ep.error_plotter(bound_error_array, int_error_array, quality_array, i, i_matrix, grid_quality)
    plt.show()


def uneven_experiment():
    cells_matrix = np.array([[9, 9], [12, 12], [15, 15]])
    uneven_matrix = np.array([[0.6, 0.6], [0.9, 0.9], [1.5, 1.5], [2.0, 2.0], [3.0, 3.0]])
    quality_error(cells_matrix, uneven_matrix, grid_quality = 1)
    return -1

def non_orthogonal_experiment():
    cells_matrix = np.array([[9, 9], [12, 12], [15, 15]])
    skew_matrix = np.array([[0.1, 0.1], [0.3, 0.3], [0.5, 0.5], [0.7, 0.7], [0.9, 0.9]])
    quality_error(cells_matrix, skew_matrix, 0)
    return - 1

uneven_experiment()
non_orthogonal_experiment()


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