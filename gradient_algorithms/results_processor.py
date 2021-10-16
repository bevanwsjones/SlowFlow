import numpy as np
import functools as ft
from mesh_generation import mesh_generator as mg
from mesh_preprocessor import preprocessor as pp
from gradient_algorithms import error_analysis as ea
from gradient_algorithms import errorplotter as ep
from gradient_algorithms import gridquality as gq
from matplotlib import pyplot as plt
from post_processor import graph as gr
def naming_fuc(grid_quality, met, phi_function):
    method = ['MGG', 'IGG', 'LS', 'NGG']
    grid_quality_name = ['nonorth','uneven','skew']
    phi_function_array = ['exp_cos', 'sin_cos', 'xysquared', 'sin']
    return method[met]+'_'+grid_quality_name[grid_quality]+'_'+phi_function_array[phi_function]

# ------------------------------- CARTESIAN GRID PLOTTER & RESULTS PROCESSOR ------------------------------------------
# Error Plotter for Cartesian Grid Test
def cartesian_error(cells_matrix, met = 0, phi_function = 0):
    """
    Determines the 3 grid errors (L1, L2 (LRms), Linf) in a Cartesian Grid, and shows how grid refinement changes the error
    accuracy.

    :param cells_matrix: various sizes of grid cells, [[N_x_1, N_y_1], [N_x_2, N_y_2], [N_x_3, N_y_3]]
    :type cells_matrix: numpy.array
    :param met: indicates the gradient algorithm method that is used in analysis. met = 0 is mean GG; met = 1 is interpolated GG; met = 2 is LS unweighted
    :type met: integer
    """
    matrix_size = len(cells_matrix)
    size_store = np.empty(shape=(matrix_size, ))
    bound_error_array = np.empty(shape=(matrix_size, 3, 2))
    int_error_array = np.empty(shape=(matrix_size, 3, 2))
    for i, i_matrix in enumerate(cells_matrix):
        size_store[i] = i_matrix[0]*i_matrix[1]
        #print("For a", i_matrix, "grid:...")
        # define mesh, setup & preprocess mesh, then find error of the mesh using gradient algorithm
        number_of_cells, start_co_ordinate, domain_size = i_matrix, [0.0, 0.0], [1.0, 1.0]
        [vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size)
        cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
        error = ea.cells_error_analysis(cell_centre_mesh, met, phi_function)
        vol_table = cell_centre_mesh.cell_table.volume
        quality_metrics = gq.cells_grid_quality(cell_centre_mesh)
        # seperate internal and external error cells from each other
        bound_error, int_error, bound_size, int_size, ext_vol_table, int_vol_table, bound_qual, int_qual = \
            ea.seperate_int_ext(cell_centre_mesh, error, vol_table, quality_metrics)
        #print("Boundary Cells Analysis:", bound_size, "external cells")
        # process the boundary error
        norm_one_bound, norm_two_bound, norm_inf_bound = ea.error_package(bound_error, bound_size, ext_vol_table)
        bound_error_array[i][0], bound_error_array[i][1], bound_error_array[i][2] = norm_one_bound.T, norm_two_bound.T, norm_inf_bound.T
        #print("Internal Cells Analysis:",int_size, "internal cells")
        # process the internal error
        norm_one_int, norm_two_int, norm_inf_int = ea.error_package(int_error, int_size, int_vol_table)
        int_error_array[i][0], int_error_array[i][1], int_error_array[i][2] = norm_one_int.T, norm_two_int.T, norm_inf_int.T
    h = size_store**(-0.5)
    ep.cartesian_plotter(int_error_array, bound_error_array, h)

def grid_refinement_error(cells_matrix, grid_metric, grid_quality = 0, met = 0, phi_function = 0):
    matrix_size = len(cells_matrix)
    metric_size = len(grid_metric)
    size_store = np.empty(shape=(matrix_size, ))
    qual_name_store = np.empty(shape=(metric_size, matrix_size))
    bound_error_array = np.empty(shape=(metric_size, matrix_size, 2))
    int_error_array = np.empty(shape=(metric_size, matrix_size, 2))
    for j, j_metric in enumerate(grid_metric):
        for i, i_matrix in enumerate(cells_matrix):
            size_store[i] = i_matrix[0]*i_matrix[1]

            # setup the mesh, find the error of the mesh using a Gradient Algorithm
            number_of_cells, start_co_ordinate, domain_size = i_matrix, [0.0, 0.0], [1.0, 1.0]
            if grid_quality == 0:
                [vertex_coordinates, cell_vertex_connectivity, cell_type] = \
                    mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size, ft.partial(mg.parallelogram, False, j_metric))
            elif grid_quality == 1:
                [vertex_coordinates, cell_vertex_connectivity, cell_type] = \
                    mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size, ft.partial(mg.stretch, j_metric))
            elif grid_quality == 2:     # skewness setup
                [vertex_coordinates, cell_vertex_connectivity, cell_type] = \
                    mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size)
                vertex_coordinates = mg.skew_strech(j_metric, number_of_cells, vertex_coordinates)

            cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)

            error = ea.cells_error_analysis(cell_centre_mesh, met, phi_function)
            vol_table = cell_centre_mesh.cell_table.volume
            tot_cell = cell_centre_mesh.cell_table.max_cell

            quality_metrics = gq.cells_grid_quality(cell_centre_mesh)

            # seperate internal and external error cells - function
            bound_error, int_error, bound_size, int_size, ext_vol_table, int_vol_table, bound_qual, int_qual = \
                ea.seperate_int_ext(cell_centre_mesh, error, vol_table, quality_metrics)
            # bound_qual = np.average(bound_qual[:, quality_metrics])

            # face transform value storage (for plotting)
            qual_name_store[j][i] = np.average(int_qual[:, grid_quality])
            # int_qual_1 = np.average(int_qual[:, 0])   # INCLUDE FOR DOING SKEWNESS ANALYSIS (Y-GRAD COMP)

            # process the boundary error
            norm_one_bound, norm_two_bound, norm_inf_bound = ea.error_package(bound_error, tot_cell, ext_vol_table)
            bound_error_array[j][i] = norm_two_bound.T
            norm_one_int, norm_two_int, norm_inf_int = ea.error_package(int_error, tot_cell, int_vol_table)
            int_error_array[j][i] = norm_two_int.T
        h = size_store**(-0.5)
    grid_name = np.round(np.average(qual_name_store, axis=1), 3)
    plt_name = naming_fuc(grid_quality, met, phi_function)
    fig1, fig2 = ep.grid_error_refine(int_error_array, bound_error_array, h, grid_name, grid_quality)
    fig1.savefig('plot_images_weight/'+plt_name+'xcomp.pdf')
    fig2.savefig('plot_images_weight/'+plt_name+'ycomp.pdf')
    print("Your Graph Name is:\n", plt_name)
    #plt.show()


def single_grid_metric(cells_matrix, quality_matrix, grid_quality, met = 0, phi_function = 0):
    quality_size = len(quality_matrix)
    quality_array = np.empty(shape=(1, quality_size))
    bound_error_array = np.empty(shape=(quality_size, 3, 2))
    int_error_array =  np.empty(shape=(quality_size, 3, 2))
    for j, j_metric in enumerate(quality_matrix):
        number_of_cells, start_co_ordinate, domain_size = cells_matrix, [0.0, 0.0], [1.0, 1.0]
        if grid_quality == 0:  # non-orthogonality setup
            [vertex_coordinates, cell_vertex_connectivity, cell_type] = \
                mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size,
                                           ft.partial(mg.parallelogram, False, j_metric))
        elif grid_quality == 1:  # unevenness setup
            [vertex_coordinates, cell_vertex_connectivity, cell_type] = \
                mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size,
                                           ft.partial(mg.stretch, j_metric))
        elif grid_quality == 2:  # skewness setup
            [vertex_coordinates, cell_vertex_connectivity, cell_type] = \
                mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size)
            vertex_coordinates = mg.skew_strech(j_metric, number_of_cells, vertex_coordinates)
        cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity,
                                                                    cell_type)
        # error analysis
        error = ea.cells_error_analysis(cell_centre_mesh, met, phi_function)
        vol_table = cell_centre_mesh.cell_table.volume

        quality_metrics = gq.cells_grid_quality(cell_centre_mesh)

        # seperate internal and external error cells - function
        bound_error, int_error, bound_size, int_size, ext_vol_table, int_vol_table, bound_qual, int_qual = ea.seperate_int_ext(
            cell_centre_mesh, error, vol_table, quality_metrics)
        # print("Boundary Cells Analysis:", bound_size, "external cells")
        # process the boundary cells error
        norm_one_bound, norm_two_bound, norm_inf_bound = ea.error_package(bound_error, bound_size, ext_vol_table)
        bound_error_array[j][0], bound_error_array[j][1], bound_error_array[j][
            2] = norm_one_bound.T, norm_two_bound.T, norm_inf_bound.T
        # print("Internal Cells Analysis:",int_size, "internal cells")
        # process the internal cells error
        norm_one_int, norm_two_int, norm_inf_int = ea.error_package(int_error, int_size, int_vol_table)
        int_error_array[j][0], int_error_array[j][1], int_error_array[j][
            2] = norm_one_int.T, norm_two_int.T, norm_inf_int.T
        quality_metrics = gq.cells_grid_quality(cell_centre_mesh)
        avg_quality = gq.grid_average_quality(quality_metrics, cell_centre_mesh)
        quality_array[0][j] = avg_quality[0][grid_quality]
    ep.grid_metric_plotter(bound_error_array, int_error_array, quality_array, cells_matrix, grid_quality)
    plt.show()