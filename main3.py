import numpy as np
from gradient_algorithms import results_processor as rp
from post_processor import graph as gr

# cartesian grid error calculator and plotter
cells_matrix = np.array([[3, 3], [9, 9], [30, 30], [30, 30]])
#g = rp.cartesian_error(cells_matrix, met = 0)

# Grid Quality Experiments
def uneven_experiment():
    cells_matrix = np.array([[9, 9], [30, 30], [90, 90]])
    uneven_matrix = np.array([[0.6, 0.6], [0.9, 0.9], [1.5, 1.5], [2.0, 2.0], [3.0, 3.0]])
    rp.quality_error(cells_matrix, uneven_matrix, grid_quality = 1, met = 0)
    return -1
def non_orthogonal_experiment():
    cells_matrix = np.array([[9, 9], [30, 30], [90, 90]])
    skew_matrix = np.array([[0.1, 0.1], [0.3, 0.3], [0.5, 0.5], [0.6, 0.6], [0.7, 0.7], [0.9, 0.9]])
    rp.quality_error(cells_matrix, skew_matrix, 0, met = 0)
    return - 1
def skewness_experiment():
    cells_matrix = np.array([[9, 9], [30, 30], [90, 90]])
    stretch_matrix = np.array([0.1, 0.2, 0.5, 1.0])
    rp.quality_error(cells_matrix, stretch_matrix, 2, met = 2)

# uneven_experiment()
# non_orthogonal_experiment()
# skewness_experiment()

def nonorth_refine_experiment():
    cells_matrix = np.array([[3, 3], [9, 9], [30, 30], [50, 50]])
    skew_matrix = np.array([[0.0, 0.0], [0.1, 0.1], [0.3, 0.3], [0.5, 0.5], [0.7, 0.7], [0.9, 0.9]])
    rp.grid_refinement_error(cells_matrix, skew_matrix, grid_quality=0, met = 2)

def uneven_refine_experiment():
    cells_matrix = np.array([[3, 3], [9, 9], [30, 30], [45, 45]])
    uneven_matrix = np.array([[0.4, 0.4], [0.7, 0.7], [0.9, 0.9], [1.2, 1.2], [1.5, 1.5]])
    rp.grid_refinement_error(cells_matrix, uneven_matrix, grid_quality=1, met = 0)

def skewness_refine_experiment():
    cells_matrix = np.array([[3, 3], [9, 9], [30, 30], [45, 45]])
    skew_matrix = np.array([0.1, 0.2, 0.5, 1.0, 1.5])
    rp.grid_refinement_error(cells_matrix, skew_matrix, grid_quality=2, met = 2)

nonorth_refine_experiment()
# uneven_refine_experiment()
# skewness_refine_experiment()

# number_of_cells, start_co_ordinate, domain_size = [10, 10], [0.0, 0.0], [1.0, 1.0]
# [vertex_coordinates, cell_vertex_connectivity, cell_type] = \
#     mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size, ft.partial(mg.stretch, [2.0, 2.0]))
# cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
# gr.plot_field(cell_centre_mesh, gr.generate_random_field(4, cell_centre_mesh.cell_table.max_cell,
#                                                          [True, False, True, False]))
