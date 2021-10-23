import numpy as np
from gradient_algorithms import results_processor as rp
from post_processor import graph as gr

# cartesian grid error calculator and plotter
# cells_matrix = np.array([[10, 10], [30, 30], [60, 60]])
cells_matrix = np.array([[10, 10], [30, 30], [60, 60], [90, 90]])
# # cells_matrix = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [60, 60], [80, 80], [100, 100]])
# g = rp.cartesian_error(cells_matrix, met = 0, phi_function = 3)
# g = rp.cartesian_error(cells_matrix, met = 0, phi_function = 3)
# g = rp.cartesian_error(cells_matrix, met = 1, phi_function = 3)

def nonorth_refine_experiment():
    cells_matrix = np.array([[10, 10], [20, 20], [40, 40], [60, 60], [90, 90]])
    skew_matrix = np.array([[0.0, 0.0], [0.1, 0.2], [0.3, 0.5], [0.8, 0.5], [0.9, 0.7], [0.9, 0.9]])
    # skew_matrix = np.array([[0.0, 0.1], [0.0, 0.3], [0.0, 0.5], [0.0, 0.6], [0.0, 0.7], [0.0, 0.8], [0.0, 0.9]])
    rp.grid_refinement_error(cells_matrix, skew_matrix, grid_quality=0, met = 1 , phi_function = 1)

def uneven_refine_experiment():
    cells_matrix = np.array([[8, 8], [12, 12], [16, 16], [24, 24], [32, 32], [64, 64]])
    # cells_matrix = np.array([[10, 1], [20, 1], [40, 1], [80, 1], [90, 1]])
    # cells_matrix = np.array([[1, 10], [1, 20], [1, 40], [1, 60], [1, 80], [1, 90]])
    # uneven_matrix = np.array([[0.8, 0.8], [0.9, 0.9], [1.2, 1.2], [1.5, 1.5]])
    uneven_matrix = np.array([[0.85, 0.85], [0.9, 0.9], [0.95, 0.95],[0.98, 0.98]])
    #uneven_matrix = np.array([[0.4, 0.4], [0.7, 0.7], [0.9, 0.9], [1.2, 1.2], [1.5, 1.5]])
    rp.grid_refinement_error(cells_matrix, uneven_matrix, grid_quality=1, met = 1, phi_function = 1)

def skewness_refine_experiment():
    cells_matrix = np.array([[10, 10], [20, 20], [40, 40], [60, 60], [90, 90]])
    skew_matrix = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    # skew_matrix = np.array([0.0000, 0.0005, 0.001, 0.01, 0.05, 0.1])
    rp.grid_refinement_error(cells_matrix, skew_matrix, grid_quality=2, met = 1, phi_function = 3)

# nonorth_refine_experiment()
# nonorth_refine_experiment()
# uneven_refine_experiment()
# skewness_refine_experiment()

# Grid Quality Experiments
def uneven_experiment():
    cells_matrix = np.array([32, 32])
    uneven_matrix = np.array([[0.85, 0.85], [0.88, 0.88], [0.9, 0.9], [0.92, 0.92], [0.95, 0.95], [0.98, 0.98]])
    rp.single_grid_metric(cells_matrix, uneven_matrix, grid_quality = 1, met=2, phi_function = 1)
    return -1
def non_orthogonal_experiment():
    cells_matrix = np.array([60, 60])
    # skew_matrix = np.array([[0.0, 0.0], [0.2, 0.2], [0.4, 0.4],[0.6, 0.6], [0.8, 0.8], [0.9, 0.9]])
    skew_matrix = np.array([[0.0, 0.0], [0.1, 0.2], [0.3, 0.5], [0.8, 0.5], [0.9, 0.7], [0.9, 0.9]])
    #skew_matrix = np.array([[0.0, 0.1], [0.0, 0.3], [0.0, 0.5], [0.0, 0.6], [0.0, 0.7], [0.0, 0.8], [0.0, 0.9]])
    rp.single_grid_metric(cells_matrix, skew_matrix, grid_quality = 0, met=2, phi_function = 1)
    return - 1
def skewness_experiment():
    cells_matrix = np.array([60, 60])
    stretch_matrix = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    rp.single_grid_metric(cells_matrix, stretch_matrix, grid_quality = 2, met=2, phi_function = 3)

# non_orthogonal_experiment()
# non_orthogonal_experiment()
# uneven_experiment()
# skewness_experiment()

def triangle_experiment():
    # cells_matrix = np.array([3, 5, 10])
    cells_matrix = np.array([5, 10, 15, 20, 40])
    hex = True
    rp.triangle_error(cells_matrix, hex, met=0, phi_function=1)
    rp.triangle_error(cells_matrix, hex, met=0, phi_function=1)
    rp.triangle_error(cells_matrix, hex, met=1, phi_function=1)
    rp.triangle_error(cells_matrix, hex, met=2, phi_function=1)

    rp.triangle_error(cells_matrix, hex, met=0, phi_function=2)
    rp.triangle_error(cells_matrix, hex, met=1, phi_function=2)
    rp.triangle_error(cells_matrix, hex, met=2, phi_function=2)

    rp.triangle_error(cells_matrix, hex, met=0, phi_function=3)
    rp.triangle_error(cells_matrix, hex, met=1, phi_function=3)
    rp.triangle_error(cells_matrix, hex, met=2, phi_function=3)
    return -1

triangle_experiment()
