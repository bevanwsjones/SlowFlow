import numpy as np
import math
from gradient_algorithms import modGreenGauss as modGG

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------- FUNDAMENTAL DEFINITIONS: Error and Error Norms --------------------------------
def relative_error(x, x_true):
    return abs(x - x_true)/abs(x_true)

def abs_error(x, x_true):
    return abs(x - x_true)

def array_relative_error(x, x_true):
# what happens if the analytical function is zero?
    error = np.zeros(shape=(1, 2))
    x_0, y_0 = x[0][0], x[0][1]
    x_t0, y_t0 = x_true[0][0], x_true[0][1]
    error_x = relative_error(x_0, x_t0)
    error_y = relative_error(y_0, y_t0)
    error[0][0] = error_x
    error[0][1] = error_y
    return error

def L_norm_one(error):
    err_sum = 0.0
    for i in range(len(error)):
        err_sum += error[i]
    return err_sum

def L_norm_two(error):
    err_sum = 0.0
    for i in range(len(error)):
        err_sum += (error[i])**2
    return math.sqrt(err_sum)

def L_norm_inf(error):
    l = np.amax(error)
    return l

# ----------------------------------------------------------------------------------------------------------------------
# ---------------------- Grid Error Analysis ---------------------------------------------------------------------------

# Analytical Solution Calculator for each cell centroid
def cell_true_function(cell_centre_mesh):
    true_field = np.empty(shape=(cell_centre_mesh.cell_table.max_cell, 2), dtype=float)
    coords = cell_centre_mesh.cell_table.centroid
    true_field[:, 0] = np.cos(coords[:, 0])
    true_field[:, 1] = -1*np.sin(coords[:, 1])
    return true_field

# returns cell error table for each cell phi compared to analytical phi: [i_cell][error_x][error_y]
def cells_error_analysis(cell_centre_mesh):
    approx_field = modGG.GreenGauss(cell_centre_mesh)
    true_field = cell_true_function(cell_centre_mesh)
    error = abs_error(approx_field, true_field)
    return error

# returns L 1 Norm for error terms from the input error table: [L1_x][L1_y]
def grid_norm_one(error):
    x_error, y_error = error[:, 0], error[:, 1]
    L_norm_x, L_norm_y = L_norm_one(x_error),  L_norm_one(y_error)
    grid_norm_one = np.array([L_norm_x, L_norm_y])
    return grid_norm_one

# returns L 2 Norm for error terms from the input error table: [L2_x][L2_y]
def grid_norm_two(error):
    x_error, y_error = error[:, 0], error[:, 1]
    # print(x_error)
    #     # print(y_error)
    L_norm_x, L_norm_y = L_norm_two(x_error),  L_norm_two(y_error)
    grid_norm_two = np.array([L_norm_x, L_norm_y])
    return grid_norm_two

# returns L_inf Norm for error terms from the input error table: [L_inf_x][L_inf_y]
def grid_norm_inf(error):
    x_error, y_error = error[:, 0], error[:, 1]
    L_norm_x, L_norm_y = L_norm_inf(x_error),  L_norm_inf(y_error)
    grid_norm_two = np.array([L_norm_x, L_norm_y])
    return grid_norm_two

# return average L1 and L2 Norms: takes in L1 or L2 and then divides by cell_no
def grid_avg_norm(cell_centre_mesh, grid_norm):
    no_cells = cell_centre_mesh.cell_table.max_cell
    L_avg_x = grid_norm[0] / no_cells
    L_avg_y = grid_norm[1] / no_cells
    grid_avg_norm = np.array([L_avg_x, L_avg_y])
    return grid_avg_norm