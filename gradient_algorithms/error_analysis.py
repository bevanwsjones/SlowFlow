import numpy as np
import math
from gradient_algorithms import NewGG
from matplotlib import pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------- FUNDAMENTAL DEFINITIONS: Error and Error Norms --------------------------------
def relative_error(x, x_true):
    return abs(x - x_true)/abs(x_true)

def abs_error(x, x_true):
    hey = np.abs(x - x_true)
    return hey

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

def L_norm_one(error, vol_table):
    err_sum = 0.0
    vol_sum = 0.0
    for i in range(len(error)):
        err_sum += abs(error[i]*vol_table[i])
        vol_sum += vol_table[i]
    return err_sum/vol_sum

def L_norm_two(error, vol_table):
    err_sum = 0.0
    vol_sum = 0.0
    for i in range(len(error)):
        err_sum += (error[i]*vol_table[i])**2
        vol_sum += vol_table[i]
    return math.sqrt(err_sum)/vol_sum

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
    # true_field[:, 0] = 2*coords[:, 0]
    # true_field[:, 1] = 2*coords[:, 1]
    # true_field[:, 0] = 1/np.cosh(coords[:, 0])
    # true_field[:, 1] = 1/np.cosh(coords[:, 1])
    return true_field

# returns cell error table for each cell phi compared to analytical phi: [i_cell][error_x][error_y]
def cells_error_analysis(cell_centre_mesh):
    approx_field = NewGG.GreenGauss(cell_centre_mesh)
    true_field = cell_true_function(cell_centre_mesh)
    error = abs_error(approx_field, true_field)
    # print("approx field is here", approx_field)
    return error

# returns L 1 Norm for error terms from the input error table: [L1_x][L1_y]
def grid_norm_one(error, vol_table):
    x_error, y_error = error[:, 0], error[:, 1]
    L_norm_x, L_norm_y = L_norm_one(x_error, vol_table),  L_norm_one(y_error, vol_table)
    grid_norm_one = np.array([L_norm_x, L_norm_y])
    return grid_norm_one

# returns L 2 Norm for error terms from the input error table: [L2_x][L2_y]
def grid_norm_two(error, vol_table):
    x_error, y_error = error[:, 0], error[:, 1]
    L_norm_x, L_norm_y = L_norm_two(x_error, vol_table),  L_norm_two(y_error, vol_table)
    grid_norm_two = np.array([L_norm_x, L_norm_y])
    return grid_norm_two

# returns L_inf Norm for error terms from the input error table: [L_inf_x][L_inf_y]
def grid_norm_inf(error):
    x_error, y_error = error[:, 0], error[:, 1]
    L_norm_x, L_norm_y = L_norm_inf(x_error),  L_norm_inf(y_error)
    grid_norm_two = np.array([L_norm_x, L_norm_y])
    return grid_norm_two

# return average L1 and L2 Norms: takes in L1 or L2 and then divides by cell_no
def grid_avg_norm(grid_norm, no_cells):
    L_avg_x = grid_norm[0] / no_cells
    L_avg_y = grid_norm[1] / no_cells
    grid_avg_norm = np.array([L_avg_x, L_avg_y])
    return grid_avg_norm

# ----------------------------------------------------------------------------------------------------------------------
# ---------------------- Internal and Boundary Cell Calcs --------------------------------------------------------------
# function that separates the error table from the internal and external cells
def seperate_int_ext(cell_centre_mesh, error, vol_table):
    boundary_cells = NewGG.bound_cells(cell_centre_mesh)
    bound_size = len(boundary_cells)
    tot_cell = cell_centre_mesh.cell_table.max_cell
    int_cell = tot_cell - bound_size
    bound_error = np.zeros(shape=(bound_size, 2))                     # initilaise storage values
    ext_vol_table = np.zeros(shape=(bound_size, 1))
    for i, i_cell in enumerate(boundary_cells):
        bound_error[i] = error[i_cell]
        ext_vol_table[i] = vol_table[i_cell]
    int_error = np.delete(error, list(boundary_cells), axis = 0)
    int_vol_table = np.delete(vol_table, list(boundary_cells), axis = 0)
    return bound_error, int_error, bound_size, int_cell, ext_vol_table, int_vol_table

def error_package(error, size, vol_table):
    norm_one = grid_norm_one(error, vol_table)
    norm_two = grid_norm_two(error, vol_table)
    norm_inf = grid_norm_inf(error)
    #print("Norm one", norm_one)
    #print("Norm two", norm_two)
    #print("Norm inf", norm_inf)
    return norm_one, norm_two, norm_inf

def error_plotter(int_error_array, bound_error_array, h):
    # plot 1 - x-gradient error - internal cells
    plt.subplot(2, 2, 1)
    plt.plot(h, int_error_array[:, 0, 0], '-o', label='L1 norm')
    plt.plot(h, int_error_array[:, 1, 0], '-ok', label='L2 norm')
    plt.plot(h, int_error_array[:, 2, 0], '-or', label='Linf norm')
    plt.xlabel("Characteristic Length")
    plt.ylabel("x-gradient internal cell error")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    # plot 2 - y-gradient error - internal cells
    plt.subplot(2, 2, 2)
    plt.plot(h, int_error_array[:, 0, 1], '-o', label='L1 norm')
    plt.plot(h, int_error_array[:, 1, 1], '-ok', label='L2 norm')
    plt.plot(h, int_error_array[:, 2, 1], '-or', label='Linf norm')
    plt.xlabel("Characteristic Length")
    plt.ylabel("y-gradient internal cell error")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    # plot 3 - x-gradient error - boundary cells
    plt.subplot(2, 2, 3)
    plt.plot(h, bound_error_array[:, 0, 0], '-o', label='L1 norm')
    plt.plot(h, bound_error_array[:, 1, 0], '-ok', label='L2 norm')
    plt.plot(h, bound_error_array[:, 2, 0], '-or', label='Linf norm')
    plt.xlabel("Characteristic Length")
    plt.ylabel("x-gradient bound cell error")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    # plot 4 - y-gradient error - boundary cells
    plt.subplot(2, 2, 4)
    plt.plot(h, bound_error_array[:, 0, 1], '-o', label='L1 norm')
    plt.plot(h, bound_error_array[:, 1, 1], '-ok', label='L2 norm')
    plt.plot(h, bound_error_array[:, 2, 1], '-or', label='Linf norm')
    plt.xlabel("Characteristic Length")
    plt.ylabel("y-gradient bound cell error")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.suptitle("2D Structured Cartesian Grid Error Analysis")
    plt.tight_layout
    plt.show()
    return -1