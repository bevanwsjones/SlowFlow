import numpy as np
import math
from gradient_algorithms import NewGG

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------VECTOR CALCULATION (FOR SINGLE FACE) QUALITY METRICS -------------------------------------

def nonorthogonality(cell_centroid, neighbour_centroid, face_unit_normal):
    cell_dist = np.subtract(neighbour_centroid, cell_centroid)
    zeta = math.acos((np.dot(cell_dist, face_unit_normal))/np.dot(np.linalg.norm(cell_dist), np.linalg.norm(face_unit_normal)))
    return zeta

def close_point(cell_centroid, neighbour_centroid, face_centroid):
    drn_vector = neighbour_centroid - cell_centroid
    t = -np.dot(drn_vector, np.subtract(cell_centroid, face_centroid))/(drn_vector[0]**2 + drn_vector[1]**2)
    close_point = np.add(cell_centroid, t*drn_vector)
    #print(close_point)
    return close_point

def unevenness(cell_centroid, neighbour_centroid, face_centroid):
    cp = close_point(cell_centroid, neighbour_centroid, face_centroid)
    midpoint = 0.5*(np.add(cell_centroid, neighbour_centroid))
    unevenness = (np.linalg.norm(np.subtract(cp, midpoint)))/(np.linalg.norm(np.subtract(neighbour_centroid, cell_centroid)))
    return unevenness

def skewness(cell_centroid, neighbour_centroid, face_centroid):
    cp = close_point(cell_centroid, neighbour_centroid, face_centroid)
    #print("close-point", cp)
    unevenness = (np.linalg.norm(np.subtract(face_centroid, cp)))/(np.linalg.norm(np.subtract(neighbour_centroid, cell_centroid)))
    return unevenness

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------FACE CALCULATION (FOR ALL FACES IN GRID) QUALITY METRICS ---------------------------------
# STILL NEED TO CONSIDER WHAT HAPPENS TO BOUNDARY CELLS (IF CONNECTIVITY GIVES -1, THEN ... FOR QUALITY METRIC...)

def faces_grid_quality(cell_centre_mesh):
# return the non-orthogonality, unevenness, and skewness for all the faces in the grid
# [i_face][non-orthogonality, unevenness, skewness]
    cell_centroid = cell_centre_mesh.cell_table.centroid
    face_normals = cell_centre_mesh.face_table.normal
    face_centroid = cell_centre_mesh.face_table.centroid
    fc_connectivity = cell_centre_mesh.face_table.connected_cell
    max_face = cell_centre_mesh.face_table.max_face
    bound_face = cell_centre_mesh.face_table.max_boundary_face
    int_face = max_face - bound_face
    quality_face_table = np.zeros(shape=(max_face, 3))
    for i_face in range(bound_face, max_face):
        i_cell_0 = fc_connectivity[i_face][0]       # find the cell numbers for each face index
        i_cell_1 = fc_connectivity[i_face][1]
        x_0 = cell_centroid[i_cell_0]               # find the centroid coordinates for the cell
        x_1 = cell_centroid[i_cell_1]
        face_normal = face_normals[i_face]
        face_cent = face_centroid[i_face]
        quality_face_table[i_face][0] = nonorthogonality(x_0, x_1, face_normal)  # face connectivity
        quality_face_table[i_face][1] = unevenness(x_0, x_1, face_cent)
        quality_face_table[i_face][2] = skewness(x_0, x_1, face_cent)
    quality_face_table = np.round(quality_face_table, 4)
    #print(quality_face_table)
    return quality_face_table

# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------SINGLE CELL GRID COMPUTATION HAND CALCULATION -------------------

# return the cell-grid quality metric matrix:
# returns [i_cell][average non-orthogonality][average unevenness][average skewness]
# averages indicate face-area weightings
def single_cell_grid_quality():
    cell_centroid = np.array([[2, 2], [5, 2], [1, 10], [-2, 4], [1, -4]])
    face_normals =  np.array([[1, 0], [-0.5, 1], [-1, 2], [-1, -1]])
    face_centroid = np.array([[4, 6], [3, 7], [1, 2], [1, 1]])
    fc_connectivity = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
    face_areas = np.array([5, 6, 5, 4])
    cell_face_table = np.array([[1, 2, 3, 4]])
    cell_size = 1
    quality_metrics = np.zeros(shape=(cell_size, 3))                # store quality metrics of all cells
    # compute quality metrics for all faces
    faces_grid = faces_grid_quality(cell_centroid, face_normals, face_centroid, fc_connectivity, number_of_faces = 4)
    grid = np.zeros(shape=(1, 3))
    area_sum = 0
    for i_cell in range(cell_size):
        faces = cell_face_table[i_cell]                     # get the faces from each cell
        for i_face in range(len(faces)):
            grid = grid + faces_grid[i_face]*face_areas[i_face]   # sum all the quality metrics from adjacent faces to cell
            area_sum = area_sum + face_areas[i_face]
        quality_metrics[i_cell] = grid/area_sum           # find average of quality metrics
    return quality_metrics

# ----------------------------CELL CALCULATION (FOR ALL CELLS IN GRID) QUALITY METRICS---------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# calculate all the average grid quality metrics per cell
def cells_grid_quality(cell_centre_mesh):
    cell_face_table = cell_centre_mesh.cell_table.connected_face
    cell_size = cell_centre_mesh.cell_table.max_cell
    face_areas = cell_centre_mesh.face_table.area
    quality_metrics = np.zeros(shape=(cell_size, 3))
    # compute quality metrics for all faces
    faces_grid = faces_grid_quality(cell_centre_mesh)
    for i_cell in range(cell_size):
        grid = np.zeros(shape=(1, 3))
        faces = cell_face_table[i_cell]                     # get the faces from each cell
        area_sum = 0
        for i, i_face in enumerate(faces):
            grid += faces_grid[i_face]*face_areas[i_face]          # sum all the quality metrics from adjacent faces to cell
            area_sum += face_areas[i_face]
        #print(area_sum, "for cell", i_cell)
        quality_metrics[i_cell] = grid/area_sum         # find average of quality metrics
    return quality_metrics

def seperate_int_ext(cell_centre_mesh, quality_metrics, vol_table):
    boundary_cells = NewGG.bound_cells(cell_centre_mesh)
    bound_size = len(boundary_cells)
    tot_cell = cell_centre_mesh.cell_table.max_cell
    int_cell = tot_cell - bound_size
    bound_qual = np.zeros(shape=(bound_size, 3))                     # initilaise storage values
    ext_vol_table = np.zeros(shape=(bound_size, 1))
    for i, i_cell in enumerate(boundary_cells):
        bound_qual[i] = quality_metrics[i_cell]
        ext_vol_table[i] = vol_table[i_cell]
    int_qual = np.delete(quality_metrics, list(boundary_cells), axis = 0)
    int_vol_table = np.delete(vol_table, list(boundary_cells), axis = 0)
    return bound_qual, int_qual, bound_size, int_cell, ext_vol_table, int_vol_table

# return the grid average quality
def grid_average_quality(quality_metrics, vol_table):
    avg_array = np.empty(shape=(1, 3))
    vol_tot = 0
    for i, i_cell in enumerate(quality_metrics):
        avg_array += vol_table[i]*i_cell
        vol_tot += vol_table[i]
    avg_array = avg_array/vol_tot
    return avg_array




# def faces_grid_quality(cell_centroid, face_normals, face_centroid, fc_connectivity, number_of_faces):
# # return the non-orthogonality, unevenness, and skewness for all the faces in the grid
# # [i_face][non-orthogonality, unevenness, skewness]
#     #number_of_faces = 4
#     quality_face_table = np.zeros(shape=(number_of_faces, 3))
#     print("inside_loop", number_of_faces)
#     print(fc_connectivity)
#     for i_face in range(number_of_faces):
#         i_cell_0 = fc_connectivity[i_face][0]       # find the cell numbers for each face index
#         i_cell_1 = fc_connectivity[i_face][1]
#         if i_cell_0 or i_cell_1 == -1:
#             print("boundary")
#             continue
#         x_0 = cell_centroid[i_cell_0]               # find the centroid coordinates for the cell
#         x_1 = cell_centroid[i_cell_1]
#         face_normal = face_normals[i_face]
#         face_cent = face_centroid[i_face]
#         quality_face_table[i_face][0] = nonorthogonality(x_0, x_1, face_normal)  # face connectivity
#         quality_face_table[i_face][1] = unevenness(x_0, x_1, face_cent)
#         quality_face_table[i_face][2] = skewness(x_0, x_1, face_cent)
#     quality_face_table = np.round(quality_face_table, 4)
#     return quality_face_table