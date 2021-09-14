import numpy as np
import math
#import scipy as sp
from scipy import linalg, sparse
from mesh_generation import mesh_generator as mg
from mesh_preprocessor import preprocessor as pp

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
    return close_point

def unevenness(cell_centroid, neighbour_centroid, face_centroid):
    cp = close_point(cell_centroid, neighbour_centroid, face_centroid)
    midpoint = 0.5*(np.add(cell_centroid, neighbour_centroid))
    unevenness = (np.linalg.norm(np.subtract(cp, midpoint)))/(np.linalg.norm(np.subtract(neighbour_centroid, cell_centroid)))
    return unevenness

def skewness(cell_centroid, neighbour_centroid, face_centroid):
    cp = close_point(cell_centroid, neighbour_centroid, face_centroid)
    unevenness = (np.linalg.norm(np.subtract(face_centroid, cp)))/(np.linalg.norm(np.subtract(neighbour_centroid, cell_centroid)))
    return unevenness

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------FACE CALCULATION (FOR ALL FACES IN GRID) QUALITY METRICS ---------------------------------
# STILL NEED TO CONSIDER WHAT HAPPENS TO BOUNDARY CELLS (IF CONNECTIVITY GIVES -1, THEN ... FOR QUALITY METRIC...)

def faces_grid_quality(cell_centroid, face_normals, face_centroid, fc_connectivity):
# return the non-orthogonality, unevenness, and skewness for all the faces in the grid
# [i_face][non-orthogonality, unevenness, skewness]
    number_of_faces = 4
    quality_face_table = np.zeros(shape=(number_of_faces, 3))
    for i_face in range(number_of_faces):
        i_cell_0 = fc_connectivity[i_face][0]       # find the cell numbers for each face index
        i_cell_1 = fc_connectivity[i_face][1]
        if i_cell_0 or i_cell_1 == -1:
            continue
        x_0 = cell_centroid[i_cell_0]               # find the centroid coordinates for the cell
        x_1 = cell_centroid[i_cell_1]
        face_normal = face_normals[i_face]
        face_cent = face_centroid[i_face]
        quality_face_table[i_face][0] = nonorthogonality(x_0, x_1, face_normal)  # face connectivity
        quality_face_table[i_face][1] = unevenness(x_0, x_1, face_cent)
        quality_face_table[i_face][2] = skewness(x_0, x_1, face_cent)
    quality_face_table = np.round(quality_face_table, 4)
    return quality_face_table

# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------SINGLE CELL GRID COMPUTATION HAND CALCULATION -------------------

number_of_cells, start_co_ordinate, domain_size = [3, 3], [0.0, 0.0], [1.0, 1.0]
[vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size)
cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
def single_cell_grid_quality():
# return the cell-grid quality metric matrix:
# returns [i_cell][average non-orthogonality][average unevenness][average skewness]
    cell_centroid = np.array([[2, 2], [5, 2], [1, 10], [-2, 4], [1, -4]])
    face_normals =  np.array([[1, 0], [-0.5, 1], [-1, 2], [-1, -1]])
    face_centroid = np.array([[4, 6], [3, 7], [1, 2], [1, 1]])
    fc_connectivity = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
    cell_face_table = np.array([[1, 2, 3, 4]])
    cell_size = 1
    quality_metrics = np.zeros(shape=(cell_size, 3))                # store quality metrics of all cells
    # compute quality metrics for all faces
    faces_grid = faces_grid_quality(cell_centroid, face_normals, face_centroid, fc_connectivity)
    grid = np.zeros(shape=(1, 3))
    for i_cell in range(cell_size):
        faces = cell_face_table[i_cell]                     # get the faces from each cell
        for i_face in range(len(faces)):
            grid = grid + faces_grid[i_face]                # sum all the quality metrics from adjacent faces to cell
        quality_metrics[i_cell] = grid/len(faces)           # find average of quality metrics
    return quality_metrics

# ----------------------------CELL CALCULATION (FOR ALL CELLS IN GRID) QUALITY METRICS---------------------------------
# ---------------------------------------------------------------------------------------------------------------------
def cells_grid_quality(cell_centre_mesh):
    cell_centroid = cell_centre_mesh.cell_table.centroid
    face_normals = cell_centre_mesh.face_table.normal
    face_centroid = cell_centre_mesh.face_table.centroid
    fc_connectivity = cell_centre_mesh.face_table.connected_cell
    cell_face_table = cell_centre_mesh.cell_table.connected_face
    cell_size = len(cell_face_table)
    quality_metrics = np.zeros(shape=(cell_size, 3))
    # compute quality metrics for all faces
    faces_grid = faces_grid_quality(cell_centroid, face_normals, face_centroid, fc_connectivity)
    grid = np.zeros(shape=(1, 3))
    for i_cell in range(cell_size):
        faces = cell_face_table[i_cell]                     # get the faces from each cell
        for i_face in range(len(faces)):
            grid = grid + faces_grid[i_face]                # sum all the quality metrics from adjacent faces to cell
        quality_metrics[i_cell] = grid/len(faces)           # find average of quality metrics
    return quality_metrics

