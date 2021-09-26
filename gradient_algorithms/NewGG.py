import mesh_generation.mesh_generator as mg
import numpy as np
from mesh_preprocessor import finite_volume as fv
from mesh_preprocessor import preprocessor as pp
import math
import matplotlib.pyplot as plt

# Calculate phi_function for each internal cell. Return as [i_cell, phi_field_x, phi_field_y]
def cell_phi_function(cell_centre_mesh):
    phi_field = np.empty(shape=(cell_centre_mesh.cell_table.max_cell, ), dtype=float)
    coords = cell_centre_mesh.cell_table.centroid
    phi_field[:] = np.sin(coords[:, 0]) + np.cos(coords[:, 1])
    # phi_field[:] = coords[:, 0]**2 + coords[:, 1]**2
    return phi_field

# Calculate phi_function for each boundary cell. Return as [i_bound_cell, phi_field_x, phi_field_y]
def boundary_phi_function(cell_centre_mesh):
    boundary_no = cell_centre_mesh.face_table.max_boundary_face
    phi_bound_field = np.empty(shape=(boundary_no, ), dtype=float)
    coords = cell_centre_mesh.face_table.centroid[0:boundary_no]
    phi_bound_field[:] = np.sin(coords[:, 0]) + np.cos(coords[:, 1])
    #phi_bound_field[:] = coords[:, 0]**2 + coords[:, 1]**2
    return phi_bound_field

def arithMean(phi_field_0, phi_field_1):
    return 0.5 * (phi_field_0 + phi_field_1)

def close_point(cell_centroid, drn_vector, face_centroid):
    t = -1 * np.dot(drn_vector, np.subtract(cell_centroid, face_centroid))/(drn_vector[0]**2 + drn_vector[1]**2)
    close_point = np.add(cell_centroid, t*drn_vector)
    return close_point

# make exception case for if either x or y coordinate exhibits
def dist_vector(vec_0, vec_1):
    return math.sqrt((vec_1[0] - vec_0[0])**2 + (vec_1[1] - vec_0[1])**2)

def lineInter(phi_field_0, phi_field_1, x_0, x_1, x_i, fc):
    ptc = close_point(x_0, x_i, fc)
    return (dist_vector(x_1, ptc)*phi_field_0 + dist_vector(x_0, ptc)*phi_field_1)/dist_vector(x_0, x_1)
    #return (np.abs(ptc - x_1) * phi_field_0 + np.abs(ptc - x_0) * phi_field_1)/np.abs(x_1 - x_0)

def GreenGauss(cell_centre_mesh, status = 0):
    phi_gradient_field = np.zeros(shape=(cell_centre_mesh.cell_table.max_cell, 2), dtype=float)
    phi_field = cell_phi_function(cell_centre_mesh)                             # phi values for all cells
    phi_boundary_field = boundary_phi_function(cell_centre_mesh)                # phi values for all boundaries
    # face_contribution is the analytical value - Dirichlet Condition
    for ibound_face in range(cell_centre_mesh.face_table.max_boundary_face):
        i_cell_0 = cell_centre_mesh.face_table.connected_cell[ibound_face][0]
        face_area = cell_centre_mesh.face_table.area[ibound_face]
        face_normal = cell_centre_mesh.face_table.normal[ibound_face]
        face_contribution = phi_boundary_field[ibound_face]
        phi_gradient_field[i_cell_0] += face_contribution * face_area * face_normal/cell_centre_mesh.cell_table.volume[i_cell_0]
    # Compute gradients for internal faces
    for i_face in range(cell_centre_mesh.face_table.max_boundary_face, cell_centre_mesh.face_table.max_face):
        i_cell_0 = cell_centre_mesh.face_table.connected_cell[i_face][0]
        i_cell_1 = cell_centre_mesh.face_table.connected_cell[i_face][1]
        phi_field_0 = phi_field[i_cell_0]                               # find phi field for each cell
        phi_field_1 = phi_field[i_cell_1]
        if status == 0:
            face_contribution = arithMean(phi_field_0, phi_field_1)       # can turn this into a face operator function?
        elif status == 1:
            x_0 = cell_centre_mesh.cell_table.centroid[i_cell_0]  # determine coordinates for cell centroids
            x_1 = cell_centre_mesh.cell_table.centroid[i_cell_1]
            x_i = cell_centre_mesh.face_table.cc_unit_vector[i_face]  # direction vector
            fc = cell_centre_mesh.face_table.centroid[i_face]
            face_contribution = lineInter(phi_field_0, phi_field_1, x_0, x_1, x_i, fc)
        face_area = cell_centre_mesh.face_table.area[i_face]
        face_normal = cell_centre_mesh.face_table.normal[i_face]
        phi_gradient_field[i_cell_0] += face_contribution*face_area*face_normal/cell_centre_mesh.cell_table.volume[i_cell_0]
        phi_gradient_field[i_cell_1] -= face_contribution*face_area*face_normal/cell_centre_mesh.cell_table.volume[i_cell_1]
    return phi_gradient_field

# function that makes an array of boundary cell numbers in grid
def bound_cells(cell_centre_mesh):
    cell_storage = set()
    for ibound_face in range(cell_centre_mesh.face_table.max_boundary_face):
        i_cell_0 = cell_centre_mesh.face_table.connected_cell[ibound_face][0]
        cell_storage.add(i_cell_0)
    return cell_storage
