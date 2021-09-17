import mesh_generation.mesh_generator as mg
import numpy as np
from mesh_preprocessor import finite_volume as fv
from mesh_preprocessor import preprocessor as pp
import math
import matplotlib.pyplot as plt

# Computing Gradients for a 2D Mesh - Phi Field
number_of_cells, start_co_ordinate, domain_size = [90, 90], [0.0, 0.0], [1.0, 1.0]
[vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size)
cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)


# Define arrays for phi_function at cells and an empty gradient field array that accepts 2 component values
phi_field = np.empty(shape=(cell_centre_mesh.cell_table.max_cell, 2), dtype=float) # getting phi|_i_cell -> [i_cell]
phi_gradient_field = np.zeros(shape=(cell_centre_mesh.cell_table.max_cell, 2), dtype=float)

# Calculate phi_function for each cell. Return as [i_cell, phi_field_x, phi_field_y]
def cell_phi_function(cell_centre_mesh):
    phi_field = np.empty(shape=(cell_centre_mesh.cell_table.max_cell, 2), dtype=float)
    coords = cell_centre_mesh.cell_table.centroid
    phi_field[:, 0] = np.tanh(coords[:, 0])
    phi_field[:, 1] = np.tanh(coords[:, 1])
    # phi_field[:, 0] = np.sin(coords[:, 0])
    # phi_field[:, 1] = np.cos(coords[:, 1])
    return phi_field

def boundary_phi_function(cell_centre_mesh):
    boundary_no = cell_centre_mesh.face_table.max_boundary_face
    phi_bound_field = np.empty(shape=(boundary_no, 2), dtype=float)
    coords = cell_centre_mesh.face_table.centroid[0:boundary_no]
    phi_bound_field[:, 0] = np.tanh(coords[:, 0])
    phi_bound_field[:, 1] = np.tanh(coords[:, 1])
    # phi_bound_field[:, 0] = np.sin(coords[:, 0])
    # phi_bound_field[:, 1] = np.cos(coords[:, 1])
    return phi_bound_field

# phi_field = np.cos(x) x is the cell centroid.

def arithMean(phi_field_0, phi_field_1):
    return 0.5 * (phi_field_0 + phi_field_1)

def close_point(cell_centroid, drn_vector, face_centroid):
    t = -np.dot(drn_vector, np.subtract(cell_centroid, face_centroid))/(drn_vector[0]**2 + drn_vector[1]**2)
    close_point = np.add(cell_centroid, t*drn_vector)
    return close_point

# def lineInter(phi_field_0, phi_field_1, x_0, x_1, x_i, fc):         #contains BUGS - Must be fixed
#     ptc = close_point(x_0, x_i, fc)
#     return (np.abs(ptc - x_1) * phi_field_0 + np.abs(ptc - x_0) * phi_field_1)/np.abs(x_1 - x_0)
#     # does lerp


def GreenGauss(cell_centre_mesh):
    phi_field = cell_phi_function(cell_centre_mesh)                             # phi values for all cells
    phi_boundary_field = boundary_phi_function(cell_centre_mesh)                # phi values for all boundaries
    # face_contribution is the analytical value - Dirichlet Condition
    for ibound_face in range(cell_centre_mesh.face_table.max_boundary_face):
        #continue
        i_cell_0 = cell_centre_mesh.face_table.connected_cell[ibound_face][0]
        face_area = cell_centre_mesh.face_table.area[ibound_face]
        face_normal = cell_centre_mesh.face_table.area[ibound_face]
        face_contribution = phi_boundary_field[ibound_face]
        phi_gradient_field[i_cell_0] += face_contribution * face_area * face_normal / \
                                        cell_centre_mesh.cell_table.volume[i_cell_0]
    # Compute gradients for internal faces
    for i_face in range(cell_centre_mesh.face_table.max_boundary_face, cell_centre_mesh.face_table.max_face):
        i_cell_0 = cell_centre_mesh.face_table.connected_cell[i_face][0]
        i_cell_1 = cell_centre_mesh.face_table.connected_cell[i_face][1]
        phi_field_0 = phi_field[i_cell_0]                               # find phi field for each cell
        phi_field_1 = phi_field[i_cell_1]
        x_0 = cell_centre_mesh.cell_table.centroid[i_cell_0]            # determine coordinates for cell centroids
        x_1 = cell_centre_mesh.cell_table.centroid[i_cell_1]
        x_i = cell_centre_mesh.face_table.cc_unit_vector[i_face]             # direction vector
        fc = cell_centre_mesh.face_table.centroid[i_face]
        face_contribution = arithMean(phi_field_0, phi_field_1)       # can turn this into a face operator function?
        # face_contribution_0 = lineInter(phi_field_0, phi_field_1, x_0, x_1, x_i, fc)
        face_area = cell_centre_mesh.face_table.area[i_face]
        face_normal = cell_centre_mesh.face_table.area[i_face]
        phi_gradient_field[i_cell_0] += face_contribution*face_area*face_normal/cell_centre_mesh.cell_table.volume[i_cell_0]
        phi_gradient_field[i_cell_1] -= face_contribution*face_area*face_normal/cell_centre_mesh.cell_table.volume[i_cell_1]
    # return face_contribution_0, face_contribution
    return phi_gradient_field

def bound_cells(cell_centre_mesh):
    cell_storage = set()
    for ibound_face in range(cell_centre_mesh.face_table.max_boundary_face):
        i_cell_0 = cell_centre_mesh.face_table.connected_cell[ibound_face][0]
        cell_storage.add(i_cell_0)
    return cell_storage

    # cell_storage = []
    # for ibound_face in range(cell_centre_mesh.face_table.max_boundary_face):
    #     i_cell_0 = cell_centre_mesh.face_table.connected_cell[ibound_face][0]
    #     cell_storage.append(i_cell_0)
    # final_list = set()
    # for i in cell_storage:
    #     final_list.add(i)
    # return final_list