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
    # phi_field[:] = np.sin(coords[:, 0]) + np.cos(coords[:, 1])
    phi_field[:] = np.exp(coords[:, 0]) * np.cos(coords[:, 1])
    # phi_field[:] = coords[:, 0]**2 + coords[:, 1]**2
    # phi_field[:] = 2 * np.multiply(coords[:, 0], coords[:, 1])
    return phi_field

# Calculate phi_function for each boundary cell. Return as [i_bound_cell, phi_field_x, phi_field_y]
def boundary_phi_function(cell_centre_mesh):
    boundary_no = cell_centre_mesh.face_table.max_boundary_face
    phi_bound_field = np.empty(shape=(boundary_no, ), dtype=float)
    coords = cell_centre_mesh.face_table.centroid[0:boundary_no]
    # phi_bound_field[:] = np.sin(coords[:, 0]) + np.cos(coords[:, 1])
    phi_bound_field[:] = np.exp(coords[:, 0]) * np.cos(coords[:, 1])
    # phi_bound_field[:] = coords[:, 0]**2 + coords[:, 1]**2
    # phi_bound_field[:] = 2 * np.multiply(coords[:, 0], coords[:, 1])
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

def lineInter(phi_field_0, phi_field_1, x_0, x_1, x_i, fc):                 # unit test this for cartesian grids
    ptc = close_point(x_0, x_i, fc)
    return (dist_vector(x_1, ptc)*phi_field_0 + dist_vector(x_0, ptc)*phi_field_1)/dist_vector(x_0, x_1)
    #return (np.abs(ptc - x_1) * phi_field_0 + np.abs(ptc - x_0) * phi_field_1)/np.abs(x_1 - x_0)

def calcbeta(x_0, x_1, fc):
    return math.sqrt((x_0[0] - fc[0])**2 + (x_0[1] - fc[1])**2)/math.sqrt((x_0[0] - x_1[0])**2 + (x_0[1] - x_1[1])**2)

def linInt(beta, phi_field_0, phi_field_1):
    return beta * phi_field_1 + (1 - beta) * phi_field_0

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
            fc = cell_centre_mesh.face_table.centroid[i_face]
            # x_i = cell_centre_mesh.face_table.cc_unit_vector[i_face]  # direction vector
            # face_contribution = lineInter(phi_field_0, phi_field_1, x_0, x_1, x_i, fc)
            beta = calcbeta(x_0, x_1, fc)
            face_contribution = linInt(beta, phi_field_0, phi_field_1)
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

def vertex_phi(vertex_coordinates, cell_centroids, cell_phi_function, vertex_cell_connect, max_vertex):
    vertex_phi = np.zeros(shape=(max_vertex, ))
    for i, i_cell in enumerate(vertex_cell_connect):
        # if len(i_cell) == 4:
        tot_dist = 0
        phi_dist = 0
        for j, j_cell in enumerate(i_cell):
            vert_coord = vertex_coordinates[i]
            cell_coord = cell_centroids[j_cell]
            dist = math.sqrt((cell_coord[0] - vert_coord[0])**2 + (cell_coord[1] - vert_coord[1])**2)
            cell_phi = cell_phi_function[j_cell]
            tot_dist += 1/dist
            phi_dist += (1/dist) * cell_phi
        vertex_phi[i] = phi_dist/tot_dist
        # else:
        #     vertex_phi[i] = 0
    return vertex_phi

def node_GreenGauss(cell_centre_mesh):
    phi_gradient_field = np.zeros(shape=(cell_centre_mesh.cell_table.max_cell, 2), dtype=float)
    phi_boundary_field = boundary_phi_function(cell_centre_mesh)                # phi values for all boundaries
    cell_function = cell_phi_function(cell_centre_mesh)
    face_vertex_connect = cell_centre_mesh.face_table.connected_vertex

    vertex_coordinates = cell_centre_mesh.vertex_table.coordinate
    cell_centroids = cell_centre_mesh.cell_table.centroid
    vertex_cell_connect = cell_centre_mesh.vertex_table.connected_cell
    max_vertex = cell_centre_mesh.vertex_table.max_vertex
    phi_vertex = vertex_phi(vertex_coordinates, cell_centroids, cell_function, vertex_cell_connect, max_vertex)
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

        face_contribution = (phi_vertex[face_vertex_connect[i_face][0]] + phi_vertex[face_vertex_connect[i_face][1]])/2

        face_area = cell_centre_mesh.face_table.area[i_face]
        face_normal = cell_centre_mesh.face_table.normal[i_face]
        phi_gradient_field[i_cell_0] += face_contribution*face_area*face_normal/cell_centre_mesh.cell_table.volume[i_cell_0]
        phi_gradient_field[i_cell_1] -= face_contribution*face_area*face_normal/cell_centre_mesh.cell_table.volume[i_cell_1]
    return phi_gradient_field
