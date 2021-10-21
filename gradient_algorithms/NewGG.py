import numpy as np
import math

def exp_cos(_coor_list):
    return np.exp(_coor_list[:, 0]) * np.cos(_coor_list[:, 1])
def gradient_exp_cos(_coor_list):
    true_field = np.zeros(shape=(len(_coor_list), 2), dtype=float)
    true_field[:,0] = np.exp(_coor_list[:, 0]) * np.cos(_coor_list[:, 1])
    true_field[:, 1] = - np.exp(_coor_list[:, 0]) * np.sin(_coor_list[:, 1])
    return true_field

def sin_cos(_coor_list):
    return np.sin(_coor_list[:, 0]) + np.cos(_coor_list[:, 1])
def gradient_sin_cos(_coor_list):
    true_field = np.zeros(shape=(len(_coor_list), 2), dtype=float)
    true_field[:, 0] = np.cos(_coor_list[:, 0])
    true_field[:, 1] = -1*np.sin(_coor_list[:, 1])
    return true_field

def xycubed(_coor_list):
    return 2 * _coor_list[:, 0]**3 + 2 * _coor_list[:, 1]**3
def gradient_xycubed(_coor_list):
    true_field = np.zeros(shape=(len(_coor_list), 2), dtype=float)
    true_field[:, 0] = 6*_coor_list[:, 0]**2
    true_field[:, 1] = 6*_coor_list[:, 1]**2
    return true_field

def tanh(_coor_list):
    return np.tanh(_coor_list[:, 0]) + np.tanh(_coor_list[:, 1])
def gradient_tanh(_coor_list):
    true_field = np.zeros(shape=(len(_coor_list), 2), dtype=float)
    true_field[:, 0] = 1/np.cosh(_coor_list[:, 0])**2
    true_field[:, 1] = 1/np.cosh(_coor_list[:, 1])**2
    return true_field

def cell_boundary_face_phi_dphi_calculation(_cell_centre_mesh, _phi_function = 0):
    boundary_no = _cell_centre_mesh.face_table.max_boundary_face
    cell_centroids = _cell_centre_mesh.cell_table.centroid
    boundary_face_centroids = _cell_centre_mesh.face_table.centroid[0:boundary_no]
    
    if _phi_function == 0:
        return [exp_cos(cell_centroids), exp_cos(boundary_face_centroids), gradient_exp_cos(cell_centroids)]
    elif _phi_function == 1:
        return [sin_cos(cell_centroids), sin_cos(boundary_face_centroids), gradient_sin_cos(cell_centroids)]
    elif _phi_function == 2:
        return [xycubed(cell_centroids), xycubed(boundary_face_centroids), gradient_xycubed(cell_centroids)]
    elif _phi_function == 3:
        return [tanh(cell_centroids), tanh(boundary_face_centroids), gradient_tanh(cell_centroids)]
    else:
        raise NotImplemented("More phi fields to be entered")

def arithMean(phi_field_0, phi_field_1):
    return 0.5 * (phi_field_0 + phi_field_1)

def calcbeta(x_0, x_1, fc):
    # BUT CHECK THIS => np.linalg.norm(x_0 - fc) same as math.sqrt((vec_1[0] - vec_0[0])**2 + (vec_1[1] - vec_0[1])**2) 
    # Point on line x_i = x_0 + lambda * t  where t = x_1 - x_0 | lambda \in [0, 1] and x_0 and x_1 are the line segement end points.
    return math.sqrt((fc[0] - x_0[0])**2 + (fc[1] - x_0[1])**2)/math.sqrt((x_1[0] - x_0[0])**2 + (x_1[1] - x_0[1])**2)

def linInt(beta, phi_field_0, phi_field_1):
    # linInt(phi_field_0, phi_field_1, x_0, x_1, x_i) 
    # CHECK below not supposed to be (1 - beta)*phi_field_1 + beta*phi_field_0?+
    return beta * phi_field_1 + (1 - beta) * phi_field_0

def GreenGauss(cell_centre_mesh, status = 0, phi_function = 0):
    phi_gradient_field = np.zeros(shape=(cell_centre_mesh.cell_table.max_cell, 2), dtype=float)
    phi_field, phi_boundary_field, dphi_analytical = cell_boundary_face_phi_dphi_calculation(cell_centre_mesh, phi_function)

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
            face_contribution = arithMean(phi_field_0, phi_field_1)     # can turn this into a face operator function?
        elif status == 1:
            x_0 = cell_centre_mesh.cell_table.centroid[i_cell_0]        # determine coordinates for cell centroids
            x_1 = cell_centre_mesh.cell_table.centroid[i_cell_1]
            fc = cell_centre_mesh.face_table.centroid[i_face]
            beta = calcbeta(x_0, x_1, fc)
            face_contribution = linInt(beta, phi_field_0, phi_field_1)
        face_area = cell_centre_mesh.face_table.area[i_face]
        face_normal = cell_centre_mesh.face_table.normal[i_face]
        phi_gradient_field[i_cell_0] += face_contribution*face_area*face_normal/cell_centre_mesh.cell_table.volume[i_cell_0]
        phi_gradient_field[i_cell_1] -= face_contribution*face_area*face_normal/cell_centre_mesh.cell_table.volume[i_cell_1]
    return phi_gradient_field, dphi_analytical

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
    return vertex_phi

# NODE GAUSS TO BE IMPLEMENTED
def node_GreenGauss(cell_centre_mesh, phi_function = 0):
    phi_gradient_field = np.zeros(shape=(cell_centre_mesh.cell_table.max_cell, 2), dtype=float)
    cell_function, phi_boundary_field, dphi_analytical = cell_boundary_face_phi_dphi_calculation(cell_centre_mesh, phi_function)

    face_vertex_connect = cell_centre_mesh.face_table.connected_vertex
    # print(face_vertex_connect)
    vertex_coordinates = cell_centre_mesh.vertex_table.coordinate
    cell_centroids = cell_centre_mesh.cell_table.centroid
    vertex_cell_connect = cell_centre_mesh.vertex_table.connected_cell
    max_vertex = cell_centre_mesh.vertex_table.max_vertex

    phi_vertex = vertex_phi(vertex_coordinates, cell_centroids, cell_function, vertex_cell_connect, max_vertex)
    # print(vertex_cell_connect)
    # print(phi_vertex)
    # face_contribution is the analytical value - Dirichlet Condition
    for ibound_face in range(cell_centre_mesh.face_table.max_boundary_face):
        i_cell_0 = cell_centre_mesh.face_table.connected_cell[ibound_face][0]
        face_area = cell_centre_mesh.face_table.area[ibound_face]
        face_normal = cell_centre_mesh.face_table.normal[ibound_face]
        face_contribution = (phi_vertex[face_vertex_connect[ibound_face][0]] + phi_vertex[face_vertex_connect[ibound_face][1]]) / 2

        # face_contribution = phi_boundary_field[ibound_face]
        # use the average of two analytical values from the two vertices connected to the boundary face. More consistent.
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
    return phi_gradient_field, dphi_analytical
