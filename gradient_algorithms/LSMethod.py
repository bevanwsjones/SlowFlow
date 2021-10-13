import numpy as np
from gradient_algorithms import NewGG

def ind_cell_LS():
    phi_neighbours = np.array([18, 20, 10, 8])
    phi_centre_cell = np.ones(4)*13
    phi_diff = phi_neighbours - phi_centre_cell
    distance_matrix = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    dist_trans = distance_matrix.T
    G = np.matmul(dist_trans, distance_matrix)
    G_inv = np.linalg.inv(G)
    G_dist_t = np.matmul(G_inv, dist_trans)
    grad_phi = np.matmul(G_dist_t, phi_diff)
    return grad_phi

def inv_cell(dist, cells_phi_neighbour, cell_phi_centre):
    cell_array = np.ones(4) * cell_phi_centre
    phi_diff = (cells_phi_neighbour - cell_array).T
    dist_trans = dist.T
    G = np.matmul(dist_trans, dist)
    G_inv = np.linalg.inv(G)
    G_dist_t = np.matmul(G_inv, dist_trans)
    grad_phi = np.matmul(G_dist_t, phi_diff)
    return grad_phi.T

# from face connectivity table (for each face), only return the cell number that neighbours the centre cell
def cell_face_neighbour(i_cell, face_cell_connect):
    face_cell_connect = face_cell_connect[face_cell_connect != i_cell]
    return face_cell_connect

# LS unweighted method
def cell_ls(cell_centre_mesh, phi_function):
    # store required metrics from the cell_centre_mesh
    connected_face = cell_centre_mesh.cell_table.connected_face
    connected_cell = cell_centre_mesh.face_table.connected_cell
    face_bound_max = cell_centre_mesh.face_table.max_boundary_face
    cell_max_no = cell_centre_mesh.cell_table.max_cell

    # Evaluate the phi_field at cell centroids and BC face centroids
    cells_phi_field, bound_faces_phi_field, dphi_analytical = NewGG.cell_boundary_face_phi_dphi_calculation(cell_centre_mesh, phi_function)
    store_phi = np.zeros(shape=(cell_max_no, 2))
    # distance vector for each face-cell
    dist_matrix = cell_centre_mesh.face_table.cc_length[:, None] * cell_centre_mesh.face_table.cc_unit_vector

    for i, i_face in enumerate(connected_face):
        neighbour_phi = np.zeros(shape=(1, 4))      # assuming four neighbour cells (can be generalised)
        dist_array_store = np.zeros(shape=(1, 4, 2))
        for j, j_face in enumerate(i_face):
            if j_face < face_bound_max:
                 adjacent_phi = bound_faces_phi_field[j_face]
                 neighbour_cell = i
            else:
                neighbour_cell = cell_face_neighbour(i, connected_cell[j_face])
                adjacent_phi = cells_phi_field[neighbour_cell]
            if neighbour_cell < i:
                dist_array = -1*dist_matrix[j_face]
            else:
                dist_array = dist_matrix[j_face]
            neighbour_phi[0][j] = adjacent_phi
            dist_array_store[0][j] = dist_array
        store_phi[i] = inv_cell(dist_array_store[0], neighbour_phi, cells_phi_field[i])
    return store_phi, dphi_analytical
