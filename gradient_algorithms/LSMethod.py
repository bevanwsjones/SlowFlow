import numpy as np
from mesh_preprocessor import preprocessor as pp
from mesh_generation import mesh_generator as mg

# vector field that is applied to the mesh
def phi_field(max_item, centroid):
    phi_field = np.zeros(shape=(max_item, ), dtype=float)
    # phi_field[:] = centroid[0:max_item, 0]**2 + centroid[0:max_item, 1]**2
    # phi_field[:] = np.sin(centroid[0:max_item, 0]) + np.cos(centroid[0:max_item, 1])
    phi_field[:] = np.exp(centroid[0:max_item, 0])*np.cos(centroid[0:max_item, 1])
    # phi_field[:] = centroid[:, 0]**2 + centroid[:, 1]**2
    return phi_field

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
def cell_ls(cell_centre_mesh):
    # store required metrics from the cell_centre_mesh
    connected_face = cell_centre_mesh.cell_table.connected_face
    connected_cell = cell_centre_mesh.face_table.connected_cell
    face_bound_max = cell_centre_mesh.face_table.max_boundary_face
    cell_centroids = cell_centre_mesh.cell_table.centroid
    face_centroids = cell_centre_mesh.face_table.centroid
    cell_max_no = cell_centre_mesh.cell_table.max_cell

    # Evaluate the phi_field at cell centroids and BC face centroids
    cells_phi_field = phi_field(cell_max_no, cell_centroids)
    bound_faces_phi_field = phi_field(face_bound_max, face_centroids)

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
    return store_phi

# number_of_cells, start_co_ordinate, domain_size = [3, 3], [0.0, 0.0], [1.0, 1.0]
# [vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size)
# cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
#
# #print(cell_centre_mesh.cell_table.connected_face)
# print(cell_centre_mesh.face_table.connected_cell)