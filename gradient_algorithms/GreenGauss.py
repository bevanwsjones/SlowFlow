# demonstration code for 1D gauss-green
# draft version

import mesh_generation.mesh_generator as mg
import numpy as np
from mesh_preprocessor import finite_volume as fv
from mesh_preprocessor import preprocessor as pp
import math
import matplotlib.pyplot as plt

# 1D GreenGauss Alogrithm

# construct mesh
def GreenGauss_1D(number_of_cells, start_co_ordinate, domain_size, ratio):
    number_of_cells, start_co_ordinate, domain_size, ratio = 100, 0.0, 10.0, 1.2
    y = mg.setup_1d_mesh(number_of_cells, start_co_ordinate, domain_size, ratio)
    vertex_coord = y[0]
    cell_vertex_connectivity = y[1]
    vol = fv.calculate_edge_volume(cell_vertex_connectivity, vertex_coord)
    centroids = fv.calculate_edge_centroid(cell_vertex_connectivity, vertex_coord)[:,0]

    # function that will be used to compute the gradient
    phi = lambda x: math.sin(x)         # function
    chi = lambda x: math.cos(x)         # analytical gradient to function

    # storing variable grad (for each cell)
    grad = np.zeros(shape=(number_of_cells, 2))
    new_chi = np.zeros(shape=(number_of_cells, 1))

    # iterating through the cell-vertex connectivity, compute the gradient for each cell using vertex co-ordinates
    for i_vertex, vertex in enumerate(cell_vertex_connectivity):
        grad[i_vertex, 0] = (1/vol[i_vertex])*(-phi(vertex_coord[vertex[0]][0])+phi(vertex_coord[vertex[1]][0]))
        grad[i_vertex, 1] = centroids[i_vertex]
        new_chi[i_vertex, 0] = chi(centroids[i_vertex])
# POST-PROCESSING
    # Data storage into csv.file - commented out
    # filedata = np.asarray(grad)
    # np.savetxt('Grad_data.csv', filedata, delimiter=",")

# Plotting Curves - both analytical and algorithm solutions
    x = np.linspace(start_co_ordinate, domain_size*1.2*1.2, 1000)
    y = np.cos(x)
    plt.figure(1)
    #plt.plot(centroids, new_chi[:,0], c = "blue")
    plt.plot(x, y, c = "blue")
    plt.scatter(centroids, grad[: , 0], c = "red",marker='x')
    plt.xlabel('x-axis - (m)')
    plt.ylabel('grad_phi')
    #plt.legend('numerical solution' , 'analytical solution')
    plt.tight_layout()
    plt.show()

def phi_function(x,y):
    phi = np.array([math.cos(x), -math.sin(y)])
    return phi

def chi_function(x,y):
    chi = np.array([math.sin(x), math.cos(y)])
    return chi

# return the neighbour cell numbers that borders the current cell faces [global_face_i, cell_centre_i]
def GreenGauss_neighbourcells(number_of_cells, start_co_ordinate, domain_size):
    [vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size)
    cell_no = 4
    cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
    connected_faces = cell_centre_mesh.cell_table.connected_face[cell_no]
    face_table = cell_centre_mesh.face_table.connected_cell
    neighbours = np.zeros(shape = (len(connected_faces), 2))
    for i, i_face in enumerate(connected_faces, 1):
        lst = face_table[i_face]
        neighbour_cell = np.delete(lst, np.where(lst == cell_no))   # delete the current cell number from array
        neighbours[i-1][0] = i_face
        neighbours[i-1][1] = neighbour_cell                         # assign face number, and cell centre number
    return neighbours

# return the vector field evaluated at face centroid that surrounds the centre cell. [i_face, chi(x), chi(y)]
def facecenter_coords(number_of_cells, start_co_ordinate, domain_size):
    neighbours = GreenGauss_neighbourcells(number_of_cells, start_co_ordinate, domain_size)
    [vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size)
    cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
    cell_centroid_table = cell_centre_mesh.cell_table.centroid
    cell_number = 4
    centre_coords = cell_centroid_table[int(cell_number)]           # find the cell center coords
    chi_centre = chi_function(centre_coords[0], centre_coords[1])   # evaluate the chi(x,y) for cell centroid
    chi_faces = np.zeros(shape = (len(neighbours), 3))
    for i, i_face in enumerate(neighbours):
        cell = neighbours[i][1]                                     # loop through the face-cell table
        coords = cell_centroid_table[int(cell)]                     # find coordinates of adjacent cell i
        chi = chi_function(coords[0], coords[1])                    # evaluate chi(x,y) for adjacent cell centroid
        chi_face = 0.5*(chi_centre + chi)                           # 1/2 * (chi_centre + chi_adjacent)
        chi_faces[i][0] = int(i_face[0])                            # add global face number and the grad_chi coords
        chi_faces[i][1] = chi_face[0]
        chi_faces[i][2] = chi_face[1]
    return chi_faces

# function returns the gradient for the cell-centroid [grad_x, grad_y]
def GreenGauss_2D(number_of_cells, start_co_ordinate, domain_size):
     chi_faces = facecenter_coords(number_of_cells, start_co_ordinate, domain_size)
     [vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size)
     neighbours = GreenGauss_neighbourcells(number_of_cells, start_co_ordinate, domain_size)
     cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
     cell_no = 4
     cell_vol = cell_centre_mesh.cell_table.volume[cell_no]                 # find cell volume
     faces = cell_centre_mesh.cell_table.connected_face[cell_no]
     sum = np.array([0, 0])                                                 # storage value for grad solution
     for i in range(len(chi_faces)):                                        # go through the [face_no, grad_coords]
         face_no = int(chi_faces[i][0])
         if int(neighbours[i][1]) > cell_no:                                # multiply by -1 to correct n vector direction
             normal_vector = cell_centre_mesh.face_table.normal[face_no]    # done if cell centre > neighbour cell
         else:
             normal_vector = -1 * cell_centre_mesh.face_table.normal[face_no]
         Face_area = cell_centre_mesh.face_table.area[face_no]              # find face area from table
         chi = np.array([chi_faces[i][1], chi_faces[i][2]])                 # store chi results in array
         grad_top = Face_area * np.multiply(chi, normal_vector)             # Use 2DGG equation A*chi(x,y)*normal
         sum = np.add(sum, grad_top)                                        # sum the results
     sum = sum/cell_vol
     print(sum)
     return sum


