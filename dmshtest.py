
from mesh_generation import mesh_generator as mg
from mesh_preprocessor import preprocessor as pp
from post_processor import graph as gr
import numpy as np
from gradient_algorithms import NewGG

[vertex_coordinates, cell_vertex_connectivity, cell_type] = \
    mg.setup_2d_simplex_mesh(10, hex = True, _start_co_ordinates=np.array([0.0, 0.0]), _domain_size=np.array([1.0, 1.0]))
cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)

phi_function = 1
phi_field, phi_boundary_field, dphi_analytical = NewGG.cell_boundary_face_phi_dphi_calculation(cell_centre_mesh, phi_function)

# NewGG.GreenGauss(cell_centre_mesh, 1, phi_function)
gr.plot_field(cell_centre_mesh, [phi_field])
# gr.plot_field(cell_centre_mesh, gr.generate_random_field(4, cell_centre_mesh.cell_table.max_cell,
#                                                         [True, False, True, False]))
