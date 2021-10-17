
from mesh_generation import mesh_generator as mg
from mesh_preprocessor import preprocessor as pp
from post_processor import graph as gr
import numpy as np

[vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_simplex_mesh(10, _start_co_ordinates=np.array([0.0, 0.0]), _domain_size=np.array([1.0, 1.0]))
cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
gr.plot_field(cell_centre_mesh, gr.generate_random_field(4, cell_centre_mesh.cell_table.max_cell,
                                                         [True, False, True, False]))
