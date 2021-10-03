import unittest as ut
from mesh_generation import mesh_generator as mg
from gradient_algorithms import NewGG
from mesh_preprocessor import preprocessor as pp

class test_2DGG_multicase(ut.TestCase):
    def test_2DGG(self):
        number_of_cells, start_co_ordinate, domain_size = [3, 3], [0.5, 1.5], [3, 3]
        [vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size)
        cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
        phi_functions = NewGG.cell_phi_function(cell_centre_mesh)
        print(phi_functions)
        # test works if phi(x, y) = cos(x)*i - sin(y)*j
        self.assertEqual(round(phi_functions[4][0], 4), -0.4161)
        self.assertEqual(round(phi_functions[4][1], 4), -0.1411)
    def test_interp(self):              # make sure function returns true/interpolated faces
        number_of_cells, start_co_ordinate, domain_size = [3, 3], [0.5, 1.5], [3, 3]
        [vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size)
        cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
        h, j = NewGG.GreenGauss(cell_centre_mesh)
        self.assertEqual(h.all(), j.all())
