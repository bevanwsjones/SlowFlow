import unittest as ut
from gradient_algorithms import GreenGauss as gg
from mesh_generation import mesh_generator as mg
from gradient_algorithms import modGreenGauss as modGG
from mesh_preprocessor import preprocessor as pp

class test_function_test(ut.TestCase):
    def test_phi_function(self):
        x, y = 2, 3
        grad_phi = gg.phi_function(x, y)
        self.assertEqual(round(grad_phi[0], 4), -0.4161)  # add assertion here
        self.assertEqual(round(grad_phi[1], 4), -0.1411)

class test_2Dgg_singlecase(ut.TestCase):
    def test_return_cells(self):
        number_of_cells, start_co_ordinate, domain_size = [3, 3], [0.5, 1.5], [3, 3]
        neighbour_cells = gg.GreenGauss_neighbourcells(number_of_cells, start_co_ordinate, domain_size)
        self.assertEqual(neighbour_cells[0][1], 1)
        self.assertEqual(neighbour_cells[1][1], 3)
        self.assertEqual(neighbour_cells[2][1], 5)
        self.assertEqual(neighbour_cells[3][1], 7)

        self.assertEqual(neighbour_cells[0][0], 15)
        self.assertEqual(neighbour_cells[1][0], 17)
        self.assertEqual(neighbour_cells[2][0], 19)
        self.assertEqual(neighbour_cells[3][0], 20)
    def test_2DGG(self):
        number_of_cells, start_co_ordinate, domain_size = [3, 3], [0.5, 1.5], [3, 3]
        grad_phi = gg.GreenGauss_2D(number_of_cells, start_co_ordinate, domain_size)
        self.assertEqual(round(grad_phi[0],3), -0.350)
        self.assertEqual(round(grad_phi[1],3), -0.119)


class test_2DGG_multicase(ut.TestCase):
    def test_2DGG(self):
        number_of_cells, start_co_ordinate, domain_size = [3, 3], [0.5, 1.5], [3, 3]
        [vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size)
        cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
        phi_functions = modGG.cell_phi_function(cell_centre_mesh)
        print(phi_functions)
        # test works if phi(x, y) = cos(x)*i - sin(y)*j
        self.assertEqual(round(phi_functions[4][0], 4), -0.4161)
        self.assertEqual(round(phi_functions[4][1], 4), -0.1411)
    def test_interp(self):              # make sure function returns true/interpolated faces
        number_of_cells, start_co_ordinate, domain_size = [3, 3], [0.5, 1.5], [3, 3]
        [vertex_coordinates, cell_vertex_connectivity, cell_type] = mg.setup_2d_cartesian_mesh(number_of_cells, start_co_ordinate, domain_size)
        cell_centre_mesh = pp.setup_cell_centred_finite_volume_mesh(vertex_coordinates, cell_vertex_connectivity, cell_type)
        h, j = modGG.GreenGauss(cell_centre_mesh)
        self.assertEqual(h.all(), j.all())
