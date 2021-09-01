# ----------------------------------------------------------------------------------------------------------------------
#  This file is part of the SlowFlow distribution  (https://github.com/bevanwsjones/SlowFlow).
#  Copyright (c) 2020 Bevan Walter Stewart Jones.
#
#  This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation, version 3.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
#  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with this program. If not, see
#  <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------------------------------------------------
# filename: mesh_generator.py
# description: Contains functions for generating different types of meshes.
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import mesh.cell as cl
import meshio
import dmsh
import optimesh


# ----------------------------------------------------------------------------------------------------------------------
# Mesh Generation Functions
# ----------------------------------------------------------------------------------------------------------------------

def geometric_series_sum(_common_factor, _n_terms, _ratio):
    """
    Computes the sum of a geometric series.

    :param _common_factor: The common factor to the geometric series.
    :type _common_factor: float
    :param _n_terms: The number of terms in the series.
    :type _n_terms: int
    :param _ratio: The ratio between successive terms in the series.
    :type _ratio: float
    :return: The sum of the series.
    :type: float
    """

    if _ratio == 1.0:
        raise ZeroDivisionError("Divide by zero, _ratio: " + str(_ratio))

    return _common_factor * (1.0 - _ratio ** float(_n_terms)) / (1.0 - _ratio)


def geometric_series_common_factor(_series_sum, _n_terms, _ratio):
    """
    Computes the common factor of a geometric series.

    :param _series_sum: The sum of the series for the first n terms.
    :type _series_sum: float
    :param _n_terms: The number of terms in the series.
    :type _n_terms: int
    :param _ratio: The ratio between successive terms in the series.
    :type _ratio: float
    :return: The sum of the series.
    :type: float
    """

    if _ratio ** _n_terms == 1.0:
        raise ZeroDivisionError("Divide by zero, _ratio: " + str(_ratio))

    return _series_sum * (1.0 - _ratio) / (1.0 - _ratio ** float(_n_terms))

# def double_geometric_series_sum(_common_factor_0, _n_terms_0, _ratio_0, _common_factor_1, _n_terms_1, _ratio_1):
#     """
#     Computes the sum of a geometric series.
#
#     :param _common_factor: The common factor to the geometric series.
#     :type _common_factor: float
#     :param _n_terms: The number of terms in the series.
#     :type _n_terms: int
#     :param _ratio: The ratio between successive terms in the series.
#     :type _ratio: float
#     :return: The sum of the series.
#     :type: float
#     """
#
#     if _ratio_0 == 1.0:
#         raise ZeroDivisionError("Divide by zero, _ratio: " + str(_ratio_0))
#     if _ratio_0 == 1.0:
#         raise ZeroDivisionError("Divide by zero, _ratio: " + str(_ratio_1))
#
#     return _common_factor * (1.0 - _ratio ** float(_n_terms)) / (1.0 - _ratio)

def double_geometric_series_common_factors(_series_sum, _n_terms, _ratio_0, _ratio_1):
    """
    Computes the common factor of a geometric series.

    :param _series_sum: The sum of the series for the first n terms.
    :type _series_sum: float
    :param _n_terms: The number of terms in the series.
    :type _n_terms: int
    :param _ratio: The ratio between successive terms in the series.
    :type _ratio: float
    :return: The sum of the series.
    :type: float
    """

    if _ratio_0 == 1.0:
        raise ZeroDivisionError("Divide by zero, _ratio_0: " + str(_ratio_0))
    if _ratio_1 == 1.0:
        raise ZeroDivisionError("Divide by zero, _ratio_1: " + str(_ratio_1))

    if _n_terms % 2 is 0:
        n_terms_0 = _n_terms / 2
        n_terms_1 = n_terms_0
    else:
        n_terms_0 = _n_terms / 2 + 1
        n_terms_1 = n_terms_0

    fact = _ratio_0 ** float(n_terms_0 - 1) / _ratio_1 ** float(n_terms_1 - 1)
    _common_factor_0 = _series_sum / ((1.0 - _ratio_0 ** float(n_terms_0)) / (1.0 - _ratio_0) +
                                      fact * (1.0 - _ratio_1 ** float(n_terms_1)) / (1.0 - _ratio_1))
    _common_factor_1 = fact * _common_factor_0

    return _common_factor_0, _common_factor_1

# ----------------------------------------------------------------------------------------------------------------------
# 1D Mesh Generation
# ----------------------------------------------------------------------------------------------------------------------


def setup_1d_mesh(_number_of_cells, _start_co_ordinate=0.0, _domain_size=1.0, _ratio=1.0):
    """
    Generates the vertices, cell-vertex connectivity as well as the cell type.

    :param _number_of_cells: The number of cells in the mesh.
    :type _number_of_cells: int
    :param _start_co_ordinate: The starting co-ordinate (lower x bound).
    :type _start_co_ordinate: float
    :param _domain_size: The total length of the domain.
    :type _domain_size: float
    :param _ratio: The ratio of change between successive (increasing in x) cell sizes.
    :type _ratio: float
    :return:
    """

    # Layout memory
    cell_type = np.empty(shape=(_number_of_cells, 1), dtype=cl.CellType)
    cell_vertex_connectivity = np.zeros(shape=(_number_of_cells, 2), dtype=int)
    vertex_coordinates = np.zeros(shape=(_number_of_cells + 1, 2), dtype=float)

    # Determine starting delta x.
    delta_x = geometric_series_common_factor(_domain_size, _number_of_cells, _ratio) \
        if _ratio != 1.0 else _domain_size / float(_number_of_cells)

    # Compute vertex positions for the domain.
    for i_vertex, vertex in enumerate(vertex_coordinates):
        current_length = geometric_series_sum(delta_x, i_vertex + 2, _ratio) \
            if _ratio != 1.0 else float(i_vertex) * delta_x
        vertex[0] = _start_co_ordinate + current_length
        vertex[1] = 0.0

    # Setup cells
    cell_type[:] = cl.CellType.edge
    cell_vertex_connectivity[:, ] = [[i_cell, i_cell + 1] for i_cell in range(_number_of_cells)]

    return [vertex_coordinates, cell_vertex_connectivity, cell_type]


# ----------------------------------------------------------------------------------------------------------------------
# 2D Mesh Generation - Cartesian Mesh
# ----------------------------------------------------------------------------------------------------------------------


def setup_2d_cartesian_mesh(_number_of_cells, _start_co_ordinates=None, _domain_size=None, _ratio=None):
    """
    Generates the vertices and cells for a 2D structured cartesian mesh. The

    :param _number_of_cells: Number of cells in x and y
    :type _number_of_cells: list
    :param _start_co_ordinates: bottom left co-ordinate of the domain.
    :type _start_co_ordinates: np.array
    :param _domain_size:
    :type _domain_size: np.array
    :param _ratio: The ratio of change between successive (increasing in x,y) cell sizes.
    :type _ratio: list
    :return:
    """

    # Assign default value.
    if _start_co_ordinates is None:
        _start_co_ordinates = np.array((0.0, 0.0))
    if _domain_size is None:
        _domain_size = np.array((1.0, 1.0))
    if _ratio is None:
        _ratio = [1.0, 1.0]

    # Layout memory
    _total_cells = _number_of_cells[0]*_number_of_cells[1]
    cell_type = np.empty(shape=(_total_cells, 1), dtype=cl.CellType)
    cell_vertex_connectivity = np.zeros(shape=(_total_cells, 2), dtype=int)
    vertex_coordinates = np.zeros(shape=((_number_of_cells[0] + 1)*(_number_of_cells[1] + 1), 2), dtype=float)

    # Determine starting delta x.
    delta_x = geometric_series_common_factor(_domain_size[0], _number_of_cells[0] + 2, _ratio[0]) \
        if _ratio[0] != 1.0 else _domain_size[0] / float(_number_of_cells[0])
    delta_y = geometric_series_common_factor(_domain_size[1], _number_of_cells[1] + 2, _ratio[1]) \
        if _ratio[1] != 1.0 else _domain_size[1] / float(_number_of_cells[1])

    # Compute vertex positions for the domain.
    for i_vertex, vertex in enumerate(vertex_coordinates):

        x_index = i_vertex % (_number_of_cells[0] + 1)
        y_index = int(i_vertex / _number_of_cells[1])

        current_length_x = geometric_series_sum(delta_x, x_index, _ratio[0]) \
            if _ratio[0] != 1.0 else float(x_index) * delta_x

        current_length_y = geometric_series_sum(delta_y, y_index, _ratio[1]) \
            if _ratio[1] != 1.0 else float(y_index) * delta_y

        print(str(x_index) + "  " + str(y_index) + "  " + str(current_length_x))

        vertex[0] = _start_co_ordinates[0] + current_length_x
        vertex[1] = _start_co_ordinates[1] + current_length_y



    # Setup cells
    cell_type[:] = cl.CellType.quadrilateral
    # cell_vertex_connectivity[:, ] = [[i_cell, i_cell + 1] for i_cell in range(_number_of_cells)]

    return [vertex_coordinates, cell_vertex_connectivity, cell_type]
    # new_mesh = mesh.Mesh()
    # return new_mesh


# ----------------------------------------------------------------------------------------------------------------------
# 2D Mesh Generation - Simplex Mesh
# ----------------------------------------------------------------------------------------------------------------------


def setup_2d_simplex_mesh():
    """

    :return:
    """
    geo = dmsh.Polygon([[0.0, 0.0], [1.0, 0.0], [1.0 + np.cos(np.pi / 3.0), np.sin(np.pi / 3.0)],
                        [1.0, 2.0 * np.sin(np.pi / 3.0)], [0.0, 2.0 * np.sin(np.pi / 3.0)],
                        [0.0 - np.cos(np.pi / 3.0), np.sin(np.pi / 3.0)]])
    verticies, cells = dmsh.generate(geo, 0.5)
    dmsh.helpers.show(verticies, cells, geo)  # Put on to print mesh
    # new_mesh = mesh.Mesh()
    # return new_mesh
