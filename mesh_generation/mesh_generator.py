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
import mesh_entities.cell as cl


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
    :param _ratio_0: The ratio between successive terms in the series.
    :type _ratio_0: float
    :param _ratio_1: The ratio between successive terms in the series.
    :type _ratio_1: float
    :return: The sum of the series.
    :type: float
    """

    if _ratio_0 == 1.0:
        raise ZeroDivisionError("Divide by zero, _ratio_0: " + str(_ratio_0))
    if _ratio_1 == 1.0:
        raise ZeroDivisionError("Divide by zero, _ratio_1: " + str(_ratio_1))

    if _n_terms % 2 == 0:
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
# Standard Mesh Transformation Functions
# ----------------------------------------------------------------------------------------------------------------------

def structured(_n, _x_0, _ds, _x):
    """
    Standard structured mesh_entities, the passed co-ordinate is the returned co-ordindate.

    :param _n: number of cells in x and y, [number_of_cells_x, number of cells_y].
    :type _n: list
    :param _x_0: Starting co-ordinate of structured grid, [start_co_ordinate_x, start_co_ordinate_y].
    :type _x_0: list
    :param _ds: Domain size/length, [size_x, size_y]
    :type _ds: list
    :param _x: current co-ordinate to transform, [co_ordinate_x, co_ordinate_y].
    :type _x: numpy.array
    :return: parsed co-ordinate, [co_ordinate_x, co_ordinate_y].
    :type: numpy.array
    """

    return _x


def stretch(_ratio, _n, _x_0, _ds, _x):
    """
    Stretches the mesh_entities in the two cardinal directions, each successive cell size in respective directions differs by the
    respective constant ratio. This is akin to a geometric series.

    :param _ratio: stretching ratio in x and y, [stretch_x, stretch_y].
    :type _ratio: numpy.array
    :param _n: number of cells in x and y, [number_of_cells_x, number of cells_y].
    :type _n: list
    :param _x_0: Starting co-ordinate of structured grid, [start_co_ordinate_x, start_co_ordinate_y].
    :type _x_0: numpy.array
    :param _ds: Domain size/length, [size_x, size_y]
    :type _ds: numpy.array
    :param _x: current co-ordinate to transform, [co_ordinate_x, co_ordinate_y].
    :type _x: numpy.array
    :return: transformed co-ordinate, [co_ordinate_x, co_ordinate_y].
    :type: numpy.array
    """

    if _ratio[0] == 1.0 or _ratio[1] == 1.0:
        raise RuntimeError("Attempting to use a stretch ratio of 1.0. " + str(_ratio))

    return np.array([_x_0[0] + _ds[0] * (1.0 - _ratio[0]) / (1.0 - np.power(_ratio[0], _n[0])) *
                     (1.0 - np.power(_ratio[0], ((_x[0] - _x_0[0]) / (_ds[0] / _n[0])))) / (1.0 - _ratio[0]),
                     _x_0[1] + _ds[1] * (1.0 - _ratio[1]) / (1.0 - np.power(_ratio[1], _n[1])) *
                     (1.0 - np.power(_ratio[1], ((_x[1] - _x_0[1]) / (_ds[1] / _n[1])))) / (1.0 - _ratio[1])])


def parallelogram(_normalise, _gradient, _n, _x_0, _ds, _x):
    """
    Converts the domain into a parallelogram, by using the two gradients. The first gradient controls the y-ordinate as
    the x-ordinates are laid down and visa versa for the second gradient and the x- and y-ordinates.

    Note:
    This function operates in 2 modes, normalised and un-normalised. In normalised mode the start co-ordinate _x_0 and
    domain size _dx are maintained, however the gradient will be 'adjusted'. In the un-normalised state _x_0 and the
    gradients _gradient are maintained, [x'_i = x_i + m_1*(y_i - y_0), y'_i = y_i + m_0*(x_i - x_0], i.e. m_1 and m_2
    are unaltered and the domain size is adjusted.

    :param _normalise: Domain is normalised if set to true, else remains un-normalised.
    :type _normalise: bool
    :param _gradient: gradient of cell faces in x and y, [gradient_x (horizontal edges), gradient_y (vertical edges)].
    :type _gradient: numpy.array
    :param _n: number of cells in x and y, [number_of_cells_x, number of cells_y].
    :type _n: list
    :param _x_0: Starting co-ordinate of structured grid, [start_co_ordinate_x, start_co_ordinate_y].
    :type _x_0: numpy.array
    :param _ds: Domain size/length, [size_x, size_y]
    :type _ds: numpy.array
    :param _x: current co-ordinate to transform, [co_ordinate_x, co_ordinate_y].
    :type _x: numpy.array
    :return: transformed co-ordinate, [co_ordinate_x, co_ordinate_y].
    :type: numpy.array
    """

    if _gradient[0] >= 1.0 or _gradient[1] >= 1.0:
        raise RuntimeError("Attempting to use a skewing ratio of 1.0 or more. " + str(_gradient))

    if _normalise:
        box_size = np.array([_ds[0] + _gradient[1] * _ds[1], _ds[1] + _gradient[0] * _ds[0]])
        return np.array([_x_0[0] + _ds[0] * (_x[0] - _x_0[0] + _gradient[1] * (_x[1] - _x_0[1])) / box_size[0],
                         _x_0[1] + _ds[1] * (_x[1] - _x_0[1] + _gradient[0] * (_x[0] - _x_0[0])) / box_size[1]])
    else:
        return np.array([_x[0] + _gradient[1] * (_x[1] - _x_0[1]), _x[1] + _gradient[0] * (_x[0] - _x_0[0])])


# ----------------------------------------------------------------------------------------------------------------------
# 1D Mesh Generation
# ----------------------------------------------------------------------------------------------------------------------

def setup_1d_mesh(_number_of_cells, _start_co_ordinate=0.0, _domain_size=1.0):
    """
    Not implemented
    """

    raise NotImplementedError("Needs to be implemented.")


# ----------------------------------------------------------------------------------------------------------------------
# 2D Mesh Generation - Structured Meshes
# ----------------------------------------------------------------------------------------------------------------------

def setup_2d_cartesian_mesh(_number_of_cells, _start_co_ordinates=None, _domain_size=None, _transform=structured):
    """
    Generates the vertices and cells-vertex connectivity for a 2D structured cartesian mesh_entities. A regular equi-spaced
    Cartesian grid is first generated using the start co-ordinates and the domain size. The cell-vertex connectivity is
    ordered in an anti-clockwise manner.

    The grid can be transformed by passing in a transformation lambda. In this case each vertex will be mapped to a new
    co-ordinate using the pared function.

    Note: This can cause the anti-clockwise ordering to be violated. Thus the user should ensure the transformation
          maintains ordering if desired.

    :param _number_of_cells: Number of cells in x and y
    :type _number_of_cells: numpy.array
    :param _start_co_ordinates: bottom left co-ordinate of the domain. (defaults to 0.0, 0.0)
    :type _start_co_ordinates: numpy.array
    :param _domain_size: Size of the domain in x and y (defaults to 1.0, 1.0)
    :type _domain_size: numpy.array
    :param _transform: The transformation lambda for grid points, must take in and return a numpy array of co-ordinates.
    :type _transform: lambda = f(_number_of_cells, _start_co_ordiantes, _domain_size, [x_i, y_i]): return [x'_i, y'_i]
    :return: [list of vertex co-ordinate, list of cell-vertex connectivity, type of cells]
    :type: [numpy.array, numpy.array. cell.CellType]
    """

    # Compute geometric constants.
    if _start_co_ordinates is None:
        _start_co_ordinates = np.array((0.0, 0.0))
    if _domain_size is None:
        _domain_size = np.array((1.0, 1.0))

    # Determine starting delta x and y.
    number_vertices = (_number_of_cells[0] + 1) * (_number_of_cells[1] + 1)
    delta = np.array([_domain_size[0] / float(_number_of_cells[0]), _domain_size[1] / float(_number_of_cells[1])])

    # Compute and return the co-ordinates, connectivity, and cell type for the mesh_entities.
    return [
        np.array([_transform(_number_of_cells, _start_co_ordinates, _domain_size,
                             np.array([_start_co_ordinates[0]
                                       + float(i_vertex % (_number_of_cells[0] + 1)) * delta[0],
                                       _start_co_ordinates[1]
                                       + float(int(i_vertex / (_number_of_cells[0] + 1))) * delta[1]]))
                  for i_vertex in range(number_vertices)], dtype=float),
        np.array([np.array([i_x + i_y * (_number_of_cells[0] + 1),
                            i_x + 1 + i_y * (_number_of_cells[0] + 1),
                            i_x + 1 + (i_y + 1) * (_number_of_cells[0] + 1),
                            i_x + (i_y + 1) * (_number_of_cells[0] + 1)])
                  for i_y in range(_number_of_cells[1])
                  for i_x in range(_number_of_cells[0])], dtype=int),
        cl.CellType.quadrilateral
    ]


# ----------------------------------------------------------------------------------------------------------------------
# 2D Mesh Generation - Simplex Mesh
# ----------------------------------------------------------------------------------------------------------------------

def setup_2d_simplex_mesh():
    """
    Not implemented
    """

    raise NotImplementedError("Needs to be implemented, if implementing see meshio, dmsh, and optimesh")
