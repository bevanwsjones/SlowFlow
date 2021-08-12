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
import mesh.mesh as mesh
import meshio
import dmsh
import optimesh


# ----------------------------------------------------------------------------------------------------------------------
# 1D Mesh Generation
# ----------------------------------------------------------------------------------------------------------------------


def setup_1d_unit_mesh(_number_of_cells, _start_co_ordinate=0.0, _domain_size=1.0, _growth_factor=1.0):
    """

    :param _number_of_cells:
    :type _number_of_cells:
    :param _start_co_ordinate:
    :type _start_co_ordinate:
    :param _domain_size:
    :type _domain_size:
    :param _growth_factor:
    :type _growth_factor:
    :return:
    """

    new_mesh = mesh.Mesh()
    return new_mesh


# ----------------------------------------------------------------------------------------------------------------------
# 2D Mesh Generation - Cartesian Mesh
# ----------------------------------------------------------------------------------------------------------------------


def setup_2d_cartesian_mesh(_number_of_cells, _start_co_ordinates=None, _domain_size=None, _growth_factor=None):
    """

    :param _number_of_cells:
    :type _number_of_cells:
    :param _start_co_ordinates:
    :type _start_co_ordinates:
    :param _domain_size:
    :type _domain_size:
    :param _growth_factor:
    :type _growth_factor:
    :return:
    """

    if _growth_factor is None:
        _growth_factor = [1.0, 1.0]
    if _domain_size is None:
        _domain_size = [1.0, 1.0]
    if _start_co_ordinates is None:
        _start_co_ordinates = [0.0, 0.0]

    new_mesh = mesh.Mesh()
    return new_mesh


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
    new_mesh = mesh.Mesh()
    return new_mesh



