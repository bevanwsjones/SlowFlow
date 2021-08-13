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
# filename: vertex.py
# description: Definition of mesh vertices.
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np


class VertexTable:
    """
    The vertex table, containing basic mesh vertex geometry data.
    """

    def __init__(self, _number_of_vertices):
        """
        Initialises, allocates the memory, for the vertices in the mesh given a number of faces.

        :param _number_of_vertices: Number of vertices to allocate.
        :type _number_of_vertices: int
        """

        self.max_vertex = _number_of_vertices

        self.connected_cell = [np.empty(shape=(0,), dtype=int) for _ in range(_number_of_vertices)]  # not square
        self.coordinate = np.zeros([_number_of_vertices, 2], dtype=float)
