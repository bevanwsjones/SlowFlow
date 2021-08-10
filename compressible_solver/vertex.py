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
# description: todo
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np


class VertexTable:
    """
    Contains cell vertex geometric and connectivity data.
    """

    def __init__(self, max_vertex):
        self.max_vertex = max_vertex
        self.connected_cell = [np.empty(shape=(0,), dtype=int) for _ in range(max_vertex)]  # not square

        self.coordinate = np.zeros([max_vertex, 2], dtype=float)
