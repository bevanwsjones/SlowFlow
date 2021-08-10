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
# filename: face.py
# description: todo
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np


class Face:

    def __init__(self):
        self.boundary = False
        self.connected_node = np.array([-1, -1],  dtype=int)
        self.node_first = np.empty(shape=[0], dtype=int)
        self.node_last = np.empty(shape=[0], dtype=int)

        self.length = 0.0
        self.tangent = np.zeros([2, 1], dtype=float)
        self.coefficient = np.zeros([2, 1], dtype=float)


class FaceTable:
    """
    Contains both cell face geometric and connectivity data.
    """

    def __init__(self, max_face):
        self.max_face = max_face
        self.boundary = np.zeros([max_face, ], dtype=bool)
        self.connected_cell = np.zeros([max_face, 2], dtype=int)
        self.connected_vertex = np.zeros([max_face, 2],  dtype=int)
        self.cell_first = np.empty([max_face, 0], dtype=int)
        self.cell_last = np.empty([max_face, 0], dtype=int)

        self.length = np.zeros([max_face, ], dtype=float)
        self.tangent = np.zeros([max_face, 2], dtype=float)
        self.coefficient = np.zeros([max_face, 2], dtype=float)
