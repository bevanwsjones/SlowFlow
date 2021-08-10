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
# description: Definition of mesh faces.
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np


class FaceTable:
    """
    The face table, containing basic mesh cell face geometry data.
    """

    def __init__(self, number_of_face):
        """
        Initialises, allocates the memory, for the faces in the mesh given a number of faces.

        :param number_of_face: Number of faces to allocate.
        :type number_of_face: int
        """
        self.max_face = number_of_face
        self.boundary = np.zeros([number_of_face, ], dtype=bool)
        self.connected_cell = np.zeros([number_of_face, 2], dtype=int)
        self.connected_vertex = np.zeros([number_of_face, 2], dtype=int)
        self.cell_first = np.empty([number_of_face, 0], dtype=int)
        self.cell_last = np.empty([number_of_face, 0], dtype=int)

        self.length = np.zeros([number_of_face, ], dtype=float)
        self.tangent = np.zeros([number_of_face, 2], dtype=float)
        self.coefficient = np.zeros([number_of_face, 2], dtype=float)