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

    def __init__(self, _number_of_faces):
        """
        Initialises, allocates the memory, for the faces in the mesh given a number of faces.

        :param _number_of_faces: Number of faces to allocate.
        :type _number_of_faces: int
        """
        self.max_face = _number_of_faces

        self.boundary = np.zeros([_number_of_faces, ], dtype=bool)
        self.connected_cell = -1*np.ones([_number_of_faces, 2], dtype=int)
        self.connected_vertex = -1*np.ones([_number_of_faces, 2], dtype=int)

        self.cc_length = np.zeros([_number_of_faces, ], dtype=float)
        self.cc_unit_vector = np.zeros([_number_of_faces, 2], dtype=float)
        self.area = np.zeros([_number_of_faces, ], dtype=float)
        self.normal = np.zeros([_number_of_faces, 2], dtype=float)
