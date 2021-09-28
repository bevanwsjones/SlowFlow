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
# description: Definition of mesh_entities faces.
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np


class FaceTable:
    """
    The face table, containing basic mesh_entities cell face geometry data.
    """

    def __init__(self):
        """
        Default constructor
        """

        self.max_face = 0
        self.max_boundary_face = 0

        self.boundary = np.zeros([0, ], dtype=bool)
        self.connected_cell = -1*np.ones([0, 2], dtype=int)
        self.connected_vertex = -1*np.ones([0, 2], dtype=int)

        self.centroid = np.zeros([0, 2], dtype=float)
        self.cc_length = np.zeros([0, ], dtype=float)
        self.cc_unit_vector = np.zeros([0, 2], dtype=float)
        self.area = np.zeros([0, ], dtype=float)
        self.normal = np.zeros([0, 2], dtype=float)
