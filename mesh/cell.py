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
# filename: cell.py
# description: Definition of mesh cells.
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import mesh_type as mt


class CellTable:
    """
    The cell table, containing basic mesh cell geometry data.
    """

    def __init__(self, number_of_cell, mesh_type):
        """
        Initialises, allocates the memory, for the cells in the mesh given a number of cells and the mesh type.

        :param number_of_cell: Number of cells to allocate.
        :type number_of_cell: int
        :param mesh_type: The type of mesh, triangle, quadrilaterals, etc.
        :type mesh_type: mt.MeshType
        """

        self.type = mesh_type
        self.max_cell = number_of_cell
        self.max_ghost_cell = 0
        self.connected_face = -1 * np.ones(shape=[number_of_cell, mt.number_of_face(mesh_type)], dtype=int)
        self.connected_vertex = -1 * np.ones(shape=[number_of_cell, mt.number_of_vertex(mesh_type)], dtype=int)
        self.boundary = np.zeros(shape=[number_of_cell, ], dtype=bool)
        self.volume = np.zeros(shape=[number_of_cell, ], dtype=float)
        self.coordinate = np.zeros([number_of_cell, 2], dtype=float)

    def add_ghost_cells(self, number_of_ghost):
        """
        Adds ghost cells to the table by resizing all the data containers and recording the number of ghost cells.

        :param number_of_ghost the number of ghost cells to add.
        :type number_of_ghost: int
        """

        if number_of_ghost < 0:
            raise ValueError("Adding negative ghost cells.")
        self.max_ghost_cell = number_of_ghost
        self.connected_face = \
            np.concatenate((self.connected_face, -1 * np.ones(shape=[self.max_ghost_cell, 3], dtype=int)))
        self.connected_vertex = \
            np.concatenate((self.connected_vertex, -1 * np.ones(shape=[self.max_ghost_cell, 3], dtype=int)))
        self.boundary = np.concatenate((self.boundary, np.zeros(shape=[self.max_ghost_cell, ], dtype=bool)))
        self.volume = np.concatenate((self.volume, np.zeros(shape=[self.max_ghost_cell, ], dtype=float)))
        self.coordinate = np.concatenate((self.coordinate, np.zeros([self.max_ghost_cell, 2], dtype=float)))
