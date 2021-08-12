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

    def __init__(self, _number_of_cells, _mesh_type):
        """
        Initialises, allocates the memory, for the cells in the mesh given a number of cells and the mesh type.

        :param _number_of_cells: Number of cells to allocate.
        :type _number_of_cells: int
        :param _mesh_type: The type of mesh, triangle, quadrilaterals, etc.
        :type _mesh_type: mt.MeshType
        """

        self.type = _mesh_type
        self.max_cell = _number_of_cells
        self.max_ghost_cell = 0
        self.connected_face = -1 * np.ones(shape=[_number_of_cells, mt.number_of_face(_mesh_type)], dtype=int)
        self.connected_vertex = -1 * np.ones(shape=[_number_of_cells, mt.number_of_vertex(_mesh_type)], dtype=int)
        self.boundary = np.zeros(shape=[_number_of_cells, ], dtype=bool)
        self.volume = np.zeros(shape=[_number_of_cells, ], dtype=float)
        self.coordinate = np.zeros([_number_of_cells, 2], dtype=float)

    def add_ghost_cells(self, _number_of_ghosts):
        """
        Adds ghost cells to the table by resizing all the data containers and recording the number of ghost cells.

        :param _number_of_ghosts the number of ghost cells to add.
        :type _number_of_ghosts: int
        """

        if _number_of_ghosts < 0:
            raise ValueError("Adding negative ghost cells.")
        self.max_ghost_cell = _number_of_ghosts
        self.connected_face = \
            np.concatenate((self.connected_face, -1 * np.ones(shape=[self.max_ghost_cell, 3], dtype=int)))
        self.connected_vertex = \
            np.concatenate((self.connected_vertex, -1 * np.ones(shape=[self.max_ghost_cell, 3], dtype=int)))
        self.boundary = np.concatenate((self.boundary, np.zeros(shape=[self.max_ghost_cell, ], dtype=bool)))
        self.volume = np.concatenate((self.volume, np.zeros(shape=[self.max_ghost_cell, ], dtype=float)))
        self.coordinate = np.concatenate((self.coordinate, np.zeros([self.max_ghost_cell, 2], dtype=float)))
