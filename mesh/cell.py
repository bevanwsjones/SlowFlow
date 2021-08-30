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
from enum import Enum, auto


class CellType(Enum):
    """
    Enumerators for the supported cell types.
    """

    edge = auto()
    triangle = auto()
    quadrilateral = auto()
    hexagon = auto()


def number_of_vertex_face(_cell_type):
    """
    Returns the number of vertices and/or faces for a given cell type

    :param _cell_type: cell type enumerator.
    :type _cell_type: CellType
    :return: number of vertices/faces for the cell type
    """

    if _cell_type == CellType.edge:
        return 2
    elif _cell_type == CellType.triangle:
        return 3
    if _cell_type == CellType.quadrilateral:
        return 4
    elif _cell_type == CellType.hexagon:
        return 6
    else:
        raise ValueError("Enumerator not found")


class CellTable:
    """
    The cell table, containing basic mesh cell geometry data.
    """

    def __init__(self, _number_of_cells, _cell_type):
        """
        Initialises, allocates the memory, for the cells in the mesh given a number of cells and the mesh type.

        :param _number_of_cells: Number of cells to allocate.
        :type _number_of_cells: int
        :param _cell_type: The type of mesh, triangle, quadrilaterals, etc.
        :type _cell_type: mt.MeshType
        """

        self.type = _cell_type
        self.max_cell = _number_of_cells

        self.connected_face = -1 * np.ones(shape=[_number_of_cells, number_of_vertex_face(_cell_type)], dtype=int)
        self.connected_vertex = -1 * np.ones(shape=[_number_of_cells, number_of_vertex_face(_cell_type)], dtype=int)
        self.boundary = np.zeros(shape=[_number_of_cells, ], dtype=bool)

        self.volume = np.zeros(shape=[_number_of_cells, ], dtype=float)
        self.coordinate = np.zeros([_number_of_cells, 2], dtype=float)
