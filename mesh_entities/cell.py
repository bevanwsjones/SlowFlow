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
# description: Definition of mesh_entities cells.
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

    if _cell_type.value == CellType.edge.value:
        return 2
    elif _cell_type.value == CellType.triangle.value:
        return 3
    if _cell_type.value == CellType.quadrilateral.value:
        return 4
    elif _cell_type.value == CellType.hexagon.value:
        return 6
    else:
        raise ValueError("Enumerator not found")


class CellTable:
    """
    The cell table, containing basic mesh_entities cell geometry data.
    """

    def __init__(self, _cell_type):
        """
        Sets the cell type for the cells for the mesh_entities.

        :param _cell_type: The type of mesh_entities, triangle, quadrilaterals, etc.
        :type _cell_type: mt.MeshType
        """

        self.type = _cell_type
        self.max_cell = 0

        self.connected_face = -1 * np.ones(shape=[0, number_of_vertex_face(_cell_type)], dtype=int)
        self.connected_vertex = -1 * np.ones(shape=[0, number_of_vertex_face(_cell_type)], dtype=int)
        self.boundary = np.zeros(shape=[0, ], dtype=bool)

        self.volume = np.zeros(shape=[0, ], dtype=float)
        self.centroid = np.zeros([0, 2], dtype=float)
