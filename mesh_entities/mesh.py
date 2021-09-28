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
# filename: mesh_entities.py
# description: Contains all data for the mesh_entities
# ----------------------------------------------------------------------------------------------------------------------

from mesh_entities import cell as ct
from mesh_entities import face as ft
from mesh_entities import vertex as vt


class CellCenteredMesh:
    """
    Structure to hold all mesh_entities entities.
    """

    def __init__(self, _cell_type):
        """
        Constructor to set the cell type

        :param _cell_type: Cell type enumerator.
        :type _cell_type: cell.CellType
        """

        self.cell_table = ct.CellTable(_cell_type)
        self.face_table = ft.FaceTable()
        self.vertex_table = vt.VertexTable()
