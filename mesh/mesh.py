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
# filename: mesh.py
# description: Contains all data for the mesh
# ----------------------------------------------------------------------------------------------------------------------

import cell as ct
import face as ft
import vertex as vt


class Mesh:
    """
    Structure to hold all mesh entities.
    """

    def __init__(self, _number_of_cells, _number_of_faces, _number_of_vertices, _mesh_type):
        """
        Allocates initial memory for all mesh entities.

        :param _number_of_cells: Number of cells to allocate.
        :type _number_of_cells: int
        :param _number_of_faces: Number of faces to allocate.
        :type _number_of_faces: int
        :param _number_of_vertices: Number of vertices to allocate.
        :type _number_of_vertices: int
        :param _mesh_type: Mesh type enumerator.
        :type _mesh_type: MeshType
        """

        self.cell_table = ct.CellTable(_number_of_cells, _mesh_type)
        self.face_table = ft.FaceTable(_number_of_faces)
        self.vertex_table = vt.VertexTable(_number_of_vertices)
        self.mesh_type = _mesh_type

