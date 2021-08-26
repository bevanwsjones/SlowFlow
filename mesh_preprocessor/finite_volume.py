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
# filename: finite_volume.py
# description: todo
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# Cell Central Co-Ordinate
# ----------------------------------------------------------------------------------------------------------------------

def calculate_edge_center(_cell_table, _vertex_table):
    # triangle center
    max_cell = _cell_table.max_cell
    _cell_table.coordinate[0:max_cell] = (_vertex_table.coordinate[_cell_table.connected_vertex[0:max_cell, 0]]
                                          + _vertex_table.coordinate[_cell_table.connected_vertex[0:max_cell, 1]]) / 2.0


def calculate_triangle_center(_cell_table, _vertex_table):
    # triangle center
    max_cell = _cell_table.max_cell
    _cell_table.coordinate[0:max_cell] = (_vertex_table.coordinate[_cell_table.connected_vertex[0:max_cell, 0]]
                                          + _vertex_table.coordinate[_cell_table.connected_vertex[0:max_cell, 1]]
                                          + _vertex_table.coordinate[_cell_table.connected_vertex[0:max_cell, 2]]) / 3.0


def calculate_quadrilateral_center(_cell_table, _vertex_table):
    # Quad center
    max_cell = _cell_table.max_cell
    _cell_table.coordinate[0:max_cell][0] = 0.50 * (
            _vertex_table.coordinate[_cell_table.connected_vertex[0:max_cell, 0]][0]
            + _vertex_table.coordinate[_cell_table.connected_vertex[0:max_cell, 1]][0])
    _cell_table.coordinate[0:max_cell][1] = 0.50 * (
            _vertex_table.coordinate[_cell_table.connected_vertex[0:max_cell, 1]][1]
            + _vertex_table.coordinate[_cell_table.connected_vertex[0:max_cell, 2]][1])


def calculate_hexagon_center(_cell_table, _vertex_table):
    raise RuntimeError("Still a todo")


def calculate_cell_center(_cell_table, _vertex_table):
    raise RuntimeError("Still a todo")


# ----------------------------------------------------------------------------------------------------------------------
# Cell Volume
# ----------------------------------------------------------------------------------------------------------------------

def calculate_edge_volume(_cell_table, _vertex_table):
    max_cell = _cell_table.max_cell
    _cell_table.volume[0:max_cell] = (_vertex_table.coordinate[_cell_table.connected_vertex[0:max_cell, 0]]
                                      - _vertex_table.coordinate[_cell_table.connected_vertex[0:max_cell, 1]])


def calculate_triangle_volume():
    raise RuntimeError("Still a todo")


def calculate_quadrilateral_volume():
    raise RuntimeError("Still a todo")


def calculate_hexagon_volume():
    raise RuntimeError("Still a todo")


def calculate_cell_volume(cell_table, vertex_table):
    raise RuntimeError("Still a todo")


# ----------------------------------------------------------------------------------------------------------------------
# Face Geometry
# ----------------------------------------------------------------------------------------------------------------------

def calculate_face_cell_cell_length(_face_table, _cell_co_ordinate):
    max_face = _face_table.max_face
    _face_table.length[0:max_face] = np.linalg.norm(_cell_co_ordinate[_face_table.connected_cell[0:max_face, 0]]
                                                    - _cell_co_ordinate[_face_table.connected_cell[0:max_face, 1]])

    raise RuntimeError("handle boundaries")


def calculate_face_cell_cell_tangent(_face_table, _cell_co_ordinate):
    max_face = _face_table.max_face
    _face_table.length[0:max_face] = (_cell_co_ordinate[_face_table.connected_cell[0:max_face, 1]]
                                      - _cell_co_ordinate[_face_table.connected_cell[0:max_face, 0]]) \
                                     / _face_table.length[0:max_face]
    raise RuntimeError("handle boundaries")


def calculate_face_area(_face_table, _vertex_co_ordinate):
    max_face = _face_table.max_face
    _face_table.length[0:max_face] = np.linalg.norm(_vertex_co_ordinate[_face_table.connected_vertex[0:max_face, 1]]
                                                    - _vertex_co_ordinate[_face_table.connected_vertex[0:max_face, 0]])


def calculate_face_normal(_face_table, _vertex_co_ordinate):
    max_face = _face_table.max_face
    _face_table.length[0:max_face] = np.cross(_vertex_co_ordinate[_face_table.connected_vertex[0:max_face, 1]]
                                              - _vertex_co_ordinate[_face_table.connected_vertex[0:max_face, 0]],
                                              np.array((0, 0, 1), dtype=float))[0:max_face, 2] / 2.0
    raise RuntimeError("dot check")


# ----------------------------------------------------------------------------------------------------------------------
# finite volume setup
# ----------------------------------------------------------------------------------------------------------------------

def setup_finite_volume_geometry(_mesh):
    calculate_cell_center(_mesh.cell_table, _mesh.vertex_table)

    calculate_face_cell_cell_length(_mesh.face_table, _mesh.cell_table.coordinate)
    calculate_face_cell_cell_tangent(_mesh.face_table, _mesh.cell_table.coordinate)
    calculate_face_area(_mesh.face_table, _mesh.vertex_table.vertex_coordinate)
    calculate_face_normal(_mesh.face_table, _mesh.vertex_table.vertex_coordinate)

    calculate_cell_volume(_mesh.cell_table, _mesh.vertex_table)
