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
# filename: preprocessor.py
# description: Contains the pre-processor function which convert a basic geometric mesh with vertices to a full mesh.
# ----------------------------------------------------------------------------------------------------------------------

import connectivity as ct
import finite_volume as fv
from mesh import mesh
from mesh import cell as cl


# ----------------------------------------------------------------------------------------------------------------------
# Mesh Connectivity Preprocessing
# ----------------------------------------------------------------------------------------------------------------------

def connect_mesh(_cell_vertex_connectivity, _number_of_vertices, _cell_type):
    """

    :param _cell_vertex_connectivity:
    :param _cell_vertex_connectivity:
    :param _number_of_vertices:
    :param _cell_type:
    :return:
    """
    vertex_cell_connectivity = ct.connect_vertices_to_cells(_cell_vertex_connectivity, _number_of_vertices)
    vertex_vertex_connectivity = ct.connect_vertices_to_vertices(_cell_vertex_connectivity, _number_of_vertices)
    [number_of_faces, number_of_boundary_faces] = ct.compute_number_of_faces(vertex_cell_connectivity,
                                                                             _cell_vertex_connectivity)
    face_vertex_connectivity = ct.connect_faces_to_vertex(_cell_vertex_connectivity)
    face_cell_connectivity = ct.connect_faces_to_cells(vertex_cell_connectivity, face_vertex_connectivity)
    cell_face_connectivity = ct.connect_cells_to_faces(face_cell_connectivity, len(_cell_vertex_connectivity),
                                                       _cell_type)
    face_boundary = ct.determine_face_boundary_status(face_cell_connectivity)
    cell_boundary = ct.determine_cell_boundary_status(face_cell_connectivity, face_boundary)
    return [vertex_cell_connectivity, vertex_vertex_connectivity, face_vertex_connectivity, face_cell_connectivity,
            cell_face_connectivity, face_boundary, cell_boundary, number_of_faces, number_of_boundary_faces]


# ----------------------------------------------------------------------------------------------------------------------
# Mesh Geometry-Preprocessing
# ----------------------------------------------------------------------------------------------------------------------

def setup_finite_volume_geometry(_cell_table, _face_table, _vertex_table):
    """

    :param _cell_table:
    :param _face_table:
    :param _vertex_table:
    :return:
    """

    _cell_table.centroid = fv.calculate_cell_centroid(_cell_table.type, _cell_table.connected_vertex,
                                                      _vertex_table.coordinate)
    _face_table.cc_length = fv.calculate_face_cell_cell_length(_face_table.max_boundary_face,
                                                               _face_table.connected_cell, _face_table.connected_vertex,
                                                               _cell_table.centroid, _vertex_table.coordinate)
    _face_table.cc_unit_vector = fv.calculate_face_cell_cell_unit_vector(_face_table.max_boundary_face,
                                                                         _face_table.connected_cell,
                                                                         _face_table.connected_vertex,
                                                                         _cell_table.centroid, _vertex_table.coordinate)
    _face_table.area = fv.calculate_face_area(_cell_table.type, _face_table.connected_vertex, _vertex_table.coordinate)
    _face_table.normal = fv.calculate_face_normal(_cell_table.type, _face_table.connected_vertex,
                                                  _vertex_table.coordinate, _face_table.cc_unit_vector)
    if _cell_table.type == cl.CellType.edge:
        _cell_table.volume = fv.calculate_edge_volume(_cell_table.connected_vertex, _vertex_table.coordinate)
    else:
        _cell_table.volume = fv.calculate_2d_cell_volume(_cell_table.connected_face, _face_table.connected_cell,
                                                         _face_table.connected_vertex, _cell_table.centroid,
                                                         _face_table.normal, _face_table.area, _vertex_table.coordinate)


# ----------------------------------------------------------------------------------------------------------------------
# Preprocessing Cell Centred Finite Volume Meshes
# ----------------------------------------------------------------------------------------------------------------------

def setup_cell_centred_finite_volume_mesh(_vertex_coordinates, _cell_vertex_connectivity, _cell_type):
    """

    :param _vertex_coordinates:
    :type _vertex_coordinates:
    :param _cell_vertex_connectivity:
    :type _cell_vertex_connectivity:
    :param _cell_type:
    :type _cell_type:
    :return:
    """

    new_mesh = mesh.CellCenteredMesh(_cell_type)
    new_mesh.vertex_table.coordinate = _vertex_coordinates
    new_mesh.vertex_table.max_vertex = len(_vertex_coordinates)
    new_mesh.cell_table.max_cell = len(_cell_vertex_connectivity)
    new_mesh.cell_table.connected_vertex = _cell_vertex_connectivity

    [new_mesh.vertex_table.connected_cell, new_mesh.vertex_table.connected_vertex, new_mesh.face_table.connected_vertex,
     new_mesh.face_table.connected_cell, new_mesh.cell_table.connected_face, new_mesh.face_table.boundary,
     new_mesh.cell_table.boundary, new_mesh.face_table.max_face, new_mesh.face_table.max_boundary_face] =\
        connect_mesh(_cell_vertex_connectivity, new_mesh.vertex_table.max_vertex, _cell_type)

    setup_finite_volume_geometry(new_mesh.cell_table, new_mesh.face_table, new_mesh.vertex_table)

    return new_mesh
