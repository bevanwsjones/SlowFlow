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
# filename: mesh_type.py
# description: Contains helper functions and types for the preset supported mesh types.
# ----------------------------------------------------------------------------------------------------------------------

from enum import Enum, auto


class MeshType(Enum):
    """
    Enumerators for the supported mesh types.
    """

    simplex = auto()
    cartesian = auto()


def number_of_vertex(mesh_type):
    """
    Returns the number of nodes for a given mesh type, i.e. for triangle 3, quad 4.... hexagon is the bestagon?

    :param mesh_type: Mesh type enumerator.
    :type mesh_type: MeshType
    :return: number of vertices
    """

    if mesh_type == MeshType.simplex:
        return 3
    elif mesh_type == MeshType.cartesian:
        return 4
    else:
        raise ValueError("Enumerator not found")


def number_of_face(mesh_type):
    """
    Returns the number of faces for a given mesh type, i.e. for triangle 3, quad 4.... hexagon is the bestagon?

    :param mesh_type:
    :return: number of vertices
    """
    if mesh_type == MeshType.simplex:
        return 3
    elif mesh_type == MeshType.cartesian:
        return 4
    else:
        raise ValueError("Enumerator not found")
