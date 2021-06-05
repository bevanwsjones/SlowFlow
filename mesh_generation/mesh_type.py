"""
SlowFlow
cell.py
Bevan Jones

Contains definitions for the cells of a mesh.
"""

from enum import Enum, auto

class MeshType(Enum):
    simplex = auto()
    cartesian = auto()


def mvertex(mesh_type):
    if mesh_type == MeshType.simplex:
        return 3
    elif mesh_type == MeshType.cartesian:
        return 4
    else:
        raise ValueError("Enumerator not found")


def mface(mesh_type):
    if mesh_type == MeshType.simplex:
        return 3
    elif mesh_type == MeshType.cartesian:
        return 4
    else:
        raise ValueError("Enumerator not found")
