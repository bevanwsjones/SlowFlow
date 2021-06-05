"""
SlowFlow
cell.py
Bevan Jones

Contains definitions for the cells of a mesh.
"""

import numpy as np
from mesh_type import MeshType as mt


class CellTable:
    """
    The node table, containing basic mesh geometry data.
    """

    def __init__(self, max_cell, mesh_type):
        self.type = mesh_type
        self.max_cell = max_cell
        self.max_ghost_cell = 0
        self.connected_face = -1 * np.ones(shape=[max_cell, mt.mFace(mesh_type)], dtype=int)
        self.connected_vertex = -1 * np.ones(shape=[max_cell, mt.mVertex(mesh_type)], dtype=int)
        self.boundary = np.zeros(shape=[max_cell, ], dtype=bool)
        self.volume = np.zeros(shape=[max_cell, ], dtype=float)
        self.coordinate = np.zeros([max_cell, 2], dtype=float)

    def add_ghost_cells(self, max_ghost):
        """
        Adds ghost cells to the table by resizing all the data containers and recording the number of ghost cells.

        :param max_ghost the number of ghost cells to add.
        """
        if max_ghost < 0:
            raise ValueError("Adding negative ghost cells.")
        self.max_ghost_cell = max_ghost
        self.connected_face = \
            np.concatenate((self.connected_face, -1 * np.ones(shape=[self.max_ghost_cell, 3], dtype=int)))
        self.connected_vertex = \
            np.concatenate((self.connected_vertex, -1 * np.ones(shape=[self.max_ghost_cell, 3], dtype=int)))
        self.boundary = np.concatenate((self.boundary, np.zeros(shape=[self.max_ghost_cell, ], dtype=bool)))
        self.volume = np.concatenate((self.volume, np.zeros(shape=[self.max_ghost_cell, ], dtype=float)))
        self.coordinate = np.concatenate((self.coordinate, np.zeros([self.max_ghost_cell, 2], dtype=float)))
