import numpy as np


class Face:

    def __init__(self):
        self.boundary = False
        self.connected_node = np.array([-1, -1],  dtype=int)
        self.node_first = np.empty(shape=[0], dtype=int)
        self.node_last = np.empty(shape=[0], dtype=int)

        self.length = 0.0
        self.tangent = np.zeros([2, 1], dtype=float)
        self.coefficient = np.zeros([2, 1], dtype=float)


class FaceTable:
    """
    Contains both cell face geometric and connectivity data.
    """

    def __init__(self, max_face):
        self.max_face = max_face
        self.boundary = np.zeros([max_face, ], dtype=bool)
        self.connected_cell = np.zeros([max_face, 2], dtype=int)
        self.connected_vertex = np.zeros([max_face, 2],  dtype=int)
        self.cell_first = np.empty([max_face, 0], dtype=int)
        self.cell_last = np.empty([max_face, 0], dtype=int)

        self.length = np.zeros([max_face, ], dtype=float)
        self.tangent = np.zeros([max_face, 2], dtype=float)
        self.coefficient = np.zeros([max_face, 2], dtype=float)
