import numpy as np


class VertexTable:
    """
    Contains cell vertex geometric and connectivity data.
    """

    def __init__(self, max_vertex):
        self.max_vertex = max_vertex
        self.connected_cell = [np.empty(shape=(0,), dtype=int) for _ in range(max_vertex)]  # not square

        self.coordinate = np.zeros([max_vertex, 2], dtype=float)
