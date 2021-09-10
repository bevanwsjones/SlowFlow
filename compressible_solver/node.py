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
# filename: node.py
# description: todo
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import copy as cp

gamma = 1.4

density = 0
momentum_x = 1
momentum_y = 2
density_energy = 3
max_variables = 4

n = 0
rhs = n + max_variables
residual = rhs + max_variables
np1 = residual + max_variables
variable_container_size = np1 + max_variables


def velocity(state_variables):
    return np.array((state_variables[momentum_x] / state_variables[density],
                     state_variables[momentum_y] / state_variables[density]), dtype=float)


def momentum(state_variables):
    return np.array((state_variables[momentum_x], state_variables[momentum_y]), dtype=float)


def acoustic_speed(state_variables):
    acoustic_speed_squared = (gamma - 1.0) * (enthalpy(state_variables, gamma)
                                              - 0.5 * np.dot(velocity(state_variables), velocity(state_variables)))
    if acoustic_speed_squared < 0.0:
        raise ValueError("Negative acoustic speed: "
                         "\nEnthalpy:       " + str(enthalpy(state_variables, gamma)) +
                         "\nVelocity:       " + str(velocity(state_variables)) +
                         "\nAcoustic speed: " + str(acoustic_speed_squared))
    return np.sqrt(acoustic_speed_squared)


def internal_energy(state_variables):
    return state_variables[density_energy] / state_variables[density] - 0.5 * np.dot(velocity(state_variables),
                                                                                     velocity(state_variables))


def pressure(state_variables):
    return (gamma - 1.0) * state_variables[density] * internal_energy(state_variables)


def energy(state_variables):
    return state_variables[density_energy] / state_variables[density]


def enthalpy(state_variables):
    return (energy(state_variables) + pressure(state_variables)) / state_variables[density]


def flux(state_variables):
    _velocity = velocity(state_variables)
    return np.outer(state_variables, _velocity) + pressure(state_variables) * np.array((np.zeros(2), np.eye(2)[0],
                                                                                        np.eye(2)[1], _velocity))


class Node:

    def __init__(self, dimension):
        self.connected_face = np.zeros(shape=[dimension * 2, ], dtype=int)
        self.boundary = False
        self.corner = False
        self.volume = 0.0
        self.coordinate = np.zeros([dimension, 1], dtype=float)


class CompressibleNode(Node):

    def __init__(self, dimension):
        super().__init__(dimension)
        self.variable = np.zeros([variable_container_size], dtype=float)
        self.gradient_variable = np.zeros([max_variables, 2], dtype=float)

    def density(self, temporal_index):
        return self.variable[temporal_index + density]

    def velocity(self, temporal_index):
        return velocity(self.variable[temporal_index:temporal_index + max_variables])

    def pressure(self, temporal_index, gamma):
        return pressure(self.variable[temporal_index:temporal_index + max_variables], gamma)

    def energy(self, temporal_index):
        return energy(self.variable[temporal_index:temporal_index + max_variables])

    def internal_energy(self, temporal_index):
        return internal_energy(self.variable[temporal_index:temporal_index + max_variables])

    def enthalpy(self, temporal_index, gamma):
        return enthalpy(self.variable[temporal_index:temporal_index + max_variables], gamma)


class CellTable:
    """
    The node table, containing basic mesh geometry data.
    """

    def __init__(self, max_cell):
        self.max_cell = max_cell
        self.max_ghost_cell = 0
        self.connected_face = -1*np.ones(shape=[max_cell, 3], dtype=int)
        self.connected_vertex = -1*np.ones(shape=[max_cell, 3], dtype=int)
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
            np.concatenate((self.connected_face, -1*np.ones(shape=[self.max_ghost_cell, 3], dtype=int)))
        self.connected_vertex =\
            np.concatenate((self.connected_vertex, -1*np.ones(shape=[self.max_ghost_cell, 3], dtype=int)))
        self.boundary = np.concatenate((self.boundary, np.zeros(shape=[self.max_ghost_cell, ], dtype=bool)))
        self.volume = np.concatenate((self.volume, np.zeros(shape=[self.max_ghost_cell, ], dtype=float)))
        self.coordinate = np.concatenate((self.coordinate, np.zeros([self.max_ghost_cell, 2], dtype=float)))


class CompressibleCellTable(CellTable):
    """
    The compressible node table, containing the conserved flow variables for the compressible Euler equations.
    Additionally, the class contains the additional data required for Runge-Kutta time stepping.
    """

    def __init__(self, max_cell, dimension):
        super().__init__(max_cell)
        self.variable = np.zeros([4, max_cell, max_variables], dtype=float)  # stored [n, rhs, residual, np1]
        self.variable_np1 = np.zeros([max_cell, max_variables], dtype=float)
        self.variable_residual = np.zeros([max_cell, max_variables], dtype=float)
        self.variable_rhs = np.zeros([max_cell, max_variables], dtype=float)
        self.gradient = np.zeros([max_cell, max_variables, 2], dtype=float)

    def advance_to_np1(self):
        """
        Copies the variables from n to np1, officially moving the simulation forward.
        """
        self.variable_n = cp.copy(self.variable_np1)

    # ------------------------------------------------------------------------------------------------------------------
    # Conserved Variable Accessors
    # ------------------------------------------------------------------------------------------------------------------

    def density(self, cell_index, temporal_index):
        return self.variable_n[temporal_index][cell_index][density]

    def momentum(self, cell_index, temporal_index):
        return np.array((self.variable_n[temporal_index][cell_index][momentum_x],
                         self.variable_n[temporal_index][cell_index][momentum_y]))

    def densityEnergy(self, cell_index, temporal_index):
        return self.variable_n[temporal_index][cell_index][density_energy]

    # ------------------------------------------------------------------------------------------------------------------
    # Primitive Variable Accessors
    # ------------------------------------------------------------------------------------------------------------------

    def velocity(self, cell_index, temporal_index):
        return velocity(self.variable[temporal_index][cell_index])

    def pressure(self, cell_index, temporal_index):
        return pressure(self.variable[temporal_index][cell_index])

    def energy(self, cell_index, temporal_index):
        return energy(self.variable[temporal_index][cell_index])

    # ------------------------------------------------------------------------------------------------------------------
    # Derived Quantities
    # ------------------------------------------------------------------------------------------------------------------

    def internal_energy(self, cell_index, temporal_index):
        return internal_energy(self.variable[temporal_index][cell_index])

    def enthalpy(self, cell_index, temporal_index):
        return enthalpy(self.variable[temporal_index][cell_index])
