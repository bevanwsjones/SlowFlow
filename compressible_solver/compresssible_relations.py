# ----------------------------------------------------------------------------------------------------------------------
#  This file is part of the SlowFlow distribution  (https://github.com/bevanwsjones/SlowFlow).
#  (Copyright (c) 2020 Bevan Walter Stewart Jones.
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
# filename: compressible_relations.py
# description: todo
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# State Variable Indexing
# ----------------------------------------------------------------------------------------------------------------------

density = 0
momentum_x = 1
momentum_y = 2
density_energy = 3
max_variables = 4

# ----------------------------------------------------------------------------------------------------------------------
# Physical Constants and Relations
# ----------------------------------------------------------------------------------------------------------------------

gamma = 1.4  # Ratio of Specific Heats


def acoustic_speed(state_variables):
    """
    Computes the speed of sound given a set of (conserved) variables.

    a = sqrt((gamma - 1) * e - 0.5 u.u)

    :param state_variables: [density, momentum_x, momentum_y, density_energy]
    :return: a
    """
    _velocity = velocity(state_variables)
    acoustic_speed_squared = (gamma - 1.0) * (enthalpy(state_variables) - 0.5 * np.dot(_velocity, _velocity))
    if acoustic_speed_squared < 0.0:
        raise ValueError("Negative acoustic speed: "
                         "\nEnthalpy:       " + str(enthalpy(state_variables)) +
                         "\nVelocity:       " + str(velocity(state_variables)) +
                         "\nAcoustic speed: " + str(acoustic_speed_squared))
    return np.sqrt(acoustic_speed_squared)


def flux(state_variables):
    """
    Computes the Flux given a set of (conserved) variables.

    U = [rho, rho u_x, rho u_y, rho E]
    u = [u_x, u_y]
    I = 2x2 Identity matrix
    F = Uu^T + p [0, I, u^T]

    :param state_variables: [density, momentum_x, momentum_y, density_energy]
    :return: F
    """
    _velocity = velocity(state_variables)
    return np.outer(state_variables, _velocity) + pressure(state_variables) * np.array((np.zeros(2), np.eye(2)[0],
                                                                                        np.eye(2)[1], _velocity))


def specific_internal_energy(state_variables):
    """
    Computes the specific internal energy given a set of (conserved) variables.

    e = rho * E / rho - 0.5 * u.u

    :param state_variables: [density, momentum_x, momentum_y, density_energy]
    :return: e
    """
    _velocity = velocity(state_variables)
    return state_variables[density_energy] / state_variables[density] - 0.5 * np.dot(_velocity, _velocity)


def enthalpy(state_variables):
    """
    Computes the enthalpy given a set of (conserved) variables.

    H = (e + p) / rho

    :param state_variables: state_variables: [density, momentum_x, momentum_y, density_energy]
    :return: H
    """
    return (specific_internal_energy(state_variables) + pressure(state_variables)) / state_variables[density]


def specific_energy(state_variables):
    """
    Computes the specific energy given a set of (conserved) variables.

    E = rho * E / rho

    :param state_variables: [density, momentum_x, momentum_y, density_energy]
    :return: E
    """
    return state_variables[density_energy] / state_variables[density]


# ----------------------------------------------------------------------------------------------------------------------
# Conserved Variables
# ----------------------------------------------------------------------------------------------------------------------


def momentum(state_variables):
    """
    Extracts the momentum from the given set of (conserved) variables.
    :param state_variables: [density, momentum_x, momentum_y, density_energy]
    :return: [rho u_x, rho u_y]
    """
    return np.array((state_variables[momentum_x], state_variables[momentum_y]), dtype=float)


# ----------------------------------------------------------------------------------------------------------------------
# Primitive Variables
# ----------------------------------------------------------------------------------------------------------------------


def velocity(state_variables):
    """
    Computes the velocity given a set of (conserved) variables.

    [u_x, u_y] = [rho * u_x, rho * u_y] / rho

    :param state_variables: [density, momentum_x, momentum_y, density_energy]
    :return: [u_x, u_y]
    """
    return momentum(state_variables) / state_variables[density]


def pressure(state_variables):
    """
    Computes from the pressure given a set of (conserved) variables.

    p = (gamma - 1) * rho * e

    :param state_variables: [density, momentum_x, momentum_y, density_energy]
    :return: p
    """
    return (gamma - 1.0) * state_variables[density] * specific_internal_energy(state_variables)
