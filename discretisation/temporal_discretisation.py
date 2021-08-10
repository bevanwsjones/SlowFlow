
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
# filename: temporal_discretisation.py
# description: todo
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Runge-Kutta explicit time integration
# ----------------------------------------------------------------------------------------------------------------------

stage_weights = [[1.0],
                 [1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0]]
stage_coefficient = [[],
                     [0.0, 0.5, 0.5, 1.0]]
stage_nodes = [[0.0],
               [0.0, 0.5, 0.5, 1.0]]


def method_to_index(mstage):
    """
    Look up method for the above RK tables.
    :param mstage: The maximum number of RK stages
    :return:
    """
    if mstage == 1:
        return 0
    elif mstage == 4:
        return 1
    else:
        raise ValueError("Unsupported Runge-Kutta method requested.")


def runge_kutta(phi_n, dphi_dt, phi_rhs, delta_time, istage, mstage):
    """
    Facilities the Runge-Kutta explicit time stepping, using reduced memory storage.

    phi_rhs += b_i*dphi/dt
    phi^(i+1) = phi^n + c_i * dphi/dt
    @i + 1 = miter: phi^(n+1) = phi_rhs

    phi^(i+1) - solution variable at the next iteration
    phi^n - Varaible at the start of the time step
    dphi/dt - Variable change with resepct to time
    phi_rhs - the accumulate weighted values of dphi/dt
    b_i - stage weight
    c_i - stage coefficent

    :param phi_n: Variable value at the start of the time step.
    :param dphi_dt: The change of the variable with respect to time.
    :param phi_rhs: The accumulated value for the RK method.
    :param delta_time: The time step size
    :param istage: The current iteration/stage, starting at 0.
    :param mstage: The maximum iteration/stage (eg RK4 = 4)
    :return: Variable to use at the next iteration (if last iteration then next time step), accumulated rhs.
    """

    index = method_to_index(mstage)

    if istage == 0:
        phi_rhs = 0.0

    phi_rhs += stage_weights[index][istage] * dphi_dt
    if istage < mstage - 1:
        return phi_n + delta_time * stage_coefficient[index][istage + 1] * dphi_dt, phi_rhs
    else:
        return phi_n + delta_time*phi_rhs, phi_rhs
