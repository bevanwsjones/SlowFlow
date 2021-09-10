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
# filename: reimann_solver.py
# description: todo
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from compressible_solver import node as cn


def flux_1d(state, gamma):
    _velocity = cn.velocity(state)
    pressure = cn.pressure(state, gamma)
    return np.array((state[0]*_velocity[0], state[1]*_velocity[0] + pressure, state[2]*_velocity[0],
                    (state[3] + pressure)*_velocity[0]))


def hllc(state_l, state_r, gamma, face_coefficient):

    unit_face_coefficient = face_coefficient / np.linalg.norm(face_coefficient) # projection vector
    rotation_matrix = np.array(([unit_face_coefficient[0], unit_face_coefficient[1]],
                                [-unit_face_coefficient[1], unit_face_coefficient[0]]))
    rotation_matrix_inverse = np.array(([unit_face_coefficient[0], -unit_face_coefficient[1]],
                                        [unit_face_coefficient[1], unit_face_coefficient[0]]))

    momentum_l_rotated = np.dot(rotation_matrix, state_l[1:3])
    momentum_r_rotated = np.dot(rotation_matrix, state_r[1:3])
    state_l_reorient = np.array((state_l[0], momentum_l_rotated[0], momentum_l_rotated[1], state_l[3]))
    state_r_reorient = np.array((state_r[0], momentum_r_rotated[0], momentum_r_rotated[1], state_r[3]))

    density_l = state_l_reorient[cn.density]
    density_r = state_r_reorient[cn.density]
    root_density_l = np.sqrt(density_l)
    root_density_r = np.sqrt(density_r)
    velocity_l = cn.velocity(state_l_reorient)[0]
    velocity_r = cn.velocity(state_r_reorient)[0]
    momentum_l = state_l_reorient[1]
    momentum_r = state_r_reorient[1]
    pressure_l = cn.pressure(state_l_reorient, gamma)
    pressure_r = cn.pressure(state_r_reorient, gamma)
    enthalpy_l = cn.enthalpy(state_l_reorient, gamma)
    enthalpy_r = cn.enthalpy(state_r_reorient, gamma)

    velocity_bar = (root_density_l * velocity_l + root_density_r * velocity_r) / (root_density_l + root_density_r)
    enthalpy_bar = (root_density_l * enthalpy_l + root_density_r * enthalpy_r) / (root_density_l + root_density_r)
    acoustic_velocity_bar = np.sqrt((gamma - 1.0) * (enthalpy_bar - 0.5 * velocity_bar ** 2))
    s_l = velocity_bar - acoustic_velocity_bar
    s_r = velocity_bar + acoustic_velocity_bar
    s_star = (pressure_r - pressure_l + momentum_l * (s_l - velocity_l) - momentum_r * (s_r - velocity_r)) / \
             (density_l * (s_l - velocity_l) - density_r * (s_r - velocity_r))

    if 0 <= s_l:
        flux = flux_1d(state_l_reorient, gamma)
        momentum_flux_reoriented = np.dot(rotation_matrix_inverse, flux[1:3])
        return np.array([flux[0], momentum_flux_reoriented[0], momentum_flux_reoriented[1], flux[3]])
    elif s_l <= 0 <= s_star:
        flux_l = flux_1d(state_l_reorient, gamma)
        d_star = np.array([0.0, 1.0, 0.0, s_star], dtype=float)
        p_lr = 0.5 * (pressure_l + pressure_r + density_l * (s_l - velocity_l) * (s_star - velocity_l)
                      + density_r * (s_r - velocity_r) * (s_star - velocity_r))
        f_star_l = (s_star * (s_l * state_l_reorient - flux_l) + s_l * p_lr * d_star) / (s_l - s_star)
        f_star_l_deconstructed = np.dot(rotation_matrix_inverse, f_star_l[1:3])
        return np.array([f_star_l[0], f_star_l_deconstructed[0], f_star_l_deconstructed[1], f_star_l[3]])
    elif s_star <= 0 <= s_r:
        flux_r = flux_1d(state_r_reorient, gamma)
        d_star = np.array([0.0, 1.0, 0.0, s_star], dtype=float)
        p_lr = 0.5 * (pressure_l + pressure_r + density_l * (s_l - velocity_l) * (s_star - velocity_l)
                      + density_r * (s_r - velocity_r) * (s_star - velocity_r))
        f_star_r = (s_star * (s_r * state_r_reorient - flux_r) + s_r * p_lr * d_star) / (s_r - s_star)
        f_star_r_deconstructed = np.dot(rotation_matrix_inverse, f_star_r[1:3])
        return np.array([f_star_r[0], f_star_r_deconstructed[0], f_star_r_deconstructed[1], f_star_r[3]])
    elif 0 >= s_r:
        flux = flux_1d(state_r_reorient, gamma)
        momentum_flux_reoriented = np.dot(rotation_matrix_inverse, flux[1:3])
        return np.array([flux[0], momentum_flux_reoriented[0], momentum_flux_reoriented[1], flux[3]])
    else:
        ValueError("WHAT")
