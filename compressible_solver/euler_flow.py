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
# filename: euler_flow.py
# description: todo
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import sys
from mesh_generation import mesh_generator as mg
from discretisation import face_operator as fo, temporal_discretisation as td
import compressible_solver.node as cn
import copy
import time

class EulerFlowSolver:

    def __init__(self, gamma, number_of_nodes, domain_size, max_time, cfl, max_runge_kutta_stages):

        # Material properties
        self.dimension = 1 if len(domain_size) == 1 else 2
        self.gamma = gamma

        if self.dimension == 1:
            self.node_table, self.face_table = mg.generete_1d_mesh(number_of_nodes, domain_size)
            self.delta_space = np.array([domain_size[0] / float(number_of_nodes[0] - 1.0)])
        else:
            self.node_table, self.face_table = mg.generate_2d_cartesian_mesh(number_of_nodes, domain_size)
            self.delta_space = np.array([domain_size[0] / float(number_of_nodes[0] - 1.0),
                                         domain_size[1] / float(number_of_nodes[1] - 1.0)])

        self.number_of_nodes_xy = number_of_nodes
        self.mnode = len(self.node_table)
        self.mface = len(self.face_table)

        # Temporal integration stuff
        self.cfl = cfl
        self.max_time = max_time
        self.time = 0.0
        self.delta_time = 0.0
        self.max_runge_kutta_stages = max_runge_kutta_stages

    def initialise_field(self, case_name):

        if case_name == '1D_shock_tube':
            for node in self.node_table:
                is_l = node.coordinate[0] > 0.7

                density = 1.0 if is_l else 0.125
                pressure = 1.0 if is_l else 0.1
                velocity = np.array([-0.75, 0.0]) if is_l else np.array([0.0, 0.0])
                internal_energy = pressure / ((self.gamma - 1.0) * density)
                node.variable[cn.n + cn.density] = density
                node.variable[cn.np1 + cn.density] = density
                node.variable[cn.n + cn.momentum_x] = density * velocity[0]
                node.variable[cn.n + cn.momentum_y] = density * velocity[0]
                node.variable[cn.np1 + cn.momentum_x] = density * velocity[0]
                node.variable[cn.np1 + cn.momentum_y] = density * velocity[1]
                node.variable[cn.n + cn.density_energy] = (internal_energy + 0.5 * np.dot(velocity, velocity)) * density
                node.variable[cn.np1 + cn.density_energy] = (internal_energy + 0.5 * np.dot(velocity, velocity)) * density
        elif case_name == '2D_blast':
            for node in self.node_table:
                radius = np.sqrt((node.coordinate[0] - 1.0)**2 + (node.coordinate[1] - 1.0)**2)
                is_in = (radius - 0.4) < 0.0

                density = 1.0 if is_in else 0.125
                pressure = 1.0 if is_in else 0.1
                velocity = np.array([0.0, 0.0])
                internal_energy = pressure / ((self.gamma - 1.0) * density)

                node.variable[cn.n + cn.density] = density
                node.variable[cn.np1 + cn.density] = density
                node.variable[cn.n + cn.momentum_x] = density*velocity[0]
                node.variable[cn.n + cn.momentum_y] = density*velocity[1]
                node.variable[cn.np1 + cn.momentum_x] = density*velocity[0]
                node.variable[cn.np1 + cn.momentum_y] = density*velocity[1]
                node.variable[cn.n + cn.density_energy] = (internal_energy + 0.5*np.dot(velocity, velocity))*density
                node.variable[cn.np1 + cn.density_energy] = (internal_energy + 0.5*np.dot(velocity, velocity))*density
        else:
            raise KeyError(case_name + ' is an unrecognised case name.')

        self.plot()

    def solve_problem(self):
        itime_step = 0
        while self.time < self.max_time:

            if itime_step % 15 == 0:
                self.plot()

            # copy values from np1 to n to start the time step.
            for node in self.node_table:
                node.variable[cn.n:cn.n + cn.max_variables] = copy.copy(node.variable[cn.np1:cn.np1 + cn.max_variables])

            itime_step += 1

            self.compute_timestep_size()
            self.time += self.delta_time

            for iteration in range(self.max_runge_kutta_stages):

                sys.stdout.write('\n' + str(itime_step) + " Time: " + str(self.time) +
                                 " Delta_time: " + str(self.delta_time) +
                                 " Iter: " + str(iteration + 1))
                sys.stdout.flush()

                self.compute_gradients()
                self.iterate(iteration)

            def main():
                """Print the latest tutorial from Real Python"""
                tic = time.perf_counter()
                toc = time.perf_counter()
                print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")

            # self.plot()

    def compute_timestep_size(self):
        self.delta_time = 100000.0
        for node in self.node_table:
            self.delta_time = min(self.delta_time, self.cfl * np.min(self.delta_space)*(0.5 if node.boundary else 1.0) /
                                  (np.linalg.norm(node.velocity(cn.n))
                                   + cn.acoustic_speed(node.variable[cn.n:cn.n + cn.max_variables], self.gamma)))
        if self.max_time < self.time + self.delta_time:
            self.delta_time = self.max_time - self.time
        return

    def compute_gradients(self):
        for face in self.face_table:
            for ivariable in range(cn.max_variables):
                for ifirst in face.node_first:
                    self.node_table[ifirst].gradient_variable[ivariable] = 0.0

                inode0 = face.connected_cell[0]
                inode1 = face.connected_cell[1]
                if inode0 != -1 and inode1 != -1:
                    face_contribution = fo.arithmetic_mean(self.node_table[inode0].variable[cn.np1 + ivariable],
                                                           self.node_table[inode1].variable[cn.np1 + ivariable]) \
                                        * face.coefficient
                    self.node_table[inode0].gradient_variable[ivariable] += face_contribution
                    self.node_table[inode1].gradient_variable[ivariable] -= face_contribution
                else:
                    node = inode0 if inode0 != -1 else inode1
                    self.node_table[node].gradient_variable[ivariable] += self.node_table[node].variable[cn.np1 + ivariable]\
                                                                          * face.coefficient

                for ilast in face.node_last:
                    self.node_table[ilast].gradient_variable[ivariable] /= self.node_table[ilast].volume
        return

    def iterate(self, iteration):
        for face in self.face_table:
            for ifirst in face.node_first:
                self.node_table[ifirst].variable[cn.residual:cn.residual + cn.max_variables] = 0.0

            if not face.boundary:
                flux_contribution = -self.compute_face_flux(face)*np.linalg.norm(face.coefficient)
                self.node_table[face.connected_cell[0]].variable[cn.residual:cn.residual + cn.max_variables] += flux_contribution
                self.node_table[face.connected_cell[1]].variable[cn.residual:cn.residual + cn.max_variables] -= flux_contribution
            else:
                inode = face.connected_cell[0] if face.connected_cell[0] != -1 else face.connected_cell[1]
                self.node_table[inode].variable[cn.residual:cn.residual + cn.max_variables] += \
                    -self.compute_boundary_flux(face)*np.linalg.norm(face.coefficient)

            for ilast in face.node_last:
                self.node_table[ilast].variable[cn.residual:cn.residual + cn.max_variables] /= self.node_table[ilast].volume
                self.node_table[ilast].variable[cn.np1:cn.np1 + cn.max_variables],\
                self.node_table[ilast].variable[cn.rhs:cn.rhs + cn.max_variables]\
                    = td.runge_kutta(self.node_table[ilast].variable[cn.n:cn.n + cn.max_variables],
                                     self.node_table[ilast].variable[cn.residual:cn.residual + cn.max_variables],
                                     self.node_table[ilast].variable[cn.rhs:cn.rhs + cn.max_variables],
                                     self.delta_time, iteration, self.max_runge_kutta_stages)
        return

    def compute_face_flux(self, face):
        inode0 = face.connected_cell[0]
        inode1 = face.connected_cell[1]
        state_variables_l = np.empty([cn.max_variables], dtype=float)
        state_variables_r = np.empty([cn.max_variables], dtype=float)
        for ivariable in range(cn.max_variables):
            state_variables_l[ivariable], state_variables_r[ivariable] = \
                fo.muscl(face.tangent, face.length,
                         self.node_table[inode0].variable[cn.np1 + ivariable],
                         self.node_table[inode1].variable[cn.np1 + ivariable],
                         self.node_table[inode0].gradient_variable[ivariable],
                         self.node_table[inode1].gradient_variable[ivariable])
        return fo.hllc(state_variables_l, state_variables_r, self.gamma, face.coefficient)

    def compute_boundary_flux(self, face):

        boundary_state = 0.0

        if self.dimension == 1:
            is_l = self.node_table[face.connected_cell[0]].coordinate[0] == 1.0
            density = 1.0 if is_l else 0.125
            pressure = 1.0 if is_l else 0.1
            velocity = np.array([-0.75, 0.0]) if is_l else np.array([0.0, 0.0])
            internal_energy = pressure / ((self.gamma - 1.0) * density)
            boundary_state = np.array([density, density * velocity[0], density * velocity[1],
                                       density*(internal_energy + 0.5 * np.dot(velocity, velocity))])

        else:
            density = 1.25
            pressure = 0.1
            velocity = np.array([0.0, 0.0])
            internal_energy = pressure / ((self.gamma - 1.0) * density)
            boundary_state = np.array([density, density * velocity[0], density * velocity[1],
                                       density*(internal_energy + 0.5 * np.dot(velocity, velocity))])

        inode = face.connected_cell[0] if face.connected_cell[0] != -1 else face.connected_cell[1]
        node_state = self.node_table[inode].variable[cn.np1:cn.np1 + cn.max_variables]
        return fo.hllc(node_state, boundary_state, self.gamma, face.coefficient)

    def plot(self):

        if self.dimension == 1:
            plt.close('all')
            self.fig, ax = plt.subplots(2, 2, sharex='all')
            x_coordinate = np.array([node.coordinate[0] for node in self.node_table])
            density = np.array([node.density(cn.np1) for node in self.node_table])
            velocity_x = np.array([node.velocity(cn.np1)[0] for node in self.node_table])
            pressure = np.array([node.pressure(cn.np1, self.gamma) for node in self.node_table])
            internal_energy = np.array([node.internal_energy(cn.np1) for node in self.node_table])

            linemarks = '-' if self.mnode > 25 else '.-'

            ax[0, 0].plot(x_coordinate, density, linemarks)
            ax[0, 0].set_title('density')
            ax[0, 1].plot(x_coordinate, velocity_x, linemarks)
            ax[0, 1].set_title('velocity_x')
            ax[1, 0].plot(x_coordinate, pressure, linemarks)
            ax[1, 0].set_title('pressure')
            ax[1, 1].plot(x_coordinate, internal_energy, linemarks)
            ax[1, 1].set_title('internal_energy')
            plt.show()

        elif self.dimension == 2:
            plt.close('all')
            mnode_x = self.number_of_nodes_xy[0]
            mnode_y = self.number_of_nodes_xy[1]
            self.fig, ax = plt.subplots(2, 2, sharex='all', sharey='all')
            x_coordinate = np.array([node.coordinate[0] for node in self.node_table]).reshape((mnode_x, mnode_y))
            y_coordinate = np.array([node.coordinate[1] for node in self.node_table]).reshape((mnode_x, mnode_y))

            density = np.array([node.density(cn.np1) for node in self.node_table]).reshape((mnode_x, mnode_y))
            velocity_norm = np.array([np.linalg.norm(node.velocity(cn.np1)) for node in self.node_table]).reshape((mnode_x, mnode_y))
            pressure = np.array([node.pressure(cn.np1, self.gamma) for node in self.node_table]).reshape((mnode_x, mnode_y))
            internal_energy = np.array([node.internal_energy(cn.np1) for node in self.node_table]).reshape((mnode_x, mnode_y))

            im = ax[0, 0].pcolormesh(x_coordinate, y_coordinate, density, cmap='RdBu_r', shading='gouraud')
            ax[0, 0].set_title('density')
            colorbar = self.fig.colorbar(im, ax=ax[0, 0])

            im = ax[0, 1].pcolormesh(x_coordinate, y_coordinate, velocity_norm, cmap='RdBu_r',shading='gouraud')
            ax[0, 1].set_title('velocity mangitude')
            self.fig.colorbar(im, ax=ax[0, 1])

            im = ax[1, 0].pcolormesh(x_coordinate, y_coordinate, pressure, cmap='RdBu_r', shading='gouraud')
            ax[1, 0].set_title('pressure')
            self.fig.colorbar(im, ax=ax[1, 0])

            im = ax[1, 1].pcolormesh(x_coordinate, y_coordinate, internal_energy, cmap='RdBu_r', shading='gouraud')
            ax[1, 1].set_title('internal energy')
            self.fig.colorbar(im, ax=ax[1, 1])
            plt.show()
        else:
            raise ValueError(str(self.dimension) + " not supported. ")
