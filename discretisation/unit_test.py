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
# filename: unit_test.py
# description: todo
# ----------------------------------------------------------------------------------------------------------------------

import unittest
import numpy as np
import face_operator as fcop
import limiter as lm
import temporal_discretisation as td
from mesh_generation import mesh_generator as mg

# ----------------------------------------------------------------------------------------------------------------------
# Face Operator Unit tests
# ----------------------------------------------------------------------------------------------------------------------


def f(x):
    return 2.0*x[0] #+ 3.0*x[1]


def dfdx(x):
    return 2.0


def dfdy(x):
    #return 3.0
    return 0.0


def L2Norm(cell_error, cell_volume):
    errorl2 = 0
    for ierror, error in enumerate(cell_error):
        errorl2 += cell_volume[ierror] * error ** 2
    return errorl2 ** 0.5


class TestFaceOperators(unittest.TestCase):

    def test_muscl(self):
        """
        The ultimate value of a muscl upwinded value is dependent on the flux limiter used. Van Albada is the 'go to'
        limiter and so is used here. One or two hand calculations are used to tests as well as ensuring for r <= 0 FOU is
        recovered, r = 1 2nd order CD is recovered.
        """




    def test_central_differnce(self):
        """
        Test central differencing operator to second order.
        """
        self.assertAlmostEqual(fcop.arithmetic_mean(10.0, 0.0), 5.0)
        self.assertAlmostEqual(fcop.arithmetic_mean(10.0, -10.0), 0.0)
        self.assertAlmostEqual(fcop.arithmetic_mean(-10.0, 2.0), -4.0)

        np.testing.assert_array_almost_equal(fcop.arithmetic_mean(np.array([2.0, 2.0]), np.array([-10.0, -2.0])),
                                             np.array([-4.0, 0.0]))
        np.testing.assert_array_almost_equal(fcop.arithmetic_mean(np.array([6.0, -3.0]), np.array([16.0, 7.0])),
                                             np.array([11.0, 2.0]))

    def test_central_difference_recovery_construct_gauss_green_coefficient_matrix(self):
        """
        For 1D linear equi-spaced mesh tests the recovery of a 2D order accuracy for the gauss green node based gradient
        calculation.
        """

        # Cell spacing to tests
        max_cell_list = [10, 20, 40]
        errorL2 = []
        for max_cell in max_cell_list:
            cell_table, face_table, vertex_table = mg.setup_1d_unit_mesh(max_cell)
            gradient_coef = fcop.construct_gauss_green_coefficient_matrix(cell_table, face_table, vertex_table)
            fx = np.array([f(x) for x in cell_table.coordinate[:]])
            error = np.abs(gradient_coef.dot(fx).reshape((max_cell, 2))[:, 0]
                           - np.array([dfdx(x) for x in cell_table.coordinate[0:cell_table.max_cell, ]]))
            errorL2.append(L2Norm(error, cell_table.volume[0:cell_table.max_cell]))

        # note log(dx_0/dx_1) = log((1.0/mc_0)/(1.0/mc_1)) = log(mc_1/mc_0)
        self.assertGreater(np.log(errorL2[0]/errorL2[1])/np.log(max_cell_list[1]/max_cell_list[0]), 2.0)
        self.assertGreater(np.log(errorL2[1]/errorL2[2])/np.log(max_cell_list[2]/max_cell_list[1]), 2.0)

    def test_2nd_order_recovery_construct_gauss_green_coefficient_matrix(self):
        """

        :return:
        """

        cell_table, face_table, vertex_table = mg.setup_connectivity()
        mg.setup_finite_volume_geometry(cell_table, face_table, vertex_table)

        # Cell spacing to tests
        errorL2 = []
        gradient_coef = fcop.construct_gauss_green_coefficient_matrix(cell_table, face_table, vertex_table)
        # print(gradient_coef)

        fx = np.array([f(x) for x in cell_table.coordinate[:]])
        dfdx_x = [np.array([dfdx(x), dfdy(x)]) for x in cell_table.coordinate[0:cell_table.max_cell]]
        gradient = gradient_coef.dot(fx).reshape((cell_table.max_cell, 2))[:]

        print(cell_table.volume)
        print(cell_table.coordinate)
        print(face_table.coefficient)
        print(fx)
        #print(np.shape(gradient_coef))
        #print(np.shape(fx))

        #print(dfdx_x)
        print(gradient)

# ----------------------------------------------------------------------------------------------------------------------
# Slope Limiter Unit tests
# ----------------------------------------------------------------------------------------------------------------------


class TestSlopeLimiter(unittest.TestCase):

    def test_limiter(self):
        """
        Tests that the correct limiters is returned for each enumerator
        """

        ratio_all = np.linspace(start=0.0, stop=4.0, num=40, dtype=float)

        for ratio in ratio_all:
            self.assertEqual(lm.limiter(ratio, lm.Limiter.superbee), lm.superbee(ratio))
            self.assertEqual(lm.limiter(ratio, lm.Limiter.minmod), lm.minmod(ratio))
            self.assertEqual(lm.limiter(ratio, lm.Limiter.van_albada), lm.van_albada(ratio))


    def test_superbee(self):
        """
        Test the superbee limiter obtains the correct values of its 4 2nd order TVD regions.
        """
        ratio_all = np.linspace(start=0.0, stop=4.0, num=40, dtype=float)

        for ratio in ratio_all:
            if ratio <= 0.5:
                self.assertEqual(lm.superbee(ratio), 2.0*ratio)
            elif 0.5 < ratio <= 1.0:
                self.assertEqual(lm.superbee(ratio), 1.0)
            elif 1.0 < ratio <= 2:
                self.assertEqual(lm.superbee(ratio), ratio)
            else:
                self.assertLessEqual(lm.superbee(ratio), 2.0)


    def test_minmod(self):
        """
        Test the minmod limiter obtains the correct values of its 2 2nd order TVD regions.
        """

        ratio_all = np.linspace(start=0.0, stop=4.0, num=40, dtype=float)

        for ratio in ratio_all:
            if ratio <= 1:
                self.assertEqual(lm.minmod(ratio), ratio)
            else:
                self.assertEqual(lm.minmod(ratio), 1.0)

    def test_van_albad(self):
        """
        Test the van Albada limiter, ensures it is both symmetric requirements as well as 2nd order TVD properties.
        """

        ratio_all = np.linspace(start=0.0, stop=4.0, num=40, dtype=float)

        for ratio in ratio_all:
            if ratio < 1:
                self.assertGreaterEqual(lm.van_albada(ratio), ratio)
                self.assertLessEqual(lm.van_albada(ratio), 2.0 * ratio)
            elif 1.0 < ratio <= 2:
                self.assertGreaterEqual(lm.van_albada(ratio), 1.0)
                self.assertLessEqual(lm.van_albada(ratio), ratio)
            else:
                self.assertGreaterEqual(lm.van_albada(ratio), 1.0)
                self.assertLessEqual(lm.van_albada(ratio), 2.0)
        self.assertEqual(lm.van_albada(1.0), 1.0)  # Specifically check 1 = 1

# ----------------------------------------------------------------------------------------------------------------------
# Temporal Discretisation Unit Tests
# ----------------------------------------------------------------------------------------------------------------------


def analytical_solution(t):
    """
    Analytical solution to for temporal accuracy

    y(t) = 1/5*t*e^(3t) - 1/25*t*e^(3t) + 1/25*t*e^(-2t)

    :param t: time
    :return: y(t), the analytical solution at time t.
    """
    return t*np.exp(3.0*t)/5.0 - np.exp(3.0*t)/25.0 + np.exp(-2.0*t)/25.0


def dydt(t, y):
    """
    Numerical gradient (rhs) for the ODE.

    dy/dt = t*e^(3t) - 2y

    :param t: time
    :param y: solution variable
    :return: dy/dt the change of the solution with respect to time at t and y.
    """
    return t*np.exp(3.0*t) - 2.0*y


class TestRungeKutta(unittest.TestCase):

    def test_1st_order(self):
        """
        Ensure the 1st order Runge-Kutta approach is achieving a 1st order approximation by reducing the time step size
        and checking convergence of the the numerical solution using th analytical solution.

        dy/dt = f(y,t), y|_0 = 0, 0 <= t <= 1
        """
        mtime = 1.0
        mtime_step = np.array([21, 51, 101])
        error = np.empty(shape=[3], dtype=float)
        test_delta_time = np.empty(shape=[3], dtype=float)
        for itest, itest_mtime_step in enumerate(mtime_step):
            x_rhs = 0.0
            x_n = analytical_solution(0)
            x_np1 = analytical_solution(0)
            time_hist = np.linspace(0.0, mtime, itest_mtime_step)
            stage_nodes = td.stage_nodes[td.method_to_index(1)]

            for itime in range(itest_mtime_step - 1):
                delta_time = time_hist[itime + 1] - time_hist[itime]
                x_np1, x_rhs = td.runge_kutta(x_n, dydt(time_hist[itime] + stage_nodes[0] * delta_time, x_np1), x_rhs, delta_time, 0, 1)
                x_n = x_np1

                if itime == 0:
                    test_delta_time[itest] = delta_time
            error[itest] = np.abs(x_np1 - analytical_solution(mtime))
        self.assertGreater(np.log(error[0]/error[1])/np.log(test_delta_time[0]/test_delta_time[1]), 1.0)
        self.assertGreater(np.log(error[1]/error[2])/np.log(test_delta_time[1]/test_delta_time[2]), 1.0)

    def test_4th_order(self):
        """
        Ensure the 4th order Runge-Kutta approach is achieving a 4th order approximation by reducing the time step size
        and checking convergence of the the numerical solution using th analytical solution.

        dy/dt = f(y,t), y|_0 = 0, 0 <= t <= 1
        """
        mtime = 1.0
        mtime_step = np.array([21, 51, 101])
        error = np.empty(shape=[3], dtype=float)
        test_delta_time = np.empty(shape=[3], dtype=float)
        stage_nodes = td.stage_nodes[td.method_to_index(4)]
        for itest, itest_mtime_step in enumerate(mtime_step):
            x_rhs = 0.0
            x_n = analytical_solution(0)
            x_np1 = analytical_solution(0)
            time_hist = np.linspace(0.0, mtime, itest_mtime_step)

            for itime in range(itest_mtime_step - 1):
                delta_time = time_hist[itime + 1] - time_hist[itime]
                for iter in range(4):
                    x_np1, x_rhs = \
                        td.runge_kutta(x_n, dydt(time_hist[itime] + stage_nodes[iter] * delta_time, x_np1), x_rhs,
                                       delta_time, iter, 4)
                x_n = x_np1

                if itime == 0:
                    test_delta_time[itest] = delta_time
            error[itest] = np.abs(x_np1 - analytical_solution(mtime))
        self.assertGreater(np.log(error[0]/error[1])/np.log(test_delta_time[0]/test_delta_time[1]), 4.0)
        self.assertGreater(np.log(error[1]/error[2])/np.log(test_delta_time[1]/test_delta_time[2]), 4.0)

