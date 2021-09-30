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
# filename: test_error_analysis.py
# description: todo
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import unittest as ut
from ...post_processor import error_analysis as ea


# ----------------------------------------------------------------------------------------------------------------------
# Error Analysis
# ----------------------------------------------------------------------------------------------------------------------

class ErrorAnalysisTest(ut.TestCase):

    def test_l1_norm(self):

        error = np.array([1, 2, 3])
        volume = np.array([1, 2, 3])
        l2 = ea.l_norm_finite_volume(error, volume, _order=2)

        self.assertEqual(l2, 2)

    def test_l2_norm(self):

        error = np.array([1, 2, 3])
        volume = np.array([1, 2, 3])
        l2 = ea.l_norm_finite_volume(error, volume, _order=2)

        self.assertEqual(l2, 2)

    def test_linf_norm(self):

        error = np.array([1, 2, 3])
        volume = np.array([1, 2, 3])
        l2 = ea.l_norm_finite_volume(error, volume, _order=2)

        self.assertEqual(l2, 2)