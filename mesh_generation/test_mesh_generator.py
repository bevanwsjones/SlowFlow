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
# filename: test_mesh_generator.py
# description: Contains unit tests for the mesh generator.
# ----------------------------------------------------------------------------------------------------------------------

import mesh_generator as mg
import unittest as ut
import mesh.cell as cl


class Setup2dCartesianMeshTest(ut.TestCase):

    def test_equispaced_unit_mesh(self):
        [verticies, cells, cell_type] = mg.setup_2d_cartesian_mesh([2, 4], _ratio=[1.5, 1.0])

        print(verticies)
        for c_type in cell_type:
            self.assertEqual(c_type, cl.CellType.quadrilateral)
        print(cells)
