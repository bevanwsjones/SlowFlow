#import numpy as np
#from compressible_solver import euler_flow as cs
#from compressible_solver import face as fc
#from compressible_solver import node as cn

#import dmsh
# cfl = 0.8
# max_time = 0.25
# gamma = 1.4
# max_runge_kutta_stages = 4
# max_node = [51, 51]
# domain_size = [2.0, 2.0]
# compressible_solver = cs.EulerFlowSolver(gamma, max_node, domain_size, max_time, cfl, max_runge_kutta_stages)
# compressible_solver.initialise_field('2D_blast')

# cfl = 0.8
# max_time = 0.2
# gamma = 1.4
# max_runge_kutta_stages = 1
# max_node = [51]
# domain_size = [1.0]
# compressible_solver = cs.EulerFlowSolver(gamma, max_node, domain_size, max_time, cfl, max_runge_kutta_stages)
# compressible_solver.initialise_field('1D_shock_tube')


# compressible_solver.solve_problem()
# compressible_solver.plot()

#from mesh_generation import mesh_generator as mg
from mesh_generation import mesh_generator as mg
import numpy as np
mg.create_cell_vertices(np.array(((0.0, 0.0), (1.0, 1.0))), [4, 4])


#mg.setup_connectivity()

