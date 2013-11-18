from clawpack import pyclaw
from clawpack import riemann
import peanoclaw
  
def init(state):
  state.q[:] = 1.0
  return 1.0

def runTests(dim):
  solver = pyclaw.ClawSolver2D(riemann.shallow_roe_with_efix_2D)
  if dim==2:
    x = pyclaw.Dimension('x',0,1,10)
    y = pyclaw.Dimension('y',0,1,10)
    domain = pyclaw.Domain([x,y])
  elif dim==3:
    x = pyclaw.Dimension('x',0,1,10)
    y = pyclaw.Dimension('y',0,1,10)
    z = pyclaw.Dimension('z',0,1,10)
    domain = pyclaw.Domain([x,y,z])
  else:
    raise Exception("Dimension " + str(dim) + " not supported.")
  state = pyclaw.State(domain,solver.num_eqn)
  
  solution = peanoclaw.Solution(state, domain)
  
  init(state)
  
  solver = peanoclaw.Solver(solver, 1.0, init)
  solver.solution = solution
  
  solver.run_tests(solution)
  
  solver.teardown()
  
if __name__=="__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser()
  parser.add_argument("dim", nargs="?", type=int, default=2)
  arguments = parser.parse_args()
  
  runTests(arguments.dim)