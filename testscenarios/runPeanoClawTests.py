from clawpack import pyclaw
from clawpack import riemann
import peanoclaw

def init(state):
  state.q[:] = 1.0
  return 1.0

solver = pyclaw.ClawSolver2D(riemann.shallow_roe_with_efix_2D)
x = pyclaw.Dimension('x',0,1,10)
y = pyclaw.Dimension('y',0,1,10)
domain = pyclaw.Domain([x,y])
state = pyclaw.State(domain,solver.num_eqn)

solution = peanoclaw.Solution(state, domain)

init(state)

solver = peanoclaw.Solver(solver, 1.0, init)
solver.solution = solution

solver.run_tests(solution)

solver.teardown()