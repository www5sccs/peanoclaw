'''
Created on Feb 7, 2012

@author: Kristof
'''
from clawpack.pyclaw.solver import Solver
from clawpack.peanoclaw.callbacks.initializationcallback import InitializationCallback 
from clawpack.peanoclaw.callbacks.solvercallback import SolverCallback
from clawpack.peanoclaw.callbacks.boundaryconditioncallback import BoundaryConditionCallback
from clawpack.peanoclaw.peano import Peano

from DimensionConverter import get_dimension

class Solver(Solver):
    r"""
        This solver class wraps the AMR functionality of Peano. It holds a normal PyClaw-solver
        to advance separate subgrids in time. Therefore it provides the callbacks callback_solver(...)
        and callback_boundary_conditions(...) as an interface for Peano to use the PyClaw-solver.
        
        A Solver is typically instantiated as follows::

        >>> import pyclaw
        >>> solver = pyclaw.ClawSolver2D()
        >>> import peanoclaw
        >>> peanoclaw_solver = peanoclaw.Solver(solver, 1.0/18.0)
    """
    
    def __init__(self, solver, initial_minimal_mesh_width, q_initialization, aux_initialization=None, refinement_criterion=None):
        r"""
        Initializes the Peano-solver. This keeps the Peano-spacetree internally and wraps the given PyClaw-solver.
        
        :Input:
         -  *solver* - (:class:`pyclaw.Solver`) The PyClaw-solver used internally.
         -  *initial_minimal_mesh_width* - The initial mesh width for the Peano mesh. I.e. Peano refines the mesh regularly
                                             until it is at least as fine as stated in this parameter.
        """
        #Initialize PeanoClaw solver
        self.solver = solver
        self.initial_minimal_mesh_width = initial_minimal_mesh_width
        self.q_initialization = q_initialization
        self.aux_initialization = aux_initialization
        self.refinement_criterion = refinement_criterion
        self.dt_initial = solver.dt_initial
        self.num_ghost = solver.num_ghost
        self.rp = solver.rp

        #Create callbacks
        self.initialization_callback = InitializationCallback(self, refinement_criterion, q_initialization, aux_initialization, initial_minimal_mesh_width)
        self.solver_callback = SolverCallback(self, refinement_criterion, initial_minimal_mesh_width)
        self.boundary_condition_callback = BoundaryConditionCallback(self)
        
    def setup(self, solution):
        r"""
        Initialize a Solver object. This method loads the library of Peano and prepares the initial mesh.
        
        See :class:`Solver` for full documentation
        """
        self.bc_lower = self.solver.bc_lower[:]
        self.bc_upper = self.solver.bc_upper[:]
        self.user_bc_lower = self.solver.user_bc_lower
        self.user_bc_upper = self.solver.user_bc_upper
        
        self.solution = solution
        
        self.peano = Peano(
                           solution,
                           self.initial_minimal_mesh_width,
                           self.solver.dimensional_split,
                           self.num_ghost,
                           self.solver.dt_initial,
                           self.initialization_callback,
                           self.solver_callback,
                           self.boundary_condition_callback,
                           None,
                           None)
                
    def teardown(self):
        r"""
        See :class:`Solver` for full documentation
        """ 
        self.peano.teardown()
        self.solver.teardown()
    
    def evolve_to_time(self, solution, tend=None):
        r"""
        Performs one global timestep until all patches in the mesh reach the given end time.
        
        See :class:`Solver` for full documentation
        """ 
        if(tend == None) :
            raise Exception("Not yet implemented.")
        
        self.solution = solution
        self.peano.evolve_to_time(tend)
                
    def solve_one_timestep(self, q, qbc):
        r"""
        """
        self.solver.step(self.solution)
        
