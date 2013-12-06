'''
Created on Mar 18, 2012

@author: kristof
'''
import clawpack.pyclaw as pyclaw
from peanoclaw.converter import get_number_of_dimensions
from peanoclaw.converter import create_domain
from peanoclaw.converter import create_subgrid_state

class SubgridSolver(object):
    r"""
    The subgrid solver holds all information needed for the PyClaw/Clawpack solver
    to work on a single patch. It has to be thread safe to allow easy shared memory
    parallelization.
    
     
    """
    
    def __init__(self, solver, global_state, q, qbc, aux, position, size, subdivision_factor, unknowns_per_cell, aux_fields_per_cell, current_time):
        r"""
        Initializes this subgrid solver. It get's all information to prepare a domain and state for a
        single subgrid. 
        
        :Input:
         -  *solver* - (:class:`pyclaw.Solver`) The PyClaw-solver used for advancing this subgrid in time.
         -  *global_state* - (:class:`pyclaw.State`) The global state. This is not the state used for
                            for the actual solving of the timestep on the subgrid.
         -  *q* - The array storing the current solution.
         -  *qbc* - The array storing the solution of the last timestep including the ghostlayer.
         -  *position* - A d-dimensional tuple holding the position of the grid in the computational domain.
                         This measures the continuous real-world position in floats, not the integer position 
                         in terms of cells. 
         -  *size* - A d-dimensional tuple holding the size of the grid in the computational domain. This
                     measures the continuous real-world size in floats, not the size in terms of cells.
         -  *subdivision_factor* - The number of cells in one dimension of this subgrid. At the moment only
                                    square subgrids are allowed, so the total number of cells in the subgrid
                                    (excluding the ghostlayer) is subdivision_factor x subdivision_factor.
         -  *unknowns_per_cell* - The number of equations or unknowns that are stored per cell of the subgrid.
        
        """
        self.solver = solver
        number_of_dimensions = get_number_of_dimensions(q)

        self.domain = create_domain(number_of_dimensions, position, size, subdivision_factor)
        subgrid_state = create_subgrid_state(global_state, self.domain, q, qbc, aux, unknowns_per_cell, aux_fields_per_cell)
        subgrid_state.problem_data = global_state.problem_data
        self.solution = pyclaw.Solution(subgrid_state, self.domain)
        self.solution.t = current_time
        
        self.solver.bc_lower[:] = [pyclaw.BC.custom] * len(self.solver.bc_lower)
        self.solver.bc_upper[:] = [pyclaw.BC.custom] * len(self.solver.bc_upper)
        
        self.qbc = qbc
        self.solver.user_bc_lower = self.user_bc_lower
        self.solver.user_bc_upper = self.user_bc_upper
        
        self.recover_ghostlayers = False
        
    def step(self, maximum_timestep_size, estimated_next_dt, fixed_timestep_size):
        r"""
        Performs one timestep on the subgrid. This might result in several runs of the
        solver to find the maximum allowed timestep size in terms of stability.
        
        :Input:
         -  *maximum_timestep_size* - This is the maximum allowed timestep size in terms
                                      of the grid topology and the global timestep. I.e. 
                                      neighboring subgrids might forbid a timestep on this 
                                      subgrid. Also this subgrid is not allowed to advance 
                                      further than the the global timestep.
         -  *estimated_next_dt*- This is the estimation for the maximum allowed timestep size
                                 in terms of stability and results from the cfl number of the
                                 last timestep performed on this grid.
        """
        if fixed_timestep_size != None:
          self.solver.dt = fixed_timestep_size
        else:
          self.solver.dt = min(maximum_timestep_size, estimated_next_dt)
        self.number_of_rollbacks = 0
        # Set qbc and timestep for the current patch
        self.solver.qbc = self.qbc
        self.solver.dt_max = maximum_timestep_size
        
        success = False
        t_start = self.solution.t
        
        while not success:
          self.solver.evolve_to_time(self.solution)
          cfl = self.solver.cfl.get_cached_max()
          if abs(cfl) > 1e-12:
            self.solver.dt = min(self.solver.dt_max,self.solver.dt * self.solver.cfl_desired / self.solver.cfl.get_cached_max())
          else:
            self.solver.dt = self.solver.dt_max
          success = self.solution.t > t_start
        
        return self.solution.state.q, self.number_of_rollbacks
        
    def user_bc_lower(self, grid,dim,t,qbc,mbc):

        if len(self.domain.patch.dimensions) is 2:
            if dim == self.domain.patch.dimensions[0]:
                qbc[:,0:mbc,:] = self.qbc[:,0:mbc,:]
            else:
                qbc[:,:,0:mbc] = self.qbc[:,:,0:mbc]                

        elif len(self.domain.patch.dimensions) is 3:
            if dim == self.domain.patch.dimensions[0]:
                qbc[:,0:mbc,:,:] = self.qbc[:,0:mbc,:,:]
            elif dim == self.domain.patch.dimensions[1]:
                qbc[:,:,0:mbc,:] = self.qbc[:,:,0:mbc,:] 
            elif dim == self.domain.patch.dimensions[2]:
                qbc[:,:,:,0:mbc] = self.qbc[:,:,:,0:mbc]
        
    def user_bc_upper(self, grid,dim,t,qbc,mbc):

        if len(self.domain.patch.dimensions) is 2:
            shiftx = self.domain.patch.dimensions[0].num_cells+mbc
            shifty = self.domain.patch.dimensions[1].num_cells+mbc
            if dim == self.domain.patch.dimensions[0]:
                qbc[:,shiftx+0:shiftx+mbc,:] = self.qbc[:,shiftx+0:shiftx+mbc,:]
            else:
                qbc[:,:,shifty+0:shifty+mbc] = self.qbc[:,:,shifty+0:shifty+mbc]

        elif len(self.domain.patch.dimensions) is 3:

            shiftx = self.domain.patch.dimensions[0].num_cells+mbc
            shifty = self.domain.patch.dimensions[1].num_cells+mbc
            shiftz = self.domain.patch.dimensions[2].num_cells+mbc

            if dim == self.domain.patch.dimensions[0]:
                qbc[:,shiftx+0:shiftx+mbc,:,:] = self.qbc[:,shiftx+0:shiftx+mbc,:,:]
            elif dim == self.domain.patch.dimensions[1]:
                qbc[:,:,shifty+0:shifty+mbc,:] = self.qbc[:,:,shifty+0:shifty+mbc,:]
            elif dim == self.domain.patch.dimensions[2]:
                qbc[:,:,:,shiftz+0:shiftz+mbc] = self.qbc[:,:,:,shiftz+0:shiftz+mbc]
