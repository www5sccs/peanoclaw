'''
Created on Jan 29, 2013

@author: kristof
'''
import time
from ctypes import c_bool
from ctypes import c_double
from ctypes import c_int
from ctypes import CFUNCTYPE
from ctypes import py_object
from ctypes import POINTER


class SolverCallback(object):
  '''
  This class encapsulates the callback for solving one timestep on
  a single subgrid.
  '''
  
  #Callback definition
  CALLBACK_SOLVER = CFUNCTYPE(c_double, 
                            POINTER(c_double), #Return array
                            py_object, #q
                            py_object, #qbc
                            py_object, #aux
                            c_int,     #subdivision factor X0
                            c_int,     #subdivision factor X1
                            c_int,     #subdivision factor X2
                            c_int,     #unknowns per cell
                            c_int,     #aux fields per cell
                            c_double, c_double, c_double, #size
                            c_double, c_double, c_double, #position
                            c_double, #current time
                            c_double, #maximum timestep size
                            c_double, #estimated next timestep size
                            c_bool)   #use dimensional splitting


  def __init__(self, solver, refinement_criterion, initial_minimal_mesh_width):
    '''
    Constructor
    '''
    self.solver = solver
    self.refinement_criterion = refinement_criterion
    self.initial_minimal_mesh_width = initial_minimal_mesh_width
    
    #Statistics
    self.number_of_non_disposed_cells = 0
    self.number_of_rollbacks = 0
    self.total_solver_time = 0.0

    self.callback = None
    
  def get_solver_callback(self):
    r"""
    Creates a closure for the solver callback method.
    """
    def callback_solver(return_dt_and_estimated_next_dt, q, qbc, aux, subdivision_factor_x0, subdivision_factor_x1, subdivision_factor_x2, unknowns_per_cell, aux_fields_per_cell, size_x, size_y, size_z, position_x, position_y, position_z, current_time, maximum_timestep_size, estimated_next_dt, use_dimensional_splitting):
        return self.call_subgridsolver(return_dt_and_estimated_next_dt, q, qbc, aux, subdivision_factor_x0, subdivision_factor_x1, subdivision_factor_x2, unknowns_per_cell, aux_fields_per_cell, size_x, size_y, size_z, position_x, position_y, position_z, current_time, maximum_timestep_size, estimated_next_dt, use_dimensional_splitting)
    
    if not self.callback:
        self.callback = self.CALLBACK_SOLVER(callback_solver)

    return self.callback
        
  def call_subgridsolver(self, return_dt_and_estimated_next_dt, q, qbc, aux, subdivision_factor_x0, subdivision_factor_x1, subdivision_factor_x2, unknowns_per_cell, aux_fields_per_cell, size_x, size_y, size_z, position_x, position_y, position_z, current_time, maximum_timestep_size, estimated_next_dt, use_dimensional_splitting):
    """
    Sets up a subgridsolver and calls it for doing the timestep. 
    Retrieves the new data and calls the refinement criterion.
    """
    starttime = time.time()
    # Fix aux array
    if(aux_fields_per_cell == 0):
      aux = None
      
    # Set up grid information for current patch
    import peanoclaw as peanoclaw
    subgridsolver = peanoclaw.SubgridSolver(self.solver.solver, self.solver.solution.state, q, qbc, aux, (position_x, position_y,position_z), (size_x, size_y,size_z), (subdivision_factor_x0, subdivision_factor_x1, subdivision_factor_x2), unknowns_per_cell, aux_fields_per_cell, current_time)
    
    new_q, number_of_rollbacks = subgridsolver.step(maximum_timestep_size, estimated_next_dt)
    
    # Copy back the array with new values    
    q[:]= new_q[:]
    self.solver.solution.t = subgridsolver.solution.t
    self.number_of_rollbacks += number_of_rollbacks
    
    return_dt_and_estimated_next_dt[0] = self.solver.solution.t - current_time
    cfl = self.solver.solver.cfl.get_cached_max()
    
    if(cfl == 0.0):
      return_dt_and_estimated_next_dt[1] = maximum_timestep_size - return_dt_and_estimated_next_dt[0]
    else:
      return_dt_and_estimated_next_dt[1] = return_dt_and_estimated_next_dt[0] * self.solver.solver.cfl_desired / cfl
      
    #Clean up
    if self.number_of_non_disposed_cells >= 1e6:
      import gc
      gc.collect()
      self.number_of_non_disposed_cells = 0
    else:
      self.number_of_non_disposed_cells += qbc.shape[1] * qbc.shape[2]
      
    #Steer refinement
    if self.refinement_criterion == None:
      return self.initial_minimal_mesh_width
    else:
      return self.refinement_criterion(subgridsolver.solution.state)      
    
    
