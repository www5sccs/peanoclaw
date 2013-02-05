'''
Created on Jan 29, 2013

@author: kristof
'''

from ctypes import c_double
from ctypes import c_int
from ctypes import CFUNCTYPE
from ctypes import py_object

class InitializationCallback(object):
  '''
  This class encapsulates the callback for initializing a single subgrid.
  '''

  #Callback definition
  CALLBACK_INITIALIZATION = CFUNCTYPE(c_double, 
                                      py_object, #q
                                      py_object, #qbc
                                      py_object, #aux
                                      c_int,     #subdivision factor X0
                                      c_int,     #subdivision factor X1
                                      c_int,     #subdivision factor X2
                                      c_int,     #unknowns per cell
                                      c_int,     #aux fields per cell
                                      c_double, c_double, c_double, #size
                                      c_double, c_double, c_double) #position


  def __init__(self, solver, refinement_criterion, q_initialization, aux_initialization, initial_minimal_mesh_width):
    '''
    Constructor
    '''
    self.solver = solver
    self.refinement_criterion = refinement_criterion
    self.q_initialization = q_initialization
    self.aux_initialization = aux_initialization
    self.initial_minimal_mesh_width = initial_minimal_mesh_width
    
  def get_initialization_callback(self):
    r"""
    Creates a closure for initializing the grid
    """
    def callback_initialization(q, qbc, aux, subdivision_factor_x0, subdivision_factor_x1, subdivision_factor_x2, unknowns_per_subcell, aux_fields_per_subcell, size_x, size_y, size_z, position_x, position_y, position_z):
        import clawpack.pyclaw as pyclaw
        self.dim_x = pyclaw.Dimension('x',position_x,position_x + size_x,subdivision_factor_x0)
        self.dim_y = pyclaw.Dimension('y',position_y,position_y + size_y,subdivision_factor_x1)
        #TODO 3D: use size_z and position_z
        domain = pyclaw.Domain([self.dim_x,self.dim_y])
        subgrid_state = pyclaw.State(domain, unknowns_per_subcell, aux_fields_per_subcell)
        subgrid_state.q = q
        if(aux_fields_per_subcell > 0):
          subgrid_state.aux = aux
        subgrid_state.problem_data = self.solver.solution.state.problem_data
        
        self.q_initialization(subgrid_state)

        if(self.aux_initialization != None and aux_fields_per_subcell > 0):
          self.aux_initialization(subgrid_state)
          
        #Steer refinement
        if self.refinement_criterion != None:
          return self.refinement_criterion(subgrid_state)
        else:
          return self.initial_minimal_mesh_width
        
    return self.CALLBACK_INITIALIZATION(callback_initialization)
