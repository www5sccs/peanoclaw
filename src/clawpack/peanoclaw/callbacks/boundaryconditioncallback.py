'''
Created on Jan 29, 2013

@author: kristof
'''
from ctypes import CFUNCTYPE
from ctypes import py_object
from ctypes import c_int

class BoundaryConditionCallback(object):
  '''
  Callback for setting the boundary conditions on a certain subgrid.
  '''

  #Callback definition
  CALLBACK_BOUNDARY_CONDITIONS = CFUNCTYPE(None, py_object, py_object, c_int, c_int)

  def __init__(self, solver):
    '''
    Constructor
    '''
    self.solver = solver
    self.callback = None

  def get_boundary_condition_callback(self):
    r"""
    Creates a closure for the boundary condition callback method.
    """
    def callback_boundary_conditions(q, qbc, dimension, setUpper):
      import numpy
      
      #TODO unterweg debug
      print("Filling boundary condition dimension=" + str(dimension) + " setUpper=" + str(setUpper))
      
      if(setUpper == 1):
        self.solver._qbc_upper(
              self.solver.solution.state, 
              self.solver.solution.state.grid.dimensions[dimension], 
              self.solver.solution.state.t, 
              numpy.rollaxis(qbc,dimension+1,1), 
              dimension
        )
      else:
        self.solver._qbc_lower(
              self.solver.solution.state, 
              self.solver.solution.state.grid.dimensions[dimension], 
              self.solver.solution.state.t, 
              numpy.rollaxis(qbc,dimension+1,1), 
              dimension
        )
 
    if not self.callback:
        self.callback = self.CALLBACK_BOUNDARY_CONDITIONS(callback_boundary_conditions)
    
    return self.callback
        
