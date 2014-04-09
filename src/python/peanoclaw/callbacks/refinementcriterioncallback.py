'''
Created on Jan 29, 2013

@author: kristof
'''

from ctypes import c_double
from ctypes import c_int
from ctypes import CFUNCTYPE
from ctypes import py_object

from peanoclaw.converter import get_number_of_dimensions

class RefinementCriterionCallback(object):
  '''
  This class encapsulates the callback for retrieving the demanded mesh width for a single subgrid
  given by the refinement criterion.
  '''

  #Callback definition
  CALLBACK_REFINEMENT_CRITERION = CFUNCTYPE(c_double, 
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


  def __init__(self, refinement_criterion, initial_minimal_mesh_width):
    '''
    Constructor
    '''
    self.refinement_criterion = refinement_criterion
    self.initial_minimal_mesh_width = initial_minimal_mesh_width
    self.callback = None

  def get_refinement_criterion_callback(self):
    r"""
    Creates a closure for the callback
    """
    def callback_refinement_criterion(q, qbc, aux, subdivision_factor_x0, subdivision_factor_x1, subdivision_factor_x2, unknowns_per_subcell, aux_fields_per_subcell, size_x, size_y, size_z, position_x, position_y, position_z):
        #Steer refinement
        if self.refinement_criterion != None:
          return self.refinement_criterion(subgrid_state)
        else:
          return self.initial_minimal_mesh_width
   
    if not self.callback:
        self.callback = self.CALLBACK_REFINEMENT_CRITERION(callback_refinement_criterion)

    return self.callback
