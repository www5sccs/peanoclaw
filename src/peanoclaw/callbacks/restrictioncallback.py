'''
Created on Mar 6, 2013

@author: kristof
'''

from clawpack.peanoclaw.converter import get_number_of_dimensions
from clawpack.peanoclaw.converter import create_domain
from clawpack.peanoclaw.converter import create_subgrid_state

class RestrictionCallback(object):
    '''
    Encapsulating the functionality when restricting from a fine to a coarse subgrid.
    '''

    from ctypes import py_object
    from ctypes import c_int
    from ctypes import c_double
    from ctypes import c_bool
    from ctypes import POINTER
    from ctypes import CFUNCTYPE

    #Callback definition
    CALLBACK_RESTRICTION = CFUNCTYPE(None, 
                          py_object, #source q
                          py_object, #source qbc
                          py_object, #source aux
                          c_int, c_int, c_int, #source subdivision factor
                          c_double, c_double, c_double, #source size
                          c_double, c_double, c_double, #source position
                          c_double, #source current time
                          c_double, #source timestep size
                          py_object, #destination q
                          py_object, #destination qbc
                          py_object, #destination aux
                          c_int, c_int, c_int, #destination subdivision factor
                          c_double, c_double, c_double, #destination size
                          c_double, c_double, c_double, #destination position
                          c_double, #destination current time
                          c_double, #destination timestep size
                          c_int,     #unknowns per cell
                          c_int,     #aux fields per cell
                          )

    def __init__(self, restriction, solver):
      '''
      Constructor
      '''
      self.restriction = restriction
      self.solver = solver
      
    def get_restriction_callback(self):
      r"""
      Creates a closure for the solver callback method.
      """
      def callback_restriction(
                                 source_q, 
                                 source_qbc, 
                                 source_aux, 
                                 source_subdivision_factor_x0, 
                                 source_subdivision_factor_x1, 
                                 source_subdivision_factor_x2, 
                                 source_size_x0,
                                 source_size_x1,
                                 source_size_x2,
                                 source_position_x0,
                                 source_position_x1,
                                 source_position_x2,
                                 source_current_time, 
                                 source_timestep_size,
                                 destination_q, 
                                 destination_qbc, 
                                 destination_aux, 
                                 destination_subdivision_factor_x0, 
                                 destination_subdivision_factor_x1, 
                                 destination_subdivision_factor_x2, 
                                 destination_size_x0,
                                 destination_size_x1,
                                 destination_size_x2,
                                 destination_position_x0,
                                 destination_position_x1,
                                 destination_position_x2,
                                 destination_current_time, 
                                 destination_timestep_size,
                                 unknowns_per_cell,
                                 aux_fields_per_cell):
        # Fix aux array
        if(aux_fields_per_cell == 0):
          source_aux = None
          destination_aux = None
          
        number_of_dimensions = get_number_of_dimensions(source_q)
        source_domain = create_domain(number_of_dimensions, 
                               (source_position_x0, source_position_x1, source_position_x2), 
                               (source_size_x0, source_size_x1, source_size_x2),
                               (source_subdivision_factor_x0, source_subdivision_factor_x1, source_subdivision_factor_x2))
        source_subgrid_state = create_subgrid_state(
                               self.solver.solution.state, 
                               source_domain, 
                               source_q, 
                               source_qbc, 
                               source_aux, 
                               unknowns_per_cell, 
                               aux_fields_per_cell)
        
        destination_domain = create_domain(number_of_dimensions, 
                               (destination_position_x0, destination_position_x1, destination_position_x2), 
                               (destination_size_x0, destination_size_x1, destination_size_x2),
                               (destination_subdivision_factor_x0, destination_subdivision_factor_x1, destination_subdivision_factor_x2))
        destination_subgrid_state = create_subgrid_state(
                                    self.solver.solution.state, 
                                    destination_domain, 
                                    destination_q, 
                                    destination_qbc, 
                                    destination_aux, 
                                    unknowns_per_cell, 
                                    aux_fields_per_cell)
          
        self.restriction(source_subgrid_state, source_qbc, destination_subgrid_state, destination_qbc)
        
      if(self.restriction == None):
        return None
      else:
        return self.CALLBACK_RESTRICTION(callback_restriction)
        