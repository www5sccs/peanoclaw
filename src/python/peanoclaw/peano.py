'''
Created on Jan 29, 2013

@author: kristof
'''

import logging
from ctypes import CDLL
from ctypes import c_bool
from ctypes import c_double
from ctypes import c_int
from ctypes import c_void_p
from ctypes import c_char_p
from ctypes import byref
from ctypes import RTLD_GLOBAL
import signal
from peanoclaw.converter import get_number_of_dimensions

class Peano(object):
  '''
  Encapsulation of the Peano library.
  '''


  def __init__(self, 
               solution,
               initial_minimal_mesh_width, 
               use_dimensional_splitting_optimization, 
               ghostlayer_width, 
               dt_initial,
               initialization_callback, 
               solver_callback,
               boundary_condition_callback,
               interpolation_callback,
               restriction_callback,
               flux_correction_callback,
               internal_settings):
    '''
    Constructor
    '''
    dim = len(solution.state.grid.dimensions)
    self.internal_settings = internal_settings
    
    logging.getLogger('peanoclaw').info("Loading Peano-library...")
    self.libpeano = CDLL(self.get_lib_path(dim),mode=RTLD_GLOBAL)
    logging.getLogger('peanoclaw').info("Peano loaded successfully.")
    self.libpeano.pyclaw_peano_new.restype = c_void_p
    self.libpeano.pyclaw_peano_destroy.argtypes = [c_void_p]
    self.libpeano.pyclaw_peano_evolveToTime.argtypes = [c_double, c_void_p]
    
    self.boundary_condition_callback = boundary_condition_callback
    self.solver_callback = solver_callback
    self.interpolation_callback = interpolation_callback
    self.restriction_callback = restriction_callback
    self.flux_correction_callback = flux_correction_callback
    
    # Get parameters for Peano
    dimensions = solution.state.grid.dimensions
    subdivision_factor_x0 = solution.state.grid.dimensions[0].num_cells
    subdivision_factor_x1 = solution.state.grid.dimensions[1].num_cells

    if dim is 3:
        subdivision_factor_x2 = solution.state.grid.dimensions[2].num_cells 
    else:
        subdivision_factor_x2 = 0
    number_of_unknowns = solution.state.num_eqn 
    number_of_auxiliar_fields = solution.state.num_aux
    import os, sys
    configuration_file = os.path.join(sys.path[0], 'peanoclaw-config.xml')
    
    if dim is 2:
      domain_position_x2 = 0
      domain_size_x2 = 0
    else:
      domain_position_x2 = dimensions[2].lower
      domain_size_x2 = dimensions[2].upper - dimensions[2].lower
      
    self.crank = c_int()
    self.libpeano.pyclaw_peano_new.argtypes = [ c_double, #Initial mesh width
                                                c_double, #Domain position X0
                                                c_double, #Domain position X1
                                                c_double, #Domain position X2
                                                c_double, #Domain size X0
                                                c_double, #Domain size X1
                                                c_double, #Domain size X2
                                                c_int,    #Subdivision factor X0
                                                c_int,    #Subdivision factor X1
                                                c_int,    #Subdivision factor X2
                                                c_int,    #Number of unknowns
                                                c_int,    #Number of auxiliar fields
                                                c_int,    #Ghostlayer width
                                                c_double, #Initial timestep size
                                                c_char_p, #Config file
                                                c_bool,   #Use dimensional splitting
                                                c_void_p, #q Initialization callback
                                                c_void_p, #Boundary condition callback
                                                c_void_p, #Solver callback
                                                c_void_p, #Solution callback
                                                c_void_p, #Interpolation callback
                                                c_void_p, #Restriction callback
                                                c_void_p, #Flux correction callback
                                                c_bool,   #Enable Peano logging
                                                c_void_p  #rank
                                                ] 
    self.peano = self.libpeano.pyclaw_peano_new(c_double(initial_minimal_mesh_width),
                                                c_double(dimensions[0].lower),
                                                c_double(dimensions[1].lower),
                                                c_double(domain_position_x2),
                                                c_double(dimensions[0].upper - dimensions[0].lower),
                                                c_double(dimensions[1].upper - dimensions[1].lower),
                                                c_double(domain_size_x2),
                                                subdivision_factor_x0,
                                                subdivision_factor_x1,
                                                subdivision_factor_x2,
                                                number_of_unknowns,
                                                number_of_auxiliar_fields,
                                                ghostlayer_width,
                                                dt_initial,
                                                c_char_p(configuration_file),
                                                use_dimensional_splitting_optimization,
                                                self.internal_settings.reduce_reductions,
                                                initialization_callback.get_initialization_callback(),
                                                boundary_condition_callback.get_boundary_condition_callback(),
                                                solver_callback.get_solver_callback(),
                                                solution.get_add_to_solution_callback(),
                                                interpolation_callback.get_interpolation_callback(),
                                                restriction_callback.get_restriction_callback(),
                                                flux_correction_callback.get_flux_correction_callback(),
                                                self.internal_settings.enable_peano_logging,
                                                self.internal_settings.fork_level_increment,
                                                byref(self.crank)
                                                )

    self.rank = self.crank.value
    
    print 'peano instance: got rank: ', self.rank

    # Set PeanoSolution
    import peanoclaw as peanoclaw
    if(isinstance(solution, peanoclaw.Solution)):
        solution.peano = self.peano
        solution.libpeano = self.libpeano
    else:
        logging.getLogger('peanoclaw').warning("Use peanoclaw.Solution instead of pyclaw.Solution together with peanoclaw.Solver to provide plotting functionality.")
    
    #Causes Ctrl+C to quit Peano
    signal.signal(signal.SIGINT, signal.SIG_DFL)

  def run_tests(self):
    self.libpeano.pyclaw_peano_runTests()
            
  def get_lib_path(self, dim):
    r"""
    Returns the path in which the shared library of Peano is located in.
    """
    import os
    import platform
    import peanoclaw as peanoclaw
    if platform.system() == 'Linux':
        shared_library_extension = 'so'
    elif platform.system() == 'Darwin':
        shared_library_extension = 'dylib'
    else:
        raise("Unsupported operating system. Only Linux and MacOS supported currently.")
      
    libraryFileName = os.path.join(os.path.dirname(peanoclaw.__file__), 'libpeano-claw-'+ str(dim)+ 'd' + self.internal_settings.getFilenameSuffix() + '.' + shared_library_extension)
    logging.getLogger('peanoclaw').info(libraryFileName)
    return os.path.join(libraryFileName)
        
  
  def evolve_to_time(self, tend):
    self.libpeano.pyclaw_peano_evolveToTime(
      tend, 
      self.peano
    )

  def teardown(self):
    self.libpeano.pyclaw_peano_destroy(self.peano)
    
  def getRank(self):
      return self.rank

  def runWorker(self):
    self.libpeano.pyclaw_peano_runWorker.argtypes = [c_void_p]
    self.libpeano.pyclaw_peano_runWorker(self.peano)
    #self.teardown()
