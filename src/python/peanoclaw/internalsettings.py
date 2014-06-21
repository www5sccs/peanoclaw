'''
Created on Sep 17, 2013

@author: kristof
'''

class InternalSettings(object):
  
  def __init__(self, 
               use_heap_compression=True, 
               enable_peano_logging=False, 
               fixed_timestep_size=None, 
               plot_unknowns_filter=None, 
               fork_level_increment=1,
               reduce_reductions=True,
               use_dimensional_splitting_optimization=False,
               plot_name='adaptive'):
    self.use_heap_compression = use_heap_compression
    self.enable_peano_logging = enable_peano_logging
    self.fixed_timestep_size = fixed_timestep_size
    self.plot_unknowns_filter = plot_unknowns_filter
    self.fork_level_increment = fork_level_increment
    self.reduce_reductions = reduce_reductions
    self.use_dimensional_splitting_optimization = use_dimensional_splitting_optimization
    self.plot_name = plot_name
    
  def getFilenameSuffix(self):
    suffix = ''
    if not self.use_heap_compression:
      suffix += '_noHeapCompression'
    return suffix
  
  