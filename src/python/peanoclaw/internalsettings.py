'''
Created on Sep 17, 2013

@author: kristof
'''

class InternalSettings(object):
  
  def __init__(self, use_heap_compression=True, enable_peano_logging=False, fixed_timestep_size=None):
    self.use_heap_compression = use_heap_compression
    self.enable_peano_logging = enable_peano_logging
    self.fixed_timestep_size = fixed_timestep_size
    
  def getFilenameSuffix(self):
    suffix = ''
    if not self.use_heap_compression:
      suffix += '_noHeapCompression'
    return suffix