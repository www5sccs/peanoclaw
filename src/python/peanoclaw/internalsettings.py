'''
Created on Sep 17, 2013

@author: kristof
'''

class InternalSettings(object):
  
  def __init__(self, useHeapCompression=True, enablePeanoLogging=False):
    self.useHeapCompression = useHeapCompression
    self.enablePeanoLogging = enablePeanoLogging
    
  def getFilenameSuffix(self):
    suffix = ''
    if not self.useHeapCompression:
      suffix += '_noHeapCompression'
    return suffix