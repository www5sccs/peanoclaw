'''
Created on Sep 17, 2013

@author: kristof
'''

class InternalSettings(object):
  
  def __init__(self, useHeapCompression=True):
    self.useHeapCompression = useHeapCompression
    
  def getFilenameSuffix(self):
    suffix = ''
    if not self.useHeapCompression:
      suffix += '_noHeapCompression'
    return suffix