#
# Tries to determine the mpi settings from environment variables.
#
import re
import os

def getMPIIncludes():
  return ['/bgsys/drivers/ppcfloor/comm/include']
      
def getMPILibrarypaths():
  return['/bgsys/drivers/ppcfloor/comm/lib']
  
def getMPILibraries():
  return ['mpich.cnk']
