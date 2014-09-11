#
# Tries to determine the mpi settings from environment variables.
#
import re
import os

def getMPIIncludes():
  includes = []
  includeString = os.getenv('MPI_INC')
  if includeString != None:
    for token in includeString.split():
      if token.startswith('-I'):
        includes.append(re.sub('^\-I', '', token))
    return includes
  else:
    raise Exception('No default MPI found!')
      
def getMPILibrarypaths():
  paths = []
  librariesString = os.getenv('MPI_LIB')
  if librariesString != None:
    for token in librariesString.split():
      if token.startswith('-L'):
        paths.append(re.sub('^\-L', '', token))
    return paths
  else:
    raise Exception('No default MPI found!')
  
def getMPILibraries():
  libraries = []
  librariesString = os.getenv('MPI_LIB')
  if librariesString != None:
    for token in librariesString.split():
      if token.startswith('-l'):
        libraries.append(re.sub('^\-l', '', token))
    return libraries
  else:
    raise Exception('No default MPI found!')
