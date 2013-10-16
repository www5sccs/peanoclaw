#
# Definitions for compiling with OpenMPI on Linux
#


def getMPIIncludes():
  return ['/usr/lib/openmpi/include']

def getMPILibrarypaths():
  return ['/usr/lib/openmpi/lib']
  
def getMPILibraries():
  return ['mpi', 'pthread']
