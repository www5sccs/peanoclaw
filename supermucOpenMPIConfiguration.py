#
# Definitions for compiling with OpenMPI on SuperMUC
#


def getMPIIncludes():
  return ['/lrz/sys/parallel/openmpi/1.6.5/intel13_ib_sles11_ll/include']

def getMPILibrarypaths():
  return ['/lrz/sys/parallel/openmpi/1.6.5/intel13_ib_sles11_ll/lib']
  
def getMPILibraries():
  return ['mpi']

