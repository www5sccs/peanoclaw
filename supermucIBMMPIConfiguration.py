#
# Definitions for compiling with IBM-MPI on SuperMUC
#


def getMPIIncludes():
  return ['/opt/ibmhpc/pecurrent/mpich2/intel/include64']

def getMPILibrarypaths():
  return ['/opt/ibmhpc/pecurrent/mpich2/intel/lib64', '/opt/ibmhpc/pecurrent/mpich2/../pempi/intel/lib64', '/opt/ibmhpc/pecurrent/ppe.pami/intel/lib64/pami64']
  
def getMPILibraries():
  return ['cxxmpich', 'pthread', 'mpich', 'opa', 'mpl', 'dl', 'poe', 'pami']


