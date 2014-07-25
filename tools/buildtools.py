from os.path import join
from os.path import dirname
import sys
import os
    
def addPython(cppdefines, cpppath, libpath, libs):
   if sys.version_info[0] == 2 and sys.version_info[1] < 7:
       pythonVersion = str(sys.version_info[0]) + '.' + str(sys.version_info[1]) #For Python 2.6
   else: 
       pythonVersion = str(sys.version_info.major) + '.' + str(sys.version_info.minor) #For Python 2.7
   # Determine python version from environment variable:
   peanoClawPythonVersion = os.getenv ('PEANOCLAW_PYTHONVERSION')
   if (peanoClawPythonVersion != None):
      pythonVersion = peanoClawPythonVersion

   # Determine python root path from environment variable:
   pythonHome = os.getenv ('PYTHONHOME')
   peanoClawPythonHome = os.getenv ('PEANOCLAW_PYTHONHOME')
   if (peanoClawPythonHome != None):
      print 'Using environment variable PEANOCLAW_PYTHONHOME =', peanoClawPythonHome
      pythonHome = peanoClawPythonHome
   elif (pythonHome != None):
      print 'Using environment variable PYTHONHOME =', pythonHome
   else:
      print('Environment variables PYTHONHOME and PEANOCLAW_PYTHONHOME not defined. Using path depending on the interpreter\'s path ' + sys.executable)
      pythonHome = join(dirname(sys.executable), '..')
      print pythonHome
   cppdefines.append('NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION')
      
   # Add paths and lib
   libpath.append(pythonHome + '/lib')
   libpath.append(pythonHome + '/lib/python' + pythonVersion)
   cpppath.append(pythonHome + '/include/python' + pythonVersion)
   cpppath.append(pythonHome + '/lib/python' + pythonVersion + '/site-packages/numpy/core/include')
   cpppath.append(os.getenv("HOME") + '/.local/lib/python' + pythonVersion + '/site-packages/numpy/core/include')
   libs.append('python' + pythonVersion)

    
def getPeanoSources(Glob, buildpath, multicore):
  sourcesTLa = [
     Glob(join(buildpath, 'kernel/tarch/la/*.cpp')),
     Glob(join(buildpath, 'kernel/tarch/la/tests/*.cpp'))
     ]
  
  sourcesTCompiler = [
     Glob(join(buildpath, 'kernel/tarch/compiler/*.cpp'))
     ]
  
  sourcesTConfiguration = [
     Glob(join(buildpath, 'kernel/tarch/configuration/*.cpp'))
     ]
  
  sourcesTIrr = [
     Glob(join(buildpath, 'kernel/tarch/irr/*.cpp'))
   ]
  
  sourcesTLogging = [
    Glob(join(buildpath, 'kernel/tarch/logging/*.cpp')),
    Glob(join(buildpath, 'kernel/tarch/logging/configurations/*.cpp'))
  ]
  
  sourcesTServices = [
    Glob(join(buildpath, 'kernel/tarch/services/*.cpp'))
  ]
  
  sourcesTTests = [
    Glob(join(buildpath, 'kernel/tarch/tests/*.cpp')),
    Glob(join(buildpath, 'kernel/tarch/tests/configurations/*.cpp'))
    ]
  
  sourcesTUtils = [
    Glob(join(buildpath, 'kernel/tarch/utils/*.cpp'))
  ]
  
  sourcesTTiming = [
    Glob(join(buildpath, 'kernel/tarch/timing/*.cpp'))
  ]
  
  sourcesTPlotter = [ 
     Glob(join(buildpath, 'kernel/tarch/plotter/*.cpp')),
     Glob(join(buildpath, 'kernel/tarch/plotter/globaldata/*.cpp')),
     Glob(join(buildpath, 'kernel/tarch/plotter/globaldata/tests/*.cpp')),
     Glob(join(buildpath, 'kernel/tarch/plotter/griddata/*.cpp')),
     Glob(join(buildpath, 'kernel/tarch/plotter/griddata/multiscale/*.cpp')),
     Glob(join(buildpath, 'kernel/tarch/plotter/griddata/unstructured/*.cpp')),
     Glob(join(buildpath, 'kernel/tarch/plotter/griddata/unstructured/configurations/*.cpp')),
     Glob(join(buildpath, 'kernel/tarch/plotter/griddata/unstructured/vtk/*.cpp')),
     Glob(join(buildpath, 'kernel/tarch/plotter/griddata/unstructured/vtk/tests/*.cpp'))
     ]   
     
  ##### Define sources T-components
  #            
  sourcesTComponents = [
     sourcesTCompiler,
     sourcesTConfiguration,
     sourcesTIrr,
     sourcesTLa,
     sourcesTLogging,
     sourcesTPlotter,
     sourcesTServices,
     sourcesTTests,
     sourcesTTiming,
     sourcesTUtils
     ]
  
  
  ##### Define sources for multicore support
  #    
  sourcesDatatraversal = [
      Glob(join(buildpath, 'kernel/peano/datatraversal/*.cpp')),
      Glob(join(buildpath, 'kernel/peano/datatraversal/configurations/*.cpp')),
      Glob(join(buildpath, 'kernel/peano/datatraversal/tests/*.cpp')),
      Glob(join(buildpath, 'kernel/peano/datatraversal/autotuning/*.cpp')),
      Glob(join(buildpath, 'kernel/tarch/multicore/configurations/*.cpp')),
      Glob(join(buildpath, 'kernel/tarch/multicore/*.cpp'))
    ]       
        
  if multicore == 'no' or multicore == 'multicore_no':
     pass
  elif multicore == 'openmp':
     sourcesDatatraversal = sourcesDatatraversal + [
       Glob(join(buildpath, 'kernel/tarch/multicore/openMP/*.cpp'))
     ]
  elif multicore == 'tbb':
     sourcesDatatraversal = sourcesDatatraversal + [
       Glob(join(buildpath, 'kernel/tarch/multicore/tbb/*.cpp'))
     ]
  
  sourcesParallel = [
   Glob(join(buildpath, 'kernel/tarch/parallel/configuration/*.cpp')),
   Glob(join(buildpath, 'kernel/tarch/parallel/*.cpp')),
   Glob(join(buildpath, 'kernel/tarch/parallel/strategy/*.cpp')),
   Glob(join(buildpath, 'kernel/tarch/parallel/messages/*.cpp')),
   Glob(join(buildpath, 'kernel/tarch/parallel/dastgen/*.cpp')),
   Glob(join(buildpath, 'kernel/tarch/parallel/configurations/*.cpp')),
   Glob(join(buildpath, 'kernel/peano/parallel/*.cpp')),
   Glob(join(buildpath, 'kernel/peano/parallel/*.cc')),
   Glob(join(buildpath, 'kernel/peano/parallel/configurations/*.cpp')),
   Glob(join(buildpath, 'kernel/peano/parallel/loadbalancing/*.cpp')),
   Glob(join(buildpath, 'kernel/peano/parallel/messages/*.cpp')),
   Glob(join(buildpath, 'kernel/peano/parallel/tests/*.cpp')),
   Glob(join(buildpath, 'kernel/tarch/mpianalysis/*.cpp')),
   Glob(join(buildpath, 'kernel/tarch/analysis/*.cpp'))
  ]
  
  
  #### Peano Utils
  sourcesPeanoUtils = [
    Glob(join(buildpath, 'kernel/peano/utils/*.cpp'))
  ]
  
  
  # ## Peano partition coupling
  sourcesPartitionCoupling = [
    Glob(join(buildpath, 'kernel/peano/integration/partitioncoupling/*.cpp')),
    Glob(join(buildpath, 'kernel/peano/integration/partitioncoupling/builtin/*.cpp')),
    Glob(join(buildpath, 'kernel/peano/integration/partitioncoupling/builtin/configurations/*.cpp')),
    Glob(join(buildpath, 'kernel/peano/integration/partitioncoupling/builtin/tests/*.cpp')),
    Glob(join(buildpath, 'kernel/peano/integration/partitioncoupling/builtin/records/*.cpp')),
    Glob(join(buildpath, 'kernel/peano/integration/partitioncoupling/services/*.cpp'))
  ]
  
  # ## Kernel
  sourcesKernelConfiguration = [
     Glob(join(buildpath, 'kernel/peano/configurations/*.cpp'))
     ]
  
  sourcesGridInterface = [
     Glob(join(buildpath, 'kernel/peano/gridinterface/*.cpp'))
     ]
     
  sourcesGrid = [
     Glob(join(buildpath, 'kernel/peano/grid/*.cpp')),
     Glob(join(buildpath, 'kernel/peano/grid/aspects/*.cpp')),
     Glob(join(buildpath, 'kernel/peano/grid/nodes/*.cpp')),
     Glob(join(buildpath, 'kernel/peano/grid/nodes/loops/*.cpp')),
     Glob(join(buildpath, 'kernel/peano/grid/nodes/tasks/*.cpp')),
     Glob(join(buildpath, 'kernel/peano/grid/tests/*.cpp')),
     Glob(join(buildpath, 'kernel/peano/grid/tests/records/*.cpp'))
     ]
  sourcesStacks = [
      Glob(join(buildpath, 'kernel/peano/stacks/*.cpp')),
      Glob(join(buildpath, 'kernel/peano/stacks/implementation/*.cpp'))
      ]
  sourcesHeap = [
      Glob(join(buildpath, 'kernel/peano/heap/records/*.cpp'))
      ]
  sourcesPeanoKernel = [
     sourcesKernelConfiguration,
     sourcesGridInterface,
     sourcesGrid,
     sourcesStacks,
     sourcesHeap
     ]
  
  #### Geometry
  ##### Builtin Geometry
  sourcesBuiltinGeometry = [
     Glob(join(buildpath, 'kernel/peano/geometry/builtin/*.cpp')),
     Glob(join(buildpath, 'kernel/peano/geometry/builtin/services/*.cpp')),
     Glob(join(buildpath, 'kernel/peano/geometry/extensions/*.cpp')),
     Glob(join(buildpath, 'kernel/peano/geometry/builtin/configurations/*.cpp')),
     Glob(join(buildpath, 'kernel/peano/geometry/builtin/tests/*.cpp'))
     ]
  
  
  sourcesPeanoGeometry = [
     Glob(join(buildpath, 'kernel/peano/geometry/*.cpp')),
     Glob(join(buildpath, 'kernel/peano/geometry/tests/*.cpp')),
     Glob(join(buildpath, 'kernel/peano/geometry/configurations/*.cpp')),
     Glob(join(buildpath, 'kernel/peano/geometry/services/*.cpp')),
     sourcesBuiltinGeometry
     ]
  
  sourcesPeanoBase = [
    sourcesPeanoKernel,
    sourcesPeanoGeometry,
    sourcesPeanoUtils,
    sourcesDatatraversal,
    # sourcesQueries,
    Glob(join(buildpath, 'kernel/peano/*.cpp')),
    Glob(join(buildpath, 'kernel/*.cpp'))
  ]
  
  sourcesToolBox = [
    Glob(join(buildpath, 'toolboxes/ControlLoopLoadBalancer/*.cpp')),
    Glob(join(buildpath, 'toolboxes/ControlLoopLoadBalancer/strategies/*.cpp'))
  ]
  
  sourcesToolBoxVHH = [
    Glob(join(buildpath, 'kernel/peano/toolbox/solver/vhh/*.cpp')),
    Glob(join(buildpath, 'kernel/peano/toolbox/solver/vhh/tests/*.cpp')),
    Glob(join(buildpath, 'kernel/peano/toolbox/solver/vhh/opencl/*.cpp')),
    Glob(join(buildpath, 'kernel/peano/toolbox/solver/vhh/opencl/tests/*.cpp'))
  ]
  
  peanoSources = []
  peanoSources.extend(sourcesTComponents)
  peanoSources.extend(sourcesPeanoBase)
  peanoSources.extend(sourcesToolBox)
  peanoSources.extend(sourcesToolBoxVHH)
  peanoSources.extend(sourcesParallel)
  
  return  peanoSources

def getPeanoClawSources(Glob, buildpath):
    # ## Applications
  sourcesPeanoClaw = [
    Glob(join(buildpath, 'peanoclaw/*.cpp')),
    Glob(join(buildpath, 'peanoclaw/adapters/*.cpp')),
    Glob(join(buildpath, 'peanoclaw/configurations/*.cpp')),
    Glob(join(buildpath, 'peanoclaw/grid/*.cpp')),
    Glob(join(buildpath, 'peanoclaw/grid/plotter/*.cpp')),
    Glob(join(buildpath, 'peanoclaw/interSubgridCommunication/*.cpp')),
    Glob(join(buildpath, 'peanoclaw/interSubgridCommunication/aspects/*.cpp')),
    Glob(join(buildpath, 'peanoclaw/mappings/*.cpp')),
    Glob(join(buildpath, 'peanoclaw/parallel/*.cpp')),
    Glob(join(buildpath, 'peanoclaw/records/*.cpp')),
    Glob(join(buildpath, 'peanoclaw/repositories/*.cpp')),
    Glob(join(buildpath, 'peanoclaw/runners/*.cpp')),
    Glob(join(buildpath, 'peanoclaw/statistics/*.cpp')),
    Glob(join(buildpath, 'peanoclaw/tests/*.cpp')),
    ]
    
  ##### PeanoClaw-specific tarch
  sourcesPeanoClawTarch = [
      Glob(join(buildpath, 'tarch/plotter/griddata/unstructured/binaryvtu/*.cpp'))
    ]
  
  sourcesPeanoClaw.extend(sourcesPeanoClawTarch)
  return sourcesPeanoClaw

