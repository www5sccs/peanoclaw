# Peano/SConstruct

import os;
import sys;

#########################################################################
##### FUNCTION DEFINITIONS
#########################################################################

def addPeanoClawFlags(libpath,libs,cpppath,cppdefines):
   ccflags.append('-g3')
   ccflags.append('-g')
   ccflags.append('-march=native')
   
   pythonVersion = '2.7'
   #Determine python version from environment variable:
   peanoClawPythonVersion = os.getenv ( 'PEANOCLAW_PYTHONVERSION' )
   if ( peanoClawPythonVersion != None ):
      pythonVersion = peanoClawPythonVersion

   # Determine python root path from environment variable:
   pythonHome = os.getenv ( 'PYTHONHOME' )
   peanoClawPythonHome = os.getenv ( 'PEANOCLAW_PYTHONHOME' )
   if ( peanoClawPythonHome != None ):
      print 'Using environment variable PEANOCLAW_PYTHONHOME =', peanoClawPythonHome
      pythonHome = peanoClawPythonHome
   elif ( pythonHome != None ):
      print 'Using environment variable PYTHONHOME =', pythonHome
   else:
      print('Environment variables PYTHONHOME and PEANOCLAW_PYTHONHOME not defined. Using default \'/usr\'')
      pythonHome = '/usr'
      
   # Add paths and lib
   libpath.append(pythonHome + '/lib')
   libpath.append(pythonHome + '/lib/python' + pythonVersion)
   cpppath.append(pythonHome + '/include/python' + pythonVersion)
   cpppath.append(pythonHome + '/lib/python' + pythonVersion + '/site-packages/numpy/core/include')
   libs.append('python' + pythonVersion)
   if(environment['PLATFORM'] == 'darwin'):
     ccflags.append('-flat_namespace')
     linkerflags.append('-flat_namespace')
   if build == 'release' and (environment['PLATFORM'] != 'darwin'):
     cppdefines.append('_GLIBCXX_DEBUG')
   if '-Werror' in ccflags:
     ccflags.remove('-Werror')
     
#########################################################################
##### MAIN CODE
#########################################################################

##### Initialize build variables
#
cxx = ''
cppdefines = []
cpppath = ['./src']
ccflags = []
linkerflags = []
libpath = []
libs = []

p3Path='../p3/src'
cpppath.append(p3Path)
clawpackPath='../src/clawpack'

#Platform specific settings
environment = Environment()
#Only include library rt if not compiling on Mac OS.
if(environment['PLATFORM'] != 'darwin'):
    libs.append('rt')

##### Determine dimension for which to build
#
dim = ARGUMENTS.get('dim', 2)   # Read command line parameter
if int(dim) == 2:
   cppdefines.append('Dim2')
elif int(dim) == 3:
   cppdefines.append('Dim3')
elif int(dim) == 4:
   cppdefines.append('Dim4')
elif int(dim) == 5:
   cppdefines.append('Dim5')
else:
   print "ERROR: dim must be either '2', '3', '4', or '5'!"
   sys.exit(1)

##### Add build parameter specific build variable settings:
# This section only defines Peano-specific flags. It does not
# set compiler specific stuff.
#
build = ARGUMENTS.get('build', 'debug')   # Read command line parameter
if build == 'debug':
   cppdefines.append('Debug')
   cppdefines.append('Asserts')
   cppdefines.append('LogTrace')
   cppdefines.append('LogSeparator')
elif build == 'release':
   pass
elif build == 'asserts':
   cppdefines.append('Asserts')
   pass
else:
   print "ERROR: build must be 'debug', 'asserts', or 'release'!"
   sys.exit(1)
   
##### Determine MPI-Parallelization
#
parallel = ARGUMENTS.get('parallel', 'parallel_no') # Read command line parameter
if parallel == 'yes' or parallel == 'parallel_yes':
   cppdefines.append('Parallel')
   cpppath.append(mpiIncludePath)
   libpath.append(mpiLibraryPath)
   libs.append (mpiLibrary)
   libs.append ('pthread')
   cxx = 'mpicxx'
elif parallel == 'no' or parallel == 'parallel_no':
   pass
else:
   print "ERROR: parallel must be = 'yes', 'parallel_yes', 'no' or 'parallel_no'!"
   sys.exit(1)

##### Determine Multicore usage
#   
multicore = ARGUMENTS.get('multicore', 'multicore_no')   # Read command line parameter

if multicore == 'no' or multicore == 'multicore_no':
   pass
elif multicore == 'openmp':
   ompDir = os.getenv ( 'OMP_DIR', '' )
   cppdefines.append('SharedOMP')
   cpppath.append(ompDir + '/include')   
   #libpath.append(ompDir + '/build/' + tbbArch)
   pass
elif multicore == 'tbb':
   libs.append('pthread')
   libs.append('dl')
   # Determine tbb directory and architecture from environment variables:
   tbbDir = os.getenv ( 'TBB_DIR' )
   
   if ( tbbDir == None ):
      print 'ERROR: Environment variable TBB_DIR not defined!'
      sys.exit(1)
   else:
      print 'Using environment variable TBB_DIR =', tbbDir
      
   tbbArch = os.getenv ( 'TBB_ARCH' );
   if( tbbArch == None ):
      print 'ERROR: Environment variable TBB_ARCH not defined!'
      sys.exit(1)
   else:
      print 'Using environment variable TBB_ARCH =', tbbArch
          
   if ( build == 'debug' ):
      libs.append ('tbb_debug')
      tbbArch = tbbArch + '_debug'     
   else:
      libs.append ('tbb')
      tbbArch = tbbArch + '_release'      

   cppdefines.append('SharedTBB')
   cpppath.append(tbbDir + '/include')   
   libpath.append(tbbDir+'/build/'+tbbArch)
elif multicore == 'opencl':
   libs.append('OpenCL')
   libs.append ('pthread')
   cppdefines.append('SIMD_OpenCL')
else:
   print "ERROR: multicore must be = 'tbb',  'openmp', 'no' or 'multicore_no'!"
   sys.exit(1)

##### Determine Valgrind usage
# 
valgrind = ARGUMENTS.get('valgrind', 'no')
if valgrind == 'no':
   pass
elif valgrind=='yes':
   ccflags.append('-g')
   cppdefines.append('USE_VALGRIND')
   cpppath.append(os.getenv ( 'VALGRIND_ROOT' ) + "/include")
   cpppath.append(os.getenv ( 'VALGRIND_ROOT' ) + "/callgrind")
else:
   print "ERROR: valgrind must be = 'yes' or 'no'!"
   sys.exit(1)
   
##### Switch Compiler
#
compiler = ARGUMENTS.get('compiler', 'gcc') # Read command line parameter
if compiler == 'gcc':
   if(parallel=='parallel_no' or parallel=='no'):
     cxx = 'g++'
   else:
     cxx = 'mpicxx'
     cppdefines.append('MPICH_SKIP_MPICXX')
   ccflags.append('-Wall')
   #if(cca=='cca_no' or cca=='no'):
   #ccflags.append('-Werror')
   #	ccflags.append('-pedantic')
   ccflags.append('-pedantic-errors')
   ccflags.append('-Wstrict-aliasing')
   ccflags.append('-fstrict-aliasing')
   #ccflags.append('-fno-exceptions')
   #ccflags.append('-fno-rtti')
   ccflags.append('-Wno-long-long')
   ccflags.append('-Wno-unknown-pragmas')
   #if multicore == 'no' or multicore == 'multicore_no':
      #ccflags.append('-Wconversion')
   ccflags.append('-Wno-non-virtual-dtor')
   if build == 'debug':
      ccflags.append('-g3')
      ccflags.append('-O0')
   elif build == 'asserts"':
      ccflags.append('-O3') 
   elif build == 'release':
      ccflags.append('-O3') 
   if multicore == 'openmp':
      ccflags.append('-fopenmp')
      linkerflags.append('-fopenmp')
elif compiler == 'xlc':
   if(parallel=='parallel_no' or parallel=='no'):
     cxx = 'xlc++'
   else:
     cxx = 'mpixlcxx'
   if build == 'debug':
      ccflags.append('-g3')
      ccflags.append('-O0')
   elif build == 'asserts':
      ccflags.append('-qstrict')
      ccflags.append('-O3')
   elif build == 'release':
      ccflags.append('-qstrict')
      ccflags.append('-O3')
   if multicore == 'openmp':
      ccflags.append('-qsmp=omp')
      linkerflags.append('-qsmp=omp')
      cxx = cxx + '_r'
      

elif compiler == 'icc':
   cxx = 'icpc'
   #ccflags.append('-fast')
   ccflags.append('-fstrict-aliasing')
   ccflags.append('-qpack_semantic=gnu')
   if build == 'debug':
      ccflags.append('-O0')
   elif build == 'asserts':
#      ccflags.append('-fast')
      #ccflags.append('-vec-report5') # is supressed by -ipo (included in -fast)
      #ccflags.append('-xHost')       # done by -fast
      #ccflags.append('-O3')          # done by -fast
      #ccflags.append('-no-prec-div') # done by -fast
      #ccflags.append('-static')      # done by -fast
      ccflags.append('-w')
#     ccflags.append('-Werror-all')
      ccflags.append('-align')
      ccflags.append('-ansi-alias')
   elif build == 'release':
#      ccflags.append('-fast')
      #ccflags.append('-vec-report5') # is supressed by -ipo (included in -fast)
      #ccflags.append('-xHost')       # done by -fast
      #ccflags.append('-O3')          # done by -fast
      #ccflags.append('-no-prec-div') # done by -fast
      #ccflags.append('-static')      # done by -fast
      ccflags.append('-w')
#     ccflags.append('-Werror-all')
      ccflags.append('-align')
      ccflags.append('-ansi-alias')
   #PN: If -fast is used for linking, the tbb-lib cannot be found :-(   
   #linkerflags.append('-fast')
   if multicore == 'openmp':
      ccflags.append('-openmp')
      linkerflags.append('-openmp')
else:
   print "ERROR: compiler must be = 'gcc', 'xlc' or 'icc'!"
   sys.exit(1)
   
##### Determine Scalasca Usage
#
scalasca = ARGUMENTS.get('scalasca', 'scalasca_no') # Read command line parameter
if scalasca == 'yes' or scalasca == 'scalasca_yes':
   cxx = 'scalasca -instrument ' + cxx
elif scalasca == 'no' or scalasca == 'scalasca_no':
   pass
else:
   print "ERROR: scalasca must be = 'scalasca_yes', 'yes', 'scalasca_no' or 'no'!"
   sys.exit(1)
   
##### Determine build path
#
build_offset = ARGUMENTS.get('buildoffset', 'build')
buildpath = build_offset  + '/' + str(build) + '/dim' + str(dim) + '/' 
if multicore == 'tbb':
   buildpath = buildpath + 'tbb/'
elif multicore == 'openmp':
   buildpath = buildpath + 'openmp/'
elif multicore == 'opencl':
   buildpath = buildpath + 'openCL/'
else:
   buildpath = buildpath + 'multicore_no/'
if parallel == 'yes' or parallel == 'parallel_yes':
   buildpath = buildpath + 'parallel_yes/'
else:
   buildpath = buildpath + 'parallel_no/'
if compiler == 'icc':
   buildpath = buildpath + 'icc/'
if compiler == 'gcc':
   buildpath = buildpath + 'gcc/'
if compiler == 'xlc':
   buildpath = buildpath + 'xlc/'
if scalasca == 'yes' or scalasca == 'scalasca_yes':
   buildpath = buildpath + 'scalasca/'

   
##### Specify build settings
#
addPeanoClawFlags(libpath,libs,cpppath,cppdefines)

##### Print options used to build
#
print
print "Building PeanoClaw"
print "Options: build = " + str(build) + ", dim = " + str(dim) + ", build-offset = " + str(build_offset) + ", parallel = " + str(parallel) + ", multicore = " + str(multicore) + ", compiler = " + str(compiler)
print "Buildpath: " + buildpath
print

VariantDir (buildpath, './', duplicate=0)  # Change build directory


##### Setup construction environment:
#
env = Environment ( 
   CPPDEFINES = cppdefines,
   LIBPATH    = libpath,
   LIBS       = libs, 
   CPPPATH    = cpppath,
   CCFLAGS    = ccflags,
   LINKFLAGS  = linkerflags,
   CXX        = cxx,
   ENV        = os.environ # Makes environment variables visible to scons
   #tools      = compiler_tools
   )

################################################################################
#
# Define sources
#

##### Sub T-components

sourcesTLa = [
   Glob(buildpath + p3Path + '/tarch/la/*.cpp'),
   Glob(buildpath + p3Path + '/tarch/la/tests/*.cpp')
   ]

sourcesTCompiler = [
   Glob(buildpath + p3Path + '/tarch/compiler/*.cpp')
   ]

sourcesTConfiguration = [
   Glob(buildpath + p3Path + '/tarch/configuration/*.cpp')
   ]

sourcesTIrr = [
   Glob(buildpath + p3Path + '/tarch/irr/*.cpp')
 ]

sourcesTLogging = [
  Glob(buildpath + p3Path + '/tarch/logging/*.cpp'),
  Glob(buildpath + p3Path + '/tarch/logging/configurations/*.cpp')
]

sourcesTServices = [
  Glob(buildpath + p3Path + '/tarch/services/*.cpp')
]

sourcesTTests = [
  Glob(buildpath + p3Path + '/tarch/tests/*.cpp'),
  Glob(buildpath + p3Path + '/tarch/tests/configurations/*.cpp')
  ]

sourcesTUtils = [
  Glob(buildpath + p3Path + '/tarch/utils/*.cpp')
]

sourcesTTiming = [
  Glob(buildpath + p3Path + '/tarch/timing/*.cpp')
]

sourcesTPlotter = [ 
   Glob(buildpath + p3Path + '/tarch/plotter/*.cpp'),
   Glob(buildpath + p3Path + '/tarch/plotter/globaldata/*.cpp'),
   Glob(buildpath + p3Path + '/tarch/plotter/globaldata/tests/*.cpp'),
   Glob(buildpath + p3Path + '/tarch/plotter/griddata/*.cpp'),
   Glob(buildpath + p3Path + '/tarch/plotter/griddata/multiscale/*.cpp'),
   Glob(buildpath + p3Path + '/tarch/plotter/griddata/unstructured/*.cpp'),
   Glob(buildpath + p3Path + '/tarch/plotter/griddata/unstructured/configurations/*.cpp'),
   Glob(buildpath + p3Path + '/tarch/plotter/griddata/unstructured/vtk/*.cpp'),
   Glob(buildpath + p3Path + '/tarch/plotter/griddata/unstructured/vtk/tests/*.cpp')
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
    Glob(buildpath + p3Path + '/peano/kernel/datatraversal/*.cpp'),
    Glob(buildpath + p3Path + '/peano/kernel/datatraversal/configurations/*.cpp'),
    Glob(buildpath + p3Path + '/peano/kernel/datatraversal/tests/*.cpp'),
    Glob(buildpath + p3Path + '/peano/kernel/datatraversal/autotuning/*.cpp'),
    Glob(buildpath + p3Path + '/tarch/multicore/configurations/*.cpp'),
    Glob(buildpath + p3Path + '/tarch/multicore/*.cpp')
  ]       
      
if multicore == 'no' or multicore == 'multicore_no':
   pass
elif multicore == 'openmp':
   sourcesDatatraversal = sourcesDatatraversal + [
     Glob(buildpath + p3Path + '/tarch/multicore/openMP/*.cpp')
   ]
elif multicore == 'tbb':
   sourcesDatatraversal = sourcesDatatraversal + [
     Glob(buildpath + p3Path + '/tarch/multicore/tbb/*.cpp')
   ]


if parallel == 'yes' or parallel == 'parallel_yes':
   sourcesParallel = [
     Glob(buildpath + p3Path + '/tarch/parallel/configuration/*.cpp'),
     Glob(buildpath + p3Path + '/tarch/parallel/*.cpp'),
     Glob(buildpath + p3Path + '/tarch/parallel/strategy/*.cpp'),
     Glob(buildpath + p3Path + '/tarch/parallel/messages/*.cpp'),
     Glob(buildpath + p3Path + '/tarch/parallel/dastgen/*.cpp'),
     Glob(buildpath + p3Path + '/tarch/parallel/configurations/*.cpp'),
     Glob(buildpath + p3Path + '/peano/kernel/parallel/*.cpp'),
     Glob(buildpath + p3Path + '/peano/kernel/parallel/configurations/*.cpp'),
     Glob(buildpath + p3Path + '/peano/kernel/parallel/loadbalancing/*.cpp'),
     Glob(buildpath + p3Path + '/peano/kernel/parallel/messages/*.cpp'),
     Glob(buildpath + p3Path + '/peano/kernel/parallel/tests/*.cpp')
   ]
   pass
else:
  sourcesParallel = []


#### Peano Utils
sourcesPeanoUtils = [
  Glob(buildpath + p3Path + '/peano/utils/*.cpp')
]


### Peano partition coupling
sourcesPartitionCoupling = [
  Glob(buildpath + p3Path + '/peano/integration/partitioncoupling/*.cpp'),
  Glob(buildpath + p3Path + '/peano/integration/partitioncoupling/builtin/*.cpp'),
  Glob(buildpath + p3Path + '/peano/integration/partitioncoupling/builtin/configurations/*.cpp'),
  Glob(buildpath + p3Path + '/peano/integration/partitioncoupling/builtin/tests/*.cpp'),
  Glob(buildpath + p3Path + '/peano/integration/partitioncoupling/builtin/records/*.cpp'),
  Glob(buildpath + p3Path + '/peano/integration/partitioncoupling/services/*.cpp')
]

### Kernel
sourcesKernelConfiguration = [
   Glob(buildpath + p3Path + '/peano/kernel/configurations/*.cpp')
   ]

sourcesGridInterface = [
   Glob(buildpath + p3Path + '/peano/kernel/gridinterface/*.cpp')
   ]
   
sourcesSpacetreeGrid = [
   Glob(buildpath + p3Path + '/peano/kernel/spacetreegrid/*.cpp'),
   Glob(buildpath + p3Path + '/peano/kernel/spacetreegrid/aspects/*.cpp'),
   Glob(buildpath + p3Path + '/peano/kernel/spacetreegrid/nodes/*.cpp'),
   Glob(buildpath + p3Path + '/peano/kernel/spacetreegrid/tests/*.cpp'),
   Glob(buildpath + p3Path + '/peano/kernel/spacetreegrid/tests/records/*.cpp')
   ]
sourcesStacks = [
    Glob(buildpath + p3Path + '/peano/kernel/stacks/*.cpp'),
    Glob(buildpath + p3Path + '/peano/kernel/stacks/implementation/*.cpp')
    ]
sourcesHeap = [
    Glob(buildpath + p3Path + '/peano/kernel/heap/records/*.cpp')
    ]
sourcesPeanoKernel = [
   Glob(buildpath + p3Path + '/peano/kernel/*.cpp'),
   sourcesKernelConfiguration,
   sourcesGridInterface,
   #sourcesRegularGrid,
   sourcesSpacetreeGrid,
   sourcesStacks,
   sourcesHeap
   ]

#### Geometry
##### Builtin Geometry
sourcesBuiltinGeometry = [
   Glob(buildpath + p3Path + '/peano/geometry/builtin/*.cpp'),
   Glob(buildpath + p3Path + '/peano/geometry/builtin/services/*.cpp'),
   Glob(buildpath + p3Path + '/peano/geometry/extensions/*.cpp'),
   Glob(buildpath + p3Path + '/peano/geometry/builtin/configurations/*.cpp'),
   Glob(buildpath + p3Path + '/peano/geometry/builtin/tests/*.cpp')
   ]


sourcesPeanoGeometry = [
   Glob(buildpath + p3Path + '/peano/geometry/*.cpp'),
   Glob(buildpath + p3Path + '/peano/geometry/tests/*.cpp'),
   Glob(buildpath + p3Path + '/peano/geometry/configurations/*.cpp'),
   Glob(buildpath + p3Path + '/peano/geometry/services/*.cpp'),
   sourcesBuiltinGeometry
   ]

sourcesPeanoBase = [
  sourcesPeanoKernel,
  sourcesPeanoGeometry,
  sourcesPeanoUtils,
  sourcesDatatraversal,
  #sourcesQueries,
  Glob(buildpath + p3Path + '/peano/*.cpp'),
  Glob(buildpath + p3Path + '/*.cpp')
]

sourcesToolBox = [
  Glob(buildpath + p3Path + '/peano/toolbox/solver/*.cpp'),
  Glob(buildpath + p3Path + '/peano/toolbox/solver/configurations/*.cpp'),
  Glob(buildpath + p3Path + '/peano/toolbox/solver/tests/*.cpp'),
  Glob(buildpath + p3Path + '/peano/toolbox/stencil/*.cpp'),
  Glob(buildpath + p3Path + '/peano/toolbox/refinement/*.cpp')
]

sourcesToolBoxVHH = [
  Glob(buildpath + p3Path + '/peano/toolbox/solver/vhh/*.cpp'),
  Glob(buildpath + p3Path + '/peano/toolbox/solver/vhh/tests/*.cpp'),
  Glob(buildpath + p3Path + '/peano/toolbox/solver/vhh/opencl/*.cpp'),
  Glob(buildpath + p3Path + '/peano/toolbox/solver/vhh/opencl/tests/*.cpp')
]

### Applications
##### Define sources of application peanoclaw
sourcesPeanoClaw = [
  Glob(buildpath + 'src/*.cpp'),
	Glob(buildpath + 'src/adapters/*.cpp'),
	Glob(buildpath + 'src/configurations/*.cpp'),
	Glob(buildpath + 'src/mappings/*.cpp'),
	Glob(buildpath + 'src/records/*.cpp'),
	Glob(buildpath + 'src/repositories/*.cpp'),
	Glob(buildpath + 'src/runners/*.cpp'),
  Glob(buildpath + 'src/statistics/*.cpp'),
  Glob(buildpath + 'src/tests/*.cpp')
	]
################################################################################

##### Build selected target
#
targetfilename = 'libpeano-claw-' + str(dim) + 'd'
target = buildpath + targetfilename
env.SharedLibrary (
#env.Program (
  target = target,
  source = [
     sourcesTComponents,
     sourcesPeanoBase,
     sourcesPeanoClaw,
     sourcesParallel
     ]
  )
  
##### Copy library to Clawpack
#
import shutil
if(environment['PLATFORM'] == 'darwin'):
  libExtension = '.dylib'
elif(environment['PLATFORM'] == 'posix'):
  libExtension = '.so'
else:
  raise "Only Linux and MacOS supported, yet!"

shutil.copyfile(target + libExtension, clawpackPath + '/pyclaw/src/peanoclaw/' + targetfilename)

