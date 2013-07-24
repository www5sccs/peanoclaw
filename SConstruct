# Peano/SConstruct

import os;
import sys;

#########################################################################
##### FUNCTION DEFINITIONS
#########################################################################

def addPeanoClawFlags(libpath, libs, cpppath, cppdefines):
   ccflags.append('-g3')
   ccflags.append('-g')
   ccflags.append('-march=native')
   
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
      print('Environment variables PYTHONHOME and PEANOCLAW_PYTHONHOME not defined. Using default \'/usr\'')
      pythonHome = '/usr'
   cppdefines.append('NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION')
      
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

p3Path = '../p3'
cpppath.append(p3Path)
clawpackPath = '../src/clawpack'

# Platform specific settings
environment = Environment()
# Only include library rt if not compiling on Mac OS.
if(environment['PLATFORM'] != 'darwin'):
    libs.append('rt')

##### Determine dimension for which to build
#
dim = ARGUMENTS.get('dim', 2)  # Read command line parameter
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
build = ARGUMENTS.get('build', 'debug')  # Read command line parameter
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
parallel = ARGUMENTS.get('parallel', 'parallel_no')  # Read command line parameter
if parallel == 'yes' or parallel == 'parallel_yes':
   cppdefines.append('Parallel')
   cpppath.append('/usr/lib/openmpi/include')
   libpath.append('/usr/lib/openmpi/lib')
   libs.append ('mpi')
   libs.append ('pthread')
   cxx = 'mpicxx'
elif parallel == 'no' or parallel == 'parallel_no':
   pass
else:
   print "ERROR: parallel must be = 'yes', 'parallel_yes', 'no' or 'parallel_no'!"
   sys.exit(1)

##### Determine Multicore usage
#   
multicore = ARGUMENTS.get('multicore', 'multicore_no')  # Read command line parameter

if multicore == 'no' or multicore == 'multicore_no':
   pass
elif multicore == 'openmp':
   ompDir = os.getenv ('OMP_DIR', '')
   cppdefines.append('SharedOMP')
   cpppath.append(ompDir + '/include')   
   pass
elif multicore == 'tbb':
   libs.append('pthread')
   libs.append('dl')
   # Determine tbb directory and architecture from environment variables:
   tbbDir = os.getenv ('TBB_DIR')
   
   # if ( tbbDir == None ):
   #   print 'ERROR: Environment variable TBB_DIR not defined!'
   #   sys.exit(1)
   # else:
   #   print 'Using environment variable TBB_DIR =', tbbDir
      
   # tbbArch = os.getenv ( 'TBB_ARCH' );
   # if( tbbArch == None ):
   #   print 'ERROR: Environment variable TBB_ARCH not defined!'
   #   sys.exit(1)
   # else:
   #   print 'Using environment variable TBB_ARCH =', tbbArch
          
   if (build == 'debug'):
      libs.append ('tbb_debug')
      # tbbArch = tbbArch     
   else:
      libs.append ('tbb')
      # tbbArch = tbbArch

   cppdefines.append('SharedTBB')
   # cpppath.append(tbbDir + '/include')   
   # libpath.append(tbbDir + '/lib/'+tbbArch)
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
elif valgrind == 'yes':
   ccflags.append('-g')
   cppdefines.append('USE_VALGRIND')
   cpppath.append(os.getenv ('VALGRIND_ROOT') + "/include")
   cpppath.append(os.getenv ('VALGRIND_ROOT') + "/callgrind")
else:
   print "ERROR: valgrind must be = 'yes' or 'no'!"
   sys.exit(1)
   
##### Switch Compiler
#
compiler = ARGUMENTS.get('compiler', 'gcc')  # Read command line parameter
if compiler == 'gcc':
   if(parallel == 'parallel_no' or parallel == 'no'):
     cxx = 'g++'
   else:
     cxx = 'mpicxx'
     cppdefines.append('MPICH_SKIP_MPICXX')
   ccflags.append('-Wall')
   # if(cca=='cca_no' or cca=='no'):
   # ccflags.append('-Werror')
   # 	ccflags.append('-pedantic')
   # ccflags.append('-pedantic-errors')
   ccflags.append('-Wstrict-aliasing')
   ccflags.append('-fstrict-aliasing')
   # ccflags.append('-fno-exceptions')
   # ccflags.append('-fno-rtti')
   ccflags.append('-Wno-long-long')
   ccflags.append('-Wno-unknown-pragmas')
   # if multicore == 'no' or multicore == 'multicore_no':
      # ccflags.append('-Wconversion')
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
   if(parallel == 'parallel_no' or parallel == 'no'):
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
   # ccflags.append('-fast')
   ccflags.append('-fstrict-aliasing')
   ccflags.append('-qpack_semantic=gnu')
   if build == 'debug':
      ccflags.append('-O0')
   elif build == 'asserts':
#      ccflags.append('-fast')
      # ccflags.append('-vec-report5') # is supressed by -ipo (included in -fast)
      # ccflags.append('-xHost')       # done by -fast
      # ccflags.append('-O3')          # done by -fast
      # ccflags.append('-no-prec-div') # done by -fast
      # ccflags.append('-static')      # done by -fast
      ccflags.append('-w')
#     ccflags.append('-Werror-all')
      ccflags.append('-align')
      ccflags.append('-ansi-alias')
   elif build == 'release':
#      ccflags.append('-fast')
      # ccflags.append('-vec-report5') # is supressed by -ipo (included in -fast)
      # ccflags.append('-xHost')       # done by -fast
      # ccflags.append('-O3')          # done by -fast
      # ccflags.append('-no-prec-div') # done by -fast
      # ccflags.append('-static')      # done by -fast
      ccflags.append('-w')
#     ccflags.append('-Werror-all')
      ccflags.append('-align')
      ccflags.append('-ansi-alias')
   # PN: If -fast is used for linking, the tbb-lib cannot be found :-(   
   # linkerflags.append('-fast')
   if multicore == 'openmp':
      ccflags.append('-openmp')
      linkerflags.append('-openmp')
else:
   print "ERROR: compiler must be = 'gcc', 'xlc' or 'icc'!"
   sys.exit(1)
   
##### Determine Scalasca Usage
#
scalasca = ARGUMENTS.get('scalasca', 'scalasca_no')  # Read command line parameter
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
buildpath = build_offset + '/' + str(build) + '/dim' + str(dim) + '/' 
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
addPeanoClawFlags(libpath, libs, cpppath, cppdefines)

##### Print options used to build
#
print
print "Building PeanoClaw"
print "Options: build = " + str(build) + ", dim = " + str(dim) + ", build-offset = " + str(build_offset) + ", parallel = " + str(parallel) + ", multicore = " + str(multicore) + ", compiler = " + str(compiler)
print "Buildpath: " + buildpath
print

VariantDir (buildpath, './src', duplicate=0)  # Set build directory for PeanoClaw sources
VariantDir (buildpath + 'kernel', p3Path, duplicate=0)  # Set build directory for Peano sources

##### Setup construction environment:
#
env = Environment (
   CPPDEFINES=cppdefines,
   LIBPATH=libpath,
   LIBS=libs,
   CPPPATH=cpppath,
   CCFLAGS=ccflags,
   LINKFLAGS=linkerflags,
   CXX=cxx,
   ENV=os.environ  # Makes environment variables visible to scons
   # tools      = compiler_tools
   )

################################################################################
#
# Define sources
#

##### Sub T-components

sourcesTLa = [
   Glob(buildpath + 'kernel/tarch/la/*.cpp'),
   Glob(buildpath + 'kernel/tarch/la/tests/*.cpp')
   ]

sourcesTCompiler = [
   Glob(buildpath + 'kernel/tarch/compiler/*.cpp')
   ]

sourcesTConfiguration = [
   Glob(buildpath + 'kernel/tarch/configuration/*.cpp')
   ]

sourcesTIrr = [
   Glob(buildpath + 'kernel/tarch/irr/*.cpp')
 ]

sourcesTLogging = [
  Glob(buildpath + 'kernel/tarch/logging/*.cpp'),
  Glob(buildpath + 'kernel/tarch/logging/configurations/*.cpp')
]

sourcesTServices = [
  Glob(buildpath + 'kernel/tarch/services/*.cpp')
]

sourcesTTests = [
  Glob(buildpath + 'kernel/tarch/tests/*.cpp'),
  Glob(buildpath + 'kernel/tarch/tests/configurations/*.cpp')
  ]

sourcesTUtils = [
  Glob(buildpath + 'kernel/tarch/utils/*.cpp')
]

sourcesTTiming = [
  Glob(buildpath + 'kernel/tarch/timing/*.cpp')
]

sourcesTPlotter = [ 
   Glob(buildpath + 'kernel/tarch/plotter/*.cpp'),
   Glob(buildpath + 'kernel/tarch/plotter/globaldata/*.cpp'),
   Glob(buildpath + 'kernel/tarch/plotter/globaldata/tests/*.cpp'),
   Glob(buildpath + 'kernel/tarch/plotter/griddata/*.cpp'),
   Glob(buildpath + 'kernel/tarch/plotter/griddata/multiscale/*.cpp'),
   Glob(buildpath + 'kernel/tarch/plotter/griddata/unstructured/*.cpp'),
   Glob(buildpath + 'kernel/tarch/plotter/griddata/unstructured/configurations/*.cpp'),
   Glob(buildpath + 'kernel/tarch/plotter/griddata/unstructured/vtk/*.cpp'),
   Glob(buildpath + 'kernel/tarch/plotter/griddata/unstructured/vtk/tests/*.cpp')
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
    Glob(buildpath + 'kernel/peano/datatraversal/*.cpp'),
    Glob(buildpath + 'kernel/peano/datatraversal/configurations/*.cpp'),
    Glob(buildpath + 'kernel/peano/datatraversal/tests/*.cpp'),
    Glob(buildpath + 'kernel/peano/datatraversal/autotuning/*.cpp'),
    Glob(buildpath + 'kernel/tarch/multicore/configurations/*.cpp'),
    Glob(buildpath + 'kernel/tarch/multicore/*.cpp')
  ]       
      
if multicore == 'no' or multicore == 'multicore_no':
   pass
elif multicore == 'openmp':
   sourcesDatatraversal = sourcesDatatraversal + [
     Glob(buildpath + 'kernel/tarch/multicore/openMP/*.cpp')
   ]
elif multicore == 'tbb':
   sourcesDatatraversal = sourcesDatatraversal + [
     Glob(buildpath + 'kernel/tarch/multicore/tbb/*.cpp')
   ]

sourcesParallel = [
 Glob(buildpath + 'kernel/tarch/parallel/configuration/*.cpp'),
 Glob(buildpath + 'kernel/tarch/parallel/*.cpp'),
 Glob(buildpath + 'kernel/tarch/parallel/strategy/*.cpp'),
 Glob(buildpath + 'kernel/tarch/parallel/messages/*.cpp'),
 Glob(buildpath + 'kernel/tarch/parallel/dastgen/*.cpp'),
 Glob(buildpath + 'kernel/tarch/parallel/configurations/*.cpp'),
 Glob(buildpath + 'kernel/peano/parallel/*.cpp'),
 Glob(buildpath + 'kernel/peano/parallel/configurations/*.cpp'),
 Glob(buildpath + 'kernel/peano/parallel/loadbalancing/*.cpp'),
 Glob(buildpath + 'kernel/peano/parallel/messages/*.cpp'),
 Glob(buildpath + 'kernel/peano/parallel/tests/*.cpp'),
 Glob(buildpath + 'kernel/tarch/mpianalysis/*.cpp')
]


#### Peano Utils
sourcesPeanoUtils = [
  Glob(buildpath + 'kernel/peano/utils/*.cpp')
]


# ## Peano partition coupling
sourcesPartitionCoupling = [
  Glob(buildpath + 'kernel/peano/integration/partitioncoupling/*.cpp'),
  Glob(buildpath + 'kernel/peano/integration/partitioncoupling/builtin/*.cpp'),
  Glob(buildpath + 'kernel/peano/integration/partitioncoupling/builtin/configurations/*.cpp'),
  Glob(buildpath + 'kernel/peano/integration/partitioncoupling/builtin/tests/*.cpp'),
  Glob(buildpath + 'kernel/peano/integration/partitioncoupling/builtin/records/*.cpp'),
  Glob(buildpath + 'kernel/peano/integration/partitioncoupling/services/*.cpp')
]

# ## Kernel
sourcesKernelConfiguration = [
   Glob(buildpath + 'kernel/peano/configurations/*.cpp')
   ]

sourcesGridInterface = [
   Glob(buildpath + 'kernel/peano/gridinterface/*.cpp')
   ]
   
sourcesGrid = [
   Glob(buildpath + 'kernel/peano/grid/*.cpp'),
   Glob(buildpath + 'kernel/peano/grid/aspects/*.cpp'),
   Glob(buildpath + 'kernel/peano/grid/nodes/*.cpp'),
   Glob(buildpath + 'kernel/peano/grid/nodes/loops/*.cpp'),
   Glob(buildpath + 'kernel/peano/grid/nodes/tasks/*.cpp'),
   Glob(buildpath + 'kernel/peano/grid/tests/*.cpp'),
   Glob(buildpath + 'kernel/peano/grid/tests/records/*.cpp')
   ]
sourcesStacks = [
    Glob(buildpath + 'kernel/peano/stacks/*.cpp'),
    Glob(buildpath + 'kernel/peano/stacks/implementation/*.cpp')
    ]
sourcesHeap = [
    Glob(buildpath + 'kernel/peano/heap/records/*.cpp')
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
   Glob(buildpath + 'kernel/peano/geometry/builtin/*.cpp'),
   Glob(buildpath + 'kernel/peano/geometry/builtin/services/*.cpp'),
   Glob(buildpath + 'kernel/peano/geometry/extensions/*.cpp'),
   Glob(buildpath + 'kernel/peano/geometry/builtin/configurations/*.cpp'),
   Glob(buildpath + 'kernel/peano/geometry/builtin/tests/*.cpp')
   ]


sourcesPeanoGeometry = [
   Glob(buildpath + 'kernel/peano/geometry/*.cpp'),
   Glob(buildpath + 'kernel/peano/geometry/tests/*.cpp'),
   Glob(buildpath + 'kernel/peano/geometry/configurations/*.cpp'),
   Glob(buildpath + 'kernel/peano/geometry/services/*.cpp'),
   sourcesBuiltinGeometry
   ]

sourcesPeanoBase = [
  sourcesPeanoKernel,
  sourcesPeanoGeometry,
  sourcesPeanoUtils,
  sourcesDatatraversal,
  # sourcesQueries,
  Glob(buildpath + 'kernel/peano/*.cpp'),
  Glob(buildpath + 'kernel/*.cpp')
]

sourcesToolBox = [
  Glob(buildpath + 'kernel/peano/toolbox/solver/*.cpp'),
  Glob(buildpath + 'kernel/peano/toolbox/solver/configurations/*.cpp'),
  Glob(buildpath + 'kernel/peano/toolbox/solver/tests/*.cpp'),
  Glob(buildpath + 'kernel/peano/toolbox/stencil/*.cpp'),
  Glob(buildpath + 'kernel/peano/toolbox/refinement/*.cpp')
]

sourcesToolBoxVHH = [
  Glob(buildpath + 'kernel/peano/toolbox/solver/vhh/*.cpp'),
  Glob(buildpath + 'kernel/peano/toolbox/solver/vhh/tests/*.cpp'),
  Glob(buildpath + 'kernel/peano/toolbox/solver/vhh/opencl/*.cpp'),
  Glob(buildpath + 'kernel/peano/toolbox/solver/vhh/opencl/tests/*.cpp')
]

# ## Applications
##### Define sources of application peanoclaw
sourcesPeanoClaw = [
  Glob(buildpath + 'peanoclaw/*.cpp'),
  Glob(buildpath + 'peanoclaw/adapters/*.cpp'),
  Glob(buildpath + 'peanoclaw/configurations/*.cpp'),
  Glob(buildpath + 'peanoclaw/interSubgridCommunication/*.cpp'),
  Glob(buildpath + 'peanoclaw/interSubgridCommunication/aspects/*.cpp'),
  Glob(buildpath + 'peanoclaw/mappings/*.cpp'),
  Glob(buildpath + 'peanoclaw/native/*.cpp'),
  Glob(buildpath + 'peanoclaw/parallel/*.cpp'),
  Glob(buildpath + 'peanoclaw/pyclaw/*.cpp'),
  Glob(buildpath + 'peanoclaw/records/*.cpp'),
  Glob(buildpath + 'peanoclaw/repositories/*.cpp'),
  Glob(buildpath + 'peanoclaw/runners/*.cpp'),
  Glob(buildpath + 'peanoclaw/statistics/*.cpp'),
  Glob(buildpath + 'peanoclaw/tests/*.cpp')
	]
################################################################################

##### Build selected target
#
targetfilename = 'libpeano-claw-' + str(dim) + 'd'
target = buildpath + targetfilename
library = env.SharedLibrary (
# env.Program (
  target=target,
  source=[
     sourcesTComponents,
     sourcesPeanoBase,
     sourcesPeanoClaw,
     sourcesParallel
     ]
  )
  
##### Copy library to Clawpack
#
installation = env.Alias('install', env.Install(clawpackPath + '/pyclaw/src/peanoclaw', library))

Default(installation)

