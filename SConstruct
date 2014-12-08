# Peano/SConstruct

import os
from os.path import join
from os.path import dirname
import sys
import shutil
from tools import buildtools

#########################################################################
##### MAIN CODE
#########################################################################

##### Initialize build variables
#
environment = Environment()
cxx = ''
linker = ''
archive = ''
cppdefines = []
cpppath = ['./src']
ccflags = []
linkerflags = []
libpath = []
libs = []
environmentVariables = {}

librarySuffix = ''
executableSuffix = ''

#Configure Peano 3
p3Path = '../peano3'
try:
  import peanoConfiguration
  p3Path = peanoConfiguration.getPeano3Path()
except ImportError:
  pass
p3SourcePath = join(p3Path, 'src')
toolboxSourcePath = join(p3Path, 'toolboxes')
cpppath.append(p3SourcePath)
cpppath.append(toolboxSourcePath)
cpppath.append(join(toolboxSourcePath, 'ControlLoopLoadBalancer'))

if not os.path.isdir(join(toolboxSourcePath, 'ControlLoopLoadBalancer')):
  shutil.copytree('tools/ControlLoopLoadBalancer/ControlLoopLoadBalancer', join(toolboxSourcePath, 'ControlLoopLoadBalancer'))

# Platform specific settings
# Only include library rt if not compiling on Mac OS.
if(environment['PLATFORM'] != 'darwin'):
    libs.append('rt')
    
##### Determine dimension for which to build
#
dim = int(ARGUMENTS.get('dim', 2))  # Read command line parameter
if dim == 2:
   cppdefines.append('Dim2')
elif dim == 3:
   cppdefines.append('Dim3')
else:
   print "ERROR: dim must be either 2 or 3!"
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
mpiConfigurationFile = ARGUMENTS.get('mpiconfig', 'openMPIConfiguration')
mpiConfiguration = __import__(mpiConfigurationFile)

parallel = ARGUMENTS.get('parallel', 'parallel_no')  # Read command line parameter
if parallel == 'yes' or parallel == 'parallel_yes':
   cppdefines.append('Parallel')
   cppdefines.append('MPICH_IGNORE_CXX_SEEK')
   cppdefines.append('MPICH_SKIP_MPICXX')
   cpppath.extend(mpiConfiguration.getMPIIncludes())
   libpath.extend(mpiConfiguration.getMPILibrarypaths())
   libs.extend(mpiConfiguration.getMPILibraries())
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
  buildtools.addTBB(cppdefines, cpppath, libpath, libs)
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
   valgrindRoot = os.getenv ('VALGRIND_ROOT')
   if(valgrindRoot == None):
     valgrindRoot = "/usr"
   cpppath.append(join(valgrindRoot, "include"))
   cpppath.append(join(valgrindRoot, "callgrind"))
else:
   print "ERROR: valgrind must be = 'yes' or 'no'!"
   sys.exit(1)

##### Determine gprof usage
# 
gprof = ARGUMENTS.get('gprof', 'no')
if gprof == 'no':
   pass
elif gprof == 'yes':
   ccflags.append('-pg')
   linkerflags.append('-pg')
else:
   print "ERROR: gprof must be = 'yes' or 'no'!"
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
     environmentVariables['OMPI_CXX'] = 'g++-4.8'
   ccflags.append('-Wall')
   ccflags.append('-Wstrict-aliasing')
   ccflags.append('-fstrict-aliasing')
   # ccflags.append('-fno-exceptions')
   # ccflags.append('-fno-rtti')
   ccflags.append('-Wno-long-long')
   ccflags.append('-Wno-unknown-pragmas')
   # if multicore == 'no' or multicore == 'multicore_no':
      # ccflags.append('-Wconversion')
   ccflags.append('-Wno-non-virtual-dtor')

   gccversion = environment['CCVERSION'].split('.')
   if int(gccversion[0]) > 4 or int(gccversion[1]) > 6:
     ccflags.append('-std=c++11')
   else:
     ccflags.append('-std=c++0x')
   if build == 'debug':
      ccflags.append('-g3')
      ccflags.append('-O0')
   elif build == 'asserts"':
      ccflags.append('-O2')
      ccflags.append('-g3') 
      ccflags.append('-ggdb')
   elif build == 'release':
      ccflags.append('-O3') 
   if multicore == 'openmp':
      ccflags.append('-fopenmp')
      linkerflags.append('-fopenmp')
elif compiler == 'xlc':
   ccflags.append('-qlanglvl=extended0x')
   ccflags.append('-qnoeh')
   if(parallel == 'parallel_no' or parallel == 'no'):
     cxx = 'xlc++'
   else:
     cxx = 'mpixlcxx'
     ccflags.append('-cxx=xlc++')
   if build == 'debug':
      ccflags.append('-g3')
      ccflags.append('-O0')
   elif build == 'asserts':
      ccflags.append('-qstrict')
      ccflags.append('-O2')
      ccflags.append('-g')
   elif build == 'release':
      ccflags.append('-qstrict')
      ccflags.append('-O3')
   if multicore == 'openmp':
      ccflags.append('-qsmp=omp')
      linkerflags.append('-qsmp=omp')
      cxx = cxx + '_r'
elif compiler == 'clang':
   if(parallel == 'parallel_no' or parallel == 'no'):
     cxx = 'clang'
   else:
     cxx = 'mpicxx'
     environmentVariables['OMPI_CXX'] = 'clang++'
   ccflags.append('-std=c++11')
   if build == 'debug':
     ccflags.append('-g3')
     ccflags.append('-O0')
   elif build == 'asserts"':
     ccflags.append('-O2')
     ccflags.append('-g3') 
   elif build == 'release':
     ccflags.append('-O3') 
   #if multicore == 'openmp':
   #  ccflags.append('-fopenmp')
   #  linkerflags.append('-fopenmp')
elif compiler == 'icc':
   if(parallel == 'parallel_no' or parallel == 'no'):
     cxx = 'icpc'
   else:
     cxx = 'mpiCC'
#    linker = 'xild'
#    archive = 'xiar'
   ccflags.append('-fstrict-aliasing')
   ccflags.append('-qpack_semantic=gnu')
   ccflags.append('-std=c++11')
   if build == 'debug':
      ccflags.append('-O0')
   elif build == 'asserts':
      ccflags.append('-w')
      ccflags.append('-align')
      ccflags.append('-ansi-alias')
      ccflags.append('-O2')
   elif build == 'release':
      ccflags.append('-w')
      ccflags.append('-align')
      ccflags.append('-ansi-alias')
      ccflags.append('-O3')
      ccflags.append('-xSSE4_2')
   if multicore == 'openmp':
      ccflags.append('-openmp')
      linkerflags.append('-openmp')
else:
   print "ERROR: compiler must be = 'gcc', 'xlc', 'clang', or 'icc'!"
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
   
##### Determine Solver
#
solver = ARGUMENTS.get('solver', 'pyclaw')
if solver == 'pyclaw':
  cppdefines.append('PYCLAW')
  cppdefines.append('PEANOCLAW_PYCLAW')
  cppdefines.append('AssertForPositiveValues')
elif solver == 'swe':
  #Configure SWE-Sources
  swePath = '../SWE/src'
  if dim != 2:
    raise Exception("The SWE solver can only be used in 2D.")
  try:
    import sweConfiguration
    swePath = sweConfiguration.getSWEPath()
  except ImportError:
    pass
  cpppath.append(swePath)
  cppdefines.append('SWE')
  cppdefines.append('PEANOCLAW_SWE')
  cppdefines.append('NDEBUG')
  
  solverType = ARGUMENTS.get('solverType', 'none')
  if solverType == 'f-wave':
    WAVE_PROPAGATION_SOLVER = 1
    executableSuffix += '-f-wave'
  elif solverType == 'augmentedRiemann':
    WAVE_PROPAGATION_SOLVER = 2
    executableSuffix += '-augmentedRiemann'
  elif solverType == 'f-wave-vectorized':
    WAVE_PROPAGATION_SOLVER = 4
    executableSuffix += ''
  else:
    raise Exception("Please specify solverType for SWE: f-wave, augmentedRiemann, or f-wave-vectorized")
  cppdefines.append('WAVE_PROPAGATION_SOLVER=' + str(WAVE_PROPAGATION_SOLVER))
  cppdefines.append('VECTORIZE')
  
  cppdefines.append('AssertForPositiveValues')
elif solver == 'fullswof2d':
  #Configure FullSWOF-Sources
  fullSWOF2DPath = '../FullSWOF_2D'
  if dim != 2:
    raise Exception("The FullSWOF2D solver can only be used in 2D.")
  try:
    import fullSWOF2DConfiguration
    fullSWOF2DPath = fullSWOF2DConfiguration.getFullSWOF2DPath()
  except ImportError:
    pass
  cpppath.append(fullSWOF2DPath)
  cppdefines.append('NDEBUG')

  cppdefines.append('DoNotAssertForPositiveValues')

  cppdefines.append('PEANOCLAW_FULLSWOF2D')
  cpppath.append( join(fullSWOF2DPath, 'Headers/liblimitations') )
  cpppath.append( join(fullSWOF2DPath, 'Headers/libfrictions') )
  cpppath.append( join(fullSWOF2DPath, 'Headers/libparser') )
  cpppath.append( join(fullSWOF2DPath, 'Headers/libflux') )
  cpppath.append( join(fullSWOF2DPath, 'Headers/libsave') )
  cpppath.append( join(fullSWOF2DPath, 'Headers/libschemes') )
  cpppath.append( join(fullSWOF2DPath, 'Headers/libreconstructions') )
  cpppath.append( join(fullSWOF2DPath, 'Headers/libinitializations') )
  cpppath.append( join(fullSWOF2DPath, 'Headers/librain_infiltration') )
  cpppath.append( join(fullSWOF2DPath, 'Headers/libboundaryconditions') )
  cpppath.append( join(fullSWOF2DPath, 'Headers/libparameters') )
  libs.append('png') # for texture file
  
  swashes = ARGUMENTS.get('swashes', 'swashes_no')
  swashesPath = '../swashes'
  if swashes == 'yes' or swashes == 'swashes_yes':
    cppdefines.append('PEANOCLAW_SWASHES')
    try:
      import swashesConfiguration
      swashesPath = swashesConfiguration.getSWEPath()
    except ImportError:
      pass
    cpppath.append(join(swashesPath, 'Include'))
  elif swashes == 'no' or swashes == 'swashes_no':
    pass
  else:
    raise Exception("Please specify 'yes' or 'no' for parameter 'swashes'.")
elif solver == 'euler3d':
  cppdefines.append('PEANOCLAW_EULER3D')
  if dim != 3:
    raise Exception("The Euler3D solver can only be used in 3D.")
  euler3DEulerEquationsPath = '../euler3DEulerEquations'
  euler3DUniPath = '../euler3DUni'
  try:
    import euler3DConfiguration
    euler3DEulerEquationsPath = euler3DConfiguration.getEulerEquationsPath()
    euler3DUniPath = euler3DConfiguration.getUniPath()
  except ImportError:
    pass
  #cpppath.append(join(euler3DEulerEquationsPath, 'source'))
  cpppath.append(join(euler3DUniPath, 'source'))
  cpppath.append(join(euler3DUniPath, 'include'))
  cpppath.append('/usr/include/eigen3')
  cppdefines.append('AssertForPositiveValues')
  buildtools.addBoost(environment, cpppath, libpath)
  if multicore != 'tbb':
    buildtools.addTBB(cppdefines, cpppath, libpath, libs, enablePeanoTBBSupport=False)
    
  solverType = ARGUMENTS.get('solverType', 'vectorized')
  if solverType == 'vectorized':
    pass
  elif solverType == 'non-vectorized':
    executableSuffix += '-non-vectorized'
    cppdefines.append('EIGEN_DONT_VECTORIZE')
  else:
    raise Exception("Please specify solverType for Euler3D: 'vectorized' or 'non-vectorized'")
else:
  raise Exception("ERROR: solver must be 'pyclaw', 'swe', 'fullswof2d', or 'euler3d'")

##### Determine Heap Compression
#
heapCompression = ARGUMENTS.get('heapCompression', 'yes')
if heapCompression == 'no':
  cppdefines.append('noPackedEmptyHeapMessages')
  librarySuffix += '_noHeapCompression'
elif heapCompression == 'yes':
  pass
else:
  raise Exception("ERROR: heapCompression must be 'yes' or 'no'")

##### Determine HDF5 usage
# 
hdf5 = ARGUMENTS.get('hdf5', 'no')
if hdf5 == 'yes':
  cppdefines.append('PEANOCLAW_USE_HDF5')
  libs.append('hdf5')
  libs.append('hdf5_hl')
  if 'HDF5_INC' in os.environ:
    cpppath.append(os.environ['HDF5_INC'])
    cpppath.insert(0,'/home/unterweg/.local/include')
    libpath.append(join(os.environ['HDF5_BASE'], 'lib'))
    libpath.insert(0, '/home/unterweg/local/lib')
  
##### Determine VTU usage
# 
vtu = ARGUMENTS.get('vtu', 'yes')
if vtu == 'yes':
  cppdefines.append('PEANOCLAW_USE_VTU')
   
##### Determine build path
#
build_offset = ARGUMENTS.get('buildoffset', 'build')
buildpath = build_offset + '/' + str(build) + '/dim' + str(dim) + '/' 
if multicore == 'tbb':
   buildpath = join(buildpath, 'tbb')
   executableSuffix += '-tbb'
elif multicore == 'openmp':
   buildpath = join(buildpath, 'openmp')
   executableSuffix += '-openmp'
elif multicore == 'opencl':
   buildpath = join(buildpath, 'openCL')
   executableSuffix += '-opencl'
else:
   buildpath = join(buildpath, 'multicore_no')
if parallel == 'yes' or parallel == 'parallel_yes':
   buildpath = join(buildpath, 'parallel_yes')
   executableSuffix += '-parallel'
else:
   buildpath = join(buildpath, 'parallel_no')
buildpath = join(buildpath, compiler)
buildpath = join(buildpath, solver)
if scalasca == 'yes' or scalasca == 'scalasca_yes':
   buildpath = join(buildpath, 'scalasca')
if heapCompression == 'no':
   buildpath = join(buildpath, 'noHeapCompression')
if gprof == 'yes':
  buildpath = join(buildpath, 'gprof') 

buildpath = buildpath + '/'
   
##### Specify build settings
#
buildtools.addPeanoClawFlags(environment, build, libpath, libs, cpppath, cppdefines, ccflags, solver)

##### Print options used to build
#
print
print "Building PeanoClaw"
print "Options: build = " + str(build) + ", dim = " + str(dim) + ", build-offset = " + str(build_offset) + ", parallel = " + str(parallel) + ", multicore = " + str(multicore) + ", compiler = " + str(compiler)
print "Buildpath: " + buildpath
print

VariantDir (buildpath, './src', duplicate=0)  # Set build directory for PeanoClaw sources
VariantDir (join(buildpath, 'kernel'), p3SourcePath, duplicate=0)  # Set build directory for Peano sources
VariantDir (join(buildpath, 'toolboxes'), toolboxSourcePath, duplicate=0)  # Set build directory for Toolbox sources
if solver == 'swe':
  VariantDir (join(buildpath, 'swe'), swePath, duplicate=0)  # Set build directory for SWE sources
elif solver == 'fullswof2d':
  VariantDir (join(buildpath, 'fullswof2d'), fullSWOF2DPath, duplicate=0)  # Set build directory for FullSWOF2D sources
  VariantDir (join(buildpath, 'swashes'), swashesPath, duplicate=0)  # Set build directory for Euler3D sources
elif solver == 'euler3d':
  VariantDir (join(buildpath, 'euler3dUni'), euler3DUniPath, duplicate=0)  # Set build directory for Euler3D sources
  
  
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
   LINK=linker if linker!='' else cxx,
   AR=archive if archive!='' else 'ar',
   ENV=dict(os.environ.items() + environmentVariables.items())  # Makes environment variables visible to scons
   # tools      = compiler_tools
   )

################################################################################
#
# Define sources
#
sourcesPeano = buildtools.getPeanoSources(Glob, buildpath, multicore)

sourcesPeanoClaw = buildtools.getPeanoClawSources(Glob, buildpath)

##### Define sources of application peanoclaw
if solver == 'swe':
  sourcesSolver = [
    Glob(join(buildpath, 'peanoclaw/native/main.cpp')),
    Glob(join(buildpath, 'peanoclaw/native/sweMain.cpp')),
    Glob(join(buildpath, 'peanoclaw/native/SWEKernel.cpp')),
    Glob(join(buildpath, 'peanoclaw/native/SWE_WavePropagationBlock_patch.cpp')),
    Glob(join(buildpath, 'peanoclaw/native/SWECommandLineParser.cpp')),
    Glob(join(buildpath, 'peanoclaw/native/scenarios/*.cpp')),
    Glob(join(buildpath, 'swe/tools/Logger.cpp')),
    Glob(join(buildpath, 'swe/writer/VtkWriter.cpp')),
    Glob(join(buildpath, 'swe/blocks/SWE_Block.cpp'))
    ]
  if(WAVE_PROPAGATION_SOLVER == 1 or WAVE_PROPAGATION_SOLVER == 2 or WAVE_PROPAGATION_SOLVER == 3):
    sourcesSolver.append(Glob(join(buildpath, 'swe/blocks/SWE_WavePropagationBlock.cpp')))
  else:
    sourcesSolver.append(Glob(join(buildpath, 'swe/blocks/SWE_WaveAccumulationBlock.cpp')))
elif solver == 'pyclaw':
  sourcesSolver = [
     Glob(join(buildpath, 'peanoclaw/pyclaw/*.cpp'))
     ]
elif solver == 'fullswof2d':
  sourcesSolver = [
     Glob(join(buildpath, 'peanoclaw/native/FullSWOF2D.cpp')),
     Glob(join(buildpath, 'peanoclaw/native/MekkaFlood_solver.cpp')),
     Glob(join(buildpath, 'peanoclaw/native/dem.cpp')),
     Glob(join(buildpath, 'peanoclaw/native/main.cpp')),
     Glob(join(buildpath, 'peanoclaw/native/fullswof2DMain.cpp')),
     Glob(join(buildpath, 'peanoclaw/native/scenarios/*.cpp')),
     Glob(join(buildpath, 'fullswof2d/Sources/liblimitations/*.cpp')),
     Glob(join(buildpath, 'fullswof2d/Sources/libfrictions/*.cpp')),
     Glob(join(buildpath, 'fullswof2d/Sources/libparser/*.cpp')),
     Glob(join(buildpath, 'fullswof2d/Sources/libflux/*.cpp')),
     Glob(join(buildpath, 'fullswof2d/Sources/libsave/*.cpp')),
     Glob(join(buildpath, 'fullswof2d/Sources/libschemes/*.cpp')),
     Glob(join(buildpath, 'fullswof2d/Sources/libreconstructions/*.cpp')),
     Glob(join(buildpath, 'fullswof2d/Sources/libinitializations/*.cpp')),
     Glob(join(buildpath, 'fullswof2d/Sources/librain_infiltration/*.cpp')),
     Glob(join(buildpath, 'fullswof2d/Sources/libboundaryconditions/*.cpp')),
     Glob(join(buildpath, 'fullswof2d/Sources/libparameters/*.cpp'))
     ]
  if swashes == 'yes' or swashes == 'swashes_yes':
    sourcesSolver.append(Glob(join(buildpath, 'peanoclaw/native/scenarios/swashes/*.cpp')))
    sourcesSolver.append(Glob(join(buildpath, 'swashes/Sources/*.cpp')))
elif solver == 'euler3d':
  sourcesSolver = [
     Glob(join(buildpath, 'euler3dUni/source/EulerEquations/*.cpp')),
     Glob(join(buildpath, 'euler3dUni/source/Logging/*.cpp')),
     Glob(join(buildpath, 'peanoclaw/native/main.cpp')),
     Glob(join(buildpath, 'peanoclaw/native/scenarios/*.cpp')),
     Glob(join(buildpath, 'peanoclaw/solver/euler3d/*.cpp'))
     ]
sourcesPeanoClaw.extend(sourcesSolver)

source = [
   sourcesPeano,
   sourcesPeanoClaw
   ]

################################################################################

##### Configure
if os.path.isfile(join(p3SourcePath, 'peano/parallel/MeshCommunication.h')):
  print 'Using RMK'
  env['CPPDEFINES'].append('UseBlockedMeshCommunication')
else:
  print 'Using Peano classic communication'
  env['CPPDEFINES'].append('DoNotUseBlockedMeshCommunication')

##### Build selected target
#
if solver == 'pyclaw':
  targetfilename = 'libpeano-claw-' + str(dim) + 'd' + librarySuffix
  target = buildpath + targetfilename
  library = env.SharedLibrary (
    target=target,
    source=source
    )
    
  ##### Copy library to Clawpack
  #
  installation = env.Alias('install', env.Install('src/python/peanoclaw', library))
elif solver == 'swe':
  targetfilename = 'peano-claw-swe' + executableSuffix
  target = buildpath + targetfilename
  executable = env.Program ( 
    target=target,
    source=source
    )
  ##### Copy executable to bin directory
  #
  installation = env.Alias('install', env.Install('bin', executable))    
elif solver == 'fullswof2d':
  targetfilename = 'peano-claw-fullswof2d' + executableSuffix
  target = buildpath + targetfilename
  executable = env.Program ( 
    target=target,
    source=source
    )
  ##### Copy executable to bin directory
  #
  installation = env.Alias('install', env.Install('bin', executable))
elif solver == 'euler3d':
  targetfilename = 'peano-claw-euler3d' + executableSuffix
  target = buildpath + targetfilename
  executable = env.Program ( 
    target=target,
    source=source
    )
  ##### Copy executable to bin directory
  #
  installation = env.Alias('install', env.Install('bin', executable))

#installation = solverSpecification.getInstallationTarget() 
Default(installation)

