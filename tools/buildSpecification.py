
class BuildSpecification:
  """
  Specifies the settings for building against a certain solver.
  """
  def __init__(self, solverName, asLibrary, assertForPositiveValues, ARGUMENTS, cppdefines, cppflags, cpppath, libpath, requiredDim=None):
    self.name = name
    self.asLibrary = asLibrary
    
    ##### Determine dimension for which to build
    #
    self.dim = int(ARGUMENTS.get('dim', 2))  # Read command line parameter
    if self.dim == 2:
       cppdefines.append('Dim2')
    elif self.dim == 3:
       cppdefines.append('Dim3')
    else:
       print "ERROR: dim must be either 2 or 3!"
       sys.exit(1)
    
    ##### Add build parameter specific build variable settings:
    # This section only defines Peano-specific flags. It does not
    # set compiler specific stuff.
    #
    self.build = ARGUMENTS.get('build', 'debug')  # Read command line parameter
    if self.build == 'debug':
       cppdefines.append('Debug')
       cppdefines.append('Asserts')
       cppdefines.append('LogTrace')
       cppdefines.append('LogSeparator')
    elif self.build == 'release':
       pass
    elif self.build == 'asserts':
       cppdefines.append('Asserts')
       pass
    else:
       print "ERROR: build must be 'debug', 'asserts', or 'release'!"
       sys.exit(1)
       
    ##### Determine MPI-Parallelization
    #
    mpiConfigurationFile = ARGUMENTS.get('mpiconfig', 'openMPIConfiguration')
    mpiConfiguration = __import__(mpiConfigurationFile)
    
    self.parallel = ARGUMENTS.get('parallel', 'parallel_no')  # Read command line parameter
    if self.parallel == 'yes' or self.parallel == 'parallel_yes':
       cppdefines.append('Parallel')
       cppdefines.append('MPICH_IGNORE_CXX_SEEK')
       cppdefines.append('MPICH_SKIP_MPICXX')
       cpppath.extend(mpiConfiguration.getMPIIncludes())
       libpath.extend(mpiConfiguration.getMPILibrarypaths())
       libs.extend(mpiConfiguration.getMPILibraries())
    elif self.parallel == 'no' or self.parallel == 'parallel_no':
       pass
    else:
       print "ERROR: parallel must be = 'yes', 'parallel_yes', 'no' or 'parallel_no'!"
       sys.exit(1)
    
    ##### Determine Multicore usage
    #   
    self.multicore = ARGUMENTS.get('multicore', 'multicore_no')  # Read command line parameter
    if self.multicore == 'no' or multicore == 'multicore_no':
       pass
    elif self.multicore == 'openmp':
       ompDir = os.getenv ('OMP_DIR', '')
       cppdefines.append('SharedOMP')
       cpppath.append(ompDir + '/include')   
       pass
    elif self.multicore == 'tbb':
       libs.append('pthread')
       libs.append('dl')
       # Determine tbb directory and architecture from environment variables:
       tbbDir = os.getenv ('TBB_DIR')
              
       libs.append ('tbb')
    
       cppdefines.append('SharedTBB')
    elif self.multicore == 'opencl':
       libs.append('OpenCL')
       libs.append ('pthread')
       cppdefines.append('SIMD_OpenCL')
    else:
       print "ERROR: multicore must be = 'tbb',  'openmp', 'no' or 'multicore_no'!"
       sys.exit(1)
    
    ##### Determine Valgrind usage
    # 
    self.valgrind = ARGUMENTS.get('valgrind', 'no')
    if self.valgrind == 'no':
       pass
    elif self.valgrind == 'yes':
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
    self.compiler = ARGUMENTS.get('compiler', 'gcc')  # Read command line parameter
    if self.compiler == 'gcc':
       if(parallel == 'parallel_no' or parallel == 'no'):
         cxx = 'g++'
       else:
         cxx = 'mpicxx'
         cppdefines.append('MPICH_SKIP_MPICXX')
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
       if self.build == 'debug':
          ccflags.append('-g3')
          ccflags.append('-O0')
       elif self.build == 'asserts"':
          ccflags.append('-O2')
          ccflags.append('-g3') 
          ccflags.append('-ggdb')
       elif self.build == 'release':
          ccflags.append('-O3') 
       if self.multicore == 'openmp':
          ccflags.append('-fopenmp')
          linkerflags.append('-fopenmp')
    elif self.compiler == 'xlc':
       if(self.parallel == 'parallel_no' or self.parallel == 'no'):
         cxx = 'xlc++'
       else:
         cxx = 'mpixlcxx'
       if self.build == 'debug':
          ccflags.append('-g3')
          ccflags.append('-O0')
       elif self.build == 'asserts':
          ccflags.append('-qstrict')
          ccflags.append('-O2')
       elif self.build == 'release':
          ccflags.append('-qstrict')
          ccflags.append('-O3')
       if self.multicore == 'openmp':
          ccflags.append('-qsmp=omp')
          linkerflags.append('-qsmp=omp')
          cxx = cxx + '_r'
    elif self.compiler == 'icc':
       if(parallel == 'parallel_no' or parallel == 'no'):
         cxx = 'icpc'
       else:
         cxx = 'mpiCC'
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
       if multicore == 'openmp':
          ccflags.append('-openmp')
          linkerflags.append('-openmp')
    else:
       print "ERROR: compiler must be = 'gcc', 'xlc' or 'icc'!"
       sys.exit(1)
       
    ##### Determine Scalasca Usage
    #
    self.scalasca = ARGUMENTS.get('scalasca', 'scalasca_no')  # Read command line parameter
    if self.scalasca == 'yes' or self.scalasca == 'scalasca_yes':
       cxx = 'scalasca -instrument ' + cxx
    elif self.scalasca == 'no' or self.scalasca == 'scalasca_no':
       pass
    else:
       print "ERROR: scalasca must be = 'scalasca_yes', 'yes', 'scalasca_no' or 'no'!"
       sys.exit(1)
       
    if requiredDim != None:
      if self.dim != requiredDim:
        raise Exception('Solver "' + solverName + '" has to be compiled with ' + str(requiredDim) + ' dimensions.')
  
  def getInstallationTarget(self, source, parallel):
    if self.asLibrary:
      targetfilename = 'libpeano-claw-' + str(dim) + 'd' + filenameSuffix
      target = buildpath + targetfilename
      library = env.SharedLibrary (
        target=target,
        source=source
        )
        
      ##### Copy library to Clawpack
      #
      return env.Alias('install', env.Install('src/python/peanoclaw', library))      
    else:
      targetfilename = 'peano-claw-' + self.name
      target = buildpath + targetfilename
      executable = env.Program ( 
        target=target,
        source=source
        )
      ##### Copy executable to bin directory
      #
      return env.Alias('install', env.Install('bin', executable))