/*
 * pyclawBindings.c
 *
 *  Created on: Feb 7, 2012
 *      Author: Kristof Unterweger
 */

#include <Python.h>
#include "peano/utils/Globals.h"
#include "tarch/logging/Log.h"
#include "tarch/logging/CommandLineLogger.h"

#include "peano/peano.h"

#include <list>
#include <numpy/arrayobject.h>

#include "peanoclaw/Numerics.h"
#include "peanoclaw/NumericsFactory.h"
#include "peanoclaw/configurations/PeanoClawConfigurationForSpacetreeGrid.h"
#include "peanoclaw/runners/PeanoClawLibraryRunner.h"
#include "tarch/tests/TestCaseRegistry.h"

#include "peanoclaw/pyclaw/PyClawCallbacks.h"

#if USE_VALGRIND
#include <callgrind.h>
#endif

static bool _calledFromPython;
static PyGILState_STATE _pythonState;
static peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid* _configuration;

void importArrays() {
  import_array();
}

void configureLogFilter() {
  // Configure the output
  tarch::logging::CommandLineLogger::getInstance().clearFilterList();
  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "info", false ) );
  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "debug", true ) );

  //Disable minimal time subgrid
//  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "info", -1, "peanoclaw::statistics::SubgridStatistics::logStatistics", true ) );

  //Selective Tracing
  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "info", -1, "peano::parallel", true ) );

  std::ostringstream logFileName;
  #ifdef Parallel
  logFileName << "rank-" << tarch::parallel::Node::getInstance().getRank() << "-trace.txt";
  #endif
  tarch::logging::CommandLineLogger::getInstance().setLogFormat( " ", false, false, true, false, true, logFileName.str() );
}

tarch::la::Vector<DIMENSIONS, double> convertToVector(double v0, double v1, double v2) {
  tarch::la::Vector<DIMENSIONS, double> vector;
  vector(0) = v0;
  vector(1) = v1;
  #ifdef Dim3
  vector(2) = v2;
  #endif
  return vector;
}

tarch::la::Vector<DIMENSIONS, int> convertToVector(int v0, int v1, int v2) {
  tarch::la::Vector<DIMENSIONS, int> vector;
  vector(0) = v0;
  vector(1) = v1;
  #ifdef Dim3
  vector(2) = v2;
  #endif
  return vector;
}

extern "C"
peanoclaw::runners::PeanoClawLibraryRunner* pyclaw_peano_new (
  double initialMinimalMeshWidthScalar,
  double domainOffsetX0,
  double domainOffsetX1,
  double domainOffsetX2,
  double domainSizeX0,
  double domainSizeX1,
  double domainSizeX2,
  int subdivisionFactorX0,
  int subdivisionFactorX1,
  int subdivisionFactorX2,
  int unknownsPerSubcell,
  int auxiliarFieldsPerSubcell,
  int ghostlayerWidth,
  double initialTimestepSize,
  char* configurationFile,
  bool useDimensionalSplittingOptimization,
  InitializationCallback initializationCallback,
  BoundaryConditionCallback boundaryConditionCallback,
  SolverCallback solverCallback,
  AddPatchToSolutionCallback addPatchToSolutionCallback,
  InterPatchCommunicationCallback interpolationCallback,
  InterPatchCommunicationCallback restrictionCallback,
  InterPatchCommunicationCallback fluxCorrectionCallback,
  int *rank
) {
  peano::fillLookupTables();

#if defined(Parallel)
  char argv[2][256];
  int argc = 1;
  //sprintf(argv[0], "%s", "peanoclaw");
  peano::initParallelEnvironment(&argc,(char ***)&argv);
  peano::initSharedMemoryEnvironment();
#endif

  //Initialize Python
  _calledFromPython = Py_IsInitialized();

  if(_calledFromPython) {
    //Needed to take over the Python context from the calling process.
    _pythonState = PyGILState_Ensure();
  } else {
    Py_Initialize();
  }

  importArrays();

  //Initialize Logger
  static tarch::logging::Log _log("::pyclawBindings");
  logInfo("pyclaw_peano_new(...)", "Initializing Peano");

  configureLogFilter();

  //Numerics -- this object is copied to the runner and is stored there.
  peanoclaw::NumericsFactory numericsFactory;
  peanoclaw::Numerics* numerics = numericsFactory.createPyClawNumerics(
    initializationCallback,
    boundaryConditionCallback,
    solverCallback,
    addPatchToSolutionCallback,
    interpolationCallback,
    restrictionCallback,
    fluxCorrectionCallback
  );

  _configuration = new peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid;

  //Construct parameters
  tarch::la::Vector<DIMENSIONS, double> domainOffset = convertToVector(domainOffsetX0, domainOffsetX1, domainOffsetX2);
  tarch::la::Vector<DIMENSIONS, double> domainSize = convertToVector(domainSizeX0, domainSizeX1, domainSizeX2);
  
  tarch::la::Vector<DIMENSIONS, double> initialMinimalMeshWidth(initialMinimalMeshWidthScalar);
  tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = convertToVector(subdivisionFactorX0, subdivisionFactorX1, subdivisionFactorX2);

  //Check parameters
  assertion1(tarch::la::greater(domainSizeX0, 0.0) && tarch::la::greater(domainSizeX1, 0.0), domainSize);
  if(initialMinimalMeshWidthScalar > domainSizeX0 || initialMinimalMeshWidthScalar > domainSizeX1) {
    logError("pyclaw_peano_new(...)", "Domainsize or initialMinimalMeshWidth not set properly.");
  }
  if(tarch::la::oneGreater(tarch::la::Vector<DIMENSIONS, int>(1), subdivisionFactor) ) {
    logError("pyclaw_peano_new(...)", "subdivisionFactor not set properly.");
  }
 
  //Create runner
  peanoclaw::runners::PeanoClawLibraryRunner* runner
    = new peanoclaw::runners::PeanoClawLibraryRunner(
    *_configuration,
    *numerics,
    domainOffset,
    domainSize,
    initialMinimalMeshWidth,
    subdivisionFactor,
    ghostlayerWidth,
    unknownsPerSubcell,
    auxiliarFieldsPerSubcell,
    initialTimestepSize,
    useDimensionalSplittingOptimization
  );

  assertion(runner != 0);
 
#if defined(Parallel) 
  *rank = tarch::parallel::Node::getInstance().getRank();
#else
  *rank = 0;
#endif

  if(_calledFromPython) {
    PyGILState_Release(_pythonState);
  }

  return runner;
}

extern "C"
void pyclaw_peano_runTests() {
  if(_calledFromPython) {
    _pythonState = PyGILState_Ensure();
  }

  tarch::tests::TestCaseRegistry::getInstance().getTestCaseCollection().run();
  if(tarch::tests::TestCaseRegistry::getInstance().getTestCaseCollection().getNumberOfErrors() > 0) {
    exit(1);
  }

  if(_calledFromPython) {
    PyGILState_Release(_pythonState);
  }
}

extern "C"
void pyclaw_peano_destroy(peanoclaw::runners::PeanoClawLibraryRunner* runner) {
  static tarch::logging::Log _log("::pyclawBindings");
  logTraceIn("pyclaw_peano_destroy");
  assertionMsg(runner!=0, "call pyclaw_peano_new before calling pyclaw_peano_destroy.");

  delete runner;

  if(_configuration != 0) {
    delete _configuration;
  }

  if(!_calledFromPython) {
    Py_Finalize();
  }
 
  logTraceOut("pyclaw_peano_destroy");
}

extern "C"
void pyclaw_peano_evolveToTime(double time, peanoclaw::runners::PeanoClawLibraryRunner* runner) {
  #ifdef USE_VALGRIND
  CALLGRIND_START_INSTRUMENTATION;
  CALLGRIND_ZERO_STATS;
  #endif

  static tarch::logging::Log _log("::pyclawBindings");
  logTraceInWith1Argument("pyclaw_peano_evolveToTime", time);
  assertionMsg(runner!=0, "call pyclaw_peano_new before calling pyclaw_peano_run.");

  if(_calledFromPython) {
    _pythonState = PyGILState_Ensure();
  }

  runner->evolveToTime(time);

  if(_calledFromPython) {
    PyGILState_Release(_pythonState);
  }

  #ifdef USE_VALGRIND
  CALLGRIND_STOP_INSTRUMENTATION;
  #endif
  logTraceOut("pyclaw_peano_evolveToTime");
}

extern "C"
void pyclaw_peano_gatherSolution(peanoclaw::runners::PeanoClawLibraryRunner* runner) {
  static tarch::logging::Log _log("::pyclawBindings");
  logTraceIn("pyclaw_peano_gatherSolution")
  if(_calledFromPython) {
    _pythonState = PyGILState_Ensure();
  }

#if defined(Parallel)
  if (tarch::parallel::Node::getInstance().isGlobalMaster()) {
#endif

    runner->gatherCurrentSolution();

#if defined(Parallel)
  }
#endif

  if(_calledFromPython) {
    PyGILState_Release(_pythonState);
  }
  logTraceOut("pyclaw_peano_gatherSolution");
}

extern "C"
int pyclaw_peano_runWorker(peanoclaw::runners::PeanoClawLibraryRunner* runner)
{
  if(_calledFromPython) {
    _pythonState = PyGILState_Ensure();
  }

  int result = runner->runWorker();

  if(_calledFromPython) {
    PyGILState_Release(_pythonState);
  }

  return result;
}
