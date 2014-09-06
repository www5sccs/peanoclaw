/*
 * pyclawBindings.c
 *
 *  Created on: Feb 7, 2012
 *      Author: Kristof Unterweger
 */

#include "peano/utils/Globals.h"
#include "tarch/logging/Log.h"
#include "tarch/logging/CommandLineLogger.h"

#include "peano/peano.h"

#include <list>
#include <vector>

#include "peanoclaw/Patch.h"
#include "peanoclaw/Numerics.h"
#include "peanoclaw/NumericsFactory.h"
#include "peanoclaw/configurations/PeanoClawConfigurationForSpacetreeGrid.h"
#include "peanoclaw/native/SWEKernel.h"
#include "peanoclaw/native/SWECommandLineParser.h"
#include "peanoclaw/native/scenarios/SWEScenario.h"
#include "peanoclaw/runners/PeanoClawLibraryRunner.h"
#include "tarch/logging/LogFilterFileReader.h"
#include "tarch/logging/Log.h"
#include "tarch/tests/TestCaseRegistry.h"

#ifdef PEANOCLAW_SWE
#include "peanoclaw/native/sweMain.h"
#elif PEANOCLAW_FULLSWOF2D
#include "peanoclaw/native/fullswof2DMain.h"
#endif

#if USE_VALGRIND
#include <callgrind.h>
#endif

static peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid* _configuration;

static tarch::logging::Log _log("::main");

void initializeLogFilter() {
  //Initialize Logger
  static tarch::logging::Log _log("peanoclaw");
  logInfo("main(...)", "Initializing Peano");

  // Configure the output
  tarch::logging::CommandLineLogger::getInstance().clearFilterList();
  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "info", false ) );

  //Validation
  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "info", -1, "peanoclaw::statistics::ParallelGridValidator", true ) );
  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "info", -1, "peanoclaw::mappings::ValidateGrid", true ) );

  //Selective Tracing
  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "info", -1, "tarch::mpianalysis", true ) );
  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "info", -1, "peano::", true ) );

  std::string logFilterFileName = "peanoclaw.logfilter";
  std::ifstream logFilterFile(logFilterFileName.c_str());
  if(logFilterFile) {
    tarch::logging::LogFilterFileReader::parsePlainTextFile(logFilterFileName);
  }

  std::ostringstream logFileName;
  #ifdef Parallel
  logFileName << "rank-" << tarch::parallel::Node::getInstance().getRank() << "-trace.txt";
  #endif
  tarch::logging::CommandLineLogger::getInstance().setLogFormat( " ", false, false, true, false, true, logFileName.str() );
}

int readOptionalArguments(int argc, char** argv, bool& usePeanoClaw, std::string& plotName, int& numberOfThreads) {
  int remaining = argc;

  //Default values
  usePeanoClaw = false;
  plotName = "adaptive";
  numberOfThreads = -1;

  for(int i = 0; i < argc; i++) {
    std::string key(argv[i]);

    if(key == "--usePeano") {
      usePeanoClaw = true;
      remaining--;
    } else if (key == "--plotName") {
      plotName = argv[i+1];
      remaining -= 2;
    } else if (key == "--threads") {
      std::stringstream s(argv[i+1]);
      s >> numberOfThreads;
      remaining -= 2;
    }

  }

  return remaining;
}

void runSimulation(
  peanoclaw::native::scenarios::SWEScenario& scenario,
  peanoclaw::Numerics& numerics,
  std::string& plotName,
  bool useCornerExtrapolation,
  int numberOfThreads
) {
  tarch::la::Vector<DIMENSIONS,double> domainOffset = scenario.getDomainOffset();
  tarch::la::Vector<DIMENSIONS,double> domainSize = scenario.getDomainSize();
  tarch::la::Vector<DIMENSIONS,double> initialMinimalMeshWidth = scenario.getInitialMinimalMeshWidth();
  tarch::la::Vector<DIMENSIONS,int>    subdivisionFactor = scenario.getSubdivisionFactor();

  //Check parameters
  assertion1(tarch::la::greater(domainSize(0), 0.0) && tarch::la::greater(domainSize(1), 0.0), domainSize);
  if(initialMinimalMeshWidth(0) > domainSize(0) || initialMinimalMeshWidth(1) > domainSize(1)) {
    logError("main(...)", "Domainsize or initialMinimalMeshWidth not set properly.");
  }
  if(tarch::la::oneGreater(tarch::la::Vector<DIMENSIONS, int>(1), subdivisionFactor(0)) ) {
    logError("main(...)", "subdivisionFactor not set properly.");
  }

  //Create runner
  peanoclaw::runners::PeanoClawLibraryRunner* runner
    = new peanoclaw::runners::PeanoClawLibraryRunner(
    *_configuration,
    numerics,
    domainOffset,
    domainSize,
    initialMinimalMeshWidth,
    subdivisionFactor,
    scenario.getInitialTimestepSize(),
    useCornerExtrapolation,
    numberOfThreads,
    true, //Reduce reductions
    1, //Fork level increment
    plotName
  );

#if defined(Parallel)
  std::cout << tarch::parallel::Node::getInstance().getRank() << ": peano instance created" << std::endl;
#endif

  assertion(runner != 0);

  // run experiment
#if defined(Parallel)
  if (tarch::parallel::Node::getInstance().isGlobalMaster()) {
#endif
    double time = 0.0;
    do {
      time += scenario.getGlobalTimestepSize();
        peanoclaw::State& state = runner->getState();
      runner->evolveToTime(std::min(time, scenario.getEndTime()));
      //runner->gatherCurrentSolution();
      //std::cout << "time " << time << " numberOfCells " << state.getNumberOfInnerCells() << std::endl;
    } while(tarch::la::smaller(time, scenario.getEndTime()));
#if defined(Parallel)
  } else {
    runner->runWorker();
  }
#endif

  // experiment done -> cleanup
  delete runner;

  if(_configuration != 0) {
    delete _configuration;
  }
}

int main(int argc, char **argv) {
  peano::fillLookupTables();

#if defined(Parallel)
  int parallelSetup = peano::initParallelEnvironment(&argc,(char ***)&argv);
  int sharedMemorySetup = peano::initSharedMemoryEnvironment();
#endif

  initializeLogFilter();

  //Tests
  if(false) {
    tarch::tests::TestCaseRegistry::getInstance().getTestCaseCollection().run();
  }

#if defined(PEANOCLAW_SWE) || defined(PEANOCLAW_FULLSWOF2D) || defined(PEANOCLAW_EULER3D)
  _configuration = new peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid;

  bool usePeanoClaw;
  std::string plotName;
  int numberOfThreads;
  argc = readOptionalArguments(argc, argv, usePeanoClaw, plotName, numberOfThreads);

  //Create Scenario
  peanoclaw::native::scenarios::SWEScenario* scenario;
  try {
     scenario
      = peanoclaw::native::scenarios::SWEScenario::createScenario(argc, argv);
  } catch(...) {
    scenario = 0;
  }

  if(scenario == 0) {
    std::cerr << "Optional arguments: [--plotName <plotName>] [--usePeano]" << std::endl;
    return 0;
  }

  peanoclaw::NumericsFactory numericsFactory;
  #if defined(PEANOCLAW_SWE)
  peanoclaw::Numerics* numerics = numericsFactory.createSWENumerics(*scenario);
  #elif defined(PEANOCLAW_FULLSWOF2D)
  peanoclaw::Numerics* numerics = numericsFactory.createFullSWOF2DNumerics(*scenario);
  #elif defined(PEANOCLAW_EULER3D)
  peanoclaw::Numerics* numerics = numericsFactory.createEuler3DNumerics(*scenario);
  #endif

  if(usePeanoClaw) {
    runSimulation(
      *scenario,
      *numerics,
      plotName,
      false,
      numberOfThreads
    );
  } else {
  #if defined(PEANOCLAW_SWE)
  tarch::la::Vector<DIMENSIONS,int> numberOfCells = scenario->getSubdivisionFactor();
  peanoclaw::native::sweMain(*scenario, numberOfCells);
  #elif defined(PEANOCLAW_FULLSWOF2D)
  tarch::la::Vector<DIMENSIONS,int> numberOfCells = scenario->getSubdivisionFactor();
  peanoclaw::native::fullswof2DMain(*scenario, numberOfCells);
  #elif defined(PEANOCLAW_EULER3D)

  #endif
  }

  delete scenario;
#endif
  return 0;
}
