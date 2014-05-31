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

#include "peanoclaw/Patch.h"
#include "peanoclaw/Numerics.h"
#include "peanoclaw/NumericsFactory.h"
#include "peanoclaw/configurations/PeanoClawConfigurationForSpacetreeGrid.h"
#include "peanoclaw/native/sweMain.h"
#include "peanoclaw/native/SWEKernel.h"
#include "peanoclaw/native/SWECommandLineParser.h"
#include "peanoclaw/runners/PeanoClawLibraryRunner.h"
#include "tarch/logging/LogFilterFileReader.h"
#include "tarch/logging/Log.h"
#include "tarch/tests/TestCaseRegistry.h"

#if USE_VALGRIND
#include <callgrind.h>
#endif

#if defined(SWE)
#include "peanoclaw/native/BreakingDam.h"
#endif

#if defined(PEANOCLAW_FULLSWOF2D)
#include "peanoclaw/native/MekkaFlood.h"
#include "peanoclaw/native/dem.h"
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

void runSimulation(
  peanoclaw::native::SWEKernelScenario& scenario,
  peanoclaw::Numerics& numerics,
  bool useCornerExtrapolation
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
    useCornerExtrapolation
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
      runner->evolveToTime(time);
      //runner->gatherCurrentSolution();
      std::cout << "time " << time << " numberOfCells " << state.getNumberOfInnerCells() << std::endl;
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

#if defined(SWE) || defined(PEANOCLAW_FULLSWOF2D)
  _configuration = new peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid;

  //PyClaw - this object is copied to the runner and is stored there.
  peanoclaw::NumericsFactory numericsFactory;

  //Construct parameters
#if defined(PEANOCLAW_FULLSWOF2D)

    DEM dem;

    dem.load("DEM_400cm.bin");
 
//    tarch::la::Vector<DIMENSIONS, double> domainOffset(0);

    // keep aspect ratio of map: 4000 3000: ratio 4:3
    tarch::la::Vector<DIMENSIONS, int> subdivisionFactor;
    subdivisionFactor(0) = static_cast<int>(16); // 96 //  6 * 4, optimum in non optimized version
    subdivisionFactor(1) = static_cast<int>(9);  // 54 //  6 * 3, optimum in non optimized version

    tarch::la::Vector<DIMENSIONS, double> maximalMeshWidth(1.0/(9.0 * subdivisionFactor(0)));
    tarch::la::Vector<DIMENSIONS, double> minimalMeshWidth(1.0/(81.0 * subdivisionFactor(0)));

    double globalTimestepSize = 2.0; //0.1;//1.0;
    double endTime = 3600.0; // 100.0;
//    double initialTimestepSize = 1.0;

    bool usePeanoClaw = true;

    peanoclaw::native::MekkaFlood_SWEKernelScenario scenario(
      dem,
      subdivisionFactor,
      minimalMeshWidth,
      maximalMeshWidth,
      globalTimestepSize,
      endTime
    );
    peanoclaw::Numerics* numerics = numericsFactory.createFullSWOF2DNumerics(scenario);
    std::cout << "domainSize " << scenario.getDomainSize() << std::endl;
#else
    tarch::la::Vector<DIMENSIONS, double> domainOffset(0);
    tarch::la::Vector<DIMENSIONS, double> domainSize(10.0);
//    tarch::la::Vector<DIMENSIONS, double> minimalMeshWidth(domainSize/6.0/9.0);
//    tarch::la::Vector<DIMENSIONS, double> maximalMeshWidth(domainSize/6.0/3.0);
    tarch::la::Vector<DIMENSIONS, int> finestSubgridTopology(9);
    tarch::la::Vector<DIMENSIONS, int> coarsestSubgridTopology(3);

    double globalTimestepSize = 0.1;
// double endTime = 2.0; //0.5; // 100.0;
    double endTime = 0.5; // 100.0;
//    int initialTimestepSize = 0.5;
    bool usePeanoClaw = true;
    tarch::la::Vector<DIMENSIONS, int> subdivisionFactor(6);
    if (argc > 1) {
      domainSize = tarch::la::Vector<DIMENSIONS, double>(1.0);

      peanoclaw::native::SWECommandLineParser commandLineParser(argc, argv);
      globalTimestepSize = commandLineParser.getGlobalTimestepSize();
      endTime = commandLineParser.getEndTime();
      finestSubgridTopology = commandLineParser.getFinestSubgridTopology();
      coarsestSubgridTopology = commandLineParser.getCoarsestSubgridTopology();
      subdivisionFactor = commandLineParser.getSubdivisionFactor();
      usePeanoClaw = commandLineParser.runSimulationWithPeanoClaw();
    }

    peanoclaw::native::BreakingDam_SWEKernelScenario scenario(
      domainOffset,
      domainSize,
      finestSubgridTopology,
      coarsestSubgridTopology,
      subdivisionFactor,
      globalTimestepSize,
      endTime
    );
    peanoclaw::Numerics* numerics = numericsFactory.createSWENumerics(scenario);

#endif

  if(usePeanoClaw) {
    runSimulation(
      scenario,
      *numerics,
      true
    );
  } else {
  #ifdef PEANOCLAW_SWE
  sweMain(scenario, finestSubgridTopology);
  #else
  #error Pure solver use not implemented
  #endif
  }

#endif
  return 0;
}
