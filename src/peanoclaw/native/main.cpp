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
#include "peanoclaw/runners/PeanoClawLibraryRunner.h"
#include "tarch/logging/LogFilterFileReader.h"
#include "tarch/logging/Log.h"
#include "tarch/tests/TestCaseRegistry.h"

#include "peanoclaw/native/SWEKernel.h"

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
  //tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "debug", true ) );
//  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "trace", true ) );

  //Validation
  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "info", -1, "peanoclaw::statistics::ParallelGridValidator", true ) );
  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "info", -1, "peanoclaw::mappings::ValidateGrid", true ) );

  //Selective Tracing
  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "info", -1, "tarch::mpianalysis", true ) );
  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "info", -1, "peano::", true ) );
//  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "debug", -1, "peanoclaw::mappings::Remesh", false ) );
//  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "debug", -1, "peanoclaw::mappings::Remesh::destroyVertex", false ) );
//  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "debug", -1, "peanoclaw::mappings::Remesh::endIteration", false ) );
//  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "debug", -1, "peanoclaw::mappings::Remesh::touchVertex", false ) );

  //tarch::logging::CommandLineLogger::getInstance().setLogFormat( ... please consult source code documentation );
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

  //PyClaw - this object is copied to the runner and is stored there.
  peanoclaw::NumericsFactory numericsFactory;

#if defined(SWE) || defined(PEANOCLAW_FULLSWOF2D)
  _configuration = new peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid;

  //Construct parameters
#if defined(PEANOCLAW_FULLSWOF2D)
    DEM dem;

    dem.load("DEM_400cm.bin");

    MekkaFlood_SWEKernelScenario scenario(dem);
    peanoclaw::Numerics* numerics = numericsFactory.createFullSWOF2DNumerics(scenario);
 
    tarch::la::Vector<DIMENSIONS, double> domainOffset;

    int parametersWithoutGhostlayerPerSubcell = 1;
    int parametersWithGhostlayerPerSubcell = 1;
    int ghostlayerWidth = 2;
    int unknownsPerSubcell = 6;

    // TODO: aaarg Y U NO PLOT CORRECTLY! -> work around established
    //domainOffset(0) = dem.lower_left(0);
    //domainOffset(1) = dem.lower_left(1);
    domainOffset(0) = 0.0;
    domainOffset(1) = 0.0;

    tarch::la::Vector<DIMENSIONS, double> domainSize;
    double upper_right_0 = dem.upper_right(0);
    double upper_right_1 = dem.upper_right(1);
 
    double lower_left_0 = dem.lower_left(0);
    double lower_left_1 = dem.lower_left(1);
 
    double x_size = (upper_right_0 - lower_left_0)/scenario.scale;
    double y_size = (upper_right_1 - lower_left_1)/scenario.scale;

    double timestep = 2.0; //0.1;//1.0;
    double endtime = 3600.0; // 100.0;
 
    // TODO: make central scale parameter in MekkaFlood class
    // currently we have to change here, meshToCoordinates and initializePatch and computeMeshWidth
    domainSize(0) = x_size;
    domainSize(1) = y_size;
 
    std::cout << "domainSize " << domainSize(0) << " " << domainSize(1) << std::endl;

    // keep aspect ratio of map: 4000 3000: ratio 4:3
    tarch::la::Vector<DIMENSIONS, int> subdivisionFactor;
    subdivisionFactor(0) = static_cast<int>(16); // 96 //  6 * 4, optimum in non optimized version
    subdivisionFactor(1) = static_cast<int>(9);  // 54 //  6 * 3, optimum in non optimized version

    double min_domainSize = std::min(domainSize(0),domainSize(1));
    double max_domainSize = std::max(domainSize(0),domainSize(1));

    int min_subdivisionFactor = std::min(subdivisionFactor(0),subdivisionFactor(1));
    int max_subdivisionFactor = std::max(subdivisionFactor(0),subdivisionFactor(1));

    tarch::la::Vector<DIMENSIONS, double> initialMinimalMeshWidth(min_domainSize/(3.0 * max_subdivisionFactor));

    double initialTimestepSize = 1.0;


#else
    BreakingDam_SWEKernelScenario scenario;
    peanoclaw::Numerics* numerics = numericsFactory.createSWENumerics(scenario);

    int parametersWithoutGhostlayerPerSubcell = 1;
    int parametersWithGhostlayerPerSubcell = 0;
    int ghostlayerWidth = 1;
    int unknownsPerSubcell = 3;
 
    tarch::la::Vector<DIMENSIONS, double> domainOffset(0);
    tarch::la::Vector<DIMENSIONS, double> domainSize(10.0);
    tarch::la::Vector<DIMENSIONS, double> initialMinimalMeshWidth(domainSize/6.0/9.0);

    double timestep = 0.1;
//    double endtime = 2.0; //0.5; // 100.0;
    double endtime = 0.5; // 100.0;
    int initialTimestepSize = 0.5;
 
    tarch::la::Vector<DIMENSIONS, int> subdivisionFactor(6);
#endif
  
  bool useDimensionalSplittingOptimization = true;

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
    *numerics,
    domainOffset,
    domainSize,
    initialMinimalMeshWidth,
    subdivisionFactor,
    ghostlayerWidth,
    unknownsPerSubcell,
    parametersWithoutGhostlayerPerSubcell,
    parametersWithGhostlayerPerSubcell,
    initialTimestepSize,
    useDimensionalSplittingOptimization,
    1
  );

#if defined(Parallel) 
  std::cout << tarch::parallel::Node::getInstance().getRank() << ": peano instance created" << std::endl;
#endif

  assertion(runner != 0);
 
  // run experiment
#if defined(Parallel)
  if (tarch::parallel::Node::getInstance().isGlobalMaster()) {
#endif
      for (double time=timestep; time <= endtime; time+=timestep) {
          peanoclaw::State& state = runner->getState();
        runner->evolveToTime(time);
        //runner->gatherCurrentSolution();
        std::cout << "time " << time << " numberOfCells " << state.getNumberOfInnerCells() << std::endl;
      }
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
#endif
  return 0;
}
