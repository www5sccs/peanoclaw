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
#include "tarch/tests/TestCaseRegistry.h"

#include "peanoclaw/native/SWEKernel.h"

#if USE_VALGRIND
#include <callgrind.h>
#endif

#if defined(SWE)
class BreakingDam_SWEKernelScenario : public peanoclaw::native::SWEKernelScenario {
    public:
        BreakingDam_SWEKernelScenario() {}
        ~BreakingDam_SWEKernelScenario() {}

        virtual void initializePatch(peanoclaw::Patch& patch) {
            // dam coordinates
            double x0=0.5;
            double y0=0.5;
            
            // Riemann states of the dam break problem
            double radDam = 0.1;
            double hl = 2.;
            double ul = 0.;
            double vl = 0.;
            double hr = 1.;
            double ur = 0.;
            double vr = 0.;
            
            // compute from mesh data
            const tarch::la::Vector<DIMENSIONS, double> patchSize = patch.getSize();
            const tarch::la::Vector<DIMENSIONS, double> patchPosition = patch.getPosition();
            const tarch::la::Vector<DIMENSIONS, double> meshWidth = patch.getSubcellSize();

            tarch::la::Vector<DIMENSIONS, int> subcellIndex;
            for (int yi = 0; yi < patch.getSubdivisionFactor()(1); yi++) {
                for (int xi = 0; xi < patch.getSubdivisionFactor()(0); xi++) {
                    subcellIndex(0) = xi;
                    subcellIndex(1) = yi;
         
                    double X = patchPosition(0) + xi*meshWidth(0);
                    double Y = patchPosition(1) + yi*meshWidth(1);
         
                    double r = sqrt((X-x0)*(X-x0) + (Y-y0)*(Y-y0));
                    double q0 = hl*(r<=radDam) + hr*(r>radDam);
                    double q1 = hl*ul*(r<=radDam) + hr*ur*(r>radDam);
                    double q2 = hl*vl*(r<=radDam) + hr*vr*(r>radDam);
          
                    patch.setValueUNew(subcellIndex, 0, q0);
                    patch.setValueUNew(subcellIndex, 1, q1);
                    patch.setValueUNew(subcellIndex, 2, q2);
                }
            }
        }

        virtual double computeDemandedMeshWidth(peanoclaw::Patch& patch) {
            double max_gradient = 0.0;
            const tarch::la::Vector<DIMENSIONS, double> meshWidth = patch.getSubcellSize();
            
            tarch::la::Vector<DIMENSIONS, int> this_subcellIndex;
            tarch::la::Vector<DIMENSIONS, int> next_subcellIndex_x;
            tarch::la::Vector<DIMENSIONS, int> next_subcellIndex_y;
            for (int yi = 0; yi < patch.getSubdivisionFactor()(1)-1; yi++) {
                for (int xi = 0; xi < patch.getSubdivisionFactor()(0)-1; xi++) {
                    this_subcellIndex(0) = xi;
                    this_subcellIndex(1) = yi;
          
                    next_subcellIndex_x(0) = xi+1;
                    next_subcellIndex_x(1) = yi;
          
                    next_subcellIndex_y(0) = xi;
                    next_subcellIndex_y(1) = yi+1;
         
                    double q0 =  patch.getValueUNew(this_subcellIndex, 0);
                    double q0_x =  (patch.getValueUNew(next_subcellIndex_x, 0) - q0) / meshWidth(0);
                    double q0_y =  (patch.getValueUNew(next_subcellIndex_y, 0) - q0) / meshWidth(1);

                    max_gradient = fmax(max_gradient, q0_x);
                    max_gradient = fmax(max_gradient, q0_y);
                }
            }
          
            double demandedMeshWidth = 0;
            if (max_gradient > 0.05) {
                //demandedMeshWidth = 1.0/243;
                demandedMeshWidth = 1.0/81;
            } else {
                demandedMeshWidth = 1.0/81;
            }

            return demandedMeshWidth;
        }
};
#endif

static peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid* _configuration;

/*void importArrays() {
  import_array();
}*/

int main(int argc, char **argv) {
 /* double initialMinimalMeshWidthScalar,
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
) {*/
  peano::fillLookupTables();

#if defined(Parallel)
  int parallelSetup = peano::initParallelEnvironment(&argc,(char ***)&argv);
  int sharedMemorySetup = peano::initSharedMemoryEnvironment();
#endif

  //importArrays();

  //Initialize Logger
  static tarch::logging::Log _log("peanoclaw");
  logInfo("main(...)", "Initializing Peano");

  // Configure the output
  tarch::logging::CommandLineLogger::getInstance().clearFilterList();
  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "", true ) );
  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "info", true ) );
  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "debug", true ) );
//  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "trace", true ) );

  //Selective Tracing
//  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "debug", -1, "peanoclaw::mappings::Remesh", false ) );
//  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "debug", -1, "peanoclaw::mappings::Remesh::destroyVertex", false ) );
//  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "debug", -1, "peanoclaw::mappings::Remesh::endIteration", false ) );
//  tarch::logging::CommandLineLogger::getInstance().addFilterListEntry( ::tarch::logging::CommandLineLogger::FilterListEntry( "debug", -1, "peanoclaw::mappings::Remesh::touchVertex", false ) );

  //tarch::logging::CommandLineLogger::getInstance().setLogFormat( ... please consult source code documentation );

  std::ostringstream logFileName;
  #ifdef Parallel
  logFileName << "rank-" << tarch::parallel::Node::getInstance().getRank() << "-trace.txt";
  #endif
  tarch::logging::CommandLineLogger::getInstance().setLogFormat( " ", false, false, true, false, true, logFileName.str() );

  //Tests
  if(false) {
    tarch::tests::TestCaseRegistry::getInstance().getTestCaseCollection().run();
  }

  //PyClaw - this object is copied to the runner and is stored there.
  peanoclaw::NumericsFactory numericsFactory;

#if defined(SWE)
  BreakingDam_SWEKernelScenario scenario;
  peanoclaw::Numerics* numerics = numericsFactory.createSWENumerics(scenario);

  _configuration = new peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid;
  // assertion1(_configuration->isValid(), _configuration);

  //Construct parameters
  tarch::la::Vector<DIMENSIONS, double> domainOffset(0);
  tarch::la::Vector<DIMENSIONS, double> domainSize(1.0);
  tarch::la::Vector<DIMENSIONS, double> initialMinimalMeshWidth(0.05);
  tarch::la::Vector<DIMENSIONS, int> subdivisionFactor(9);
  int ghostlayerWidth = 1;
  int unknownsPerSubcell = 3;
  int auxiliarFieldsPerSubcell = 0;
  int initialTimestepSize = 0.1;
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
    auxiliarFieldsPerSubcell,
    initialTimestepSize,
    useDimensionalSplittingOptimization
  );

#if defined(Parallel) 
  std::cout << tarch::parallel::Node::getInstance().getRank() << ": peano instance created" << std::endl;
#endif

  assertion(runner != 0);
 
  // run experiment
  double timestep = 0.01;
  double endtime = 2.0;
#if defined(Parallel)
  if (tarch::parallel::Node::getInstance().isGlobalMaster()) {
#endif
      for (double time=0.0; time < endtime; time+=timestep) {
        runner->evolveToTime(time);
        //runner->gatherCurrentSolution();
        std::cout << "time " << time << " done " << std::endl;
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
