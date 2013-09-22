/*
 * PeanoClawLibraryRunner.cpp
 *
 *  Created on: Feb 7, 2012
 *      Author: Kristof Unterweger
 */

#include "peanoclaw/runners/PeanoClawLibraryRunner.h"

#include "peanoclaw/runners/Runner.h"
#include "peanoclaw/repositories/RepositoryFactory.h"
#include "peanoclaw/State.h"
#include "peanoclaw/configurations/PeanoClawConfigurationForSpacetreeGrid.h"
#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/records/VertexDescription.h"
#include "peanoclaw/records/Data.h"
#include "peanoclaw/records/Cell.h"
#include "peanoclaw/records/Vertex.h"
#include "peanoclaw/statistics/LevelStatistics.h"

#include "peano/heap/Heap.h"
#include "peano/utils/UserInterface.h"
#include "peano/peano.h"

#if defined(Parallel)
#include "tarch/parallel/NodePool.h"
#include "peano/parallel/SendReceiveBufferPool.h"
#include "peano/parallel/JoinDataBufferPool.h"
#include "peano/parallel/messages/ForkMessage.h"
#include "tarch/parallel/FCFSNodePoolStrategy.h"
#include "peano/parallel/loadbalancing/Oracle.h"
#include "peano/parallel/loadbalancing/OracleForOnePhaseWithGreedyPartitioning.h"
#endif

#ifdef SharedTBB
#include "tarch/multicore/tbb/Core.h"
#endif

#include "peano/datatraversal/autotuning/Oracle.h"
#include "peano/datatraversal/autotuning/OracleForOnePhaseDummy.h"

tarch::logging::Log peanoclaw::runners::PeanoClawLibraryRunner::_log("peanoclaw::runners::PeanoClawLibraryRunner");

void peanoclaw::runners::PeanoClawLibraryRunner::initializePeano(
  tarch::la::Vector<DIMENSIONS, double> domainOffset,
  tarch::la::Vector<DIMENSIONS, double> domainSize
) {
  //Initialize heap data
  peano::heap::Heap<peanoclaw::records::CellDescription>::getInstance().setName("CellDescription");
  peano::heap::Heap<peanoclaw::records::Data>::getInstance().setName("Data");
  peano::heap::Heap<peanoclaw::records::VertexDescription>::getInstance().setName("VertexDescription");
  peano::heap::Heap<peanoclaw::statistics::LevelStatistics>::getInstance().setName("LevelStatistics");
}

void peanoclaw::runners::PeanoClawLibraryRunner::initializeParallelEnvironment() {
  //Distributed Memory
  #if defined(Parallel)
  tarch::parallel::Node::getInstance().setTimeOutWarning(45);
  tarch::parallel::Node::getInstance().setDeadlockTimeOut(90);

  if (tarch::parallel::Node::getInstance().isGlobalMaster()) {
    tarch::parallel::NodePool::getInstance().setStrategy( new tarch::parallel::FCFSNodePoolStrategy() );
  }
  tarch::parallel::NodePool::getInstance().restart();

  peano::parallel::loadbalancing::Oracle::getInstance().setOracle(
    new peano::parallel::loadbalancing::OracleForOnePhaseWithGreedyPartitioning(true)
  );

  // have to be the same for all ranks
  peano::parallel::SendReceiveBufferPool::getInstance().setBufferSize(64);
  peano::parallel::JoinDataBufferPool::getInstance().setBufferSize(64);
  #endif

  //Shared Memory
  #ifdef SharedMemoryParallelisation
  tarch::multicore::tbb::Core::getInstance().configure(1);
  #endif
  peano::datatraversal::autotuning::Oracle::getInstance().setOracle( new peano::datatraversal::autotuning::OracleForOnePhaseDummy(true) );
}

void peanoclaw::runners::PeanoClawLibraryRunner::iterateInitialiseGrid() {
  if(_validateGrid) {
    _repository->switchToInitialiseAndValidateGrid();
  } else {
    _repository->switchToInitialiseGrid();
  }
  _repository->iterate();
}

void peanoclaw::runners::PeanoClawLibraryRunner::iteratePlot() {
  if(_validateGrid) {
    _repository->switchToPlotAndValidateGrid();
  } else {
    _repository->switchToPlot();
  }
  _repository->iterate();
}

void peanoclaw::runners::PeanoClawLibraryRunner::iterateSolveTimestep(bool plotSubsteps) {
  if(plotSubsteps) {
    _repository->getState().setPlotNumber(_plotNumber);
    if(_validateGrid) {
      _repository->switchToSolveTimestepAndPlotAndValidateGrid();
    } else {
      _repository->switchToSolveTimestepAndPlot();
    }
    _repository->iterate();
    _plotNumber++;
  } else {
    if(_validateGrid) {
      _repository->switchToSolveTimestepAndValidateGrid();
    } else {
      _repository->switchToSolveTimestep();
    }
    _repository->iterate();
  }
}

void peanoclaw::runners::PeanoClawLibraryRunner::iterateGatherSolution() {
  if(_validateGrid) {
    _repository->switchToGatherCurrentSolutionAndValidateGrid(); _repository->iterate();
  } else {
    _repository->switchToGatherCurrentSolution(); _repository->iterate();
  }
}

peanoclaw::runners::PeanoClawLibraryRunner::PeanoClawLibraryRunner(
  peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid& configuration,
  peanoclaw::Numerics& numerics,
  const tarch::la::Vector<DIMENSIONS, double>& domainOffset,
  const tarch::la::Vector<DIMENSIONS, double>& domainSize,
  const tarch::la::Vector<DIMENSIONS, double>& initialMinimalMeshWidth,
  const tarch::la::Vector<DIMENSIONS, int>& subdivisionFactor,
  int defaultGhostLayerWidth,
  int unknownsPerSubcell,
  int auxiliarFieldsPerSubcell,
  double initialTimestepSize,
  bool useDimensionalSplittingOptimization
) :
  _plotNumber(1),
  _configuration(configuration),
  _iterationTimer("peanoclaw::runners::PeanoClawLibraryRunner", "iteration", false),
  _totalRuntime(0.0),
  _numerics(numerics),
  _validateGrid(false)
{
  #ifndef Asserts
  _validateGrid = false;
  #endif

  //User interface
  peano::utils::UserInterface userInterface;
  userInterface.writeHeader();

  initializePeano(domainOffset, domainSize);
  initializeParallelEnvironment();

  //Initialize pseudo geometry (Has to be done after initializeParallelEnvironment()
  _geometry =
    new  peano::geometry::Hexahedron (
      domainSize,  // width
      domainOffset // offset
    );
  _repository =
    peanoclaw::repositories::RepositoryFactory::getInstance().createWithSTDStackImplementation(
      *_geometry,
      domainSize,   // domainSize,
      domainOffset  // computationalDomainOffset
    );
  assertion(_repository != 0);

  logInfo("PeanoClawLibraryRunner", "Initial values: "
      << "Domain size = [" << domainSize << "]"
      << ", default subdivision factor = " << subdivisionFactor
      << ", default ghostlayer width = " << defaultGhostLayerWidth
      << ", unknowns per cell = " << unknownsPerSubcell);

  State& state = _repository->getState();
  state.setDefaultSubdivisionFactor(subdivisionFactor);
  state.setDefaultGhostLayerWidth(defaultGhostLayerWidth);
  state.setUnknownsPerSubcell(unknownsPerSubcell);
  state.setAuxiliarFieldsPerSubcell(auxiliarFieldsPerSubcell);
  tarch::la::Vector<DIMENSIONS, double> initialMinimalSubcellSize = tarch::la::multiplyComponents(initialMinimalMeshWidth, subdivisionFactor.convertScalar<double>());
  state.setInitialMinimalMeshWidth(initialMinimalSubcellSize);
  state.setNumerics(numerics);
  state.resetTotalNumberOfCellUpdates();
  state.setInitialTimestepSize(initialTimestepSize);
  state.setDomain(domainOffset, domainSize);
  state.setUseDimensionalSplittingOptimization(useDimensionalSplittingOptimization && !_configuration.disableDimensionalSplittingOptimization());

  //Initialise Grid (two iterations needed to set the initial ghostlayers of patches neighboring refined patches)
  state.setIsInitializing(true);

  #ifdef Parallel
  if (tarch::parallel::Node::getInstance().isGlobalMaster()) {
   tarch::parallel::NodePool::getInstance().waitForAllNodesToBecomeIdle();
  #endif

   state.enableRefinementCriterion(false);
   tarch::la::Vector<DIMENSIONS, double> currentMinimalSubcellSize;
   int maximumLevel = 2;
   do {

     logInfo("", "Iterating with maximumLevel=" << maximumLevel);

     for(int d = 0; d < DIMENSIONS; d++) {
       currentMinimalSubcellSize(d) = std::max(initialMinimalMeshWidth(d), domainSize(d) / pow(3.0, maximumLevel) / subdivisionFactor(d));
     }

     state.setIsInitializing(true);
     do {
       logInfo("", "Iterate");
       iterateInitialiseGrid();
       iterateInitialiseGrid();

       logInfo("", "stationary: " << _repository->getState().isGridStationary() << ", balanced: " << _repository->getState().isGridBalanced());
     } while(!_repository->getState().isGridStationary() || !_repository->getState().isGridBalanced());

     maximumLevel += 2;
   } while(tarch::la::oneGreater(currentMinimalSubcellSize, initialMinimalMeshWidth));

   state.enableRefinementCriterion(true);
   do {
     logInfo("", "Iterate with Refinement Criterion");
     iterateInitialiseGrid();
     iterateInitialiseGrid();
     iterateInitialiseGrid();
     iterateInitialiseGrid();
   } while(!_repository->getState().isGridStationary() || !_repository->getState().isGridBalanced());

   _repository->getState().setPlotNumber(0);
   if(_configuration.plotAtOutputTimes() || _configuration.plotSubsteps()) {
     iteratePlot();
   }

//  tarch::la::Vector<DIMENSIONS, double> current_initialMinimalSubcellSize(0.1);
//  tarch::la::Vector<DIMENSIONS, double> next_initialMinimalSubcellSize(0.1);
//
//  if (tarch::la::oneGreater(next_initialMinimalSubcellSize,initialMinimalSubcellSize)) {
//        current_initialMinimalSubcellSize = next_initialMinimalSubcellSize;
//    } else {
//        current_initialMinimalSubcellSize = initialMinimalSubcellSize;
//    }

//#ifdef Parallel
//  if (tarch::parallel::Node::getInstance().isGlobalMaster()) {
//
//    state.setIsInitializing(true);
//    while (tarch::la::oneGreater(current_initialMinimalSubcellSize,initialMinimalSubcellSize) || initialMinimalSubcellSize == current_initialMinimalSubcellSize) {
//        state.setInitialMinimalMeshWidth(current_initialMinimalSubcellSize);
//
//        bool hasChangedState = false;
//        iterateInitialiseGrid();
//        hasChangedState |= (!_repository->getState().isGridStationary() || !_repository->getState().isGridBalanced());
//
//        _repository->switchToRemesh();
//
//        do {
//            // TODO: not sure if we need all 3 iterations of them or if it is sufficient to check for a stationary grid
//
//            hasChangedState = false;
//
//            _repository->iterate();
//            hasChangedState |= (!_repository->getState().isGridStationary() || !_repository->getState().isGridBalanced());
//
//        } while (hasChangedState);
//
//        next_initialMinimalSubcellSize = current_initialMinimalSubcellSize * (1.0/subdivisionFactor[0]);
//        std::cout << "next initial minimal subcell size " << next_initialMinimalSubcellSize
//                    << " " << current_initialMinimalSubcellSize
//                    << " " << initialMinimalSubcellSize
//                    << std::endl;
//
//        if (tarch::la::oneGreater(next_initialMinimalSubcellSize,initialMinimalSubcellSize)) {
//            current_initialMinimalSubcellSize = next_initialMinimalSubcellSize;
//        } else if (initialMinimalSubcellSize == current_initialMinimalSubcellSize) {
//            current_initialMinimalSubcellSize *= (1.0/3.0); // abort loop
//        } else {
//            current_initialMinimalSubcellSize = initialMinimalSubcellSize;
//        }
//    }
//
//      do {
//        state.setInitialRefinementTriggered(false);
//        iterateInitialiseGrid();
//        std::cout << "second iteration block (1)" << _repository->getState() << std::endl;
//      } while(state.getInitialRefinementTriggered());
//      state.setIsInitializing(false);
//
//      _repository->getState().setPlotNumber(0);
//      if(_configuration.plotAtOutputTimes() || _configuration.plotSubsteps()) {
//        iteratePlot();
//      }
//  }
//
//  }
//#endif

  #ifdef Parallel
  }
  #endif
}

peanoclaw::runners::PeanoClawLibraryRunner::~PeanoClawLibraryRunner()
{
  logTraceIn("~PeanoClawLibraryRunner");
//  logInfo("~PeanoClawLibraryRunner()", "Total number of cell updates: " << _repository->getState().getTotalNumberOfCellUpdates());

  //Delete remaining heap data
  peano::heap::Heap<peanoclaw::records::CellDescription>& cellDescriptionHeap = peano::heap::Heap<peanoclaw::records::CellDescription>::getInstance();
//  if(cellDescriptionHeap.getNumberOfAllocatedEntries() > 0) {
//    logWarning("~PeanoClawLibraryRunner()", "The heap for CellDescriptions still contains " << cellDescriptionHeap.getNumberOfAllocatedEntries() << " undeleted entries.");
//  }
  peano::heap::Heap<peanoclaw::records::Data>& dataHeap = peano::heap::Heap<peanoclaw::records::Data>::getInstance();
//  if(dataHeap.getNumberOfAllocatedEntries() > 0) {
//    logWarning("~PeanoClawLibraryRunner()", "The heap for patch data still contains " << dataHeap.getNumberOfAllocatedEntries() << " undeleted entries.");
//  }
  peano::heap::Heap<peanoclaw::records::VertexDescription>& vertexDescriptionHeap = peano::heap::Heap<peanoclaw::records::VertexDescription>::getInstance();
//  if(vertexDescriptionHeap.getNumberOfAllocatedEntries() > 0) {
//    logWarning("~PeanoClawLibraryRunner()", "The heap for VertexDescriptions still contains " << vertexDescriptionHeap.getNumberOfAllocatedEntries() << " undeleted entries.");
//  }
  cellDescriptionHeap.deleteAllData();
  dataHeap.deleteAllData();
  vertexDescriptionHeap.deleteAllData();

  peano::heap::Heap<peanoclaw::records::CellDescription>::getInstance().plotStatistics();
  peano::heap::Heap<peanoclaw::records::Data>::getInstance().plotStatistics();
  peano::heap::Heap<peanoclaw::records::VertexDescription>::getInstance().plotStatistics();

  #ifdef Parallel
  if(tarch::parallel::Node::getInstance().isGlobalMaster()) {
  #endif
    _repository->getState().plotTotalStatistics();

    _repository->logIterationStatistics();
    _repository->terminate();
  #ifdef Parallel
  }
  #endif
  delete _repository;
  delete _geometry;

  #ifdef Parallel
  tarch::parallel::NodePool::getInstance().terminate();
  #endif
  peano::shutdownParallelEnvironment();
  peano::shutdownSharedMemoryEnvironment();

  logTraceOut("~PeanoClawLibraryRunner");
}

void peanoclaw::runners::PeanoClawLibraryRunner::evolveToTime(
  double time
) {
  logTraceIn("evolveToTime");
  
  bool plotSubsteps = _configuration.plotSubsteps()
      || (_configuration.plotSubstepsAfterOutputTime() != -1 && _configuration.plotSubstepsAfterOutputTime() <= _plotNumber);
 
  configureGlobalTimestep(time);
  do {
      runNextPossibleTimestep();
  } while(!_repository->getState().getAllPatchesEvolvedToGlobalTimestep());

  if(_configuration.plotAtOutputTimes() && !plotSubsteps) {
    _repository->getState().setPlotNumber(_plotNumber);
    iteratePlot();
    _plotNumber++;
  } else if (!_configuration.plotAtOutputTimes() && !plotSubsteps) {
    _plotNumber++;
  }

  peano::heap::Heap<peanoclaw::records::CellDescription>::getInstance().plotStatistics();
  peano::heap::Heap<peanoclaw::records::Data>::getInstance().plotStatistics();
  peano::heap::Heap<peanoclaw::records::VertexDescription>::getInstance().plotStatistics();

  logTraceOut("evolveToTime");
}

void peanoclaw::runners::PeanoClawLibraryRunner::gatherCurrentSolution() {
  logTraceIn("gatherCurrentSolution");
  assertion(_repository != 0);

  iterateGatherSolution();

  logTraceOut("gatherCurrentSolution");
}

int peanoclaw::runners::PeanoClawLibraryRunner::runWorker() {
  #ifdef Parallel
  int newMasterNode = tarch::parallel::NodePool::getInstance().waitForJob();
  while ( newMasterNode != tarch::parallel::NodePool::JobRequestMessageAnswerValues::Terminate ) {
    if ( newMasterNode >= tarch::parallel::NodePool::JobRequestMessageAnswerValues::NewMaster ) {
      peano::parallel::messages::ForkMessage forkMessage;
      forkMessage.receive(tarch::parallel::NodePool::getInstance().getMasterRank(),tarch::parallel::NodePool::getInstance().getTagForForkMessages(), true);
      _repository->restart(
        forkMessage.getH(),
        forkMessage.getDomainOffset(),
        forkMessage.getLevel(),
        forkMessage.getPositionOfFineGridCellRelativeToCoarseGridCell()
      );

      bool continueToIterate = true;
      while (continueToIterate) {
        switch (_repository->continueToIterate()) {
          case peanoclaw::repositories::Repository::Continue:
            _repository->iterate();
            break;
          case peanoclaw::repositories::Repository::Terminate:
            continueToIterate = false;
            break;
          case peanoclaw::repositories::Repository::RunGlobalStep:
            //runGlobalStep();
            break;
        }
      }

      _repository->terminate();
    }
    else if ( newMasterNode == tarch::parallel::NodePool::JobRequestMessageAnswerValues::RunAllNodes ) {
      //runGlobalStep();
    }
    newMasterNode = tarch::parallel::NodePool::getInstance().waitForJob();
  }

  #endif
  return 0;
}

void peanoclaw::runners::PeanoClawLibraryRunner::configureGlobalTimestep(double time) {
  _repository->getState().setGlobalTimestepEndTime(time);
  _repository->getState().setNumerics(_numerics);
  _repository->getState().setPlotNumber(_plotNumber);
}

void peanoclaw::runners::PeanoClawLibraryRunner::runNextPossibleTimestep() {
    bool plotSubsteps = _configuration.plotSubsteps()
      || (_configuration.plotSubstepsAfterOutputTime() != -1 && _configuration.plotSubstepsAfterOutputTime() <= _plotNumber);

    logInfo("evolveToTime", "Solving timestep " << (_plotNumber-1) << " with maximum global time interval ("
        << _repository->getState().getStartMaximumGlobalTimeInterval() << ", " << _repository->getState().getEndMaximumGlobalTimeInterval() << ")"
        << " and minimum global time interval (" << _repository->getState().getStartMinimumGlobalTimeInterval() << ", " << _repository->getState().getEndMinimumGlobalTimeInterval() << ")");
    _iterationTimer.startTimer();

    _repository->getState().resetGlobalTimeIntervals();
    _repository->getState().resetMinimalTimestep();
    _repository->getState().setAllPatchesEvolvedToGlobalTimestep(true);

    iterateSolveTimestep(plotSubsteps);

    _repository->getState().plotStatisticsForLastGridIteration();

    _iterationTimer.stopTimer();
    _totalRuntime += (double)_iterationTimer.getCPUTicks() / (double)CLOCKS_PER_SEC;
    logInfo("evolveToTime", "Wallclock time for this grid iteration/Total runtime: " << _iterationTimer.getCalendarTime() << "s/" << _totalRuntime << "s");
    logInfo("evolveToTime", "Minimal timestep for this grid iteration: " << _repository->getState().getMinimalTimestep());

    assertion(_repository->getState().getMinimalTimestep() < std::numeric_limits<double>::infinity());
}
