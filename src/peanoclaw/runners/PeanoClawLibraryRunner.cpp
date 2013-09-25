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

#include "peanoclaw/Heap.h"
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
  CellDescriptionHeap::getInstance().setName("CellDescription");
  DataHeap::getInstance().setName("Data");
  LevelStatisticsHeap::getInstance().setName("LevelStatistics");
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
  _validateGrid(true)
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

    state.enableRefinementCriterion(false);
    tarch::la::Vector<DIMENSIONS, double> currentMinimalSubcellSize;
    int maximumLevel = 2;
    do {

      logDebug("PeanoClawLibraryRunner", "Iterating with maximumLevel=" << maximumLevel);

      for(int d = 0; d < DIMENSIONS; d++) {
        currentMinimalSubcellSize(d) = std::max(initialMinimalMeshWidth(d), domainSize(d) / pow(3.0, maximumLevel) / subdivisionFactor(d));
      }
      _repository->getState().setInitialMinimalMeshWidth(currentMinimalSubcellSize);

      do {
        iterateInitialiseGrid();
        iterateInitialiseGrid();

        logDebug("PeanoClawLibraryRunner", "stationary: " << _repository->getState().isGridStationary() << ", balanced: " << _repository->getState().isGridBalanced());
      } while(!_repository->getState().isGridStationary() || !_repository->getState().isGridBalanced());

      maximumLevel += 2;
    } while(tarch::la::oneGreater(currentMinimalSubcellSize, initialMinimalMeshWidth));
    #endif

    state.enableRefinementCriterion(true);
    do {
      logDebug("PeanoClawLibraryRunner", "Iterate with Refinement Criterion");
      iterateInitialiseGrid();
      iterateInitialiseGrid();
    } while(!_repository->getState().isGridStationary() || !_repository->getState().isGridBalanced());

    //Plot initial grid
    _repository->getState().setPlotNumber(0);
    if(_configuration.plotAtOutputTimes() || _configuration.plotSubsteps()) {
      iteratePlot();
    }

    state.setIsInitializing(false);
  #ifdef Parallel
  }
  #endif
}

peanoclaw::runners::PeanoClawLibraryRunner::~PeanoClawLibraryRunner()
{
  logTraceIn("~PeanoClawLibraryRunner");
//  logInfo("~PeanoClawLibraryRunner()", "Total number of cell updates: " << _repository->getState().getTotalNumberOfCellUpdates());

  //Delete remaining heap data
//  if(cellDescriptionHeap.getNumberOfAllocatedEntries() > 0) {
//    logWarning("~PeanoClawLibraryRunner()", "The heap for CellDescriptions still contains " << cellDescriptionHeap.getNumberOfAllocatedEntries() << " undeleted entries.");
//  }
//  if(dataHeap.getNumberOfAllocatedEntries() > 0) {
//    logWarning("~PeanoClawLibraryRunner()", "The heap for patch data still contains " << dataHeap.getNumberOfAllocatedEntries() << " undeleted entries.");
//  }
  CellDescriptionHeap::getInstance().deleteAllData();
  DataHeap::getInstance().deleteAllData();

  CellDescriptionHeap::getInstance().plotStatistics();
  DataHeap::getInstance().plotStatistics();

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

  CellDescriptionHeap::getInstance().plotStatistics();
  DataHeap::getInstance().plotStatistics();

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
