/*
 * PeanoClawLibraryRunner.cpp
 *
 *  Created on: Feb 7, 2012
 *      Author: Kristof Unterweger
 */

#include "peanoclaw/runners/PeanoClawLibraryRunner.h"

#include "peanoclaw/repositories/RepositoryFactory.h"
#include "peanoclaw/State.h"
#include "peanoclaw/configurations/PeanoClawConfigurationForSpacetreeGrid.h"
#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/records/VertexDescription.h"
#include "peanoclaw/records/Data.h"
#include "peano/heap/Heap.h"

#include "peano/utils/UserInterface.h"

#if defined(Parallel)
#include "peano/parallel/messages/ForkMessage.h"
#endif

#include "tarch/parallel/FCFSNodePoolStrategy.h"
#include "peano/parallel/loadbalancing/Oracle.h"
#include "peano/parallel/loadbalancing/OracleForOnePhaseWithGreedyPartitioning.h"
#include "peano/parallel/SendReceiveBufferPool.h"
#include "peano/parallel/JoinDataBufferPool.h"


#include "peano/datatraversal/autotuning/Oracle.h"
#include "peano/datatraversal/autotuning/OracleForOnePhaseDummy.h"

#include <ctime>

// roland MARK
// TODO: implement/port RunnerParallerWorker part

tarch::logging::Log peanoclaw::runners::PeanoClawLibraryRunner::_log("peanoclaw::runners::PeanoClawLibraryRunner");

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
  bool useDimensionalSplitting
) :
  _plotNumber(1),
  _configuration(configuration),
  _iterationTimer("peanoclaw::runners::PeanoClawLibraryRunner", "iteration", false),
  _totalRuntime(0.0) {

  peano::utils::UserInterface userInterface;
  userInterface.writeHeader();

  //Initialize heap data
  peano::heap::Heap<peanoclaw::records::CellDescription>::getInstance().setName("CellDescription");
  peano::heap::Heap<peanoclaw::records::Data>::getInstance().setName("Data");
  peano::heap::Heap<peanoclaw::records::VertexDescription>::getInstance().setName("VertexDescription");

#if defined(Parallel)
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

  peano::datatraversal::autotuning::Oracle::getInstance().setOracle( new peano::datatraversal::autotuning::OracleForOnePhaseDummy(true) );

  //Initialize pseudo geometry
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

//  UserInterface userInterface;
//  userInterface.writeHeader();
 

  logInfo("PeanoClawLibraryRunner", "Initial values: "
      << "Domain size = [" << domainSize << "]"
      << ", default subdivision factor = " << subdivisionFactor
      << ", default ghostlayer width = " << defaultGhostLayerWidth
      << ", unknowns per cell = " << unknownsPerSubcell);

  assertion(_repository != 0);
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
  state.setAdditionalLevelsForPredefinedRefinement(_configuration.getAdditionalLevelsForPredefinedRefinement());
  state.setUseDimensionalSplitting(useDimensionalSplitting && !_configuration.disableDimensionalSplittingOptimization());

  //Initialise Grid (two iterations needed to set the initial ghostlayers of patches neighboring refined patches)
  state.setIsInitializing(true);
  _repository->switchToInitialiseGrid(); _repository->iterate();
  do {
    state.setInitialRefinementTriggered(false);
    _repository->iterate();
  } while(state.getInitialRefinementTriggered());
  state.setIsInitializing(false);

  _repository->getState().setPlotNumber(0);
  if(_configuration.plotAtOutputTimes() || _configuration.plotSubsteps()) {
    _repository->switchToPlot(); _repository->iterate();
  }
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

  _repository->getState().plotTotalStatistics();

  _repository->logIterationStatistics();
  _repository->terminate();
  delete _repository;
  delete _geometry;
  logTraceOut("~PeanoClawLibraryRunner");
}

void peanoclaw::runners::PeanoClawLibraryRunner::evolveToTime(
  double time,
  peanoclaw::Numerics& numerics
) {
  logTraceIn("evolveToTime");

#if defined(Parallel)
  if (!tarch::parallel::Node::getInstance().isGlobalMaster()) {
    if (!_repository->continueToIterate()) {
        _repository->terminate();

        if ( tarch::parallel::NodePool::getInstance().waitForJob() >= tarch::parallel::NodePool::JobRequestMessageAnswerValues::NewMaster ) {
            peano::parallel::messages::ForkMessage forkMessage;
            forkMessage.receive(tarch::parallel::NodePool::getInstance().getMasterRank(),tarch::parallel::NodePool::getInstance().getTagForForkMessages(), true);

             _repository->restart(
              forkMessage.getH(),
              forkMessage.getDomainOffset(),
              forkMessage.getLevel()
           );
        }
    }
  }
#endif
  
  bool plotSubsteps = _configuration.plotSubsteps()
      || (_configuration.plotSubstepsAfterOutputTime() != -1 && _configuration.plotSubstepsAfterOutputTime() <= _plotNumber);

  _repository->getState().setGlobalTimestepEndTime(time);
  _repository->getState().setNumerics(numerics);
  _repository->getState().setPlotNumber(_plotNumber);
  do {
    logInfo("evolveToTime", "Solving timestep " << (_plotNumber-1) << " with maximum global time interval ("
        << _repository->getState().getStartMaximumGlobalTimeInterval() << ", " << _repository->getState().getEndMaximumGlobalTimeInterval() << ")"
        << " and minimum global time interval (" << _repository->getState().getStartMinimumGlobalTimeInterval() << ", " << _repository->getState().getEndMinimumGlobalTimeInterval() << ")");
    _iterationTimer.startTimer();

    _repository->getState().resetGlobalTimeIntervals();
    _repository->getState().resetMinimalTimestep();
    _repository->getState().setAllPatchesEvolvedToGlobalTimestep(true);

    if(plotSubsteps) {
      _repository->getState().setPlotNumber(_plotNumber);
      _repository->switchToSolveTimestepAndPlot(); _repository->iterate();
      _plotNumber++;
    } else {
      _repository->switchToSolveTimestep(); _repository->iterate();
    }

    _repository->getState().plotStatisticsForLastGridIteration();

    _iterationTimer.stopTimer();
    _totalRuntime += _iterationTimer.getCPUTicks() / CLOCKS_PER_SEC;
    logInfo("evolveToTime", "Wallclock time for this grid iteration/Total runtime: " << _iterationTimer.getCalendarTime() << "s/" << _totalRuntime << "s");
    logInfo("evolveToTime", "Minimal timestep for this grid iteration: " << _repository->getState().getMinimalTimestep());
  } while(!_repository->getState().getAllPatchesEvolvedToGlobalTimestep());

  if(_configuration.plotAtOutputTimes() && !plotSubsteps) {
    _repository->getState().setPlotNumber(_plotNumber);
    _repository->switchToPlot(); _repository->iterate();
    _plotNumber++;
  } else if (!_configuration.plotAtOutputTimes() && !plotSubsteps) {
    _plotNumber++;
  }
  logTraceOut("evolveToTime");
}

void peanoclaw::runners::PeanoClawLibraryRunner::gatherCurrentSolution(
  peanoclaw::Numerics& numerics
) {
  logTraceIn("gatherCurrentSolution");
  assertion(_repository != 0);
  _repository->getState().setNumerics(numerics);

  _repository->switchToGatherCurrentSolution();
  _repository->iterate();
  logTraceOut("gatherCurrentSolution");
}
