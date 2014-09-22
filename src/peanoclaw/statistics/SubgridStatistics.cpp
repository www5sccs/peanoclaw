/*
 * SubgridStatistics.cpp
 *
 *  Created on: Jul 29, 2013
 *      Author: kristof
 */
#include "peanoclaw/statistics/SubgridStatistics.h"

#include "peanoclaw/Vertex.h"

#include "peano/grid/aspects/VertexStateAnalysis.h"
#include "peanoclaw/Heap.h"
#include "peanoclaw/State.h"

#include "tarch/parallel/Node.h"

#include <algorithm>
#include <limits>
#include <iomanip>

#ifdef Parallel
#include "mpi.h"
#endif

tarch::logging::Log peanoclaw::statistics::SubgridStatistics::_log("peanoclaw::statistics::SubgridStatistics");
tarch::multicore::BooleanSemaphore peanoclaw::statistics::SubgridStatistics::_heapSemaphore;


bool peanoclaw::statistics::smaller(
  const ProcessStatisticsEntry& entry1,
  const ProcessStatisticsEntry& entry2
) {
  return entry1.getRank() < entry2.getRank();
}

void peanoclaw::statistics::SubgridStatistics::initializeLevelAndProcessStatistics() {
  tarch::multicore::Lock lock(_heapSemaphore);
  _levelStatisticsIndex = LevelStatisticsHeap::getInstance().createData();

  _processStatisticsIndex = ProcessStatisticsHeap::getInstance().createData();
  std::vector<ProcessStatisticsEntry>& processStatistics = ProcessStatisticsHeap::getInstance().getData(_processStatisticsIndex);

  ProcessStatisticsEntry processStatisticsEntry;
  #ifdef Parallel
  processStatisticsEntry.setRank(tarch::parallel::Node::getInstance().getRank());
  #else
  processStatisticsEntry.setRank(0);
  #endif
  processStatisticsEntry.setProcessorHashCode(computeProcessorHashCode());
  processStatisticsEntry.setNumberOfCellUpdates(0);
  processStatistics.push_back(processStatisticsEntry);

  //TODO unterweg debug
//  std::cout << "Initialized subgrid statistics levelStatisticsIndex=" << _levelStatisticsIndex << ", _processStatisticsIndex=" << _processStatisticsIndex << std::endl;
}

void peanoclaw::statistics::SubgridStatistics::logStatistics() const {
  //TODO unterweg debug
//  if(_minimalPatchIndex != -1) {
//    Patch minimalTimePatch(CellDescriptionHeap::getInstance().getData(_minimalPatchIndex).at(0));
//    if(minimalTimePatch.isValid()) {
//      logInfo("logStatistics()", "Minimal time subgrid" << ": " << minimalTimePatch);
//
//      //Parent
//      if(_minimalPatchParentIndex >= 0) {
//        Patch minimalTimePatchParent(
//          CellDescriptionHeap::getInstance().getData(_minimalPatchParentIndex).at(0)
//        );
//        if(minimalTimePatchParent.isValid()) {
//          logInfo("logStatistics()", "\tMinimal time subgrid parent: " << minimalTimePatchParent);
//        }
//      }
//
//      //Constraining patch
//      if(minimalTimePatch.getTimeIntervals().getConstrainingNeighborIndex() != -1) {
//        Patch constrainingPatch(CellDescriptionHeap::getInstance().getData(
//          minimalTimePatch.getTimeIntervals().getConstrainingNeighborIndex()).at(0)
//        );
//        logInfo("logStatistics()", "\tConstrained by " << constrainingPatch);
//
//        if(constrainingPatch.getTimeIntervals().getConstrainingNeighborIndex() != -1) {
//          Patch constrainingConstrainingPatch(
//            CellDescriptionHeap::getInstance().getData(constrainingPatch.getTimeIntervals().getConstrainingNeighborIndex()).at(0)
//          );
//          logInfo("logStatistics()", "\t\tConstrained^2 by " << constrainingConstrainingPatch);
//        }
//      }
//
//      logInfo("logStatistics()", "Minimal time subgrid blocked due to coarsening: " << _minimalPatchBlockedDueToCoarsening);
//      logInfo("logStatistics()", "Minimal time subgrid blocked due to global timestep: " << _minimalPatchBlockedDueToGlobalTimestep);
//    }
//  }
}

void peanoclaw::statistics::SubgridStatistics::addLevelToLevelStatistics(int level, std::vector<LevelStatistics>& levelStatistics) {
  while(static_cast<int>(levelStatistics.size()) < level + 1) {
    LevelStatistics levelStatisticsEntry;
    memset(&levelStatisticsEntry, 0, sizeof(LevelStatistics));
    levelStatisticsEntry.setMinimalTimestepSize(std::numeric_limits<double>::max());
    levelStatistics.push_back(levelStatisticsEntry);
  }
}

void peanoclaw::statistics::SubgridStatistics::addSubgridToLevelStatistics(
  const Patch& subgrid
) {
  std::vector<LevelStatistics>& levelStatistics = LevelStatisticsHeap::getInstance().getData(_levelStatisticsIndex);
  addLevelToLevelStatistics(subgrid.getLevel(), levelStatistics);

  peanoclaw::statistics::LevelStatistics& level = levelStatistics[subgrid.getLevel()-1];
  if(subgrid.isLeaf()) {
    level.setNumberOfPatches(level.getNumberOfPatches() + 1);
    level.setNumberOfCells(level.getNumberOfCells() + (tarch::la::volume(subgrid.getSubdivisionFactor())));
    level.setArea(level.getArea() + tarch::la::volume(subgrid.getSize()));
    level.setEstimatedNumberOfRemainingIterationsToGlobalTimestep(
        std::max(level.getEstimatedNumberOfRemainingIterationsToGlobalTimestep(),
                estimateRemainingIterationsUntilGlobalSubgrid(subgrid))
    );
  }
}

int peanoclaw::statistics::SubgridStatistics::estimateRemainingIterationsUntilGlobalSubgrid(Patch subgrid) const {
  if(!tarch::la::equals(subgrid.getTimeIntervals().getTimestepSize(), 0.0)) {
    double timeToGlobalTimestep = _globalTimestepEndTime - (subgrid.getTimeIntervals().getCurrentTime() + subgrid.getTimeIntervals().getTimestepSize());
    double numberOfRestrictingSubgrids = THREE_POWER_D-1;

    assertion2(tarch::la::greaterEquals(timeToGlobalTimestep, 0.0), timeToGlobalTimestep, subgrid);
    return (int)(std::ceil(timeToGlobalTimestep / subgrid.getTimeIntervals().getTimestepSize() * numberOfRestrictingSubgrids));
  } else {
    return 1;
  }
}

void peanoclaw::statistics::SubgridStatistics::copy(const SubgridStatistics& other) {
  _levelStatisticsIndex = other._levelStatisticsIndex;
  _processStatisticsIndex = other._processStatisticsIndex;
  _minimalPatchIndex = other._minimalPatchIndex;
  _minimalPatchParentIndex = other._minimalPatchParentIndex;
  _minimalPatchTime = other._minimalPatchTime;
  _startMaximumLocalTimeInterval = other._startMaximumLocalTimeInterval;
  _endMaximumLocalTimeInterval = other._endMaximumLocalTimeInterval;
  _startMinimumLocalTimeInterval = other._startMinimumLocalTimeInterval;
  _endMinimumLocalTimeInterval = other._endMinimumLocalTimeInterval;
  _minimalTimestep = other._minimalTimestep;
  _allPatchesEvolvedToGlobalTimestep = other._allPatchesEvolvedToGlobalTimestep;
  _averageGlobalTimeInterval = other._averageGlobalTimeInterval;
  _globalTimestepEndTime = other._globalTimestepEndTime;
  _minimalPatchBlockedDueToCoarsening = other._minimalPatchBlockedDueToCoarsening;
  _minimalPatchBlockedDueToGlobalTimestep = other._minimalPatchBlockedDueToGlobalTimestep;
  _isFinalized = other._isFinalized;
}

int peanoclaw::statistics::SubgridStatistics::computeProcessorHashCode() {
  #ifdef Parallel
  char processorName[MPI_MAX_PROCESSOR_NAME];
  int nameLength;
  MPI_Get_processor_name(processorName, &nameLength);
  int hashCode = 0;
  for(size_t i = 0; i < nameLength; i++) {
      hashCode = 65599 * hashCode + processorName[i];
  }
  return hashCode ^ (hashCode >> 16);
  #else
  return 0;
  #endif
}

peanoclaw::statistics::SubgridStatistics::SubgridStatistics()
: _levelStatisticsIndex(-1),
  _processStatisticsIndex(-1),
  _minimalPatchIndex(-1),
  _minimalPatchParentIndex(-1),
  _minimalPatchTime(std::numeric_limits<double>::max()),
  _startMaximumLocalTimeInterval(std::numeric_limits<double>::max()),
  _endMaximumLocalTimeInterval(-std::numeric_limits<double>::max()),
  _startMinimumLocalTimeInterval(-std::numeric_limits<double>::max()),
  _endMinimumLocalTimeInterval(std::numeric_limits<double>::max()),
  _minimalTimestep(std::numeric_limits<double>::max()),
  _allPatchesEvolvedToGlobalTimestep(true),
  _averageGlobalTimeInterval(0.0),
  _globalTimestepEndTime(-1.0),
  _minimalPatchBlockedDueToCoarsening(false),
  _minimalPatchBlockedDueToGlobalTimestep(false),
  _isFinalized(false) {
  #ifndef SharedMemoryParallelisation
  initializeLevelAndProcessStatistics();
  #endif
}

peanoclaw::statistics::SubgridStatistics::SubgridStatistics(
  double globalTimestepEndTime
): _levelStatisticsIndex(-1),
   _processStatisticsIndex(-1),
   _minimalPatchIndex(-1),
   _minimalPatchParentIndex(-1),
   _minimalPatchTime(std::numeric_limits<double>::max()),
   _startMaximumLocalTimeInterval(std::numeric_limits<double>::max()),
   _endMaximumLocalTimeInterval(-std::numeric_limits<double>::max()),
   _startMinimumLocalTimeInterval(-std::numeric_limits<double>::max()),
   _endMinimumLocalTimeInterval(std::numeric_limits<double>::max()),
   _minimalTimestep(std::numeric_limits<double>::max()),
   _allPatchesEvolvedToGlobalTimestep(true),
   _averageGlobalTimeInterval(0.0),
   _globalTimestepEndTime(globalTimestepEndTime),
   _minimalPatchBlockedDueToCoarsening(false),
   _minimalPatchBlockedDueToGlobalTimestep(false),
   _isFinalized(false) {
  #ifndef SharedMemoryParallelisation
  initializeLevelAndProcessStatistics();
  #endif
}

peanoclaw::statistics::SubgridStatistics::SubgridStatistics(const peanoclaw::State& state)
 : _levelStatisticsIndex(-1),
   //_levelStatistics(0),
   _processStatisticsIndex(-1),
   //_processStatistics(0),
   _minimalPatchIndex(-1),
   _minimalPatchParentIndex(-1),
   _minimalPatchTime(std::numeric_limits<double>::max()),
   _startMaximumLocalTimeInterval(std::numeric_limits<double>::max()),
   _endMaximumLocalTimeInterval(-std::numeric_limits<double>::max()),
   _startMinimumLocalTimeInterval(-std::numeric_limits<double>::max()),
   _endMinimumLocalTimeInterval(std::numeric_limits<double>::max()),
   _minimalTimestep(std::numeric_limits<double>::max()),
   _allPatchesEvolvedToGlobalTimestep(state.getAllPatchesEvolvedToGlobalTimestep()),
   _averageGlobalTimeInterval(0.0),
   _globalTimestepEndTime(state.getGlobalTimestepEndTime()),
   _minimalPatchBlockedDueToCoarsening(false),
   _minimalPatchBlockedDueToGlobalTimestep(false),
   _isFinalized(false) {
  #ifndef SharedMemoryParallelisation
  initializeLevelAndProcessStatistics();
  #endif
}

peanoclaw::statistics::SubgridStatistics::SubgridStatistics(
  const std::vector<LevelStatistics>& otherLevelStatistics
) : _levelStatisticsIndex(-1),
    _processStatisticsIndex(-1),
    _minimalPatchIndex(-1),
    _minimalPatchParentIndex(-1),
    _minimalPatchTime(std::numeric_limits<double>::max()),
    _startMaximumLocalTimeInterval(std::numeric_limits<double>::max()),
    _endMaximumLocalTimeInterval(-std::numeric_limits<double>::max()),
    _startMinimumLocalTimeInterval(-std::numeric_limits<double>::max()),
    _endMinimumLocalTimeInterval(std::numeric_limits<double>::max()),
    _minimalTimestep(std::numeric_limits<double>::max()),
    _allPatchesEvolvedToGlobalTimestep(true),
    _averageGlobalTimeInterval(0.0),
    _globalTimestepEndTime(0.0),
    _minimalPatchBlockedDueToCoarsening(false),
    _minimalPatchBlockedDueToGlobalTimestep(false),
    _isFinalized(false) {
  #ifndef SharedMemoryParallelisation
  initializeLevelAndProcessStatistics();
  std::vector<LevelStatistics>& levelStatistics = LevelStatisticsHeap::getInstance().getData(_levelStatisticsIndex);
  for(std::vector<LevelStatistics>::const_iterator i = levelStatistics.begin(); i != levelStatistics.end(); i++) {
    levelStatistics.push_back(*i);
  }
  #endif
}

peanoclaw::statistics::SubgridStatistics::SubgridStatistics(
  int workerRank
) : _levelStatisticsIndex(-1),
    _processStatisticsIndex(-1),
    _minimalPatchIndex(-1),
    _minimalPatchParentIndex(-1),
    _minimalPatchTime(std::numeric_limits<double>::max()),
    _startMaximumLocalTimeInterval(std::numeric_limits<double>::max()),
    _endMaximumLocalTimeInterval(-std::numeric_limits<double>::max()),
    _startMinimumLocalTimeInterval(-std::numeric_limits<double>::max()),
    _endMinimumLocalTimeInterval(std::numeric_limits<double>::max()),
    _minimalTimestep(std::numeric_limits<double>::max()),
    _allPatchesEvolvedToGlobalTimestep(true),
    _averageGlobalTimeInterval(0.0),
    _globalTimestepEndTime(0.0),
    _minimalPatchBlockedDueToCoarsening(false),
    _minimalPatchBlockedDueToGlobalTimestep(false),
    _isFinalized(false) {
  #ifndef SharedMemoryParallelisation
  _levelStatisticsIndex = LevelStatisticsHeap::getInstance().createData();
  _processStatisticsIndex = ProcessStatisticsHeap::getInstance().createData();
  tarch::multicore::Lock lock(_heapSemaphore);
  LevelStatisticsHeap::getInstance().receiveData(_levelStatisticsIndex, workerRank, 0, 0, peano::heap::MasterWorkerCommunication);

  ProcessStatisticsHeap::getInstance().receiveData(_processStatisticsIndex, workerRank, 0, 0, peano::heap::MasterWorkerCommunication);
  #endif
}

peanoclaw::statistics::SubgridStatistics::SubgridStatistics(SubgridStatistics& other)
: _levelStatisticsIndex(other._levelStatisticsIndex),
  _processStatisticsIndex(other._processStatisticsIndex),
  _minimalPatchIndex(other._minimalPatchIndex),
  _minimalPatchParentIndex(other._minimalPatchParentIndex),
  _minimalPatchTime(other._minimalPatchTime),
  _startMaximumLocalTimeInterval(other._startMaximumLocalTimeInterval),
  _endMaximumLocalTimeInterval(other._endMaximumLocalTimeInterval),
  _startMinimumLocalTimeInterval(other._startMinimumLocalTimeInterval),
  _endMinimumLocalTimeInterval(other._endMinimumLocalTimeInterval),
  _minimalTimestep(other._minimalTimestep),
  _allPatchesEvolvedToGlobalTimestep(other._allPatchesEvolvedToGlobalTimestep),
  _averageGlobalTimeInterval(other._averageGlobalTimeInterval),
  _globalTimestepEndTime(other._globalTimestepEndTime),
  _minimalPatchBlockedDueToCoarsening(other._minimalPatchBlockedDueToCoarsening),
  _minimalPatchBlockedDueToGlobalTimestep(other._minimalPatchBlockedDueToGlobalTimestep),
  _isFinalized(other._isFinalized)
{
  other._levelStatisticsIndex = -1;
  other._processStatisticsIndex = -1;
}

peanoclaw::statistics::SubgridStatistics::SubgridStatistics(const SubgridStatistics& other)
: _levelStatisticsIndex(other._levelStatisticsIndex),
  _processStatisticsIndex(other._processStatisticsIndex),
  _minimalPatchIndex(other._minimalPatchIndex),
  _minimalPatchParentIndex(other._minimalPatchParentIndex),
  _minimalPatchTime(other._minimalPatchTime),
  _startMaximumLocalTimeInterval(other._startMaximumLocalTimeInterval),
  _endMaximumLocalTimeInterval(other._endMaximumLocalTimeInterval),
  _startMinimumLocalTimeInterval(other._startMinimumLocalTimeInterval),
  _endMinimumLocalTimeInterval(other._endMinimumLocalTimeInterval),
  _minimalTimestep(other._minimalTimestep),
  _allPatchesEvolvedToGlobalTimestep(other._allPatchesEvolvedToGlobalTimestep),
  _averageGlobalTimeInterval(other._averageGlobalTimeInterval),
  _globalTimestepEndTime(other._globalTimestepEndTime),
  _minimalPatchBlockedDueToCoarsening(other._minimalPatchBlockedDueToCoarsening),
  _minimalPatchBlockedDueToGlobalTimestep(other._minimalPatchBlockedDueToGlobalTimestep),
  _isFinalized(other._isFinalized)
{
  #ifndef SharedMemoryParallelisation
  initializeLevelAndProcessStatistics();
  std::vector<LevelStatistics>& levelStatistics = LevelStatisticsHeap::getInstance().getData(_levelStatisticsIndex);
  std::vector<LevelStatistics>& otherLevelStatistics = LevelStatisticsHeap::getInstance().getData(other._levelStatisticsIndex);
  levelStatistics.resize(otherLevelStatistics.size());
  std::copy(otherLevelStatistics.begin(), otherLevelStatistics.end(), levelStatistics.begin());

  std::vector<ProcessStatisticsEntry>& processStatistics = ProcessStatisticsHeap::getInstance().getData(_processStatisticsIndex);
  std::vector<ProcessStatisticsEntry>& otherProcessStatistics = ProcessStatisticsHeap::getInstance().getData(other._processStatisticsIndex);
  processStatistics.resize(otherProcessStatistics.size());
  std::copy(otherProcessStatistics.begin(), otherProcessStatistics.end(), processStatistics.begin());
  #endif
}

peanoclaw::statistics::SubgridStatistics& peanoclaw::statistics::SubgridStatistics::operator=(
  SubgridStatistics& other
) {
  copy(other);

  other._levelStatisticsIndex = -1;
  other._processStatisticsIndex = -1;

  return *this;
}

const peanoclaw::statistics::SubgridStatistics& peanoclaw::statistics::SubgridStatistics::operator=(
  const SubgridStatistics& other
) {

  copy(other);

  #ifndef SharedMemoryParallelisation
  initializeLevelAndProcessStatistics();
  std::vector<LevelStatistics>& levelStatistics = LevelStatisticsHeap::getInstance().getData(_levelStatisticsIndex);
  std::vector<LevelStatistics>& otherLevelStatistics = LevelStatisticsHeap::getInstance().getData(other._levelStatisticsIndex);
  levelStatistics.resize(otherLevelStatistics.size());
  std::copy(otherLevelStatistics.begin(), otherLevelStatistics.end(), levelStatistics.begin());

  std::vector<ProcessStatisticsEntry>& processStatistics = ProcessStatisticsHeap::getInstance().getData(_processStatisticsIndex);
  std::vector<ProcessStatisticsEntry>& otherProcessStatistics = ProcessStatisticsHeap::getInstance().getData(other._processStatisticsIndex);
  processStatistics.resize(otherProcessStatistics.size());
  std::copy(otherProcessStatistics.begin(), otherProcessStatistics.end(), processStatistics.begin());

  #endif
  return *this;
}

peanoclaw::statistics::SubgridStatistics::~SubgridStatistics() {
  #ifndef SharedMemoryParallelisation
  tarch::multicore::Lock lock(_heapSemaphore);
  if(_levelStatisticsIndex != -1) {
    LevelStatisticsHeap::getInstance().deleteData(_levelStatisticsIndex);
  }

  if(_processStatisticsIndex != -1) {
    ProcessStatisticsHeap::getInstance().deleteData(_processStatisticsIndex);
  }
  #endif
}

void peanoclaw::statistics::SubgridStatistics::processSubgrid(
  const peanoclaw::Patch& patch,
  int parentIndex
) {
  #ifndef SharedMemoryParallelisation
  addSubgridToLevelStatistics(patch);
  #endif

  if(patch.getTimeIntervals().getCurrentTime() + patch.getTimeIntervals().getTimestepSize() < _minimalPatchTime) {
    _minimalPatchIndex = patch.getCellDescriptionIndex();
    _minimalPatchParentIndex = parentIndex;
    _minimalPatchTime = patch.getTimeIntervals().getCurrentTime() + patch.getTimeIntervals().getTimestepSize();

    //Stopping criterion for global timestep
    if(tarch::la::smaller(patch.getTimeIntervals().getCurrentTime() + patch.getTimeIntervals().getTimestepSize(), _globalTimestepEndTime)) {
      _allPatchesEvolvedToGlobalTimestep = false;
    }
  }

  _startMaximumLocalTimeInterval = std::min(patch.getTimeIntervals().getCurrentTime(), _startMaximumLocalTimeInterval);
  _endMaximumLocalTimeInterval = std::max(patch.getTimeIntervals().getCurrentTime() + patch.getTimeIntervals().getTimestepSize(), _endMaximumLocalTimeInterval);
  _startMinimumLocalTimeInterval = std::max(patch.getTimeIntervals().getCurrentTime(), _startMinimumLocalTimeInterval);
  _endMinimumLocalTimeInterval = std::min(patch.getTimeIntervals().getCurrentTime() + patch.getTimeIntervals().getTimestepSize(), _endMinimumLocalTimeInterval);

}

void peanoclaw::statistics::SubgridStatistics::processSubgridAfterUpdate(const peanoclaw::Patch& patch, int parentIndex) {

  _minimalTimestep = std::min(_minimalTimestep, patch.getTimeIntervals().getTimestepSize());

  processSubgrid(patch, parentIndex);

  #ifndef SharedMemoryParallelisation
  std::vector<LevelStatistics>& levelStatistics = LevelStatisticsHeap::getInstance().getData(_levelStatisticsIndex);
  LevelStatistics& level = levelStatistics[patch.getLevel()-1];
  level.setNumberOfCellUpdates(
    level.getNumberOfCellUpdates() + tarch::la::volume(patch.getSubdivisionFactor())
  );
  level.setAverageTimestepSize(level.getAverageTimestepSize() + patch.getTimeIntervals().getTimestepSize());
  level.setMinimalTimestepSize(std::min(level.getMinimalTimestepSize(), patch.getTimeIntervals().getTimestepSize()));

  std::vector<ProcessStatisticsEntry>& processStatistics = ProcessStatisticsHeap::getInstance().getData(_processStatisticsIndex);
  ProcessStatisticsEntry& processStatisticsEntry = processStatistics[0];
  processStatisticsEntry.setNumberOfCellUpdates(processStatisticsEntry.getNumberOfCellUpdates() + tarch::la::volume(patch.getSubdivisionFactor()));

  assertion(processStatisticsEntry.getNumberOfCellUpdates() > 0);
  assertion(processStatistics[0].getNumberOfCellUpdates() > 0);
  #endif
}

void peanoclaw::statistics::SubgridStatistics::updateMinimalSubgridBlockReason(
  const peanoclaw::Patch&              subgrid,
  peanoclaw::Vertex * const            coarseGridVertices,
  const peano::grid::VertexEnumerator& coarseGridVerticesEnumerator,
  double                               globalTimestep
) {
  if(subgrid.getCellDescriptionIndex() == _minimalPatchIndex) {
    _minimalPatchBlockedDueToCoarsening = peano::grid::aspects::VertexStateAnalysis::doesOneVertexCarryRefinementFlag
                            (
                              coarseGridVertices,
                              coarseGridVerticesEnumerator,
                              peanoclaw::records::Vertex::Erasing
                            );
    _minimalPatchBlockedDueToGlobalTimestep
      = tarch::la::greaterEquals(subgrid.getTimeIntervals().getCurrentTime() + subgrid.getTimeIntervals().getTimestepSize(), globalTimestep);
  }
}

void peanoclaw::statistics::SubgridStatistics::destroyedSubgrid(int cellDescriptionIndex) {
  if(_minimalPatchIndex == cellDescriptionIndex) {
    _minimalPatchIndex = -1;
  }
  if(_minimalPatchParentIndex == cellDescriptionIndex) {
    _minimalPatchParentIndex = -1;
  }
}

void peanoclaw::statistics::SubgridStatistics::finalizeIteration(peanoclaw::State& state) {
  state.setAllPatchesEvolvedToGlobalTimestep(
    state.getAllPatchesEvolvedToGlobalTimestep()
    && _allPatchesEvolvedToGlobalTimestep
  );
  state.updateGlobalTimeIntervals(
    _startMaximumLocalTimeInterval,
    _endMaximumLocalTimeInterval,
    _startMinimumLocalTimeInterval,
    _endMinimumLocalTimeInterval
  );
  state.updateMinimalTimestep(_minimalTimestep);

  //Finalize statistics
  _averageGlobalTimeInterval = (state.getStartMaximumGlobalTimeInterval() + state.getEndMaximumGlobalTimeInterval()) / 2.0;

  #ifndef SharedMemoryParallelisation
  std::vector<LevelStatistics>& levelStatistics = LevelStatisticsHeap::getInstance().getData(_levelStatisticsIndex);
  for(std::vector<LevelStatistics>::iterator i = levelStatistics.begin(); i != levelStatistics.end(); i++) {
    i->setAverageTimestepSize(i->getAverageTimestepSize() / i->getNumberOfPatches() * i->getNumberOfCells() / i->getNumberOfCellUpdates());
  }
  #endif
 
  if(tarch::parallel::Node::getInstance().isGlobalMaster()) {
    state.setSubgridStatisticsForLastGridIteration(*this);
  }

  _isFinalized = true;

  logStatistics();
}

void peanoclaw::statistics::SubgridStatistics::logLevelStatistics(std::string description) const {
  #ifndef SharedMemoryParallelisation
  std::vector<LevelStatistics>& levelStatistics = LevelStatisticsHeap::getInstance().getData(_levelStatisticsIndex);
  logInfo("logLevelStatistics", description << ": Spacetree height: " <<  levelStatistics.size());

  double totalArea = 0.0;
  double totalNumberOfPatches = 0.0;
  double totalNumberOfCells = 0.0;
  double totalNumberOfCellUpdates = 0.0;
  double totalBlockedPatchesDueToNeighbors = 0.0;
  double totalBlockedPatchesDueToGlobalTimestep = 0.0;
  double totalSkippingPatches = 0.0;
  double totalCoarseningPatches = 0.0;
  int    totalEstimatedIterationsToGlobalTimestep = 0;

  for(size_t i = 0; i < levelStatistics.size(); i++) {
    const LevelStatistics& level = levelStatistics.at(i);
    logInfo("logLevelStatistics", "\tLevel " << i << ": " << level.getNumberOfPatches() << " patches (area=" << level.getArea() <<  "), "
        << level.getNumberOfCells() << " cells, " << level.getNumberOfCellUpdates() << "cell updates, "
        << totalEstimatedIterationsToGlobalTimestep << " remaining iterations. "
        << "minDt=" << level.getMinimalTimestepSize() << " averageDt=" << level.getAverageTimestepSize());

    totalArea += level.getArea();
    totalNumberOfPatches += level.getNumberOfPatches();
    totalNumberOfCells += level.getNumberOfCells();
    totalNumberOfCellUpdates += level.getNumberOfCellUpdates();
    totalBlockedPatchesDueToNeighbors += level.getPatchesBlockedDueToNeighbors();
    totalBlockedPatchesDueToGlobalTimestep += level.getPatchesBlockedDueToGlobalTimestep();
    totalSkippingPatches += level.getPatchesSkippingIteration();
    totalCoarseningPatches += level.getPatchesCoarsening();
    totalEstimatedIterationsToGlobalTimestep = std::max(level.getEstimatedNumberOfRemainingIterationsToGlobalTimestep(), totalEstimatedIterationsToGlobalTimestep);
  }
  logInfo("logLevelStatistics",
    "Sum: max. " << totalNumberOfPatches << " patches (area=" << totalArea <<  "), max. "
    << totalNumberOfCells << " cells, " << totalNumberOfCellUpdates << " cell updates, "
    << totalEstimatedIterationsToGlobalTimestep << " remaining iterations. "
    << " Blocking: " << totalBlockedPatchesDueToNeighbors << ", " << totalBlockedPatchesDueToGlobalTimestep
    << ", " << totalSkippingPatches << ", " << totalCoarseningPatches
    );
  #endif
}

void peanoclaw::statistics::SubgridStatistics::logProcessStatistics(std::string description) const {
  #ifndef SharedMemoryParallelisation
  int totalCellUpdates = 0;
  int maximumLocalCellUpdates = 0;
  int numberOfWorkers = 0;
  int numberOfIgnoredProcesses = 0;

  std::vector<ProcessStatisticsEntry>& processStatistics = ProcessStatisticsHeap::getInstance().getData(_processStatisticsIndex);
  logInfo("logLevelStatistics", description << ": #ranks=" <<  processStatistics.size());
  for(std::vector<ProcessStatisticsEntry>::iterator i = processStatistics.begin(); i != processStatistics.end(); i++) {
    totalCellUpdates += i->getNumberOfCellUpdates();
    maximumLocalCellUpdates = std::max(maximumLocalCellUpdates, i->getNumberOfCellUpdates());

    if(i->getNumberOfCellUpdates() > 0) {
      numberOfWorkers++;
    } else {
      numberOfIgnoredProcesses++;
    }

    logInfo("logProcessStatistics(...)", "Rank " << i->getRank() << " (Processor: " << i->getProcessorHashCode() << "): #cell updates=" << i->getNumberOfCellUpdates());
  }

  double averageCellUpdates = (double)totalCellUpdates / numberOfWorkers;
  double imbalance = maximumLocalCellUpdates / averageCellUpdates;
  logInfo("logProcessStatistics(...)", "Workers: " << numberOfWorkers << " Ignored: " << numberOfIgnoredProcesses
      << " Average: " << std::fixed << std::setprecision(2) << averageCellUpdates << " Imbalance: " << imbalance);
  #endif
}

void peanoclaw::statistics::SubgridStatistics::addBlockedPatchDueToGlobalTimestep(const Patch& subgrid) {
  #ifndef SharedMemoryParallelisation
  std::vector<LevelStatistics>& levelStatistics = LevelStatisticsHeap::getInstance().getData(_levelStatisticsIndex);
  addLevelToLevelStatistics(subgrid.getLevel(), levelStatistics);
  levelStatistics[subgrid.getLevel() - 1].setPatchesBlockedDueToGlobalTimestep(
      levelStatistics[subgrid.getLevel() - 1].getPatchesBlockedDueToGlobalTimestep() + 1
  );
  #endif
}

void peanoclaw::statistics::SubgridStatistics::addBlockedPatchDueToNeighborTimeConstraint(const Patch& subgrid) {
  #ifndef SharedMemoryParallelisation
  std::vector<LevelStatistics>& levelStatistics = LevelStatisticsHeap::getInstance().getData(_levelStatisticsIndex);
  addLevelToLevelStatistics(subgrid.getLevel(), levelStatistics);
  levelStatistics[subgrid.getLevel() - 1].setPatchesBlockedDueToNeighbors(
      levelStatistics[subgrid.getLevel() - 1].getPatchesBlockedDueToNeighbors() + 1
  );
  #endif
}
void peanoclaw::statistics::SubgridStatistics::addBlockedPatchDueToSkipIteration(const Patch& subgrid) {
  #ifndef SharedMemoryParallelisation
  std::vector<LevelStatistics>& levelStatistics = LevelStatisticsHeap::getInstance().getData(_levelStatisticsIndex);
  addLevelToLevelStatistics(subgrid.getLevel(), levelStatistics);
  levelStatistics[subgrid.getLevel() - 1].setPatchesSkippingIteration(
      levelStatistics[subgrid.getLevel() - 1].getPatchesSkippingIteration() + 1
  );
  #endif
}
void peanoclaw::statistics::SubgridStatistics::addBlockedPatchDueToCoarsening(const Patch& subgrid) {
  #ifndef SharedMemoryParallelisation
  std::vector<LevelStatistics>& levelStatistics = LevelStatisticsHeap::getInstance().getData(_levelStatisticsIndex);
  addLevelToLevelStatistics(subgrid.getLevel(), levelStatistics);
  levelStatistics[subgrid.getLevel() - 1].setPatchesCoarsening(
      levelStatistics[subgrid.getLevel() - 1].getPatchesCoarsening() + 1
  );
  #endif
}

void peanoclaw::statistics::SubgridStatistics::merge(const SubgridStatistics& subgridStatistics) {

  //Subgrid statistics
  if(_minimalPatchTime > subgridStatistics._minimalPatchTime) {
    _minimalPatchTime = subgridStatistics._minimalPatchTime;
    _minimalPatchIndex = subgridStatistics._minimalPatchIndex;
    _minimalPatchParentIndex = subgridStatistics._minimalPatchParentIndex;
  }

  _startMaximumLocalTimeInterval     = std::min(_startMaximumLocalTimeInterval, subgridStatistics._startMaximumLocalTimeInterval);
  _endMaximumLocalTimeInterval       = std::max(_endMaximumLocalTimeInterval, subgridStatistics._endMaximumLocalTimeInterval);
  _startMinimumLocalTimeInterval     = std::max(_startMinimumLocalTimeInterval, subgridStatistics._startMinimumLocalTimeInterval);
  _endMinimumLocalTimeInterval       = std::min(_endMinimumLocalTimeInterval, subgridStatistics._endMinimumLocalTimeInterval);
  _minimalTimestep                   = std::min(_minimalTimestep, subgridStatistics._minimalTimestep);
  _allPatchesEvolvedToGlobalTimestep &= subgridStatistics._allPatchesEvolvedToGlobalTimestep;
  _averageGlobalTimeInterval         = (_averageGlobalTimeInterval + subgridStatistics._averageGlobalTimeInterval) / 2.0;

  #ifndef SharedMemoryParallelisation
  //Level statistics
  std::vector<LevelStatistics>& otherLevelStatistics = LevelStatisticsHeap::getInstance().getData(subgridStatistics._levelStatisticsIndex);
  std::vector<LevelStatistics>& levelStatistics = LevelStatisticsHeap::getInstance().getData(_levelStatisticsIndex);
  addLevelToLevelStatistics(otherLevelStatistics.size()-1, levelStatistics);
  for(int level = 0; level < (int)otherLevelStatistics.size(); level++) {

    LevelStatistics& thisLevel = levelStatistics[level];
    LevelStatistics& otherLevel = otherLevelStatistics[level];

    thisLevel.setArea(thisLevel.getArea() + otherLevel.getArea());
    thisLevel.setCreatedPatches(thisLevel.getCreatedPatches() + otherLevel.getCreatedPatches());
    thisLevel.setDestroyedPatches(thisLevel.getDestroyedPatches() + otherLevel.getDestroyedPatches());
    thisLevel.setNumberOfCellUpdates(thisLevel.getNumberOfCellUpdates() + otherLevel.getNumberOfCellUpdates());
    thisLevel.setNumberOfCells(thisLevel.getNumberOfCells() + otherLevel.getNumberOfCells());
    thisLevel.setNumberOfPatches(thisLevel.getNumberOfPatches() + otherLevel.getNumberOfPatches());
    thisLevel.setPatchesBlockedDueToGlobalTimestep(thisLevel.getPatchesBlockedDueToGlobalTimestep() + otherLevel.getPatchesBlockedDueToGlobalTimestep());
    thisLevel.setPatchesBlockedDueToNeighbors(thisLevel.getPatchesBlockedDueToNeighbors() + otherLevel.getPatchesBlockedDueToNeighbors());
    thisLevel.setPatchesCoarsening(thisLevel.getPatchesCoarsening() + otherLevel.getPatchesCoarsening());
    thisLevel.setPatchesSkippingIteration(thisLevel.getPatchesSkippingIteration() + otherLevel.getPatchesSkippingIteration());
    thisLevel.setEstimatedNumberOfRemainingIterationsToGlobalTimestep(
      std::max(thisLevel.getEstimatedNumberOfRemainingIterationsToGlobalTimestep(), otherLevel.getEstimatedNumberOfRemainingIterationsToGlobalTimestep())
    );
  }

  //Merge process statistics
  std::vector<ProcessStatisticsEntry>& otherProcessStatistics = ProcessStatisticsHeap::getInstance().getData(subgridStatistics._processStatisticsIndex);
  std::vector<ProcessStatisticsEntry>& processStatistics = ProcessStatisticsHeap::getInstance().getData(_processStatisticsIndex);
  std::sort(processStatistics.begin(), processStatistics.end(), smaller);
  std::sort(otherProcessStatistics.begin(), otherProcessStatistics.end(), smaller);
  size_t ownIndex = 0;

  for(std::vector<ProcessStatisticsEntry>::iterator other = otherProcessStatistics.begin(); other != otherProcessStatistics.end(); other++) {
    while(ownIndex < processStatistics.size() && processStatistics[ownIndex].getRank() < other->getRank()) {
      ownIndex++;
    }

    if(ownIndex == processStatistics.size()) {
      processStatistics.push_back(*other);
    } else if(processStatistics[ownIndex].getRank() == other->getRank()) {
      processStatistics[ownIndex].setNumberOfCellUpdates(processStatistics[ownIndex].getNumberOfCellUpdates() + other->getNumberOfCellUpdates());
    } else {
      processStatistics.push_back(*other);
    }
  }
  #endif
}

void peanoclaw::statistics::SubgridStatistics::averageTotalSimulationValues(int numberOfEntries) {
  #ifndef SharedMemoryParallelisation
  std::vector<LevelStatistics>& levelStatistics = LevelStatisticsHeap::getInstance().getData(_levelStatisticsIndex);
  for(int i = 0; i < (int)levelStatistics.size(); i++) {
    LevelStatistics& level = levelStatistics[i];
    level.setArea(level.getArea() / numberOfEntries);
    level.setNumberOfCells(level.getNumberOfCells() / numberOfEntries);
    level.setNumberOfPatches(level.getNumberOfPatches() / numberOfEntries);
  }
  #endif
}

#ifdef Parallel
void peanoclaw::statistics::SubgridStatistics::sendToMaster(int masterRank) {
  LevelStatisticsHeap::getInstance().sendData(
    _levelStatisticsIndex,
    masterRank,
    0,
    0,
    peano::heap::MasterWorkerCommunication
  );
  ProcessStatisticsHeap::getInstance().sendData(
    _processStatisticsIndex,
    masterRank,
    0,
    0,
    peano::heap::MasterWorkerCommunication
  );
}

int peanoclaw::statistics::SubgridStatistics::getEstimatedIterationsUntilGlobalTimestep() const {
  int maximumEstimationOfLevels = 0;
  #ifndef SharedMemoryParallelisation
  std::vector<LevelStatistics>& levelStatistics = LevelStatisticsHeap::getInstance().getData(_levelStatisticsIndex);
  for(std::vector<LevelStatistics>::iterator i = levelStatistics.begin(); i != levelStatistics.end(); i++) {
    maximumEstimationOfLevels = std::max(maximumEstimationOfLevels, i->getEstimatedNumberOfRemainingIterationsToGlobalTimestep());
  }
  #endif
  return maximumEstimationOfLevels;
}

void peanoclaw::statistics::SubgridStatistics::restrictionFromWorkerSkipped() {
  _allPatchesEvolvedToGlobalTimestep = false;
}
#endif
