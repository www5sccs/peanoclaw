/*
 * SubgridStatistics.cpp
 *
 *  Created on: Jul 29, 2013
 *      Author: kristof
 */
#include "peanoclaw/statistics/SubgridStatistics.h"

#include "peanoclaw/Vertex.h"

#include "peano/grid/aspects/VertexStateAnalysis.h"
#include "peano/heap/Heap.h"

#include <limits>

tarch::logging::Log peanoclaw::statistics::SubgridStatistics::_log("peanoclaw::statistics::SubgridStatistics");

void peanoclaw::statistics::SubgridStatistics::initializeLevelStatistics() {
  peano::heap::Heap<LevelStatistics>& heap = peano::heap::Heap<LevelStatistics>::getInstance();
  _levelStatisticsIndex = heap.createData();
  _levelStatistics = &heap.getData(_levelStatisticsIndex);
}

void peanoclaw::statistics::SubgridStatistics::logStatistics() const {
  if(_minimalPatchIndex != -1) {
    Patch minimalTimePatch(peano::heap::Heap<CellDescription>::getInstance().getData(_minimalPatchIndex).at(0));
    if(minimalTimePatch.isValid()) {
      logInfo("logStatistics()", "Minimal time subgrid" << ": " << minimalTimePatch);

      //Parent
      if(_minimalPatchParentIndex >= 0) {
        Patch minimalTimePatchParent(
          peano::heap::Heap<CellDescription>::getInstance().getData(_minimalPatchParentIndex).at(0)
        );
        if(minimalTimePatchParent.isValid()) {
          logInfo("logStatistics()", "\tMinimal time subgrid parent: " << minimalTimePatchParent);
        }
      }

      //Constraining patch
      if(minimalTimePatch.getConstrainingNeighborIndex() != -1) {
        Patch constrainingPatch(peano::heap::Heap<CellDescription>::getInstance().getData(minimalTimePatch.getConstrainingNeighborIndex()).at(0));
        logInfo("logStatistics()", "\tConstrained by " << constrainingPatch);
      }

      logInfo("logStatistics()", "Minimal time subgrid blocked due to coarsening: " << _minimalPatchBlockedDueToCoarsening);
      logInfo("logStatistics()", "Minimal time subgrid blocked due to global timestep: " << _minimalPatchBlockedDueToGlobalTimestep);
    }
  }
}

void peanoclaw::statistics::SubgridStatistics::addLevelToLevelStatistics(int level) {
  while(static_cast<int>(_levelStatistics->size()) < level + 1) {
    LevelStatistics levelStatistics;
    memset(&levelStatistics, 0, sizeof(LevelStatistics));
    _levelStatistics->push_back(levelStatistics);
  }
}

void peanoclaw::statistics::SubgridStatistics::addSubgridToLevelStatistics(
  const Patch& subgrid
) {
  addLevelToLevelStatistics(subgrid.getLevel());
  peanoclaw::statistics::LevelStatistics& level = _levelStatistics->at(subgrid.getLevel()-1);
  if(subgrid.isLeaf()) {
    level.setNumberOfPatches(level.getNumberOfPatches() + 1);
    level.setNumberOfCells(level.getNumberOfCells() + (tarch::la::volume(subgrid.getSubdivisionFactor())));
    level.setArea(level.getArea() + tarch::la::volume(subgrid.getSize()));
  }
}

peanoclaw::statistics::SubgridStatistics::SubgridStatistics()
: _levelStatisticsIndex(-1),
  _levelStatistics(0),
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
  initializeLevelStatistics();
}

peanoclaw::statistics::SubgridStatistics::SubgridStatistics(const peanoclaw::State& state)
 : _levelStatisticsIndex(-1),
   _levelStatistics(0),
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
  initializeLevelStatistics();
}

peanoclaw::statistics::SubgridStatistics::SubgridStatistics(
  const std::vector<LevelStatistics>& levelStatistics
) : _levelStatisticsIndex(-1),
    _levelStatistics(0),
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
  initializeLevelStatistics();
  for(std::vector<LevelStatistics>::const_iterator i = levelStatistics.begin(); i != levelStatistics.end(); i++) {
    _levelStatistics->push_back(*i);
  }
}

//peanoclaw::statistics::SubgridStatistics::SubgridStatistics(const SubgridStatistics& toCopy)
//: _levelStatisticsIndex(-1),
//  _levelStatistics(0),
//  _minimalPatchIndex(-1),
//  _minimalPatchParentIndex(-1),
//  _minimalPatchTime(std::numeric_limits<double>::max()),
//  _startMaximumLocalTimeInterval(std::numeric_limits<double>::max()),
//  _endMaximumLocalTimeInterval(-std::numeric_limits<double>::max()),
//  _startMinimumLocalTimeInterval(-std::numeric_limits<double>::max()),
//  _endMinimumLocalTimeInterval(std::numeric_limits<double>::max()),
//  _minimalTimestep(std::numeric_limits<double>::max()),
//  _allPatchesEvolvedToGlobalTimestep(true),
//  _averageGlobalTimeInterval(0.0),
//  _globalTimestepEndTime(0.0),
//  _isFinalized(false) {
//  initializeLevelStatistics();
//  for(std::vector<LevelStatistics>::iterator i = toCopy._levelStatistics->begin(); i != toCopy._levelStatistics->end(); i++) {
//    _levelStatistics->push_back(*i);
//  }
//}

peanoclaw::statistics::SubgridStatistics::~SubgridStatistics() {
  //peano::heap::Heap<LevelStatistics>::getInstance().deleteData(_levelStatisticsIndex);
  //_levelStatistics = 0;
}

void peanoclaw::statistics::SubgridStatistics::processSubgrid(
  const peanoclaw::Patch& patch,
  int parentIndex
) {
  addSubgridToLevelStatistics(patch);

  if(patch.getCurrentTime() + patch.getTimestepSize() < _minimalPatchTime) {
    _minimalPatchIndex = patch.getCellDescriptionIndex();
    _minimalPatchParentIndex = parentIndex;
  }

  //Stopping criterion for global timestep
  if(tarch::la::smaller(patch.getCurrentTime() + patch.getTimestepSize(), _globalTimestepEndTime)) {

    //TODO unterweg debug
    logInfo("", "Blocking global timestep: " << patch);

    _allPatchesEvolvedToGlobalTimestep = false;
  }

  _startMaximumLocalTimeInterval = std::min(patch.getCurrentTime(), _startMaximumLocalTimeInterval);
  _endMaximumLocalTimeInterval = std::max(patch.getCurrentTime() + patch.getTimestepSize(), _endMaximumLocalTimeInterval);
  _startMinimumLocalTimeInterval = std::max(patch.getCurrentTime(), _startMinimumLocalTimeInterval);
  _endMinimumLocalTimeInterval = std::min(patch.getCurrentTime() + patch.getTimestepSize(), _endMinimumLocalTimeInterval);
}

void peanoclaw::statistics::SubgridStatistics::processSubgridAfterUpdate(const peanoclaw::Patch& patch, int parentIndex) {
  _minimalTimestep = std::min(_minimalTimestep, patch.getTimestepSize());

  processSubgrid(patch, parentIndex);
  LevelStatistics& level = _levelStatistics->at(patch.getLevel()-1);
  level.setNumberOfCellUpdates(
    level.getNumberOfCellUpdates() + tarch::la::volume(patch.getSubdivisionFactor())
  );
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
      = tarch::la::greaterEquals(subgrid.getCurrentTime() + subgrid.getTimestepSize(), globalTimestep);
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

  state.setLevelStatisticsForLastGridIteration(*_levelStatistics);

  _isFinalized = true;

  logStatistics();
}

void peanoclaw::statistics::SubgridStatistics::logLevelStatistics(std::string description) {
  logInfo("logLevelStatistics", description << ": Spacetree height: " <<  _levelStatistics->size());

  double totalArea = 0.0;
  double totalNumberOfPatches = 0.0;
  double totalNumberOfCells = 0.0;
  double totalNumberOfCellUpdates = 0.0;
  double totalBlockedPatchesDueToNeighbors = 0.0;
  double totalBlockedPatchesDueToGlobalTimestep = 0.0;
  double totalSkippingPatches = 0.0;
  double totalCoarseningPatches = 0.0;

  for(size_t i = 0; i < _levelStatistics->size(); i++) {
    const LevelStatistics& level = _levelStatistics->at(i);
    logInfo("logLevelStatistics", "\tLevel " << i << ": " << level.getNumberOfPatches() << " patches (area=" << level.getArea() <<  "), "
        << level.getNumberOfCells() << " cells, " << level.getNumberOfCellUpdates() << " cell updates.");

    totalArea += level.getArea();
    totalNumberOfPatches += level.getNumberOfPatches();
    totalNumberOfCells += level.getNumberOfCells();
    totalNumberOfCellUpdates += level.getNumberOfCellUpdates();
    totalBlockedPatchesDueToNeighbors += level.getPatchesBlockedDueToNeighbors();
    totalBlockedPatchesDueToGlobalTimestep += level.getPatchesBlockedDueToGlobalTimestep();
    totalSkippingPatches += level.getPatchesSkippingIteration();
    totalCoarseningPatches += level.getPatchesCoarsening();
  }
  logInfo("logLevelStatistics",
    "Sum: max. " << totalNumberOfPatches << " patches (area=" << totalArea <<  "), max. "
    << totalNumberOfCells << " cells, " << totalNumberOfCellUpdates << " cell updates."
    << " Blocking: " << totalBlockedPatchesDueToNeighbors << ", " << totalBlockedPatchesDueToGlobalTimestep
    << ", " << totalSkippingPatches << ", " << totalCoarseningPatches
    );
}

void peanoclaw::statistics::SubgridStatistics::addBlockedPatchDueToGlobalTimestep(const Patch& subgrid) {
  addLevelToLevelStatistics(subgrid.getLevel());
  _levelStatistics->at(subgrid.getLevel() - 1).setPatchesBlockedDueToGlobalTimestep(
      _levelStatistics->at(subgrid.getLevel() - 1).getPatchesBlockedDueToGlobalTimestep() + 1
  );
}
void peanoclaw::statistics::SubgridStatistics::addBlockedPatchDueToNeighborTimeConstraint(const Patch& subgrid) {
  addLevelToLevelStatistics(subgrid.getLevel());
  _levelStatistics->at(subgrid.getLevel() - 1).setPatchesBlockedDueToNeighbors(
      _levelStatistics->at(subgrid.getLevel() - 1).getPatchesBlockedDueToNeighbors() + 1
  );
}
void peanoclaw::statistics::SubgridStatistics::addBlockedPatchDueToSkipIteration(const Patch& subgrid) {
  addLevelToLevelStatistics(subgrid.getLevel());
  _levelStatistics->at(subgrid.getLevel() - 1).setPatchesSkippingIteration(
      _levelStatistics->at(subgrid.getLevel() - 1).getPatchesSkippingIteration() + 1
  );
}
void peanoclaw::statistics::SubgridStatistics::addBlockedPatchDueToCoarsening(const Patch& subgrid) {
  addLevelToLevelStatistics(subgrid.getLevel());
  _levelStatistics->at(subgrid.getLevel() - 1).setPatchesCoarsening(
      _levelStatistics->at(subgrid.getLevel() - 1).getPatchesCoarsening() + 1
  );
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

  //Level statistics
  addLevelToLevelStatistics(subgridStatistics._levelStatistics->size()-1);
  for(int level = 0; level < (int)subgridStatistics._levelStatistics->size(); level++) {

    LevelStatistics& thisLevel = _levelStatistics->at(level);
    LevelStatistics& otherLevel = subgridStatistics._levelStatistics->at(level);

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
  }
}

void peanoclaw::statistics::SubgridStatistics::averageTotalSimulationValues(int numberOfEntries) {
  for(int i = 0; i < (int)_levelStatistics->size(); i++) {
    LevelStatistics& level = _levelStatistics->at(i);
    level.setArea(level.getArea() / numberOfEntries);
    level.setNumberOfCells(level.getNumberOfCells() / numberOfEntries);
    level.setNumberOfPatches(level.getNumberOfPatches() / numberOfEntries);
  }
}

#ifdef Parallel
void peanoclaw::statistics::SubgridStatistics::sendToMaster(int masterRank) {
  peano::heap::Heap<LevelStatistics>::getInstance().sendData(
    _levelStatisticsIndex,
    masterRank,
    0,
    0,
    peano::heap::MasterWorkerCommunication
  );
}

void peanoclaw::statistics::SubgridStatistics::receiveFromWorker(int workerRank) {
  SubgridStatistics remoteStatistics(
    peano::heap::Heap<LevelStatistics>::getInstance().receiveData(workerRank, 0, 0, peano::heap::MasterWorkerCommunication)
  );
  merge(remoteStatistics);
}
#endif
