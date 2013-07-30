/*
 * SubgridStatistics.cpp
 *
 *  Created on: Jul 29, 2013
 *      Author: kristof
 */
#include "peanoclaw/statistics/SubgridStatistics.h"

#include "peano/heap/Heap.h"

#include <limits>

tarch::logging::Log peanoclaw::statistics::SubgridStatistics::_log("peanoclaw::statistics::SubgridStatistics");

void peanoclaw::statistics::SubgridStatistics::logStatistics() const {
  if(_minimalPatchIndex != -1) {
    Patch minimalTimePatch(peano::heap::Heap<CellDescription>::getInstance().getData(_minimalPatchIndex).at(0));
    if(minimalTimePatch.isValid()) {
      logInfo("logStatistics()", "Minimal time patch" << ": " << minimalTimePatch);

      //Parent
      if(_minimalPatchParentIndex != -1) {
        Patch minimalTimePatchParent(
          peano::heap::Heap<CellDescription>::getInstance().getData(_minimalPatchParentIndex).at(0)
        );
        if(minimalTimePatchParent.isValid()) {
          logInfo("logStatistics()", "\tMinimal time patch parent: " << minimalTimePatchParent);
        }
      }

      //Constraining patch
      if(minimalTimePatch.getConstrainingNeighborIndex() != -1) {
        Patch constrainingPatch(peano::heap::Heap<CellDescription>::getInstance().getData(minimalTimePatch.getConstrainingNeighborIndex()).at(0));
        logInfo("logStatistics()", "\tConstrained by " << constrainingPatch);
      }
    }
  }
}

peanoclaw::statistics::SubgridStatistics::SubgridStatistics()
: _minimalPatchIndex(-1),
  _minimalPatchParentIndex(-1),
  _minimalPatchTime(std::numeric_limits<double>::max()),
  _startMaximumLocalTimeInterval(std::numeric_limits<double>::max()),
  _endMaximumLocalTimeInterval(-std::numeric_limits<double>::max()),
  _startMinimumLocalTimeInterval(-std::numeric_limits<double>::max()),
  _endMinimumLocalTimeInterval(std::numeric_limits<double>::max()),
  _minimalTimestep(std::numeric_limits<double>::max()),
  _allPatchesEvolvedToGlobalTimestep(-1.0),
  _averageGlobalTimeInterval(0.0),
  _globalTimestepEndTime(-1.0),
  _isFinalized(false) {
}

peanoclaw::statistics::SubgridStatistics::SubgridStatistics(const peanoclaw::State& state)
 : _minimalPatchIndex(-1),
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
   _isFinalized(false) {

}

void peanoclaw::statistics::SubgridStatistics::processSubgrid(const peanoclaw::Patch& patch, int parentIndex) {
  if(patch.getCurrentTime() + patch.getTimestepSize() < _minimalPatchTime) {
    _minimalPatchIndex = patch.getCellDescriptionIndex();
    _minimalPatchParentIndex = parentIndex;
  }

  //Stopping criterion for global timestep
  if(tarch::la::smaller(patch.getCurrentTime() + patch.getTimestepSize(), _globalTimestepEndTime)) {
    _allPatchesEvolvedToGlobalTimestep = false;
  }

  _startMaximumLocalTimeInterval = std::min(patch.getCurrentTime(), _startMaximumLocalTimeInterval);
  _endMaximumLocalTimeInterval = std::max(patch.getCurrentTime() + patch.getTimestepSize(), _endMaximumLocalTimeInterval);
  _startMinimumLocalTimeInterval = std::max(patch.getCurrentTime(), _startMinimumLocalTimeInterval);
  _endMinimumLocalTimeInterval = std::min(patch.getCurrentTime() + patch.getTimestepSize(), _endMinimumLocalTimeInterval);

  //TODO unterweg: We cannot do this here. If we need it -> new method that is called in leaveCell(...).
//  if(finePatch.isLeaf() && (finePatch.getCurrentTime() + finePatch.getTimestepSize() < _minimalPatchTime)) {
//    _minimalPatchTime = finePatch.getCurrentTime() + finePatch.getTimestepSize();
//    _minimalTimePatch = finePatch;
//    _minimalPatchCoarsening = peano::grid::aspects::VertexStateAnalysis::doesOneVertexCarryRefinementFlag
//                            (
//                              coarseGridVertices,
//                              coarseGridVerticesEnumerator,
//                              peanoclaw::records::Vertex::Erasing
//                            );
//    _minimalPatchIsAllowedToAdvanceInTime = finePatch.isAllowedToAdvanceInTime();
//    _minimalPatchShouldSkipGridIteration = finePatch.shouldSkipNextGridIteration();
//
//    if(coarseGridCell.getCellDescriptionIndex() > 0) {
//      Patch coarsePatch(coarseGridCell);
//      _minimalTimePatchParent = coarsePatch;
//    }
//  }
}

void peanoclaw::statistics::SubgridStatistics::processSubgridAfterUpdate(const peanoclaw::Patch& patch, int parentIndex) {
  _minimalTimestep = std::min(_minimalTimestep, patch.getTimestepSize());

  processSubgrid(patch, parentIndex);
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

  _isFinalized = true;

  logStatistics();
}
