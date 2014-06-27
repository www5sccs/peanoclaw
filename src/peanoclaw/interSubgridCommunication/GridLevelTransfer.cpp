/*
 * GridLevelTransfer.cpp
 *
 *  Created on: Mar 19, 2012
 *      Author: Kristof Unterweger
 */
#include "peanoclaw/interSubgridCommunication/GridLevelTransfer.h"

#include <limits>

#include "peanoclaw/Heap.h"
#include "peanoclaw/Patch.h"
#include "peanoclaw/Numerics.h"
#include "peanoclaw/Vertex.h"

#include "peano/grid/VertexEnumerator.h"
#include "peanoclaw/Heap.h"
#include "peano/grid/aspects/VertexStateAnalysis.h"
#include "peano/utils/Loop.h"

#include "tarch/multicore/Lock.h"

tarch::logging::Log peanoclaw::interSubgridCommunication::GridLevelTransfer::_log("peanoclaw::interSubgridCommunication::GridLevelTransfer");

void peanoclaw::interSubgridCommunication::GridLevelTransfer::vetoCoarseningIfNecessary (
  Patch&                               patch,
  peanoclaw::Vertex * const            fineGridVertices,
  const peano::grid::VertexEnumerator& fineGridVerticesEnumerator
) {
  assertion(!patch.isLeaf());

  if(tarch::la::smaller(patch.getTimeIntervals().getTimestepSize(), 0.0)) {
    assertion(!patch.isLeaf());
    bool patchBlocksErasing = false;
    for( int i = 0; i < TWO_POWER_D; i++ ) {
      peanoclaw::Vertex& vertex = fineGridVertices[fineGridVerticesEnumerator(i)];
      if(vertex.shouldErase()) {
        patchBlocksErasing = true;
      }
      vertex.setSubcellEraseVeto(i);
    }

    if(patchBlocksErasing) {
      patch.getTimeIntervals().setFineGridsSynchronize(true);
    }
  } else {
    patch.getTimeIntervals().setFineGridsSynchronize(false);
  }
}

bool peanoclaw::interSubgridCommunication::GridLevelTransfer::isPatchAdjacentToRemoteRank (
  peanoclaw::Vertex * const            fineGridVertices,
  const peano::grid::VertexEnumerator& fineGridVerticesEnumerator
) {
  #ifdef Parallel
  bool isAdjacentToRemoteRank = false;
  for(int i = 0; i < TWO_POWER_D; i++) {
    isAdjacentToRemoteRank |= fineGridVertices[fineGridVerticesEnumerator(i)].isAdjacentToRemoteRank();
  }
  return isAdjacentToRemoteRank;
  #else
  return false;
  #endif
}

bool peanoclaw::interSubgridCommunication::GridLevelTransfer::shouldBecomeVirtualSubgrid(
  const Patch&                         fineSubgrid,
  peanoclaw::Vertex * const            fineGridVertices,
  const peano::grid::VertexEnumerator& fineGridVerticesEnumerator,
  bool                                 isInitializing,
  bool                                 isPeanoCellLeaf
) {
  //Check wether the patch should become a virtual patch
  bool createVirtualPatch = false;
  if(!fineSubgrid.isLeaf()) {
    if(peano::grid::aspects::VertexStateAnalysis::doesOneVertexCarryRefinementFlag
      (
        fineGridVertices,
        fineGridVerticesEnumerator,
        peanoclaw::records::Vertex::Unrefined
      )
    ) {
      createVirtualPatch = true;
    }
    //TODO unterweg: Dokumentation
    if(!isInitializing
      && !peano::grid::aspects::VertexStateAnalysis::doAllVerticesCarryRefinementFlag
      (
        fineGridVertices,
        fineGridVerticesEnumerator,
        peanoclaw::records::Vertex::Refined
      )
    ) {
      createVirtualPatch = true;
    }
//      TODO unterweg debug
//      if(finePatch.getLevel() > 1) {
//        createVirtualPatch = true;
//      }
  }
//  else {
//    createVirtualPatch = !isPeanoCellLeaf;
//  }

  return createVirtualPatch;
}

void peanoclaw::interSubgridCommunication::GridLevelTransfer::switchToAndAddVirtualSubgrid(
  Patch& subgrid
) {
  tarch::multicore::Lock lock(_virtualPatchListSemaphore);

  //Push virtual stack
  tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> virtualSubgridKey = createVirtualSubgridKey(subgrid.getPosition(), subgrid.getLevel());
  _virtualPatchDescriptionIndices[virtualSubgridKey] = subgrid.getCellDescriptionIndex();
//  _virtualPatchTimeConstraints[virtualSubgridKey] = subgrid.getMinimalNeighborTimeConstraint();

  if(static_cast<int>(_virtualPatchDescriptionIndices.size()) > _maximumNumberOfSimultaneousVirtualPatches) {
    _maximumNumberOfSimultaneousVirtualPatches = _virtualPatchDescriptionIndices.size();
  }

  //Create virtual patch
  if(subgrid.isVirtual()) {
    subgrid.getAccessor().clearRegion(
      tarch::la::Vector<DIMENSIONS, int>(0),
      subgrid.getSubdivisionFactor(),
      false
    );
    subgrid.getAccessor().clearRegion(
      tarch::la::Vector<DIMENSIONS, int>(0),
      subgrid.getSubdivisionFactor(),
      true
    );
  } else {
    subgrid.switchToVirtual();
  }
  _numerics.update(subgrid);
}

void peanoclaw::interSubgridCommunication::GridLevelTransfer::restrictToOverlappingVirtualSubgrids(
  Patch&           subgrid,
  ParallelSubgrid& parallelSubgrid
) {
  tarch::multicore::Lock lock(_virtualPatchListSemaphore);

  //Restrict to all
  //for(int i = 0;  i < (int)_virtualPatchDescriptionIndices.size(); i++) {
  for(VirtualSubgridMap::iterator i = _virtualPatchDescriptionIndices.begin();
      i != _virtualPatchDescriptionIndices.end();
      i++) {
    int virtualSubgridDescriptionIndex = i->second;
    CellDescription& virtualSubgridDescription = CellDescriptionHeap::getInstance().getData(virtualSubgridDescriptionIndex).at(0);
    Patch virtualSubgrid(virtualSubgridDescription);
    ParallelSubgrid virtualParallelSubgrid(virtualSubgridDescription);

    // Restrict only if coarse patches can advance in time
    bool areAllCoarseSubgridsBlocked
    = tarch::la::smaller(
        subgrid.getTimeIntervals().getCurrentTime() + subgrid.getTimeIntervals().getTimestepSize(),
        virtualSubgrid.getTimeIntervals().getMinimalLeafNeighborTimeConstraint()
      );
    // Restrict only if this patch is overlapped by neighboring ghostlayers
    bool isOverlappedByCoarseGhostlayers
      = tarch::la::oneGreater(virtualSubgrid.getUpperNeighboringGhostlayerBounds(), subgrid.getPosition())
        || tarch::la::oneGreater(subgrid.getPosition() + subgrid.getSize(), virtualSubgrid.getLowerNeighboringGhostlayerBounds());

    bool subgridOverlapsVirtualSubgrid = !tarch::la::oneGreater(virtualSubgrid.getPosition(), subgrid.getPosition())
                                      && !tarch::la::oneGreater(subgrid.getPosition() + subgrid.getSize(), virtualSubgrid.getPosition() + virtualSubgrid.getSize());

    //TODO unterweg debug
//    if(tarch::la::equals(subgrid.getPosition()(0), 10.0*2.0/3.0)
//      &&tarch::la::equals(subgrid.getPosition()(1), 10.0*2.0/3.0)) {
//      std::cout << "Restricting from " << subgrid
//          << ", isOverlapped=" << isOverlappedByCoarseGhostlayers
//          << ", areAllCoarseSubgridsBlocked=" << areAllCoarseSubgridsBlocked
//          << ", willCoarsen=" << virtualSubgrid.willCoarsen()
//          << std::endl << subgrid.toStringUNew() << std::endl
//          << " to " << virtualSubgrid << std::endl << virtualSubgrid.toStringUNew() << std::endl;
//    }

    if(
      subgridOverlapsVirtualSubgrid &&
      (
        // Restrict if virtual subgrid is coarsening or if the data on the virtual subgrid is required for timestepping
        virtualSubgrid.willCoarsen()
        || (!areAllCoarseSubgridsBlocked && isOverlappedByCoarseGhostlayers)
        //TODO unterweg dissertation: Es kann sein, dass ein Nachbarsubgitter vom groben Subgitter noch nicht angekommen ist, wenn
        //das Gitter gerade verteilt wurde.
        || subgrid.getAge() < 2
        //|| true
      )
    ) {
      assertion2(virtualSubgrid.isVirtual(), subgrid.toString(), virtualSubgrid.toString());
      assertion2(!tarch::la::oneGreater(virtualSubgrid.getPosition(), subgrid.getPosition())
          && !tarch::la::oneGreater(subgrid.getPosition() + subgrid.getSize(), virtualSubgrid.getPosition() + virtualSubgrid.getSize()),
          subgrid.toString(), virtualSubgrid.toString());

      _numerics.restrict(subgrid, virtualSubgrid, !virtualSubgrid.willCoarsen());

      virtualSubgrid.getTimeIntervals().setEstimatedNextTimestepSize(
        subgrid.getTimeIntervals().getEstimatedNextTimestepSize()
      );
    }

    if(!parallelSubgrid.wasCurrentStateSent()) {
      //virtualParallelSubgrid.markCurrentStateAsSent(virtualParallelSubgrid.wasCurrentStateSent() || parallelSubgrid.wasCurrentStateSent());
      virtualParallelSubgrid.markCurrentStateAsSent(false);
    }
  }
}

void peanoclaw::interSubgridCommunication::GridLevelTransfer::finalizeVirtualSubgrid(
  Patch&                               subgrid,
  peanoclaw::Vertex * const            fineGridVertices,
  const peano::grid::VertexEnumerator& fineGridVerticesEnumerator,
  bool                                 isPeanoCellLeaf
) {
  tarch::multicore::Lock lock(_virtualPatchListSemaphore);
  assertion1(_virtualPatchDescriptionIndices.size() >= 0, subgrid.toString());

  tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> virtualSubgridKey = createVirtualSubgridKey(subgrid.getPosition(), subgrid.getLevel());
  int virtualPatchDescriptionIndex = _virtualPatchDescriptionIndices[virtualSubgridKey];
  _virtualPatchDescriptionIndices.erase(virtualSubgridKey);
//  _virtualPatchTimeConstraints.erase(virtualSubgridKey);
  CellDescription& virtualPatchDescription = CellDescriptionHeap::getInstance().getData(virtualPatchDescriptionIndex).at(0);
  Patch virtualPatch(virtualPatchDescription);

  //Assert that we're working on the correct virtual patch
  assertionEquals3(subgrid.getCellDescriptionIndex(), virtualPatchDescriptionIndex, subgrid, virtualPatch, _virtualPatchDescriptionIndices.size());
  assertionNumericalEquals(subgrid.getPosition(), virtualPatch.getPosition());
  assertionNumericalEquals(subgrid.getSize(), virtualPatch.getSize());
  assertionEquals(subgrid.getLevel(), virtualPatch.getLevel());
  assertionEquals(subgrid.getUIndex(), virtualPatch.getUIndex());
//    assertionEquals(finePatch.getUOldIndex(), virtualPatch.getUOldIndex());

  _numerics.postProcessRestriction(subgrid, !subgrid.willCoarsen());

  //Fill ghostlayer
  for(int i = 0; i < TWO_POWER_D; i++) {
    fineGridVertices[fineGridVerticesEnumerator(i)].fillAdjacentGhostLayers(
      subgrid.getLevel(),
      _useDimensionalSplitting,
      _numerics,
      fineGridVerticesEnumerator.getVertexPosition(i)
    );
  }

  //Switch to leaf or non-virtual
  if(isPeanoCellLeaf) {
    assertion1(tarch::la::greaterEquals(subgrid.getTimeIntervals().getTimestepSize(), 0.0), subgrid);
    subgrid.switchToLeaf();
    _numerics.update(subgrid);
    ParallelSubgrid parallelSubgrid(subgrid);
    parallelSubgrid.markCurrentStateAsSent(false);
  } else {
    if(!isPatchAdjacentToRemoteRank(
      fineGridVertices,
      fineGridVerticesEnumerator
    )) {
      subgrid.switchToNonVirtual();
    }
  }

  assertion1(!subgrid.isVirtual()
    || isPatchAdjacentToRemoteRank(
        fineGridVertices,
        fineGridVerticesEnumerator),
    subgrid);
}

tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> peanoclaw::interSubgridCommunication::GridLevelTransfer::createVirtualSubgridKey(
  tarch::la::Vector<DIMENSIONS, double> position,
  int                                   level
) const {
  tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> virtualSubgridKey;
  for(int d = 0; d < DIMENSIONS; d++) {
    virtualSubgridKey(d) = position(d);
  }
  virtualSubgridKey(DIMENSIONS) = level;
  return virtualSubgridKey;
}

peanoclaw::interSubgridCommunication::GridLevelTransfer::GridLevelTransfer(
  bool useDimensionalSplitting,
  peanoclaw::Numerics& numerics
) :
  _numerics(numerics),
  _maximumNumberOfSimultaneousVirtualPatches(0),
  _useDimensionalSplitting(useDimensionalSplitting)
{
}

peanoclaw::interSubgridCommunication::GridLevelTransfer::~GridLevelTransfer() {
  tarch::multicore::Lock lock(_virtualPatchListSemaphore);
  assertion1(_virtualPatchDescriptionIndices.empty(), _virtualPatchDescriptionIndices.size());

  logDebug("~GridLevelTransfer", "Maximum number of simultaneously held virtual patches: " << _maximumNumberOfSimultaneousVirtualPatches);
}

#ifdef Parallel
void peanoclaw::interSubgridCommunication::GridLevelTransfer::updatePatchStateDuringMergeWithWorker(
  int localCellDescriptionIndex,
  int remoteCellDescriptionIndex
) {
  logTraceInWith1Argument("updatePatchStateDuringMergeWithWorker", localCellDescriptionIndex);

  assertion(localCellDescriptionIndex != -1);
  CellDescription& localCellDescription = CellDescriptionHeap::getInstance().getData(localCellDescriptionIndex).at(0);
  Patch localPatch(localCellDescription);

  if(remoteCellDescriptionIndex != -1) {
    CellDescription& remoteCellDescription = CellDescriptionHeap::getInstance().getData(remoteCellDescriptionIndex).at(0);

    assertion1(localCellDescriptionIndex != -1, localPatch);
    if(remoteCellDescription.getUIndex() != -1) {
      assertion1(localPatch.isVirtual() || localPatch.isLeaf(), localPatch);

      //Delete current content of patch
      DataHeap::getInstance().deleteData(localPatch.getUIndex());

      //Merge
      localCellDescription.setUIndex(remoteCellDescription.getUIndex());
    }

    CellDescriptionHeap::getInstance().deleteData(remoteCellDescriptionIndex);
  }

  logTraceOut("updatePatchStateDuringMergeWithWorker");
}
#endif

void peanoclaw::interSubgridCommunication::GridLevelTransfer::stepDown(
  Patch*                               coarseSubgrid,
  Patch&                               fineSubgrid,
  peanoclaw::Vertex * const            fineGridVertices,
  const peano::grid::VertexEnumerator& fineGridVerticesEnumerator,
  bool                                 isInitializing,
  bool                                 isPeanoCellLeaf
) {

  //Switch to virtual subgrid if necessary
  if(shouldBecomeVirtualSubgrid(
      fineSubgrid,
      fineGridVertices,
      fineGridVerticesEnumerator,
      isInitializing,
      isPeanoCellLeaf
  )) {
    switchToAndAddVirtualSubgrid(fineSubgrid);
  } else if(fineSubgrid.isVirtual()) {
    //Switch to non-virtual if still virtual
    fineSubgrid.switchToNonVirtual();
  }

  //Prepare flags for subgrid
  if(!fineSubgrid.isLeaf()) {
    fineSubgrid.setWillCoarsen(peano::grid::aspects::VertexStateAnalysis::doesOneVertexCarryRefinementFlag
      (
        fineGridVertices,
        fineGridVerticesEnumerator,
        peanoclaw::records::Vertex::Erasing
      )
    );
  }
  fineSubgrid.getTimeIntervals().resetMinimalNeighborTimeConstraint();
  fineSubgrid.getTimeIntervals().resetMaximalNeighborTimeInterval();
  fineSubgrid.resetNeighboringGhostlayerBounds();
  fineSubgrid.getTimeIntervals().resetMinimalFineGridTimeInterval();

  //Get data from neighbors:
  //  - Ghostlayers data
  //  - Ghostlayer bounds
  //  - Neighbor times
  for(int i = 0; i < TWO_POWER_D; i++) {
    assertion1(fineSubgrid.getCellDescriptionIndex() != -1, fineSubgrid);
    fineGridVertices[fineGridVerticesEnumerator(i)].setAdjacentCellDescriptionIndex(i, fineSubgrid.getCellDescriptionIndex());
    fineGridVertices[fineGridVerticesEnumerator(i)].fillAdjacentGhostLayers(
      fineGridVerticesEnumerator.getLevel(),
      _useDimensionalSplitting,
      _numerics,
      fineGridVerticesEnumerator.getVertexPosition(i),
      i
    );
  }

  //Data from coarse patch:
  // -> Update minimal time constraint of coarse neighbors
  if(coarseSubgrid != 0) {
    //Patch coarsePatch(coarseCellDescriptionIndex);
    if(coarseSubgrid->getTimeIntervals().shouldFineGridsSynchronize()) {
      //Set time constraint of fine grid to time of coarse grid to synch
      //on that time.
      fineSubgrid.getTimeIntervals().updateMinimalNeighborTimeConstraint(
        coarseSubgrid->getTimeIntervals().getCurrentTime() + coarseSubgrid->getTimeIntervals().getTimestepSize(),
        coarseSubgrid->getCellDescriptionIndex()
      );
    }
  }
}

void peanoclaw::interSubgridCommunication::GridLevelTransfer::stepUp(
  //int                                  coarseCellDescriptionIndex,
  Patch*                               coarseSubgrid,
  Patch&                               finePatch,
  ParallelSubgrid&                     fineParallelSubgrid,
  bool                                 isPeanoCellLeaf,
  peanoclaw::Vertex * const            fineGridVertices,
  const peano::grid::VertexEnumerator& fineGridVerticesEnumerator
) {

  //Correct time intervals for virtual subgrid or going-to-be virtual subgrid
  if(!finePatch.isLeaf() || !isPeanoCellLeaf) {
    finePatch.switchValuesAndTimeIntervalToMinimalFineGridTimeInterval();
    assertion1(tarch::la::greaterEquals(finePatch.getTimeIntervals().getTimestepSize(), 0) || !isPeanoCellLeaf, finePatch);
  }

  //Update fine grid time interval on next coarser patch if possible
  if(coarseSubgrid != 0) {
    //Patch coarsePatch(coarseCellDescriptionIndex);
    coarseSubgrid->getTimeIntervals().updateMinimalFineGridTimeInterval(
      finePatch.getTimeIntervals().getCurrentTime(),
      finePatch.getTimeIntervals().getTimestepSize()
    );
  }

  if(finePatch.isLeaf()) {
    restrictToOverlappingVirtualSubgrids(finePatch, fineParallelSubgrid);

    //TODO unterweg dissertation:
    //If the patch is leaf, but the Peano cell is not, it got refined.
    //Thus, the patch was not turned to a virtual patch to avoid
    //restriction to this patch, which would lead to invalid data, since
    //the patch is not initialized with zeros. So, the patch needs to
    //be switched to refined (i.e. non-virtual) here...
    if(!isPeanoCellLeaf) {
      finePatch.switchToVirtual();

      //Fill ghostlayer
      for(int i = 0; i < TWO_POWER_D; i++) {
        fineGridVertices[fineGridVerticesEnumerator(i)].fillAdjacentGhostLayers(
          finePatch.getLevel(),
          _useDimensionalSplitting,
          _numerics,
          fineGridVerticesEnumerator.getVertexPosition(i));
      }

      finePatch.switchToNonVirtual();

      ParallelSubgrid parallelSubgrid(finePatch);
      parallelSubgrid.markCurrentStateAsSent(false);
    }
  } else if (finePatch.isVirtual()) {
    finalizeVirtualSubgrid(
      finePatch,
      fineGridVertices,
      fineGridVerticesEnumerator,
      isPeanoCellLeaf
    );
  }

  //If patch wasn't refined -> look if veto for coarsening is necessary
  if(!finePatch.isLeaf()) {
    vetoCoarseningIfNecessary(
      finePatch,
      fineGridVertices,
      fineGridVerticesEnumerator
    );
  }

  //Reset time constraint for optimization of ghostlayer filling
//  finePatch.resetMinimalNeighborTimeConstraint();
}

void peanoclaw::interSubgridCommunication::GridLevelTransfer::restrictRefinementFlagsToCoarseVertices(
  peanoclaw::Vertex*        coarseGridVertices,
  const peano::grid::VertexEnumerator&      coarseGridVerticesEnumerator,
  const peanoclaw::Vertex& fineGridVertex,
  const tarch::la::Vector<DIMENSIONS,int>&                   localPositionOfHangingNode
) {
  logTraceInWith2Arguments( "restrictRefinementFlagsToCoarseVertices(...)", fineGridVertex, localPositionOfHangingNode );

  tarch::la::Vector<DIMENSIONS,int>   toCoarseGridVertex;
  for (int d=0; d<DIMENSIONS; d++) {
    if(localPositionOfHangingNode(d) < 2) {
      toCoarseGridVertex(d) = 0;
    } else {
      toCoarseGridVertex(d) = 1;
    }
  }

  #ifdef Asserts
  int toCoarseGridVertexScalar = peano::utils::dLinearised(toCoarseGridVertex, 2);
  assertion(toCoarseGridVertexScalar >= 0 && toCoarseGridVertexScalar < TWO_POWER_D);
  #endif

  logTraceOut( "restrictRefinementFlagsToCoarseVertices(...)" );
}

void peanoclaw::interSubgridCommunication::GridLevelTransfer::restrictDestroyedSubgrid(
  const Patch&                         destroyedSubgrid,
  Patch&                               coarseSubgrid,
  peanoclaw::Vertex * const            fineGridVertices,
  const peano::grid::VertexEnumerator& fineGridVerticesEnumerator
) {
  assertion2(tarch::la::greaterEquals(coarseSubgrid.getTimeIntervals().getTimestepSize(), 0.0), destroyedSubgrid, coarseSubgrid);

  //Fix timestep size
  assertion1(tarch::la::greaterEquals(coarseSubgrid.getTimeIntervals().getTimestepSize(), 0), coarseSubgrid);
  coarseSubgrid.getTimeIntervals().setTimestepSize(std::max(0.0, coarseSubgrid.getTimeIntervals().getTimestepSize()));

  //Set indices on coarse adjacent vertices
  for(int i = 0; i < TWO_POWER_D; i++) {
    fineGridVertices[fineGridVerticesEnumerator(i)].setAdjacentCellDescriptionIndex(i, coarseSubgrid.getCellDescriptionIndex());
  }

  //Skip update for coarse patch in next grid iteration
  coarseSubgrid.setSkipNextGridIteration(2);

  //Set demanded mesh width for coarse cell to coarse cell size. Otherwise
  //the coarse patch might get refined immediately.
  coarseSubgrid.setDemandedMeshWidth(coarseSubgrid.getSubcellSize());
}

