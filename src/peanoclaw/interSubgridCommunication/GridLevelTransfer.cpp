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

  if(tarch::la::smaller(patch.getTimestepSize(), 0.0)) {
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
      patch.setFineGridsSynchronize(true);
    }
  } else {
    patch.setFineGridsSynchronize(false);
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
  bool                                 isInitializing
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

  return createVirtualPatch;
}

void peanoclaw::interSubgridCommunication::GridLevelTransfer::switchToAndAddVirtualSubgrid(
  Patch& subgrid
) {
  tarch::multicore::Lock lock(_virtualPatchListSemaphore);

  //Push virtual stack
  _virtualPatchDescriptionIndices.push_back(subgrid.getCellDescriptionIndex());
  _virtualPatchTimeConstraints.push_back(subgrid.getMinimalNeighborTimeConstraint());

  if(static_cast<int>(_virtualPatchDescriptionIndices.size()) > _maximumNumberOfSimultaneousVirtualPatches) {
    _maximumNumberOfSimultaneousVirtualPatches = _virtualPatchDescriptionIndices.size();
  }

  //Create virtual patch
  if(subgrid.isVirtual()) {
    subgrid.clearRegion(
      tarch::la::Vector<DIMENSIONS, int>(0),
      subgrid.getSubdivisionFactor(),
      false
    );
    subgrid.clearRegion(
      tarch::la::Vector<DIMENSIONS, int>(0),
      subgrid.getSubdivisionFactor(),
      true
    );
  } else {
    subgrid.switchToVirtual();
  }
}

void peanoclaw::interSubgridCommunication::GridLevelTransfer::restrictToAllVirtualSubgrids(
  const Patch& fineSubgrid
) {
  tarch::multicore::Lock lock(_virtualPatchListSemaphore);

  //Restrict to all
  for(int i = 0;  i < (int)_virtualPatchDescriptionIndices.size(); i++) {
    int virtualSubgridDescriptionIndex = _virtualPatchDescriptionIndices[i];
    CellDescription& virtualSubgridDescription = CellDescriptionHeap::getInstance().getData(virtualSubgridDescriptionIndex).at(0);
    Patch virtualSubgrid(virtualSubgridDescription);
    if(
//          true
        // Restrict if virtual patch is coarsening
        virtualSubgrid.willCoarsen()
        #ifdef Dim2
        || (
          // Restrict only if coarse patches can advance in time
          (tarch::la::greaterEquals(fineSubgrid.getCurrentTime() + fineSubgrid.getTimestepSize(), virtualSubgrid.getMinimalLeafNeighborTimeConstraint()))
        &&
          // Restrict only if this patch is overlapped by neighboring ghostlayers
          (tarch::la::oneGreater(virtualSubgrid.getUpperNeighboringGhostlayerBounds(), fineSubgrid.getPosition())
          || tarch::la::oneGreater(fineSubgrid.getPosition() + fineSubgrid.getSize(), virtualSubgrid.getLowerNeighboringGhostlayerBounds()))
        )
        #else
         || true
        #endif
    ) {
      assertion2(virtualSubgrid.isVirtual(), fineSubgrid.toString(), virtualSubgrid.toString());
      assertion2(!tarch::la::oneGreater(virtualSubgrid.getPosition(), fineSubgrid.getPosition())
          && !tarch::la::oneGreater(fineSubgrid.getPosition() + fineSubgrid.getSize(), virtualSubgrid.getPosition() + virtualSubgrid.getSize()),
          fineSubgrid.toString(), virtualSubgrid.toString());

      _numerics.restrict(fineSubgrid, virtualSubgrid, !virtualSubgrid.willCoarsen());

      virtualSubgrid.setEstimatedNextTimestepSize(fineSubgrid.getEstimatedNextTimestepSize());
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
  assertion1(_virtualPatchDescriptionIndices.size() > 0, subgrid.toString());

  int virtualPatchDescriptionIndex = _virtualPatchDescriptionIndices[_virtualPatchDescriptionIndices.size()-1];
  _virtualPatchDescriptionIndices.pop_back();
  _virtualPatchTimeConstraints.pop_back();
  CellDescription& virtualPatchDescription = CellDescriptionHeap::getInstance().getData(virtualPatchDescriptionIndex).at(0);
  Patch virtualPatch(virtualPatchDescription);

  //Assert that we're working on the correct virtual patch
  assertionEquals3(subgrid.getCellDescriptionIndex(), virtualPatchDescriptionIndex, subgrid, virtualPatch, _virtualPatchDescriptionIndices.size());
  assertionNumericalEquals(subgrid.getPosition(), virtualPatch.getPosition());
  assertionNumericalEquals(subgrid.getSize(), virtualPatch.getSize());
  assertionEquals(subgrid.getLevel(), virtualPatch.getLevel());
  assertionEquals(subgrid.getUNewIndex(), virtualPatch.getUNewIndex());
//    assertionEquals(finePatch.getUOldIndex(), virtualPatch.getUOldIndex());

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
    assertion1(tarch::la::greaterEquals(subgrid.getTimestepSize(), 0.0), subgrid);
    subgrid.switchToLeaf();
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
    if(remoteCellDescription.getUNewIndex() != -1) {
      assertion1(localPatch.isVirtual() || localPatch.isLeaf(), localPatch);

      //Delete current content of patch
      DataHeap::getInstance().deleteData(localPatch.getUNewIndex());
//      DataHeap::getInstance().deleteData(localPatch.getUOldIndex());
//      if(localPatch.getAuxIndex() != -1) {
//        DataHeap::getInstance().deleteData(localPatch.getAuxIndex());
//      }

      //Merge
      localCellDescription.setUNewIndex(remoteCellDescription.getUNewIndex());
//      localCellDescription.setUOldIndex(remoteCellDescription.getUOldIndex());
//      localCellDescription.setAuxIndex(remoteCellDescription.getAuxIndex());
    }

    CellDescriptionHeap::getInstance().deleteData(remoteCellDescriptionIndex);
  }

  logTraceOut("updatePatchStateDuringMergeWithWorker");
}
#endif

void peanoclaw::interSubgridCommunication::GridLevelTransfer::stepDown(
  int                                  coarseCellDescriptionIndex,
  Patch&                               fineSubgrid,
  peanoclaw::Vertex * const            fineGridVertices,
  const peano::grid::VertexEnumerator& fineGridVerticesEnumerator,
  bool                                 isInitializing
) {

  //Switch to virtual subgrid if necessary
  if(shouldBecomeVirtualSubgrid(
      fineSubgrid,
      fineGridVertices,
      fineGridVerticesEnumerator,
      isInitializing
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
  fineSubgrid.resetMinimalNeighborTimeConstraint();
  fineSubgrid.resetMaximalNeighborTimeInterval();
  fineSubgrid.resetNeighboringGhostlayerBounds();
  fineSubgrid.resetMinimalFineGridTimeInterval();

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
  if(coarseCellDescriptionIndex > -1) {
    CellDescription& coarsePatchDescription = CellDescriptionHeap::getInstance().getData(coarseCellDescriptionIndex).at(0);
    Patch coarsePatch(coarsePatchDescription);
    if(coarsePatch.shouldFineGridsSynchronize()) {
      //Set time constraint of fine grid to time of coarse grid to synch
      //on that time.
      fineSubgrid.updateMinimalNeighborTimeConstraint(coarsePatch.getCurrentTime(), coarsePatch.getCellDescriptionIndex());
    }
  }
}

void peanoclaw::interSubgridCommunication::GridLevelTransfer::stepUp(
  int                                  coarseCellDescriptionIndex,
  Patch&                               finePatch,
  bool                                 isPeanoCellLeaf,
  peanoclaw::Vertex * const            fineGridVertices,
  const peano::grid::VertexEnumerator& fineGridVerticesEnumerator
) {

  if(!finePatch.isLeaf() || !isPeanoCellLeaf) {
    finePatch.switchValuesAndTimeIntervalToMinimalFineGridTimeInterval();
    assertion1(tarch::la::greaterEquals(finePatch.getTimestepSize(), 0) || !isPeanoCellLeaf, finePatch);
  }

  if(!isPeanoCellLeaf) {
    //Avoid to refine immediately after coarsening
    finePatch.setDemandedMeshWidth(finePatch.getSubcellSize()(0));

    if(!finePatch.isLeaf()) {
      //If patch wasn't refined -> look if veto for coarsening is necessary
      vetoCoarseningIfNecessary(
        finePatch,
        fineGridVertices,
        fineGridVerticesEnumerator
      );
    }
  }

  //Update fine grid time interval on next coarser patch if possible
  if(coarseCellDescriptionIndex > 0) {
    CellDescription& coarsePatchDescription = CellDescriptionHeap::getInstance().getData(coarseCellDescriptionIndex).at(0);
    Patch coarsePatch(coarsePatchDescription);
    coarsePatch.updateMinimalFineGridTimeInterval(finePatch.getCurrentTime(), finePatch.getTimestepSize());
  }

  if(finePatch.isLeaf()) {
    restrictToAllVirtualSubgrids(finePatch);

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
    }
  } else if (finePatch.isVirtual()) {
    finalizeVirtualSubgrid(
      finePatch,
      fineGridVertices,
      fineGridVerticesEnumerator,
      isPeanoCellLeaf
    );
  }

  //Reset time constraint for optimization of ghostlayer filling
//  finePatch.resetMinimalNeighborTimeConstraint();
}

void peanoclaw::interSubgridCommunication::GridLevelTransfer::fillAdjacentPatchIndicesFromCoarseVertices(
  const peanoclaw::Vertex* coarseGridVertices,
  const peano::grid::VertexEnumerator&      coarseGridVerticesEnumerator,
  peanoclaw::Vertex&       fineGridVertex,
  const tarch::la::Vector<DIMENSIONS,int>&                   localPositionOfHangingNode
) {
  logTraceInWith2Arguments( "fillAdjacentPatchIndicesFromCoarseVertices(...)", fineGridVertex, localPositionOfHangingNode );

  tarch::la::Vector<DIMENSIONS,int>   fromCoarseGridVertex;
  tarch::la::Vector<DIMENSIONS,int>   coarseGridVertexAdjacentPatchIndex;

  dfor2(k)
    for (int d=0; d<DIMENSIONS; d++) {
      if (localPositionOfHangingNode(d)==0) {
        fromCoarseGridVertex(d)          = 0;
        coarseGridVertexAdjacentPatchIndex(d) = k(d);
      }
      else if (localPositionOfHangingNode(d)==3) {
        fromCoarseGridVertex(d)          = 1;
        coarseGridVertexAdjacentPatchIndex(d) = k(d);
      }
      else if (k(d)==0) {
        fromCoarseGridVertex(d)          = 0;
        coarseGridVertexAdjacentPatchIndex(d) = 1;
      }
      else {
        fromCoarseGridVertex(d)          = 1;
        coarseGridVertexAdjacentPatchIndex(d) = 0;
      }
    }
    int coarseGridVertexIndex = coarseGridVerticesEnumerator(peano::utils::dLinearised(fromCoarseGridVertex,2));
    int coarseGridVertexEntry = TWO_POWER_D_MINUS_ONE-peano::utils::dLinearised(coarseGridVertexAdjacentPatchIndex,2);
    fineGridVertex.setAdjacentCellDescriptionIndex(
      TWO_POWER_D_MINUS_ONE-kScalar,
      coarseGridVertices[coarseGridVertexIndex].getAdjacentCellDescriptionIndex(coarseGridVertexEntry)
    );
  enddforx

  logTraceOut( "fillAdjacentPatchIndicesFromCoarseVertices(...)" );
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
  assertion1(tarch::la::greaterEquals(coarseSubgrid.getTimestepSize(), 0.0), destroyedSubgrid);

  //Fix timestep size
  assertion1(tarch::la::greaterEquals(coarseSubgrid.getTimestepSize(), 0), coarseSubgrid);
  coarseSubgrid.setTimestepSize(std::max(0.0, coarseSubgrid.getTimestepSize()));

  //Set indices on coarse adjacent vertices
  for(int i = 0; i < TWO_POWER_D; i++) {
    fineGridVertices[fineGridVerticesEnumerator(i)].setAdjacentCellDescriptionIndex(i, coarseSubgrid.getCellDescriptionIndex());
  }

  //Skip update for coarse patch in next grid iteration
  coarseSubgrid.setSkipNextGridIteration(1);

  //Set demanded mesh width for coarse cell to coarse cell size. Otherwise
  //the coarse patch might get refined immediately.
  coarseSubgrid.setDemandedMeshWidth(coarseSubgrid.getSubcellSize()(0));
}
