/*
 * GridLevelTransfer.cpp
 *
 *  Created on: Mar 19, 2012
 *      Author: Kristof Unterweger
 */
#include "peanoclaw/interSubgridCommunication/GridLevelTransfer.h"

#include <limits>

#include "peanoclaw/Patch.h"
#include "peanoclaw/Numerics.h"
#include "peanoclaw/Vertex.h"
#include "peano/grid/VertexEnumerator.h"
#include "peano/heap/Heap.h"
#include "peano/grid/aspects/VertexStateAnalysis.h"

#include "peano/utils/Loop.h"

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
  assertion1(_virtualPatchDescriptionIndices.empty(), _virtualPatchDescriptionIndices.size());

  logDebug("~GridLevelTransfer", "Maximum number of simultaneosly held virtual patches: " << _maximumNumberOfSimultaneousVirtualPatches);
}

//void peanoclaw::interSubgridCommunication::GridLevelTransfer::updatePatchStateDuringForkOrJoin(
//  Patch& finePatch,
//  bool   needsToHoldGridData
//) {
//  logTraceInWith1Argument("updatePatchStateDuringForkOrJoin", finePatch);
//
//  if(needsToHoldGridData && !finePatch.isVirtual() && !finePatch.isLeaf()) {
//    //Push virtual stack
//    _virtualPatchDescriptionIndices.push_back(finePatch.getCellDescriptionIndex());
//    _virtualPatchTimeConstraints.push_back(finePatch.getMinimalNeighborTimeConstraint());
//
//    if(static_cast<int>(_virtualPatchDescriptionIndices.size()) > _maximumNumberOfSimultaneousVirtualPatches) {
//      _maximumNumberOfSimultaneousVirtualPatches = _virtualPatchDescriptionIndices.size();
//    }
//
//    //Create virtual patch
//    finePatch.switchToVirtual();
//  }
//
//  logTraceOut("updatePatchStateDuringForkOrJoin");
//}
//
//void peanoclaw::interSubgridCommunication::GridLevelTransfer::updatePatchStateBeforeStepDown(
//  Patch& finePatch,
//  bool   needsToHoldGridData
//) {
//  logTraceInWith1Argument("updatePatchStateBeforeStepDown", finePatch);
//
//  updatePatchStateDuringForkOrJoin(
//    finePatch,
//    needsToHoldGridData
//  );
//  finePatch.increaseNumberOfPatchStateUpdate();
//
//  //TODO unterweg debug
//  if(tarch::la::equals(finePatch.getSize(), 1.0/3.0)
//    && tarch::la::equals(finePatch.getPosition()(0), 0.0)
//    && tarch::la::equals(finePatch.getPosition()(1), 0.0)) {
//    #ifdef Parallel
//    std::cout << "Updated patch (parallel) on rank " << tarch::parallel::Node::getInstance().getRank() << ": " << finePatch << std::endl;
//    #endif
//  }
//
//  logTraceOut("updatePatchStateBeforeStepDown");
//}

void peanoclaw::interSubgridCommunication::GridLevelTransfer::updatePatchStateBeforeStepDown(
  Patch&                               finePatch,
  peanoclaw::Vertex * const            fineGridVertices,
  const peano::grid::VertexEnumerator& fineGridVerticesEnumerator,
  bool                                 isInitializing,
  bool                                 isInvolvedInFork
) {
  logTraceInWith1Argument("updatePatchStateBeforeStepDown", finePatch);

  assertion1(finePatch.getNumberOfPatchStateUpdates() >= 0, finePatch);

  if(!finePatch.isVirtual()) {
    //Check wether the patch should become a virtual patch
    bool createVirtualPatch = false;
    if(!finePatch.isLeaf()) {
      if(peano::grid::aspects::VertexStateAnalysis::doesOneVertexCarryRefinementFlag
        (
          fineGridVertices,
          fineGridVerticesEnumerator,
          peanoclaw::records::Vertex::Unrefined
        )
      ) {
        createVirtualPatch = true;
      }
      if(!isInitializing
        //&& finePatch.isLeaf()
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
//        if(finePatch.getLevel() > 1) {
//          createVirtualPatch = true;
//        }

      finePatch.setWillCoarsen(peano::grid::aspects::VertexStateAnalysis::doesOneVertexCarryRefinementFlag
                              (
                                fineGridVertices,
                                fineGridVerticesEnumerator,
                                peanoclaw::records::Vertex::Erasing
                              )
        );
      }

    if(createVirtualPatch) {
      //Push virtual stack
      _virtualPatchDescriptionIndices.push_back(finePatch.getCellDescriptionIndex());
      _virtualPatchTimeConstraints.push_back(finePatch.getMinimalNeighborTimeConstraint());

      if(static_cast<int>(_virtualPatchDescriptionIndices.size()) > _maximumNumberOfSimultaneousVirtualPatches) {
        _maximumNumberOfSimultaneousVirtualPatches = _virtualPatchDescriptionIndices.size();
      }

      //Create virtual patch
      finePatch.switchToVirtual();
    }
  }

  finePatch.increaseNumberOfPatchStateUpdate();
  logTraceOut("updatePatchStateBeforeStepDown");
}

void peanoclaw::interSubgridCommunication::GridLevelTransfer::stepDown(
  int                                  coarseCellDescriptionIndex,
  Patch&                               finePatch,
  peanoclaw::Vertex * const            fineGridVertices,
  const peano::grid::VertexEnumerator& fineGridVerticesEnumerator
) {
  //TODO unterweg debug
//  assertion(!finePatch.isVirtual() || finePatch.getNumberOfPatchStateUpdates() > 0);

  finePatch.resetMinimalNeighborTimeConstraint();
  finePatch.resetMaximalNeighborTimeInterval();
  finePatch.resetNeighboringGhostlayerBounds();
  finePatch.resetMinimalFineGridTimeInterval();

  //Get data from neighbors:
  //  - Ghostlayers data
  //  - Ghostlayer bounds
  //  - Neighbor times
  for(int i = 0; i < TWO_POWER_D; i++) {
    fineGridVertices[fineGridVerticesEnumerator(i)].setAdjacentCellDescriptionIndex(i, finePatch.getCellDescriptionIndex());
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
  //TODO Kann man die Root-Zelle richtig setzen? Dann kann man diese Abfrage rausschmeissen.
//  if(fineGridVerticesEnumerator.getLevel() > 1) {
  if(coarseCellDescriptionIndex > -1) {
    CellDescription& coarsePatchDescription = peano::heap::Heap<CellDescription>::getInstance().getData(coarseCellDescriptionIndex).at(0);
    Patch coarsePatch(coarsePatchDescription);
    if(coarsePatch.shouldFineGridsSynchronize()) {
      //Set time constraint of fine grid to time of coarse grid to synch
      //on that time.
      finePatch.updateMinimalNeighborTimeConstraint(coarsePatch.getCurrentTime(), coarsePatch.getCellDescriptionIndex());
    }
  }
}

void peanoclaw::interSubgridCommunication::GridLevelTransfer::stepUp(
  int                                                         coarseCellDescriptionIndex,
  Patch&                                                      finePatch,
  bool                                                        isPeanoCellLeaf,
  peanoclaw::Vertex * const fineGridVertices,
  const peano::grid::VertexEnumerator&       fineGridVerticesEnumerator
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
    CellDescription& coarsePatchDescription = peano::heap::Heap<CellDescription>::getInstance().getData(coarseCellDescriptionIndex).at(0);
    Patch coarsePatch(coarsePatchDescription);
    coarsePatch.updateMinimalFineGridTimeInterval(finePatch.getCurrentTime(), finePatch.getTimestepSize());
  }

  if(finePatch.isLeaf()) {
    //Restrict to all
    for(int i = 0;  i < (int)_virtualPatchDescriptionIndices.size(); i++) {
      int virtualPatchDescriptionIndex = _virtualPatchDescriptionIndices[i];
      CellDescription& virtualPatchDescription = peano::heap::Heap<CellDescription>::getInstance().getData(virtualPatchDescriptionIndex).at(0);
      Patch virtualPatch(virtualPatchDescription);
      if(
//          true
          // Restrict if virtual patch is coarsening
          virtualPatch.willCoarsen()
          #ifdef Dim2
          || (
            // Restrict only if coarse patches can advance in time
            (tarch::la::greaterEquals(finePatch.getCurrentTime() + finePatch.getTimestepSize(), virtualPatch.getMinimalLeafNeighborTimeConstraint()))
          &&
            // Restrict only if this patch is overlapped by neighboring ghostlayers
            (tarch::la::oneGreater(virtualPatch.getUpperNeighboringGhostlayerBounds(), finePatch.getPosition())
            || tarch::la::oneGreater(finePatch.getPosition() + finePatch.getSize(), virtualPatch.getLowerNeighboringGhostlayerBounds()))
          )
          #else
           || true
          #endif
      ) {
        assertion2(virtualPatch.isVirtual(), finePatch.toString(), virtualPatch.toString());
        assertion2(!tarch::la::oneGreater(virtualPatch.getPosition(), finePatch.getPosition())
            && !tarch::la::oneGreater(finePatch.getPosition() + finePatch.getSize(), virtualPatch.getPosition() + virtualPatch.getSize()),
            finePatch.toString(), virtualPatch.toString());

        _numerics.restrict(finePatch, virtualPatch, !virtualPatch.willCoarsen());

        virtualPatch.setEstimatedNextTimestepSize(finePatch.getEstimatedNextTimestepSize());
      }
    }

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
    assertion1(_virtualPatchDescriptionIndices.size() > 0, finePatch.toString());

    int virtualPatchDescriptionIndex = _virtualPatchDescriptionIndices[_virtualPatchDescriptionIndices.size()-1];
    _virtualPatchDescriptionIndices.pop_back();
    _virtualPatchTimeConstraints.pop_back();
    CellDescription& virtualPatchDescription = peano::heap::Heap<CellDescription>::getInstance().getData(virtualPatchDescriptionIndex).at(0);
    Patch virtualPatch(virtualPatchDescription);

    //Assert that we're working on the correct virtual patch
    assertionEquals(finePatch.getCellDescriptionIndex(), virtualPatchDescriptionIndex);
    assertionNumericalEquals(finePatch.getPosition(), virtualPatch.getPosition());
    assertionNumericalEquals(finePatch.getSize(), virtualPatch.getSize());
    assertionEquals(finePatch.getLevel(), virtualPatch.getLevel());
    assertionEquals(finePatch.getUNewIndex(), virtualPatch.getUNewIndex());
    assertionEquals(finePatch.getUOldIndex(), virtualPatch.getUOldIndex());

    //Fill ghostlayer
    for(int i = 0; i < TWO_POWER_D; i++) {
      fineGridVertices[fineGridVerticesEnumerator(i)].fillAdjacentGhostLayers(
        finePatch.getLevel(),
        _useDimensionalSplitting,
        _numerics,
        fineGridVerticesEnumerator.getVertexPosition(i));
    }

    //Switch to leaf or non-virtual
    //TODO unterweg: move to updatePatchStateAfterStepUp
    if(isPeanoCellLeaf) {
      assertion1(tarch::la::greaterEquals(finePatch.getTimestepSize(), 0.0), finePatch);
      finePatch.switchToLeaf();
    } else {
      finePatch.switchToNonVirtual();
    }
    assertion(!finePatch.isVirtual());
  }

  //Reset time constraint for optimization of ghostlayer filling
  finePatch.resetMinimalNeighborTimeConstraint();

  assertion(!finePatch.isVirtual());
}

//void peanoclaw::interSubgridCommunication::GridLevelTransfer::updatePatchStateAfterStepUp(
//  Patch& finePatch,
//  bool   isPeanoCellLeaf
//) {
//  finePatch.decreaseNumberOfPatchStateUpdate();
//  assertion1(finePatch.getNumberOfPatchStateUpdates() >= 0, finePatch);
//
//  if(finePatch.getNumberOfPatchStateUpdates() == 0 && finePatch.isVirtual()) {
//    //Switch to leaf or non-virtual
//    if(isPeanoCellLeaf) {
//      assertion1(finePatch.getTimestepSize() >= 0.0, finePatch);
//      finePatch.switchToLeaf();
//    } else {
//      finePatch.switchToNonVirtual();
//    }
//    assertion(!finePatch.isVirtual());
//  }
//}

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


