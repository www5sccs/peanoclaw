/*
 * GhostLayerCompositor.cpp
 *
 *  Created on: Feb 16, 2012
 *      Author: Kristof Unterweger
 */

#include "peanoclaw/interSubgridCommunication/GhostLayerCompositor.h"

#include "peanoclaw/interSubgridCommunication/GhostlayerCompositorFunctors.h"
#include "peanoclaw/interSubgridCommunication/aspects/FaceAdjacentPatchTraversal.h"
#include "peanoclaw/interSubgridCommunication/aspects/EdgeAdjacentPatchTraversal.h"
#include "peanoclaw/interSubgridCommunication/aspects/CornerAdjacentPatchTraversal.h"
#include "peanoclaw/Patch.h"

#include "peano/utils/Loop.h"
#include "tarch/parallel/Node.h"

tarch::logging::Log peanoclaw::interSubgridCommunication::GhostLayerCompositor::_log("peanoclaw::interSubgridCommunication::GhostLayerCompositor");

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::copyGhostLayerDataBlock(
  const tarch::la::Vector<DIMENSIONS, int>& size,
  const tarch::la::Vector<DIMENSIONS, int>& sourceOffset,
  const tarch::la::Vector<DIMENSIONS, int>& destinationOffset,
  const peanoclaw::Patch& source,
  peanoclaw::Patch& destination
) {
  logTraceInWith3Arguments("copyGhostLayerDataBlock", size, sourceOffset, destinationOffset);

  assertionEquals(source.getUnknownsPerSubcell(), destination.getUnknownsPerSubcell());

  double timeFactor;
  if(source.isVirtual()) {
    //TODO unterweg: Restricting to interval [0, 1]
    //timeFactor = (destination.getTimeUNew() - 0.0) / (1.0 - source.getTimeUOld());
    timeFactor = (destination.getTimeUNew() - 0.0) / 1.0;
  } else {
    if(tarch::la::greater(source.getTimeUNew() - source.getTimeUOld(), 0.0)) {
      timeFactor = (destination.getTimeUNew() - source.getTimeUOld()) / (source.getTimeUNew() - source.getTimeUOld());
    } else {
      timeFactor = 1.0;
    }
  }

  int sourceUnknownsPerSubcell = source.getUnknownsPerSubcell();
  dfor(subcellindex, size) {
    int linearSourceUNewIndex = source.getLinearIndexUNew(subcellindex + sourceOffset);
    int linearSourceUOldIndex = source.getLinearIndexUOld(subcellindex + sourceOffset);
    int linearDestinationUOldIndex = destination.getLinearIndexUOld(subcellindex + destinationOffset);

    for(int unknown = 0; unknown < sourceUnknownsPerSubcell; unknown++) {
      double valueUNew = source.getValueUNew(linearSourceUNewIndex, unknown);
      double valueUOld = source.getValueUOld(linearSourceUOldIndex, unknown);

      double value = valueUNew * timeFactor + valueUOld * (1.0 - timeFactor);

      destination.setValueUOld(linearDestinationUOldIndex, unknown, value);

      logDebug("copyGhostLayerDataBlock(...)", "Copied cell " << (subcellindex+sourceOffset) << " with value " << value << " to " << (subcellindex+destinationOffset));
    }
  }

  #ifdef Asserts
  dfor(subcellIndex, size) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndexInDestinationPatch = subcellIndex + destinationOffset;
    if(destination.getValueUOld(subcellIndexInDestinationPatch, 0) < 0.0) {
      std::cout << "Invalid copy "
          #ifdef Parallel
          << "on rank " << tarch::parallel::Node::getInstance().getRank() << " "
          #endif
          << "from patch " << std::endl << source.toString() << std::endl << source.toStringUNew() << std::endl << source.toStringUOldWithGhostLayer()
          << std::endl << "to patch" << std::endl << destination.toString() << std::endl << destination.toStringUNew() << std::endl << destination.toStringUOldWithGhostLayer()
          << std::endl << "value=" << destination.getValueUOld(subcellIndexInDestinationPatch, 0) << std::endl;
      assertion2(false, subcellIndexInDestinationPatch, destination.getValueUOld(subcellIndexInDestinationPatch, 0));
      throw "";
    }
  }
  #endif

  logTraceOut("copyGhostLayerDataBlock");
}

bool peanoclaw::interSubgridCommunication::GhostLayerCompositor::shouldTransferGhostlayerData(Patch& source, Patch& destination) {
  bool sourceHoldsGridData = (source.isVirtual() || source.isLeaf());
  return destination.isLeaf()
            && (( sourceHoldsGridData
                && !tarch::la::greater(destination.getCurrentTime() + destination.getTimestepSize(), source.getCurrentTime() + source.getTimestepSize()))
            || (source.isLeaf() && destination.isAllowedToAdvanceInTime()));
}

peanoclaw::interSubgridCommunication::GhostLayerCompositor::GhostLayerCompositor(
  peanoclaw::Patch patches[TWO_POWER_D],
  int level,
  peanoclaw::Numerics& numerics,
  bool useDimensionalSplittingOptimization
) :
  _patches(patches),
  _level(level),
  _numerics(numerics),
  _useDimensionalSplittingOptimization(/*useDimensionalSplittingOptimization*/false) //TODO unterweg debug
{
}

peanoclaw::interSubgridCommunication::GhostLayerCompositor::~GhostLayerCompositor() {
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::updateGhostlayerBound(
  int updatedPatchIndex,
  int neighborPatchIndex,
  int dimension
) {
  if(neighborPatchIndex < updatedPatchIndex) {
    updateUpperGhostlayerBound(updatedPatchIndex, neighborPatchIndex, dimension);
  } else {
    updateLowerGhostlayerBound(updatedPatchIndex, neighborPatchIndex, dimension);
  }
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::updateGhostlayerBound (
  int updatedPatchIndex,
  int neighborPatchIndex,
  tarch::la::Vector<DIMENSIONS, int> direction
) {
  tarch::la::Vector<DIMENSIONS, double> lowerBounds
    = _patches[neighborPatchIndex].getPosition()
      - (double)(_patches[neighborPatchIndex].getGhostLayerWidth()) * _patches[neighborPatchIndex].getSubcellSize();
  tarch::la::Vector<DIMENSIONS, double> upperBounds
    = _patches[neighborPatchIndex].getPosition() + _patches[neighborPatchIndex].getSize()
      + (double)(_patches[neighborPatchIndex].getGhostLayerWidth()) * _patches[neighborPatchIndex].getSubcellSize();

  bool hasToUpdate = true;
  for(int d = 0; d < DIMENSIONS; d++) {
    if( direction(d) > 0 ) {
      //Check upper bounds
      hasToUpdate &= (_patches[updatedPatchIndex].getUpperNeighboringGhostlayerBounds()(d) < upperBounds(d));
    } else if( direction(d) < 0 ) {
      //Check lower bounds
      hasToUpdate &= (_patches[updatedPatchIndex].getLowerNeighboringGhostlayerBounds()(d) > lowerBounds(d));
    }
  }

  if( hasToUpdate ) {
    for(int d = 0; d < DIMENSIONS; d++) {
      if( direction(d) > 0 ) {
        _patches[updatedPatchIndex].updateUpperNeighboringGhostlayerBound(d, upperBounds(d));
        return;
      } else if( direction(d) < 0 ) {
        _patches[updatedPatchIndex].updateLowerNeighboringGhostlayerBound(d, lowerBounds(d));
        return;
      }
    }
  }
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::updateLowerGhostlayerBound(
  int updatedPatchIndex,
  int neighborPatchIndex,
  int dimension
) {
  _patches[updatedPatchIndex].updateLowerNeighboringGhostlayerBound(
    dimension,
    _patches[neighborPatchIndex].getPosition()(dimension)
    - _patches[neighborPatchIndex].getGhostLayerWidth() * _patches[neighborPatchIndex].getSubcellSize()(dimension)
  );

  logDebug("updateLowerGhostlayerBound", "Updating lower ghostlayer from patch " << _patches[neighborPatchIndex]
          << " to patch " << _patches[updatedPatchIndex] << ", new lower ghostlayer bound is "
          << _patches[updatedPatchIndex].getLowerNeighboringGhostlayerBounds()(dimension));
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::updateUpperGhostlayerBound(
  int updatedPatchIndex,
  int neighborPatchIndex,
  int dimension
) {
  _patches[updatedPatchIndex].updateUpperNeighboringGhostlayerBound(
    dimension,
    _patches[neighborPatchIndex].getPosition()(dimension) + _patches[neighborPatchIndex].getSize()(dimension)
    + _patches[neighborPatchIndex].getGhostLayerWidth() * _patches[neighborPatchIndex].getSubcellSize()(dimension)
  );

  logDebug("updateUpperGhostlayerBound", "Updating upper ghostlayer from patch " << _patches[neighborPatchIndex]
          << " to patch " << _patches[updatedPatchIndex] << ", new upper ghostlayer bound is "
          << _patches[updatedPatchIndex].getUpperNeighboringGhostlayerBounds()(dimension));
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::updateNeighborTime(int updatedPatchIndex, int neighborPatchIndex) {
  if(_patches[updatedPatchIndex].isValid() && _patches[neighborPatchIndex].isValid()) {
    _patches[updatedPatchIndex].updateMinimalNeighborTimeConstraint(
      _patches[neighborPatchIndex].getTimeConstraint(),
      _patches[neighborPatchIndex].getCellDescriptionIndex()
    );
    _patches[updatedPatchIndex].updateMaximalNeighborTimeInterval(
      _patches[neighborPatchIndex].getCurrentTime(),
      _patches[neighborPatchIndex].getTimestepSize()
    );

    if(_patches[neighborPatchIndex].isLeaf()) {
      _patches[updatedPatchIndex].updateMinimalLeafNeighborTimeConstraint(_patches[neighborPatchIndex].getTimeConstraint());
    }
  }
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::fillGhostLayers(int destinationPatchIndex) {
  logTraceIn("fillGhostLayers()");

  //Faces
  FillGhostlayerFaceFunctor faceFunctor(
    *this,
    destinationPatchIndex
  );
  peanoclaw::interSubgridCommunication::aspects::FaceAdjacentPatchTraversal<FillGhostlayerFaceFunctor>(
    _patches,
    faceFunctor
  );

  if(!_useDimensionalSplittingOptimization) {
    //Edges
    FillGhostlayerEdgeFunctor edgeFunctor(
      *this,
      destinationPatchIndex
    );
    peanoclaw::interSubgridCommunication::aspects::EdgeAdjacentPatchTraversal<FillGhostlayerEdgeFunctor>(
      _patches,
      edgeFunctor
    );

    //Corners
    #ifdef Dim3
    FillGhostlayerCornerFunctor cornerFunctor(
      *this,
      destinationPatchIndex
    );
    peanoclaw::interSubgridCommunication::aspects::CornerAdjacentPatchTraversal<FillGhostlayerCornerFunctor>(
      _patches,
      cornerFunctor
    );
    #endif
  }
  logTraceOut("fillGhostLayers()");
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::updateNeighborTimes() {

  UpdateNeighborTimeFunctor functor(
    *this
  );

  //Faces
  peanoclaw::interSubgridCommunication::aspects::FaceAdjacentPatchTraversal<UpdateNeighborTimeFunctor>(
    _patches,
    functor
  );

  if(!_useDimensionalSplittingOptimization) {
    //Edges
    peanoclaw::interSubgridCommunication::aspects::EdgeAdjacentPatchTraversal<UpdateNeighborTimeFunctor>(
      _patches,
      functor
    );

    //Corners
    #ifdef Dim3
    peanoclaw::interSubgridCommunication::aspects::CornerAdjacentPatchTraversal<UpdateNeighborTimeFunctor>(
      _patches,
      functor
    );
    #endif
  }
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::updateGhostlayerBounds() {
  //Faces
  UpdateGhostlayerBoundsFaceFunctor faceFunctor(*this);
  peanoclaw::interSubgridCommunication::aspects::FaceAdjacentPatchTraversal<UpdateGhostlayerBoundsFaceFunctor>(
    _patches,
    faceFunctor
  );

  if(!_useDimensionalSplittingOptimization) {
    //Edges
    UpdateGhostlayerBoundsEdgeFunctor edgeFunctor(*this);
    peanoclaw::interSubgridCommunication::aspects::EdgeAdjacentPatchTraversal<UpdateGhostlayerBoundsEdgeFunctor>(
      _patches,
      edgeFunctor
    );

    //Corners
    #ifdef Dim3
    UpdateGhostlayerBoundsCornerFunctor cornerFunctor(*this);
    peanoclaw::interSubgridCommunication::aspects::CornerAdjacentPatchTraversal<UpdateGhostlayerBoundsCornerFunctor>(
      _patches,
      cornerFunctor
    );
    #endif
  }
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::applyFluxCorrection() {
  FluxCorrectionFunctor functor(_numerics);
  peanoclaw::interSubgridCommunication::aspects::FaceAdjacentPatchTraversal<FluxCorrectionFunctor>(
    _patches,
    functor
  );
}
