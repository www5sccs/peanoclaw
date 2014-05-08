/*
 * GhostLayerCompositor.cpp
 *
 *  Created on: Feb 16, 2012
 *      Author: Kristof Unterweger
 */

#include "peanoclaw/interSubgridCommunication/GhostLayerCompositor.h"

#include "peanoclaw/grid/SubgridAccessor.h"
#include "peanoclaw/interSubgridCommunication/DefaultTransfer.h"
#include "peanoclaw/interSubgridCommunication/Extrapolation.h"
#include "peanoclaw/interSubgridCommunication/GhostlayerCompositorFunctors.h"
#include "peanoclaw/interSubgridCommunication/aspects/FaceAdjacentPatchTraversal.h"
#include "peanoclaw/interSubgridCommunication/aspects/EdgeAdjacentPatchTraversal.h"
#include "peanoclaw/interSubgridCommunication/aspects/CornerAdjacentPatchTraversal.h"
#include "peanoclaw/Patch.h"

#include "peano/utils/Loop.h"
#include "tarch/parallel/Node.h"

tarch::logging::Log peanoclaw::interSubgridCommunication::GhostLayerCompositor::_log("peanoclaw::interSubgridCommunication::GhostLayerCompositor");

//void peanoclaw::interSubgridCommunication::GhostLayerCompositor::copyGhostLayerDataBlock(
//  const tarch::la::Vector<DIMENSIONS, int>& size,
//  const tarch::la::Vector<DIMENSIONS, int>& sourceOffset,
//  const tarch::la::Vector<DIMENSIONS, int>& destinationOffset,
//  peanoclaw::Patch& source,
//  peanoclaw::Patch& destination
//) {
//  logTraceInWith3Arguments("copyGhostLayerDataBlock", size, sourceOffset, destinationOffset);
//
//  //TODO unterweg debug
//// std::cout << "Copying from " << source << std::endl << " to " << destination << std::endl
//// << " size=" << size << " sourceOffset=" << sourceOffset << " destinationOffset=" << destinationOffset << std::endl;
//
//  assertionEquals(source.getUnknownsPerSubcell(), destination.getUnknownsPerSubcell());
//
//  double timeFactor;
//  if(source.isVirtual()) {
//    //TODO unterweg: Restricting to interval [0, 1]
//    //timeFactor = (destination.getTimeIntervals().getTimeUNew() - 0.0) / (1.0 - source.getTimeIntervals().getTimeUOld());
//    timeFactor = (destination.getTimeIntervals().getTimeUNew() - 0.0) / 1.0;
//  } else {
//    if(tarch::la::greater(source.getTimeIntervals().getTimeUNew() - source.getTimeIntervals().getTimeUOld(), 0.0)) {
//      timeFactor = (destination.getTimeIntervals().getTimeUNew() - source.getTimeIntervals().getTimeUOld()) / (source.getTimeIntervals().getTimeUNew() - source.getTimeIntervals().getTimeUOld());
//    } else {
//      timeFactor = 1.0;
//    }
//  }
//
//  int sourceUnknownsPerSubcell = source.getUnknownsPerSubcell();
//  dfor(subcellindex, size) {
//    int linearSourceUNewIndex = source.getLinearIndexUNew(subcellindex + sourceOffset);
//    int linearSourceUOldIndex = source.getLinearIndexUOld(subcellindex + sourceOffset);
//    int linearDestinationUOldIndex = destination.getLinearIndexUOld(subcellindex + destinationOffset);
//
//    //TODO unterweg debug
//// std::cout << "Copying " << (subcellindex + sourceOffset) << std::endl;
//
//    for(int unknown = 0; unknown < sourceUnknownsPerSubcell; unknown++) {
//      double valueUNew = source.getValueUNew(linearSourceUNewIndex, unknown);
//      double valueUOld = source.getValueUOld(linearSourceUOldIndex, unknown);
//
//      double value = valueUNew * timeFactor + valueUOld * (1.0 - timeFactor);
//
//      destination.setValueUOld(linearDestinationUOldIndex, unknown, value);
//
//      logDebug("copyGhostLayerDataBlock(...)", "Copied cell " << (subcellindex+sourceOffset) << " with value " << value << " to " << (subcellindex+destinationOffset));
//    }
//  }
//
//#if defined(Asserts) && defined(AssertForPositiveValues)
//  dfor(subcellIndex, size) {
//    tarch::la::Vector<DIMENSIONS, int> subcellIndexInDestinationPatch = subcellIndex + destinationOffset;
//    if(destination.getValueUOld(subcellIndexInDestinationPatch, 0) < 0.0) {
//      std::cout << "Invalid copy "
//          << " sourceOffset=" << sourceOffset << ", size=" << size << " "
//#ifdef Parallel
//          << "on rank " << tarch::parallel::Node::getInstance().getRank() << " "
//#endif
//          << "from patch " << std::endl << source.toString() << std::endl << source.toStringUNew() << std::endl << source.toStringUOldWithGhostLayer()
//          << std::endl << "to patch" << std::endl << destination.toString() << std::endl << destination.toStringUNew() << std::endl << destination.toStringUOldWithGhostLayer()
//          << std::endl << "value=" << destination.getValueUOld(subcellIndexInDestinationPatch, 0) << std::endl;
//      assertion2(false, subcellIndexInDestinationPatch, destination.getValueUOld(subcellIndexInDestinationPatch, 0));
//      throw "";
//    }
//  }
//#endif
//
//  logTraceOut("copyGhostLayerDataBlock");
//}

//void peanoclaw::interSubgridCommunication::GhostLayerCompositor::copyGhostLayerDataBlock(
//  const tarch::la::Vector<DIMENSIONS, int>& size,
//  const tarch::la::Vector<DIMENSIONS, int>& sourceOffset,
//  const tarch::la::Vector<DIMENSIONS, int>& destinationOffset,
//  peanoclaw::Patch& source,
//  peanoclaw::Patch& destination
//) {
//  assertionEquals(source.getUnknownsPerSubcell(), destination.getUnknownsPerSubcell());
//
//  peanoclaw::interSubgridCommunication::DefaultTransfer transfer;
//  transfer.transfer(size, sourceOffset, destinationOffset, source, destination);
//}

bool peanoclaw::interSubgridCommunication::GhostLayerCompositor::shouldTransferGhostlayerData(Patch& source, Patch& destination) {
  if(!source.isValid()) {
    return false;
  }
  bool sourceHoldsGridData = (source.isVirtual() || source.isLeaf());
  return (destination.isValid() && destination.isLeaf())
            #ifdef Parallel
            && !destination.isRemote()
            #endif
            && (
                (
                 sourceHoldsGridData
                 && !tarch::la::greater(destination.getTimeIntervals().getCurrentTime() + destination.getTimeIntervals().getTimestepSize(), source.getTimeIntervals().getCurrentTime() + source.getTimeIntervals().getTimestepSize())
                )
                || (source.isLeaf() && !destination.getTimeIntervals().isBlockedByNeighbors())
            );
}

double peanoclaw::interSubgridCommunication::GhostLayerCompositor::fillGhostlayerManifolds(
  int destinationSubgridIndex,
  bool fillFromNeighbor,
  int dimensionality
) {
  Extrapolation extrapolation(_patches[destinationSubgridIndex]);
  double maximumLinearError = 0.0;
  if(dimensionality == DIMENSIONS - 1) {
    //Faces
    FillGhostlayerFaceFunctor faceFunctor(
      *this,
      destinationSubgridIndex
    );
    peanoclaw::interSubgridCommunication::aspects::FaceAdjacentPatchTraversal<FillGhostlayerFaceFunctor>(
        _patches,
        faceFunctor
    );
  } else if(dimensionality == DIMENSIONS - 2) {
    //Edges
    FillGhostlayerEdgeFunctor edgeFunctor(
      *this,
      extrapolation,
      fillFromNeighbor,
      destinationSubgridIndex
    );
    peanoclaw::interSubgridCommunication::aspects::EdgeAdjacentPatchTraversal<FillGhostlayerEdgeFunctor>(
        _patches,
        edgeFunctor
    );
    maximumLinearError = edgeFunctor.getMaximumLinearError();
  } else if(dimensionality == DIMENSIONS - 3) {
    #ifdef Dim3
    //Corners
    FillGhostlayerCornerFunctor cornerFunctor(
      *this,
      extrapolation,
      fillFromNeighbor,
      destinationSubgridIndex
    );
    peanoclaw::interSubgridCommunication::aspects::CornerAdjacentPatchTraversal<FillGhostlayerCornerFunctor>(
        _patches,
        cornerFunctor
    );
    maximumLinearError = cornerFunctor.getMaximumLinearError();
    #else
    assertionFail("Only valid for 3D!");
    #endif
  }
  return maximumLinearError;
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::updateNeighborTimeForManifolds(
  int destinationSubgridIndex,
  int dimensionality
) {
  UpdateNeighborTimeFunctor functor(
    *this
  );

  if(dimensionality == DIMENSIONS - 1) {
    //Faces
    peanoclaw::interSubgridCommunication::aspects::FaceAdjacentPatchTraversal<UpdateNeighborTimeFunctor>(
        _patches,
        functor
    );
  } else if(dimensionality == DIMENSIONS - 2) {
    //Edges
    peanoclaw::interSubgridCommunication::aspects::EdgeAdjacentPatchTraversal<UpdateNeighborTimeFunctor>(
        _patches,
        functor
    );
  } else if(dimensionality == DIMENSIONS - 3) {
    #ifdef Dim3
    //Corners
    peanoclaw::interSubgridCommunication::aspects::CornerAdjacentPatchTraversal<UpdateNeighborTimeFunctor>(
        _patches,
        functor
    );
    #else
    assertionFail("Only valid for 3D!");
    #endif
  }
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::fillOrExtrapolateGhostlayerAndUpdateNeighborTime(
  int destinationSubgridIndex
) {
  int extrapolationUpToDimension = DIMENSIONS-2;
  int fillingUpToDimension = DIMENSIONS-1;
  Extrapolation extrapolation(_patches[destinationSubgridIndex]);

  double maximalGradient;
  do {
    maximalGradient = 0.0;

    //Fill from neighbors
    for(int dimensionality = extrapolationUpToDimension + 1; dimensionality <= fillingUpToDimension; dimensionality++) {
      fillGhostlayerManifolds(destinationSubgridIndex, true, dimensionality);
    }
    fillingUpToDimension = extrapolationUpToDimension;

    //Extrapolate from ghostlayer
    for(int dimensionality = extrapolationUpToDimension; dimensionality >= 0 ; dimensionality--) {
      double maximalGradientForDimensionality = fillGhostlayerManifolds(destinationSubgridIndex, false, dimensionality);
      maximalGradient = std::max(maximalGradient, maximalGradientForDimensionality);
    }

    extrapolationUpToDimension--;
    assertion1(extrapolationUpToDimension >= -2, extrapolationUpToDimension);
  } while(maximalGradient > 0.1/(extrapolationUpToDimension+2));

  //Update neighbor time
  for(int dimensionality = extrapolationUpToDimension + 2; dimensionality < DIMENSIONS; dimensionality++) {
    updateNeighborTimeForManifolds(destinationSubgridIndex, dimensionality);
  }
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
  _useDimensionalSplittingOptimization(useDimensionalSplittingOptimization)
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
  double neighborGhostlayerWidth = (double)(_patches[neighborPatchIndex].getGhostlayerWidth());
  tarch::la::Vector<DIMENSIONS, double> neighborSubcellSize = _patches[neighborPatchIndex].getSubcellSize();
  tarch::la::Vector<DIMENSIONS, double> lowerBounds
    = _patches[neighborPatchIndex].getPosition()
      - neighborGhostlayerWidth * neighborSubcellSize;
  tarch::la::Vector<DIMENSIONS, double> upperBounds
    = _patches[neighborPatchIndex].getPosition() + _patches[neighborPatchIndex].getSize()
      + neighborGhostlayerWidth * neighborSubcellSize;

  bool hasToUpdate = true;
  tarch::la::Vector<DIMENSIONS, double> upperNeighboringGhostlayerBounds = _patches[updatedPatchIndex].getUpperNeighboringGhostlayerBounds();
  tarch::la::Vector<DIMENSIONS, double> lowerNeighboringGhostlayerBounds = _patches[updatedPatchIndex].getLowerNeighboringGhostlayerBounds();
  for(int d = 0; d < DIMENSIONS; d++) {
    if( direction(d) > 0 ) {
      //Check upper bounds
      hasToUpdate &= (upperNeighboringGhostlayerBounds(d) < upperBounds(d));
    } else if( direction(d) < 0 ) {
      //Check lower bounds
      hasToUpdate &= (lowerNeighboringGhostlayerBounds(d) > lowerBounds(d));
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
    - _patches[neighborPatchIndex].getGhostlayerWidth() * _patches[neighborPatchIndex].getSubcellSize()(dimension)
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
    + _patches[neighborPatchIndex].getGhostlayerWidth() * _patches[neighborPatchIndex].getSubcellSize()(dimension)
  );

  logDebug("updateUpperGhostlayerBound", "Updating upper ghostlayer from patch " << _patches[neighborPatchIndex]
          << " to patch " << _patches[updatedPatchIndex] << ", new upper ghostlayer bound is "
          << _patches[updatedPatchIndex].getUpperNeighboringGhostlayerBounds()(dimension));
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::updateNeighborTime(int updatedPatchIndex, int neighborPatchIndex) {
  if(_patches[updatedPatchIndex].isValid() && _patches[neighborPatchIndex].isValid()) {
    double neighborTimeConstraint = _patches[neighborPatchIndex].getTimeIntervals().getTimeConstraint();

    _patches[updatedPatchIndex].getTimeIntervals().updateMinimalNeighborTimeConstraint(
      neighborTimeConstraint,
      _patches[neighborPatchIndex].getCellDescriptionIndex()
    );
    _patches[updatedPatchIndex].getTimeIntervals().updateMaximalNeighborTimeInterval(
      _patches[neighborPatchIndex].getTimeIntervals().getCurrentTime(),
      _patches[neighborPatchIndex].getTimeIntervals().getTimestepSize()
    );

    if(_patches[neighborPatchIndex].isLeaf()) {
      _patches[updatedPatchIndex].getTimeIntervals().updateMinimalLeafNeighborTimeConstraint(neighborTimeConstraint);
    }
  }
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::fillGhostLayersAndUpdateNeighborTimes(
  int destinationSubgridIndex
) {
  logTraceIn("fillGhostLayersAndUpdateNeighborTimes(int)");

  if(_useDimensionalSplittingOptimization) {
    if(destinationSubgridIndex == -1) {
      for(int i = 0; i < TWO_POWER_D; i++) {
        if(_patches[i].isValid() && _patches[i].isLeaf()) {
          fillOrExtrapolateGhostlayerAndUpdateNeighborTime(i);
        }
      }
    } else {
      if(_patches[destinationSubgridIndex].isValid()
          && (_patches[destinationSubgridIndex].isLeaf() || _patches[destinationSubgridIndex].isVirtual())) {
        fillOrExtrapolateGhostlayerAndUpdateNeighborTime(destinationSubgridIndex);
      }
    }
  } else {
    for(int dimensionality = 0; dimensionality < DIMENSIONS; dimensionality++) {
      fillGhostlayerManifolds(destinationSubgridIndex, true, dimensionality);
      updateNeighborTimeForManifolds(destinationSubgridIndex, dimensionality);
    }
  }
  logTraceOut("fillGhostLayersAndUpdateNeighborTimes(int)");
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::updateGhostlayerBounds() {
  //Faces
  UpdateGhostlayerBoundsFaceFunctor faceFunctor(*this);
  peanoclaw::interSubgridCommunication::aspects::FaceAdjacentPatchTraversal<UpdateGhostlayerBoundsFaceFunctor>(
    _patches,
    faceFunctor
  );

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

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::applyFluxCorrection() {
  FluxCorrectionFunctor functor(_numerics);
  peanoclaw::interSubgridCommunication::aspects::FaceAdjacentPatchTraversal<FluxCorrectionFunctor>(
    _patches,
    functor
  );
}
