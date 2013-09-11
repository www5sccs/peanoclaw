/*
 * GhostLayerCompositor.cpp
 *
 *  Created on: Feb 16, 2012
 *      Author: Kristof Unterweger
 */

#include "peanoclaw/interSubgridCommunication/GhostLayerCompositor.h"
#include "peanoclaw/interSubgridCommunication/aspects/FaceAdjacentPatchTraversal.h"
#include "peanoclaw/interSubgridCommunication/aspects/EdgeAdjacentPatchTraversal.h"
#include "peanoclaw/interSubgridCommunication/aspects/EdgeAdjacentPatchTraversalWithCommonFaceNeighbors.h"
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
  _useDimensionalSplittingOptimization(useDimensionalSplittingOptimization)
{
}

peanoclaw::interSubgridCommunication::GhostLayerCompositor::~GhostLayerCompositor() {
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
    peanoclaw::interSubgridCommunication::aspects::CornerAdjacentPatchTraversal<FillGhostlayerConrerFunctor>(
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
    UpdateNeighborTimeCornerFunctor cornerFunctor(
      *this
    );
    peanoclaw::interSubgridCommunication::aspects::CornerAdjacentPatchTraversal<UpdateNeighborTimeFunctor>(
      _patches,
      cornerFunctor
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
    peanoclaw::interSubgridCommunication::aspects::EdgeAdjacentPatchTraversalWithCommonFaceNeighbors<UpdateGhostlayerBoundsEdgeFunctor>(
      _patches,
      edgeFunctor
    );

    //Corners
    #ifdef Dim3
    UpdateGhostlayerBoundsCornerFunctor cornerFunctor(*this);
    peanoclaw::interSubgridCommunication::aspects::CornerAdjacentPatchTraversalWithCommonFaceAndEdgeNeighbors<UpdateGhostlayerBoundsCornerFunctor>(
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

peanoclaw::interSubgridCommunication::GhostLayerCompositor::FillGhostlayerFaceFunctor::FillGhostlayerFaceFunctor(
  GhostLayerCompositor& ghostlayerCompositor,
  int                   destinationPatchIndex
) : _ghostlayerCompositor(ghostlayerCompositor), _destinationPatchIndex(destinationPatchIndex) {
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::FillGhostlayerFaceFunctor::operator()
(
  peanoclaw::Patch&                  source,
  int                                sourceIndex,
  peanoclaw::Patch&                  destination,
  int                                destinationIndex,
  tarch::la::Vector<DIMENSIONS, int> direction
) {
  if(
    (_destinationPatchIndex == -1 || _destinationPatchIndex == destinationIndex)
    && _ghostlayerCompositor.shouldTransferGhostlayerData(source, destination)
  ) {
    int dimension = -1;
    int offset = 0;
    for(int d = 0; d < DIMENSIONS; d++) {
      if(abs(direction(d)) == 1) {
        dimension = d;
        offset = direction(d);
      }
    }
    assertion3(dimension!=-1 && offset != 0, dimension, offset, direction);

    assertionEquals2(destination.getSubdivisionFactor(), source.getSubdivisionFactor(), source, destination);
    tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = source.getSubdivisionFactor();
    tarch::la::Vector<DIMENSIONS, int> faceSize(subdivisionFactor);
    faceSize(dimension) = destination.getGhostLayerWidth();
    tarch::la::Vector<DIMENSIONS, int> destinationOffset(0);
    destinationOffset(dimension) = (offset==1) ? -destination.getGhostLayerWidth() : subdivisionFactor(dimension);

    if(source.getLevel() == destination.getLevel() && source.getLevel() == _ghostlayerCompositor._level) {
      tarch::la::Vector<DIMENSIONS, int> sourceOffset(0);
      sourceOffset(dimension)
        = (offset==1) ? (source.getSubdivisionFactor()(dimension) - source.getGhostLayerWidth()) : 0;

      _ghostlayerCompositor.copyGhostLayerDataBlock(faceSize, sourceOffset, destinationOffset, source, destination);
    } else if(source.getLevel() < destination.getLevel() && destination.getLevel() == _ghostlayerCompositor._level && source.isLeaf()) {
      _ghostlayerCompositor._numerics.interpolate(
        faceSize,
        destinationOffset,
        source,
        destination,
        true,
        false
      );
    }
  }
}

peanoclaw::interSubgridCommunication::GhostLayerCompositor::FillGhostlayerEdgeFunctor::FillGhostlayerEdgeFunctor(
  GhostLayerCompositor& ghostlayerCompositor,
  int                   destinationPatchIndex
) : _ghostlayerCompositor(ghostlayerCompositor), _destinationPatchIndex(destinationPatchIndex) {
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::FillGhostlayerEdgeFunctor::operator() (
  peanoclaw::Patch&                  source,
  int                                sourceIndex,
  peanoclaw::Patch&                  destination,
  int                                destinationIndex,
  tarch::la::Vector<DIMENSIONS, int> direction
) {
  if(
    (_destinationPatchIndex == -1 || _destinationPatchIndex == destinationIndex)
    && _ghostlayerCompositor.shouldTransferGhostlayerData(source, destination)
  ) {
    assertionEquals2(source.getSubdivisionFactor(), destination.getSubdivisionFactor(), source, destination);
    int ghostlayerWidth = destination.getGhostLayerWidth();
    tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = source.getSubdivisionFactor();
    tarch::la::Vector<DIMENSIONS, int> edgeSize(ghostlayerWidth);
    tarch::la::Vector<DIMENSIONS, int> sourceOffset(0);
    tarch::la::Vector<DIMENSIONS, int> destinationOffset(0);
    for(int d = 0; d < DIMENSIONS; d++) {
      if(direction(d) == 0) {
        edgeSize(d) = subdivisionFactor(d);
      } else if(direction(d) == 1) {
        sourceOffset(d) = subdivisionFactor(d) - ghostlayerWidth;
        destinationOffset(d) = -ghostlayerWidth;
      } else if(direction(d) == -1) {
        destinationOffset(d) = subdivisionFactor(d);
      } else {
        assertionFail("Direction " << direction << " is invalid!");
      }
    }
    if(source.getLevel() == destination.getLevel() && source.getLevel() == _ghostlayerCompositor._level) {
      _ghostlayerCompositor.copyGhostLayerDataBlock(
        edgeSize,
        sourceOffset,
        destinationOffset,
        source,
        destination
      );
    } else if(source.getLevel() < destination.getLevel() && destination.getLevel() == _ghostlayerCompositor._level && source.isLeaf()) {
      _ghostlayerCompositor._numerics.interpolate(
        edgeSize,
        destinationOffset,
        source,
        destination,
        true,
        false
      );
    }
  }
}

peanoclaw::interSubgridCommunication::GhostLayerCompositor::UpdateNeighborTimeFunctor::UpdateNeighborTimeFunctor(
  GhostLayerCompositor& ghostlayerCompositor
) : _ghostlayerCompositor(ghostlayerCompositor) {
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::UpdateNeighborTimeFunctor::operator() (
  peanoclaw::Patch&                  neighborPatch,
  int                                neighborPatchIndex,
  peanoclaw::Patch&                  updatedPatch,
  int                                updatedPatchIndexIndex,
  tarch::la::Vector<DIMENSIONS, int> direction
) {
  if(updatedPatch.isValid() && neighborPatch.isValid()) {
    updatedPatch.updateMinimalNeighborTimeConstraint(
      neighborPatch.getTimeConstraint(),
      neighborPatch.getCellDescriptionIndex()
    );
    updatedPatch.updateMaximalNeighborTimeInterval(
      neighborPatch.getCurrentTime(),
      neighborPatch.getTimestepSize()
    );

    if(neighborPatch.isLeaf()) {
      updatedPatch.updateMinimalLeafNeighborTimeConstraint(neighborPatch.getTimeConstraint());
    }
  }
}

peanoclaw::interSubgridCommunication::GhostLayerCompositor::FluxCorrectionFunctor::FluxCorrectionFunctor(
  Numerics& numerics
) : _numerics(numerics) {
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::FluxCorrectionFunctor::operator() (
  peanoclaw::Patch&                  patch1,
  int                                index1,
  peanoclaw::Patch&                  patch2,
  int                                index2,
  tarch::la::Vector<DIMENSIONS, int> direction
) {
  if(patch1.isLeaf()
      && patch2.isLeaf()) {
    int dimension = -1;
    int offset = 0;
    for(int d = 0; d < DIMENSIONS; d++) {
      if(abs(direction(d)) == 1) {
        dimension = d;
        offset = direction(d);
      }
    }
    assertion3(dimension!=-1 && offset != 0, dimension, offset, direction);

    //Correct from patch1 to patch2
    if(patch1.getLevel() > patch2.getLevel()) {
      _numerics.applyFluxCorrection(patch1, patch2, dimension, offset);
    }

    //Correct from right to left
    if(patch2.getLevel() > patch1.getLevel()) {
      _numerics.applyFluxCorrection(patch2, patch1, dimension, offset);
    }
  }
}

peanoclaw::interSubgridCommunication::GhostLayerCompositor::UpdateGhostlayerBoundsFaceFunctor::UpdateGhostlayerBoundsFaceFunctor (
  GhostLayerCompositor& ghostlayerCompositor
) : _ghostlayerCompositor(ghostlayerCompositor) {
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::UpdateGhostlayerBoundsFaceFunctor::operator() (
  peanoclaw::Patch&                  patch1,
  int                                index1,
  peanoclaw::Patch&                  patch2,
  int                                index2,
  tarch::la::Vector<DIMENSIONS, int> direction
) {
  int dimension = -1;
  for(int d = 0; d < DIMENSIONS; d++) {
    if(abs(direction(d)) == 1) {
      dimension = d;
    }
  }

  if(patch1.isLeaf() && patch2.isValid()
      && patch1.getLevel() == patch2.getLevel()) {
    if(index1 < index2) {
      _ghostlayerCompositor.updateUpperGhostlayerBound(index2, index1, dimension);
    } else {
      _ghostlayerCompositor.updateLowerGhostlayerBound(index2, index1, dimension);
    }
  }
}

peanoclaw::interSubgridCommunication::GhostLayerCompositor::UpdateGhostlayerBoundsEdgeFunctor::UpdateGhostlayerBoundsEdgeFunctor (
  GhostLayerCompositor& ghostlayerCompositor
) : _ghostlayerCompositor(ghostlayerCompositor) {
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::UpdateGhostlayerBoundsEdgeFunctor::operator() (
  peanoclaw::Patch&                  patch1,
  int                                index1,
  peanoclaw::Patch&                  patch2,
  int                                index2,
  peanoclaw::Patch&                  faceNeighbor1,
  int                                indexFaceNeighbor1,
  peanoclaw::Patch&                  faceNeighbor2,
  int                                indexFaceNeighbor2,
  tarch::la::Vector<DIMENSIONS, int> direction
) {
  if((!faceNeighbor1.isValid() || !faceNeighbor1.isLeaf()) && (!faceNeighbor2.isValid() || !faceNeighbor2.isLeaf())) {
    if(patch1.isValid() && patch2.isValid() && (patch1.getLevel() == patch2.getLevel())) {
      for(int d = 0; d < DIMENSIONS; d++) {
        if(direction(d) != 0) {
          if(index1 < index2) {
            _ghostlayerCompositor.updateUpperGhostlayerBound(index2, index1, d);
          } else {
            _ghostlayerCompositor.updateLowerGhostlayerBound(index2, index1, d);
          }
        }
      }
    }
  }
}
