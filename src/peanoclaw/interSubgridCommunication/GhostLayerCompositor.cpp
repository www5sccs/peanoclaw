/*
 * GhostLayerCompositor.cpp
 *
 *  Created on: Feb 16, 2012
 *      Author: Kristof Unterweger
 */

#include "peanoclaw/interSubgridCommunication/GhostLayerCompositor.h"
#include "peanoclaw/interSubgridCommunication/aspects/FaceAdjacentPatchTraversal.h"
#include "peanoclaw/interSubgridCommunication/aspects/EdgeAdjacentPatchTraversal.h"
#include "peanoclaw/Patch.h"

#include "peano/utils/Loop.h"

#include "tarch/parallel/Node.h"

tarch::logging::Log peanoclaw::interSubgridCommunication::GhostLayerCompositor::_log("peanoclaw::interSubgridCommunication::GhostLayerCompositor");

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::copyGhostLayerDataBlock(
    const tarch::la::Vector<DIMENSIONS, int>& size,
    const tarch::la::Vector<DIMENSIONS, int>& sourceOffset,
    const tarch::la::Vector<DIMENSIONS, int>& destinationOffset,
    const peanoclaw::Patch& source,
    peanoclaw::Patch& destination) {
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

  //TODO unterweg As soon as the virtual patches work correctly, the time interpolation can be activated
//  timeFactor = 1.0;

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

  //Faces
  UpdateNeighborTimeFaceFunctor faceFunctor(
    *this
  );
  peanoclaw::interSubgridCommunication::aspects::FaceAdjacentPatchTraversal<UpdateNeighborTimeFaceFunctor>(
    _patches,
    faceFunctor
  );

  if(!_useDimensionalSplittingOptimization) {
    UpdateNeighborTimeEdgeFunctor edgeFunctor(
      *this
    );
    //Edges
    peanoclaw::interSubgridCommunication::aspects::EdgeAdjacentPatchTraversal<UpdateNeighborTimeEdgeFunctor>(
      _patches,
      edgeFunctor
    );

    //Corners
    #ifdef Dim3
    UpdateNeighborTimeCornerFunctor cornerFunctor(
      *this
    );
    peanoclaw::interSubgridCommunication::aspects::CornerAdjacentPatchTraversal<UpdateNeighborTimeCornerFunctor>(
      _patches,
      cornerFunctor
    );
    #endif
  }
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::updateGhostlayerBounds() {
  int rightPatchIndex = 0;

  //Faces
  for(int d = 0; d < DIMENSIONS; d++) {
    int leftPatchIndex = rightPatchIndex + tarch::la::aPowI(d,2);

    if(_patches[leftPatchIndex].isLeaf()
        && _patches[rightPatchIndex].isValid()
        && _patches[leftPatchIndex].getLevel() == _patches[rightPatchIndex].getLevel()) {
      updateUpperGhostlayerBound(rightPatchIndex, leftPatchIndex, d);
    }
    if(_patches[rightPatchIndex].isLeaf()
        && _patches[leftPatchIndex].isValid()
        && _patches[rightPatchIndex].getLevel() == _patches[leftPatchIndex].getLevel()) {
      updateLowerGhostlayerBound(leftPatchIndex, rightPatchIndex, d);
    }
  }

  //Corners (2D)/Edges (3D)
//  2->4
//  3->12
//
//  2D:
//  0->3
//  1->2
//
//  3D:
//  0->3
//  0->5
//  0->6
//  1->2
//  1->4
//  1->7
//  2->4
//  2->7
//  3->5
//  3->6
//  4->7
//  5->6

  if((!_patches[1].isValid() || !_patches[1].isLeaf()) && (!_patches[2].isValid() || !_patches[2].isLeaf())) {
    rightPatchIndex = 0;
    int leftPatchIndex = 3;
    if(_patches[rightPatchIndex].isValid() && _patches[leftPatchIndex].isValid()) {
      if(_patches[rightPatchIndex].getLevel() == _patches[leftPatchIndex].getLevel()) {
        //Upper right corner
        if(_patches[rightPatchIndex].isLeaf()) {
          updateLowerGhostlayerBound(leftPatchIndex, rightPatchIndex, 0);
          updateLowerGhostlayerBound(leftPatchIndex, rightPatchIndex, 1);
        }
        //Lower left corner
        if(_patches[leftPatchIndex].isLeaf()) {
          updateUpperGhostlayerBound(rightPatchIndex, leftPatchIndex, 0);
          updateUpperGhostlayerBound(rightPatchIndex, leftPatchIndex, 1);
        }
      }
    }
  }

  if((!_patches[0].isValid() || !_patches[0].isLeaf()) && (!_patches[3].isValid() || !_patches[3].isLeaf())) {
    rightPatchIndex = 1;
    int leftPatchIndex = 2;
    if(_patches[rightPatchIndex].isValid() && _patches[leftPatchIndex].isValid()) {
      if(_patches[rightPatchIndex].getLevel() == _patches[leftPatchIndex].getLevel()) {
        //Upper left corner
        if(_patches[rightPatchIndex].isLeaf()) {
          updateUpperGhostlayerBound(leftPatchIndex, rightPatchIndex, 0);
          updateLowerGhostlayerBound(leftPatchIndex, rightPatchIndex, 1);
        }
        //Lower right corner
        if(_patches[leftPatchIndex].isLeaf()) {
          updateLowerGhostlayerBound(rightPatchIndex, leftPatchIndex, 0);
          updateUpperGhostlayerBound(rightPatchIndex, leftPatchIndex, 1);
        }
      }
    }
  }
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::applyCoarseGridCorrection() {

  //Left and right patch indices need to be understood in a "per-dimension-way". I.e. in the following
  //loop over dimensions
  //for d=0 leftPatchIndex is the left patch, rightPatchIndex the right patch
  //for d=1 leftPatchIndex is the lower patch, rightPatchIndex is the upper patch and so on.
  int rightPatchIndex = 0;

  for(int d = 0; d < DIMENSIONS; d++) {
    int leftPatchIndex = rightPatchIndex + tarch::la::aPowI(d,2);

    if(_patches[rightPatchIndex].isLeaf()
        && _patches[leftPatchIndex].isLeaf()) {
      //Correct from left to right
      if(_patches[leftPatchIndex].getLevel() > _patches[rightPatchIndex].getLevel()) {
        _numerics.applyFluxCorrection(_patches[leftPatchIndex], _patches[rightPatchIndex], d, -1);
      }

      //Correct from right to left
      if(_patches[rightPatchIndex].getLevel() > _patches[leftPatchIndex].getLevel()) {
        _numerics.applyFluxCorrection(_patches[rightPatchIndex], _patches[leftPatchIndex], d, 1);
      }
    }
  }
}

peanoclaw::interSubgridCommunication::GhostLayerCompositor::FillGhostlayerFaceFunctor::FillGhostlayerFaceFunctor(
  GhostLayerCompositor& ghostlayerCompositor,
  int                   destinationPatchIndex
) : _ghostlayerCompositor(ghostlayerCompositor), _destinationPatchIndex(destinationPatchIndex) {
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::FillGhostlayerFaceFunctor::operator()
(
  peanoclaw::Patch& source,
  int               sourceIndex,
  peanoclaw::Patch& destination,
  int               destinationIndex,
  int               dimension,
  int               direction
) {

  //TODO unterweg debug
//  std::cout << "Copy? " << sourceIndex << "->" << destinationIndex << " destinationPatchIndex=" << _destinationPatchIndex << std::endl
//      << "source: " << source
//      << ", destination: " << destination << std::endl;


  if(
    (_destinationPatchIndex == -1 || _destinationPatchIndex == destinationIndex)
    && _ghostlayerCompositor.shouldTransferGhostlayerData(source, destination)
  ) {
    assertionEquals2(destination.getSubdivisionFactor(), source.getSubdivisionFactor(), source, destination);
    assertionEquals2(source.getGhostLayerWidth(), destination.getGhostLayerWidth(), source, destination);
    int ghostLayerWidth = source.getGhostLayerWidth();
    tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = source.getSubdivisionFactor();
    tarch::la::Vector<DIMENSIONS, int> faceSize(subdivisionFactor);
    faceSize(dimension) = ghostLayerWidth;
    tarch::la::Vector<DIMENSIONS, int> destinationOffset(0);
    destinationOffset(dimension) = (direction==1) ? -ghostLayerWidth : subdivisionFactor(dimension);

    if(source.getLevel() == destination.getLevel() && source.getLevel() == _ghostlayerCompositor._level) {
      tarch::la::Vector<DIMENSIONS, int> sourceOffset(0);
      sourceOffset(dimension)
        = (direction==1) ? (source.getSubdivisionFactor()(dimension) - ghostLayerWidth) : 0;

      //TODO unterweg debug
//      std::cout << "CopyGhostlayerDataBlock: " << index1 << "->" << index2 << " sourceOffset=" << sourceOffset
//          << ", destinationOffset=" << destinationOffset << ", faceSize=" << faceSize
//          << " patch1=" << patch1 << ", patch2=" << patch2 << std::endl;

      _ghostlayerCompositor.copyGhostLayerDataBlock(faceSize, sourceOffset, destinationOffset, source, destination);
    } else if(source.getLevel() < destination.getLevel() && destination.getLevel() == _ghostlayerCompositor._level && destination.isLeaf()) {
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
    assertionEquals2(source.getGhostLayerWidth(), destination.getGhostLayerWidth(), source, destination);
    int ghostLayerWidth = source.getGhostLayerWidth();
    tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = source.getSubdivisionFactor();
    tarch::la::Vector<DIMENSIONS, int> edgeSize(ghostLayerWidth);
    tarch::la::Vector<DIMENSIONS, int> sourceOffset(0);
    tarch::la::Vector<DIMENSIONS, int> destinationOffset(0);
    for(int d = 0; d < DIMENSIONS; d++) {
      if(direction(d) == 0) {
        edgeSize(d) = subdivisionFactor(d);
      } else if(direction(d) == 1) {
        sourceOffset(d) = subdivisionFactor(d) - ghostLayerWidth;
        destinationOffset(d) = -ghostLayerWidth;
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

peanoclaw::interSubgridCommunication::GhostLayerCompositor::UpdateNeighborTimeFaceFunctor::UpdateNeighborTimeFaceFunctor(
  GhostLayerCompositor& ghostlayerCompositor
) : _ghostlayerCompositor(ghostlayerCompositor) {
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::UpdateNeighborTimeFaceFunctor::operator() (
  peanoclaw::Patch&                  neighborPatch,
  int                                neighborPatchIndex,
  peanoclaw::Patch&                  updatedPatch,
  int                                updatedPatchIndexIndex,
  int                                dimension,
  int                                direction
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

peanoclaw::interSubgridCommunication::GhostLayerCompositor::UpdateNeighborTimeEdgeFunctor::UpdateNeighborTimeEdgeFunctor(
  GhostLayerCompositor& ghostlayerCompositor
) : _ghostlayerCompositor(ghostlayerCompositor) {
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::UpdateNeighborTimeEdgeFunctor::operator() (
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
