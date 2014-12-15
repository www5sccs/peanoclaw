/*
 * GhostlayerCompositorFunctors.cpp
 *
 *  Created on: Sep 5, 2013
 *      Author: unterweg
 */
#include "peanoclaw/interSubgridCommunication/GhostlayerCompositorFunctors.h"

#include "peanoclaw/interSubgridCommunication/Extrapolation.h"
#include "peanoclaw/interSubgridCommunication/GhostLayerCompositor.h"
#include "peanoclaw/Numerics.h"
#include "peanoclaw/Patch.h"

peanoclaw::interSubgridCommunication::FillGhostlayerFaceFunctor::FillGhostlayerFaceFunctor(
  GhostLayerCompositor& ghostlayerCompositor,
  int                   destinationPatchIndex
) : _ghostlayerCompositor(ghostlayerCompositor),
    _destinationPatchIndex(destinationPatchIndex),
    _maximumLinearError(0)
{
}

void peanoclaw::interSubgridCommunication::FillGhostlayerFaceFunctor::operator()
(
  peanoclaw::Patch&                         source,
  int                                       sourceIndex,
  peanoclaw::Patch&                         destination,
  int                                       destinationIndex,
  const tarch::la::Vector<DIMENSIONS, int>& direction
) {
  if(
    (_destinationPatchIndex == -1 || _destinationPatchIndex == destinationIndex)
    && _ghostlayerCompositor.shouldTransferGhostlayerData(source, destination)
  ) {
    assertion1(source.isLeaf() || source.isVirtual(), source);

    int dimension = -1;
    int offset = 0;
    for(int d = 0; d < DIMENSIONS; d++) {
      if(std::abs(direction(d)) == 1) {
        dimension = d;
        offset = direction(d);
      }
    }
    assertion3(dimension!=-1 && offset != 0, dimension, offset, direction);

    assertionEquals2(destination.getSubdivisionFactor(), source.getSubdivisionFactor(), source, destination);
    tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = source.getSubdivisionFactor();
    tarch::la::Vector<DIMENSIONS, int> faceSize(subdivisionFactor);
    faceSize(dimension) = destination.getGhostlayerWidth();
    tarch::la::Vector<DIMENSIONS, int> destinationOffset(0);
    destinationOffset(dimension) = (offset==1) ? -destination.getGhostlayerWidth() : subdivisionFactor(dimension);

    if(source.getLevel() == destination.getLevel() && source.getLevel() == _ghostlayerCompositor._level) {
      tarch::la::Vector<DIMENSIONS, int> sourceOffset(0);
      sourceOffset(dimension)
        = (offset==1) ? (subdivisionFactor(dimension) - source.getGhostlayerWidth()) : 0;

      _ghostlayerCompositor._numerics.transferGhostlayer(faceSize, sourceOffset, destinationOffset, source, destination);
    } else if(source.getLevel() < destination.getLevel() && destination.getLevel() == _ghostlayerCompositor._level && source.isLeaf()) {
      _ghostlayerCompositor._numerics.interpolateSolution(
        faceSize,
        destinationOffset,
        source,
        destination,
        true,
        false,
        true
      );
    }
  }
}

peanoclaw::interSubgridCommunication::FillGhostlayerEdgeFunctor::FillGhostlayerEdgeFunctor(
  GhostLayerCompositor& ghostlayerCompositor,
  Extrapolation&        extrapolation,
  bool                  fillFromNeighbor,
  int                   destinationPatchIndex
) : _ghostlayerCompositor(ghostlayerCompositor),
    _extrapolation(extrapolation),
    _fillFromNeighbor(fillFromNeighbor),
    _destinationPatchIndex(destinationPatchIndex),
    _maximumLinearError(0)
{
}

void peanoclaw::interSubgridCommunication::FillGhostlayerEdgeFunctor::operator() (
  peanoclaw::Patch&                         source,
  int                                       sourceIndex,
  peanoclaw::Patch&                         destination,
  int                                       destinationIndex,
  const tarch::la::Vector<DIMENSIONS, int>& direction
) {

  //TODO unterweg debug
//  std::cout << "Filling ghostlayer edge from " << sourceIndex << " to " << destinationIndex
//      << " _destinationPatchIndex=" << _destinationPatchIndex << " should transfer: " << _ghostlayerCompositor.shouldTransferGhostlayerData(source, destination) << std::endl
//      << "source: " << source << std::endl
//      << "destiantion: " << destination << std::endl;
  if(
    (_destinationPatchIndex == -1 || _destinationPatchIndex == destinationIndex)
    && _ghostlayerCompositor.shouldTransferGhostlayerData(source, destination)
  ) {
    assertionEquals2(source.getSubdivisionFactor(), destination.getSubdivisionFactor(), source, destination);
    int ghostlayerWidth = destination.getGhostlayerWidth();
    tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = source.getSubdivisionFactor();
    tarch::la::Vector<DIMENSIONS, int> edgeSize(ghostlayerWidth);
    tarch::la::Vector<DIMENSIONS, int> sourceOffset(0);
    tarch::la::Vector<DIMENSIONS, int> destinationOffset(0);
    int sourceLevel = source.getLevel();
    int destinationLevel = destination.getLevel();
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
    if(sourceLevel == destinationLevel && sourceLevel == _ghostlayerCompositor._level) {
      _ghostlayerCompositor._numerics.transferGhostlayer(
        edgeSize,
        sourceOffset,
        destinationOffset,
        source,
        destination
      );
    } else if(source.getLevel() < destinationLevel && destinationLevel == _ghostlayerCompositor._level && source.isLeaf()) {

      //TODO unterweg debug
//      std::cout << "Interpolating ghostlayer edge from " << sourceIndex << " to " << destinationIndex << std::endl;

      _ghostlayerCompositor._numerics.interpolateSolution(
        edgeSize,
        destinationOffset,
        source,
        destination,
        true,
        false,
        true
      );
    }
  }
}

double peanoclaw::interSubgridCommunication::FillGhostlayerEdgeFunctor::getMaximumLinearError() const {
  return _maximumLinearError;
}

peanoclaw::interSubgridCommunication::FillGhostlayerCornerFunctor::FillGhostlayerCornerFunctor(
  GhostLayerCompositor& ghostlayerCompositor,
  Extrapolation&        extrapolation,
  bool                  fillFromNeighbor,
  int                   destinationPatchIndex
) : _ghostlayerCompositor(ghostlayerCompositor),
    _extrapolation(extrapolation),
    _fillFromNeighbor(fillFromNeighbor),
    _destinationPatchIndex(destinationPatchIndex),
    _maximumLinearError(0)
{
}

void peanoclaw::interSubgridCommunication::FillGhostlayerCornerFunctor::operator() (
  peanoclaw::Patch&                         source,
  int                                       sourceIndex,
  peanoclaw::Patch&                         destination,
  int                                       destinationIndex,
  const tarch::la::Vector<DIMENSIONS, int>& direction
) {

  if(
    (_destinationPatchIndex == -1 || _destinationPatchIndex == destinationIndex)
    && _ghostlayerCompositor.shouldTransferGhostlayerData(source, destination)
  ) {
    assertionEquals2(source.getSubdivisionFactor(), destination.getSubdivisionFactor(), source, destination);
    int ghostlayerWidth = destination.getGhostlayerWidth();
    tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = destination.getSubdivisionFactor();
    tarch::la::Vector<DIMENSIONS, int> cornerSize(ghostlayerWidth);
    tarch::la::Vector<DIMENSIONS, int> sourceOffset(0);
    tarch::la::Vector<DIMENSIONS, int> destinationOffset(0);
    int sourceLevel = source.getLevel();
    int destinationLevel = destination.getLevel();
    for(int d = 0; d < DIMENSIONS; d++) {
      if(direction(d) == 1) {
        sourceOffset(d) = subdivisionFactor(d) - ghostlayerWidth;
        destinationOffset(d) = -ghostlayerWidth;
      } else if(direction(d) == -1) {
        destinationOffset(d) = subdivisionFactor(d);
      } else {
        assertionFail("Direction " << direction << " is invalid!");
      }
    }
    if(sourceLevel == destinationLevel && sourceLevel == _ghostlayerCompositor._level) {
      _ghostlayerCompositor._numerics.transferGhostlayer(
        cornerSize,
        sourceOffset,
        destinationOffset,
        source,
        destination
      );
    } else if(sourceLevel < destinationLevel && destinationLevel == _ghostlayerCompositor._level && source.isLeaf()) {
      _ghostlayerCompositor._numerics.interpolateSolution(
        cornerSize,
        destinationOffset,
        source,
        destination,
        true,
        false,
        true
      );
    }
  }
}

double peanoclaw::interSubgridCommunication::FillGhostlayerCornerFunctor::getMaximumLinearError() const {
  return _maximumLinearError;
}

peanoclaw::interSubgridCommunication::UpdateNeighborTimeFunctor::UpdateNeighborTimeFunctor(
  GhostLayerCompositor& ghostlayerCompositor
) : _ghostlayerCompositor(ghostlayerCompositor) {
}

void peanoclaw::interSubgridCommunication::UpdateNeighborTimeFunctor::operator() (
  peanoclaw::Patch&                         neighborPatch,
  int                                       neighborPatchIndex,
  peanoclaw::Patch&                         updatedPatch,
  int                                       updatedPatchIndex,
  const tarch::la::Vector<DIMENSIONS, int>& direction
) {
  if(updatedPatch.isValid() && neighborPatch.isValid()) {
    peanoclaw::grid::TimeIntervals updatedTimeIntervals = updatedPatch.getTimeIntervals();
    peanoclaw::grid::TimeIntervals neighborTimeIntervals = neighborPatch.getTimeIntervals();

    double neighborTimeConstraint = neighborTimeIntervals.getTimeConstraint();
    updatedTimeIntervals.updateMinimalNeighborTimeConstraint(
      neighborTimeConstraint,
      neighborPatch.getCellDescriptionIndex()
    );
    updatedTimeIntervals.updateMaximalNeighborTimeInterval(
      neighborTimeIntervals.getCurrentTime(),
      neighborTimeIntervals.getTimestepSize()
    );

    if(neighborPatch.isLeaf()) {
      updatedTimeIntervals.updateMinimalLeafNeighborTimeConstraint(
        neighborTimeConstraint
      );
    }

    if(tarch::la::greaterEquals(neighborTimeIntervals.getNeighborInducedMaximumTimestepSize(), 0.0)) {
      updatedTimeIntervals.setNeighborInducedMaximumTimestepSize(
        std::min(
          updatedTimeIntervals.getNeighborInducedMaximumTimestepSize(),
          std::min(
            neighborTimeIntervals.getNeighborInducedMaximumTimestepSize() * tarch::la::min(neighborPatch.getSubdivisionFactor()),
            neighborTimeIntervals.getEstimatedNextTimestepSize()
          )
        )
      );
    }
  }
}

peanoclaw::interSubgridCommunication::FluxCorrectionFunctor::FluxCorrectionFunctor(
  Numerics& numerics,
  int       sourceSubgridIndex
) : _numerics(numerics), _sourceSubgridIndex(sourceSubgridIndex) {
}

void peanoclaw::interSubgridCommunication::FluxCorrectionFunctor::operator() (
  peanoclaw::Patch&                         patch1,
  int                                       index1,
  peanoclaw::Patch&                         patch2,
  int                                       index2,
  const tarch::la::Vector<DIMENSIONS, int>& direction
) {
  if(index1 == _sourceSubgridIndex) {
    if(patch1.isValid() && patch1.isLeaf() && patch2.isValid() && patch2.isLeaf()) {
      int dimension = -1;
      int offset = 0;
      for(int d = 0; d < DIMENSIONS; d++) {
        if(abs(direction(d)) == 1) {
          dimension = d;
          offset = direction(d);
        }
      }
      assertion3(dimension!=-1 && offset != 0, dimension, offset, direction);

      _numerics.applyFluxCorrection(patch1, patch2, dimension, offset);
    }
  }
}

peanoclaw::interSubgridCommunication::UpdateGhostlayerBoundsFaceFunctor::UpdateGhostlayerBoundsFaceFunctor (
  GhostLayerCompositor& ghostlayerCompositor
) : _ghostlayerCompositor(ghostlayerCompositor) {
}

void peanoclaw::interSubgridCommunication::UpdateGhostlayerBoundsFaceFunctor::operator() (
  peanoclaw::Patch&                         patch1,
  int                                       index1,
  peanoclaw::Patch&                         patch2,
  int                                       index2,
  const tarch::la::Vector<DIMENSIONS, int>& direction
) {
  if(index1 < index2 && patch1.isValid() && patch2.isValid()) {
    int dimension = -1;
    for(int d = 0; d < DIMENSIONS; d++) {
      if(abs(direction(d)) == 1) {
        dimension = d;
      }
    }

    if(patch1.getLevel() == patch2.getLevel()) {
      if(patch1.isLeaf()) {
        _ghostlayerCompositor.updateUpperGhostlayerBound(index2, index1, dimension);
      }

      if(patch2.isLeaf()) {
        _ghostlayerCompositor.updateLowerGhostlayerBound(index1, index2, dimension);
      }
    }
  }
}

peanoclaw::interSubgridCommunication::UpdateGhostlayerBoundsEdgeFunctor::UpdateGhostlayerBoundsEdgeFunctor (
  GhostLayerCompositor& ghostlayerCompositor
) : _ghostlayerCompositor(ghostlayerCompositor) {
}

void peanoclaw::interSubgridCommunication::UpdateGhostlayerBoundsEdgeFunctor::operator() (
  peanoclaw::Patch&                         patch1,
  int                                       index1,
  peanoclaw::Patch&                         patch2,
  int                                       index2,
  const tarch::la::Vector<DIMENSIONS, int>& direction
) {
  if(index1 < index2) {
    if(patch1.isValid() && patch2.isValid() && (patch1.getLevel() == patch2.getLevel())) {
      _ghostlayerCompositor.updateGhostlayerBound(index2, index1, direction);
      _ghostlayerCompositor.updateGhostlayerBound(index1, index2, direction*-1);
    }
  }
}

peanoclaw::interSubgridCommunication::UpdateGhostlayerBoundsCornerFunctor::UpdateGhostlayerBoundsCornerFunctor (
  GhostLayerCompositor& ghostlayerCompositor
) : _ghostlayerCompositor(ghostlayerCompositor) {
}

void peanoclaw::interSubgridCommunication::UpdateGhostlayerBoundsCornerFunctor::operator() (
  peanoclaw::Patch&                         patch1,
  int                                       index1,
  peanoclaw::Patch&                         patch2,
  int                                       index2,
  const tarch::la::Vector<DIMENSIONS, int>& direction
) {
  assertion("Not implemented, yet!");
//  if((!faceNeighbor1.isValid() || !faceNeighbor1.isLeaf()) && (!faceNeighbor2.isValid() || !faceNeighbor2.isLeaf())) {
//    if(patch1.isValid() && patch2.isValid() && (patch1.getLevel() == patch2.getLevel())) {
//      for(int d = 0; d < DIMENSIONS; d++) {
//        if(direction(d) != 0) {
//          if(index1 < index2) {
//            _ghostlayerCompositor.updateUpperGhostlayerBound(index2, index1, d);
//          } else {
//            _ghostlayerCompositor.updateLowerGhostlayerBound(index2, index1, d);
//          }
//        }
//      }
//    }
//  }
}
