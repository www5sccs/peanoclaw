/*
 * GhostLayerCompositor.cpp
 *
 *  Created on: Feb 16, 2012
 *      Author: Kristof Unterweger
 */

#include "peanoclaw/interSubgridCommunication/GhostLayerCompositor.h"
#include "peanoclaw/PatchOperations.h"
#include "peanoclaw/Patch.h"

#include "peano/utils/Loop.h"

tarch::logging::Log peanoclaw::interSubgridCommunication::GhostLayerCompositor::_log("peanoclaw::interSubgridCommunication::GhostLayerCompositor");

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::interpolate(
  const tarch::la::Vector<DIMENSIONS, int>&    destinationSize,
  const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
  const peanoclaw::Patch& source,
  peanoclaw::Patch&       destination,
  bool interpolateToUOld,
  bool interpolateToCurrentTime
) {
  peanoclaw::interpolate(
    destinationSize,
    destinationOffset,
    source,
    destination,
    interpolateToUOld,
    interpolateToCurrentTime
  );

//  peanoclaw::interpolateOldVersion(
//    destinationSize,
//    destinationOffset,
//    source,
//    destination,
//    interpolateToUOld
//  );
}

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
    timeFactor = (destination.getTimeUNew() - 0.0) / (1.0 - source.getTimeUOld());
  } else {
    if(tarch::la::greater(source.getTimeUNew() - source.getTimeUOld(), 0.0)) {
      timeFactor = (destination.getTimeUNew() - source.getTimeUOld()) / (source.getTimeUNew() - source.getTimeUOld());
    } else {
      timeFactor = 1.0;
    }
  }

  bool copyFromUOld = tarch::la::equals(timeFactor, 0.0);
  bool copyFromUNew = tarch::la::equals(timeFactor, 1.0);

  //TODO unterweg debug
//  if(tarch::la::equals(timeFactor, 0.0)) {
//    std::cout << "TRIVIAL CASE 0" << std::endl;
//  }else if (tarch::la::equals(timeFactor, 1.0)) {
//    std::cout << "TRIVIAL CASE 1" << std::endl;
//  } else {
//    std::cout << "COMPLEX CASE" << std::endl;
//  }

  //TODO unterweg As soon as the virtual patches work correctly, the time interpolation can be activated
//  timeFactor = 1.0;

  int sourceUnknownsPerSubcell = source.getUnknownsPerSubcell();

  if(copyFromUOld) {
    dfor(subcellindex, size) {
      int linearSourceUOldIndex = source.getLinearIndexUOld(subcellindex + sourceOffset);
      int linearDestinationUOldIndex = destination.getLinearIndexUOld(subcellindex + destinationOffset);

      for(int unknown = 0; unknown < sourceUnknownsPerSubcell; unknown++) {
        double valueUOld = source.getValueUOld(linearSourceUOldIndex, unknown);

        destination.setValueUOld(linearDestinationUOldIndex, unknown, valueUOld);

        logDebug("copyGhostLayerDataBlock(...)", "Copied cell " << (subcellindex+sourceOffset) << " with value " << valueUOld << " to " << (subcellindex+destinationOffset));
      }
    }
  } else if (copyFromUNew) {
    dfor(subcellindex, size) {
      int linearSourceUNewIndex = source.getLinearIndexUNew(subcellindex + sourceOffset);
      int linearDestinationUOldIndex = destination.getLinearIndexUOld(subcellindex + destinationOffset);

      for(int unknown = 0; unknown < sourceUnknownsPerSubcell; unknown++) {
        double valueUNew = source.getValueUNew(linearSourceUNewIndex, unknown);

        destination.setValueUOld(linearDestinationUOldIndex, unknown, valueUNew);

        logDebug("copyGhostLayerDataBlock(...)", "Copied cell " << (subcellindex+sourceOffset) << " with value " << valueUNew << " to " << (subcellindex+destinationOffset));
      }
    }
  } else {
    dfor(subcellindex, size) {
      int linearSourceUNewIndex = source.getLinearIndexUNew(subcellindex + sourceOffset);
      int linearSourceUOldIndex = source.getLinearIndexUOld(subcellindex + sourceOffset);
      int linearDestinationUOldIndex = destination.getLinearIndexUOld(subcellindex + destinationOffset);

      for(int unknown = 0; unknown < sourceUnknownsPerSubcell; unknown++) {
        double valueUNew = source.getValueUNew(linearSourceUNewIndex, unknown);
        double valueUOld = source.getValueUOld(linearSourceUOldIndex, unknown);

        double value = valueUNew * timeFactor + valueUOld * (1.0 - timeFactor);

        #ifdef Asserts
        if(unknown == 0) {
          assertion8(tarch::la::greaterEquals(value, 0.0),
              value,
              valueUNew,
              valueUOld,
              timeFactor,
              source,
              destination,
              source.toStringUNew() << source.toStringUOldWithGhostLayer(),
              destination.toStringUNew() << destination.toStringUOldWithGhostLayer()
              );
        }
        #endif

        destination.setValueUOld(linearDestinationUOldIndex, unknown, value);

        logDebug("copyGhostLayerDataBlock(...)", "Copied cell " << (subcellindex+sourceOffset) << " with value " << value << " to " << (subcellindex+destinationOffset));
      }
    }
  }

  logTraceOut("copyGhostLayerDataBlock");
}

peanoclaw::interSubgridCommunication::GhostLayerCompositor::GhostLayerCompositor(
  peanoclaw::Patch patches[TWO_POWER_D],
  int level,
  peanoclaw::pyclaw::PyClaw& pyClaw,
  bool useDimensionalSplitting
) : _patches(patches), _level(level), _pyClaw(pyClaw), _useDimensionalSplitting(useDimensionalSplitting) {
}

peanoclaw::interSubgridCommunication::GhostLayerCompositor::~GhostLayerCompositor() {
}

//Ghost layers, i.e. copy from one patch to another.
void peanoclaw::interSubgridCommunication::GhostLayerCompositor::fillLeftGhostLayer() {
  assertionEquals(_patches[0].getSubdivisionFactor(), _patches[1].getSubdivisionFactor());
  tarch::la::Vector<DIMENSIONS, int> columnSize;
  int ghostLayerWidth = _patches[0].getGhostLayerWidth();
  tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = _patches[0].getSubdivisionFactor();
  columnSize(0) = ghostLayerWidth;
  columnSize(1) = subdivisionFactor(1);
  tarch::la::Vector<DIMENSIONS, int> destinationOffset;
  destinationOffset(0) = -ghostLayerWidth;
  destinationOffset(1) = 0;

  if(_patches[0].getLevel() == _patches[1].getLevel() && _patches[0].getLevel() == _level) {
    tarch::la::Vector<DIMENSIONS, int> sourceOffset;
    sourceOffset(0) = subdivisionFactor(0) - ghostLayerWidth;
    sourceOffset(1) = 0;
    copyGhostLayerDataBlock(columnSize, sourceOffset, destinationOffset, _patches[1], _patches[0]);
  } else if(_patches[1].getLevel() < _patches[0].getLevel() && _patches[0].getLevel() == _level && _patches[1].isLeaf()) {
    assertion2(_patches[1].getLevel() < _patches[0].getLevel(), _patches[1].getLevel(), _patches[0].getLevel());
    interpolate(columnSize, destinationOffset, _patches[1], _patches[0], true, false);
  } else if (_patches[1].getLevel() > _patches[0].getLevel() && _patches[1].getLevel() == _level) {
    assertion2(_patches[1].getLevel() > _patches[0].getLevel(), _patches[1].getLevel(), _patches[0].getLevel());
    applyCoarseGridCorrection(_patches[1], _patches[0], 0, 1);
  }
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::fillLowerLeftGhostLayer() {
  int ghostLayerWidth = _patches[0].getGhostLayerWidth();
  tarch::la::Vector<DIMENSIONS, int> cornerSize(ghostLayerWidth);
  tarch::la::Vector<DIMENSIONS, int> destinationOffset;
  destinationOffset(0) = -ghostLayerWidth;
  destinationOffset(1) = -ghostLayerWidth;

  if(_patches[0].getLevel() == _patches[3].getLevel() && _patches[0].getLevel() == _level) {
    tarch::la::Vector<DIMENSIONS, int> sourceOffset;
    sourceOffset(0) = _patches[3].getSubdivisionFactor()(0) - _patches[0].getGhostLayerWidth();
    sourceOffset(1) = _patches[3].getSubdivisionFactor()(1) - _patches[0].getGhostLayerWidth();

    copyGhostLayerDataBlock(cornerSize, sourceOffset, destinationOffset, _patches[3], _patches[0]);
  } else if(_patches[3].getLevel() < _patches[0].getLevel() && _patches[0].getLevel() == _level && _patches[3].isLeaf()) {
    assertion2(_patches[3].getLevel() < _patches[0].getLevel(), _patches[3].getLevel(), _patches[0].getLevel());
    interpolate(cornerSize, destinationOffset, _patches[3], _patches[0], true, false);
  } else if(_patches[3].getLevel() > _patches[0].getLevel() && _patches[3].getLevel() == _level){
    assertion2(_patches[3].getLevel() > _patches[0].getLevel(), _patches[3].getLevel(), _patches[0].getLevel());
  }
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::fillLowerGhostLayer() {
  int ghostLayerWidth = _patches[0].getGhostLayerWidth();
  tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = _patches[0].getSubdivisionFactor();
  tarch::la::Vector<DIMENSIONS, int> rowSize;
  rowSize(0) = subdivisionFactor(0);
  rowSize(1) = ghostLayerWidth;

  tarch::la::Vector<DIMENSIONS, int> destinationOffset;
  destinationOffset(0) = 0;
  destinationOffset(1) = -ghostLayerWidth;
  if(_patches[0].getLevel() == _patches[2].getLevel() && _patches[0].getLevel() == _level) {
    tarch::la::Vector<DIMENSIONS, int> sourceOffset;
    sourceOffset(0) = 0;
    sourceOffset(1) = subdivisionFactor(1) - ghostLayerWidth;
    copyGhostLayerDataBlock(rowSize, sourceOffset, destinationOffset, _patches[2], _patches[0]);
  } else if(_patches[2].getLevel() < _patches[0].getLevel() && _patches[0].getLevel() == _level && _patches[2].isLeaf()) {
    assertion2(_patches[2].getLevel() < _patches[0].getLevel(), _patches[2].getLevel(), _patches[0].getLevel());
    interpolate(rowSize, destinationOffset, _patches[2], _patches[0], true, false);
  } else if(_patches[2].getLevel() > _patches[0].getLevel() && _patches[2].getLevel() == _level){
    assertion2(_patches[2].getLevel() > _patches[0].getLevel(), _patches[2].getLevel(), _patches[0].getLevel());
    applyCoarseGridCorrection(_patches[2], _patches[0], 1, 1);
  }

}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::fillUpperLeftGhostLayer() {
  int ghostLayerWidth = _patches[2].getGhostLayerWidth();
  tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = _patches[2].getSubdivisionFactor();
  tarch::la::Vector<DIMENSIONS, int> cornerSize(ghostLayerWidth);
  tarch::la::Vector<DIMENSIONS, int> destinationOffset;
  destinationOffset(0) = -ghostLayerWidth;
  destinationOffset(1) = subdivisionFactor(1);

  if(_patches[1].getLevel() == _patches[2].getLevel() && _patches[1].getLevel() == _level) {
    tarch::la::Vector<DIMENSIONS, int> sourceOffset;
    sourceOffset(0) = subdivisionFactor(0) - ghostLayerWidth;
    sourceOffset(1) = 0;

    copyGhostLayerDataBlock(cornerSize, sourceOffset, destinationOffset, _patches[1], _patches[2]);
  } else if(_patches[1].getLevel() < _patches[2].getLevel() && _patches[2].getLevel() == _level && _patches[1].isLeaf()) {
    assertion2(_patches[1].getLevel() < _patches[2].getLevel(), _patches[1].getLevel(), _patches[2].getLevel());
    interpolate(cornerSize, destinationOffset, _patches[1], _patches[2], true, false);
  } else if(_patches[1].getLevel() > _patches[2].getLevel() && _patches[1].getLevel() == _level){
    assertion2(_patches[1].getLevel() > _patches[2].getLevel(), _patches[1].getLevel(), _patches[2].getLevel());
  }
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::fillUpperGhostLayer() {
  int ghostLayerWidth = _patches[2].getGhostLayerWidth();
  tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = _patches[2].getSubdivisionFactor();
  tarch::la::Vector<DIMENSIONS, int> rowSize;
  rowSize(0) = subdivisionFactor(0);
  rowSize(1) = ghostLayerWidth;
  tarch::la::Vector<DIMENSIONS, int> destinationOffset;
  destinationOffset(0) = 0;
  destinationOffset(1) = subdivisionFactor(1);
  tarch::la::Vector<DIMENSIONS, int> normal;
  assignList(normal) = 0, -1;

  if(_patches[0].getLevel() == _patches[2].getLevel() && _patches[0].getLevel() == _level) {
    tarch::la::Vector<DIMENSIONS, int> sourceOffset;
    sourceOffset(0) = 0;
    sourceOffset(1) = 0;

    copyGhostLayerDataBlock(rowSize, sourceOffset, destinationOffset, _patches[0], _patches[2]);
  } else if(_patches[0].getLevel() < _patches[2].getLevel() && _patches[2].getLevel() == _level && _patches[0].isLeaf()) {
    assertion2(_patches[0].getLevel() < _patches[2].getLevel(), _patches[0].getLevel(), _patches[2].getLevel());
    interpolate(rowSize, destinationOffset, _patches[0], _patches[2], true, false);
  } else if(_patches[0].getLevel() > _patches[2].getLevel() && _patches[0].getLevel() == _level){
    assertion2(_patches[0].getLevel() > _patches[2].getLevel(), _patches[0].getLevel(), _patches[2].getLevel());
    applyCoarseGridCorrection(_patches[0], _patches[2], 1, -1);
  }
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::fillUpperRightGhostLayer() {
  int ghostLayerWidth = _patches[3].getGhostLayerWidth();
  tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = _patches[3].getSubdivisionFactor();
  tarch::la::Vector<DIMENSIONS, int> cornerSize(ghostLayerWidth);
  tarch::la::Vector<DIMENSIONS, int> destinationOffset;
  destinationOffset(0) = subdivisionFactor(0);
  destinationOffset(1) = subdivisionFactor(1);

  if(_patches[0].getLevel() == _patches[3].getLevel() && _patches[0].getLevel() == _level) {
    tarch::la::Vector<DIMENSIONS, int> sourceOffset;
    sourceOffset(0) = 0;
    sourceOffset(1) = 0;

    copyGhostLayerDataBlock(cornerSize, sourceOffset, destinationOffset, _patches[0], _patches[3]);
  } else if(_patches[0].getLevel() < _patches[3].getLevel() && _patches[3].getLevel() == _level && _patches[0].isLeaf()) {
    assertion2(_patches[0].getLevel() < _patches[3].getLevel(), _patches[0].getLevel(), _patches[3].getLevel());
    interpolate(cornerSize, destinationOffset, _patches[0], _patches[3], true, false);
  } else if(_patches[0].getLevel() > _patches[3].getLevel() && _patches[0].getLevel() == _level){
    assertion2(_patches[0].getLevel() > _patches[3].getLevel(), _patches[0].getLevel(), _patches[3].getLevel());
  }
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::fillRightGhostLayer() {
  int ghostLayerWidth = _patches[3].getGhostLayerWidth();
  tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = _patches[3].getSubdivisionFactor();
  tarch::la::Vector<DIMENSIONS, int> columnSize;
  columnSize(0) = ghostLayerWidth;
  columnSize(1) = subdivisionFactor(1);
  tarch::la::Vector<DIMENSIONS, int> destinationOffset;
  destinationOffset(0) = subdivisionFactor(0);
  destinationOffset(1) = 0;

  if(_patches[2].getLevel() == _patches[3].getLevel() && _patches[2].getLevel() == _level) {
    tarch::la::Vector<DIMENSIONS, int> sourceOffset;
    sourceOffset(0) = 0;
    sourceOffset(1) = 0;

    copyGhostLayerDataBlock(columnSize, sourceOffset, destinationOffset, _patches[2], _patches[3]);
  } else if(_patches[2].getLevel() < _patches[3].getLevel() && _patches[3].getLevel() == _level && _patches[2].isLeaf()) {
    assertion2(_patches[2].getLevel() < _patches[3].getLevel(), _patches[2].getLevel(), _patches[3].getLevel());
    interpolate(columnSize, destinationOffset, _patches[2], _patches[3], true, false);
  } else if(_patches[2].getLevel() > _patches[3].getLevel() && _patches[2].getLevel() == _level) {
    assertion2(_patches[2].getLevel() > _patches[3].getLevel(), _patches[2].getLevel(), _patches[3].getLevel());
    applyCoarseGridCorrection(_patches[2], _patches[3], 0, -1);
  }
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::fillLowerRightGhostLayer() {
//  assertionEquals(_patches[1].getGhostLayerWidth(), _patches[2].getGhostLayerWidth());
//  assertionEquals(_patches[1].getSubdivisionFactor(), _patches[2].getSubdivisionFactor());
  int ghostLayerWidth = _patches[1].getGhostLayerWidth();
  tarch::la::Vector<DIMENSIONS, int> subdivisionFactor = _patches[1].getSubdivisionFactor();
  tarch::la::Vector<DIMENSIONS, int> cornerSize(ghostLayerWidth);
  tarch::la::Vector<DIMENSIONS, int> destinationOffset;
  destinationOffset(0) = subdivisionFactor(0);
  destinationOffset(1) = -ghostLayerWidth;

  if(_patches[1].getLevel() == _patches[2].getLevel() && _patches[1].getLevel() == _level) {
    tarch::la::Vector<DIMENSIONS, int> sourceOffset;
    sourceOffset(0) = 0;
    sourceOffset(1) = subdivisionFactor(1) - ghostLayerWidth;

    copyGhostLayerDataBlock(cornerSize, sourceOffset, destinationOffset, _patches[2], _patches[1]);
  } else if(_patches[2].getLevel() < _patches[1].getLevel() && _patches[1].getLevel() == _level && _patches[2].isLeaf()) {
    assertion2(_patches[2].getLevel() < _patches[1].getLevel(), _patches[2].getLevel(), _patches[1].getLevel());
    interpolate(cornerSize, destinationOffset, _patches[2], _patches[1], true, false);
  } else if(_patches[2].getLevel() > _patches[1].getLevel() && _patches[2].getLevel() == _level) {
    assertion2(_patches[2].getLevel() > _patches[1].getLevel(), _patches[2].getLevel(), _patches[1].getLevel());
  }
}

double peanoclaw::interSubgridCommunication::GhostLayerCompositor::calculateOverlappingArea(
  tarch::la::Vector<DIMENSIONS, double> position1,
  tarch::la::Vector<DIMENSIONS, double> size1,
  tarch::la::Vector<DIMENSIONS, double> position2,
  tarch::la::Vector<DIMENSIONS, double> size2,
  int projectionAxis
) const {
  double area = 1.0;

  for(int d = 0; d < DIMENSIONS; d++) {
    if(d != projectionAxis) {
      double overlappingInterval =
          std::min(position1(d)+size1(d), position2(d)+size2(d))
            - std::max(position1(d), position2(d));
      area *= overlappingInterval;

      if(area < 0.0) {
        area = 0.0;
      }
    }
  }

  return area;
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

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::applyCoarseGridCorrection(
  const Patch& finePatch,
  Patch& coarsePatch,
  int dimension,
  int direction
) const {

  logTraceInWith4Arguments("applyCoarseGridCorrection", finePatch.toString(), coarsePatch.toString(), dimension, direction);

  //Create description of the fine patch's face to be traversed
  tarch::la::Vector<DIMENSIONS, int> face = finePatch.getSubdivisionFactor();
  face(dimension) = 1;
  tarch::la::Vector<DIMENSIONS, int> offset(0);
  if(direction == 1) {
    offset(dimension) = finePatch.getSubdivisionFactor()(dimension) - 1;
  }

  //Create search area that needs to be considered around the neighboring cell in the coarse patch
  tarch::la::Vector<DIMENSIONS, int> searchArea = tarch::la::Vector<DIMENSIONS, int>(2);
  searchArea(dimension) = 1;

  logDebug("applyCoarseGridCorrection", "face=" << face << ", offset=" << offset << ", searchArea=" << searchArea);

  tarch::la::Vector<DIMENSIONS, double> fineSubcellSize = finePatch.getSubcellSize();
  tarch::la::Vector<DIMENSIONS, double> coarseSubcellSize = coarsePatch.getSubcellSize();

  double refinementFactor = 1.0;
  for( int d = 0; d < DIMENSIONS; d++ ) {
    if( d != dimension ) {
      refinementFactor *= coarseSubcellSize(d) / fineSubcellSize(d);
    }
  }


  dfor(subcellIndexInFace, face) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndexInFinePatch = subcellIndexInFace + offset;
    tarch::la::Vector<DIMENSIONS, double> subcellPositionInFinePatch = finePatch.getSubcellPosition(subcellIndexInFinePatch);
    tarch::la::Vector<DIMENSIONS, double> neighboringSubcellPositionInCoarsePatch = subcellPositionInFinePatch;
    neighboringSubcellPositionInCoarsePatch(dimension) += fineSubcellSize(dimension) * direction;
    tarch::la::Vector<DIMENSIONS, int> neighboringSubcellIndexInCoarsePatch =
      (tarch::la::multiplyComponents((neighboringSubcellPositionInCoarsePatch.convertScalar<double>() -coarsePatch.getPosition()), tarch::la::invertEntries(coarseSubcellSize))).convertScalar<int>();

    logDebug("applyCoarseGridCorrection", "Correcting from cell " << subcellIndexInFinePatch);

    dfor(neighborOffset, searchArea) {
      tarch::la::Vector<DIMENSIONS, int> adjacentSubcellIndexInCoarsePatch = neighboringSubcellIndexInCoarsePatch + neighborOffset;

      logDebug("applyCoarseGridCorrection", "Correcting cell " << adjacentSubcellIndexInCoarsePatch);

      if(
        !tarch::la::oneGreater(tarch::la::Vector<DIMENSIONS, int>(0), adjacentSubcellIndexInCoarsePatch)
        && !tarch::la::oneGreaterEquals(adjacentSubcellIndexInCoarsePatch, coarsePatch.getSubdivisionFactor())
      ) {
        //Get interface area
        double interfaceArea = calculateOverlappingArea(
          finePatch.getSubcellPosition(subcellIndexInFinePatch),
          fineSubcellSize,
          coarsePatch.getSubcellPosition(adjacentSubcellIndexInCoarsePatch),
          coarseSubcellSize,
          dimension
        );

        //TODO This depends on the application, since we assume that variable 0 holds the height of the shallow water equation
        //and variables 1 and 2 hold the velocity
        //Estimate the flux through the interface from fine to coarse grid, once from the point of view of the fine subcell and
        //once from the of the coarse subcell
        double fineGridFlux = finePatch.getValueUNew(subcellIndexInFinePatch, 0) * finePatch.getValueUNew(subcellIndexInFinePatch, 1 + dimension) * interfaceArea;
        double coarseGridFlux = coarsePatch.getValueUNew(adjacentSubcellIndexInCoarsePatch, 0) * coarsePatch.getValueUNew(adjacentSubcellIndexInCoarsePatch, 1 + dimension) * interfaceArea;

        //Estimate the according transfered volume during the fine patch's timestep
        double transferedVolumeFineGrid = fineGridFlux * finePatch.getTimestepSize();
        double transferedVolumeCoarseGrid = coarseGridFlux * finePatch.getTimestepSize();

//        double delta = (transferedVolumeFineGrid * refinementFactor - transferedVolumeCoarseGrid) / coarseSubcellSize(dimension) / refinementFactor;
        double delta = (transferedVolumeFineGrid * refinementFactor - transferedVolumeCoarseGrid) / refinementFactor;

        logDebug("applyCoarseGridCorrection", "Correcting neighbor cell " << adjacentSubcellIndexInCoarsePatch << " from cell " << subcellIndexInFinePatch << " with interfaceArea=" << interfaceArea << std::endl
            << "\tu0=" << finePatch.getValueUNew(subcellIndexInFinePatch, 0) << " u1=" << finePatch.getValueUNew(subcellIndexInFinePatch, 1) << " u2=" << finePatch.getValueUNew(subcellIndexInFinePatch, 2) << std::endl
            << "\tfineGridFlux=" << fineGridFlux << std::endl
            << "\tcoarseGridFlux=" << coarseGridFlux << std::endl
            << "\ttransferedVolumeFineGrid=" << transferedVolumeFineGrid << std::endl
            << "\ttransferedVolumeCoarseGrid=" << transferedVolumeCoarseGrid << std::endl
            << "\tdelta=" << delta << std::endl
            << "\told u=" << coarsePatch.getValueUNew(adjacentSubcellIndexInCoarsePatch, 0) << std::endl
            << "\tnew u=" << coarsePatch.getValueUNew(adjacentSubcellIndexInCoarsePatch, 0) + delta);

//        assertion(tarch::la::greater(coarsePatch.getValueUNew(adjacentSubcellIndexInCoarsePatch, 0) + delta, 0.0));

        //Scaled down due to inaccuracy
        coarsePatch.setValueUNew(adjacentSubcellIndexInCoarsePatch, 0,
          std::max(0.0, coarsePatch.getValueUNew(adjacentSubcellIndexInCoarsePatch, 0) + delta * 10e-8));
      }
    }
  }

  logTraceOut("applyCoarseGridCorrection");
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::updateNeighborTime(int updatedPatchIndex, int neighborPatchIndex) {
  _patches[updatedPatchIndex].updateMinimalNeighborTimeConstraint(_patches[neighborPatchIndex].getTimeConstraint());
  _patches[updatedPatchIndex].updateMaximalNeighborTimeInterval(_patches[neighborPatchIndex].getCurrentTime(), _patches[neighborPatchIndex].getTimestepSize());

  if(_patches[neighborPatchIndex].isLeaf()) {
    _patches[updatedPatchIndex].updateMinimalLeafNeighborTimeConstraint(_patches[neighborPatchIndex].getTimeConstraint());
  }
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::fillGhostLayers(int destinationPatchIndex) {
  logTraceIn("fillGhostLayers()");

  if(destinationPatchIndex == -1 || destinationPatchIndex == 0) {
    //Copy from cell 1 to cell 0
    if(
        shouldTransferGhostlayerData(_patches[1], _patches[0])
    ) {
      fillLeftGhostLayer();
    }

    //Copy from cell 2 to cell 0
    if(
        shouldTransferGhostlayerData(_patches[2], _patches[0])
    ) {
      fillLowerGhostLayer();
    }

    //Copy from cell 3 to cell 0
    if(!_useDimensionalSplitting) {
      if(
          shouldTransferGhostlayerData(_patches[3], _patches[0])
      ) {
        fillLowerLeftGhostLayer();
      }
    }
  }

  //Copy from cell 2 to cell 1
  if(destinationPatchIndex == -1 || destinationPatchIndex == 1) {
    if(!_useDimensionalSplitting) {
        if(
            shouldTransferGhostlayerData(_patches[2], _patches[1])
        ) {
          fillLowerRightGhostLayer();
        }
      }
  }

  if(destinationPatchIndex == -1 || destinationPatchIndex == 2) {
    //Copy from cell 0 to cell 2
    if(
        shouldTransferGhostlayerData(_patches[0], _patches[2])
    ) {
      fillUpperGhostLayer();
    }

    //Copy from cell 1 to cell 2
    if(!_useDimensionalSplitting) {
      if(
          shouldTransferGhostlayerData(_patches[1], _patches[2])
      ) {
        fillUpperLeftGhostLayer();
      }
    }
  }

  if(destinationPatchIndex == -1 || destinationPatchIndex == 3) {
    //Copy from cell 0 to cell 3
    if(!_useDimensionalSplitting) {
      if(
          shouldTransferGhostlayerData(_patches[0], _patches[3])
      ) {
        fillUpperRightGhostLayer();
      }
    }

    //Copy from cell 2 to cell 3
    if(
        shouldTransferGhostlayerData(_patches[2], _patches[3])
    ) {
      fillRightGhostLayer();
    }
  }

  logTraceOut("fillGhostLayers()");
}

void peanoclaw::interSubgridCommunication::GhostLayerCompositor::updateNeighborTimes() {
  //Check cell 0 and 1
  if(_patches[0].isValid()
      && _patches[1].isValid()) {
    updateNeighborTime(0, 1);
    updateNeighborTime(1, 0);
  }

  //Check cell 0 and 2
  if(_patches[0].isValid()
      && _patches[2].isValid()) {
    updateNeighborTime(0, 2);
    updateNeighborTime(2, 0);
  }

  //Check cell 0 and 3
  if(!_useDimensionalSplitting) {
    if(_patches[0].isValid()
        && _patches[3].isValid()) {
      updateNeighborTime(0, 3);
      updateNeighborTime(3, 0);
    }
  }

  //Check cell 1 and 2
  if(!_useDimensionalSplitting) {
    if(_patches[1].isValid()
        && _patches[2].isValid()) {
      updateNeighborTime(1, 2);
      updateNeighborTime(2, 1);
    }
  }

  //Check cell 1 and 3
  if(_patches[1].isValid()
      && _patches[3].isValid()) {
    updateNeighborTime(1, 3);
    updateNeighborTime(3, 1);
  }

  //Check cell 2 and 3
  if(_patches[2].isValid()
      && _patches[3].isValid()) {
    updateNeighborTime(2, 3);
    updateNeighborTime(3, 2);
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
        applyCoarseGridCorrection(_patches[leftPatchIndex], _patches[rightPatchIndex], d, -1);
      }

      //Correct from right to left
      if(_patches[rightPatchIndex].getLevel() > _patches[leftPatchIndex].getLevel()) {
        applyCoarseGridCorrection(_patches[rightPatchIndex], _patches[leftPatchIndex], d, 1);
      }
    }
  }
}

bool peanoclaw::interSubgridCommunication::GhostLayerCompositor::shouldTransferGhostlayerData(Patch& source, Patch& destination) {
  return destination.isLeaf()
            && (((source.isVirtual() || source.isLeaf()) && !tarch::la::greater(destination.getCurrentTime() + destination.getTimestepSize(),
                                  source.getCurrentTime() + source.getTimestepSize()))
            || (source.isLeaf() && source.isAllowedToAdvanceInTime()));
}