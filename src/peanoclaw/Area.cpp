/*
 * Area.cpp
 *
 *  Created on: Jan 28, 2013
 *      Author: kristof
 */
#include "Area.h"

#include "Patch.h"

namespace peano {
  namespace applications {
    namespace peanoclaw {
      class Area;
    }
  }
}

peanoclaw::Area::Area() {
}

peanoclaw::Area::Area(
  tarch::la::Vector<DIMENSIONS, int> offset,
  tarch::la::Vector<DIMENSIONS, int> size
) : _offset(offset), _size(size) {
}

peanoclaw::Area peanoclaw::Area::mapToPatch(
  const Patch& source,
  const Patch& destination,
  double epsilon
) const {
  Area destinationArea;

  //Offset
  tarch::la::Vector<DIMENSIONS, double> sourceSubcellSize = source.getSubcellSize();
  tarch::la::Vector<DIMENSIONS, double> position = tarch::la::multiplyComponents(sourceSubcellSize, _offset.convertScalar<double>());
  position += source.getPosition() - destination.getPosition();

  tarch::la::Vector<DIMENSIONS, double> offsetTemp = (position+epsilon).convertScalar<double>();
  for (int i=0; i < DIMENSIONS; i++) {
    offsetTemp[i] /= destination.getSubcellSize()[i];
  }
  destinationArea._offset = offsetTemp.convertScalar<int>();

  //Size
  tarch::la::Vector<DIMENSIONS, double> size = tarch::la::multiplyComponents(sourceSubcellSize, (_offset + _size).convertScalar<double>());
  size += (source.getPosition() - destination.getPosition());
  
  tarch::la::Vector<DIMENSIONS, double> sizeTemp = (size - epsilon).convertScalar<double>();
  for (int i=0; i < DIMENSIONS; i++) {
    sizeTemp(i) /= destination.getSubcellSize()(i);
  }

  sizeTemp -= destinationArea._offset.convertScalar<double>();
  sizeTemp += 1.0;
  destinationArea._size = sizeTemp.convertScalar<int>();

  return destinationArea;
}

peanoclaw::Area peanoclaw::Area::mapCellToPatch(
  const tarch::la::Vector<DIMENSIONS, double>& finePosition,
  const tarch::la::Vector<DIMENSIONS, double>& fineSubcellSize,
  const tarch::la::Vector<DIMENSIONS, double>& coarseSubcellSize,
  const tarch::la::Vector<DIMENSIONS, int>& coarseSubcellIndex,
  const tarch::la::Vector<DIMENSIONS, double>& coarseSubcellPosition,
  const double& epsilon
) const {
  Area cellArea;

  //cellArea._offset = ((coarseSubcellPosition - finePosition + epsilon) / fineSubcellSize).convertScalar<int>();
  tarch::la::Vector<DIMENSIONS, double> offsetTemp = (coarseSubcellPosition - finePosition + epsilon);
  for (int d=0; d < DIMENSIONS; d++) {
    offsetTemp[d] /= fineSubcellSize[d];
  }
  cellArea._offset = offsetTemp.convertScalar<int>();

 
  //cellArea._size = ((coarseSubcellPosition + coarseSubcellSize - finePosition - epsilon) / fineSubcellSize - cellArea._offset.convertScalar<double>() + 1.0).convertScalar<int>();
  tarch::la::Vector<DIMENSIONS, double> sizeTemp = coarseSubcellPosition + coarseSubcellSize - finePosition - epsilon;
  for (int d=0; d < DIMENSIONS; d++) {
    sizeTemp[d] /= fineSubcellSize[d];
  }
  sizeTemp -= offsetTemp;
  sizeTemp += 1.0;
  cellArea._size = sizeTemp.convertScalar<int>();

  tarch::la::Vector<DIMENSIONS, int> cellAreaUpperBound = cellArea._offset + cellArea._size;
  tarch::la::Vector<DIMENSIONS, int> areaUpperBound = _offset + _size;
  for(int d = 0; d < DIMENSIONS; d++) {
    cellArea._offset(d) = std::max(cellArea._offset(d), _offset(d));
    cellArea._size(d) = std::min(cellAreaUpperBound(d), areaUpperBound(d));
  }
  cellArea._size -= cellArea._offset;

  return cellArea;
}


int peanoclaw::Area::getAreasOverlappedByNeighboringGhostlayers (
  const tarch::la::Vector<DIMENSIONS, double>& lowerNeighboringGhostlayerBounds,
  const tarch::la::Vector<DIMENSIONS, double>& upperNeighboringGhostlayerBounds,
  const tarch::la::Vector<DIMENSIONS, double>& sourcePosition,
  const tarch::la::Vector<DIMENSIONS, double>& sourceSize,
  const tarch::la::Vector<DIMENSIONS, double>& sourceSubcellSize,
  const tarch::la::Vector<DIMENSIONS, int>& sourceSubdivisionFactor,
  Area areas[DIMENSIONS_TIMES_TWO]
) {

  //Check if bounds overlap
  if(tarch::la::oneGreater(upperNeighboringGhostlayerBounds + sourceSubcellSize, lowerNeighboringGhostlayerBounds)
    || tarch::la::oneGreaterEquals(upperNeighboringGhostlayerBounds, sourcePosition + sourceSize)
    || !tarch::la::allGreater(lowerNeighboringGhostlayerBounds, sourcePosition)) {
    //If whole patch is overlapped -> One area holds whole patch, others are empty
    areas[0]._offset = tarch::la::Vector<DIMENSIONS, int>(0);
    areas[0]._size = sourceSubdivisionFactor;
    areas[1]._offset = tarch::la::Vector<DIMENSIONS, int>(0);
    areas[1]._size = tarch::la::Vector<DIMENSIONS, int>(0);
    for(int d = 1; d < DIMENSIONS; d++) {
      areas[2*d]._offset = tarch::la::Vector<DIMENSIONS, int>(0);
      areas[2*d]._size = tarch::la::Vector<DIMENSIONS, int>(0);
      areas[2*d + 1]._offset = tarch::la::Vector<DIMENSIONS, int>(0);
      areas[2*d + 1]._size = tarch::la::Vector<DIMENSIONS, int>(0);
    }
    return 1;
  } else {
    double epsilon = 1e-12;
    tarch::la::Vector<DIMENSIONS, double> doubleLowerBoundsInSourcePatch = tarch::la::multiplyComponents((lowerNeighboringGhostlayerBounds - sourcePosition), tarch::la::invertEntries(sourceSubcellSize)) + epsilon;
    tarch::la::Vector<DIMENSIONS, double> doubleUpperBoundsInSourcePatch = tarch::la::multiplyComponents((upperNeighboringGhostlayerBounds - sourcePosition), tarch::la::invertEntries(sourceSubcellSize)) - epsilon;
    tarch::la::Vector<DIMENSIONS, int> lowerBoundsInSourcePatch;
    tarch::la::Vector<DIMENSIONS, int> upperBoundsInSourcePatch;

    for(int d = 0; d < DIMENSIONS; d++) {
      //TODO unterweg: The constants +/-1e10 are choosen sufficiently large -> need a good heuristic for this...
      doubleLowerBoundsInSourcePatch(d) = std::min((double)std::numeric_limits<int>::max(), doubleLowerBoundsInSourcePatch(d));
      doubleUpperBoundsInSourcePatch(d) = std::max((double)std::numeric_limits<int>::min(), doubleUpperBoundsInSourcePatch(d));
      lowerBoundsInSourcePatch(d) = std::max(std::min((int)std::floor(doubleLowerBoundsInSourcePatch(d)), sourceSubdivisionFactor(d)), -1);
      upperBoundsInSourcePatch(d) = std::max(std::min((int)std::floor(doubleUpperBoundsInSourcePatch(d)), sourceSubdivisionFactor(d)), -1);
    }

    for(int d = 0; d < DIMENSIONS; d++) {
      areas[2*d]._size(d) = upperBoundsInSourcePatch(d) + 1;
      areas[2*d]._offset(d) = 0;
      areas[2*d+1]._size(d) = sourceSubdivisionFactor(d) - lowerBoundsInSourcePatch(d);
      areas[2*d+1]._offset(d) = lowerBoundsInSourcePatch(d);

      //Restricted on lower dimensions
      for(int i = 0; i < d; i++) {
        areas[2*d]._size(i) = -upperBoundsInSourcePatch(i) - 1 + lowerBoundsInSourcePatch(i);
        areas[2*d]._offset(i) = upperBoundsInSourcePatch(i) + 1;
        areas[2*d+1]._size(i) = areas[2*d]._size(i);
        areas[2*d+1]._offset(i) = areas[2*d]._offset(i);
      }
      //Spread over whole patch in higher dimensions
      for(int i = d + 1; i < DIMENSIONS; i++) {
        areas[2*d]._size(i) = sourceSubdivisionFactor(i);
        areas[2*d]._offset(i) = 0;
        areas[2*d+1]._size(i) = sourceSubdivisionFactor(i);
        areas[2*d+1]._offset(i) = 0;
      }

      assertion1(tarch::la::allGreaterEquals(areas[2*d]._size, 0), areas[2*d]);
      assertion1(tarch::la::allGreaterEquals(areas[2*d+1]._size, 0), areas[2*d+1]);
    }

    return DIMENSIONS_TIMES_TWO;
  }
}

std::ostream& operator<<(std::ostream& out, const peanoclaw::Area& area){
  out << "offset=[" << area._offset << "],size=[" << area._size << "]" << std::endl;
  return out;
}

