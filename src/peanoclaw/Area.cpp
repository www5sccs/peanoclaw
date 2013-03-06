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
  destinationArea._offset = tarch::la::multiplyComponents((position+epsilon).convertScalar<double>(), tarch::la::invertEntries(destination.getSubcellSize())).convertScalar<int>();

  //Size
  tarch::la::Vector<DIMENSIONS, double> size = tarch::la::multiplyComponents(sourceSubcellSize, (_offset + _size).convertScalar<double>());
  size += (source.getPosition() - destination.getPosition());
  destinationArea._size = (tarch::la::multiplyComponents((size - epsilon).convertScalar<double>(), tarch::la::invertEntries(destination.getSubcellSize())) - destinationArea._offset.convertScalar<double>() + 1.0).convertScalar<int>();

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

  cellArea._offset = tarch::la::multiplyComponents((coarseSubcellPosition - finePosition + epsilon), tarch::la::invertEntries(fineSubcellSize)).convertScalar<int>();
  cellArea._size = (tarch::la::multiplyComponents((coarseSubcellPosition + coarseSubcellSize - finePosition - epsilon), tarch::la::invertEntries(fineSubcellSize)) - cellArea._offset.convertScalar<double>() + 1.0).convertScalar<int>();

  tarch::la::Vector<DIMENSIONS, int> cellAreaUpperBound = cellArea._offset + cellArea._size;
  tarch::la::Vector<DIMENSIONS, int> areaUpperBound = _offset + _size;
  for(int d = 0; d < DIMENSIONS; d++) {
    cellArea._offset(d) = std::max(cellArea._offset(d), _offset(d));
    cellArea._size(d) = std::min(cellAreaUpperBound(d), areaUpperBound(d));
  }
  cellArea._size -= cellArea._offset;

  return cellArea;
}

std::ostream& operator<<(std::ostream& out, const peanoclaw::Area& area){
  out << "offset=[" << area._offset << "],size=[" << area._size << "]" << std::endl;
  return out;
}

