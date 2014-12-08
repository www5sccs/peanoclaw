/*
 * Region.cpp
 *
 *  Created on: Jan 28, 2013
 *      Author: kristof
 */
#include "peanoclaw/geometry/Region.h"

#include "peanoclaw/Patch.h"

tarch::logging::Log peanoclaw::geometry::Region::_log("peanoclaw::geometry::Region");

int peanoclaw::geometry::Region::factorial(int i) {
  int f = 1;
  while(i > 1) {
    f *= i--;
  }
  return f;
}

void peanoclaw::geometry::Region::incrementIndices(tarch::la::Vector<DIMENSIONS, int>& indices, int highestIndex, int upperBound) {
  int i = highestIndex;
  do {
    indices(i)++;
    if(i < DIMENSIONS-1) {
      indices(i+1) = indices(i) + 1;
    }
    i--;
  } while(indices(i+1) >= upperBound && i>=0);
}

int peanoclaw::geometry::Region::getNumberOfManifolds(int dimensionality) {
  #ifdef Dim2
  return 4;
  #elif Dim3
  int s = (DIMENSIONS - dimensionality);
  return pow(2, s) * factorial(DIMENSIONS) / (factorial(s) * factorial(DIMENSIONS-s));
  #endif
}

tarch::la::Vector<DIMENSIONS, int> peanoclaw::geometry::Region::getManifold(int dimensionality, int index) {
  tarch::la::Vector<DIMENSIONS, int> modifyIndices(0);
  for(int d = 0; d < DIMENSIONS; d++) {
    modifyIndices(d) = d;
  }

  int s = DIMENSIONS - dimensionality;
  int combinations = pow(2, s);
  while(index >= combinations) {
    incrementIndices(modifyIndices, s-1, DIMENSIONS);
    index-=combinations;
  }

  tarch::la::Vector<DIMENSIONS, int> manifoldPosition(0);
  for(int i = 0; i < s; i++) {
    int newEntry = ((index & (1<<i)) != 0) ? 1 : -1;
    manifoldPosition(modifyIndices(i)) = newEntry;
  }
  return manifoldPosition;
}

int peanoclaw::geometry::Region::getNumberOfAdjacentManifolds(
  const tarch::la::Vector<DIMENSIONS, int>& manifoldPosition,
  int dimensionality,
  int adjacentDimensionality
) {
  assertion1(dimensionality != adjacentDimensionality, dimensionality);

  int n = 0;
  for(int d = 0; d < DIMENSIONS; d++) {
    n += (manifoldPosition(d) == 0) ? 1 : 0;
  }

  int s = std::abs(dimensionality - adjacentDimensionality);
  if(dimensionality < adjacentDimensionality) {
    return factorial(DIMENSIONS - n) / (factorial(s) * factorial(DIMENSIONS-n-s));
  } else {
    return std::pow(2.0, factorial(n) / (factorial(s) * factorial(n-s)) * s);
  }
}

tarch::la::Vector<DIMENSIONS, int> peanoclaw::geometry::Region::getIndexOfAdjacentManifold(
  tarch::la::Vector<DIMENSIONS, int> manifoldPosition,
  int dimensionality,
  int adjacentDimensionality,
  int adjacentIndex
) {
  tarch::la::Vector<DIMENSIONS, int> zeroIndices(0);
  tarch::la::Vector<DIMENSIONS, int> nonzeroIndices(0);
  tarch::la::Vector<DIMENSIONS, int> modifyIndices(0);

  int s = std::abs(dimensionality - adjacentDimensionality);
  int n = 0;
  for(int d = 0; d < DIMENSIONS; d++) {
    if(manifoldPosition(d) == 0) {
      zeroIndices(n++) = d;
    } else {
      nonzeroIndices(d - n) = d;
    }
  }
  for(int i = 0; i < s; i++) {
    modifyIndices(i) = i;
  }

  if(dimensionality < adjacentDimensionality) {
    while(adjacentIndex > 0) {
      incrementIndices(modifyIndices, s-1, DIMENSIONS - n);
      adjacentIndex--;
    }
    for(int i = 0; i < s; i++) {
      manifoldPosition(nonzeroIndices(modifyIndices(i))) = 0;
    }
    return manifoldPosition;
  } else {
    int combinations = pow(2, s);
    while(adjacentIndex >= combinations) {
      incrementIndices(modifyIndices, s-1, DIMENSIONS - n);
      adjacentIndex -= combinations;
    }
    for(int i = 0; i < s; i++) {
      int newEntry = ((adjacentIndex & (1<<i)) != 0) ? 1 : -1;
      manifoldPosition(zeroIndices(modifyIndices(i))) = newEntry;
    }

    return manifoldPosition;
  }
}

bool peanoclaw::geometry::Region::checkHigherDimensionalManifoldForOverlap(
  const tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int>& adjacentRanks,
  tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int>&       overlapOfRemoteGhostlayers,
  const tarch::la::Vector<DIMENSIONS, int>& manifoldPosition,
  int dimensionality,
  int manifoldEntry,
  int rank
) {
  for(int adjacentDimensionality = dimensionality + 1; adjacentDimensionality < DIMENSIONS; adjacentDimensionality++) {
    int numberOfAdjacentManifolds = getNumberOfAdjacentManifolds(manifoldPosition, dimensionality, adjacentDimensionality);
    for(int adjacentManifoldIndex = 0; adjacentManifoldIndex < numberOfAdjacentManifolds; adjacentManifoldIndex++) {
      tarch::la::Vector<DIMENSIONS, int> adjacentManifoldPosition = getIndexOfAdjacentManifold(manifoldPosition, dimensionality, adjacentDimensionality, adjacentManifoldIndex);
      int adjacentManifoldEntry = linearizeManifoldPosition(adjacentManifoldPosition);

      if(adjacentRanks[adjacentManifoldEntry] == rank && overlapOfRemoteGhostlayers[adjacentManifoldEntry] >= overlapOfRemoteGhostlayers[manifoldEntry]){
        overlapOfRemoteGhostlayers[manifoldEntry] = 0;
        return true;
      }
    }
  }

  return false;
}

peanoclaw::geometry::Region::Region() {
}

peanoclaw::geometry::Region::Region(
  tarch::la::Vector<DIMENSIONS, int> offset,
  tarch::la::Vector<DIMENSIONS, int> size
) : _offset(offset), _size(size) {
}

int peanoclaw::geometry::Region::linearizeManifoldPosition(
  const tarch::la::Vector<DIMENSIONS, int>& manifoldPosition
) {
  assertion1(tarch::la::allSmaller(manifoldPosition, 2), manifoldPosition);

  int entry = peano::utils::dLinearised(manifoldPosition + 1, 3);
  if(entry > THREE_POWER_D_MINUS_ONE/2) {
    entry--;
  }
  return entry;
}

peanoclaw::geometry::Region peanoclaw::geometry::Region::mapToPatch(
  const Patch& source,
  const Patch& destination,
  double epsilon
) const {
  Region destinationRegion;

  //Offset
  tarch::la::Vector<DIMENSIONS, double> sourceSubcellSize = source.getSubcellSize();
  tarch::la::Vector<DIMENSIONS, double> position = tarch::la::multiplyComponents(sourceSubcellSize, _offset.convertScalar<double>());
  position += source.getPosition() - destination.getPosition();

  tarch::la::Vector<DIMENSIONS, double> offsetTemp = (position+epsilon).convertScalar<double>();
  for (int i=0; i < DIMENSIONS; i++) {
    offsetTemp[i] /= destination.getSubcellSize()[i];
  }
  destinationRegion._offset = offsetTemp.convertScalar<int>();

  //Size
  tarch::la::Vector<DIMENSIONS, double> size = tarch::la::multiplyComponents(sourceSubcellSize, (_offset + _size).convertScalar<double>());
  size += (source.getPosition() - destination.getPosition());
  
  tarch::la::Vector<DIMENSIONS, double> sizeTemp = (size - epsilon).convertScalar<double>();
  for (int i=0; i < DIMENSIONS; i++) {
    sizeTemp(i) /= destination.getSubcellSize()(i);
  }

  sizeTemp -= destinationRegion._offset.convertScalar<double>();
  sizeTemp += 1.0;
  destinationRegion._size = sizeTemp.convertScalar<int>();

  return destinationRegion;
}

peanoclaw::geometry::Region peanoclaw::geometry::Region::mapCellToPatch(
  const tarch::la::Vector<DIMENSIONS, double>& finePosition,
  const tarch::la::Vector<DIMENSIONS, double>& fineSubcellSize,
  const tarch::la::Vector<DIMENSIONS, double>& coarseSubcellSize,
  const tarch::la::Vector<DIMENSIONS, int>& coarseSubcellIndex,
  const tarch::la::Vector<DIMENSIONS, double>& coarseSubcellPosition,
  const double& epsilon
) const {
  Region cellRegion;

  //cellRegion._offset = ((coarseSubcellPosition - finePosition + epsilon) / fineSubcellSize).convertScalar<int>();
  tarch::la::Vector<DIMENSIONS, double> offsetTemp = (coarseSubcellPosition - finePosition + epsilon);
  for (int d=0; d < DIMENSIONS; d++) {
    offsetTemp[d] /= fineSubcellSize[d];
  }
  cellRegion._offset = offsetTemp.convertScalar<int>();

 
  //cellRegion._size = ((coarseSubcellPosition + coarseSubcellSize - finePosition - epsilon) / fineSubcellSize - cellRegion._offset.convertScalar<double>() + 1.0).convertScalar<int>();
  tarch::la::Vector<DIMENSIONS, double> sizeTemp = coarseSubcellPosition + coarseSubcellSize - finePosition - epsilon;
  for (int d=0; d < DIMENSIONS; d++) {
    sizeTemp[d] /= fineSubcellSize[d];
  }
  sizeTemp -= offsetTemp;
  sizeTemp += 1.0;
  cellRegion._size = sizeTemp.convertScalar<int>();

  tarch::la::Vector<DIMENSIONS, int> cellRegionUpperBound = cellRegion._offset + cellRegion._size;
  tarch::la::Vector<DIMENSIONS, int> regionUpperBound = _offset + _size;
  for(int d = 0; d < DIMENSIONS; d++) {
    cellRegion._offset(d) = std::max(cellRegion._offset(d), _offset(d));
    cellRegion._size(d) = std::min(cellRegionUpperBound(d), regionUpperBound(d));
  }
  cellRegion._size -= cellRegion._offset;

  return cellRegion;
}


int peanoclaw::geometry::Region::getRegionsOverlappedByNeighboringGhostlayers (
  const tarch::la::Vector<DIMENSIONS, double>& lowerNeighboringGhostlayerBounds,
  const tarch::la::Vector<DIMENSIONS, double>& upperNeighboringGhostlayerBounds,
  const tarch::la::Vector<DIMENSIONS, double>& sourcePosition,
  const tarch::la::Vector<DIMENSIONS, double>& sourceSize,
  const tarch::la::Vector<DIMENSIONS, double>& sourceSubcellSize,
  const tarch::la::Vector<DIMENSIONS, int>& sourceSubdivisionFactor,
  Region regions[DIMENSIONS_TIMES_TWO]
) {

  //Check if bounds overlap
  if(tarch::la::oneGreater(upperNeighboringGhostlayerBounds + sourceSubcellSize, lowerNeighboringGhostlayerBounds)
    || tarch::la::oneGreaterEquals(upperNeighboringGhostlayerBounds, sourcePosition + sourceSize)
    || !tarch::la::allGreater(lowerNeighboringGhostlayerBounds, sourcePosition)) {
    //If whole patch is overlapped -> One region holds whole patch, others are empty
    regions[0]._offset = tarch::la::Vector<DIMENSIONS, int>(0);
    regions[0]._size = sourceSubdivisionFactor;
    regions[1]._offset = tarch::la::Vector<DIMENSIONS, int>(0);
    regions[1]._size = tarch::la::Vector<DIMENSIONS, int>(0);
    for(int d = 1; d < DIMENSIONS; d++) {
      regions[2*d]._offset = tarch::la::Vector<DIMENSIONS, int>(0);
      regions[2*d]._size = tarch::la::Vector<DIMENSIONS, int>(0);
      regions[2*d + 1]._offset = tarch::la::Vector<DIMENSIONS, int>(0);
      regions[2*d + 1]._size = tarch::la::Vector<DIMENSIONS, int>(0);
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
      regions[2*d]._size(d) = upperBoundsInSourcePatch(d) + 1;
      regions[2*d]._offset(d) = 0;
      regions[2*d+1]._size(d) = sourceSubdivisionFactor(d) - lowerBoundsInSourcePatch(d);
      regions[2*d+1]._offset(d) = lowerBoundsInSourcePatch(d);

      //Restricted on lower dimensions
      for(int i = 0; i < d; i++) {
        regions[2*d]._size(i) = -upperBoundsInSourcePatch(i) - 1 + lowerBoundsInSourcePatch(i);
        regions[2*d]._offset(i) = upperBoundsInSourcePatch(i) + 1;
        regions[2*d+1]._size(i) = regions[2*d]._size(i);
        regions[2*d+1]._offset(i) = regions[2*d]._offset(i);
      }
      //Spread over whole patch in higher dimensions
      for(int i = d + 1; i < DIMENSIONS; i++) {
        regions[2*d]._size(i) = sourceSubdivisionFactor(i);
        regions[2*d]._offset(i) = 0;
        regions[2*d+1]._size(i) = sourceSubdivisionFactor(i);
        regions[2*d+1]._offset(i) = 0;
      }

      assertion1(tarch::la::allGreaterEquals(regions[2*d]._size, 0), regions[2*d]);
      assertion1(tarch::la::allGreaterEquals(regions[2*d+1]._size, 0), regions[2*d+1]);
    }

    return DIMENSIONS_TIMES_TWO;
  }
}

int peanoclaw::geometry::Region::getRegionsOverlappedByRemoteGhostlayers(
  const tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int>& adjacentRanks,
  tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int>        overlapOfRemoteGhostlayers,
  const tarch::la::Vector<DIMENSIONS, int>&              subdivisionFactor,
  int                                                    rank,
  Region                                                 regions[THREE_POWER_D_MINUS_ONE]
) {
  logTraceInWith2Arguments("getRegionsOverlappedByRemoteGhostlayers(...)", adjacentRanks, overlapOfRemoteGhostlayers);
  int  numberOfRegions = 0;
  bool oneRegionCoversCompleteSubgrid = false;

  for(int dimensionality = 0; dimensionality < DIMENSIONS; dimensionality++) {
    int numberOfManifolds = getNumberOfManifolds(dimensionality);
    for(int manifoldIndex = 0; manifoldIndex < numberOfManifolds; manifoldIndex++) {
      tarch::la::Vector<DIMENSIONS, int> manifoldPosition = getManifold(dimensionality, manifoldIndex);
      int manifoldEntry = linearizeManifoldPosition(manifoldPosition);

      logDebug("getRegionsOverlappedByRemoteGhostlayers(...)", "Manifold " << manifoldPosition << " of dimensions " << dimensionality
          << ": entry=" << manifoldEntry << ", rank=" << adjacentRanks[manifoldEntry] << ", overlap=" << overlapOfRemoteGhostlayers[manifoldEntry]);

      if(adjacentRanks[manifoldEntry] == rank && overlapOfRemoteGhostlayers[manifoldEntry] > 0) {
        //Reduce lower-dimensional manifolds
        bool canBeOmitted = checkHigherDimensionalManifoldForOverlap(
          adjacentRanks,
          overlapOfRemoteGhostlayers,
          manifoldPosition,
          dimensionality,
          manifoldEntry,
          rank
        );

        //Restrict size and offset by higher-dimensional manifolds
        logDebug("getRegionsOverlappedByRemoteGhostlayers(...)","Testing manifold " << manifoldPosition << " of dimensions " << dimensionality
            << ": " << (canBeOmitted ? "omitting" : "consider"));

        if(!canBeOmitted) {
          tarch::la::Vector<DIMENSIONS, int> size;
          tarch::la::Vector<DIMENSIONS, int> offset;

          //Initialise size and offset
          for(int d = 0; d < DIMENSIONS; d++) {
            size(d) = (manifoldPosition(d) == 0) ? subdivisionFactor(d) : std::min(subdivisionFactor(d), overlapOfRemoteGhostlayers[manifoldEntry]);
            offset(d) = (manifoldPosition(d) == 1) ? subdivisionFactor(d) - size(d) : 0;
          }

          //TODO unterweg debug
//          std::cout << "offset: " << offset << ", size: " << size << std::endl;

          for(int adjacentDimensionality = dimensionality - 1; adjacentDimensionality >= 0; adjacentDimensionality--) {
            int numberOfAdjacentManifolds = getNumberOfAdjacentManifolds(manifoldPosition, dimensionality, adjacentDimensionality);
            for(int adjacentManifoldIndex = 0; adjacentManifoldIndex < numberOfAdjacentManifolds; adjacentManifoldIndex++) {
              tarch::la::Vector<DIMENSIONS, int> adjacentManifoldPosition = getIndexOfAdjacentManifold(
                manifoldPosition,
                dimensionality,
                adjacentDimensionality,
                adjacentManifoldIndex
              );

              //TODO unterweg debug
//              std::cout << "adj. manifold " << adjacentManifoldPosition << std::endl;

              int adjacentEntry = linearizeManifoldPosition(adjacentManifoldPosition);

              if(adjacentRanks[adjacentEntry] == rank) {
                for(int d = 0; d < DIMENSIONS; d++) {
                  if(manifoldPosition(d) == 0) {
                    if(adjacentManifoldPosition(d) < 0) {
                      int overlap = std::max(0, overlapOfRemoteGhostlayers[adjacentEntry] - offset(d));
                      offset(d) += overlap;
                      size(d) -= overlap;

                      //TODO unterweg debug
//                      std::cout << "Reducing bottom " << overlap << std::endl;
                    } else if(adjacentManifoldPosition(d) > 0) {
                      assertion2(adjacentManifoldPosition(d) > 0, adjacentManifoldPosition, d);
                      int overlap = std::max(0, offset(d) + size(d) - (subdivisionFactor(d) - overlapOfRemoteGhostlayers[adjacentEntry]));
                      size(d) -= overlap;

                      //TODO unterweg debug
//                      std::cout << "Reducing top " << overlap << std::endl;
                    }
                  }
                }
              }
            }
          }

          logDebug("getRegionsOverlappedByRemoteGhostlayers(...)", "offset: " << offset << ", size: " << size << ", dimensionality=" << dimensionality << ", manifoldIndex=" << manifoldIndex << ", manifoldEntry=" << manifoldEntry);
          if(tarch::la::allGreater(size, 0)) {
            regions[numberOfRegions++] = Region(offset, size);

            oneRegionCoversCompleteSubgrid |= (tarch::la::volume(size) >= tarch::la::volume(subdivisionFactor));

            assertion1(tarch::la::allGreaterEquals(offset, 0), offset);
            assertion3(tarch::la::allGreaterEquals(subdivisionFactor, offset + size), offset, size, subdivisionFactor);
          }
        }
      }
    }
  }

  //Optimize regions
  if(oneRegionCoversCompleteSubgrid) {
    numberOfRegions = 1;
    regions[0] = Region(0, subdivisionFactor);
  }

  logTraceOutWith2Arguments("getRegionsOverlappedByRemoteGhostlayers(...)", numberOfRegions, regions);
  return numberOfRegions;
}

std::ostream& operator<<(std::ostream& out, const peanoclaw::geometry::Region& region){
  out << "offset=[" << region._offset << "],size=[" << region._size << "]" << std::endl;
  return out;
}

