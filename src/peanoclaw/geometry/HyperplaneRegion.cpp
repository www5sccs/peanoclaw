/*
 * HyperplaneRegion.cpp
 *
 *  Created on: Dec 5, 2014
 *      Author: kristof
 */

#include "peanoclaw/geometry/HyperplaneRegion.h"

#include "peanoclaw/Patch.h"

double peanoclaw::geometry::HyperplaneRegion::epsilon = 1e-10;

peanoclaw::geometry::HyperplaneRegion::HyperplaneRegion(
  const tarch::la::Vector<DIMENSIONS_MINUS_ONE,int>& offset,
  const tarch::la::Vector<DIMENSIONS_MINUS_ONE,int>& size,
  int dimension,
  int direction
) : _offset(offset),
    _size(size),
    _dimension(dimension),
    _direction(direction)
{
}

peanoclaw::geometry::HyperplaneRegion peanoclaw::geometry::HyperplaneRegion::getInterfaceRegion(
  const Patch& subgrid,
  const Patch& neighboringSubgrid
) {
  tarch::la::Vector<DIMENSIONS,double> position = subgrid.getPosition();
  tarch::la::Vector<DIMENSIONS,double> subgridSize = subgrid.getSize();
  tarch::la::Vector<DIMENSIONS,double> subcellSize = subgrid.getSubcellSize();
  tarch::la::Vector<DIMENSIONS,double> neighborPosition = neighboringSubgrid.getPosition();
  tarch::la::Vector<DIMENSIONS,double> neighborSize = neighboringSubgrid.getSize();

  int dimension;
  int direction;

  for(int d = 0; d < DIMENSIONS; d++) {
    if(tarch::la::smaller(position[d], neighborPosition[d])) {
      dimension = d;
      direction = 1;
    } else if(tarch::la::greater(position[d], neighborPosition[d])) {
      dimension = d;
      direction = -1;
    }
  }

  tarch::la::Vector<DIMENSIONS_MINUS_ONE, int> offset;
  tarch::la::Vector<DIMENSIONS_MINUS_ONE, int> size;
  for(int d = 0; d < DIMENSIONS - 1; d++) {
    int subgridDimension = peanoclaw::grid::Linearization::getGlobalDimension(d, dimension);
    double lowerBound = std::max(position[subgridDimension], neighborPosition[subgridDimension]);
    double upperBound = std::min(
                          position[subgridDimension] + subgridSize[subgridDimension],
                          neighborPosition[subgridDimension] + neighborSize[subgridDimension]
                        );
    int lowerBoundCells = (lowerBound - position[subgridDimension] + epsilon) / subcellSize[subgridDimension];
    int upperBoundCells = std::ceil((upperBound - position[subgridDimension] - epsilon) / subcellSize[subgridDimension]);

    offset[d] = lowerBoundCells;
    size[d] = upperBoundCells - lowerBoundCells;
  }

  return HyperplaneRegion(offset, size, dimension, direction);
}

peanoclaw::geometry::HyperplaneRegion peanoclaw::geometry::HyperplaneRegion::getInterfaceRegionForSubcell(
  const Patch& subgrid,
  const Patch& neighboringSubgrid,
  tarch::la::Vector<DIMENSIONS_MINUS_ONE,int> neighborSubcellIndex,
  int projectionDimension,
  int direction
) {
  tarch::la::Vector<DIMENSIONS,double> position = subgrid.getPosition();
  tarch::la::Vector<DIMENSIONS,double> subgridSize = subgrid.getSubcellSize();
  tarch::la::Vector<DIMENSIONS,double> subcellSize = subgrid.getSubcellSize();
  tarch::la::Vector<DIMENSIONS,double> neighborPosition = neighboringSubgrid.getPosition();
  tarch::la::Vector<DIMENSIONS,double> neighborSubcellSize = neighboringSubgrid.getSubcellSize();

  for(int d = 0; d < DIMENSIONS_MINUS_ONE; d++) {
    int subgridDimension = peanoclaw::grid::Linearization::getGlobalDimension(d, projectionDimension);
    neighborPosition[subgridDimension]
      += neighborSubcellSize[subgridDimension] * neighborSubcellIndex[d];
  }

  tarch::la::Vector<DIMENSIONS_MINUS_ONE, int> offset;
  tarch::la::Vector<DIMENSIONS_MINUS_ONE, int> size;
  for(int d = 0; d < DIMENSIONS - 1; d++) {
    int subgridDimension = peanoclaw::grid::Linearization::getGlobalDimension(d, projectionDimension);
    double lowerBound = std::max(position[subgridDimension], neighborPosition[subgridDimension]);
    double upperBound = std::min(
                          position[subgridDimension] + subgridSize[subgridDimension],
                          neighborPosition[subgridDimension] + neighborSubcellSize[subgridDimension]
                        );
    int lowerBoundCells = (lowerBound - position[subgridDimension] + epsilon) / subcellSize[subgridDimension];
    int upperBoundCells = std::ceil((upperBound - position[subgridDimension] - epsilon) / subcellSize[subgridDimension]);

    offset[d] = lowerBoundCells;
    size[d] = upperBoundCells - lowerBoundCells;
  }

  return HyperplaneRegion(offset, size, projectionDimension, direction);
}

std::ostream& operator<<(std::ostream& out, const peanoclaw::geometry::HyperplaneRegion& region){
  out << "offset=" << region._offset << ",size=" << region._size << "";
  return out;
}
