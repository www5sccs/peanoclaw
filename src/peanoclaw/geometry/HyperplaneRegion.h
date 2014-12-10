/*
 * HyperplanRegion.h
 *
 *  Created on: Dec 5, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_GEOMETRY_HYPERPLANEREGION_H_
#define PEANOCLAW_GEOMETRY_HYPERPLANEREGION_H_

#include "peano/utils/Dimensions.h"
#include "tarch/la/Vector.h"

#define DIMENSIONS_MINUS_ONE (DIMENSIONS-1)

namespace peanoclaw {
  class Patch;

  namespace geometry {
    class HyperplaneRegion;
  }
}

class peanoclaw::geometry::HyperplaneRegion {

  private:
    static double epsilon;

  public:
    tarch::la::Vector<DIMENSIONS_MINUS_ONE,int> _offset;
    tarch::la::Vector<DIMENSIONS_MINUS_ONE,int> _size;
    int _dimension;
    int _direction;

    HyperplaneRegion(
      const tarch::la::Vector<DIMENSIONS_MINUS_ONE,int>& offset,
      const tarch::la::Vector<DIMENSIONS_MINUS_ONE,int>& size,
      int dimension,
      int direction
    );

    /**
     * Returns the interface region of a subgrid on which this
     * subgrids touches the neighboring subgrid.
     */
    static HyperplaneRegion getInterfaceRegion(
      const Patch& subgrid,
      const Patch& neighboringSubgrid
    );

    /**
     * Returns the interface region between two subgrids that
     * only covers the cell in the neighbor subgrid given by
     * the neighbor subcell index.
     */
    static HyperplaneRegion getInterfaceRegionForSubcell(
      const Patch& subgrid,
      const Patch& neighboringSubgrid,
      tarch::la::Vector<DIMENSIONS_MINUS_ONE,int> neighborSubcellIndex,
      int projectionDimension,
      int direction
    );
};

std::ostream& operator<<(std::ostream& out, const peanoclaw::geometry::HyperplaneRegion& region);

#endif /* PEANOCLAW_GEOMETRY_HYPERPLANEREGION_H_ */
