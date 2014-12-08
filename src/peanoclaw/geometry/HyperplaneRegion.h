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
  namespace geometry {
    class HyperplaneRegion;
  }
}

class peanoclaw::geometry::HyperplaneRegion {

  public:
    tarch::la::Vector<DIMENSIONS_MINUS_ONE,int> _offset;
    tarch::la::Vector<DIMENSIONS_MINUS_ONE,int> _size;

    HyperplaneRegion(
      const tarch::la::Vector<DIMENSIONS_MINUS_ONE,int>& offset,
      const tarch::la::Vector<DIMENSIONS_MINUS_ONE,int>& size
    );
};


#endif /* PEANOCLAW_GEOMETRY_HYPERPLANEREGION_H_ */
