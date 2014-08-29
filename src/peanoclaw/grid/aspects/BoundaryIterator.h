/*
 * BoundaryIterator.h
 *
 *  Created on: Aug 26, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_GRID_ASPECTS_BOUNDARYITERATOR_H_
#define PEANOCLAW_GRID_ASPECTS_BOUNDARYITERATOR_H_

#include "peanoclaw/Patch.h"

namespace peanoclaw {
  namespace grid {
    namespace aspects {
      template<class BoundaryCondition>
      class BoundaryIterator;
    }
  }
}

template<class BoundaryCondition>
class peanoclaw::grid::aspects::BoundaryIterator {
  private:
    BoundaryCondition& _boundaryCondition;

  public:
    BoundaryIterator(BoundaryCondition& boundaryCondition);

    void iterate(
      peanoclaw::Patch& subgrid,
      peanoclaw::grid::SubgridAccessor& accessor,
      int dimension,
      bool isUpper
    );
};

#include "peanoclaw/grid/aspects/BoundaryIterator.cpph"

#endif /* PEANOCLAW_GRID_ASPECTS_BOUNDARYITERATOR_H_ */
