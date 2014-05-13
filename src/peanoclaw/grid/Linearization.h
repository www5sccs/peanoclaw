/*
 * Linearization.h
 *
 *  Created on: May 8, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_GRID_LINEARIZATION_H_
#define PEANOCLAW_GRID_LINEARIZATION_H_

#include "peanoclaw/grid/LinearizationQZYX.h"

namespace peanoclaw {
  namespace grid {
    /**
     * Typedef to the used linearization. A linearization class must support the
     * following interface:
     *
     *   int linearize(int unknown,const tarch::la::Vector<DIMENSIONS, int>& subcellIndex) const;
     *
     *   int linearizeWithGhostlayer(int unknown,const tarch::la::Vector<DIMENSIONS, int>& subcellIndex) const;
     *
     */
    typedef LinearizationQZYX Linearization;
  }
}

#endif /* PEANOCLAW_GRID_LINEARIZATION_H_ */
