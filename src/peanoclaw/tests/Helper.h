/*
 * Helper.h
 *
 *  Created on: Jun 10, 2012
 *      Author: kristof
 */

#ifndef PEANOCLAW_TESTS_HELPER_H_
#define PEANOCLAW_TESTS_HELPER_H_

#include "tarch/la/Vector.h"
#include "peano/utils/Dimensions.h"

namespace peanoclaw {
  /**
   * Forward declaration.
   */
  class Patch;
}

namespace peanoclaw {
  namespace tests {
    Patch createPatch(
      int unknownsPerSubcell,
      int parametersWithoutPerSubcell,
      int parametersWithPerSubcell,
      int subdivisionFactor,
      int ghostlayerWidth,
      tarch::la::Vector<DIMENSIONS, double> position,
      tarch::la::Vector<DIMENSIONS, double> size,
      int level,
      double time,
      double timestepSize,
      double minimalNeighborTime,
      bool virtualPatch = false
    );
  }
}
#endif /* PEANOCLAW_TESTS_HELPER_H_ */
