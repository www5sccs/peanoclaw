/*
 * Helper.h
 *
 *  Created on: Jun 10, 2012
 *      Author: kristof
 */

#ifndef PEANO_APPLICATIONS_PEANOCLAW_TESTS_HELPER_H_
#define PEANO_APPLICATIONS_PEANOCLAW_TESTS_HELPER_H_

#include "tarch/la/Vector.h"
#include "peano/utils/Dimensions.h"

namespace peano {
  namespace applications {
    namespace peanoclaw {
      /**
       * Forward declaration.
       */
      class Patch;
    }
  }
}

namespace peano {
  namespace applications {
    namespace peanoclaw {
      namespace tests {
        Patch createPatch(
          int unknownsPerSubcell,
          int auxPerSubcell,
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
  }
}
#endif /* PEANO_APPLICATIONS_PEANOCLAW_TESTS_HELPER_H_ */
