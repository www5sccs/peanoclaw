/*
 * NumericsTestStump.h
 *
 *  Created on: May 24, 2012
 *      Author: unterweg
 */

#ifndef PEANO_APPLICATIONS_PEANOCLAW_TESTS_NUMERICSTESTSTUMP_H_
#define PEANO_APPLICATIONS_PEANOCLAW_TESTS_NUMERICSTESTSTUMP_H_

#include "peanoclaw/Numerics.h"

namespace peanoclaw {
  namespace tests {
    class NumericsTestStump;
  }
}

class peanoclaw::tests::NumericsTestStump : public peanoclaw::Numerics {

  public:
    NumericsTestStump();

    void addPatchToSolution(Patch& patch){};

    double initializePatch(Patch& patch){return 1.0;};

    void fillBoundaryLayer(
      Patch& patch,
      int dimension,
      bool setUpper
    ){};

    double solveTimestep(
      Patch& patch,
      double maximumTimestepSize,
      bool useDimensionalSplitting
    ){return 1.0;};
};
#endif /* PEANO_APPLICATIONS_PEANOCLAW_TESTS_NUMERICSTESTSTUMP_H_ */
