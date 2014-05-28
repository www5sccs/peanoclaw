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

    void initializePatch(Patch& patch){};

    void fillBoundaryLayer(
      Patch& patch,
      int dimension,
      bool setUpper
    ){};

    void solveTimestep(
      Patch& patch,
      double maximumTimestepSize,
      bool useDimensionalSplitting
    ){};

    tarch::la::Vector<DIMENSIONS, double> getDemandedMeshWidth(
      Patch& patch,
      bool isInitializing
    ) {
      return 1;
    }

    int getNumberOfUnknownsPerCell() const {
      return 1;
    }

    int getNumberOfParameterFieldsWithoutGhostlayer() const {
      return 0;
    }

    int getNumberOfParameterFieldsWithGhostlayer() const  {
      return 0;
    }

    int getGhostlayerWidth() const {
      return 1;
    }
};
#endif /* PEANO_APPLICATIONS_PEANOCLAW_TESTS_NUMERICSTESTSTUMP_H_ */
