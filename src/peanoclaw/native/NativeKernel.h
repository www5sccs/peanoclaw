/*
 * NativeKernel.h
 *
 *  Created on: Jun 24, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_NATIVE_NATIVEKERNEL_H_
#define PEANOCLAW_NATIVE_NATIVEKERNEL_H_

#include "peanoclaw/Numerics.h"

namespace peanoclaw {

  class Patch;

  namespace native {
    class NativeKernel;
  }
}

class peanoclaw::native::NativeKernel : public peanoclaw::Numerics {

  public:

    NativeKernel();

    /**
     * @see peanoclaw::Numerics
     */
    void addPatchToSolution(Patch& patch);

    /**
     * @see peanoclaw::Numerics
     */
    double initializePatch(Patch& patch);

    /**
     * @see peanoclaw::Numerics
     */
    void fillBoundaryLayer(
      Patch& patch,
      int dimension,
      bool setUpper
    ) const;

    /**
     * @see peanoclaw::Numerics
     */
    double solveTimestep(
      Patch& patch,
      double maximumTimestepSize,
      bool useDimensionalSplitting
    );
};

#endif /* PEANOCLAW_NATIVE_NATIVEKERNEL_H_ */
