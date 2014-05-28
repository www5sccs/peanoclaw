/*
 * NumericsFactory.h
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_PYCLAW_NUMERICSFACTORY_H_
#define PEANOCLAW_PYCLAW_NUMERICSFACTORY_H_

#include "peanoclaw/pyclaw/PyClawCallbacks.h"

#if defined(SWE)
    #include "peanoclaw/native/SWEKernel.h" 
#endif

#if defined(PEANOCLAW_FULLSWOF2D)
    #include "peanoclaw/native/FullSWOF2D.h" 
#endif

#include "tarch/logging/Log.h"

namespace peanoclaw {
  class Numerics;
  class NumericsFactory;
}

/**
 * Factory class for creating a Numerics object for using PyClaw.
 */
class peanoclaw::NumericsFactory {

  private:
    /**
     * Logging device.
     */
    static tarch::logging::Log _log;

  public:

#if defined(SWE)
    peanoclaw::Numerics* createSWENumerics(
      peanoclaw::native::SWEKernelScenario& scenario
    );
#endif

#if defined(PEANOCLAW_PYCLAW)
    /**
     * Creates a new Numerics object on the heap that encapsulates
     * all offered PyClaw functionality.
     */
    peanoclaw::Numerics* createPyClawNumerics(
      InitializationCallback initializationCallback,
      BoundaryConditionCallback boundaryConditionCallback,
      SolverCallback solverCallback,
      RefinementCriterionCallback refinementCriterionCallback,
      AddPatchToSolutionCallback addPatchToSolutionCallback,
      InterPatchCommunicationCallback interpolationCallback,
      InterPatchCommunicationCallback restrictionCallback,
      InterPatchCommunicationCallback fluxCorrectionCallback,
      int numberOfUnknowns,
      int numberOfAuxFields,
      int ghostlayerWidth
    );
#endif

#if defined(PEANOCLAW_FULLSWOF2D)
    peanoclaw::Numerics* createFullSWOF2DNumerics(
      peanoclaw::native::SWEKernelScenario& scenario
    );
#endif

    /**
     * Creates a new Numerics object on the heap that provides
     * native functionality.
     */
//    peanoclaw::Numerics* createNativeNumerics(
//    );
};

#endif /* NUMERICSFACTORY_H_ */
