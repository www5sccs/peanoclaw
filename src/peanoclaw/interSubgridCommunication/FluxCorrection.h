/*
 * CoarseGridCorrection.h
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_COARSEGRIDCORRECTION_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_COARSEGRIDCORRECTION_H_

#include "peano/utils/Dimensions.h"
#include "tarch/la/Vector.h"

namespace peanoclaw {
  namespace interSubgridCommunication {
    class FluxCorrection;
  }

  class Patch;

  namespace pyclaw {
    class PyClaw;
  }
}

/**
 * Interface for flux correction implementations
 */
class peanoclaw::interSubgridCommunication::FluxCorrection {

  public:

    virtual ~FluxCorrection(){};

    /**
     * Applies the correction on the given coarse patch.
     */
    virtual void applyCorrection(
      const Patch& finePatch,
      Patch& coarsePatch,
      int dimension,
      int direction
    ) const = 0;

    virtual void computeFluxes(Patch& subgrid) const = 0;

};

#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_COARSEGRIDCORRECTION_H_ */
