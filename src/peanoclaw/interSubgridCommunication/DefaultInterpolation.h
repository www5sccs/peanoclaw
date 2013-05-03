/*
 * DefaultInterpolation.h
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTINTERPOLATION_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTINTERPOLATION_H_

#include "peanoclaw/interSubgridCommunication/Interpolation.h"

#include "peano/utils/Dimensions.h"
#include "tarch/la/Vector.h"

namespace peanoclaw {
  namespace interSubgridCommunication {
    class DefaultInterpolation;
  }
}

/**
 * Default implementation for interpolation grid values from coarse
 * to fine subgrids.
 */
class peanoclaw::interSubgridCommunication::DefaultInterpolation
  : public peanoclaw::interSubgridCommunication::Interpolation {

  private:
    /**
     * Logging device for the trace macros.
     */
    static tarch::logging::Log  _log;

  public:
    /**
     * @see peanoclaw::interSubgridCommunication::Interpolation
     */
    void interpolate (
        const tarch::la::Vector<DIMENSIONS, int>&    destinationSize,
        const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
        const peanoclaw::Patch& source,
        peanoclaw::Patch&        destination,
        bool interpolateToUOld = true,
        bool interpolateToCurrentTime = true
    );
};

#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTINTERPOLATION_H_ */
