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
     *
     * TODO unterweg: useTimeUNewOrTimeUOld is only used because the
     * restriction now is done to the minimalNeighborTimeInterval. However,
     * during grid refinement the interpolation has to take the actual currentTime
     * and timestepSize of the subgrid into account and not timeUNew and timeUOld
     * from the TimeIntervals class. This could be circumvented if the restriction
     * is carried out in the ascend event and can be done to the minimalFineGrid
     * interval which becomes the new currentTime and timestepSize of the virtual
     * subgrid anyway.
     */
    void interpolate (
        const tarch::la::Vector<DIMENSIONS, int>&    destinationSize,
        const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
        const peanoclaw::Patch& source,
        peanoclaw::Patch&        destination,
        bool interpolateToUOld,
        bool interpolateToCurrentTime,
        bool useTimeUNewOrTimeUOld
    );
};

#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTINTERPOLATION_H_ */
