/*
 * InterpolationCallback.h
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_PYCLAW_INTERPOLATIONCALLBACK_H_
#define PEANOCLAW_PYCLAW_INTERPOLATIONCALLBACK_H_

#include "peanoclaw/pyclaw/PyClawCallbacks.h"
#include "peanoclaw/interSubgridCommunication/Interpolation.h"

namespace peanoclaw {
  namespace pyclaw {
    class InterpolationCallbackWrapper;
  }
}

class peanoclaw::pyclaw::InterpolationCallbackWrapper : public peanoclaw::interSubgridCommunication::Interpolation {

  private:
    InterPatchCommunicationCallback _interpolationCallback;

  public:
    InterpolationCallbackWrapper(InterPatchCommunicationCallback interpolationCallback);

    virtual ~InterpolationCallbackWrapper();

    /**
     * @see peanoclaw::interSubgridCommunication::Interpolation
     */
    void interpolate (
      const tarch::la::Vector<DIMENSIONS, int>&    destinationSize,
      const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
      peanoclaw::Patch& source,
      peanoclaw::Patch& destination,
      bool interpolateToUOld,
      bool interpolateToCurrentTime,
      bool useTimeUNewOrTimeUOld
    );
};


#endif /* INTERPOLATIONCALLBACK_H_ */
