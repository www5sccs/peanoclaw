/*
 * RestrictionCallbackWrapper.h
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_PYCLAW_RESTRICTIONCALLBACKWRAPPER_H_
#define PEANOCLAW_PYCLAW_RESTRICTIONCALLBACKWRAPPER_H_

#include "peanoclaw/pyclaw/PyClawCallbacks.h"
#include "peanoclaw/interSubgridCommunication/Restriction.h"

namespace peanoclaw {
  namespace pyclaw {
    class RestrictionCallbackWrapper;
  }
}

class peanoclaw::pyclaw::RestrictionCallbackWrapper : public peanoclaw::interSubgridCommunication::Restriction {

  private:
    InterPatchCommunicationCallback _restrictionCallback;

  public:
    RestrictionCallbackWrapper(InterPatchCommunicationCallback restrictionCallback);

    /**
     * @see peanoclaw::interSubgridCommunication::RestrictionCallbackWrapper
     */
    void restrict (
      const peanoclaw::Patch& source,
      peanoclaw::Patch&       destination,
      bool restrictOnlyOverlappedAreas
    );
};

#endif /* PEANOCLAW_PYCLAW_RESTRICTIONCALLBACKWRAPPER_H_ */
