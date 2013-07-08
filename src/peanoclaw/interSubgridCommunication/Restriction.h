/*
 * Restriction.h
 *
 *  Created on: Mar 5, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_RESTRICTION_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_RESTRICTION_H_

#include "peano/utils/Globals.h"
#include "tarch/logging/Log.h"

namespace peanoclaw {

  class Area;
  class Patch;

  namespace interSubgridCommunication {
    class Restriction;
  }
}

class peanoclaw::interSubgridCommunication::Restriction {
  public:

    virtual ~Restriction(){};

    /**
     * Restricts data from a fine patch to a coarse patch.
     */
    virtual void restrict (
      const peanoclaw::Patch& source,
      peanoclaw::Patch&       destination,
      bool restrictOnlyOverlappedAreas
    ) = 0;
};

#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_RESTRICTION_H_ */
