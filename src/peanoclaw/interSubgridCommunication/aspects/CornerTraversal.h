/*
 * GhostlayerTraversal.h
 *
 *  Created on: Jul 8, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_SUBGRIDCOMMUNICATION_ASPECTS_GHOSTLAYERTRAVERSAL_H_
#define PEANOCLAW_SUBGRIDCOMMUNICATION_ASPECTS_GHOSTLAYERTRAVERSAL_H_

#include "peanoclaw/Patch.h"

namespace peanoclaw {
  namespace interSubgridCommunication {
    namespace aspects {
      template<class LoopBody>
      class CornerTraversal;
    }
  }
}

template<class LoopBody>
class peanoclaw::interSubgridCommunication::aspects::CornerTraversal {

  public:
    /**
     * Traverses all $2^d$ corners of the given patch.
     */
    CornerTraversal(
      Patch& patch,
      LoopBody& loopBody
    );
};

#include "peanoclaw/interSubgridCommunication/aspects/CornerTraversal.cpph"

#endif /* PEANOCLAW_SUBGRIDCOMMUNICATION_ASPECTS_GHOSTLAYERTRAVERSAL_H_ */
