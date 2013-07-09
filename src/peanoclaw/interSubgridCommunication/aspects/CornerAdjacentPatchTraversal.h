/*
 * CornerAdjacentPatchTraversal.h
 *
 *  Created on: Jul 9, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ASPECTS_CORNERADJACENTPATCHTRAVERSAL_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ASPECTS_CORNERADJACENTPATCHTRAVERSAL_H_

namespace peanoclaw {
  namespace interSubgridCommunication {
    namespace aspects {
      template<class LoopBody>
      class CornerAdjacentPatchTraversal;
    }
  }
}

template<class LoopBody>
class peanoclaw::interSubgridCommunication::aspects::CornerAdjacentPatchTraversal {
  public:
    CornerAdjacentPatchTraversal(
      peanoclaw::Patch* patches,
      LoopBody& loopBody
    );
};

#include "peanoclaw/interSubgridCommunication/aspects/CornerAdjacentPatchTraversal.cpph"

#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ASPECTS_CORNERADJACENTPATCHTRAVERSAL_H_ */
