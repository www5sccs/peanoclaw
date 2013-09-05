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

/**
 * Traverses all pairs of patches that are adjacent via a corner. This is done from
 * a vertex-centered view, i.e. all patches adjacent to a vertex are taken into
 * account.
 * This class is only valid in 3D. In 2D patches adjacent via a corner can be
 * traversed via the EdgeAdjacentPatchTraversal class (since the terminology is
 * based on 3D).
 */
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
