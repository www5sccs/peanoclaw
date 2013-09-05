/*
 * CornerAdjacentPatchTraversalWithCommonFaceAndEdgeNeighbors.h
 *
 *  Created on: Sep 5, 2013
 *      Author: unterweg
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_CORNERADJACENTPATCHTRAVERSALWITHCOMMONFACEANDEDGENEIGHBORS_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_CORNERADJACENTPATCHTRAVERSALWITHCOMMONFACEANDEDGENEIGHBORS_H_

namespace peanoclaw {
  namespace interSubgridCommunication {
    namespace aspects {
      template<class LoopBody>
      class CornerAdjacentPatchTraversalWithCommonFaceAndEdgeNeighbors;
    }
  }
}

template<class LoopBody>
class peanoclaw::interSubgridCommunication::aspects::CornerAdjacentPatchTraversalWithCommonFaceAndEdgeNeighbors {
  public:
    CornerAdjacentPatchTraversalWithCommonFaceAndEdgeNeighbors(
      peanoclaw::Patch patches[TWO_POWER_D],
      LoopBody& loopBody
    );
};

#include "peanoclaw/interSubgridCommunication/aspects/CornerAdjacentPatchTraversalWithCommonFaceAndEdgeNeighbors.cpph"

#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_CORNERADJACENTPATCHTRAVERSALWITHCOMMONFACEANDEDGENEIGHBORS_H_ */
