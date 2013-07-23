/*
 * EdgeAdjacentPatchTraversal.h
 *
 *  Created on: Jul 9, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ASPECTS_EDGEADJACENTPATCHTRAVERSALWITHCOMMONFACENEIGHBORS_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ASPECTS_EDGEADJACENTPATCHTRAVERSALWITHCOMMONFACENEIGHBORS_H_

#include "peano/utils/Globals.h"

namespace peanoclaw {
  namespace interSubgridCommunication {
    namespace aspects {
      template<class LoopBody>
      class EdgeAdjacentPatchTraversalWithCommonFaceNeighbors;
    }
  }
}

/**
 * Implements a scheme for traversing through all pairs of patches adjacent to
 * a vertex which are sharing an edge (or vertex in 2D). In contrast to
 * EdgeAdjacentPatchTraversal this class also provides the two patches which are
 * face adjacent to both of the two edge adjacent patches.
 *
 * The LoopBody has to provide an operator() with the following parameters:
 * - Patch& patch1
 * - int    indexPatch1
 * - Patch& patch2
 * - int    indexPatch2
 * - Patch& faceNeighbor1
 * - int    faceNeighborIndex1
 * - Patch& faceNeighbor2
 * - int    faceNeighborIndex2
 * - Vector direction
 *
 * Here the indices describe the position of the according patch in the local patcharray
 * of the vertex.
 * direction describes the discrete direction from patch1 to patch2 in the sense of the
 * intermediate vertex. I.e. direction=[1,1] means that patch1 is lower left of the
 * vertex while patch2 is upper right.
 */
template<class LoopBody>
class peanoclaw::interSubgridCommunication::aspects::EdgeAdjacentPatchTraversalWithCommonFaceNeighbors {
  public:
    EdgeAdjacentPatchTraversalWithCommonFaceNeighbors(
      peanoclaw::Patch patches[TWO_POWER_D],
      LoopBody& loopBody
    );
};

#include "peanoclaw/interSubgridCommunication/aspects/EdgeAdjacentPatchTraversalWithCommonFaceNeighbors.cpph"


#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ASPECTS_EDGEADJACENTPATCHTRAVERSALWITHCOMMONFACENEIGHBORS_H_ */
