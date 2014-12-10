/*
 * FaceAdjacentPatchTraversal.h
 *
 *  Created on: Jul 9, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ASPECTS_MINIMALFACEADJACENTSUBGRIDTRAVERSAL_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ASPECTS_MINIMALFACEADJACENTSUBGRIDTRAVERSAL_H_

#include "peano/utils/Globals.h"

namespace peanoclaw {
  namespace interSubgridCommunication {
    namespace aspects {
      template<class LoopBody>
      class MinimalFaceAdjacentSubgridTraversal;
    }
  }
}

/**
 * Implements a scheme for traversing through all pairs of patches adjacent to
 * a vertex which are sharing a face (or an edge in 2D).
 *
 * The LoopBody has to provide an operator() with the following parameters:
 * - Patch& patch1
 * - int    indexPatch1
 * - Patch& patch2
 * - int    indexPatch2
 * - int    dimension
 * - int    direction
 *
 * Here the indices describe the position of the according patch in the local patcharray
 * of the vertex.
 * dimension describes the dimension in which the patches are adjacent and direction
 * indicates the direction from patch1 to patch2, i.e. whether patch2 is above patch1
 * in the given dimension.
 */
template<class LoopBody>
class peanoclaw::interSubgridCommunication::aspects::MinimalFaceAdjacentSubgridTraversal {

  public:
    MinimalFaceAdjacentSubgridTraversal(
      peanoclaw::Patch patches[TWO_POWER_D],
      int vertexIndex,
      LoopBody& loopBody
    );
};

#include "peanoclaw/interSubgridCommunication/aspects/MinimalFaceAdjacentSubgridTraversal.cpph"

#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ASPECTS_FACEADJACENTSUBGRIDTRAVERSAL_H_ */
