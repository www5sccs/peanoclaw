/*
 * EdgeTraversal.h
 *
 *  Created on: Feb 10, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ASPECTS_EDGETRAVERSAL_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ASPECTS_EDGETRAVERSAL_H_

namespace peanoclaw {
  namespace interSubgridCommunication {
    namespace aspects {
      template<class LoopBody>
      class EdgeTraversal;
    }
  }
}

template<class LoopBody>
class peanoclaw::interSubgridCommunication::aspects::EdgeTraversal {
  public:
    /**
     * Traverses all 12 edges (in 3D) or 4 corners of the given subgrid.
     *
     * The LoopBody has to provide an operator() with the following parameters:
     *
     *  - peanoclaw::Patch: The subgrid on which the traversal is performed
     *  - peanoclaw::Area: An area describing the ghostlayer edge of the subgrid.
     *  - tarch::la::Vector<DIMENSIONS, int>: Describing the direction of the edge.
     *              The dimension which axis is parallel to the edge is set to zero.
     */
    EdgeTraversal(
      peanoclaw::Patch& subgrid,
      LoopBody& loopBody
    );
};


#include "peanoclaw/interSubgridCommunication/aspects/EdgeTraversal.cpph"

#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ASPECTS_EDGETRAVERSAL_H_ */
