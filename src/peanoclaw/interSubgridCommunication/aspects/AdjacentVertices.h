/*
 * AdjacentVertices.h
 *
 *  Created on: Aug 8, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ASPECTS_ADJACENTVERTICES_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ASPECTS_ADJACENTVERTICES_H_

#include "peanoclaw/Vertex.h"
#include "peano/grid/VertexEnumerator.h"
#include "tarch/logging/Log.h"

namespace peanoclaw {
  class Patch;

  namespace interSubgridCommunication {
    namespace aspects {
      class AdjacentVertices;
    }
  }
}

/**
 * This class provides the subgrid-centered view on the adjacent TWO_POWER_D
 * vertices and their adjacency information. The methods implement the
 * functionality to keep the consistency of this adjacency information when
 * the grid changes.
 *
 * TODO unterweg dissertation:
 * There are four situations when the adjacency information change:
 *
 *  1. Cells/Vertices get created or destroyed while refining or coarsening the grid.
 *  2. Cells/Vertices are sent to a remote node while forking or joining in parallel.
 *  3. Vertices are sent to neighboring ranks in parallel.
 *
 *  The parallel exchange is mostly implemented in the Communicator-classes in
 *  peanoclaw::parallel.
 *  The Vertex-operations are implemented in AdjacentSubgrids.
 *  The Cell-operations are implemented in this class.
 */
class peanoclaw::interSubgridCommunication::aspects::AdjacentVertices {
  private:
    /**
     * Logging device.
     */
    static tarch::logging::Log _log;

    peanoclaw::Vertex* const             _vertices;
    const peano::grid::VertexEnumerator& _verticesEnumerator;

  public:
    AdjacentVertices(
      peanoclaw::Vertex* const             vertices,
      const peano::grid::VertexEnumerator& verticesEnumerator
    );

    /**
     * Refines the surrounding vertices according to
     * the given minimal mesh width.
     */
    bool refineIfNecessary(
      Patch&                                patch,
      tarch::la::Vector<DIMENSIONS, double> minimalMeshWidth
    );
};


#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ASPECTS_ADJACENTVERTICES_H_ */
