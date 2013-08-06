/*
 * ParallelSubgrid.h
 *
 *  Created on: Aug 5, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_PARALLELSUBGRID_H_
#define PEANOCLAW_PARALLELSUBGRID_H_

#include "peanoclaw/Vertex.h"
#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/records/Data.h"

#include "tarch/logging/Log.h"

namespace peanoclaw {
  class ParallelSubgrid;
}

class peanoclaw::ParallelSubgrid {

  private:
    /**
     * Logging device.
     */
    static tarch::logging::Log _log;
    typedef peanoclaw::records::Data Data;
    typedef peanoclaw::records::CellDescription CellDescription;

    CellDescription* _cellDescription;

  public:
  ParallelSubgrid(
    CellDescription& cellDescription
  );

  ParallelSubgrid(
    int subgridDescriptionIndex
  );

  /**
   * Sets whether this patch was sent to the neighbor ranks since the last time step.
   */
  void markCurrentStateAsSent(bool wasSent);

  /**
   * Returns whether this patch was sent to the neighbor ranks since the last time step.
   */
  bool wasCurrentStateSent() const;

  /**
   * Decreases the number of shared adjacent vertices by one.
   */
  void decreaseNumberOfSharedAdjacentVertices();

  /**
   * Returns the adjacent rank or -1 if no or more than one ranks are adjacent
   * to this subgrid.
   */
  int getAdjacentRank() const;

  /**
   * Returns the number of adjacent vertices that are shared between this and
   * the adjacent rank.
   * Returns 0 if no ranks are adjacent.
   * Returns -1 if more than one ranks are adjacent.
   */
  int getNumberOfSharedAdjacentVertices() const;

  /**
   * Counts how many of the adjacent subgrids belong to a different MPI rank
   * and how many vertices are involved in the communication.
   */
  void countNumberOfAdjacentParallelSubgridsAndResetExclusiveFlag(
    peanoclaw::Vertex * const            vertices,
    const peano::grid::VertexEnumerator& verticesEnumerator
  );
};

#endif /* PEANOCLAW_PARALLELSUBGRID_H_ */
