/*
 * ParallelSubgrid.h
 *
 *  Created on: Aug 5, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_PARALLELSUBGRID_H_
#define PEANOCLAW_PARALLELSUBGRID_H_

#include "peanoclaw/Cell.h"
#include "peanoclaw/Patch.h"
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

  ParallelSubgrid(
    const Cell& cell
  );

  ParallelSubgrid(
    Patch& subgrid
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
   * Returns the number of additional transfers for this subgrid that have to
   * be skipped.
   */
  int getNumberOfTransfersToBeSkipped() const;

  /**
   * Decreases the number of additional transfers for this subgrid that have to
   * be skipped.
   */
  void decreaseNumberOfTransfersToBeSkipped();

  /**
   * Resets the number to zero.
   */
  void resetNumberOfTransfersToBeSkipped();

  /**
   * Counts how many of the adjacent subgrids belong to a different MPI rank
   * and how many vertices are involved in the communication.
   */
  void countNumberOfAdjacentParallelSubgrids(
    peanoclaw::Vertex * const            vertices,
    const peano::grid::VertexEnumerator& verticesEnumerator
  );

  /**
   * Determines whether the subgrid is adjacent to the local subdomain. This
   * requires that the subgrid is assigned to a different rank.
   */
  bool isAdjacentToLocalSubdomain(
    const peanoclaw::Cell&               coarseGridCell,
    peanoclaw::Vertex * const            fineGridVertices,
    const peano::grid::VertexEnumerator& fineGridVerticesEnumerator
  );

  /**
   * Determines whether the subgrid is adjacent to a remote subdomain. I.e.
   * if one of the adjacent vertices is shared with a remote rank.
   */
//  bool isAdjacentToRemoteSubdomain(
//    peanoclaw::Vertex * const            fineGridVertices,
//    const peano::grid::VertexEnumerator& fineGridVerticesEnumerator
//  );
};

#endif /* PEANOCLAW_PARALLELSUBGRID_H_ */
