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

    /**
     * Inserts the given remote rank into the list of adjacent
     * ranks. Furthermore, it adds the vertex to the number
     * of shared vertices with this remote rank as long as this
     * has not been done already.
     */
    void listRemoteRankAndAddSharedVertex(
      int  remoteRank,
      bool setRanks[THREE_POWER_D_MINUS_ONE]
    );

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
   * Marks the current state of the subgrid as sent if it was flagged so in the
   * previous grid iteration. Hence, the flag markCurrentStateAsSentInNextIteration
   * is set to false.
   */
  void markCurrentStateAsSentIfAppropriate();

  /**
   * Returns whether this patch was sent to the neighbor ranks since the last time step.
   */
  bool wasCurrentStateSent() const;

  /**
   * States that the current state of the subgrid should be marked as sent from the
   * next iteration on.
   */
  void markCurrentStateAsSentInNextIteration();

  /**
   * Returns the adjacent rank or -1 if no or more than one ranks are adjacent
   * to this subgrid.
   */
  tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int> getAdjacentRanks() const;

  /**
   * Decreases the number of shared adjacent vertices by one.
   */
  void decreaseNumberOfSharedAdjacentVertices(int remoteRank);

  /**
   * Returns the number of adjacent vertices that are shared between this and
   * the adjacent rank.
   */
  tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int> getNumberOfSharedAdjacentVertices() const;
  int getNumberOfSharedAdjacentVertices(int remoteRank) const;

  /**
   * Returns the number of additional transfers for this subgrid that have to
   * be skipped.
   */
  tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int> getAllNumbersOfTransfersToBeSkipped() const;
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
   *
   * Also, it checks for all adjacent subgrids that reside on the same level
   * whether they belong to a remote rank. If they do, the rank is set to
   * the current subgrid.
   */
  void countNumberOfAdjacentParallelSubgrids(
    peanoclaw::Vertex * const            vertices,
    const peano::grid::VertexEnumerator& verticesEnumerator
  );

  /**
   *
   */
  void setOverlapOfRemoteGhostlayer(int subgridIndex, int overlap);

  /**
   * Returns the overlap of remote ghostlayers for the given
   * adjacent subgrid.
   */
  int getOverlapOfRemoteGhostlayer(int subgridIndex) const;

  /**
   * Returns all overlaps of remote ghostlayers, corresponding to the
   * adjacent ranks.
   */
  tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int> getOverlapOfRemoteGhostlayers() const;

  /**
   * Determines whether the subgrid is adjacent to the local subdomain. This
   * requires that the subgrid is assigned to a different rank.
   */
  bool isAdjacentToLocalSubdomain(
    const peanoclaw::Cell&               coarseGridCell,
    peanoclaw::Vertex * const            fineGridVertices,
    const peano::grid::VertexEnumerator& fineGridVerticesEnumerator
  );
};

#endif /* PEANOCLAW_PARALLELSUBGRID_H_ */
