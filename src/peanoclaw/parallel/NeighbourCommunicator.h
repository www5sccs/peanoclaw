/*
 * NeighbourCommunicator.h
 *
 *  Created on: Mar 15, 2013
 *      Author: unterweg
 */

#ifndef PEANOCLAW_PARALLEL_NEIGHBOURCOMMUNICATOR_H_
#define PEANOCLAW_PARALLEL_NEIGHBOURCOMMUNICATOR_H_

#include "peanoclaw/parallel/SubgridCommunicator.h"
#include "peanoclaw/Patch.h"
#include "peanoclaw/State.h"
#include "peanoclaw/Vertex.h"
#include "peanoclaw/statistics/ParallelStatistics.h"

#include "peano/heap/Heap.h"
#include "peano/utils/Dimensions.h"

#include "tarch/la/Vector.h"
#include "tarch/la/VectorCompare.h"
#include "tarch/logging/Log.h"

#include <map>

namespace peanoclaw {

  namespace records {
    class CellDescription;
    class Data;
  }

  namespace parallel {
    class NeighbourCommunicator;
  }
}

#define DIMENSIONS_PLUS_ONE (DIMENSIONS+1)

/**
 * This class encapsulates the functionality for sending Patches
 * to neighbour compute nodes. Note that the methods cannot be
 * used for sending heap data to workers or the master nor during
 * forking or joining, since the receive methods expect an inverse
 * order of the data, due to the inverse grid traversal.
 */
class peanoclaw::parallel::NeighbourCommunicator {

  private:
    typedef peanoclaw::records::CellDescription CellDescription;
    typedef peanoclaw::records::Data Data;

  public:
    typedef std::map<tarch::la::Vector<DIMENSIONS_PLUS_ONE,double>, int, tarch::la::VectorCompare<DIMENSIONS_PLUS_ONE> > RemoteSubgridMap;

  private:
    /**
     * Logging device.
     */
    static tarch::logging::Log                 _log;

    int                                        _remoteRank;
    tarch::la::Vector<DIMENSIONS,double>       _position;
    int                                        _level;
    tarch::la::Vector<DIMENSIONS,double>       _subgridSize;
    RemoteSubgridMap&                          _remoteSubgridMap;
    peanoclaw::statistics::ParallelStatistics& _statistics;

    /**
     * Tries to send subgrids only once per iteration.
     */
    const bool _avoidMultipleTransferOfSubgridsIfPossible;

    /**
     * Determines whether subgrids should be sent always despite if they
     * have been updated since the last sending or not.
     */
    const bool _onlySendSubgridsAfterChange;

    /**
     * Determines whether only the cells of a subgrid should be sent that
     * are overlapped by the remote ghostlayer.
     */
    const bool _onlySendOverlappedCells;

    const bool _packCommunication;

    SubgridCommunicator                        _subgridCommunicator;

    /**
     * Receives all necessary information for a patch defined
     * by its cell description index.
     *
     * @param localCellDescriptionIndex The current cell description
     * index for the local patch.
     *
     */
    void receiveSubgrid(
      Patch& localSubgrid
    );

    /**
     * Creates a remote subgrid before merging a received subgrid
     * from a neighboring rank.
     */
    void createOrFindRemoteSubgrid(
      Vertex& localVertex,
      int     adjacentSubgridIndexInPeanoOrder,
      const tarch::la::Vector<DIMENSIONS, double>& subgridSize
    );

    /**
     * Creates the key for the remote-subgrid-map.
     */
    tarch::la::Vector<DIMENSIONS_PLUS_ONE, double> createRemoteSubgridKey(
      const tarch::la::Vector<DIMENSIONS, double> subgridPosition,
      int                                         level
    ) const;

  public:
    NeighbourCommunicator(
      int                                         remoteRank,
      const tarch::la::Vector<DIMENSIONS,double>& position,
      int                                         level,
      const tarch::la::Vector<DIMENSIONS,double>& subgridSize,
      RemoteSubgridMap&                           remoteSubgridMap,
      peanoclaw::statistics::ParallelStatistics&  statistics
    );

    /**
     * Sends all necessary information for a patch defined by its
     * cell description index.
     */
    bool sendSubgrid(
      Patch& transferedSubgrid,
      Vertex& vertex
    );

    /**
     * Sends the parts of the given subgrid that is overlapped by
     * the neighboring subgrids.
     */
    void sendOverlap(
      const Patch& transferedSubgrid
    );

    /**
     * Send all required adjacent subgrids for a vertex.
     */
    void sendSubgridsForVertex(
      peanoclaw::Vertex&                           vertex,
      const tarch::la::Vector<DIMENSIONS, double>& vertexPosition,
      const tarch::la::Vector<DIMENSIONS, double>& adjacentSubgridSize,
      int                                          level,
      const peanoclaw::State&                      state
    );

    /**
     * Receives the subgrids adjacent to a vertex that is merged
     * from a neighbor.
     */
    void receiveSubgridsForVertex(
      peanoclaw::Vertex&                           localVertex,
      const peanoclaw::Vertex&                     remoteVertex,
      const tarch::la::Vector<DIMENSIONS, double>& vertexPosition,
      const tarch::la::Vector<DIMENSIONS, double>& adjacentSubgridSize,
      int                                          level
    );

    /**
     * Switches a given subgrid to remote.
     */
    void switchToRemote(Patch& subgrid);
};

#endif /* PEANOCLAW_PARALLEL_NEIGHBOURCOMMUNICATOR_H_ */
