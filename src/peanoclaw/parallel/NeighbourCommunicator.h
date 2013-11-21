/*
 * NeighbourCommunicator.h
 *
 *  Created on: Mar 15, 2013
 *      Author: unterweg
 */

#ifndef PEANOCLAW_PARALLEL_NEIGHBOURCOMMUNICATOR_H_
#define PEANOCLAW_PARALLEL_NEIGHBOURCOMMUNICATOR_H_

#include "peanoclaw/Patch.h"
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
//    CellDescriptionHeap&        _cellDescriptionHeap;
//    DataHeap&                   _dataHeap;
    RemoteSubgridMap&                          _remoteSubgridMap;
    peanoclaw::statistics::ParallelStatistics& _statistics;

    /**
     * Determines whether subgrids should be sent always despite if they
     * have been updated since the last sending or not.
     */
    const bool                                       _avoidMultipleTransferOfSubgridsIfPossible;
    /**
     * Tries to find the minimal number of padding subgrids to be sent to match the
     * number of received subgrids.
     */
    const bool                                       _reduceNumberOfPaddingSubgrids;
    /**
     * Tries to send subgrids only once per iteration.
     */
    const bool                                       _reduceMultipleSends;

    void sendCellDescription(int cellDescriptionIndex);

    /**
     * Sends a temporary cell description to achieve a balanced number
     * of sent and received heap data messages. Sets the given position,
     * level and size for the subgrid.
     */
    void sendPaddingCellDescription(
      const tarch::la::Vector<DIMENSIONS, double>& position = 0,
      int                                          level = 0,
      const tarch::la::Vector<DIMENSIONS, double>& subgridSize = 0
    );

    void sendDataArray(int index);

    /**
     * Sends an empty data array to achieve a balanced number
     * of sent and received heap data messages.
     */
    void sendPaddingDataArray();

    /**
     * Deletes the cell description and the according arrays.
     */
    void deleteArraysFromPatch(Patch& subgrid);

    /**
     * Receives an data array, copies it to a local heap array
     * and returns the local index.
     */
    int receiveDataArray();

    /**
     * Receives all necessary information for a patch defined
     * by its cell description index.
     *
     * @param localCellDescriptionIndex The current cell description
     * index for the local patch.
     *
     */
    void receivePatch(
      Patch& localSubgrid
    );

    void receivePaddingPatch();

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
    void sendPatch(
      const Patch& transferedSubgrid
    );

    void sendPaddingPatch(
      const tarch::la::Vector<DIMENSIONS, double>& position = 0,
      int                                          level = 0,
      const tarch::la::Vector<DIMENSIONS, double>& subgridSize = 0
    );

    /**
     * Send all required adjacent subgrids for a vertex.
     */
    void sendSubgridsForVertex(
      peanoclaw::Vertex&                           vertex,
      const tarch::la::Vector<DIMENSIONS, double>& vertexPosition,
      const tarch::la::Vector<DIMENSIONS, double>& adjacentSubgridSize,
      int                                          level
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
