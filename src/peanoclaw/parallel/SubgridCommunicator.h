/*
 * Communicator.h
 *
 *  Created on: Dec 4, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_PARALLEL_SUBBGRIDCOMMUNICATOR_H_
#define PEANOCLAW_PARALLEL_SUBBGRIDCOMMUNICATOR_H_

#include "peanoclaw/Heap.h"
#include "peanoclaw/Patch.h"
#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/records/Data.h"

#include "tarch/logging/Log.h"

namespace peanoclaw {
  namespace parallel {
    class SubgridCommunicator;
  }
}

/**
 * The SubgridCommunicator class offers functionality
 * to  * transfer subgrids to other MPI ranks. It
 * applies to both, master-worker and neighbor
 * communication and, thus, is used by the
 * MasterWorkerAndForkJoinCommunicator and the
 * NeighbourCommunicator.
 */
class peanoclaw::parallel::SubgridCommunicator {

  private:
    typedef peanoclaw::records::CellDescription CellDescription;
    typedef peanoclaw::records::Data Data;

    /**
     * Logging device.
     */
    static tarch::logging::Log _log;

    int _remoteRank;
    const tarch::la::Vector<DIMENSIONS,double>& _position;
    int                                         _level;
    peano::heap::MessageType                    _messageType;

  public:
    SubgridCommunicator(
      int                                         remoteRank,
      const tarch::la::Vector<DIMENSIONS,double>& position,
      int                                         level,
      peano::heap::MessageType                    messageType
    );

    /**
     * Sends a single cell description.
     */
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
     * Receives an data array, copies it to a local heap array
     * and returns the local index.
     */
    int receiveDataArray();

    /**
     * Deletes the cell description and the according arrays.
     */
    void deleteArraysFromSubgrid(Patch& subgrid);
};


#endif /* PEANOCLAW_PARALLEL_SUBBGRIDCOMMUNICATOR_H_ */
