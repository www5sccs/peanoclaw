/*
 * MasterWorkerAndForkJoinCommunicator.h
 *
 *  Created on: Mar 19, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_PARALLEL_MASTERWORKERANDFORKJOINCOMMUNICATOR_H_
#define PEANOCLAW_PARALLEL_MASTERWORKERANDFORKJOINCOMMUNICATOR_H_

#include "peanoclaw/State.h"
#include "peano/heap/Heap.h"
#include "peano/utils/Dimensions.h"
#include "tarch/la/Vector.h"
#include "tarch/logging/Log.h"

namespace peanoclaw {

  namespace records {
    class CellDescription;
    class Data;
  }

  namespace parallel {
    class MasterWorkerAndForkJoinCommunicator;
  }
}

/**
 * This class encapsulates the sending and receiving methods for
 * communicating with the master or a worker. This class may be
 * used during normal iteration master/worker communication as
 * well as during fork or join operations.
 */
class peanoclaw::parallel::MasterWorkerAndForkJoinCommunicator {

  private:
    typedef peanoclaw::records::CellDescription CellDescription;
    typedef peanoclaw::records::Data Data;

    /**
     * Logging device.
     */
    static tarch::logging::Log _log;

    int _remoteRank;

    tarch::la::Vector<DIMENSIONS,double> _position;

    int _level;

    peano::heap::Heap<CellDescription>& _cellDescriptionHeap;

    peano::heap::Heap<Data>& _dataHeap;

    peano::heap::MessageType _messageType;

    void sendCellDescription(int cellDescriptionIndex);

    void sendDataArray(int index);

    /**
     * Deletes the cell description and the according arrays.
     */
    void deleteArraysFromPatch(int cellDescriptionIndex);

    /**
     * Receives an data array, copies it to a local heap array
     * and returns the local index.
     */
    int receiveDataArray();

  public:
    MasterWorkerAndForkJoinCommunicator(
      int remoteRank,
      const tarch::la::Vector<DIMENSIONS,double> position,
      int level,
      bool forkOrJoin
    );

    /**
     * Sends all necessary information for a patch defined by its
     * cell description index.
     */
    void sendPatch(
      int cellDescriptionIndex
    );

    /**
     * Receives all necessary information for a patch defined
     * by its cell description index.
     *
     * @param localCellDescriptionIndex The current cell description
     * index for the local patch. If cell description exist the
     * according arrays will be deleted and replaced by the remote
     * ones.
     */
    void receivePatch(
      int localCellDescriptionIndex
    );

    /**
     * Merges an incoming cell during a fork or join operation.
     *
     * There are several cases:
     *
     *  - The local cell is not inside or the remote cell is assigned
     *    to a different rank: No merge is necessary, since the data
     *    is not relevant to the local rank.
     *  - The local cell is flagged as remote: The remote subgrid is
     *    received and discarded.
     *  - Otherwise: The remote subgrid is received and is merged into
     *    the local subgrid.
     */
    void mergeCellDuringForkOrJoin(
      peanoclaw::Cell&                      localCell,
      const peanoclaw::Cell&                remoteCell,
      tarch::la::Vector<DIMENSIONS, double> cellSize,
      const peanoclaw::State&               state
    );

    /**
     * Called when a worker returns its state to its master. The
     * worker's state is then merged into the master's state by
     * this method.
     */
    void mergeWorkerStateIntoMasterState(
      const peanoclaw::State&          workerState,
      peanoclaw::State&                masterState
    );
};



#endif /* PEANOCLAW_PARALLEL_MASTERWORKERANDFORKJOINCOMMUNICATOR_H_ */
