/*
 * NeighbourCommunicator.h
 *
 *  Created on: Mar 15, 2013
 *      Author: unterweg
 */

#ifndef PEANOCLAW_PARALLEL_NEIGHBOURCOMMUNICATOR_H_
#define PEANOCLAW_PARALLEL_NEIGHBOURCOMMUNICATOR_H_

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
    class NeighbourCommunicator;
  }
}


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

    /**
     * Logging device.
     */
    static tarch::logging::Log _log;

    int _remoteRank;
    tarch::la::Vector<DIMENSIONS,double> _position;
    int _level;
    peano::heap::Heap<CellDescription>& _cellDescriptionHeap;
    peano::heap::Heap<Data>& _dataHeap;

    void sendCellDescription(int cellDescriptionIndex);

    /**
     * Sends a temporary cell description to achieve a balanced number
     * of sent and received heap data messages.
     */
    void sendPaddingCellDescription();

    void sendDataArray(int index);

    /**
     * Sends an empty data array to achieve a balanced number
     * of sent and received heap data messages.
     */
    void sendPaddingDataArray();

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
    NeighbourCommunicator(
      int remoteRank,
      const tarch::la::Vector<DIMENSIONS,double> position,
      int level
    );

    /**
     * Sends all necessary information for a patch defined by its
     * cell description index.
     */
    void sendPatch(
      int cellDescriptionIndex
    );

    void sendPaddingPatch();

    /**
     * Receives all necessary information for a patch defined
     * by its cell description index.
     *
     * @param localCellDescriptionIndex The current cell description
     * index for the local patch.
     *
     */
    void receivePatch(
      int localCellDescriptionIndex
    );

    void receivePaddingPatch();
};

#endif /* PEANOCLAW_PARALLEL_NEIGHBOURCOMMUNICATOR_H_ */
