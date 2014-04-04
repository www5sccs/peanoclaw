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

#if !defined(UseBlockedMeshCommunication) && !defined(DoNotUseBlockedMeshCommunication)
#error "If no blocked mesh communciation should be used then please define 'DoNotUseBlockedMeshCommunication'"
#endif

#if defined(UseBlockedMeshCommunication) && defined(DoNotUseBlockedMeshCommunication)
#error "Please define 'UseBlockedMeshCommunication' or 'DoNotUseBlockedMeshCommunication', but not both!"
#endif

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
    bool                                        _onlySendOverlappedCells;
    bool                                        _packCommunication;

  public:
    /**
     * Sends a single cell description.
     */
    void sendCellDescription(int cellDescriptionIndex);

    /**
     * Sends a temporary cell description to achieve a balanced number
     * of sent and received heap data messages. Sets the given position,
     * level and size for the subgrid.
     */
    void sendPaddingCellDescription();

    /**
     * Sends the data array with the given index.
     */
    void sendDataArray(int index);

    /**
     * Sends the cells of the given data array that are overlapped by
     * the corresponding remote ghostlayer.
     */
    void sendOverlappedCells(Patch& subgrid);

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
     * Receives and returns a CellDescription, either a normal
     * CellDescription, so the resulting vector contains one
     * element, or a padding CellDescription, so the result vector
     * contains zero elements.
     */
    std::vector<CellDescription> receiveCellDescription();

    /**
     * Receives the cells of the given subgrid that are overlapped by
     * the corresponding remote ghostlayer.
     */
    void receiveOverlappedCells(
      const CellDescription& remoteCellDescription,
      Patch&                 subgrid
    );

  public:
    SubgridCommunicator(
      int                                         remoteRank,
      const tarch::la::Vector<DIMENSIONS,double>& position,
      int                                         level,
      peano::heap::MessageType                    messageType,
      bool                                        onlySendOverlappedCells,
      bool                                        packCommunication
    );

    /**
     * Sends the given subgrid to the remote rank.
     */
    void sendSubgrid(Patch& subgrid);

    /**
     * Sends a padding subgrid to balance the number of
     * sent messages.
     */
    void sendPaddingSubgrid();

    /**
     * Receives a padding subgrid that has been sent to
     * balance the number of exchanged messages.
     */
    void receivePaddingSubgrid();

    /**
     * Receives a subgrid from the remote rank and merges
     * it with the given local subgrid.
     */
    void receiveAndMergeSubgrid(Patch& subgrid);

    /**
     * Deletes the cell description and the according arrays.
     */
    void deleteArraysFromSubgrid(Patch& subgrid);
};


#endif /* PEANOCLAW_PARALLEL_SUBBGRIDCOMMUNICATOR_H_ */
