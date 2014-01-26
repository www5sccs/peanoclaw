/*
 * ParallelStatistics.h
 *
 *  Created on: Aug 5, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_STATISTICS_PARALLELSTATISTICS_H_
#define PEANOCLAW_STATISTICS_PARALLELSTATISTICS_H_

#include "tarch/logging/Log.h"

namespace peanoclaw {
  namespace statistics {
    class ParallelStatistics;
  }
}

class peanoclaw::statistics::ParallelStatistics {

  private:
    /**
     * Logging device.
     */
    static tarch::logging::Log _log;

    std::string _name;

    int _sentNeighborData;
    int _sentPaddingNeighborData;
    int _receivedNeighborData;
    int _receivedPaddingNeighborData;

    double _waitingTimeMasterWorkerSpacetreeCommunication;
    int    _samplesMasterWorkerSpacetreeCommunication;
    double _waitingTimeMasterWorkerSubgridCommunication;
    int    _samplesMasterWorkerSubgridCommunication;
    double _waitingTimeNeighborSubgridCommunication;
    int    _samplesNeighborSubgridCommunication;

  public:
    /**
     * Constructor sets all counters to zero.
     */
    ParallelStatistics(std::string name);

    /**
     * Counts a subgrid sent to a neighbor.
     */
    void sentNeighborData(int numberOfSentSubgrids=1);

    /**
     * Counts a padding subgrid sent to a neighbor.
     */
    void sentPaddingNeighborData(int numberOfSentSubgrids=1);

    /**
     * Counts a subgrid received from a neighbor.
     */
    void receivedNeighborData(int numberOfReceivedSubgrids=1);

    /**
     * Counts a padding subgrid received from a neighbor.
     */
    void receivedPaddingNeighborData(int numberOfReceivedSubgrids=1);

    /**
     * Adds the measured waiting time to the statistics of the vertical
     * spacetree communication. This is the time for the communication
     * performed in Peano along the tree.
     */
    void addWaitingTimeForMasterWorkerSpacetreeCommunication(double time);

    /**
     * This can not be measured directly.
     */
    //void addWaitingTimeForNeighborSpacetreeCommunication(double time);

    /**
     * Adds the measured waiting time to the statistics of the vertical
     * subgrid communication. This is the time for the communication
     * of subgrid data, i.e. done via peano::heap.
     */
    void addWaitingTimeForMasterWorkerSubgridCommunication(double time);

    /**
     * Adds the measured waiting time to the statistics of the communication
     * of subgrids to the neighbors, i.e. done via peano::heap.
     */
    void addWaitingTimeForNeighborSubgridCommunication(double time);

    /**
     * Logs the statistics for the last iteration.
     */
    void logIterationStatistics() const;

    /**
     * Logs the statistics for the complete simulation.
     */
    void logTotalStatistics() const;

    /**
     * Merges two statistics objects.
     */
    void merge(const ParallelStatistics& otherStatistics);

};

#endif /* PEANOCLAW_STATISTICS_PARALLELSTATISTICS_H_ */
