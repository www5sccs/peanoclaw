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
     * Logs the statistics.
     */
    void logStatistics() const;

    /**
     * Merges two statistics objects.
     */
    void merge(const ParallelStatistics& otherStatistics);

};

#endif /* PEANOCLAW_STATISTICS_PARALLELSTATISTICS_H_ */
