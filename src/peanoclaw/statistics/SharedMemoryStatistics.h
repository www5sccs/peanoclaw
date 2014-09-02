/*
 * SharedMemoryStatistics.h
 *
 *  Created on: Sep 2, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_STATISTICS_SHAREDMEMORYSTATISTICS_H_
#define PEANOCLAW_STATISTICS_SHAREDMEMORYSTATISTICS_H_

#include "tarch/logging/Log.h"

#include <map>

namespace peanoclaw {
  class Patch;

  namespace statistics {
    class SharedMemoryStatistics;
  }
}

/**
 * This class is used to track the number of cell updates per thread.
 * Hence, it provides information on the load balancing for
 * shared-memory parallelization.
 */
class peanoclaw::statistics::SharedMemoryStatistics {

  private:
    /**
     * Logging device.
     */
    static tarch::logging::Log _log;

    std::map<unsigned long int, double> _cellUpdatesPerThread;

  public:
    SharedMemoryStatistics();
    ~SharedMemoryStatistics();

    void addCellUpdatesForThread(
      const peanoclaw::Patch& subgrid
    );

    void logStatistics() const;
};


#endif /* PEANOCLAW_STATISTICS_SHAREDMEMORYSTATISTICS_H_ */
