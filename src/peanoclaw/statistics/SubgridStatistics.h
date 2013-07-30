/*
 * SubgridStatistics.h
 *
 *  Created on: Jul 29, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_STATISTICS_SUBGRIDSTATISTICS_H_
#define PEANOCLAW_STATISTICS_SUBGRIDSTATISTICS_H_

#include "peanoclaw/Patch.h"
#include "peanoclaw/State.h"
#include "peanoclaw/records/CellDescription.h"

#include "tarch/logging/Log.h"

namespace peanoclaw {
  namespace statistics {
    class SubgridStatistics;
  }
}

class peanoclaw::statistics::SubgridStatistics {
  private:
    /**
     * Logging device.
     */
    static tarch::logging::Log _log;
    typedef peanoclaw::records::CellDescription CellDescription;

    int    _minimalPatchIndex;
    int    _minimalPatchParentIndex;
    double _minimalPatchTime;
    double _startMaximumLocalTimeInterval;
    double _endMaximumLocalTimeInterval;
    double _startMinimumLocalTimeInterval;
    double _endMinimumLocalTimeInterval;
    double _minimalTimestep;
    bool   _allPatchesEvolvedToGlobalTimestep;
    double _averageGlobalTimeInterval;
    double _globalTimestepEndTime;

    bool   _isFinalized;

    /**
     * Logs the statistics to the info-logger.
     */
    void logStatistics() const;

  public:
    /**
     * Default destructor. Objects build with this constructor
     * should not be used.
     */
    SubgridStatistics();

    /**
     * Constructor to instantiate a new statistics object.
     */
    SubgridStatistics(const peanoclaw::State& state);

    /**
     * Called after processing a subgrid. Registers the
     * patch in this statistics object.
     */
    void processSubgrid(const Patch& patch, int parentIndex);

    /**
     * Called after a subgrid took a timestep. Registers the
     * patch in this statistics object.
     */
    void processSubgridAfterUpdate(const Patch& patch, int parentIndex);

    /**
     * Sets the statistics values in the state after a grid iteration.
     */
    void finalizeIteration(peanoclaw::State& state);
};
#endif /* PEANOCLAW_STATISTICS_SUBGRIDSTATISTICS_H_ */
