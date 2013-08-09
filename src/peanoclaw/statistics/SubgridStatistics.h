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
#include "peanoclaw/statistics/LevelStatistics.h"

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
    typedef peanoclaw::statistics::LevelStatistics LevelStatistics;

    int                           _levelStatisticsIndex;
    std::vector<LevelStatistics>* _levelStatistics;

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
     * Reserves a Heap-vector for the level statistics.
     */
    void initializeLevelStatistics();

    /**
     * Logs the statistics to the info-logger.
     */
    void logStatistics() const;

    /**
     * Adds entries to the level vector up to the given level
     */
    void addLevelToLevelStatistics(int level);

    /**
     * Adds the given subgrid to the level statistics.
     */
    void addSubgridToLevelStatistics(
      const Patch& subgrid
    );

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
     * Constructor to build a SubgridStatistics from given levelStatistics.
     * The rest of the statistics is empty. Used for merging in parallel.
     */
    SubgridStatistics(const std::vector<LevelStatistics>& levelStatistics);

    /**
     * Copy-constructor.
     */
//    SubgridStatistics(const SubgridStatistics& toCopy);

    /**
     * Destructor.
     */
    ~SubgridStatistics();

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
     * Called after a subgrid got destroyed -> Check if it's one of the stored
     * patches and update.
     */
    void destroyedSubgrid(int cellDescriptionIndex);

    /**
     * Sets the statistics values in the state after a grid iteration
     * and logs the statistics for this iteration.
     */
    void finalizeIteration(peanoclaw::State& state);

    /**
     * Logs the level-wise statistics.
     */
    void logLevelStatistics(std::string description);

    void addBlockedPatchDueToGlobalTimestep(const Patch& subgrid);
    void addBlockedPatchDueToNeighborTimeConstraint(const Patch& subgrid);
    void addBlockedPatchDueToSkipIteration(const Patch& subgrid);
    void addBlockedPatchDueToCoarsening(const Patch& subgrid);

    /**
     * Merges another statistics object into this one. It assumes the other
     * one belongs to a different subdomain.
     */
    void merge(const SubgridStatistics& subgridStatistics);

    /**
     * Computes the average for the values that should be shown as an average
     * over the whole simulation time.
     */
    void averageTotalSimulationValues(int numberOfEntries);

    #ifdef Parallel
    /**
     * Sends the level statistics to the master rank.
     */
    void sendToMaster(int masterRank);

    /**
     * Receives the level statistics from a worker rank.
     */
    void receiveFromWorker(int workerRank);
    #endif
};
#endif /* PEANOCLAW_STATISTICS_SUBGRIDSTATISTICS_H_ */
