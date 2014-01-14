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

#include "peano/grid/VertexEnumerator.h"

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

    bool   _minimalPatchBlockedDueToCoarsening;
    bool   _minimalPatchBlockedDueToGlobalTimestep;

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

    /**
     * Gives an estimate on how many iterations have to be done for this subgrid
     * to reach the global timestep.
     */
    int estimateRemainingIterationsUntilGlobalSubgrid(Patch subgrid) const;

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
     * Constructs a SubgridStatistics by receiving it from the worker.
     */
    SubgridStatistics(int workerRank);

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
     * Updates the reason for the minimal subgrid to be blocked for
     * timestepping.
     */
    void updateMinimalSubgridBlockReason(
      const peanoclaw::Patch&              subgrid,
      peanoclaw::Vertex * const            coarseGridVertices,
      const peano::grid::VertexEnumerator& coarseGridVerticesEnumerator,
      double                               globalTimestep
    );

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
     * Returns the estimated number of iterations to reach the global timestep
     * with all subgrids.
     */
    int getEstimatedIterationsUntilGlobalTimestep() const;

    /**
     * States that the restriktion of a worker was skipped. I.e. there is no
     * information whether this worker has evolved completely to the global
     * timestep and the statistics assumes that this is not the case.
     * Hence, as long as restrictions are skipped, the current global timestep
     * cannot be finished.
     */
    void restrictionFromWorkerSkipped();
    #endif
};
#endif /* PEANOCLAW_STATISTICS_SUBGRIDSTATISTICS_H_ */
