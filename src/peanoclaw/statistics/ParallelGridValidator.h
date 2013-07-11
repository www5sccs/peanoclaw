/*
 * ParallelGridValidator.h
 *
 *  Created on: Jun 29, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_STATISTICS_PARALLELGRIDVALIDATOR_H_
#define PEANOCLAW_STATISTICS_PARALLELGRIDVALIDATOR_H_

#include "peanoclaw/Vertex.h"
#include "peanoclaw/statistics/PatchDescriptionDatabase.h"
#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/records/PatchDescription.h"
#include "peanoclaw/tests/StatisticsTest.h"

#include "tarch/la/Vector.h"
#include "tarch/logging/Log.h"

#include <vector>

namespace peanoclaw {
  namespace statistics {
    class ParallelGridValidator;
  }
}

/**
 * Provides functionality to validate a Peano grid in
 * a parallel run.
 */
class peanoclaw::statistics::ParallelGridValidator {

private:
  /**
   * Logging device.
   */
  static tarch::logging::Log _log;
  friend class peanoclaw::tests::StatisticsTest;

  peanoclaw::statistics::PatchDescriptionDatabase _descriptions;

  /**
   * Offset of the computational domain.
   */
  tarch::la::Vector<DIMENSIONS,double> _domainOffset;

  /**
   * Size of the computational domain.
   */
  tarch::la::Vector<DIMENSIONS,double> _domainSize;

  /**
   * Flag to indicate whether edge/corner-adjacent patches have
   * to be taken into account.
   */
  bool                                 _useDimensionalSplittingOptimization;

  typedef peanoclaw::records::PatchDescription PatchDescription;
  typedef peanoclaw::records::CellDescription CellDescription;

  /**
   * Retrieves the position of the coarse patch overlapping the
   * given fine patch.
   */
  tarch::la::Vector<DIMENSIONS,double> getNeighborPositionOnLevel(
    const tarch::la::Vector<DIMENSIONS,double>& finePosition,
    const tarch::la::Vector<DIMENSIONS,double>& fineSize,
    int                                         fineLevel,
    int                                         coarseLevel,
    const tarch::la::Vector<DIMENSIONS, int>&   discreteNeighborPosition
  );

  bool validateNeighborPatchOnSameRank(
    const peanoclaw::statistics::PatchDescriptionDatabase& database,
    const PatchDescription& patch,
    const tarch::la::Vector<DIMENSIONS,double>& neighborPosition,
    int level,
    const tarch::la::Vector<DIMENSIONS, int> discreteNeighborPosition
  );

  bool validateNeighborPatchOnRemoteRank(
    const peanoclaw::statistics::PatchDescriptionDatabase& database,
    const PatchDescription& patch,
    const tarch::la::Vector<DIMENSIONS,double>& neighborPosition,
    int level,
    const tarch::la::Vector<DIMENSIONS, int> discreteNeighborPosition
  );

  void validateNeighborPatch(
    const PatchDescription& patch,
    const tarch::la::Vector<DIMENSIONS, int> discreteNeighborPosition
  );

public:
  ParallelGridValidator(
    tarch::la::Vector<DIMENSIONS,double> domainOffset,
    tarch::la::Vector<DIMENSIONS,double> domainSize,
    bool                                 useDimensionalSplittingOptimization
  );

  /**
   * Validates all patches in the database whether the adjacent patches
   * have been found.
   */
  void validatePatches();

  /**
   * Finds all adjacent patches for the given vertex and adds them to
   * the database.
   */
  void findAdjacentPatches(
    const peanoclaw::Vertex&                         fineGridVertex,
    const tarch::la::Vector<DIMENSIONS,double>&      fineGridX,
    int                                              level,
    int                                              localRank
  );

  /**
   * Deletes all adjacent patches for the given vertex from the databeas
   * that are not remote
   */
  void deleteNonRemoteAdjacentPatches(
    const peanoclaw::Vertex&                         fineGridVertex,
    const tarch::la::Vector<DIMENSIONS,double>&      fineGridX,
    int                                              level,
    int                                              localRank
  );

  std::vector<PatchDescription> getAllPatches();
};


#endif /* PEANOCLAW_STATISTICS_PARALLELGRIDVALIDATOR_H_ */
