/*
 * DefaultFluxCorrection.h
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_VELOCITYFLUXCORRECTION_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_VELOCITYFLUXCORRECTION_H_

#include "peanoclaw/interSubgridCommunication/FluxCorrection.h"
#include "peanoclaw/tests/GridLevelTransferTest.h"
#include "peanoclaw/Patch.h"
#include "peanoclaw/grid/SubgridAccessor.h"
#include "peanoclaw/grid/TimeIntervals.h"

#include "peano/utils/Dimensions.h"
#include "tarch/la/Vector.h"
#include "tarch/logging/Log.h"

namespace peanoclaw {
  namespace interSubgridCommunication {
    class VelocityFluxCorrection;
  }

  class Patch;
}

/**
 * An implementation for the flux correction that assumes that velocities and not
 * impulses are stored in the variables.
 */
class peanoclaw::interSubgridCommunication::VelocityFluxCorrection
    : public peanoclaw::interSubgridCommunication::FluxCorrection {

  private:
    /**
     * Logging device for the trace macros.
     */
    static tarch::logging::Log  _log;

    friend class peanoclaw::tests::GridLevelTransferTest;

    /**
     * Returns the region of the region where the two given
     * patches overlap.
     * This overload projects the patches along the given projection axis
     * and just calculates the overlap in this projection.
     *
     * In 2d this refers to a projection to one-dimensional intervals and
     * the intersection between these intervals.
     */
    double calculateOverlappingRegion(
      tarch::la::Vector<DIMENSIONS, double> position1,
      tarch::la::Vector<DIMENSIONS, double> size1,
      tarch::la::Vector<DIMENSIONS, double> position2,
      tarch::la::Vector<DIMENSIONS, double> size2,
      int projectionAxis
    ) const;

    void correctFluxBetweenCells(
      int dimension,
      int direction,
      double timestepOverlap,
      const peanoclaw::Patch& sourceSubgrid,
      peanoclaw::Patch& destinationSubgrid,
      peanoclaw::grid::SubgridAccessor& sourceAccessor,
      peanoclaw::grid::SubgridAccessor& destinationAccessor,
      const peanoclaw::grid::TimeIntervals& sourceTimeIntervals,
      const peanoclaw::grid::TimeIntervals& destinationTimeIntervals,
      double destinationSubcellVolume,
      const tarch::la::Vector<DIMENSIONS,double>& sourceSubcellSize,
      const tarch::la::Vector<DIMENSIONS,double>& destinationSubcellSize,
      const tarch::la::Vector<DIMENSIONS,int>& subcellIndexInSourcePatch,
      const tarch::la::Vector<DIMENSIONS,int>& ghostlayerSubcellIndexInSourcePatch,
      const tarch::la::Vector<DIMENSIONS,int>& adjacentSubcellIndexInDestinationPatch,
      const tarch::la::Vector<DIMENSIONS,int>& ghostlayerSubcellIndexInDestinationPatch
    ) const;

  public:

    virtual ~VelocityFluxCorrection();

    /**
     * Applying the default flux correction on the coarse patch.
     */
    void applyCorrection(
      Patch& sourcePatch,
      Patch& destinationPatch,
      int dimension,
      int direction
    ) const;

    void computeFluxes(Patch& subgrid) const;
};

#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTFLUXCORRECTION_H_ */
