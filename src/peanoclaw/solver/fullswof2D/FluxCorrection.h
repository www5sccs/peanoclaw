/*
 * DefaultFluxCorrection.h
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_SOLVER_FULLSWOF2D_FLUXCORRECTION_H_
#define PEANOCLAW_SOLVER_FULLSWOF2D_FLUXCORRECTION_H_

#include "peanoclaw/interSubgridCommunication/FluxCorrection.h"
#include "peanoclaw/tests/GridLevelTransferTest.h"
#include "peanoclaw/Patch.h"
#include "peanoclaw/grid/SubgridAccessor.h"
#include "peanoclaw/grid/TimeIntervals.h"
#include "peanoclaw/native/FullSWOF2D.h"

#include "peano/utils/Dimensions.h"
#include "tarch/la/Vector.h"
#include "tarch/logging/Log.h"

namespace peanoclaw {
  namespace solver {
    namespace fullswof2D {
      template<int NumberOfUnknowns>
      class FluxCorrectionTemplate;
      class FluxCorrection;
    }
  }

  namespace native {
    class FullSWOF2D;
  }

  class Patch;
}

/**
 * An implementation for the flux correction that assumes that velocities and not
 * impulses are stored in the variables.
 */
template<int NumberOfUnknowns>
class peanoclaw::solver::fullswof2D::FluxCorrectionTemplate {

  private:
    /**
     * Logging device for the trace macros.
     */
    static tarch::logging::Log  _log;

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
      tarch::la::Vector<DIMENSIONS_MINUS_ONE, int> cellIndex1,
      tarch::la::Vector<DIMENSIONS, double> subcellSize1,
      tarch::la::Vector<DIMENSIONS, double> position2,
      tarch::la::Vector<DIMENSIONS_MINUS_ONE, int> cellIndex2,
      tarch::la::Vector<DIMENSIONS, double> subcellSize2,
      int projectionAxis
    ) const;

  public:
    /**
     * Applying the default flux correction on the coarse patch.
     */
    #ifdef PEANOCLAW_FULLSWOF2D
    void computeFluxes(Patch& subgrid, peanoclaw::native::FullSWOF2D& fullswof2D) const;
    #endif

    void applyCorrection(
      Patch& sourceSubgrid,
      Patch& destinationSubgrid,
      int dimension,
      int direction
    ) const;
};

class peanoclaw::solver::fullswof2D::FluxCorrection
    : public peanoclaw::interSubgridCommunication::FluxCorrection {
    private:
      /**
       * Logging device for the trace macros.
       */
      static tarch::logging::Log  _log;

      peanoclaw::native::FullSWOF2D* _fullswof2D;

    public:

      #ifdef PEANOCLAW_FULLSWOF2D
      void setFullSWOF2D(peanoclaw::native::FullSWOF2D* fullswof2D);
      #endif

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

#include "peanoclaw/solver/fullswof2D/FluxCorrection.cpph"

#endif /* PEANOCLAW_SOLVER_FULLSWOF2D_FLUXCORRECTION_H_ */
