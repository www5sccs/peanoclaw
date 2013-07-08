/*
 * DefaultRestriction.h
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTRESTRICTION_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTRESTRICTION_H_

#include "peanoclaw/interSubgridCommunication/Restriction.h"
#include "peanoclaw/tests/GhostLayerCompositorTest.h"
#include "peanoclaw/tests/GridLevelTransferTest.h"

#include "tarch/la/Vector.h"

namespace peanoclaw {
  namespace interSubgridCommunication {
    class DefaultRestriction;
  }
}

class peanoclaw::interSubgridCommunication::DefaultRestriction
: public peanoclaw::interSubgridCommunication::Restriction {

  private:
    /**
     * Logging device for the trace macros.
     */
    static tarch::logging::Log  _log;

    friend class peanoclaw::tests::GhostLayerCompositorTest;
    friend class peanoclaw::tests::GridLevelTransferTest;

    /**
     * Returns the area of the region where the two given patches overlap.
     */
    inline double calculateOverlappingArea(
        const tarch::la::Vector<DIMENSIONS, double>& position1,
        const tarch::la::Vector<DIMENSIONS, double>& size1,
        const tarch::la::Vector<DIMENSIONS, double>& position2,
        const tarch::la::Vector<DIMENSIONS, double>& size2
    ) {
      double area = 1.0;

      for(int d = 0; d < DIMENSIONS; d++) {
        double overlappingInterval =
            std::min(position1(d)+size1(d), position2(d)+size2(d))
        - std::max(position1(d), position2(d));
        area *= overlappingInterval;

        area = std::max(area, 0.0);
      }

      return area;
    }

    /**
     * Creates the data for the 2*d areas that need to be restricted.
     * Returns the number of areas that should be processed to restrict
     * all necessary cells.
     */
    int getAreasForRestriction (
        const tarch::la::Vector<DIMENSIONS, double>& lowerNeighboringGhostlayerBounds,
        const tarch::la::Vector<DIMENSIONS, double>& upperNeighboringGhostlayerBounds,
        const tarch::la::Vector<DIMENSIONS, double>& sourcePosition,
        const tarch::la::Vector<DIMENSIONS, double>& sourceSize,
        const tarch::la::Vector<DIMENSIONS, double>& sourceSubcellSize,
        const tarch::la::Vector<DIMENSIONS, int>&    sourceSubdivisionFactor,
        Area areas[DIMENSIONS_TIMES_TWO]
    );

    /**
     * Restricts the data from the given source to the destination averaging
     * over the source cells that overlap with a destination cell.
     *
     * @param source The patch to get the data from
     * @param destination The patch to restrict the data to
     * @param destinationTime The time that should be used as the destination
     * time when interpolating in time
     * @param restrictToUOld Decides wether to restrict to UOld or to UNew
     */
    void restrictArea (
        const peanoclaw::Patch& source,
        peanoclaw::Patch&       destination,
        const Area&                                  area
    );

  public:
    void restrict (
      const peanoclaw::Patch& source,
      peanoclaw::Patch&       destination,
      bool restrictOnlyOverlappedAreas
    );
};

#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_DEFAULTRESTRICTION_H_ */
