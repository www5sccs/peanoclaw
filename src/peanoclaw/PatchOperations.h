/*
 * PatchOperations.h
 *
 *  Created on: Jun 6, 2012
 *      Author: unterweg
 */

#ifndef PEANO_APPLICATIONS_PEANOCLAW_PATCHOPERATIONS_H_
#define PEANO_APPLICATIONS_PEANOCLAW_PATCHOPERATIONS_H_

#include "tarch/la/Vector.h"
#include "peano/utils/Globals.h"

namespace peanoclaw {
  /**
   * Forward declaration
   */
  class Patch;
  class Area;
}

namespace peanoclaw {
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
   * Copies data from the source patch to the destination patch. The difference
   * to the method copyGhostLayerDataBlock is that in this method a discrete
   * part (i.e. a rectangular block of cells) is specified in the destination
   * patch while the appropriate part from the source block is determined by
   * the positions, sizes and subdivision factors of the patches. Therefore,
   * this method also works on patches which have different subcell sizes, due
   * to level or subdivision factor.
   * This method performs a mapping from continuous points in one patch to the
   * discrete cells in the other patch. We use a d-linear interpolation. Since
   * for fine cells we might need information outside the source patch, we use
   * d-linear extrapolation for this.
   *
   * The two parameters interpolateToUOld and interpolateToCurrentTime are
   * helpers while the UNew array does not hold a ghostlayer. There are three
   * situations when an interpolation is performed:
   *
   *  - Filling of ghostlayer: Interpolating to UOld array but using the time
   *    of UNew
   *  - Interpolating to UNew of new patch: Interpolating to UNew array and
   *    using the time of UNew
   *  - Interpolating to UOld of new patch: Interpolating to UOld array and
   *    using the time of UOld
   *
   */
  void interpolate (
      const tarch::la::Vector<DIMENSIONS, int>&    destinationSize,
      const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
      const peanoclaw::Patch& source,
      peanoclaw::Patch&        destination,
      bool interpolateToUOld = true,
      bool interpolateToCurrentTime = true
  );

  void interpolateVersion2 (
      const tarch::la::Vector<DIMENSIONS, int>&    destinationSize,
      const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
      const peanoclaw::Patch& source,
      peanoclaw::Patch&        destination,
      bool interpolateToUOld = true,
      bool interpolateToCurrentTime = true
  );

  /**
   * Determines the areas of the source patch that need to be
   * restricted to the destination.
   */
  void restrict (
    const peanoclaw::Patch& source,
    peanoclaw::Patch&       destination,
    bool restrictOnlyOverlappedAreas
  );

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
}

#endif /* PEANO_APPLICATIONS_PEANOCLAW_PATCHOPERATIONS_H_ */
