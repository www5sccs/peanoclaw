/*
 * Area.h
 *
 *  Created on: Jan 28, 2013
 *      Author: kristof
 */

#ifndef PENAO_APPLICATIONS_PEANOCLAW_AREA_H_
#define PENAO_APPLICATIONS_PEANOCLAW_AREA_H_

#include "tarch/la/Vector.h"
#include "peano/utils/Globals.h"

namespace peanoclaw {
  /**
   * Forward declaration
   */
  class Area;

  class Patch;
}

/**
 * This class describes a rectangular axis-aligned
 * area of a patch.
 */
class peanoclaw::Area {
  public:
    tarch::la::Vector<DIMENSIONS, int> _offset;
    tarch::la::Vector<DIMENSIONS, int> _size;

    /**
     * Default constructor.
     */
    Area();

    /**
     * Creates an area with the given offset and size.
     */
    Area(
      tarch::la::Vector<DIMENSIONS, int> offset,
      tarch::la::Vector<DIMENSIONS, int> size
    );

    /**
     * Maps the area of the source patch to the according area
     * of the destination patch.
     */
    Area mapToPatch(const Patch& source, const Patch& destination, double epsilon = 1e-12) const;

    /**
     * Maps a cell of the coarse patch to an area in the destination patch.
     */
    Area mapCellToPatch(
      const tarch::la::Vector<DIMENSIONS, double>& finePosition,
      const tarch::la::Vector<DIMENSIONS, double>& fineSubcellSize,
      const tarch::la::Vector<DIMENSIONS, double>& coarseSubcellSize,
      const tarch::la::Vector<DIMENSIONS, int>& coarseSubcellIndex,
      const tarch::la::Vector<DIMENSIONS, double>& coarseSubcellPosition,
      const double& epsilon = 1e-12
    ) const;

    /**
     * Creates the data for the 2*d areas that are overlapped by ghostlayers of
     * neighboring subgrids.
     * Returns the number of areas that are required to represent the overlapped
     * parts of the subgrid.
     */
    static int getAreasOverlappedByNeighboringGhostlayers(
      const tarch::la::Vector<DIMENSIONS, double>& lowerNeighboringGhostlayerBounds,
      const tarch::la::Vector<DIMENSIONS, double>& upperNeighboringGhostlayerBounds,
      const tarch::la::Vector<DIMENSIONS, double>& sourcePosition,
      const tarch::la::Vector<DIMENSIONS, double>& sourceSize,
      const tarch::la::Vector<DIMENSIONS, double>& sourceSubcellSize,
      const tarch::la::Vector<DIMENSIONS, int>&    sourceSubdivisionFactor,
      Area areas[DIMENSIONS_TIMES_TWO]
    );
};

std::ostream& operator<<(std::ostream& out, const peanoclaw::Area& area);

#endif /* PENAO_APPLICATIONS_PEANOCLAW_AREA_H_ */
