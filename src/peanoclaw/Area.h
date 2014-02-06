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

  namespace tests {
    class PatchTest;
  }
}

/**
 * This class describes a rectangular axis-aligned
 * area of a patch.
 */
class peanoclaw::Area {

  private:
    friend class peanoclaw::tests::PatchTest;

    static int factorial(int i);

    /**
     * Increases a given set of indices in such manner that for each i holds:
     *
     *   - indices[i] < upperBound
     *   - indices[i] < indices[i+1]
     */
    static void incrementIndices(tarch::la::Vector<DIMENSIONS, int>& indices, int highestIndex, int upperBound);

    /**
     * Returns the number of manifolds that exist for a
     * given dimensionality within  a given subgrid.
     *
     * 2D:
     * 0 -> 4
     * 1 -> 4
     *
     * 3D:
     * 0 -> 8
     * 1 -> 12
     * 2 -> 6
     *
     * s = (DIMENSIONS - dimensionality)
     * n = 2^s * (s aus DIMENSIONS)
     *
     * Nick Highham -> Tex
     *
     */
    static int getNumberOfManifolds(int dimensionality);

    /**
     * Returns the manifold of the given dimensionality and
     * the given index.
     *
     * @param index has to be between 0 (incl.) and
     * getNumberOfManifolds(dimensionality) (excl.).
     */
    static tarch::la::Vector<DIMENSIONS, int> getManifold(int dimensionality, int index);

    /**
     * 2D:
     * Edge (1D) to vertices (0D): 2
     *
     * Vertex (0D) to edges (1D):  2
     *
     * 3D:
     * Face (2D) to edges (1D):    4
     * Face (2D) to vertices (0D): 4
     * Edge (1D) to vertices (0D): 2
     *
     * Vertex (0D) to edges (1D):  3
     * Vertex (0D) to faces (2D):  3
     * Edge (1D) to faces (2D):    2
     *
     * 2D:
     * Rauf:
     * Ecke 0D auf Kante 1D: -1,-1 -> -1,0 und 0,-1
     *
     * Runter
     * Kante 1D auf Ecke 0D: -1,0 -> -1,-1 und -1,1 (Festgelegt in einer Dimension)
     *
     * 3D:
     * Rauf
     * Ecke 0D auf Kante 1D: -1,-1,-1 -> -1,-1,0   -1,0,-1   0,-1,-1
     * Ecke 0D auf Fläche 2D: -1,-1,-1 -> -1,0,0   0,-1,0    0,0,-1
     * Kante 1D auf Fläche 2D: -1,-1,0 -> -1,0,0  und  0,-1,0
     *
     * Runter
     * Fläche 2D auf Kante 1D: -1,0,0 -> -1,-1,0   -1,1,0   -1,0,-1   -1,0,1 (Festgelegt in einer Dimension)
     * Fläche 2D auf Ecke 0D:  -1,0,0 -> -1,-1,-1  -1,-1,1  -1,1,-1   -1,1,1 (Festgelegt in einer Dimension)
     * Kante 1D auf Ecke 0D:   -1,-1,0 -> -1,-1,-1  -1,-1,1 (Festgelegt in zwei Dimensionen)
     *
     *
     *
     *
     * Bei Rauf können immer alle -1 oder 1 in 0 umgewandelt werden. Und zwar maximal so viele wie Dimensionssprünge gemacht werden.
     * Bei Runter ist es interessant, welche Dimensionen festgelegt sind. Das sind immer (Dimensionalität - Dimensionalität_Ursprunngsmanigfaltigkeit) viele und diese stehen immer auf -1 oder 1, die anderen auf 0. Damit gilt folgendes: Bei Runter kann jede 0 in eine -1 oder 1 umgewandelt werden und zwar so viele wie Dimensionssprünge gemacht werden.
     *
     * Anzahl veränderbarer Stellen: n
     * Anzahl möglicher Zielwerte: k
     * Anzahl Dimensionssprünge: s
     * Anzahl Nachbarmanigfaltigkeiten: m
     *
     * n = (Anzahl 0)
     * d = DIMENSIONS
     *
     * Rauf:
     * m = s aus (d-n) = (d-n)!/(s!(d-n-s)!)
     *
     * Runter:
     * m = 2^(s aus n) = 2^(n!/(s!(n-s)!*s)
     */
    static int getNumberOfAdjacentManifolds(
      const tarch::la::Vector<DIMENSIONS, int>& manifoldPosition,
      int dimensionality,
      int adjacentDimensionality
    );

    static tarch::la::Vector<DIMENSIONS, int> getIndexOfAdjacentManifold(
      tarch::la::Vector<DIMENSIONS, int> manifoldPosition,
      int dimensionality,
      int adjacentDimensionality,
      int adjacentIndex
    );

    static bool checkHigherDimensionalManifoldForOverlap(
      const tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int>& adjacentRanks,
      tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int>&       overlapOfRemoteGhostlayers,
      const tarch::la::Vector<DIMENSIONS, int>& manifoldPosition,
      int dimensionality,
      int manifoldEntry,
      int rank
    );

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

    static int linearizeManifoldPosition(
      const tarch::la::Vector<DIMENSIONS, int>& manifoldPosition
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

    /**
     * Determines the areas required to cover all cells of a subgrid that are
     * overlapped by a remote ghostlayer belonging to the given rank.
     *
     * Currently the resulting areas are neither necessarily optimal nor
     * disjoint.
     */
   static int getAreasOverlappedByRemoteGhostlayers(
     const tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int>& adjacentRanks,
     tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int>        overlapOfRemoteGhostlayers,
     const tarch::la::Vector<DIMENSIONS, int>&              subdivisionFactor,
     int                                                    rank,
     Area                                                   areas[THREE_POWER_D_MINUS_ONE]
   );

};

std::ostream& operator<<(std::ostream& out, const peanoclaw::Area& area);

#endif /* PENAO_APPLICATIONS_PEANOCLAW_AREA_H_ */
