/*
 * Linearization.h
 *
 *  Created on: May 8, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_GRID_LINEARIZATION_H_
#define PEANOCLAW_GRID_LINEARIZATION_H_

#include "peano/utils/Dimensions.h"

#include "tarch/la/Vector.h"

namespace peanoclaw {
  namespace grid {
    /**
     * Typedef to the used linearization. A linearization class must support the
     * following interface:
     *
     *   int linearize(int unknown,const tarch::la::Vector<DIMENSIONS, int>& subcellIndex) const;
     *
     *   int linearizeWithGhostlayer(int unknown,const tarch::la::Vector<DIMENSIONS, int>& subcellIndex) const;
     *
     */
    //typedef LinearizationZYXQ Linearization;
    class Linearization;
  }
}

class peanoclaw::grid::Linearization {

private:
  int _qStrideUNew;
  tarch::la::Vector<DIMENSIONS,int> _cellStrideUNew;
  int _qStrideUOld;
  tarch::la::Vector<DIMENSIONS,int> _cellStrideUOld;
  int _ghostlayerWidth;
  int _qStrideParameterWithoutGhostlayer;
  tarch::la::Vector<DIMENSIONS,int> _cellStrideParameterWithoutGhostlayer;
  int _qStrideParameterWithGhostlayer;
  tarch::la::Vector<DIMENSIONS,int> _cellStrideParameterWithGhostlayer;

public:

  Linearization() {
  }

  inline Linearization(
      const tarch::la::Vector<DIMENSIONS, int> subdivisionFactor,
      int numberOfUnknowns,
      int numberOfParameterFieldsWithoutGhostlayer,
      int numberOfParameterFieldsWithGhostlayer,
      int ghostlayerWidth
  );

  /**
   * Linearizes the given subcell index to a columnwise stored array index.
   *
   * This method assumes that the unknown is the first index of the d-dimensional array,
   * while the subcell index-components follow as indices.
   */
  int linearize(
      int unknown,
      const tarch::la::Vector<DIMENSIONS, int>& subcellIndex
  ) const {
    int index = 0;
    for (int d = 0; d < DIMENSIONS; d++) {
      index += subcellIndex(d) * _cellStrideUNew[d];
    }
    index += unknown * _qStrideUNew;

    return index;
  }

  /**
   * Does a similar linearization like the method linearize(...), but it assumes
   * that the array is surrounded by a ghostlayer.
   */
  int linearizeWithGhostlayer(
      int unknown,
      const tarch::la::Vector<DIMENSIONS, int>& subcellIndex
  ) const {
    int index = 0;
    for(int d = 0; d < DIMENSIONS; d++) {
      index += (subcellIndex(d) + _ghostlayerWidth) * _cellStrideUOld[d];
    }
    index += unknown * _qStrideUOld;
    return index;
  }

  int linearizeParameterWithoutGhostlayer(
    int unknown,
    const tarch::la::Vector<DIMENSIONS, int>& subcellIndex
  ) const {
    int index = 0;
    for(int d = 0; d < DIMENSIONS; d++) {
      index += subcellIndex(d) * _cellStrideParameterWithoutGhostlayer[d];
    }
    index += unknown * _qStrideParameterWithoutGhostlayer;
    return index;
  }

  int linearizeParameterWithGhostlayer(
    int unknown,
    const tarch::la::Vector<DIMENSIONS, int>& subcellIndex
  ) const {
    int index = 0;
    for(int d = 0; d < DIMENSIONS; d++) {
      index += (subcellIndex(d) + _ghostlayerWidth) * _cellStrideParameterWithGhostlayer[d];
    }
    index += unknown * _qStrideParameterWithGhostlayer;
    return index;
  }

  /**
   * Returns the index shift required to go from one unknown in
   * a given cell to the next unknown in the same cell in the
   * qNew array.
   */
  inline int getQStrideUNew() const {
    return _qStrideUNew;
  }

  /**
   * Returns the index shift required to go from one unknown in
   * a given cell to the next unknown in the same cell in the
   * qOld array.
   */
  inline int getQStrideUOld() const {
    return _qStrideUOld;
  }

  inline int getQStrideParameterWithoutGhostlayer() const {
    return _qStrideParameterWithoutGhostlayer;
  }

  inline int getQStrideParameterWithGhostlayer() const {
    return _qStrideParameterWithGhostlayer;
  }

  /**
   * Returns the index shift required to go from one cell in
   * uNew to the next cell in uNew.
   */
  inline int getCellStrideUNew(int dimension) const {
    return _cellStrideUNew[dimension];
  }

  /**
   * Returns the index shift required to go from one cell in
   * uNew to the next cell in uNew.
   */
  inline int getCellStrideUOld(int dimension) const {
    return _cellStrideUOld[dimension];
  }

  inline int getCellStrideParameterWithoutGhostlayer(int dimension) const {
    return _cellStrideParameterWithoutGhostlayer[dimension];
  }

  inline int getCellStrideParameterWithGhostlayer(int dimension) const {
    return _cellStrideParameterWithGhostlayer[dimension];
  }

  /**
   * Returns the offset that is applied when starting an iterator for a subgrid.
   * I.e., this refers to one cell before the iterated part of the subgrid to
   * allow for the pattern
   *
   *   while(iterator.moveToNextCell()) {
   *      while(iterator.moveToNextUnknowns()) {
   *          ...
   *      }
   *   }
   *
   *   For this linearization this refers to minus one cell in the highest
   *   coordinate, i.e. Y in 2D and Z in 3D.
   *
   */
  inline tarch::la::Vector<DIMENSIONS,int> getInitialOffsetForIterator() const;
};

#if defined(PEANOCLAW_PYCLAW) || defined(PEANOCLAW_FULLSWOF2D) || defined(PEANOCLAW_SWE) || defined(PEANOCLAW_EULER3D)
#include "peanoclaw/grid/LinearizationZYXQ.h"
#elif defined(PEANOCLAW_EULER3D)
#include "peanoclaw/grid/LinearizationQZYX.h"
#else
#error No Linearization defined
#endif

#endif /* PEANOCLAW_GRID_LINEARIZATION_H_ */
