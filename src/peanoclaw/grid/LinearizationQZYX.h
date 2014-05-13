/*
 * LinearizationQZYX.h
 *
 *  Created on: May 8, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_GRID_LINEARIZATIONQZYX_H_
#define PEANOCLAW_GRID_LINEARIZATIONQZYX_H_

#include "tarch/la/Vector.h"
#include "peano/utils/Dimensions.h"

namespace peanoclaw {
  namespace grid {
    class LinearizationQZYX;
  }
}

/**
 * Linearization for PyClaw. Fastest-running index is X, while Q is the slowest-running index.
 *
 * Refers to Fortran-style arrays where each unknown is held in one array for the complete grid.
 */
class peanoclaw::grid::LinearizationQZYX {

  private:
    int _qStrideUNew;
    tarch::la::Vector<DIMENSIONS,int> _cellStrideUNew;
    int _qStrideUOld;
    tarch::la::Vector<DIMENSIONS,int> _cellStrideUOld;
    int _ghostlayerWidth;

  public:
    LinearizationQZYX() {
    }

    LinearizationQZYX(
      const tarch::la::Vector<DIMENSIONS, int> subdivisionFactor,
      int ghostlayerWidth
    ) : _ghostlayerWidth(ghostlayerWidth) {
      //UOld
      int stride = 1;
      for (int d = DIMENSIONS-1; d >= 0; d--) {
        _cellStrideUOld[d] = stride;
        stride *= subdivisionFactor[d] + 2 * ghostlayerWidth;
      }
      //_uOldStrideCache[0] = stride;
      _qStrideUOld = stride;

      //UNew
      stride = 1;
      for (int d = DIMENSIONS-1; d >= 0; d--) {
        _cellStrideUNew[d] = stride;
        stride *= subdivisionFactor[d];
      }
      //_uNewStrideCache[0] = stride;
      _qStrideUNew = stride;

      assertion2(tarch::la::allGreater(_cellStrideUNew, tarch::la::Vector<DIMENSIONS,int>(0)), subdivisionFactor, ghostlayerWidth);
      assertion2(tarch::la::allGreater(_cellStrideUOld, tarch::la::Vector<DIMENSIONS,int>(0)), subdivisionFactor, ghostlayerWidth);
      assertion2(_qStrideUNew > 0, subdivisionFactor, ghostlayerWidth);
      assertion2(_qStrideUOld > 0, subdivisionFactor, ghostlayerWidth);
    }

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
    //  int stride = 1;
    //  for(int d = DIMENSIONS-1; d >= 0; d--) {
      for (int d = 0; d < DIMENSIONS; d++) {
    //    assertion3(subcellIndex(d) >= 0 && subcellIndex(d) < _cellDescription->getSubdivisionFactor()(d),
    //        subcellIndex(d),
    //        _cellDescription->getSubdivisionFactor(),
    //        toString());
    //    index += subcellIndex(d) * stride;
    //    stride *= _cellDescription->getSubdivisionFactor()(d);
        index += subcellIndex(d) * _cellStrideUNew[d]; //_uNewStrideCache[d + 1];
      }
    //  index += unknown * stride;
      index += unknown * _qStrideUNew; //_uNewStrideCache[0];

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
      //int ghostlayerWidth = _cellDescription->getGhostlayerWidth();

      for(int d = 0; d < DIMENSIONS; d++) {
        index += (subcellIndex(d) + _ghostlayerWidth) * _cellStrideUOld[d];
      }
      index += unknown * _qStrideUOld;
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
};


#endif /* PEANOCLAW_GRID_LINEARIZATIONQZYX_H_ */
