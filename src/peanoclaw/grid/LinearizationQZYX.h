/*
 * LinearizationQZYX.h
 *
 *  Created on: May 8, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_GRID_LinearizationQZYX_H_
#define PEANOCLAW_GRID_LinearizationQZYX_H_

#include "tarch/la/Vector.h"
#include "peano/utils/Dimensions.h"

/**
 * Linearization for SWE. Fastest-running index is Q, while the grid is traversed column-major,
 * so Y is the second fastest index, X the slowest.
 *
 * SWE is only 2D. This linearization, however, should also work in 3D.
 *
 * Refers to Fortran-style arrays where each cell holds an array of unknowns.
 */
peanoclaw::grid::Linearization::Linearization(
    const tarch::la::Vector<DIMENSIONS, int> subdivisionFactor,
    int numberOfUnknowns,
    int ghostlayerWidth
) : _ghostlayerWidth(ghostlayerWidth) {
  //UOld
  _qStrideUOld = 1;
  int stride = numberOfUnknowns;
  for (int d = DIMENSIONS-1; d >= 0; d--) {
    _cellStrideUOld[d] = stride;
    stride *= subdivisionFactor[d] + 2 * ghostlayerWidth;
  }

  //UNew
  _qStrideUNew = 1;
  stride = numberOfUnknowns;
  for (int d = DIMENSIONS-1; d >= 0; d--) {
    _cellStrideUNew[d] = stride;
    stride *= subdivisionFactor[d];
  }

  assertion3(tarch::la::allGreater(_cellStrideUNew, tarch::la::Vector<DIMENSIONS,int>(0)), _cellStrideUNew, subdivisionFactor, ghostlayerWidth);
  assertion3(tarch::la::allGreater(_cellStrideUOld, tarch::la::Vector<DIMENSIONS,int>(0)), _cellStrideUOld, subdivisionFactor, ghostlayerWidth);
  assertion2(_qStrideUNew > 0, subdivisionFactor, ghostlayerWidth);
  assertion2(_qStrideUOld > 0, subdivisionFactor, ghostlayerWidth);
}

tarch::la::Vector<DIMENSIONS,int> peanoclaw::grid::Linearization::getInitialOffsetForIterator() const {
  tarch::la::Vector<DIMENSIONS,int> offset(0);
  offset(DIMENSIONS-1)--;
  return offset;
}

#endif /* PEANOCLAW_GRID_LinearizationQZYX_H_ */
