/*
 * LinearizationZYXQ.h
 *
 *  Created on: May 8, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_GRID_LinearizationZYXQ_H_
#define PEANOCLAW_GRID_LinearizationZYXQ_H_

#include "tarch/la/Vector.h"
#include "peano/utils/Dimensions.h"

/**
 * Linearization for PyClaw. Fastest-running index is Z (or Y in 2D), while Q is the slowest-running index.
 *
 * Refers to Fortran-style arrays where each unknown is held in one array for the complete grid.
 */

peanoclaw::grid::Linearization::Linearization(
    const tarch::la::Vector<DIMENSIONS, int> subdivisionFactor,
    int numberOfUnknowns,
    int numberOfParameterFieldsWithoutGhostlayer,
    int numberOfParameterFieldsWithGhostlayer,
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

  _qStrideParameterWithoutGhostlayer = _qStrideUNew;
  _cellStrideParameterWithoutGhostlayer = _cellStrideUNew;

  _qStrideParameterWithGhostlayer = _qStrideUOld;
  _cellStrideParameterWithGhostlayer = _cellStrideUOld;

  assertion2(tarch::la::allGreater(_cellStrideUNew, tarch::la::Vector<DIMENSIONS,int>(0)), subdivisionFactor, ghostlayerWidth);
  assertion2(tarch::la::allGreater(_cellStrideUOld, tarch::la::Vector<DIMENSIONS,int>(0)), subdivisionFactor, ghostlayerWidth);
  assertion2(_qStrideUNew > 0, subdivisionFactor, ghostlayerWidth);
  assertion2(_qStrideUOld > 0, subdivisionFactor, ghostlayerWidth);
}

tarch::la::Vector<DIMENSIONS,int> peanoclaw::grid::Linearization::getInitialOffsetForIterator() const {
  tarch::la::Vector<DIMENSIONS,int> offset(0);
  offset(DIMENSIONS-1)--;
  return offset;
}

#endif /* PEANOCLAW_GRID_LinearizationZYXQ_H_ */
