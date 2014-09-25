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
 * Linearization for Euler3D. Fastest-running index is Q, while the grid is traversed column-major,
 * so Y is the second fastest index, X the slowest.
 *
 * Euler3D is only 3D. This linearization, however, should also work in 2D.
 *
 * Refers to Fortran-style arrays where each cell holds an array of unknowns.
 */
peanoclaw::grid::Linearization::Linearization(
    const tarch::la::Vector<DIMENSIONS, int> subdivisionFactor,
    int numberOfUnknowns,
    int numberOfParameterFieldsWithoutGhostlayer,
    int numberOfParameterFieldsWithGhostlayer,
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

  //Parameter without ghostlayer
  _qStrideParameterWithoutGhostlayer = 1;
  stride = numberOfParameterFieldsWithoutGhostlayer;
  for (int d = DIMENSIONS-1; d >= 0; d--) {
    _cellStrideParameterWithoutGhostlayer[d] = stride;
    stride *= subdivisionFactor[d];
  }

  //Parameter with ghostlayer
  _qStrideParameterWithGhostlayer = 1;
  stride = numberOfParameterFieldsWithGhostlayer;
  for (int d = DIMENSIONS-1; d >= 0; d--) {
    _cellStrideParameterWithGhostlayer[d] = stride;
    stride *= subdivisionFactor[d] + 2 * ghostlayerWidth;
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
