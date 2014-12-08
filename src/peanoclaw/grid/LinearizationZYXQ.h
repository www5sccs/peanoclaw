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

  //Fluxes
  _faceOffset[0] = 0;
  for(int d = 0; d < DIMENSIONS; d++) {
    //Cell stride
    _cellStrideFlux[DIMENSIONS_MINUS_ONE * d] = 1;
    for(int i = 1; i < DIMENSIONS-1; i++) {
      _cellStrideFlux[DIMENSIONS_MINUS_ONE * d + i]
        = _cellStrideFlux[DIMENSIONS_MINUS_ONE * d + i - 1] * subdivisionFactor[getGlobalDimension(i-1, d)];
    }

    //q stride
    _qStrideFlux[d] = _cellStrideFlux[DIMENSIONS_MINUS_ONE * d + DIMENSIONS - 2] * subdivisionFactor[getGlobalDimension(DIMENSIONS - 2, d)];

    //Face offsets
    if(d > 0) {
      _faceOffset[2 * d] = _faceOffset[2 * d - 1] + _qStrideFlux[d - 1] * numberOfUnknowns;
    }
    _faceOffset[2 * d + 1] = _faceOffset[2 * d] + _qStrideFlux[d] * numberOfUnknowns;
  }

  //TODO unterweg debug
//  std::cout << " subdivisionFactor=" << subdivisionFactor << " unknowns=" << numberOfUnknowns << " faceOffset=" << _faceOffset << " cellStrideFlux=" << _cellStrideFlux
//      << " qStrideFlux=" << _qStrideFlux << std::endl;

  //Array indices
  int volumeNew = tarch::la::volume(subdivisionFactor);
  int volumeOld = tarch::la::volume(subdivisionFactor + 2*ghostlayerWidth);

  _uOldWithGhostlayerArrayIndex = volumeNew * numberOfUnknowns;
  _parameterWithoutGhostlayerArrayIndex = _uOldWithGhostlayerArrayIndex + volumeOld * numberOfUnknowns;
  _parameterWithGhostlayerArrayIndex = _parameterWithoutGhostlayerArrayIndex + volumeNew * numberOfParameterFieldsWithoutGhostlayer;
  _fluxArrayIndex = _parameterWithGhostlayerArrayIndex + volumeOld * numberOfParameterFieldsWithGhostlayer;

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
