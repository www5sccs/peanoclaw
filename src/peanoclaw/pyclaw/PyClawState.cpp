/*
 * PyClawState.cpp
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */
#include "peanoclaw/pyclaw/PyClawState.h"

#include <Python.h>
#include "peanoclaw/Patch.h"
#include <numpy/arrayobject.h>

peanoclaw::pyclaw::PyClawState::PyClawState(const Patch& patch) {

  import_array();

  //Create uNew
  npy_intp sizeUNew[1 + DIMENSIONS];
  sizeUNew[0] = patch.getUnknownsPerSubcell();
  int elementsUNew = sizeUNew[0];
  for(int d = 0; d < DIMENSIONS; d++) {
    sizeUNew[1 + d] = patch.getSubdivisionFactor()(d);
    elementsUNew *= sizeUNew[1 + d];
  }
  npy_intp sizeUOldWithGhostlayer[1 + DIMENSIONS];
  sizeUOldWithGhostlayer[0] = patch.getUnknownsPerSubcell();
  int elementsUOldWithGhostlayer = sizeUOldWithGhostlayer[0];
  for(int d = 0; d < DIMENSIONS; d++) {
    sizeUOldWithGhostlayer[1 + d] = patch.getSubdivisionFactor()(d) + 2*patch.getGhostLayerWidth();
    elementsUOldWithGhostlayer *= sizeUOldWithGhostlayer[1 + d];
  }

  _q = PyArray_SimpleNewFromData(1 + DIMENSIONS, sizeUNew, NPY_DOUBLE, patch.getUNewArray());
  _qbc = PyArray_SimpleNewFromData(1 + DIMENSIONS, sizeUOldWithGhostlayer, NPY_DOUBLE, patch.getUOldWithGhostlayerArray());

  //Build auxArray
  double* auxArray = 0;
  if(patch.getAuxiliarFieldsPerSubcell() > 0) {
    auxArray = patch.getAuxArray();
    assertion(auxArray != 0);
    npy_intp sizeAux[1 + DIMENSIONS];
    sizeAux[0] = patch.getAuxiliarFieldsPerSubcell();
    for(int d = 0; d < DIMENSIONS; d++) {
      sizeAux[1 + d] = patch.getSubdivisionFactor()(d);
    }
    _aux = PyArray_SimpleNewFromData(1 + DIMENSIONS, sizeAux, NPY_DOUBLE, auxArray);
    assertion(_aux != 0);
  } else {
    //TODO Here I'm creating an unecessary PyObject. Would be great, if we could just pass
    //"None" to Python.
    auxArray = 0;
    npy_intp sizeZero[1 + DIMENSIONS];
    for(int d = 0; d < 1+ DIMENSIONS; d++) {
      sizeZero[d] = 0;
    }
    _aux = PyArray_SimpleNewFromData(1 + DIMENSIONS, sizeZero, NPY_DOUBLE, auxArray);
  }
}

peanoclaw::pyclaw::PyClawState::~PyClawState() {
}



