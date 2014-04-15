/*
 * FluxCorrectionCallbackWrapper.cpp
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */
#include "peanoclaw/pyclaw/FluxCorrectionCallbackWrapper.h"

#include "peanoclaw/Patch.h"
#include "peanoclaw/pyclaw/PyClawState.h"

peanoclaw::pyclaw::FluxCorrectionCallbackWrapper::FluxCorrectionCallbackWrapper(
  InterPatchCommunicationCallback fluxCorrectionCallback
) : _fluxCorrectionCallback(fluxCorrectionCallback) {
}

peanoclaw::pyclaw::FluxCorrectionCallbackWrapper::~FluxCorrectionCallbackWrapper() {
}

void peanoclaw::pyclaw::FluxCorrectionCallbackWrapper::applyCorrection(
  const Patch& finePatch,
  Patch& coarsePatch,
  int dimension,
  int direction
) const {
  PyClawState fineState(finePatch);
  PyClawState coarseState(coarsePatch);

  _fluxCorrectionCallback(
    fineState._q,
    fineState._qbc,
    fineState._aux,
    finePatch.getSubdivisionFactor()(0),
    finePatch.getSubdivisionFactor()(1),
    #ifdef Dim3
    finePatch.getSubdivisionFactor()(2),
    #else
    0,
    #endif
    finePatch.getSize()(0),
    finePatch.getSize()(1),
    #ifdef Dim3
    finePatch.getSize()(2),
    #else
    0.0,
    #endif
    finePatch.getPosition()(0),
    finePatch.getPosition()(1),
    #ifdef Dim3
    finePatch.getPosition()(2),
    #else
    0.0,
    #endif
    finePatch.getTimeIntervals().getCurrentTime(),
    finePatch.getTimeIntervals().getTimestepSize(),

    coarseState._q,
    coarseState._qbc,
    coarseState._aux,
    coarsePatch.getSubdivisionFactor()(0),
    coarsePatch.getSubdivisionFactor()(1),
    #ifdef Dim3
    coarsePatch.getSubdivisionFactor()(2),
    #else
    0,
    #endif
    coarsePatch.getSize()(0),
    coarsePatch.getSize()(1),
    #ifdef Dim3
    coarsePatch.getSize()(2),
    #else
    0.0,
    #endif
    coarsePatch.getPosition()(0),
    coarsePatch.getPosition()(1),
    #ifdef Dim3
    coarsePatch.getPosition()(2),
    #else
    0.0,
    #endif
    coarsePatch.getTimeIntervals().getCurrentTime(),
    coarsePatch.getTimeIntervals().getTimestepSize(),
    finePatch.getUnknownsPerSubcell(),
    finePatch.getNumberOfParametersWithoutGhostlayerPerSubcell()
  );
}



