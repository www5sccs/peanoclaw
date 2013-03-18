/*
 * RestrictionCallbackWrapper.cpp
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */
#include "peanoclaw/pyclaw/RestrictionCallbackWrapper.h"

#include "peanoclaw/Patch.h"

#include "peanoclaw/pyclaw/PyClawState.h"

peanoclaw::pyclaw::RestrictionCallbackWrapper::RestrictionCallbackWrapper(
  InterPatchCommunicationCallback restrictionCallback
) : _restrictionCallback(restrictionCallback) {

}

void peanoclaw::pyclaw::RestrictionCallbackWrapper::restrict(
  const peanoclaw::Patch& source,
  peanoclaw::Patch&       destination,
  bool restrictOnlyOverlappedAreas
) {
  PyClawState sourceState(source);
  PyClawState destinationState(destination);

  _restrictionCallback(
    sourceState._q,
    sourceState._qbc,
    sourceState._aux,
    source.getSubdivisionFactor()(0),
    source.getSubdivisionFactor()(1),
    #ifdef Dim3
    source.getSubdivisionFactor()(2),
    #else
    0,
    #endif
    source.getSize()(0),
    source.getSize()(1),
    #ifdef Dim3
    source.getSize()(2),
    #else
    0.0,
    #endif
    source.getPosition()(0),
    source.getPosition()(1),
    #ifdef Dim3
    source.getPosition()(2),
    #else
    0.0,
    #endif
    source.getCurrentTime(),
    source.getTimestepSize(),

    destinationState._q,
    destinationState._qbc,
    destinationState._aux,
    destination.getSubdivisionFactor()(0),
    destination.getSubdivisionFactor()(1),
    #ifdef Dim3
    destination.getSubdivisionFactor()(2),
    #else
    0,
    #endif
    destination.getSize()(0),
    destination.getSize()(1),
    #ifdef Dim3
    destination.getSize()(2),
    #else
    0.0,
    #endif
    destination.getPosition()(0),
    destination.getPosition()(1),
    #ifdef Dim3
    destination.getPosition()(2),
    #else
    0.0,
    #endif
    destination.getCurrentTime(),
    destination.getTimestepSize(),
    source.getUnknownsPerSubcell(),
    source.getAuxiliarFieldsPerSubcell()
  );
}


