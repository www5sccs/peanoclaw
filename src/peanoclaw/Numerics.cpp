/*
 * Numerics.cpp
 *
 *  Created on: Jun 24, 2013
 *      Author: kristof
 */
#include "peanoclaw/Numerics.h"

peanoclaw::Numerics::Numerics(
  peanoclaw::interSubgridCommunication::Interpolation*  interpolation,
  peanoclaw::interSubgridCommunication::Restriction*    restriction,
  peanoclaw::interSubgridCommunication::FluxCorrection* fluxCorrection
) : _interpolation(interpolation), _restriction(restriction), _fluxCorrection(fluxCorrection) {
}

peanoclaw::Numerics::~Numerics() {
  delete _interpolation;
  delete _restriction;
  delete _fluxCorrection;
}

void peanoclaw::Numerics::interpolate(
  const tarch::la::Vector<DIMENSIONS, int>&    destinationSize,
  const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
  const peanoclaw::Patch& source,
  peanoclaw::Patch&        destination,
  bool interpolateToUOld,
  bool interpolateToCurrentTime
) const {
  _interpolation->interpolate(
    destinationSize,
    destinationOffset,
    source,
    destination,
    interpolateToUOld,
    interpolateToCurrentTime
  );
}

void peanoclaw::Numerics::restrict (
  const peanoclaw::Patch& source,
  peanoclaw::Patch&       destination,
  bool restrictOnlyOverlappedAreas
) const {
  _restriction->restrict(
    source,
    destination,
    restrictOnlyOverlappedAreas
  );
}

void peanoclaw::Numerics::applyFluxCorrection (
  const Patch& finePatch,
  Patch& coarsePatch,
  int dimension,
  int direction
) const {
  _fluxCorrection->applyCorrection(
    finePatch,
    coarsePatch,
    dimension,
    direction
  );
}



