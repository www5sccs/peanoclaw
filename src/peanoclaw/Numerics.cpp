/*
 * Numerics.cpp
 *
 *  Created on: Jun 24, 2013
 *      Author: kristof
 */
#include "peanoclaw/Numerics.h"

#include "peanoclaw/interSubgridCommunication/DefaultTransfer.h"

peanoclaw::Numerics::Numerics(
  peanoclaw::interSubgridCommunication::DefaultTransfer* transfer,
  peanoclaw::interSubgridCommunication::Interpolation*   interpolation,
  peanoclaw::interSubgridCommunication::Restriction*     restriction,
  peanoclaw::interSubgridCommunication::FluxCorrection*  fluxCorrection
) : _transfer(transfer), _interpolation(interpolation), _restriction(restriction), _fluxCorrection(fluxCorrection) {
}

peanoclaw::Numerics::~Numerics() {
  delete _transfer;
  delete _interpolation;
  delete _restriction;
  delete _fluxCorrection;
}

void peanoclaw::Numerics::transferGhostlayer(
  const tarch::la::Vector<DIMENSIONS, int>&    size,
  const tarch::la::Vector<DIMENSIONS, int>&    sourceOffset,
  const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
  peanoclaw::Patch& source,
  peanoclaw::Patch&       destination
) const {
  _transfer->transferGhostlayer(
    size,
    sourceOffset,
    destinationOffset,
    source,
    destination
  );
}

void peanoclaw::Numerics::interpolateSolution (
  const tarch::la::Vector<DIMENSIONS, int>&    destinationSize,
  const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
  peanoclaw::Patch& source,
  peanoclaw::Patch& destination,
  bool interpolateToUOld,
  bool interpolateToCurrentTime,
  bool useTimeUNewOrTimeUOld
) const {
  _interpolation->interpolateSolution(
    destinationSize,
    destinationOffset,
    source,
    destination,
    interpolateToUOld,
    interpolateToCurrentTime,
    useTimeUNewOrTimeUOld
  );
}

void peanoclaw::Numerics::restrictSolution (
  peanoclaw::Patch& source,
  peanoclaw::Patch& destination,
  bool              restrictOnlyOverlappedRegions
) const {
  _restriction->restrictSolution(
    source,
    destination,
    restrictOnlyOverlappedRegions
  );
}

void peanoclaw::Numerics::postProcessRestriction(
  peanoclaw::Patch& destination,
  bool              restrictOnlyOverlappedRegions
) const {
}

void peanoclaw::Numerics::applyFluxCorrection (
  const Patch& sourceSubgrid,
  Patch& destinationSubgrid,
  int dimension,
  int direction
) const {
  _fluxCorrection->applyCorrection(
    sourceSubgrid,
    destinationSubgrid,
    dimension,
    direction
  );
}

void peanoclaw::Numerics::computeFluxes(
  Patch& subgrid
) const {
  _fluxCorrection->computeFluxes(subgrid);
}

void peanoclaw::Numerics::update (Patch& finePatch) {

}
