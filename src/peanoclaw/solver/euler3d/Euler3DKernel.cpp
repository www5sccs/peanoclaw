/*
 * Euler3DKernel.cpp
 *
 *  Created on: Jul 24, 2014
 *      Author: kristof
 */
#include "peanoclaw/solver/euler3d/Euler3DKernel.h"

#include "peanoclaw/grid/SubgridAccessor.h"
#include "peanoclaw/Patch.h"

tarch::logging::Log peanoclaw::solver::euler3d::Euler3DKernel::_log("peanoclaw::solver::euler3d::Euler3DKernel");

peanoclaw::solver::euler3d::Euler3DKernel::Euler3DKernel(
  peanoclaw::native::scenarios::SWEScenario& scenario,
  peanoclaw::interSubgridCommunication::DefaultTransfer* transfer,
  peanoclaw::interSubgridCommunication::Interpolation*   interpolation,
  peanoclaw::interSubgridCommunication::Restriction*     restriction,
  peanoclaw::interSubgridCommunication::FluxCorrection*  fluxCorrection
) : Numerics(transfer, interpolation, restriction, fluxCorrection),
_scenario(scenario)
{
}


void peanoclaw::solver::euler3d::Euler3DKernel::initializePatch(
  Patch& patch
) {
  logTraceIn( "initializePatch(...)");

  _scenario.initializePatch(patch);

  logTraceOutWith1Argument( "initializePatch(...)", demandedMeshWidth);
}

void peanoclaw::solver::euler3d::Euler3DKernel::update(Patch& subgrid) {
  _scenario.update(subgrid);
}

void peanoclaw::solver::euler3d::Euler3DKernel::solveTimestep(Patch& patch, double maximumTimestepSize, bool useDimensionalSplitting) {
  peanoclaw::grid::SubgridAccessor accessor = patch.getAccessor();

  patch.getTimeIntervals().advanceInTime();
  patch.getTimeIntervals().setTimestepSize(0.1);
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::solver::euler3d::Euler3DKernel::getDemandedMeshWidth(Patch& patch, bool isInitializing) {
  return _scenario.computeDemandedMeshWidth(patch, isInitializing);
}

void peanoclaw::solver::euler3d::Euler3DKernel::addPatchToSolution(Patch& patch) {
}

void peanoclaw::solver::euler3d::Euler3DKernel::fillBoundaryLayer(Patch& patch, int dimension, bool setUpper) {
}

