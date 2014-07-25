/*
 * Euler3DKernel.cpp
 *
 *  Created on: Jul 24, 2014
 *      Author: kristof
 */
#include "peanoclaw/solver/euler3d/Euler3DKernel.h"

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

}

