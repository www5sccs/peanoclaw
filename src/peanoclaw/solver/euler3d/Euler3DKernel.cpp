/*
 * Euler3DKernel.cpp
 *
 *  Created on: Jul 24, 2014
 *      Author: kristof
 */
#include "peanoclaw/solver/euler3d/Euler3DKernel.h"

#include "peanoclaw/grid/aspects/BoundaryIterator.h"
#include "peanoclaw/solver/euler3d/Cell.h"
#include "peanoclaw/solver/euler3d/SchemeExecutor.h"
#include "peanoclaw/grid/SubgridAccessor.h"
#include "peanoclaw/Patch.h"

#include "Uni/EulerEquations/RoeSolver"

#include <iomanip>

#ifdef PEANOCLAW_EULER3D
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#endif

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

void peanoclaw::solver::euler3d::Euler3DKernel::solveTimestep(
  Patch& subgrid,
  double maximumTimestepSize,
  bool useDimensionalSplitting
) {
  #ifdef Dim3
  double dt = std::min(subgrid.getTimeIntervals().getEstimatedNextTimestepSize(), maximumTimestepSize);
  assertion2(tarch::la::greaterEquals(dt, 0.0), subgrid.getTimeIntervals().getEstimatedNextTimestepSize(), maximumTimestepSize);

  //Run update
  double estimatedDt = dt;
  double cfl;
  double maximalCFL = 0.3;
  int iterations = 0;
  do {
    logDebug("solveTimestep(...)", "Solving timestep with dt=" << estimatedDt);
    double maxLambda = computeTimestep(estimatedDt, subgrid);
    dt = estimatedDt;
    cfl = dt * maxLambda / tarch::la::min(subgrid.getSubcellSize());
    estimatedDt = estimatedDt * (maximalCFL / cfl) * 0.9;
    assertion3(tarch::la::greater(cfl, 0.0), cfl, dt, maxLambda);
    assertion(tarch::la::greaterEquals(dt, 0.0));
    iterations++;
    assertion(iterations < 10);
  } while(tarch::la::greater(cfl, maximalCFL));

  assertion2(tarch::la::greater(estimatedDt, 0.0), estimatedDt, cfl);

  subgrid.getTimeIntervals().advanceInTime();
  subgrid.getTimeIntervals().setEstimatedNextTimestepSize(estimatedDt);
  subgrid.getTimeIntervals().setTimestepSize(dt);

  #ifdef Asserts
  peanoclaw::grid::SubgridAccessor accessor = subgrid.getAccessor();
  dfor(subcellIndex, subgrid.getSubdivisionFactor()) {
    assertion5(
      tarch::la::greater(accessor.getValueUNew(subcellIndex, 0), 0.0),
      subgrid.toStringUNew(),
      subgrid.toStringUOldWithGhostLayer(),
      accessor.getValueUNew(subcellIndex, 0),
      subcellIndex,
      cfl
    );
    assertion5(
      tarch::la::greater(accessor.getValueUNew(subcellIndex, 4), 0.0),
      subgrid.toStringUNew(),
      subgrid.toStringUOldWithGhostLayer(),
      accessor.getValueUNew(subcellIndex, 4),
      subcellIndex,
      cfl
    );
  }
  #endif

  //TODO unterweg debug
//  std::cout << "After: " << std::endl << subgrid.toStringUNew() << std::endl;
//  std::cout << "After: " << std::endl << subgrid.toStringUOldWithGhostLayer() << std::endl;
//  assertionFail("");
  #endif
}

double peanoclaw::solver::euler3d::Euler3DKernel::computeTimestep(
  double dt,
  peanoclaw::Patch& subgrid
) {
  SchemeExecutor schemeExecutor(
    subgrid,
    dt
  );

  #if defined(SharedTBB) or true
  tarch::la::Vector<DIMENSIONS,int> subdivisionFactor = subgrid.getSubdivisionFactor();
  tbb::parallel_reduce(
    tbb::blocked_range<int>(0, tarch::la::volume(subdivisionFactor)),
    schemeExecutor
  );
  #else
  tarch::la::Vector<DIMENSIONS,int> subdivisionFactor = subgrid.getSubdivisionFactor();
  for(int x = 0; x < subdivisionFactor[0]; x++) {
    for(int y = 0; y < subdivisionFactor[1]; y++) {
      for(int z = 0; z < subdivisionFactor[2]; z++) {
        tarch::la::Vector<DIMENSIONS,int> subcellIndex;
        assignList(subcellIndex) = x, y, z;

        schemeExecutor(subcellIndex);
      }
    }
  }
  #endif

  return schemeExecutor.getMaximumLambda();
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::solver::euler3d::Euler3DKernel::getDemandedMeshWidth(Patch& patch, bool isInitializing) {
  return _scenario.computeDemandedMeshWidth(patch, isInitializing);
}

void peanoclaw::solver::euler3d::Euler3DKernel::addPatchToSolution(Patch& patch) {
}

void peanoclaw::solver::euler3d::Euler3DKernel::fillBoundaryLayer(Patch& subgrid, int dimension, bool setUpper) {

  peanoclaw::grid::SubgridAccessor accessor = subgrid.getAccessor();

  //Fill default boundary condition
  ExtrapolateBoundaryCondition extrapolateBoundaryCondition;
  peanoclaw::grid::aspects::BoundaryIterator<ExtrapolateBoundaryCondition> defaultBoundaryIterator(extrapolateBoundaryCondition);
  defaultBoundaryIterator.iterate(subgrid, accessor, dimension, setUpper);

  //Fill scenario boundary condition
  peanoclaw::grid::aspects::BoundaryIterator<peanoclaw::native::scenarios::SWEScenario> scenarioBoundaryIterator(_scenario);
  scenarioBoundaryIterator.iterate(subgrid, accessor, dimension, setUpper);
}


void peanoclaw::solver::euler3d::ExtrapolateBoundaryCondition::setBoundaryCondition(
    peanoclaw::Patch& subgrid,
    peanoclaw::grid::SubgridAccessor& accessor,
    int dimension,
    bool setUpper,
    tarch::la::Vector<DIMENSIONS,int> sourceSubcellIndex,
    tarch::la::Vector<DIMENSIONS,int> destinationSubcellIndex
) {
   //Copy
   for(int unknown = 0; unknown < subgrid.getUnknownsPerSubcell(); unknown++) {
     accessor.setValueUOld(destinationSubcellIndex, unknown, accessor.getValueUOld(sourceSubcellIndex, unknown));
   }
}



