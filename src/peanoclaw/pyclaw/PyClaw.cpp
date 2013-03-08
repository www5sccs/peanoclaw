/*
 * PyClaw.cpp
 *
 *  Created on: Feb 18, 2012
 *      Author: kristof
 */

#include <Python.h>
#include <numpy/arrayobject.h>
#include "peanoclaw/pyclaw/PyClaw.h"
#include "peanoclaw/pyclaw/PyClawState.h"
#include "peanoclaw/Patch.h"
#include "tarch/timing/Watch.h"

tarch::logging::Log peanoclaw::pyclaw::PyClaw::_log("peanoclaw::pyclaw::PyClaw");

peanoclaw::pyclaw::PyClaw::PyClaw(
  InitializationCallback         initializationCallback,
  BoundaryConditionCallback      boundaryConditionCallback,
  SolverCallback                 solverCallback,
  AddPatchToSolutionCallback     addPatchToSolutionCallback,
  peanoclaw::interSubgridCommunication::Interpolation*  interpolation,
  peanoclaw::interSubgridCommunication::Restriction*    restriction,
  peanoclaw::interSubgridCommunication::FluxCorrection* fluxCorrection
) : _initializationCallback(initializationCallback),
_boundaryConditionCallback(boundaryConditionCallback),
_solverCallback(solverCallback),
_addPatchToSolutionCallback(addPatchToSolutionCallback),
_interpolation(interpolation),
_restriction(restriction),
_fluxCorrection(fluxCorrection),
_totalSolverCallbackTime(0.0)
{
  import_array();
}

peanoclaw::pyclaw::PyClaw::~PyClaw()
{
  delete _interpolation;
  delete _restriction;
  delete _fluxCorrection;
}


double peanoclaw::pyclaw::PyClaw::initializePatch(
  Patch& patch
) {
  logTraceIn( "initializePatch(...)");

  PyClawState state(patch);
  double demandedMeshWidth = _initializationCallback(
    state._q,
    state._qbc,
    state._aux,
    patch.getSubdivisionFactor()(0),
    patch.getSubdivisionFactor()(1),
    #ifdef Dim3
    patch.getSubdivisionFactor()(2),
    #else
      0,
    #endif
    patch.getUnknownsPerSubcell(),
    patch.getAuxiliarFieldsPerSubcell(),
    patch.getSize()(0),
    patch.getSize()(1),
    #ifdef Dim3
    patch.getSize()(2),
    #else
      0,
    #endif
    patch.getPosition()(0),
    patch.getPosition()(1),
    #ifdef Dim3
    patch.getPosition()(2)
    #else
      0
    #endif

  );

#if 0
  patch.copyUNewToUOld(); // roland MARK: Y U CAUSE FAIL and Y U SO IMPORTANT!!
 
  demandedMeshWidth = _initializationCallback(
    state._q,
    state._qbc,
    state._aux,
    patch.getSubdivisionFactor()(0),
    patch.getSubdivisionFactor()(1),
    #ifdef Dim3
    patch.getSubdivisionFactor()(2),
    #else
      0,
    #endif
    patch.getUnknownsPerSubcell(),
    patch.getAuxiliarFieldsPerSubcell(),
    patch.getSize()(0),
    patch.getSize()(1),
    #ifdef Dim3
    patch.getSize()(2),
    #else
      0,
    #endif
    patch.getPosition()(0),
    patch.getPosition()(1),
    #ifdef Dim3
    patch.getPosition()(2)
    #else
      0
    #endif

  );
#endif

  logTraceOutWith1Argument( "initializePatch(...)", demandedMeshWidth);
  return demandedMeshWidth;
}

double peanoclaw::pyclaw::PyClaw::solveTimestep(Patch& patch, double maximumTimestepSize, bool useDimensionalSplitting) {
  logTraceInWith2Arguments( "solveTimestep(...)", maximumTimestepSize, useDimensionalSplitting);

  assertion2(tarch::la::greater(maximumTimestepSize, 0.0), "Timestepsize == 0 should be checked outside.", patch.getMinimalNeighborTimeConstraint());

  //logInfo("solveTimestep(..)", "patch before call: " << patch);

  PyClawState state(patch);

  tarch::timing::Watch pyclawWatch("", "", false);
  pyclawWatch.startTimer();
  double dtAndEstimatedNextDt[2];
  double requiredMeshWidth
    = _solverCallback(
      dtAndEstimatedNextDt,
      state._q,
      state._qbc,
      state._aux,
      patch.getSubdivisionFactor()(0),
      patch.getSubdivisionFactor()(1),
      #ifdef Dim3
      patch.getSubdivisionFactor()(2),
      #else
      0,
      #endif
      patch.getUnknownsPerSubcell(),            // meqn = nvar
      patch.getAuxiliarFieldsPerSubcell(),      // naux
      patch.getSize()(0),
      patch.getSize()(1),
      #ifdef Dim3
      patch.getSize()(2),
      #else
      0,
      #endif
      patch.getPosition()(0),
      patch.getPosition()(1),
      #ifdef Dim3
      patch.getPosition()(2),
      #else
      0,
      #endif
      patch.getCurrentTime() + patch.getTimestepSize(),
      maximumTimestepSize,
      patch.getEstimatedNextTimestepSize(),
      useDimensionalSplitting
    );

  // requiredMeshWidth =
  //    [Steer refinement]
  //        if not self.refinement_criterion == None:
  //          return self.refinement_criterion(subgridsolver.solution.state)
  //        else:
 //           return self.initial_minimal_mesh_width
  


  pyclawWatch.stopTimer();
  _totalSolverCallbackTime += pyclawWatch.getCalendarTime();
//  logInfo("solveTimestep", "Time for PyClaw solver: " << pyclawWatch.getCalendarTime());
 
  //logInfo("solveTimestep(..)", "patch after call: " << patch << " maximumTimestepSize " << maximumTimestepSize << " dtAndEstimatedNextDt[0] " << dtAndEstimatedNextDt[0] << " dtAndEstimatedNextDt[1] " << dtAndEstimatedNextDt[1]);

  assertion3(
      tarch::la::greater(patch.getTimestepSize(), 0.0)
      || tarch::la::greater(dtAndEstimatedNextDt[1], 0.0)
      || tarch::la::equals(maximumTimestepSize, 0.0)
      || tarch::la::equals(patch.getEstimatedNextTimestepSize(), 0.0),
      patch, maximumTimestepSize, dtAndEstimatedNextDt[1]);

  if (tarch::la::greater(dtAndEstimatedNextDt[0], 0.0)) {
    patch.advanceInTime();
    patch.setTimestepSize(dtAndEstimatedNextDt[0]);
  }
  patch.setEstimatedNextTimestepSize(dtAndEstimatedNextDt[1]);
 
  logTraceOutWith1Argument( "solveTimestep(...)", requiredMeshWidth);
  return requiredMeshWidth;
}

void peanoclaw::pyclaw::PyClaw::addPatchToSolution(Patch& patch) {

  PyClawState state(patch);

  assertion(_addPatchToSolutionCallback != 0);
  _addPatchToSolutionCallback(
    state._q,
    state._qbc,
    patch.getGhostLayerWidth(),
    patch.getSize()(0),
    patch.getSize()(1),
    #ifdef Dim3
    patch.getSize()(2),
    #else
    0,
    #endif
    patch.getPosition()(0),
    patch.getPosition()(1),
    #ifdef Dim3
    patch.getPosition()(2),
    #else
    0,
    #endif
    patch.getCurrentTime()+patch.getTimestepSize()
  );
}

void peanoclaw::pyclaw::PyClaw::interpolate(
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

void peanoclaw::pyclaw::PyClaw::restrict (
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

void peanoclaw::pyclaw::PyClaw::applyFluxCorrection (
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

void peanoclaw::pyclaw::PyClaw::fillBoundaryLayer(Patch& patch, int dimension, bool setUpper) const {
  logTraceInWith3Arguments("fillBoundaryLayerInPyClaw", patch, dimension, setUpper);

  logDebug("fillBoundaryLayerInPyClaw", "Setting left boundary for " << patch.getPosition() << ", dim=" << dimension << ", setUpper=" << setUpper);

  PyClawState state(patch);

  _boundaryConditionCallback(state._q, state._qbc, dimension, setUpper ? 1 : 0);

  logTraceOut("fillBoundaryLayerInPyClaw");
}


//#ifdef Dim2
//void peanoclaw::pyclaw::PyClaw::fillLeftBoundaryLayer(Patch& patch) {
//  fillBoundaryLayerInPyClaw(patch, 0, false);
//}
//
//void peanoclaw::pyclaw::PyClaw::fillUpperBoundaryLayer(Patch& patch) {
//  fillBoundaryLayerInPyClaw(patch, 1, true);
//}
//
//void peanoclaw::pyclaw::PyClaw::fillRightBoundaryLayer(Patch& patch) {
//  fillBoundaryLayerInPyClaw(patch, 0, true);
//}
//
//void peanoclaw::pyclaw::PyClaw::fillLowerBoundaryLayer(Patch& patch) {
//  fillBoundaryLayerInPyClaw(patch, 1, false);
//}
//#endif


//#ifdef Dim3
//void peanoclaw::pyclaw::PyClaw::fillLeftBoundaryLayer(Patch& patch) {
//  fillBoundaryLayerInPyClaw(patch, 0, false);
//}
//
//void peanoclaw::pyclaw::PyClaw::fillBehindBoundaryLayer(Patch& patch) {
//  fillBoundaryLayerInPyClaw(patch, 1, true);
//}
//
//void peanoclaw::pyclaw::PyClaw::fillRightBoundaryLayer(Patch& patch) {
//  fillBoundaryLayerInPyClaw(patch, 0, true);
//}
//
//void peanoclaw::pyclaw::PyClaw::fillFrontBoundaryLayer(Patch& patch) {
//  fillBoundaryLayerInPyClaw(patch, 1, false);
//}
//
//void peanoclaw::pyclaw::PyClaw::fillUpperBoundaryLayer(Patch& patch) {
//  fillBoundaryLayerInPyClaw(patch, 2, true);
//}
//
//void peanoclaw::pyclaw::PyClaw::fillLowerBoundaryLayer(Patch& patch) {
//  fillBoundaryLayerInPyClaw(patch, 2, false);
//}
//#endif
