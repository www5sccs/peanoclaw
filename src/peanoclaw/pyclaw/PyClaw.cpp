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
#include "tarch/parallel/Node.h"

tarch::logging::Log peanoclaw::pyclaw::PyClaw::_log("peanoclaw::pyclaw::PyClaw");

peanoclaw::pyclaw::PyClaw::PyClaw(
  InitializationCallback         initializationCallback,
  BoundaryConditionCallback      boundaryConditionCallback,
  SolverCallback                 solverCallback,
  AddPatchToSolutionCallback     addPatchToSolutionCallback,
  peanoclaw::interSubgridCommunication::Interpolation*  interpolation,
  peanoclaw::interSubgridCommunication::Restriction*    restriction,
  peanoclaw::interSubgridCommunication::FluxCorrection* fluxCorrection
) : Numerics(interpolation, restriction, fluxCorrection),
_initializationCallback(initializationCallback),
_boundaryConditionCallback(boundaryConditionCallback),
_solverCallback(solverCallback),
_addPatchToSolutionCallback(addPatchToSolutionCallback),
_totalSolverCallbackTime(0.0)
{
  //import_array();
}

peanoclaw::pyclaw::PyClaw::~PyClaw()
{
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

  logTraceOutWith1Argument( "initializePatch(...)", demandedMeshWidth);
  return demandedMeshWidth;
}

double peanoclaw::pyclaw::PyClaw::solveTimestep(Patch& patch, double maximumTimestepSize, bool useDimensionalSplitting) {
  logTraceInWith2Arguments( "solveTimestep(...)", maximumTimestepSize, useDimensionalSplitting);

  assertion2(tarch::la::greater(maximumTimestepSize, 0.0), "Timestepsize == 0 should be checked outside.", patch.getMinimalNeighborTimeConstraint());

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

  pyclawWatch.stopTimer();
  _totalSolverCallbackTime += pyclawWatch.getCalendarTime();

  assertion4(
      tarch::la::greater(patch.getTimestepSize(), 0.0)
      || tarch::la::greater(dtAndEstimatedNextDt[1], 0.0)
      || tarch::la::equals(maximumTimestepSize, 0.0)
      || tarch::la::equals(patch.getEstimatedNextTimestepSize(), 0.0),
      patch, maximumTimestepSize, dtAndEstimatedNextDt[1], patch.toStringUNew());
  assertion(patch.getTimestepSize() < std::numeric_limits<double>::infinity());

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

void peanoclaw::pyclaw::PyClaw::fillBoundaryLayer(Patch& patch, int dimension, bool setUpper) const {
  logTraceInWith3Arguments("fillBoundaryLayerInPyClaw", patch, dimension, setUpper);

  logDebug("fillBoundaryLayerInPyClaw", "Setting left boundary for " << patch.getPosition() << ", dim=" << dimension << ", setUpper=" << setUpper);

  PyClawState state(patch);
  _boundaryConditionCallback(state._q, state._qbc, dimension, setUpper ? 1 : 0);

  logTraceOut("fillBoundaryLayerInPyClaw");
}

