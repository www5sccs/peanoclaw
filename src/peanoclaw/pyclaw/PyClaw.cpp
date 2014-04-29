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

#include "peano/utils/Loop.h"

#include "tarch/timing/Watch.h"
#include "tarch/parallel/Node.h"
#include "tarch/multicore/Lock.h"

#include "tarch/Assertions.h"

tarch::logging::Log peanoclaw::pyclaw::PyClaw::_log("peanoclaw::pyclaw::PyClaw");

peanoclaw::pyclaw::PyClaw::PyClaw(
  InitializationCallback                                initializationCallback,
  BoundaryConditionCallback                             boundaryConditionCallback,
  SolverCallback                                        solverCallback,
  RefinementCriterionCallback                           refinementCriterionCallback,
  AddPatchToSolutionCallback                            addPatchToSolutionCallback,
  peanoclaw::interSubgridCommunication::Interpolation*  interpolation,
  peanoclaw::interSubgridCommunication::Restriction*    restriction,
  peanoclaw::interSubgridCommunication::FluxCorrection* fluxCorrection
) : Numerics(interpolation, restriction, fluxCorrection),
_initializationCallback(initializationCallback),
_boundaryConditionCallback(boundaryConditionCallback),
_solverCallback(solverCallback),
_refinementCriterionCallback(refinementCriterionCallback),
_addPatchToSolutionCallback(addPatchToSolutionCallback),
_totalSolverCallbackTime(0.0),
_cachedDemandedMeshWidth(-1.0),
_cachedSubgridPosition(0.0),
_cachedSubgridLevel(-1)
{
  //import_array();
}

peanoclaw::pyclaw::PyClaw::~PyClaw()
{
}


void peanoclaw::pyclaw::PyClaw::initializePatch(
  Patch& patch
) {
  logTraceIn( "initializePatch(...)");

  tarch::multicore::Lock lock(_semaphore);

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
    patch.getNumberOfParametersWithoutGhostlayerPerSubcell(),
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

  //Cache demanded mesh width
  _cachedSubgridPosition = patch.getPosition();
  _cachedSubgridLevel = patch.getLevel();
  _cachedDemandedMeshWidth = tarch::la::Vector<DIMENSIONS,double>(demandedMeshWidth);

  logTraceOutWith1Argument( "initializePatch(...)", demandedMeshWidth);
}

void peanoclaw::pyclaw::PyClaw::solveTimestep(Patch& patch, double maximumTimestepSize, bool useDimensionalSplitting) {
  logTraceInWith2Arguments( "solveTimestep(...)", maximumTimestepSize, useDimensionalSplitting);

  tarch::multicore::Lock lock(_semaphore);

  assertion2(tarch::la::greater(maximumTimestepSize, 0.0), "max. Timestepsize == 0 should be checked outside.", patch.getTimeIntervals().getMinimalNeighborTimeConstraint());
  assertion3(!patch.containsNaN(), patch, patch.toStringUNew(), patch.toStringUOldWithGhostLayer());

  PyClawState state(patch);

  tarch::timing::Watch pyclawWatch("", "", false);
  pyclawWatch.startTimer();

  double dtAndEstimatedNextDt[2];
  dtAndEstimatedNextDt[0] = 0.0;
  dtAndEstimatedNextDt[1] = 0.0;

  bool doDummyTimestep = false;
  double requiredMeshWidth;
  if(doDummyTimestep) {
    dtAndEstimatedNextDt[0] = std::min(maximumTimestepSize, patch.getTimeIntervals().getEstimatedNextTimestepSize());
    dtAndEstimatedNextDt[1] = patch.getTimeIntervals().getEstimatedNextTimestepSize();
    requiredMeshWidth = patch.getSubcellSize()(0);
  } else {
    requiredMeshWidth = _solverCallback(
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
      patch.getNumberOfParametersWithoutGhostlayerPerSubcell(),      // naux
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
      patch.getTimeIntervals().getCurrentTime() + patch.getTimeIntervals().getTimestepSize(),
      maximumTimestepSize,
      patch.getTimeIntervals().getEstimatedNextTimestepSize(),
      useDimensionalSplitting
    );
  }

  pyclawWatch.stopTimer();
  _totalSolverCallbackTime += pyclawWatch.getCalendarTime();

  assertion3(tarch::la::greater(dtAndEstimatedNextDt[0], 0.0), patch, patch.toStringUNew(), patch.toStringUOldWithGhostLayer());

  assertion4(
      tarch::la::greater(patch.getTimeIntervals().getTimestepSize(), 0.0)
      || tarch::la::greater(dtAndEstimatedNextDt[1], 0.0)
      || tarch::la::equals(maximumTimestepSize, 0.0)
      || tarch::la::equals(patch.getTimeIntervals().getEstimatedNextTimestepSize(), 0.0),
      patch, maximumTimestepSize, dtAndEstimatedNextDt[1], patch.toStringUNew());
  assertion(patch.getTimeIntervals().getTimestepSize() < std::numeric_limits<double>::infinity());

  //Check for zeros in solution
  assertion3(!patch.containsNonPositiveNumberInUnknownInUNew(0), patch, patch.toStringUNew(), patch.toStringUOldWithGhostLayer());

  patch.getTimeIntervals().advanceInTime();
  patch.getTimeIntervals().setTimestepSize(dtAndEstimatedNextDt[0]);

  //Cache demanded mesh width
  _cachedSubgridPosition = patch.getPosition();
  _cachedSubgridLevel = patch.getLevel();
  _cachedDemandedMeshWidth = tarch::la::Vector<DIMENSIONS,double>(requiredMeshWidth);

  logTraceOutWith1Argument( "solveTimestep(...)", requiredMeshWidth);
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::pyclaw::PyClaw::getDemandedMeshWidth(Patch& patch, bool isInitializing) {
  if(
    tarch::la::equals(patch.getPosition(), _cachedSubgridPosition)
    && patch.getLevel() == _cachedSubgridLevel
  ) {
    return _cachedDemandedMeshWidth;
  } else {
    PyClawState state(patch);
    double demandedMeshWidth = _refinementCriterionCallback(
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
      patch.getNumberOfParametersWithoutGhostlayerPerSubcell(),
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

    //Cache demanded mesh width
    _cachedSubgridPosition = patch.getPosition();
    _cachedSubgridLevel = patch.getLevel();
    _cachedDemandedMeshWidth = tarch::la::Vector<DIMENSIONS,double>(demandedMeshWidth);

    return _cachedDemandedMeshWidth;
  }
}

void peanoclaw::pyclaw::PyClaw::addPatchToSolution(Patch& patch) {

  tarch::multicore::Lock lock(_semaphore);

  PyClawState state(patch);

  assertion(_addPatchToSolutionCallback != 0);
  _addPatchToSolutionCallback(
    state._q,
    state._qbc,
    patch.getGhostlayerWidth(),
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
    patch.getTimeIntervals().getCurrentTime()+patch.getTimeIntervals().getTimestepSize()
  );
}

void peanoclaw::pyclaw::PyClaw::fillBoundaryLayer(Patch& patch, int dimension, bool setUpper) {
  logTraceInWith3Arguments("fillBoundaryLayerInPyClaw", patch, dimension, setUpper);

  tarch::multicore::Lock lock(_semaphore);

  logDebug("fillBoundaryLayerInPyClaw", "Setting boundary for " << patch.getPosition() << ", dim=" << dimension << ", setUpper=" << setUpper);

  PyClawState state(patch);
  _boundaryConditionCallback(state._q, state._qbc, dimension, setUpper ? 1 : 0);

  logTraceOut("fillBoundaryLayerInPyClaw");
}

void peanoclaw::pyclaw::PyClaw::update(Patch& finePatch) {

}

