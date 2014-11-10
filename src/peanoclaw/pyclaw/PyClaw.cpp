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

peanoclaw::pyclaw::PyClaw::initialize(Patch& subgrid, bool skipQInitialization) {
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
    patch.getPosition()(2),
    #else
      0,
    #endif
    skipQInitialization
  );

  //Cache demanded mesh width
  _cachedSubgridPosition = patch.getPosition();
  _cachedSubgridLevel = patch.getLevel();
  _cachedDemandedMeshWidth = tarch::la::Vector<DIMENSIONS,double>(demandedMeshWidth);
}

peanoclaw::pyclaw::PyClaw::PyClaw(
  InitializationCallback                                 initializationCallback,
  BoundaryConditionCallback                              boundaryConditionCallback,
  SolverCallback                                         solverCallback,
  RefinementCriterionCallback                            refinementCriterionCallback,
  AddPatchToSolutionCallback                             addPatchToSolutionCallback,
  peanoclaw::interSubgridCommunication::DefaultTransfer* transfer,
  peanoclaw::interSubgridCommunication::Interpolation*   interpolation,
  peanoclaw::interSubgridCommunication::Restriction*     restriction,
  peanoclaw::interSubgridCommunication::FluxCorrection*  fluxCorrection,
  int numberOfUnknowns,
  int numberOfAuxFields,
  int ghostlayerWidth
) : Numerics(transfer, interpolation, restriction, fluxCorrection),
_initializationCallback(initializationCallback),
_boundaryConditionCallback(boundaryConditionCallback),
_solverCallback(solverCallback),
_refinementCriterionCallback(refinementCriterionCallback),
_addPatchToSolutionCallback(addPatchToSolutionCallback),
_totalSolverCallbackTime(0.0),
_numberOfUnknowns(numberOfUnknowns),
_numberOfAuxFields(numberOfAuxFields),
_ghostlayerWidth(ghostlayerWidth),
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
  Patch& subgrid
) {
  logTraceIn( "initializePatch(...)");

  initialize(subgrid, false);

  logTraceOutWith1Argument( "initializePatch(...)", demandedMeshWidth);
}

void peanoclaw::pyclaw::PyClaw::solveTimestep(
  Patch& subgrid,
  double maximumTimestepSize,
  bool useDimensionalSplitting,
  tarch::la::Vector<DIMENSIONS_TIMES_TWO, bool> domainBoundaryFlags
) {
  logTraceInWith2Arguments( "solveTimestep(...)", maximumTimestepSize, useDimensionalSplitting);

  tarch::multicore::Lock lock(_semaphore);

  assertion2(tarch::la::greater(maximumTimestepSize, 0.0), "max. Timestepsize == 0 should be checked outside.", subgrid.getTimeIntervals().getMinimalNeighborTimeConstraint());
  assertion3(!subgrid.containsNaN(), subgrid, subgrid.toStringUNew(), subgrid.toStringUOldWithGhostLayer());

  PyClawState state(subgrid);

  tarch::timing::Watch pyclawWatch("", "", false);
  pyclawWatch.startTimer();

  double dtAndEstimatedNextDt[2];
  dtAndEstimatedNextDt[0] = 0.0;
  dtAndEstimatedNextDt[1] = 0.0;

  bool doDummyTimestep = false;
  double requiredMeshWidth;
  if(doDummyTimestep) {
    dtAndEstimatedNextDt[0] = std::min(maximumTimestepSize, subgrid.getTimeIntervals().getEstimatedNextTimestepSize());
    dtAndEstimatedNextDt[1] = subgrid.getTimeIntervals().getEstimatedNextTimestepSize();
    requiredMeshWidth = subgrid.getSubcellSize()(0);
  } else {
    requiredMeshWidth = _solverCallback(
      dtAndEstimatedNextDt,
      state._q,
      state._qbc,
      state._aux,
      subgrid.getSubdivisionFactor()(0),
      subgrid.getSubdivisionFactor()(1),
      #ifdef Dim3
      subgrid.getSubdivisionFactor()(2),
      #else
      0,
      #endif
      subgrid.getUnknownsPerSubcell(),            // meqn = nvar
      subgrid.getNumberOfParametersWithoutGhostlayerPerSubcell(),      // naux
      subgrid.getSize()(0),
      subgrid.getSize()(1),
      #ifdef Dim3
      subgrid.getSize()(2),
      #else
      0,
      #endif
      subgrid.getPosition()(0),
      subgrid.getPosition()(1),
      #ifdef Dim3
      subgrid.getPosition()(2),
      #else
      0,
      #endif
      subgrid.getTimeIntervals().getCurrentTime() + subgrid.getTimeIntervals().getTimestepSize(),
      maximumTimestepSize,
      subgrid.getTimeIntervals().getEstimatedNextTimestepSize(),
      useDimensionalSplitting
    );
  }

  pyclawWatch.stopTimer();
  _totalSolverCallbackTime += pyclawWatch.getCalendarTime();

  assertion3(tarch::la::greater(dtAndEstimatedNextDt[0], 0.0), subgrid, subgrid.toStringUNew(), subgrid.toStringUOldWithGhostLayer());

  assertion4(
      tarch::la::greater(subgrid.getTimeIntervals().getTimestepSize(), 0.0)
      || tarch::la::greater(dtAndEstimatedNextDt[1], 0.0)
      || tarch::la::equals(maximumTimestepSize, 0.0)
      || tarch::la::equals(subgrid.getTimeIntervals().getEstimatedNextTimestepSize(), 0.0),
      subgrid, maximumTimestepSize, dtAndEstimatedNextDt[1], subgrid.toStringUNew());
  assertion(subgrid.getTimeIntervals().getTimestepSize() < std::numeric_limits<double>::infinity());

  //Check for zeros in solution
  assertion3(!subgrid.containsNonPositiveNumberInUnknownInUNew(0), subgrid, subgrid.toStringUNew(), subgrid.toStringUOldWithGhostLayer());

  subgrid.getTimeIntervals().advanceInTime();
  subgrid.getTimeIntervals().setTimestepSize(dtAndEstimatedNextDt[0]);
  subgrid.getTimeIntervals().setEstimatedNextTimestepSize(dtAndEstimatedNextDt[1]);

  //Cache demanded mesh width
  _cachedSubgridPosition = subgrid.getPosition();
  _cachedSubgridLevel = subgrid.getLevel();
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

void peanoclaw::pyclaw::PyClaw::update(Patch& subgrid) {
  initialize(subgrid, true);
}

int peanoclaw::pyclaw::PyClaw::getNumberOfUnknownsPerCell() const {
  return _numberOfUnknowns;
}

int peanoclaw::pyclaw::PyClaw::getNumberOfParameterFieldsWithoutGhostlayer() const {
  return _numberOfAuxFields;
}

int peanoclaw::pyclaw::PyClaw::getGhostlayerWidth() const {
  return _ghostlayerWidth;
}
