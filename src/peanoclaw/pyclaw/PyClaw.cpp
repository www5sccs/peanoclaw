/*
 * PyClaw.cpp
 *
 *  Created on: Feb 18, 2012
 *      Author: kristof
 */

#include <Python.h>
#include "peanoclaw/pyclaw/PyClaw.h"
#include "peanoclaw/Patch.h"
#include <numpy/arrayobject.h>
#include "tarch/timing/Watch.h"

tarch::logging::Log peanoclaw::pyclaw::PyClaw::_log("peanoclaw::pyclaw::PyClaw");

peanoclaw::pyclaw::PyClawState::PyClawState(const Patch& patch) {
  //Create uNew
  npy_intp sizeUNew[1 + DIMENSIONS];
  sizeUNew[0] = patch.getUnknownsPerSubcell();
  int elementsUNew = sizeUNew[0];
  for(int d = 0; d < DIMENSIONS; d++) {
    sizeUNew[1 + d] = patch.getSubdivisionFactor()(d);
    elementsUNew *= sizeUNew[1 + d];
  }
  npy_intp sizeUOldWithGhostlayer[1 + DIMENSIONS];
  sizeUOldWithGhostlayer[0] = patch.getUnknownsPerSubcell();
  int elementsUOldWithGhostlayer = sizeUOldWithGhostlayer[0];
  for(int d = 0; d < DIMENSIONS; d++) {
    sizeUOldWithGhostlayer[1 + d] = patch.getSubdivisionFactor()(d) + 2*patch.getGhostLayerWidth();
    elementsUOldWithGhostlayer *= sizeUOldWithGhostlayer[1 + d];
  }

  _q = PyArray_SimpleNewFromData(1 + DIMENSIONS, sizeUNew, NPY_DOUBLE, patch.getUNewArray());
  _qbc = PyArray_SimpleNewFromData(1 + DIMENSIONS, sizeUOldWithGhostlayer, NPY_DOUBLE, patch.getUOldWithGhostlayerArray());

  //Build auxArray
  double* auxArray = 0;
  if(patch.getAuxiliarFieldsPerSubcell() > 0) {
    auxArray = patch.getAuxArray();
    assertion(auxArray != 0);
    npy_intp sizeAux[1 + DIMENSIONS];
    sizeAux[0] = patch.getAuxiliarFieldsPerSubcell();
    for(int d = 0; d < DIMENSIONS; d++) {
      sizeAux[1 + d] = patch.getSubdivisionFactor()(d);
    }
    _aux = PyArray_SimpleNewFromData(1 + DIMENSIONS, sizeAux, NPY_DOUBLE, auxArray);
    assertion(_aux != 0);
  } else {
    //TODO Here I'm creating an unecessary PyObject. Would be great, if we could just pass
    //"None" to Python.
    auxArray = 0;
    npy_intp sizeZero[1 + DIMENSIONS];
    for(int d = 0; d < 1+ DIMENSIONS; d++) {
      sizeZero[d] = 0;
    }
    _aux = PyArray_SimpleNewFromData(1 + DIMENSIONS, sizeZero, NPY_DOUBLE, auxArray);
  }
}

peanoclaw::pyclaw::PyClaw::PyClaw(
    InitializationCallback initializationCallback,
    BoundaryConditionCallback boundaryConditionCallback,
    SolverCallback solverCallback,
    AddPatchToSolutionCallback addPatchToSolutionCallback,
    InterPatchCommunicationCallback interpolationCallback,
    InterPatchCommunicationCallback restrictionCallback
) : _initializationCallback(initializationCallback),
_boundaryConditionCallback(boundaryConditionCallback),
_solverCallback(solverCallback),
_addPatchToSolutionCallback(addPatchToSolutionCallback),
_interpolationCallback(interpolationCallback),
_restrictionCallback(restrictionCallback),
_totalSolverCallbackTime(0.0)
{
  import_array();
}

peanoclaw::pyclaw::PyClawState::~PyClawState() {
}

double peanoclaw::pyclaw::PyClaw::initializePatch(Patch& patch) {
  logTraceIn( "initializePatch(...)");
  //TODO unterweg debug
//  std::cout << "INITIALIZE" << std::endl;

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

void peanoclaw::pyclaw::PyClaw::fillBoundaryLayerInPyClaw(Patch& patch, int dimension, bool setUpper) const {
  logTraceInWith3Arguments("fillBoundaryLayerInPyClaw", patch, dimension, setUpper);

  logDebug("fillBoundaryLayerInPyClaw", "Setting left boundary for " << patch.getPosition() << ", dim=" << dimension << ", setUpper=" << setUpper);

  PyClawState state(patch);

  _boundaryConditionCallback(state._q, state._qbc, dimension, setUpper ? 1 : 0);

  logTraceOut("fillBoundaryLayerInPyClaw");
}

peanoclaw::pyclaw::PyClaw::~PyClaw()
{
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

bool peanoclaw::pyclaw::PyClaw::providesInterpolation() const {
  return _interpolationCallback != 0;
}

void peanoclaw::pyclaw::PyClaw::interpolate(
  const Patch& source,
  Patch& destination
) const {

  PyClawState sourceState(source);
  PyClawState destinationState(destination);

  _interpolationCallback(
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

bool peanoclaw::pyclaw::PyClaw::providesRestriction() const {
  return _restrictionCallback != 0;
}

void peanoclaw::pyclaw::PyClaw::restrict (
  const peanoclaw::Patch& source,
  peanoclaw::Patch&       destination
) const {
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

#ifdef Dim2
void peanoclaw::pyclaw::PyClaw::fillLeftBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 0, false);
}

void peanoclaw::pyclaw::PyClaw::fillUpperBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 1, true);
}

void peanoclaw::pyclaw::PyClaw::fillRightBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 0, true);
}

void peanoclaw::pyclaw::PyClaw::fillLowerBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 1, false);
}
#endif


#ifdef Dim3
void peanoclaw::pyclaw::PyClaw::fillLeftBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 0, false);
}

void peanoclaw::pyclaw::PyClaw::fillBehindBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 1, true);
}

void peanoclaw::pyclaw::PyClaw::fillRightBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 0, true);
}

void peanoclaw::pyclaw::PyClaw::fillFrontBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 1, false);
}

void peanoclaw::pyclaw::PyClaw::fillUpperBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 2, true);
}

void peanoclaw::pyclaw::PyClaw::fillLowerBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 2, false);
}
#endif
