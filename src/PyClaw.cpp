/*
 * PyClaw.cpp
 *
 *  Created on: Feb 18, 2012
 *      Author: kristof
 */

#include <Python.h>
#include "PyClaw.h"
#include "Patch.h"
#include <numpy/arrayobject.h>
#include "tarch/timing/Watch.h"

tarch::logging::Log peanoclaw::PyClaw::_log("peanoclaw::PyClaw");

peanoclaw::PyClawState::PyClawState(const Patch& patch) {
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

peanoclaw::PyClawState::~PyClawState() {
}

peanoclaw::PyClaw::PyClaw(
    double (*initializationCallback)(PyObject* q,
      PyObject* qbc,
      PyObject* aux,
      int subdivisionFactorX0,
      int subdivisionFactorX1,
      int subdivisionFactorX2,
      int unknownsPerSubcell,
      int auxFieldsPerSubcell,
      double sizeX,
      double sizeY,
      double sizeZ,
      double positionX,
      double positionY,
      double positionZ),
    void (*boundaryConditionCallback)(PyObject* q,
      PyObject* qbc,
      int dimension,
      int setUpper),
    double (*solverCallback)(double* dtAndCfl,
      PyObject* q,
      PyObject* qbc,
      PyObject* aux,
      int subdivisionFactorX0,
      int subdivisionFactorX1,
      int subdivisionFactorX2,
      int unknownsPerSubcell,
      int auxFieldsPerSubcell,
      double sizeX,
      double sizeY,
      double sizeZ,
      double positionX,
      double positionY,
      double positionZ,
      double currentTime,
      double maximumTimestepSize,
      double estimatedNextTimestepSize,
      bool useDimensionalSplitting),
    void (*addPatchToSolutionCallback)(PyObject* q,
      PyObject* qbc,
      int ghostlayerWidth,
      double sizeX,
      double sizeY,
      double sizeZ,
      double positionX,
      double positionY,
      double positionZ,
      double currentTime)
) : _initializationCallback(initializationCallback),
_boundaryConditionCallback(boundaryConditionCallback),
_solverCallback(solverCallback),
_addPatchToSolutionCallback(addPatchToSolutionCallback),
_totalSolverCallbackTime(0.0)
{
  import_array();
}

double peanoclaw::PyClaw::initializePatch(Patch& patch) {
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

/*
 45     def callback_initialization(q, qbc, aux, subdivision_factor_x0, subdivision_factor_x1, subdivision_factor_x2, unknowns_per_subcell, aux_fields_per_subcell, si    ze_x, size_y, size_z, position_x, position_y, position_z):
 46         import clawpack.pyclaw as pyclaw
 47         self.dim_x = pyclaw.Dimension('x',position_x,position_x + size_x,subdivision_factor_x0)
 48         self.dim_y = pyclaw.Dimension('y',position_y,position_y + size_y,subdivision_factor_x1)
 49         #TODO 3D: use size_z and position_z
 50         domain = pyclaw.Domain([self.dim_x,self.dim_y])
 51         subgrid_state = pyclaw.State(domain, unknowns_per_subcell, aux_fields_per_subcell)
 52         subgrid_state.q = q
 53         if(aux_fields_per_subcell > 0):
 54           subgrid_state.aux = aux
 55         subgrid_state.problem_data = self.solver.solution.state.problem_data
 56 
 57         self.q_initialization(subgrid_state)
 58 
 59         if(self.aux_initialization != None and aux_fields_per_subcell > 0):
 60           self.aux_initialization(subgrid_state)
 61 
 62         #Steer refinement
 63         if self.refinement_criterion != None:
 64           return self.refinement_criterion(subgrid_state)
 65         else:
 66           return self.initial_minimal_mesh_width



    q_initialization -> qinit
    aux_initialization -> None (currently) | setaux?
    refinement_criterion -> <function>

*/

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

void peanoclaw::PyClaw::fillBoundaryLayerInPyClaw(Patch& patch, int dimension, bool setUpper) {
  logTraceInWith3Arguments("fillBoundaryLayerInPyClaw", patch, dimension, setUpper);

  logDebug("fillBoundaryLayerInPyClaw", "Setting left boundary for " << patch.getPosition() << ", dim=" << dimension << ", setUpper=" << setUpper);

  PyClawState state(patch);

  _boundaryConditionCallback(state._q, state._qbc, dimension, setUpper ? 1 : 0);

/* 28     def callback_boundary_conditions(q, qbc, dimension, setUpper):
 29       import numpy
 30       if(setUpper == 1):
 31         self.solver.qbc_upper(self.solver.solution.state, self.solver.solution.state.grid.dimensions[dimension], self.solver.solution.state.t, numpy.rollaxis(qbc,    dimension+1,1), dimension)
 32       else:
 33         self.solver.qbc_lower(self.solver.solution.state, self.solver.solution.state.grid.dimensions[dimension], self.solver.solution.state.t, numpy.rollaxis(qbc,    dimension+1,1), dimension)*/


  logTraceOut("fillBoundaryLayerInPyClaw");
}

peanoclaw::PyClaw::~PyClaw()
{
}

double peanoclaw::PyClaw::solveTimestep(Patch& patch, double maximumTimestepSize, bool useDimensionalSplitting) {
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

void peanoclaw::PyClaw::addPatchToSolution(Patch& patch) {

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


#ifdef Dim2
void peanoclaw::PyClaw::fillLeftBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 0, false);
}

void peanoclaw::PyClaw::fillUpperBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 1, true);
}

void peanoclaw::PyClaw::fillRightBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 0, true);
}

void peanoclaw::PyClaw::fillLowerBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 1, false);
}
#endif


#ifdef Dim3
void peanoclaw::PyClaw::fillLeftBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 0, false);
}

void peanoclaw::PyClaw::fillBehindBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 1, true);
}

void peanoclaw::PyClaw::fillRightBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 0, true);
}

void peanoclaw::PyClaw::fillFrontBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 1, false);
}

void peanoclaw::PyClaw::fillUpperBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 2, true);
}

void peanoclaw::PyClaw::fillLowerBoundaryLayer(Patch& patch) {
  fillBoundaryLayerInPyClaw(patch, 2, false);
}
#endif
