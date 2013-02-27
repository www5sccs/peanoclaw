/*
 * PyClaw.h
 *
 *  Created on: Feb 18, 2012
 *      Author: Kristof Unterweger
 */

#ifndef PEANO_APPLICATIONS_PEANOCLAW_PYCLAW_H_
#define PEANO_APPLICATIONS_PEANOCLAW_PYCLAW_H_

#include "tarch/logging/Log.h"

#include "PyClawCallbacks.h"

namespace peanoclaw {
//class TimestepResult;
  class PyClaw;
  class PyClawState;
  class Patch;
} /* namespace peanoclaw */

#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif

/**
 * Holds the timestepsize taken for the last timestep and the
 * resulting CFL number.
 */
//class peanoclaw::TimestepResult {
//public:
//  double _dt;
//  double _cfl;
//
//  TimestepResult(double dt, double cfl);
//
//  double getDt();
//
//  double getCfl();
//};

/**
 * A state object encapsulates all NumPy arrays for the PyClaw state.
 *
 * This class is used for a RAII like allocation of memory. I.e. during
 * the constructor the memory for the arrays is allocated dynamically
 * and it is deleted on destruction of a PyClawState object.
 */
class peanoclaw::PyClawState {
  public:
//    double* _uNew;
//    double* _uOldWithGhostlayer;
    PyObject* _q;
    PyObject* _qbc;
    PyObject* _aux;

    /**
     * Creates the Numpy arrays and allocates memory when needed.
     */
    PyClawState(const Patch& patch);

    /**
     * Deletes allocated memory.
     */
    ~PyClawState();
};

class peanoclaw::PyClaw
{
private:
  /**
   * Logging device
   */
  static tarch::logging::Log _log;

  InitializationCallback _initializationCallback;

  BoundaryConditionCallback _boundaryConditionCallback;

  SolverCallback _solverCallback;

  AddPatchToSolutionCallback _addPatchToSolutionCallback;

  double _totalSolverCallbackTime;

  void fillBoundaryLayerInPyClaw(Patch& patch, int dimension, bool setUpper);

public:
  PyClaw(InitializationCallback initializationCallback,
         BoundaryConditionCallback boundaryConditionCallback,
         SolverCallback solverCallback,
         AddPatchToSolutionCallback addPatchToSolutionCallback);

  ~PyClaw();

  /**
   * Initializes the given patch at the beginning of a simulation run.
   *
   * @return The mesh width demanded by the application.
   */
  double initializePatch(Patch& patch);

  /**
   * Solves a timestep. All updates (e.g. change of grid values, taken timestep size, new cfl number)
   * are performed on the patch object
   *
   * @param patch The Patch object holding the grid data.
   * @param maximumTimestepSize The maximal timestep size with regard to the current global timestep.
   */
  double solveTimestep(Patch& patch, double maximumTimestepSize, bool useDimensionalSplitting);

  /**
   * Adds a patch to the solution which is hold in PyClaw. This method is used for gathering a solution
   * holding the complete grid in PyClaw to plot it via VisClaw.
   */
  void addPatchToSolution(Patch& patch);
#ifdef Dim2
  /**
   * Fills in the left boundary layer.
   * This method assumes that patch.uOld already holds
   * the current solution. I.e. uNew was already copied
   * to uOld.
   *
   * TODO unterweg: Es ist nicht schoen, dass vorausgesetzt wird, dass das Umkopieren von uNew auf uOld
   * schon durchgefuehrt wurde. Das kann man auch auf PyClawseite erledigen, indem dort die Daten aus
   * q statt qbc geholt werden.
   */
  void fillLeftBoundaryLayer(Patch& patch);

  /**
   * Fills in the upper boundary layer.
   * This method assumes that patch.uOld already holds
   * the current solution. I.e. uNew was already copied
   * to uOld.
   */
  void fillUpperBoundaryLayer(Patch& patch);

  /**
   * Fills in the right boundary layer.
   * This method assumes that patch.uOld already holds
   * the current solution. I.e. uNew was already copied
   * to uOld.
   */
  void fillRightBoundaryLayer(Patch& patch);

  /**
   * Fills in the lower boundary layer.
   * This method assumes that patch.uOld already holds
   * the current solution. I.e. uNew was already copied
   * to uOld.
   */
  void fillLowerBoundaryLayer(Patch& patch);

#endif
#ifdef Dim3
  void fillLeftBoundaryLayer(Patch& patch);
  void fillBehindBoundaryLayer(Patch& patch);
  void fillRightBoundaryLayer(Patch& patch);
  void fillFrontBoundaryLayer(Patch& patch);
  void fillUpperBoundaryLayer(Patch& patch) ;
  void fillLowerBoundaryLayer(Patch& patch);
#endif

};
#endif /* PEANO_APPLICATIONS_PEANOCLAW_PYCLAW_H_ */
