/*
 * PyClaw.h
 *
 *  Created on: Feb 18, 2012
 *      Author: Kristof Unterweger
 */

#ifndef PEANOCLAW_PYCLAW_PYCLAW_H_
#define PEANOCLAW_PYCLAW_PYCLAW_H_

#include "peanoclaw/Numerics.h"
#include "peanoclaw/pyclaw/PyClawCallbacks.h"
#include "peanoclaw/pyclaw/InterpolationCallbackWrapper.h"
#include "peanoclaw/pyclaw/RestrictionCallbackWrapper.h"
#include "peanoclaw/pyclaw/FluxCorrectionCallbackWrapper.h"

#include "peanoclaw/interSubgridCommunication/Interpolation.h"
#include "peanoclaw/interSubgridCommunication/Restriction.h"
#include "peanoclaw/interSubgridCommunication/FluxCorrection.h"

#include "peano/utils/Dimensions.h"

#include "tarch/logging/Log.h"
#include "tarch/la/Vector.h"
#include "tarch/multicore/BooleanSemaphore.h"

namespace peanoclaw {
  namespace pyclaw {
    class PyClaw;
    class PyClawState;
  }
  class Patch;
} /* namespace peanoclaw */

class peanoclaw::pyclaw::PyClaw  : public peanoclaw::Numerics
{
private:
  /**
   * Logging device
   */
  static tarch::logging::Log         _log;

  InitializationCallback             _initializationCallback;

  BoundaryConditionCallback          _boundaryConditionCallback;

  SolverCallback                     _solverCallback;

  AddPatchToSolutionCallback         _addPatchToSolutionCallback;

  double                             _totalSolverCallbackTime;

  tarch::multicore::BooleanSemaphore _semaphore;

public:
  PyClaw(InitializationCallback   initializationCallback,
    BoundaryConditionCallback      boundaryConditionCallback,
    SolverCallback                 solverCallback,
    AddPatchToSolutionCallback     addPatchToSolutionCallback,
    peanoclaw::interSubgridCommunication::Interpolation*  interpolation,
    peanoclaw::interSubgridCommunication::Restriction*    restriction,
    peanoclaw::interSubgridCommunication::FluxCorrection* fluxCorrection
  );

  virtual ~PyClaw();

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

  /**
   * @see peanoclaw::Numerics
   */
  void fillBoundaryLayer(Patch& patch, int dimension, bool setUpper);

};
#endif /* PEANOCLAW_PYCLAW_PYCLAW_H_ */
