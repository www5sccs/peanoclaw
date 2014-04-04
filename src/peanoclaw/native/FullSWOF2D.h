/*
 * PyClaw.h
 *
 *  Created on: Feb 18, 2012
 *      Author: Kristof Unterweger
 */

#ifndef PEANOCLAW_NATIVE_FULLSWOF2D_H_
#define PEANOCLAW_NATIVE_FULLSWOF2D_H_

#include "tarch/logging/Log.h"
#include "tarch/la/Vector.h"
#include "peano/utils/Dimensions.h"

#include "peanoclaw/Numerics.h"
/*#include "peanoclaw/pyclaw/PyClawCallbacks.h"
#include "peanoclaw/pyclaw/InterpolationCallbackWrapper.h"
#include "peanoclaw/pyclaw/RestrictionCallbackWrapper.h"
#include "peanoclaw/pyclaw/FluxCorrectionCallbackWrapper.h"*/

#include "peanoclaw/interSubgridCommunication/Interpolation.h"
#include "peanoclaw/interSubgridCommunication/Restriction.h"
#include "peanoclaw/interSubgridCommunication/FluxCorrection.h"

#include "peanoclaw/native/SWEKernel.h"

#include "peanoclaw/native/MekkaFlood_solver.h"

#include "parameters.hpp"

#include "choice_scheme.hpp"

namespace peanoclaw {
  namespace native {
    class FullSWOF2D;
    class FullSWOF2D_Parameters;
  }
} /* namespace peanoclaw */

class peanoclaw::native::FullSWOF2D  : public peanoclaw::Numerics
{
private:
  /**
   * Logging device
   */
  static tarch::logging::Log     _log;

  /*InitializationCallback         _initializationCallback;

  BoundaryConditionCallback      _boundaryConditionCallback;

  SolverCallback                 _solverCallback;

  AddPatchToSolutionCallback     _addPatchToSolutionCallback;*/

  double _totalSolverCallbackTime;

  peanoclaw::native::SWEKernelScenario& _scenario;

public:
  FullSWOF2D(
    SWEKernelScenario& scenario,
    peanoclaw::interSubgridCommunication::Interpolation*  interpolation,
    peanoclaw::interSubgridCommunication::Restriction*    restriction,
    peanoclaw::interSubgridCommunication::FluxCorrection* fluxCorrection
  );

  virtual ~FullSWOF2D();

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
   * Performs the interpolation between the given source and destination
   * by means of the interpolation method implemented in Python. I.e. this
   * method can only be called if providesInterpolation() returns <tt>true</tt>.
   */
//  void interpolate(
//    const tarch::la::Vector<DIMENSIONS, int>&    destinationSize,
//    const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
//    const peanoclaw::Patch& source,
//    peanoclaw::Patch&        destination,
//    bool interpolateToUOld = true,
//    bool interpolateToCurrentTime = true
//  ) const;

  /**
   * Performs the restriction between the given source and destination
   * by means of the restriction method implemented in Python. I.e. this
   * method can only be called if providesRestriction() returns <tt>true</tt>.
   */
//  void restrict (
//    const peanoclaw::Patch& source,
//    peanoclaw::Patch&       destination,
//    bool restrictOnlyOverlappedAreas
//  ) const;

  /**
   * Performs the flux correction between the given source and destination
   * by means of the restriction method implemented in Python. I.e. this
   * method can only be called if providesRestriction() returns <tt>true</tt>.
   */
//  void applyFluxCorrection (
//    const Patch& finePatch,
//    Patch& coarsePatch,
//    int dimension,
//    int direction
//  ) const;

  /**
   * @see peanoclaw::Numerics
   */
  void fillBoundaryLayer(Patch& patch, int dimension, bool setUpper);

  void update(Patch& finePatch);

  void copyPatchToScheme(Patch& patch, Scheme* scheme);
  void copySchemeToPatch(Scheme* scheme, Patch& patch);

  void copyPatchToSet(Patch& patch, unsigned int *strideinfo, MekkaFlood_solver::InputArrays& input, MekkaFlood_solver::TempArrays& temp);
  void copySetToPatch(unsigned int *strideinfo, MekkaFlood_solver::InputArrays& input, MekkaFlood_solver::TempArrays& temp, Patch& patch);
};

class peanoclaw::native::FullSWOF2D_Parameters : public Parameters {
    public:
        FullSWOF2D_Parameters(int ghostlayerWidth, int nx, int ny, double meshwidth_x, double meshwidth_y, int select_order=2, int select_reconstruction=1);
        virtual ~FullSWOF2D_Parameters();
};


#endif /* PEANOCLAW_SWEKERNEL_NATIVE_H_ */
