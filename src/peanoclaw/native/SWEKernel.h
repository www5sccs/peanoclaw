/*
 * PyClaw.h
 *
 *  Created on: Feb 18, 2012
 *      Author: Kristof Unterweger
 */

#ifndef PEANOCLAW_NATIVE_SWEKERNEL_H_
#define PEANOCLAW_NATIVE_SWEKERNEL_H_

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

namespace peanoclaw {
  namespace native {
    class SWEKernel;
    class SWEKernelScenario;
  }
} /* namespace peanoclaw */

class peanoclaw::native::SWEKernel  : public peanoclaw::Numerics
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

  SWEKernelScenario& _scenario;
public:
  SWEKernel(
    SWEKernelScenario& scenario,
    peanoclaw::interSubgridCommunication::DefaultTransfer* transfer,
    peanoclaw::interSubgridCommunication::Interpolation*  interpolation,
    peanoclaw::interSubgridCommunication::Restriction*    restriction,
    peanoclaw::interSubgridCommunication::FluxCorrection* fluxCorrection
  );

  virtual ~SWEKernel();

  /**
   * Initializes the given patch at the beginning of a simulation run.
   *
   * @return The mesh width demanded by the application.
   */
  void initializePatch(Patch& patch);

  /**
   * Solves a timestep. All updates (e.g. change of grid values, taken timestep size, new cfl number)
   * are performed on the patch object
   *
   * @param patch The Patch object holding the grid data.
   * @param maximumTimestepSize The maximal timestep size with regard to the current global timestep.
   */
  void solveTimestep(Patch& patch, double maximumTimestepSize, bool useDimensionalSplitting);

  /**
   * Returns the demanded mesh width for the given subgrid.
   */
  tarch::la::Vector<DIMENSIONS, double> getDemandedMeshWidth(Patch& patch, bool isInitializing);

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

//#ifdef Dim2
//  /**
//   * Fills in the left boundary layer.
//   * This method assumes that patch.uOld already holds
//   * the current solution. I.e. uNew was already copied
//   * to uOld.
//   *
//   * TODO unterweg: Es ist nicht schoen, dass vorausgesetzt wird, dass das Umkopieren von uNew auf uOld
//   * schon durchgefuehrt wurde. Das kann man auch auf PyClawseite erledigen, indem dort die Daten aus
//   * q statt qbc geholt werden.
//   */
//  void fillLeftBoundaryLayer(Patch& patch);
//
//  /**
//   * Fills in the upper boundary layer.
//   * This method assumes that patch.uOld already holds
//   * the current solution. I.e. uNew was already copied
//   * to uOld.
//   */
//  void fillUpperBoundaryLayer(Patch& patch);
//
//  /**
//   * Fills in the right boundary layer.
//   * This method assumes that patch.uOld already holds
//   * the current solution. I.e. uNew was already copied
//   * to uOld.
//   */
//  void fillRightBoundaryLayer(Patch& patch);
//
//  /**
//   * Fills in the lower boundary layer.
//   * This method assumes that patch.uOld already holds
//   * the current solution. I.e. uNew was already copied
//   * to uOld.
//   */
//  void fillLowerBoundaryLayer(Patch& patch);
//
//#endif
//#ifdef Dim3
//  void fillLeftBoundaryLayer(Patch& patch);
//  void fillBehindBoundaryLayer(Patch& patch);
//  void fillRightBoundaryLayer(Patch& patch);
//  void fillFrontBoundaryLayer(Patch& patch);
//  void fillUpperBoundaryLayer(Patch& patch) ;
//  void fillLowerBoundaryLayer(Patch& patch);
//#endif

};

class peanoclaw::native::SWEKernelScenario {
    public:
        virtual ~SWEKernelScenario() {}
        virtual void initializePatch(Patch& patch) = 0;
        virtual tarch::la::Vector<DIMENSIONS,double> computeDemandedMeshWidth(Patch& patch, bool isInitializing) = 0;
        virtual void update(Patch& patch) = 0;
    protected:
        SWEKernelScenario() {}
};

#endif /* PEANOCLAW_SWEKERNEL_NATIVE_H_ */
