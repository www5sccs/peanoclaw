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
   * @see peanoclaw::Numerics
   */
  void fillBoundaryLayer(Patch& patch, int dimension, bool setUpper);

  /**
   * @see peanoclaw::Numerics
   */
  int getNumberOfUnknownsPerCell() const { return 3; }

  int getNumberOfParameterFieldsWithoutGhostlayer() const { return 1; }

  int getNumberOfParameterFieldsWithGhostlayer() const { return 0; }

  /**
   * @see peanoclaw::Numerics
   */
  int getGhostlayerWidth() const { return 1; }
};

class peanoclaw::native::SWEKernelScenario {
    public:
        virtual ~SWEKernelScenario() {}
        virtual void initializePatch(Patch& patch) = 0;
        virtual tarch::la::Vector<DIMENSIONS,double> computeDemandedMeshWidth(Patch& patch, bool isInitializing) = 0;
        virtual void update(Patch& patch) = 0;

        virtual tarch::la::Vector<DIMENSIONS,double> getDomainOffset() const = 0;
        virtual tarch::la::Vector<DIMENSIONS,double> getDomainSize() const = 0;
        virtual tarch::la::Vector<DIMENSIONS,double> getInitialMinimalMeshWidth() const = 0;
        virtual tarch::la::Vector<DIMENSIONS,int>    getSubdivisionFactor() const = 0;
        virtual double getGlobalTimestepSize() const = 0;
        virtual double getEndTime() const = 0;
        virtual double getInitialTimestepSize() const = 0;
    protected:
        SWEKernelScenario() {}
};

#endif /* PEANOCLAW_SWEKERNEL_NATIVE_H_ */
