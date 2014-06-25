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

#include "peanoclaw/interSubgridCommunication/Interpolation.h"
#include "peanoclaw/interSubgridCommunication/Restriction.h"
#include "peanoclaw/interSubgridCommunication/FluxCorrection.h"

#include "peanoclaw/native/scenarios/SWEScenario.h"

#include <memory>

class SWE_WavePropagationBlock_patch;
namespace peanoclaw {
  namespace native {
    class SWEKernel;
  }
} /* namespace peanoclaw */

class peanoclaw::native::SWEKernel  : public peanoclaw::Numerics
{
private:
  /**
   * Logging device
   */
  static tarch::logging::Log     _log;

  double _totalSolverCallbackTime;

  peanoclaw::native::scenarios::SWEScenario& _scenario;

  tarch::la::Vector<DIMENSIONS,int> _cachedSubdivisionFactor;
  int                               _cachedGhostlayerWidth;
  std::auto_ptr<SWE_WavePropagationBlock_patch> _cachedBlock;

public:
  SWEKernel(
    peanoclaw::native::scenarios::SWEScenario& scenario,
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
   * Initializes parameter when a subgrid is created due to refinement
   * or coarsening.
   */
  void update(Patch& subgrid);

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

  int getNumberOfParameterFieldsWithoutGhostlayer() const { return 0; }

  int getNumberOfParameterFieldsWithGhostlayer() const { return 1; }

  /**
   * @see peanoclaw::Numerics
   */
  int getGhostlayerWidth() const { return 1; }
};

#endif /* PEANOCLAW_SWEKERNEL_NATIVE_H_ */
