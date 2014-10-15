/*
 * Euler3DKernel.h
 *
 *  Created on: Jul 24, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_SOLVER_EULER3D_EULER3DKERNEL_H_
#define PEANOCLAW_SOLVER_EULER3D_EULER3DKERNEL_H_

namespace peanoclaw {
  namespace solver {
    namespace euler3d {
      class Euler3DKernel;
      class ExtrapolateBoundaryCondition;
    }
  }
}

#include "peanoclaw/solver/euler3d/Cell.h"

#include "peanoclaw/Numerics.h"
#include "peanoclaw/native/scenarios/SWEScenario.h"
#include "peanoclaw/grid/SubgridAccessor.h"

#ifdef PEANOCLAW_EULER3D
#include <tbb/task_scheduler_init.h>
#endif

class peanoclaw::solver::euler3d::Euler3DKernel : public peanoclaw::Numerics {
  private:
    /**
     * Logging device
     */
    static tarch::logging::Log     _log;

    peanoclaw::native::scenarios::SWEScenario& _scenario;

    #ifdef PEANOCLAW_EULER3D
    tbb::task_scheduler_init _task_scheduler_init;
    #endif

  public:
    Euler3DKernel(
      peanoclaw::native::scenarios::SWEScenario& scenario,
      peanoclaw::interSubgridCommunication::DefaultTransfer* transfer,
      peanoclaw::interSubgridCommunication::Interpolation*   interpolation,
      peanoclaw::interSubgridCommunication::Restriction*     restriction,
      peanoclaw::interSubgridCommunication::FluxCorrection*  fluxCorrection,
      int numberOfThreads
    );

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
    int getNumberOfUnknownsPerCell() const { return NUMBER_OF_EULER_UNKNOWNS; }

    int getNumberOfParameterFieldsWithoutGhostlayer() const { return 0; }

    int getNumberOfParameterFieldsWithGhostlayer() const { return 0; }

    /**
     * @see peanoclaw::Numerics
     */
    int getGhostlayerWidth() const { return 1; }

    double computeTimestep(
      double dt,
      peanoclaw::Patch& subgrid//,
//      std::vector<peanoclaw::solver::euler3d::Cell>& cellsUNew,
//      std::vector<peanoclaw::solver::euler3d::Cell>& cellsUOld
    );
};

class peanoclaw::solver::euler3d::ExtrapolateBoundaryCondition {
  public:
    void setBoundaryCondition(
      peanoclaw::Patch& subgrid,
      peanoclaw::grid::SubgridAccessor& accessor,
      int dimension,
      bool setUpper,
      tarch::la::Vector<DIMENSIONS,int> sourceSubcellIndex,
      tarch::la::Vector<DIMENSIONS,int> destinationSubcellIndex
    );
};

#endif /* PEANOCLAW_SOLVER_EULER3D_EULER3DKERNEL_H_ */
