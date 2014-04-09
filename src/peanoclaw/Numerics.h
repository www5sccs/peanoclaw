/*
 * Numerics.h
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_NUMERICS_H_
#define PEANOCLAW_NUMERICS_H_

#include "peanoclaw/interSubgridCommunication/Interpolation.h"
#include "peanoclaw/interSubgridCommunication/Restriction.h"
#include "peanoclaw/interSubgridCommunication/FluxCorrection.h"

#include "tarch/la/Vector.h"

namespace peanoclaw {
  class Numerics;
  class Patch;
}

/**
 * Interface for functionality provided by the coupled framework
 * (I.e. PyClaw).
 */
class peanoclaw::Numerics {

private:
  peanoclaw::interSubgridCommunication::Interpolation*  _interpolation;
  peanoclaw::interSubgridCommunication::Restriction*    _restriction;
  peanoclaw::interSubgridCommunication::FluxCorrection* _fluxCorrection;

  public:
    Numerics(
      peanoclaw::interSubgridCommunication::Interpolation*  interpolation,
      peanoclaw::interSubgridCommunication::Restriction*    restriction,
      peanoclaw::interSubgridCommunication::FluxCorrection* fluxCorrection
    );

    virtual ~Numerics();

    /**
     * Adds a patch to the solution which is hold in PyClaw. This method is used for gathering a solution
     * holding the complete grid in PyClaw to plot it via VisClaw.
     */
    virtual void addPatchToSolution(Patch& patch) = 0;

    /**
     * Initializes the given patch at the beginning of a simulation run.
     *
     */
    virtual void initializePatch(Patch& patch) = 0;

    /**
     * Performs the interpolation between the given source and destination
     * by means of the interpolation method implemented in Python. I.e. this
     * method can only be called if providesInterpolation() returns <tt>true</tt>.
     */
    virtual void interpolate(
      const tarch::la::Vector<DIMENSIONS, int>&    destinationSize,
      const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
      const peanoclaw::Patch& source,
      peanoclaw::Patch&        destination,
      bool interpolateToUOld,
      bool interpolateToCurrentTime,
      bool useTimeUNewOrTimeUOld
    ) const;

    /**
     * Performs the restriction between the given source and destination
     * by means of the restriction method implemented in Python. I.e. this
     * method can only be called if providesRestriction() returns <tt>true</tt>.
     */
    virtual void restrict (
      const peanoclaw::Patch& source,
      peanoclaw::Patch&       destination,
      bool restrictOnlyOverlappedAreas
    ) const;

    /**
     * Performs the flux correction between the given source and destination
     * by means of the restriction method implemented in Python. I.e. this
     * method can only be called if providesRestriction() returns <tt>true</tt>.
     */
    virtual void applyFluxCorrection (
      const Patch& finePatch,
      Patch& coarsePatch,
      int dimension,
      int direction
    ) const;

    /**
     * Fills the specified boundary layer.
     *
     * @param patch The patch for which the boundary needs to be set.
     * @param dimension The dimension in perpendicular to which the
     * boundary needs to be set.
     * @param setUpper Specifies, whether the upper or the lower part
     * of the boundary needs to be set.
     */
    virtual void fillBoundaryLayer(
      Patch& patch,
      int dimension,
      bool setUpper
    ) = 0;

    /**
     * Solves a timestep. All updates (e.g. change of grid values, taken timestep size, new cfl number)
     * are performed on the patch object
     *
     * @param patch The Patch object holding the grid data.
     * @param maximumTimestepSize The maximal timestep size with regard to the current global timestep.
     *
     */
    virtual void solveTimestep(
      Patch& patch,
      double maximumTimestepSize,
      bool useDimensionalSplitting
    ) = 0;

    /**
     * Retrieves the demanded mesh width for the given patch.
     *
     * @return The mesh width demanded by the application. This is assumed to be the minimal mesh width
     * over all dimensions, hence it's a scalar value.
     */
    virtual tarch::la::Vector<DIMENSIONS, double> getDemandedMeshWidth(
      Patch& patch,
      bool   isInitializing
    ) = 0;


    /**
     * this function is e.g. useful to update auxillary data like bathymetry on a finer leverl
     */
    virtual void update (Patch& finePatch);
};


#endif /* PEANOCLAW_NUMERICS_H_ */
