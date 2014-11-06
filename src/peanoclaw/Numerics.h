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

  namespace interSubgridCommunication {
    class DefaultTransfer;
  }
}

/**
 * Interface for functionality provided by the coupled framework
 * (I.e. PyClaw).
 */
class peanoclaw::Numerics {

private:
  peanoclaw::interSubgridCommunication::DefaultTransfer* _transfer;
  peanoclaw::interSubgridCommunication::Interpolation*   _interpolation;
  peanoclaw::interSubgridCommunication::Restriction*     _restriction;
  peanoclaw::interSubgridCommunication::FluxCorrection*  _fluxCorrection;

  public:
    Numerics(
        peanoclaw::interSubgridCommunication::DefaultTransfer* transfer,
      peanoclaw::interSubgridCommunication::Interpolation*     interpolation,
      peanoclaw::interSubgridCommunication::Restriction*       restriction,
      peanoclaw::interSubgridCommunication::FluxCorrection*    fluxCorrection
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
     * Transfers ghostlayer values from the source subgrid to
     * the destination subgrid.
     */
    virtual void transferGhostlayer(
      const tarch::la::Vector<DIMENSIONS, int>&    size,
      const tarch::la::Vector<DIMENSIONS, int>&    sourceOffset,
      const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
      peanoclaw::Patch& source,
      peanoclaw::Patch&       destination
    ) const;

    /**
     * Performs the interpolation between the given source and destination
     * by means of the interpolation method implemented in Python. I.e. this
     * method can only be called if providesInterpolation() returns <tt>true</tt>.
     */
    virtual void interpolateSolution (
      const tarch::la::Vector<DIMENSIONS, int>&    destinationSize,
      const tarch::la::Vector<DIMENSIONS, int>&    destinationOffset,
      peanoclaw::Patch& source,
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
    virtual void restrictSolution (
      peanoclaw::Patch& source,
      peanoclaw::Patch& destination,
      bool              restrictOnlyOverlappedAreas
    ) const;

    /**
     * As the restriction is a splitted process over all fine subgrids
     * this event allows to do operations after all restriction steps
     * are finished.
     */
    virtual void postProcessRestriction(
      peanoclaw::Patch& destination,
      bool              restrictOnlyOverlappedAreas
    ) const;

    /**
     * Performs the flux correction between the given source and destination
     * by means of the restriction method implemented in Python. I.e. this
     * method can only be called if providesRestriction() returns <tt>true</tt>.
     */
    virtual void applyFluxCorrection (
      const Patch& sourceSubgrid,
      Patch& destinationSubgrid,
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
      Patch& subgrid,
      int dimension,
      bool setUpper
    ) = 0;

    /**
     * Solves a timestep. All updates (e.g. change of grid values, taken timestep size, new cfl number)
     * are performed on the patch object
     *
     * @param patch The Patch object holding the grid data.
     * @param maximumTimestepSize The maximal timestep size with regard to the current global timestep.
     * @param useDimensionalSplitting Enables dimensional splitting for the solver if available.
     * @param domainBoundaryFlags Specifies which boundaries of the subgrid are actual boundaries of the domain.
     *
     */
    virtual void solveTimestep(
      Patch& subgrid,
      double maximumTimestepSize,
      bool useDimensionalSplitting,
      tarch::la::Vector<DIMENSIONS_TIMES_TWO, bool> domainBoundaryFlags
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

    /**
     * Returns the number of unknowns that have to be stored per cell.
     */
    virtual int getNumberOfUnknownsPerCell() const = 0;

    virtual int getNumberOfParameterFieldsWithoutGhostlayer() const = 0;

    virtual int getNumberOfParameterFieldsWithGhostlayer() const = 0;

    /**
     * Returns the ghostlayer width in cells that is required for the applied solver.
     */
    virtual int getGhostlayerWidth() const = 0;
};


#endif /* PEANOCLAW_NUMERICS_H_ */
