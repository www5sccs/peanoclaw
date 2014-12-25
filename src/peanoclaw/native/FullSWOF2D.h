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

#include "peanoclaw/interSubgridCommunication/Interpolation.h"
#include "peanoclaw/interSubgridCommunication/Restriction.h"
#include "peanoclaw/interSubgridCommunication/FluxCorrection.h"

#include "peanoclaw/native/SWEKernel.h"
#include "peanoclaw/native/scenarios/SWEScenario.h"
#include "peanoclaw/native/scenarios/FullSWOF2DBoundaryCondition.h"

#include "peanoclaw/native/MekkaFlood_solver.h"

#ifdef PEANOCLAW_FULLSWOF2D
#include "parameters.hpp"
#include "scheme.hpp"
#include "choice_scheme.hpp"
#endif

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

  static peanoclaw::native::scenarios::FullSWOF2DBoundaryCondition _interSubgridBoundaryCondition;

  double _totalSolverCallbackTime;

  peanoclaw::native::scenarios::SWEScenario& _scenario;

  #ifdef PEANOCLAW_FULLSWOF2D
  Choice_scheme* _wrapperScheme;
  #endif

  /**
   * Transforms the water height in the given subgrid from relative
   * height above ground to absolute water height.
   * Also it transforms velocities to momenta.
   */
  void transformToAbsoluteWaterHeightAndMomenta(
    peanoclaw::Patch&                  subgrid,
    const peanoclaw::geometry::Region& region,
    bool                               modifyUOld
  ) const;

  void transformToRelativeWaterHeightAndVelocities(
    peanoclaw::Patch&                  subgrid,
    const peanoclaw::geometry::Region& region,
    bool                               modifyUOld
  ) const;

public:
  FullSWOF2D(
    peanoclaw::native::scenarios::SWEScenario& scenario,
    peanoclaw::interSubgridCommunication::DefaultTransfer* transfer,
    peanoclaw::interSubgridCommunication::Interpolation*   interpolation,
    peanoclaw::interSubgridCommunication::Restriction*     restriction,
    peanoclaw::interSubgridCommunication::FluxCorrection*  fluxCorrection
  );

  virtual ~FullSWOF2D();

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
   * @param useDimensionalSplitting @see peanoclaw::Numerics
   * @param domainBoundaryFlags @see peanoclaw::Numerics
   */
  void solveTimestep(
    Patch& subgrid,
    double maximumTimestepSize,
    bool useDimensionalSplitting,
    tarch::la::Vector<DIMENSIONS_TIMES_TWO, bool> domainBoundaryFlags
  );

  /**
   * Returns the demanded mesh width for the given patch.
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
//  void interpolateSolution (
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
//    bool restrictOnlyOverlappedRegions
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

  #ifdef PEANOCLAW_FULLSWOF2D
  void copyPatchToScheme(Patch& patch, Scheme* scheme, tarch::la::Vector<DIMENSIONS_TIMES_TWO, int> margin);
  void copySchemeToPatch(Scheme* scheme, Patch& patch, tarch::la::Vector<DIMENSIONS_TIMES_TWO, int> margin);
  #endif

//  void copyPatchToSet(Patch& patch, unsigned int *strideinfo, MekkaFlood_solver::InputArrays& input, MekkaFlood_solver::TempArrays& temp);
//  void copySetToPatch(unsigned int *strideinfo, MekkaFlood_solver::InputArrays& input, MekkaFlood_solver::TempArrays& temp, Patch& patch);

  /*
   * @see peanoclaw::Numerics
   */
  int getNumberOfUnknownsPerCell() const { return 6; }

  /*
   * @see peanoclaw::Numerics
   */
  int getNumberOfParameterFieldsWithoutGhostlayer() const { return 1; }

  /*
   * @see peanoclaw::Numerics
   */
  int getNumberOfParameterFieldsWithGhostlayer() const { return 1; }

  /*
   * @see peanoclaw::Numerics
   */
  int getGhostlayerWidth() const { return 3; }

  #ifdef PEANOCLAW_FULLSWOF2D
  Scheme* getScheme() const { return _wrapperScheme->getInternalScheme(); };
  #endif

  /*
   * Modifies the source subgrid so that not the water height above
   * the seafloor is interpolated but the absolute water height. Also,
   * the velocities are transformed to momenta.
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

  /*
   * Modifies the source subgrid so that not the water height above
   * the seafloor is restricted but the absolute water height.
   */
  virtual void restrictSolution (
    peanoclaw::Patch& source,
    peanoclaw::Patch& destination,
    bool              restrictOnlyOverlappedRegions
  ) const;

  /**
   * Transforms the destination subgrid from absolute water height
   * to water height above seafloor.
   */
  virtual void postProcessRestriction(
    peanoclaw::Patch& destination,
    bool              restrictOnlyOverlappedRegions
  ) const;
};

#ifdef PEANOCLAW_FULLSWOF2D
class peanoclaw::native::FullSWOF2D_Parameters : public Parameters {
    private:
        double _endTime;

    public:
        FullSWOF2D_Parameters(
          int ghostlayerWidth,
          int nx,
          int ny,
          double meshwidth_x,
          double meshwidth_y,
          tarch::la::Vector<DIMENSIONS,double> domainSize,
          double endTime,
          bool enableRain,
          double friction,
          peanoclaw::native::scenarios::FullSWOF2DBoundaryCondition leftBoundaryCondition,
          peanoclaw::native::scenarios::FullSWOF2DBoundaryCondition rightBoundaryCondition,
          peanoclaw::native::scenarios::FullSWOF2DBoundaryCondition bottomBoundaryCondition,
          peanoclaw::native::scenarios::FullSWOF2DBoundaryCondition topBoundaryCondition,
          int select_order=2,
          int select_reconstruction=1
        );

        virtual ~FullSWOF2D_Parameters();

        double get_T() const;
};
#endif

#endif /* PEANOCLAW_SWEKERNEL_NATIVE_H_ */
