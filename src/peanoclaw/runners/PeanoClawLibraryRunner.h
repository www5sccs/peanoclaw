/*
 * PeanoClawLibraryRunner.h
 *
 *  Created on: Feb 7, 2012
 *      Author: Kristof Unterweger
 */

#ifndef PEANO_APPLICATIONS_PEANOCLAW_RUNNERS_PEANOCLAWLIBRARYRUNNER_H_
#define PEANO_APPLICATIONS_PEANOCLAW_RUNNERS_PEANOCLAWLIBRARYRUNNER_H_

#include "peano/utils/Globals.h"
#include "tarch/la/Vector.h"

#include "peanoclaw/repositories/Repository.h"
#include "peanoclaw/parallel/PeanoClawControlLoopLoadBalancer.h"

#include "tarch/logging/Log.h"
#include "tarch/timing/Watch.h"

#include "peano/geometry/Hexahedron.h"

namespace peanoclaw {

  class Numerics;

  namespace configurations {
    class PeanoClawConfigurationForSpacetreeGrid;
  }
  namespace runners {
    class PeanoClawLibraryRunner;
  }
}


class peanoclaw::runners::PeanoClawLibraryRunner
{
private:

  /**
   * Logging device
   */
  static tarch::logging::Log _log;

  peano::geometry::Hexahedron* _geometry;

  /**
   * Pointer to the repository which actually stores all grid data etc.
   */
  peanoclaw::repositories::Repository* _repository;

  int _plotNumber;

  peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid& _configuration;

  tarch::timing::Watch _iterationTimer;

  double _totalRuntime;

  peanoclaw::Numerics& _numerics;

  bool _validateGrid;

  peanoclaw::parallel::PeanoClawControlLoopLoadBalancer _controlLoopLoadBalancer;

  tarch::timing::Watch _initializationWatch;
  tarch::timing::Watch _simulationWatch;

  /**
   * Initializes geometry and grid.
   */
  void initializePeano(
    tarch::la::Vector<DIMENSIONS, double> domainOffset,
    tarch::la::Vector<DIMENSIONS, double> domainSize
  );

  /**
   * Initializes the MPI environment
   */
  void initializeParallelEnvironment();

  void iterateInitialiseGrid();

  void iteratePlot();

  void iterateSolveTimestep(bool plotSubsteps);

  void iterateGatherSolution();

public:
  /**
   * Sets everything up but does not start any grid-traversal, yet.
   */
  PeanoClawLibraryRunner(
    peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid& configuration,
    peanoclaw::Numerics& numerics,
    const tarch::la::Vector<DIMENSIONS, double>& domainOffset,
    const tarch::la::Vector<DIMENSIONS, double>& domainSize,
    const tarch::la::Vector<DIMENSIONS, double>& minimalMeshWidth,
    const tarch::la::Vector<DIMENSIONS, int>& subdivisionFactor,
    int ghostLayerWidth,
    int unknownsPerSubcell,
    int auxiliarFieldsPerSubcell,
    double initialTimestepSize,
    bool useDimensionalSplitting,
    bool reduceReductions,
    int  forkLevelIncrement = 1
  );

  virtual ~PeanoClawLibraryRunner();

  peanoclaw::Numerics& getNumerics() { return _numerics; }

  /**
   * Evolves the solution up to the given point in time.
   */
  void evolveToTime( double time  );

  /**
   * Gathers the current solution, i.e. all patches.
   */
  void gatherCurrentSolution();
  int runWorker();

  void configureGlobalTimestep(double time);
  void runNextPossibleTimestep();

  void updateOracle();
};
#endif /* PEANO_APPLICATIONS_PEANOCLAW_RUNNERS_PEANOCLAWLIBRARYRUNNER_H_ */
