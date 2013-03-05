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
#include "peanoclaw/pyclaw/PyClaw.h"

#include "tarch/logging/Log.h"
#include "tarch/timing/Watch.h"

#include "peano/geometry/Hexahedron.h"

namespace peanoclaw {
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

public:
  /**
   * Sets everything up but does not start any grid-traversal, yet.
   */
  PeanoClawLibraryRunner(
    peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid& configuration,
    peanoclaw::pyclaw::PyClaw& pyClaw,
    const tarch::la::Vector<DIMENSIONS, double>& domainOffset,
    const tarch::la::Vector<DIMENSIONS, double>& domainSize,
    const tarch::la::Vector<DIMENSIONS, double>& minimalMeshWidth,
    const tarch::la::Vector<DIMENSIONS, int>& subdivisionFactor,
    int ghostLayerWidth,
    int unknownsPerSubcell,
    int auxiliarFieldsPerSubcell,
    double initialTimestepSize,
    bool useDimensionalSplitting
  );

  virtual ~PeanoClawLibraryRunner();

  /**
   * Evolves the solution up to the given point in time.
   */
  void evolveToTime(
    double time,
    peanoclaw::pyclaw::PyClaw& pyClaw);

  /**
   * Gathers the current solution, i.e. all patches, in PyClaw.
   */
  void gatherCurrentSolution(peanoclaw::pyclaw::PyClaw& pyClaw);
};
#endif /* PEANO_APPLICATIONS_PEANOCLAW_RUNNERS_PEANOCLAWLIBRARYRUNNER_H_ */
