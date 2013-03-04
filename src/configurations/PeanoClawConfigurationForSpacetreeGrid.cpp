#include "configurations/PeanoClawConfigurationForSpacetreeGrid.h"
#include "runners/Runner.h"
#include "runners/PeanoClawLibraryRunner.h"

#include "tarch/logging/CommandLineLogger.h"

tarch::logging::Log peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid::_log("peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid");

peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid::PeanoClawConfigurationForSpacetreeGrid():
  _isValid(true),
  _plotAtOutputTimes(true),
  _plotSubsteps(false),
  _plotSubstepsAfterOutputTime(-1),
  _additionalLevelsForPredefinedRefinement(1),
  _disableDimensionalSplittingOptimization(true)
  {
}


peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid::~PeanoClawConfigurationForSpacetreeGrid() {
}


bool peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid::isValid() const { 
  return true;
}

bool peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid::plotAtOutputTimes() const {
  return _plotAtOutputTimes;
}
 
bool peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid::plotSubsteps() const {
  return _plotSubsteps;
}

int peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid::plotSubstepsAfterOutputTime() const {
  return _plotSubstepsAfterOutputTime;
}

int peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid::getAdditionalLevelsForPredefinedRefinement() const {
  return _additionalLevelsForPredefinedRefinement;
}

bool peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid::disableDimensionalSplittingOptimization() const {
  return _disableDimensionalSplittingOptimization;
}
