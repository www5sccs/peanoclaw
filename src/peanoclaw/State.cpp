#include "State.h"
#include "Cell.h"
#include "Vertex.h"

#include "peanoclaw/statistics/SubgridStatistics.h"

#include "peano/grid/Checkpoint.h"

tarch::logging::Log peanoclaw::State::_log("peanoclaw::State");

peanoclaw::State::State():
  Base(), _numerics(0) {
  resetGlobalTimeIntervals();
  resetMinimalTimestep();
  resetTotalNumberOfCellUpdates();

  //Flags that might not be initialized by Peano
  _stateData.setHasChangedVertexOrCellState(false);
  _stateData.setHasRefined(false);
  _stateData.setHasErased(false);
  _stateData.setHasTriggeredRefinementForNextIteration(false);
  _stateData.setHasTriggeredEraseForNextIteration(false);
  #ifdef Parallel
  _stateData.setCouldNotEraseDueToDecompositionFlag(false);
  _stateData.setSubWorkerIsInvolvedInJoinOrFork(false);
  #endif
}


peanoclaw::State::State(const Base::PersistentState& argument):
  Base(argument), _numerics(0) {
  // @todo Insert your code here
}


void peanoclaw::State::writeToCheckpoint( peano::grid::Checkpoint<peanoclaw::Vertex,peanoclaw::Cell>& checkpoint ) const {
  // @todo Insert your code here
}

    
void peanoclaw::State::readFromCheckpoint( const peano::grid::Checkpoint<peanoclaw::Vertex,peanoclaw::Cell>& checkpoint ) {
  // @todo Insert your code here
}

int peanoclaw::State::getPlotNumber() const {
  return _stateData.getPlotNumber();
}

void peanoclaw::State::setPlotNumber(int plotNumber) {
  _stateData.setPlotNumber(plotNumber);
}

void peanoclaw::State::setUnknownsPerSubcell(int unknownsPerSubcell) {
  _stateData.setUnknownsPerSubcell(unknownsPerSubcell);
}

int peanoclaw::State::getUnknownsPerSubcell() const {
  return _stateData.getUnknownsPerSubcell();
}

void peanoclaw::State::setNumberOfParametersWithoutGhostlayerPerSubcell(int numberOfParametersWithoutGhostlayerPerSubcell) {
  _stateData.setNumberOfParametersWithoutGhostlayerPerSubcell(numberOfParametersWithoutGhostlayerPerSubcell);
}

int peanoclaw::State::getNumberOfParametersWithoutGhostlayerPerSubcell() const {
  return _stateData.getNumberOfParametersWithoutGhostlayerPerSubcell();
}

void peanoclaw::State::setNumberOfParametersWithGhostlayerPerSubcell(int numberOfParametersWithGhostlayerPerSubcell) {
  _stateData.setNumberOfParametersWithGhostlayerPerSubcell(numberOfParametersWithGhostlayerPerSubcell);
}

int peanoclaw::State::getNumberOfParametersWithGhostlayerPerSubcell() const {
  return _stateData.getNumberOfParametersWithGhostlayerPerSubcell();
}

void peanoclaw::State::setDefaultSubdivisionFactor(const tarch::la::Vector<DIMENSIONS, int>& defaultSubdivisionFactor) {
  _stateData.setDefaultSubdivisionFactor(defaultSubdivisionFactor);
}

tarch::la::Vector<DIMENSIONS, int> peanoclaw::State::getDefaultSubdivisionFactor() const {
  return _stateData.getDefaultSubdivisionFactor();
}

void peanoclaw::State::setDefaultGhostLayerWidth(int defaultGhostLayerWidth) {
  _stateData.setDefaultGhostWidthLayer(defaultGhostLayerWidth);
}

int peanoclaw::State::getDefaultGhostLayerWidth() const {
  return _stateData.getDefaultGhostWidthLayer();
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::State::getInitialMaximalSubgridSize() const {
  return _stateData.getInitialMaximalSubgridSize();
}

void peanoclaw::State::setInitialMaximalSubgridSize(const tarch::la::Vector<DIMENSIONS, double>& h) {
  _stateData.setInitialMaximalSubgridSize(h);
}

void peanoclaw::State::setInitialTimestepSize(double initialTimestepSize) {
  _stateData.setInitialTimestepSize(initialTimestepSize);
}

double peanoclaw::State::getInitialTimestepSize() const {
  return _stateData.getInitialTimestepSize();
}

void peanoclaw::State::setNumerics(peanoclaw::Numerics& numerics) {
  _numerics = &numerics;
}

peanoclaw::Numerics* peanoclaw::State::getNumerics() const {
  return _numerics;
}

void peanoclaw::State::setProbeList(std::vector<peanoclaw::statistics::Probe> probeList) {
  _probeList = probeList;
}

std::vector<peanoclaw::statistics::Probe>& peanoclaw::State::getProbeList() {
  return _probeList;
}

void peanoclaw::State::setGlobalTimestepEndTime(double globalTimestepEndTime) {
  _stateData.setGlobalTimestepEndTime(globalTimestepEndTime);
}

double peanoclaw::State::getGlobalTimestepEndTime() const {
  return _stateData.getGlobalTimestepEndTime();
}

void peanoclaw::State::setAllPatchesEvolvedToGlobalTimestep(bool value) {
  _stateData.setAllPatchesEvolvedToGlobalTimestep(value);
}

bool peanoclaw::State::getAllPatchesEvolvedToGlobalTimestep() const {
  return _stateData.getAllPatchesEvolvedToGlobalTimestep();
}

void peanoclaw::State::setDomain(const tarch::la::Vector<DIMENSIONS, double>& offset, const tarch::la::Vector<DIMENSIONS, double>& size) {
  _stateData.setDomainOffset(offset);
  _stateData.setDomainSize(size);
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::State::getDomainOffset() {
  return _stateData.getDomainOffset();
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::State::getDomainSize() {
  return _stateData.getDomainSize();
}

void peanoclaw::State::updateGlobalTimeIntervals(
  double startMaximumLocalTimeInterval,
  double endMaximumLocalTimeInterval,
  double startMinimumLocalTimeInterval,
  double endMinimumLocalTimeInterval
) {
  _stateData.setStartMaximumGlobalTimeInterval(
    std::min(startMaximumLocalTimeInterval, _stateData.getStartMaximumGlobalTimeInterval())
  );
  _stateData.setEndMaximumGlobalTimeInterval(
    std::max(endMaximumLocalTimeInterval, _stateData.getEndMaximumGlobalTimeInterval())
  );
  _stateData.setStartMinimumGlobalTimeInterval(
    std::max(startMinimumLocalTimeInterval, _stateData.getStartMinimumGlobalTimeInterval())
  );
  _stateData.setEndMinimumGlobalTimeInterval(
    std::min(endMinimumLocalTimeInterval, _stateData.getEndMinimumGlobalTimeInterval())
  );
}

void peanoclaw::State::resetGlobalTimeIntervals() {
  _stateData.setStartMaximumGlobalTimeInterval(std::numeric_limits<double>::max());
  _stateData.setEndMaximumGlobalTimeInterval(-std::numeric_limits<double>::max());
  _stateData.setStartMinimumGlobalTimeInterval(-std::numeric_limits<double>::max());
  _stateData.setEndMinimumGlobalTimeInterval(std::numeric_limits<double>::max());
}

double peanoclaw::State::getStartMaximumGlobalTimeInterval() const {
  return _stateData.getStartMaximumGlobalTimeInterval();
}

double peanoclaw::State::getEndMaximumGlobalTimeInterval() const {
  return _stateData.getEndMaximumGlobalTimeInterval();
}

double peanoclaw::State::getStartMinimumGlobalTimeInterval() const {
  return _stateData.getStartMinimumGlobalTimeInterval();
}

double peanoclaw::State::getEndMinimumGlobalTimeInterval() const {
  return _stateData.getEndMinimumGlobalTimeInterval();
}

void peanoclaw::State::resetTotalNumberOfCellUpdates() {
  _stateData.setTotalNumberOfCellUpdates(0.0);
}

double peanoclaw::State::getTotalNumberOfCellUpdates() const {
  return _stateData.getTotalNumberOfCellUpdates();
}

void peanoclaw::State::resetMinimalTimestep() {
  _stateData.setMinimalTimestep(std::numeric_limits<double>::max());
}

void peanoclaw::State::updateMinimalTimestep(double timestep) {
  _stateData.setMinimalTimestep(std::min(timestep, _stateData.getMinimalTimestep()));
}

double peanoclaw::State::getMinimalTimestep() const {
  return _stateData.getMinimalTimestep();
}

void peanoclaw::State::setLevelStatisticsForLastGridIteration (
  const std::vector<peanoclaw::statistics::LevelStatistics>& levelStatistics
) {
  _levelStatisticsForLastGridIteration = levelStatistics;
  _levelStatisticsHistory.push_back(levelStatistics);
}

std::list<std::vector<peanoclaw::statistics::LevelStatistics> > peanoclaw::State::getLevelStatisticsHistory() const {
  return _levelStatisticsHistory;
}

void peanoclaw::State::plotStatisticsForLastGridIteration() const {
  peanoclaw::statistics::SubgridStatistics subgridStatistics(_levelStatisticsForLastGridIteration);
  subgridStatistics.logLevelStatistics("Statistics for last grid iteration");
}

void peanoclaw::State::plotTotalStatistics() const {
  peanoclaw::statistics::SubgridStatistics totalStatistics;
  for(std::list<std::vector<LevelStatistics> >::const_iterator i = _levelStatisticsHistory.begin();
      i != _levelStatisticsHistory.end(); i++) {
    totalStatistics.merge(*i);
  }
  totalStatistics.averageTotalSimulationValues(_levelStatisticsHistory.size());
  totalStatistics.logLevelStatistics("Total Statistics");
}

void peanoclaw::State::setIsInitializing(bool isInitializing) {
  _stateData.setIsInitializing(isInitializing);
}

bool peanoclaw::State::getIsInitializing() const {
  return _stateData.getIsInitializing();
}

void peanoclaw::State::enableRefinementCriterion(bool enabled) {
  _stateData.setIsRefinementCriterionEnabled(enabled);
}

bool peanoclaw::State::isRefinementCriterionEnabled() const {
  return _stateData.getIsRefinementCriterionEnabled();
}


void peanoclaw::State::setUseDimensionalSplittingOptimization(bool useDimensionalSplittingOptimization) {
  _stateData.setUseDimensionalSplittingOptimization(useDimensionalSplittingOptimization);
}

bool peanoclaw::State::useDimensionalSplittingOptimization() const {
  return _stateData.getUseDimensionalSplittingOptimization();
}

//void peanoclaw::State::resetLocalHeightOfWorkerTree() {
//  #ifdef Parallel
//  _stateData.setGlobalHeightOfWorkerTreeDuringLastIteration(
//    _stateData.getLocalHeightOfWorkerTree()
//  );
//  _stateData.setLocalHeightOfWorkerTree(0);
//  #endif
//}
//
//void peanoclaw::State::increaseLocalHeightOfWorkerTree() {
//  #ifdef Parallel
//  _stateData.setLocalHeightOfWorkerTree(_stateData.getLocalHeightOfWorkerTree() + 1);
//  #endif
//}
//
//void peanoclaw::State::updateLocalHeightOfWorkerTree(int localHeightOfWorkerTree) {
//  #ifdef Parallel
//  _stateData.setLocalHeightOfWorkerTree(
//    std::max(_stateData.getLocalHeightOfWorkerTree(), localHeightOfWorkerTree)
//  );
//  #endif
//}
//
//int peanoclaw::State::getLocalHeightOfWorkerTree() const {
//  #ifdef Parallel
//  return _stateData.getLocalHeightOfWorkerTree();
//  #else
//  return 0;
//  #endif
//}
//
//int peanoclaw::State::getGlobalHeightOfWorkerTreeDuringLastIteration() const {
//  #ifdef Parallel
//  return _stateData.getGlobalHeightOfWorkerTreeDuringLastIteration();
//  #else
//  return 0;
//  #endif
//}


void peanoclaw::State::setReduceReductions(bool reduceReductions) {
  #ifdef Parallel
  _stateData.setReduceReductions(reduceReductions);
  #endif
}

bool peanoclaw::State::shouldReduceReductions() const {
  #ifdef Parallel
  return _stateData.getReduceReductions();
  #else
  return true;
  #endif
}
