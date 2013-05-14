#include "State.h"
#include "Cell.h"
#include "Vertex.h"

#include "peano/grid/Checkpoint.h"

tarch::logging::Log peanoclaw::State::_log("peanoclaw::State");

peanoclaw::State::State():
  Base() { 
  // @todo Insert your code here
}


peanoclaw::State::State(const Base::PersistentState& argument):
  Base(argument) {
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

int peanoclaw::State::getSubPlotNumber() const {
  return _stateData.getSubPlotNumber();
}

void peanoclaw::State::setSubPlotNumber(int subPlotNumber) {
  _stateData.setSubPlotNumber(subPlotNumber);
}

void peanoclaw::State::setUnknownsPerSubcell(int unknownsPerSubcell) {
  _stateData.setUnknownsPerSubcell(unknownsPerSubcell);
}

int peanoclaw::State::getUnknownsPerSubcell() const {
  return _stateData.getUnknownsPerSubcell();
}

void peanoclaw::State::setAuxiliarFieldsPerSubcell(int auxiliarFieldsPerSubcell) {
  _stateData.setAuxiliarFieldsPerSubcell(auxiliarFieldsPerSubcell);
}

int peanoclaw::State::getAuxiliarFieldsPerSubcell() const {
  return _stateData.getAuxiliarFieldsPerSubcell();
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

tarch::la::Vector<DIMENSIONS, double> peanoclaw::State::getInitialMinimalMeshWidth() const {
  return _stateData.getInitialMinimalMeshWidth();
}

void peanoclaw::State::setInitialMinimalMeshWidth(const tarch::la::Vector<DIMENSIONS, double>& h) {
  _stateData.setInitialMinimalMeshWidth(h);
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

//void peanoclaw::State::addToTotalNumberOfCellUpdates(int cellUpdates) {
//  _stateData.setTotalNumberOfCellUpdates( _stateData.getTotalNumberOfCellUpdates() + cellUpdates );
//}

double peanoclaw::State::getTotalNumberOfCellUpdates() const {
  return _stateData.getTotalNumberOfCellUpdates();
}

void peanoclaw::State::resetMinimalTimestep() {
  _stateData.setMinimalTimestep(std::numeric_limits<double>::max());
}

void peanoclaw::State::updateMinimalTimestep(double timestep) {
  if(timestep < _stateData.getMinimalTimestep()) {
    _stateData.setMinimalTimestep(timestep);
  }
}

double peanoclaw::State::getMinimalTimestep() const {
  return _stateData.getMinimalTimestep();
}

void peanoclaw::State::setLevelStatisticsForLastGridIteration (
  const std::vector<peanoclaw::statistics::LevelInformation>& levelStatistics
) {
  _levelStatisticsForLastGridIteration = levelStatistics;

  double totalNumberOfCellUpdates = 0.0;
  for(size_t i = 0; i < _levelStatisticsForLastGridIteration.size(); i++) {
    const peanoclaw::statistics::LevelInformation& levelInformation = _levelStatisticsForLastGridIteration[i];
    totalNumberOfCellUpdates += levelInformation._numberOfCellUpdates;

    if(_totalLevelStatistics.size() <= i) {
      assertionEquals(_totalLevelStatistics.size(), i);
      _totalLevelStatistics.push_back(peanoclaw::statistics::LevelInformation());
    }

    _totalLevelStatistics.at(i)._area = std::max(_totalLevelStatistics.at(i)._area, levelInformation._area);
    _totalLevelStatistics.at(i)._numberOfPatches = std::max(_totalLevelStatistics.at(i)._numberOfPatches, levelInformation._numberOfPatches);
    _totalLevelStatistics.at(i)._numberOfCells = std::max(_totalLevelStatistics.at(i)._numberOfCells, levelInformation._numberOfCells);
    _totalLevelStatistics.at(i)._numberOfCellUpdates += levelInformation._numberOfCellUpdates;
    _totalLevelStatistics.at(i)._patchesBlockedDueToNeighbors += levelInformation._patchesBlockedDueToNeighbors;
    _totalLevelStatistics.at(i)._patchesBlockedDueToGlobalTimestep += levelInformation._patchesBlockedDueToGlobalTimestep;
    _totalLevelStatistics.at(i)._patchesSkippingIteration += levelInformation._patchesSkippingIteration;
    _totalLevelStatistics.at(i)._patchesCoarsening += levelInformation._patchesCoarsening;
  }
}

void peanoclaw::State::plotStatisticsForLastGridIteration() const {
  logInfo("plotStatisticsForLastGridIteration", "Statistics for last grid iteration: Spacetree height: " <<  _levelStatisticsForLastGridIteration.size());

  double totalArea = 0.0;
  double totalNumberOfPatches = 0.0;
  double totalNumberOfCells = 0.0;
  double totalNumberOfCellUpdates = 0.0;
  double totalCreatedPatches = 0.0;
  double totalDestroyedPatches = 0.0;
  double totalBlockedPatchesDueToNeighbors = 0.0;
  double totalBlockedPatchesDueToGlobalTimestep = 0.0;
  double totalSkippingPatches = 0.0;
  double totalCoarseningPatches = 0.0;

  for(size_t i = 0; i < _levelStatisticsForLastGridIteration.size(); i++) {
    const peanoclaw::statistics::LevelInformation& levelInformation = _levelStatisticsForLastGridIteration[i];
    logInfo("plotStatisticsForLastGridIteration", "\tLevel " << i << ": " << levelInformation._numberOfPatches << " patches (area=" << levelInformation._area <<  "), "
      << levelInformation._numberOfCells << " cells, " << levelInformation._numberOfCellUpdates << " cell updates."
      << " Blocking: (" << levelInformation._patchesBlockedDueToNeighbors << ", " << levelInformation._patchesBlockedDueToGlobalTimestep
      << ", " << levelInformation._patchesSkippingIteration << ", " << levelInformation._patchesCoarsening << ")"
    );

    totalArea += levelInformation._area;
    totalNumberOfPatches += levelInformation._numberOfPatches;
    totalNumberOfCells += levelInformation._numberOfCells;
    totalNumberOfCellUpdates += levelInformation._numberOfCellUpdates;
    totalCreatedPatches += levelInformation._createdPatches;
    totalDestroyedPatches += levelInformation._destroyedPatches;
    totalBlockedPatchesDueToNeighbors += levelInformation._patchesBlockedDueToNeighbors;
    totalBlockedPatchesDueToGlobalTimestep += levelInformation._patchesBlockedDueToGlobalTimestep;
    totalSkippingPatches += levelInformation._patchesSkippingIteration;
    totalCoarseningPatches += levelInformation._patchesCoarsening;
  }
  logInfo("plotStatisticsForLastGridIteration", "Sum for grid iteration: "
    << totalNumberOfPatches << " patches (+" << totalCreatedPatches << "/-" << totalDestroyedPatches << ",area=" << totalArea <<  "), "
    << totalNumberOfCells << " cells, " << totalNumberOfCellUpdates << " cell updates."
    << " Blocking: " << totalBlockedPatchesDueToNeighbors << ", " << totalBlockedPatchesDueToGlobalTimestep
    << ", " << totalSkippingPatches << ", " << totalCoarseningPatches
  );
}

void peanoclaw::State::plotTotalStatistics() const {
  logInfo("plotTotalStatistics", "Total statistics: Spacetree height: " <<  _totalLevelStatistics.size());

  double totalArea = 0.0;
  double totalNumberOfPatches = 0.0;
  double totalNumberOfCells = 0.0;
  double totalNumberOfCellUpdates = 0.0;
  double totalBlockedPatchesDueToNeighbors = 0.0;
  double totalBlockedPatchesDueToGlobalTimestep = 0.0;
  double totalSkippingPatches = 0.0;
  double totalCoarseningPatches = 0.0;

  for(size_t i = 0; i < _totalLevelStatistics.size(); i++) {
    const peanoclaw::statistics::LevelInformation& levelInformation = _totalLevelStatistics[i];
    logInfo("plotTotalStatistics", "\tLevel " << i << ": " << levelInformation._numberOfPatches << " patches (area=" << levelInformation._area <<  "), "
        << levelInformation._numberOfCells << " cells, " << levelInformation._numberOfCellUpdates << " cell updates.");

    totalArea += levelInformation._area;
    totalNumberOfPatches += levelInformation._numberOfPatches;
    totalNumberOfCells += levelInformation._numberOfCells;
    totalNumberOfCellUpdates += levelInformation._numberOfCellUpdates;
    totalBlockedPatchesDueToNeighbors += levelInformation._patchesBlockedDueToNeighbors;
    totalBlockedPatchesDueToGlobalTimestep += levelInformation._patchesBlockedDueToGlobalTimestep;
    totalSkippingPatches += levelInformation._patchesSkippingIteration;
    totalCoarseningPatches += levelInformation._patchesCoarsening;
  }
  logInfo("plotTotalStatistics",
    "Sum for simulation: max. " << totalNumberOfPatches << " patches (area=" << totalArea <<  "), max. "
    << totalNumberOfCells << " cells, " << totalNumberOfCellUpdates << " cell updates."
    << " Blocking: " << totalBlockedPatchesDueToNeighbors << ", " << totalBlockedPatchesDueToGlobalTimestep
    << ", " << totalSkippingPatches << ", " << totalCoarseningPatches
    );
}

void peanoclaw::State::setAdditionalLevelsForPredefinedRefinement(int levels) {
  _stateData.setAdditionalLevelsForPredefinedRefinement(levels);
}

int peanoclaw::State::getAdditionalLevelsForPredefinedRefinement() const {
  return _stateData.getAdditionalLevelsForPredefinedRefinement();
}

void peanoclaw::State::setIsInitializing(bool isInitializing) {
  _stateData.setIsInitializing(isInitializing);
}

bool peanoclaw::State::getIsInitializing() const {
  return _stateData.getIsInitializing();
}

void peanoclaw::State::setInitialRefinementTriggered(bool initialRefinementTriggered) {
  _stateData.setInitialRefinmentTriggered(initialRefinementTriggered);
}

bool peanoclaw::State::getInitialRefinementTriggered() const {
  return _stateData.getInitialRefinmentTriggered();
}

void peanoclaw::State::setUseDimensionalSplitting(bool useDimensionalSplitting) {
  _stateData.setUseDimensionalSplitting(useDimensionalSplitting);
}

bool peanoclaw::State::useDimensionalSplitting() const {
  return _stateData.getUseDimensionalSplitting();
}
