#include "peanoclaw/records/CellDescription.h"

#if defined(Parallel)
   peanoclaw::records::CellDescription::PersistentRecords::PersistentRecords() {
      
   }
   
   
   peanoclaw::records::CellDescription::PersistentRecords::PersistentRecords(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const int& level, const bool& isVirtual, const bool& isRemote, const bool& isPaddingSubgrid, const int& numberOfSkippedTransfers, const int& adjacentRank, const int& numberOfSharedAdjacentVertices, const bool& currentStateWasSend, const bool& adjacentRanksChanged, const tarch::la::Vector<DIMENSIONS,int>& upperOverlapByRemoteGhostlayer, const tarch::la::Vector<DIMENSIONS,int>& lowerOverlapByRemoteGhostlayer, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& maximumFineGridTime, const double& minimumFineGridTimestep, const bool& synchronizeFineGrids, const bool& willCoarsen, const double& minimalNeighborTimeConstraint, const int& constrainingNeighborIndex, const double& minimalLeafNeighborTimeConstraint, const double& minimalNeighborTime, const double& maximalNeighborTimestep, const double& estimatedNextTimestepSize, const int& skipGridIterations, const int& ageInGridIterations, const double& demandedMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds, const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds, const int& cellDescriptionIndex, const int& uNewIndex):
   _subdivisionFactor(subdivisionFactor),
   _ghostLayerWidth(ghostLayerWidth),
   _unknownsPerSubcell(unknownsPerSubcell),
   _auxiliarFieldsPerSubcell(auxiliarFieldsPerSubcell),
   _level(level),
   _isVirtual(isVirtual),
   _isRemote(isRemote),
   _isPaddingSubgrid(isPaddingSubgrid),
   _numberOfSkippedTransfers(numberOfSkippedTransfers),
   _adjacentRank(adjacentRank),
   _numberOfSharedAdjacentVertices(numberOfSharedAdjacentVertices),
   _currentStateWasSend(currentStateWasSend),
   _adjacentRanksChanged(adjacentRanksChanged),
   _upperOverlapByRemoteGhostlayer(upperOverlapByRemoteGhostlayer),
   _lowerOverlapByRemoteGhostlayer(lowerOverlapByRemoteGhostlayer),
   _position(position),
   _size(size),
   _time(time),
   _timestepSize(timestepSize),
   _maximumFineGridTime(maximumFineGridTime),
   _minimumFineGridTimestep(minimumFineGridTimestep),
   _synchronizeFineGrids(synchronizeFineGrids),
   _willCoarsen(willCoarsen),
   _minimalNeighborTimeConstraint(minimalNeighborTimeConstraint),
   _constrainingNeighborIndex(constrainingNeighborIndex),
   _minimalLeafNeighborTimeConstraint(minimalLeafNeighborTimeConstraint),
   _minimalNeighborTime(minimalNeighborTime),
   _maximalNeighborTimestep(maximalNeighborTimestep),
   _estimatedNextTimestepSize(estimatedNextTimestepSize),
   _skipGridIterations(skipGridIterations),
   _ageInGridIterations(ageInGridIterations),
   _demandedMeshWidth(demandedMeshWidth),
   _restrictionLowerBounds(restrictionLowerBounds),
   _restrictionUpperBounds(restrictionUpperBounds),
   _cellDescriptionIndex(cellDescriptionIndex),
   _uNewIndex(uNewIndex) {
      
   }
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::CellDescription::PersistentRecords::getSubdivisionFactor() const  {
      return _subdivisionFactor;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setSubdivisionFactor(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor)  {
      _subdivisionFactor = (subdivisionFactor);
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getGhostLayerWidth() const  {
      return _ghostLayerWidth;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setGhostLayerWidth(const int& ghostLayerWidth)  {
      _ghostLayerWidth = ghostLayerWidth;
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getUnknownsPerSubcell() const  {
      return _unknownsPerSubcell;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setUnknownsPerSubcell(const int& unknownsPerSubcell)  {
      _unknownsPerSubcell = unknownsPerSubcell;
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getAuxiliarFieldsPerSubcell() const  {
      return _auxiliarFieldsPerSubcell;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setAuxiliarFieldsPerSubcell(const int& auxiliarFieldsPerSubcell)  {
      _auxiliarFieldsPerSubcell = auxiliarFieldsPerSubcell;
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getLevel() const  {
      return _level;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setLevel(const int& level)  {
      _level = level;
   }
   
   
   
    bool peanoclaw::records::CellDescription::PersistentRecords::getIsVirtual() const  {
      return _isVirtual;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setIsVirtual(const bool& isVirtual)  {
      _isVirtual = isVirtual;
   }
   
   
   
    bool peanoclaw::records::CellDescription::PersistentRecords::getIsRemote() const  {
      return _isRemote;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setIsRemote(const bool& isRemote)  {
      _isRemote = isRemote;
   }
   
   
   
    bool peanoclaw::records::CellDescription::PersistentRecords::getIsPaddingSubgrid() const  {
      return _isPaddingSubgrid;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setIsPaddingSubgrid(const bool& isPaddingSubgrid)  {
      _isPaddingSubgrid = isPaddingSubgrid;
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getNumberOfSkippedTransfers() const  {
      return _numberOfSkippedTransfers;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setNumberOfSkippedTransfers(const int& numberOfSkippedTransfers)  {
      _numberOfSkippedTransfers = numberOfSkippedTransfers;
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getAdjacentRank() const  {
      return _adjacentRank;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setAdjacentRank(const int& adjacentRank)  {
      _adjacentRank = adjacentRank;
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getNumberOfSharedAdjacentVertices() const  {
      return _numberOfSharedAdjacentVertices;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setNumberOfSharedAdjacentVertices(const int& numberOfSharedAdjacentVertices)  {
      _numberOfSharedAdjacentVertices = numberOfSharedAdjacentVertices;
   }
   
   
   
    bool peanoclaw::records::CellDescription::PersistentRecords::getCurrentStateWasSend() const  {
      return _currentStateWasSend;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setCurrentStateWasSend(const bool& currentStateWasSend)  {
      _currentStateWasSend = currentStateWasSend;
   }
   
   
   
    bool peanoclaw::records::CellDescription::PersistentRecords::getAdjacentRanksChanged() const  {
      return _adjacentRanksChanged;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setAdjacentRanksChanged(const bool& adjacentRanksChanged)  {
      _adjacentRanksChanged = adjacentRanksChanged;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::CellDescription::PersistentRecords::getUpperOverlapByRemoteGhostlayer() const  {
      return _upperOverlapByRemoteGhostlayer;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setUpperOverlapByRemoteGhostlayer(const tarch::la::Vector<DIMENSIONS,int>& upperOverlapByRemoteGhostlayer)  {
      _upperOverlapByRemoteGhostlayer = (upperOverlapByRemoteGhostlayer);
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::CellDescription::PersistentRecords::getLowerOverlapByRemoteGhostlayer() const  {
      return _lowerOverlapByRemoteGhostlayer;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setLowerOverlapByRemoteGhostlayer(const tarch::la::Vector<DIMENSIONS,int>& lowerOverlapByRemoteGhostlayer)  {
      _lowerOverlapByRemoteGhostlayer = (lowerOverlapByRemoteGhostlayer);
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescription::PersistentRecords::getPosition() const  {
      return _position;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setPosition(const tarch::la::Vector<DIMENSIONS,double>& position)  {
      _position = (position);
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescription::PersistentRecords::getSize() const  {
      return _size;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setSize(const tarch::la::Vector<DIMENSIONS,double>& size)  {
      _size = (size);
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getTime() const  {
      return _time;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setTime(const double& time)  {
      _time = time;
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getTimestepSize() const  {
      return _timestepSize;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setTimestepSize(const double& timestepSize)  {
      _timestepSize = timestepSize;
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getMaximumFineGridTime() const  {
      return _maximumFineGridTime;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setMaximumFineGridTime(const double& maximumFineGridTime)  {
      _maximumFineGridTime = maximumFineGridTime;
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getMinimumFineGridTimestep() const  {
      return _minimumFineGridTimestep;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setMinimumFineGridTimestep(const double& minimumFineGridTimestep)  {
      _minimumFineGridTimestep = minimumFineGridTimestep;
   }
   
   
   
    bool peanoclaw::records::CellDescription::PersistentRecords::getSynchronizeFineGrids() const  {
      return _synchronizeFineGrids;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setSynchronizeFineGrids(const bool& synchronizeFineGrids)  {
      _synchronizeFineGrids = synchronizeFineGrids;
   }
   
   
   
    bool peanoclaw::records::CellDescription::PersistentRecords::getWillCoarsen() const  {
      return _willCoarsen;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setWillCoarsen(const bool& willCoarsen)  {
      _willCoarsen = willCoarsen;
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getMinimalNeighborTimeConstraint() const  {
      return _minimalNeighborTimeConstraint;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setMinimalNeighborTimeConstraint(const double& minimalNeighborTimeConstraint)  {
      _minimalNeighborTimeConstraint = minimalNeighborTimeConstraint;
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getConstrainingNeighborIndex() const  {
      return _constrainingNeighborIndex;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setConstrainingNeighborIndex(const int& constrainingNeighborIndex)  {
      _constrainingNeighborIndex = constrainingNeighborIndex;
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getMinimalLeafNeighborTimeConstraint() const  {
      return _minimalLeafNeighborTimeConstraint;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setMinimalLeafNeighborTimeConstraint(const double& minimalLeafNeighborTimeConstraint)  {
      _minimalLeafNeighborTimeConstraint = minimalLeafNeighborTimeConstraint;
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getMinimalNeighborTime() const  {
      return _minimalNeighborTime;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setMinimalNeighborTime(const double& minimalNeighborTime)  {
      _minimalNeighborTime = minimalNeighborTime;
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getMaximalNeighborTimestep() const  {
      return _maximalNeighborTimestep;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setMaximalNeighborTimestep(const double& maximalNeighborTimestep)  {
      _maximalNeighborTimestep = maximalNeighborTimestep;
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getEstimatedNextTimestepSize() const  {
      return _estimatedNextTimestepSize;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setEstimatedNextTimestepSize(const double& estimatedNextTimestepSize)  {
      _estimatedNextTimestepSize = estimatedNextTimestepSize;
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getSkipGridIterations() const  {
      return _skipGridIterations;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setSkipGridIterations(const int& skipGridIterations)  {
      _skipGridIterations = skipGridIterations;
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getAgeInGridIterations() const  {
      return _ageInGridIterations;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setAgeInGridIterations(const int& ageInGridIterations)  {
      _ageInGridIterations = ageInGridIterations;
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getDemandedMeshWidth() const  {
      return _demandedMeshWidth;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setDemandedMeshWidth(const double& demandedMeshWidth)  {
      _demandedMeshWidth = demandedMeshWidth;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescription::PersistentRecords::getRestrictionLowerBounds() const  {
      return _restrictionLowerBounds;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setRestrictionLowerBounds(const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds)  {
      _restrictionLowerBounds = (restrictionLowerBounds);
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescription::PersistentRecords::getRestrictionUpperBounds() const  {
      return _restrictionUpperBounds;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setRestrictionUpperBounds(const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds)  {
      _restrictionUpperBounds = (restrictionUpperBounds);
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getCellDescriptionIndex() const  {
      return _cellDescriptionIndex;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setCellDescriptionIndex(const int& cellDescriptionIndex)  {
      _cellDescriptionIndex = cellDescriptionIndex;
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getUNewIndex() const  {
      return _uNewIndex;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setUNewIndex(const int& uNewIndex)  {
      _uNewIndex = uNewIndex;
   }
   
   
   peanoclaw::records::CellDescription::CellDescription() {
      
   }
   
   
   peanoclaw::records::CellDescription::CellDescription(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._subdivisionFactor, persistentRecords._ghostLayerWidth, persistentRecords._unknownsPerSubcell, persistentRecords._auxiliarFieldsPerSubcell, persistentRecords._level, persistentRecords._isVirtual, persistentRecords._isRemote, persistentRecords._isPaddingSubgrid, persistentRecords._numberOfSkippedTransfers, persistentRecords._adjacentRank, persistentRecords._numberOfSharedAdjacentVertices, persistentRecords._currentStateWasSend, persistentRecords._adjacentRanksChanged, persistentRecords._upperOverlapByRemoteGhostlayer, persistentRecords._lowerOverlapByRemoteGhostlayer, persistentRecords._position, persistentRecords._size, persistentRecords._time, persistentRecords._timestepSize, persistentRecords._maximumFineGridTime, persistentRecords._minimumFineGridTimestep, persistentRecords._synchronizeFineGrids, persistentRecords._willCoarsen, persistentRecords._minimalNeighborTimeConstraint, persistentRecords._constrainingNeighborIndex, persistentRecords._minimalLeafNeighborTimeConstraint, persistentRecords._minimalNeighborTime, persistentRecords._maximalNeighborTimestep, persistentRecords._estimatedNextTimestepSize, persistentRecords._skipGridIterations, persistentRecords._ageInGridIterations, persistentRecords._demandedMeshWidth, persistentRecords._restrictionLowerBounds, persistentRecords._restrictionUpperBounds, persistentRecords._cellDescriptionIndex, persistentRecords._uNewIndex) {
      
   }
   
   
   peanoclaw::records::CellDescription::CellDescription(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const int& level, const bool& isVirtual, const bool& isRemote, const bool& isPaddingSubgrid, const int& numberOfSkippedTransfers, const int& adjacentRank, const int& numberOfSharedAdjacentVertices, const bool& currentStateWasSend, const bool& adjacentRanksChanged, const tarch::la::Vector<DIMENSIONS,int>& upperOverlapByRemoteGhostlayer, const tarch::la::Vector<DIMENSIONS,int>& lowerOverlapByRemoteGhostlayer, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& maximumFineGridTime, const double& minimumFineGridTimestep, const bool& synchronizeFineGrids, const bool& willCoarsen, const double& minimalNeighborTimeConstraint, const int& constrainingNeighborIndex, const double& minimalLeafNeighborTimeConstraint, const double& minimalNeighborTime, const double& maximalNeighborTimestep, const double& estimatedNextTimestepSize, const int& skipGridIterations, const int& ageInGridIterations, const double& demandedMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds, const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds, const int& cellDescriptionIndex, const int& uNewIndex):
   _persistentRecords(subdivisionFactor, ghostLayerWidth, unknownsPerSubcell, auxiliarFieldsPerSubcell, level, isVirtual, isRemote, isPaddingSubgrid, numberOfSkippedTransfers, adjacentRank, numberOfSharedAdjacentVertices, currentStateWasSend, adjacentRanksChanged, upperOverlapByRemoteGhostlayer, lowerOverlapByRemoteGhostlayer, position, size, time, timestepSize, maximumFineGridTime, minimumFineGridTimestep, synchronizeFineGrids, willCoarsen, minimalNeighborTimeConstraint, constrainingNeighborIndex, minimalLeafNeighborTimeConstraint, minimalNeighborTime, maximalNeighborTimestep, estimatedNextTimestepSize, skipGridIterations, ageInGridIterations, demandedMeshWidth, restrictionLowerBounds, restrictionUpperBounds, cellDescriptionIndex, uNewIndex) {
      
   }
   
   
   peanoclaw::records::CellDescription::~CellDescription() { }
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::CellDescription::getSubdivisionFactor() const  {
      return _persistentRecords._subdivisionFactor;
   }
   
   
   
    void peanoclaw::records::CellDescription::setSubdivisionFactor(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor)  {
      _persistentRecords._subdivisionFactor = (subdivisionFactor);
   }
   
   
   
    int peanoclaw::records::CellDescription::getSubdivisionFactor(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._subdivisionFactor[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescription::setSubdivisionFactor(int elementIndex, const int& subdivisionFactor)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._subdivisionFactor[elementIndex]= subdivisionFactor;
      
   }
   
   
   
    int peanoclaw::records::CellDescription::getGhostLayerWidth() const  {
      return _persistentRecords._ghostLayerWidth;
   }
   
   
   
    void peanoclaw::records::CellDescription::setGhostLayerWidth(const int& ghostLayerWidth)  {
      _persistentRecords._ghostLayerWidth = ghostLayerWidth;
   }
   
   
   
    int peanoclaw::records::CellDescription::getUnknownsPerSubcell() const  {
      return _persistentRecords._unknownsPerSubcell;
   }
   
   
   
    void peanoclaw::records::CellDescription::setUnknownsPerSubcell(const int& unknownsPerSubcell)  {
      _persistentRecords._unknownsPerSubcell = unknownsPerSubcell;
   }
   
   
   
    int peanoclaw::records::CellDescription::getAuxiliarFieldsPerSubcell() const  {
      return _persistentRecords._auxiliarFieldsPerSubcell;
   }
   
   
   
    void peanoclaw::records::CellDescription::setAuxiliarFieldsPerSubcell(const int& auxiliarFieldsPerSubcell)  {
      _persistentRecords._auxiliarFieldsPerSubcell = auxiliarFieldsPerSubcell;
   }
   
   
   
    int peanoclaw::records::CellDescription::getLevel() const  {
      return _persistentRecords._level;
   }
   
   
   
    void peanoclaw::records::CellDescription::setLevel(const int& level)  {
      _persistentRecords._level = level;
   }
   
   
   
    bool peanoclaw::records::CellDescription::getIsVirtual() const  {
      return _persistentRecords._isVirtual;
   }
   
   
   
    void peanoclaw::records::CellDescription::setIsVirtual(const bool& isVirtual)  {
      _persistentRecords._isVirtual = isVirtual;
   }
   
   
   
    bool peanoclaw::records::CellDescription::getIsRemote() const  {
      return _persistentRecords._isRemote;
   }
   
   
   
    void peanoclaw::records::CellDescription::setIsRemote(const bool& isRemote)  {
      _persistentRecords._isRemote = isRemote;
   }
   
   
   
    bool peanoclaw::records::CellDescription::getIsPaddingSubgrid() const  {
      return _persistentRecords._isPaddingSubgrid;
   }
   
   
   
    void peanoclaw::records::CellDescription::setIsPaddingSubgrid(const bool& isPaddingSubgrid)  {
      _persistentRecords._isPaddingSubgrid = isPaddingSubgrid;
   }
   
   
   
    int peanoclaw::records::CellDescription::getNumberOfSkippedTransfers() const  {
      return _persistentRecords._numberOfSkippedTransfers;
   }
   
   
   
    void peanoclaw::records::CellDescription::setNumberOfSkippedTransfers(const int& numberOfSkippedTransfers)  {
      _persistentRecords._numberOfSkippedTransfers = numberOfSkippedTransfers;
   }
   
   
   
    int peanoclaw::records::CellDescription::getAdjacentRank() const  {
      return _persistentRecords._adjacentRank;
   }
   
   
   
    void peanoclaw::records::CellDescription::setAdjacentRank(const int& adjacentRank)  {
      _persistentRecords._adjacentRank = adjacentRank;
   }
   
   
   
    int peanoclaw::records::CellDescription::getNumberOfSharedAdjacentVertices() const  {
      return _persistentRecords._numberOfSharedAdjacentVertices;
   }
   
   
   
    void peanoclaw::records::CellDescription::setNumberOfSharedAdjacentVertices(const int& numberOfSharedAdjacentVertices)  {
      _persistentRecords._numberOfSharedAdjacentVertices = numberOfSharedAdjacentVertices;
   }
   
   
   
    bool peanoclaw::records::CellDescription::getCurrentStateWasSend() const  {
      return _persistentRecords._currentStateWasSend;
   }
   
   
   
    void peanoclaw::records::CellDescription::setCurrentStateWasSend(const bool& currentStateWasSend)  {
      _persistentRecords._currentStateWasSend = currentStateWasSend;
   }
   
   
   
    bool peanoclaw::records::CellDescription::getAdjacentRanksChanged() const  {
      return _persistentRecords._adjacentRanksChanged;
   }
   
   
   
    void peanoclaw::records::CellDescription::setAdjacentRanksChanged(const bool& adjacentRanksChanged)  {
      _persistentRecords._adjacentRanksChanged = adjacentRanksChanged;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::CellDescription::getUpperOverlapByRemoteGhostlayer() const  {
      return _persistentRecords._upperOverlapByRemoteGhostlayer;
   }
   
   
   
    void peanoclaw::records::CellDescription::setUpperOverlapByRemoteGhostlayer(const tarch::la::Vector<DIMENSIONS,int>& upperOverlapByRemoteGhostlayer)  {
      _persistentRecords._upperOverlapByRemoteGhostlayer = (upperOverlapByRemoteGhostlayer);
   }
   
   
   
    int peanoclaw::records::CellDescription::getUpperOverlapByRemoteGhostlayer(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._upperOverlapByRemoteGhostlayer[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescription::setUpperOverlapByRemoteGhostlayer(int elementIndex, const int& upperOverlapByRemoteGhostlayer)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._upperOverlapByRemoteGhostlayer[elementIndex]= upperOverlapByRemoteGhostlayer;
      
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::CellDescription::getLowerOverlapByRemoteGhostlayer() const  {
      return _persistentRecords._lowerOverlapByRemoteGhostlayer;
   }
   
   
   
    void peanoclaw::records::CellDescription::setLowerOverlapByRemoteGhostlayer(const tarch::la::Vector<DIMENSIONS,int>& lowerOverlapByRemoteGhostlayer)  {
      _persistentRecords._lowerOverlapByRemoteGhostlayer = (lowerOverlapByRemoteGhostlayer);
   }
   
   
   
    int peanoclaw::records::CellDescription::getLowerOverlapByRemoteGhostlayer(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._lowerOverlapByRemoteGhostlayer[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescription::setLowerOverlapByRemoteGhostlayer(int elementIndex, const int& lowerOverlapByRemoteGhostlayer)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._lowerOverlapByRemoteGhostlayer[elementIndex]= lowerOverlapByRemoteGhostlayer;
      
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescription::getPosition() const  {
      return _persistentRecords._position;
   }
   
   
   
    void peanoclaw::records::CellDescription::setPosition(const tarch::la::Vector<DIMENSIONS,double>& position)  {
      _persistentRecords._position = (position);
   }
   
   
   
    double peanoclaw::records::CellDescription::getPosition(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._position[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescription::setPosition(int elementIndex, const double& position)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._position[elementIndex]= position;
      
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescription::getSize() const  {
      return _persistentRecords._size;
   }
   
   
   
    void peanoclaw::records::CellDescription::setSize(const tarch::la::Vector<DIMENSIONS,double>& size)  {
      _persistentRecords._size = (size);
   }
   
   
   
    double peanoclaw::records::CellDescription::getSize(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._size[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescription::setSize(int elementIndex, const double& size)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._size[elementIndex]= size;
      
   }
   
   
   
    double peanoclaw::records::CellDescription::getTime() const  {
      return _persistentRecords._time;
   }
   
   
   
    void peanoclaw::records::CellDescription::setTime(const double& time)  {
      _persistentRecords._time = time;
   }
   
   
   
    double peanoclaw::records::CellDescription::getTimestepSize() const  {
      return _persistentRecords._timestepSize;
   }
   
   
   
    void peanoclaw::records::CellDescription::setTimestepSize(const double& timestepSize)  {
      _persistentRecords._timestepSize = timestepSize;
   }
   
   
   
    double peanoclaw::records::CellDescription::getMaximumFineGridTime() const  {
      return _persistentRecords._maximumFineGridTime;
   }
   
   
   
    void peanoclaw::records::CellDescription::setMaximumFineGridTime(const double& maximumFineGridTime)  {
      _persistentRecords._maximumFineGridTime = maximumFineGridTime;
   }
   
   
   
    double peanoclaw::records::CellDescription::getMinimumFineGridTimestep() const  {
      return _persistentRecords._minimumFineGridTimestep;
   }
   
   
   
    void peanoclaw::records::CellDescription::setMinimumFineGridTimestep(const double& minimumFineGridTimestep)  {
      _persistentRecords._minimumFineGridTimestep = minimumFineGridTimestep;
   }
   
   
   
    bool peanoclaw::records::CellDescription::getSynchronizeFineGrids() const  {
      return _persistentRecords._synchronizeFineGrids;
   }
   
   
   
    void peanoclaw::records::CellDescription::setSynchronizeFineGrids(const bool& synchronizeFineGrids)  {
      _persistentRecords._synchronizeFineGrids = synchronizeFineGrids;
   }
   
   
   
    bool peanoclaw::records::CellDescription::getWillCoarsen() const  {
      return _persistentRecords._willCoarsen;
   }
   
   
   
    void peanoclaw::records::CellDescription::setWillCoarsen(const bool& willCoarsen)  {
      _persistentRecords._willCoarsen = willCoarsen;
   }
   
   
   
    double peanoclaw::records::CellDescription::getMinimalNeighborTimeConstraint() const  {
      return _persistentRecords._minimalNeighborTimeConstraint;
   }
   
   
   
    void peanoclaw::records::CellDescription::setMinimalNeighborTimeConstraint(const double& minimalNeighborTimeConstraint)  {
      _persistentRecords._minimalNeighborTimeConstraint = minimalNeighborTimeConstraint;
   }
   
   
   
    int peanoclaw::records::CellDescription::getConstrainingNeighborIndex() const  {
      return _persistentRecords._constrainingNeighborIndex;
   }
   
   
   
    void peanoclaw::records::CellDescription::setConstrainingNeighborIndex(const int& constrainingNeighborIndex)  {
      _persistentRecords._constrainingNeighborIndex = constrainingNeighborIndex;
   }
   
   
   
    double peanoclaw::records::CellDescription::getMinimalLeafNeighborTimeConstraint() const  {
      return _persistentRecords._minimalLeafNeighborTimeConstraint;
   }
   
   
   
    void peanoclaw::records::CellDescription::setMinimalLeafNeighborTimeConstraint(const double& minimalLeafNeighborTimeConstraint)  {
      _persistentRecords._minimalLeafNeighborTimeConstraint = minimalLeafNeighborTimeConstraint;
   }
   
   
   
    double peanoclaw::records::CellDescription::getMinimalNeighborTime() const  {
      return _persistentRecords._minimalNeighborTime;
   }
   
   
   
    void peanoclaw::records::CellDescription::setMinimalNeighborTime(const double& minimalNeighborTime)  {
      _persistentRecords._minimalNeighborTime = minimalNeighborTime;
   }
   
   
   
    double peanoclaw::records::CellDescription::getMaximalNeighborTimestep() const  {
      return _persistentRecords._maximalNeighborTimestep;
   }
   
   
   
    void peanoclaw::records::CellDescription::setMaximalNeighborTimestep(const double& maximalNeighborTimestep)  {
      _persistentRecords._maximalNeighborTimestep = maximalNeighborTimestep;
   }
   
   
   
    double peanoclaw::records::CellDescription::getEstimatedNextTimestepSize() const  {
      return _persistentRecords._estimatedNextTimestepSize;
   }
   
   
   
    void peanoclaw::records::CellDescription::setEstimatedNextTimestepSize(const double& estimatedNextTimestepSize)  {
      _persistentRecords._estimatedNextTimestepSize = estimatedNextTimestepSize;
   }
   
   
   
    int peanoclaw::records::CellDescription::getSkipGridIterations() const  {
      return _persistentRecords._skipGridIterations;
   }
   
   
   
    void peanoclaw::records::CellDescription::setSkipGridIterations(const int& skipGridIterations)  {
      _persistentRecords._skipGridIterations = skipGridIterations;
   }
   
   
   
    int peanoclaw::records::CellDescription::getAgeInGridIterations() const  {
      return _persistentRecords._ageInGridIterations;
   }
   
   
   
    void peanoclaw::records::CellDescription::setAgeInGridIterations(const int& ageInGridIterations)  {
      _persistentRecords._ageInGridIterations = ageInGridIterations;
   }
   
   
   
    double peanoclaw::records::CellDescription::getDemandedMeshWidth() const  {
      return _persistentRecords._demandedMeshWidth;
   }
   
   
   
    void peanoclaw::records::CellDescription::setDemandedMeshWidth(const double& demandedMeshWidth)  {
      _persistentRecords._demandedMeshWidth = demandedMeshWidth;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescription::getRestrictionLowerBounds() const  {
      return _persistentRecords._restrictionLowerBounds;
   }
   
   
   
    void peanoclaw::records::CellDescription::setRestrictionLowerBounds(const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds)  {
      _persistentRecords._restrictionLowerBounds = (restrictionLowerBounds);
   }
   
   
   
    double peanoclaw::records::CellDescription::getRestrictionLowerBounds(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._restrictionLowerBounds[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescription::setRestrictionLowerBounds(int elementIndex, const double& restrictionLowerBounds)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._restrictionLowerBounds[elementIndex]= restrictionLowerBounds;
      
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescription::getRestrictionUpperBounds() const  {
      return _persistentRecords._restrictionUpperBounds;
   }
   
   
   
    void peanoclaw::records::CellDescription::setRestrictionUpperBounds(const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds)  {
      _persistentRecords._restrictionUpperBounds = (restrictionUpperBounds);
   }
   
   
   
    double peanoclaw::records::CellDescription::getRestrictionUpperBounds(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._restrictionUpperBounds[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescription::setRestrictionUpperBounds(int elementIndex, const double& restrictionUpperBounds)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._restrictionUpperBounds[elementIndex]= restrictionUpperBounds;
      
   }
   
   
   
    int peanoclaw::records::CellDescription::getCellDescriptionIndex() const  {
      return _persistentRecords._cellDescriptionIndex;
   }
   
   
   
    void peanoclaw::records::CellDescription::setCellDescriptionIndex(const int& cellDescriptionIndex)  {
      _persistentRecords._cellDescriptionIndex = cellDescriptionIndex;
   }
   
   
   
    int peanoclaw::records::CellDescription::getUNewIndex() const  {
      return _persistentRecords._uNewIndex;
   }
   
   
   
    void peanoclaw::records::CellDescription::setUNewIndex(const int& uNewIndex)  {
      _persistentRecords._uNewIndex = uNewIndex;
   }
   
   
   
   
   std::string peanoclaw::records::CellDescription::toString() const {
      std::ostringstream stringstr;
      toString(stringstr);
      return stringstr.str();
   }
   
   void peanoclaw::records::CellDescription::toString (std::ostream& out) const {
      out << "("; 
      out << "subdivisionFactor:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getSubdivisionFactor(i) << ",";
   }
   out << getSubdivisionFactor(DIMENSIONS-1) << "]";
      out << ",";
      out << "ghostLayerWidth:" << getGhostLayerWidth();
      out << ",";
      out << "unknownsPerSubcell:" << getUnknownsPerSubcell();
      out << ",";
      out << "auxiliarFieldsPerSubcell:" << getAuxiliarFieldsPerSubcell();
      out << ",";
      out << "level:" << getLevel();
      out << ",";
      out << "isVirtual:" << getIsVirtual();
      out << ",";
      out << "isRemote:" << getIsRemote();
      out << ",";
      out << "isPaddingSubgrid:" << getIsPaddingSubgrid();
      out << ",";
      out << "numberOfSkippedTransfers:" << getNumberOfSkippedTransfers();
      out << ",";
      out << "adjacentRank:" << getAdjacentRank();
      out << ",";
      out << "numberOfSharedAdjacentVertices:" << getNumberOfSharedAdjacentVertices();
      out << ",";
      out << "currentStateWasSend:" << getCurrentStateWasSend();
      out << ",";
      out << "adjacentRanksChanged:" << getAdjacentRanksChanged();
      out << ",";
      out << "upperOverlapByRemoteGhostlayer:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getUpperOverlapByRemoteGhostlayer(i) << ",";
   }
   out << getUpperOverlapByRemoteGhostlayer(DIMENSIONS-1) << "]";
      out << ",";
      out << "lowerOverlapByRemoteGhostlayer:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getLowerOverlapByRemoteGhostlayer(i) << ",";
   }
   out << getLowerOverlapByRemoteGhostlayer(DIMENSIONS-1) << "]";
      out << ",";
      out << "position:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getPosition(i) << ",";
   }
   out << getPosition(DIMENSIONS-1) << "]";
      out << ",";
      out << "size:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getSize(i) << ",";
   }
   out << getSize(DIMENSIONS-1) << "]";
      out << ",";
      out << "time:" << getTime();
      out << ",";
      out << "timestepSize:" << getTimestepSize();
      out << ",";
      out << "maximumFineGridTime:" << getMaximumFineGridTime();
      out << ",";
      out << "minimumFineGridTimestep:" << getMinimumFineGridTimestep();
      out << ",";
      out << "synchronizeFineGrids:" << getSynchronizeFineGrids();
      out << ",";
      out << "willCoarsen:" << getWillCoarsen();
      out << ",";
      out << "minimalNeighborTimeConstraint:" << getMinimalNeighborTimeConstraint();
      out << ",";
      out << "constrainingNeighborIndex:" << getConstrainingNeighborIndex();
      out << ",";
      out << "minimalLeafNeighborTimeConstraint:" << getMinimalLeafNeighborTimeConstraint();
      out << ",";
      out << "minimalNeighborTime:" << getMinimalNeighborTime();
      out << ",";
      out << "maximalNeighborTimestep:" << getMaximalNeighborTimestep();
      out << ",";
      out << "estimatedNextTimestepSize:" << getEstimatedNextTimestepSize();
      out << ",";
      out << "skipGridIterations:" << getSkipGridIterations();
      out << ",";
      out << "ageInGridIterations:" << getAgeInGridIterations();
      out << ",";
      out << "demandedMeshWidth:" << getDemandedMeshWidth();
      out << ",";
      out << "restrictionLowerBounds:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getRestrictionLowerBounds(i) << ",";
   }
   out << getRestrictionLowerBounds(DIMENSIONS-1) << "]";
      out << ",";
      out << "restrictionUpperBounds:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getRestrictionUpperBounds(i) << ",";
   }
   out << getRestrictionUpperBounds(DIMENSIONS-1) << "]";
      out << ",";
      out << "cellDescriptionIndex:" << getCellDescriptionIndex();
      out << ",";
      out << "uNewIndex:" << getUNewIndex();
      out <<  ")";
   }
   
   
   peanoclaw::records::CellDescription::PersistentRecords peanoclaw::records::CellDescription::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::CellDescriptionPacked peanoclaw::records::CellDescription::convert() const{
      return CellDescriptionPacked(
         getSubdivisionFactor(),
         getGhostLayerWidth(),
         getUnknownsPerSubcell(),
         getAuxiliarFieldsPerSubcell(),
         getLevel(),
         getIsVirtual(),
         getIsRemote(),
         getIsPaddingSubgrid(),
         getNumberOfSkippedTransfers(),
         getAdjacentRank(),
         getNumberOfSharedAdjacentVertices(),
         getCurrentStateWasSend(),
         getAdjacentRanksChanged(),
         getUpperOverlapByRemoteGhostlayer(),
         getLowerOverlapByRemoteGhostlayer(),
         getPosition(),
         getSize(),
         getTime(),
         getTimestepSize(),
         getMaximumFineGridTime(),
         getMinimumFineGridTimestep(),
         getSynchronizeFineGrids(),
         getWillCoarsen(),
         getMinimalNeighborTimeConstraint(),
         getConstrainingNeighborIndex(),
         getMinimalLeafNeighborTimeConstraint(),
         getMinimalNeighborTime(),
         getMaximalNeighborTimestep(),
         getEstimatedNextTimestepSize(),
         getSkipGridIterations(),
         getAgeInGridIterations(),
         getDemandedMeshWidth(),
         getRestrictionLowerBounds(),
         getRestrictionUpperBounds(),
         getCellDescriptionIndex(),
         getUNewIndex()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::CellDescription::_log( "peanoclaw::records::CellDescription" );
      
      MPI_Datatype peanoclaw::records::CellDescription::Datatype = 0;
      MPI_Datatype peanoclaw::records::CellDescription::FullDatatype = 0;
      
      
      void peanoclaw::records::CellDescription::initDatatype() {
         {
            CellDescription dummyCellDescription[2];
            
            const int Attributes = 32;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //subdivisionFactor
               MPI_INT,		 //ghostLayerWidth
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_INT,		 //level
               MPI_CHAR,		 //isVirtual
               MPI_CHAR,		 //isRemote
               MPI_CHAR,		 //isPaddingSubgrid
               MPI_INT,		 //numberOfSkippedTransfers
               MPI_INT,		 //upperOverlapByRemoteGhostlayer
               MPI_INT,		 //lowerOverlapByRemoteGhostlayer
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
               MPI_DOUBLE,		 //maximumFineGridTime
               MPI_DOUBLE,		 //minimumFineGridTimestep
               MPI_CHAR,		 //synchronizeFineGrids
               MPI_CHAR,		 //willCoarsen
               MPI_DOUBLE,		 //minimalNeighborTimeConstraint
               MPI_DOUBLE,		 //minimalLeafNeighborTimeConstraint
               MPI_DOUBLE,		 //minimalNeighborTime
               MPI_DOUBLE,		 //maximalNeighborTimestep
               MPI_DOUBLE,		 //estimatedNextTimestepSize
               MPI_INT,		 //skipGridIterations
               MPI_INT,		 //ageInGridIterations
               MPI_DOUBLE,		 //demandedMeshWidth
               MPI_DOUBLE,		 //restrictionLowerBounds
               MPI_DOUBLE,		 //restrictionUpperBounds
               MPI_INT,		 //cellDescriptionIndex
               MPI_INT,		 //uNewIndex
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               DIMENSIONS,		 //subdivisionFactor
               1,		 //ghostLayerWidth
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               1,		 //level
               1,		 //isVirtual
               1,		 //isRemote
               1,		 //isPaddingSubgrid
               1,		 //numberOfSkippedTransfers
               DIMENSIONS,		 //upperOverlapByRemoteGhostlayer
               DIMENSIONS,		 //lowerOverlapByRemoteGhostlayer
               DIMENSIONS,		 //position
               DIMENSIONS,		 //size
               1,		 //time
               1,		 //timestepSize
               1,		 //maximumFineGridTime
               1,		 //minimumFineGridTimestep
               1,		 //synchronizeFineGrids
               1,		 //willCoarsen
               1,		 //minimalNeighborTimeConstraint
               1,		 //minimalLeafNeighborTimeConstraint
               1,		 //minimalNeighborTime
               1,		 //maximalNeighborTimestep
               1,		 //estimatedNextTimestepSize
               1,		 //skipGridIterations
               1,		 //ageInGridIterations
               1,		 //demandedMeshWidth
               DIMENSIONS,		 //restrictionLowerBounds
               DIMENSIONS,		 //restrictionUpperBounds
               1,		 //cellDescriptionIndex
               1,		 //uNewIndex
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._ghostLayerWidth))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._unknownsPerSubcell))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._level))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._isVirtual))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._isRemote))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._isPaddingSubgrid))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._numberOfSkippedTransfers))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._upperOverlapByRemoteGhostlayer[0]))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._lowerOverlapByRemoteGhostlayer[0]))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._position[0]))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._size[0]))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._time))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._timestepSize))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._maximumFineGridTime))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimumFineGridTimestep))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._synchronizeFineGrids))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._willCoarsen))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalNeighborTimeConstraint))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalLeafNeighborTimeConstraint))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalNeighborTime))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._maximalNeighborTimestep))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._estimatedNextTimestepSize))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._skipGridIterations))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._ageInGridIterations))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._demandedMeshWidth))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._restrictionLowerBounds[0]))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._restrictionUpperBounds[0]))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._cellDescriptionIndex))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._uNewIndex))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyCellDescription[1]._persistentRecords._subdivisionFactor[0])), 		&disp[31] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellDescription::Datatype );
            MPI_Type_commit( &CellDescription::Datatype );
            
         }
         {
            CellDescription dummyCellDescription[2];
            
            const int Attributes = 37;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //subdivisionFactor
               MPI_INT,		 //ghostLayerWidth
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_INT,		 //level
               MPI_CHAR,		 //isVirtual
               MPI_CHAR,		 //isRemote
               MPI_CHAR,		 //isPaddingSubgrid
               MPI_INT,		 //numberOfSkippedTransfers
               MPI_INT,		 //adjacentRank
               MPI_INT,		 //numberOfSharedAdjacentVertices
               MPI_CHAR,		 //currentStateWasSend
               MPI_CHAR,		 //adjacentRanksChanged
               MPI_INT,		 //upperOverlapByRemoteGhostlayer
               MPI_INT,		 //lowerOverlapByRemoteGhostlayer
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
               MPI_DOUBLE,		 //maximumFineGridTime
               MPI_DOUBLE,		 //minimumFineGridTimestep
               MPI_CHAR,		 //synchronizeFineGrids
               MPI_CHAR,		 //willCoarsen
               MPI_DOUBLE,		 //minimalNeighborTimeConstraint
               MPI_INT,		 //constrainingNeighborIndex
               MPI_DOUBLE,		 //minimalLeafNeighborTimeConstraint
               MPI_DOUBLE,		 //minimalNeighborTime
               MPI_DOUBLE,		 //maximalNeighborTimestep
               MPI_DOUBLE,		 //estimatedNextTimestepSize
               MPI_INT,		 //skipGridIterations
               MPI_INT,		 //ageInGridIterations
               MPI_DOUBLE,		 //demandedMeshWidth
               MPI_DOUBLE,		 //restrictionLowerBounds
               MPI_DOUBLE,		 //restrictionUpperBounds
               MPI_INT,		 //cellDescriptionIndex
               MPI_INT,		 //uNewIndex
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               DIMENSIONS,		 //subdivisionFactor
               1,		 //ghostLayerWidth
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               1,		 //level
               1,		 //isVirtual
               1,		 //isRemote
               1,		 //isPaddingSubgrid
               1,		 //numberOfSkippedTransfers
               1,		 //adjacentRank
               1,		 //numberOfSharedAdjacentVertices
               1,		 //currentStateWasSend
               1,		 //adjacentRanksChanged
               DIMENSIONS,		 //upperOverlapByRemoteGhostlayer
               DIMENSIONS,		 //lowerOverlapByRemoteGhostlayer
               DIMENSIONS,		 //position
               DIMENSIONS,		 //size
               1,		 //time
               1,		 //timestepSize
               1,		 //maximumFineGridTime
               1,		 //minimumFineGridTimestep
               1,		 //synchronizeFineGrids
               1,		 //willCoarsen
               1,		 //minimalNeighborTimeConstraint
               1,		 //constrainingNeighborIndex
               1,		 //minimalLeafNeighborTimeConstraint
               1,		 //minimalNeighborTime
               1,		 //maximalNeighborTimestep
               1,		 //estimatedNextTimestepSize
               1,		 //skipGridIterations
               1,		 //ageInGridIterations
               1,		 //demandedMeshWidth
               DIMENSIONS,		 //restrictionLowerBounds
               DIMENSIONS,		 //restrictionUpperBounds
               1,		 //cellDescriptionIndex
               1,		 //uNewIndex
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._ghostLayerWidth))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._unknownsPerSubcell))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._level))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._isVirtual))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._isRemote))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._isPaddingSubgrid))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._numberOfSkippedTransfers))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._adjacentRank))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._numberOfSharedAdjacentVertices))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._currentStateWasSend))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._adjacentRanksChanged))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._upperOverlapByRemoteGhostlayer[0]))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._lowerOverlapByRemoteGhostlayer[0]))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._position[0]))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._size[0]))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._time))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._timestepSize))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._maximumFineGridTime))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimumFineGridTimestep))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._synchronizeFineGrids))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._willCoarsen))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalNeighborTimeConstraint))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._constrainingNeighborIndex))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalLeafNeighborTimeConstraint))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalNeighborTime))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._maximalNeighborTimestep))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._estimatedNextTimestepSize))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._skipGridIterations))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._ageInGridIterations))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._demandedMeshWidth))), 		&disp[31] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._restrictionLowerBounds[0]))), 		&disp[32] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._restrictionUpperBounds[0]))), 		&disp[33] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._cellDescriptionIndex))), 		&disp[34] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._uNewIndex))), 		&disp[35] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyCellDescription[1]._persistentRecords._subdivisionFactor[0])), 		&disp[36] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellDescription::FullDatatype );
            MPI_Type_commit( &CellDescription::FullDatatype );
            
         }
         
      }
      
      
      void peanoclaw::records::CellDescription::shutdownDatatype() {
         MPI_Type_free( &CellDescription::Datatype );
         MPI_Type_free( &CellDescription::FullDatatype );
         
      }
      
      void peanoclaw::records::CellDescription::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
         MPI_Request* sendRequestHandle = new MPI_Request();
         MPI_Status   status;
         int          flag = 0;
         int          result;
         
         clock_t      timeOutWarning   = -1;
         clock_t      timeOutShutdown  = -1;
         bool         triggeredTimeoutWarning = false;
         
         if (exchangeOnlyAttributesMarkedWithParallelise) {
            result = MPI_Isend(
               this, 1, Datatype, destination,
               tag, tarch::parallel::Node::getInstance().getCommunicator(),
               sendRequestHandle
            );
            
         }
         else {
            result = MPI_Isend(
               this, 1, FullDatatype, destination,
               tag, tarch::parallel::Node::getInstance().getCommunicator(),
               sendRequestHandle
            );
            
         }
         if  (result!=MPI_SUCCESS) {
            std::ostringstream msg;
            msg << "was not able to send message peanoclaw::records::CellDescription "
            << toString()
            << " to node " << destination
            << ": " << tarch::parallel::MPIReturnValueToString(result);
            _log.error( "send(int)",msg.str() );
         }
         result = MPI_Test( sendRequestHandle, &flag, &status );
         while (!flag) {
            if (timeOutWarning==-1)   timeOutWarning   = tarch::parallel::Node::getInstance().getDeadlockWarningTimeStamp();
            if (timeOutShutdown==-1)  timeOutShutdown  = tarch::parallel::Node::getInstance().getDeadlockTimeOutTimeStamp();
            result = MPI_Test( sendRequestHandle, &flag, &status );
            if (result!=MPI_SUCCESS) {
               std::ostringstream msg;
               msg << "testing for finished send task for peanoclaw::records::CellDescription "
               << toString()
               << " sent to node " << destination
               << " failed: " << tarch::parallel::MPIReturnValueToString(result);
               _log.error("send(int)", msg.str() );
            }
            
            // deadlock aspect
            if (
               tarch::parallel::Node::getInstance().isTimeOutWarningEnabled() &&
               (clock()>timeOutWarning) &&
               (!triggeredTimeoutWarning)
            ) {
               tarch::parallel::Node::getInstance().writeTimeOutWarning(
               "peanoclaw::records::CellDescription",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::CellDescription",
               "send(int)", destination,tag,1
               );
            }
            tarch::parallel::Node::getInstance().receiveDanglingMessages();
         }
         
         delete sendRequestHandle;
         #ifdef Debug
         _log.debug("send(int,int)", "sent " + toString() );
         #endif
         
      }
      
      
      
      void peanoclaw::records::CellDescription::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
         MPI_Request* sendRequestHandle = new MPI_Request();
         MPI_Status   status;
         int          flag = 0;
         int          result;
         
         clock_t      timeOutWarning   = -1;
         clock_t      timeOutShutdown  = -1;
         bool         triggeredTimeoutWarning = false;
         
         if (exchangeOnlyAttributesMarkedWithParallelise) {
            result = MPI_Irecv(
               this, 1, Datatype, source, tag,
               tarch::parallel::Node::getInstance().getCommunicator(), sendRequestHandle
            );
            
         }
         else {
            result = MPI_Irecv(
               this, 1, FullDatatype, source, tag,
               tarch::parallel::Node::getInstance().getCommunicator(), sendRequestHandle
            );
            
         }
         if ( result != MPI_SUCCESS ) {
            std::ostringstream msg;
            msg << "failed to start to receive peanoclaw::records::CellDescription from node "
            << source << ": " << tarch::parallel::MPIReturnValueToString(result);
            _log.error( "receive(int)", msg.str() );
         }
         
         result = MPI_Test( sendRequestHandle, &flag, &status );
         while (!flag) {
            if (timeOutWarning==-1)   timeOutWarning   = tarch::parallel::Node::getInstance().getDeadlockWarningTimeStamp();
            if (timeOutShutdown==-1)  timeOutShutdown  = tarch::parallel::Node::getInstance().getDeadlockTimeOutTimeStamp();
            result = MPI_Test( sendRequestHandle, &flag, &status );
            if (result!=MPI_SUCCESS) {
               std::ostringstream msg;
               msg << "testing for finished receive task for peanoclaw::records::CellDescription failed: "
               << tarch::parallel::MPIReturnValueToString(result);
               _log.error("receive(int)", msg.str() );
            }
            
            // deadlock aspect
            if (
               tarch::parallel::Node::getInstance().isTimeOutWarningEnabled() &&
               (clock()>timeOutWarning) &&
               (!triggeredTimeoutWarning)
            ) {
               tarch::parallel::Node::getInstance().writeTimeOutWarning(
               "peanoclaw::records::CellDescription",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::CellDescription",
               "receive(int)", source,tag,1
               );
            }
            tarch::parallel::Node::getInstance().receiveDanglingMessages();
         }
         
         delete sendRequestHandle;
         
         #ifdef Debug
         _log.debug("receive(int,int)", "received " + toString() ); 
         #endif
         
      }
      
      
      
      bool peanoclaw::records::CellDescription::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
         MPI_Status status;
         int  flag        = 0;
         MPI_Iprobe(
            MPI_ANY_SOURCE, tag,
            tarch::parallel::Node::getInstance().getCommunicator(), &flag, &status
         );
         if (flag) {
            int  messageCounter;
            if (exchangeOnlyAttributesMarkedWithParallelise) {
               MPI_Get_count(&status, Datatype, &messageCounter);
            }
            else {
               MPI_Get_count(&status, FullDatatype, &messageCounter);
            }
            return messageCounter > 0;
         }
         else return false;
         
      }
      
      
   #endif
   
   
   peanoclaw::records::CellDescriptionPacked::PersistentRecords::PersistentRecords() {
      assertion((31 < (8 * sizeof(int))));
      
   }
   
   
   peanoclaw::records::CellDescriptionPacked::PersistentRecords::PersistentRecords(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const int& level, const bool& isVirtual, const bool& isRemote, const bool& isPaddingSubgrid, const int& numberOfSkippedTransfers, const int& adjacentRank, const int& numberOfSharedAdjacentVertices, const bool& currentStateWasSend, const bool& adjacentRanksChanged, const tarch::la::Vector<DIMENSIONS,int>& upperOverlapByRemoteGhostlayer, const tarch::la::Vector<DIMENSIONS,int>& lowerOverlapByRemoteGhostlayer, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& maximumFineGridTime, const double& minimumFineGridTimestep, const bool& synchronizeFineGrids, const bool& willCoarsen, const double& minimalNeighborTimeConstraint, const int& constrainingNeighborIndex, const double& minimalLeafNeighborTimeConstraint, const double& minimalNeighborTime, const double& maximalNeighborTimestep, const double& estimatedNextTimestepSize, const int& skipGridIterations, const int& ageInGridIterations, const double& demandedMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds, const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds, const int& cellDescriptionIndex, const int& uNewIndex):
   _subdivisionFactor(subdivisionFactor),
   _adjacentRank(adjacentRank),
   _numberOfSharedAdjacentVertices(numberOfSharedAdjacentVertices),
   _upperOverlapByRemoteGhostlayer(upperOverlapByRemoteGhostlayer),
   _lowerOverlapByRemoteGhostlayer(lowerOverlapByRemoteGhostlayer),
   _position(position),
   _size(size),
   _time(time),
   _timestepSize(timestepSize),
   _maximumFineGridTime(maximumFineGridTime),
   _minimumFineGridTimestep(minimumFineGridTimestep),
   _minimalNeighborTimeConstraint(minimalNeighborTimeConstraint),
   _constrainingNeighborIndex(constrainingNeighborIndex),
   _minimalLeafNeighborTimeConstraint(minimalLeafNeighborTimeConstraint),
   _minimalNeighborTime(minimalNeighborTime),
   _maximalNeighborTimestep(maximalNeighborTimestep),
   _estimatedNextTimestepSize(estimatedNextTimestepSize),
   _ageInGridIterations(ageInGridIterations),
   _demandedMeshWidth(demandedMeshWidth),
   _restrictionLowerBounds(restrictionLowerBounds),
   _restrictionUpperBounds(restrictionUpperBounds),
   _cellDescriptionIndex(cellDescriptionIndex),
   _uNewIndex(uNewIndex) {
      setGhostLayerWidth(ghostLayerWidth);
      setUnknownsPerSubcell(unknownsPerSubcell);
      setAuxiliarFieldsPerSubcell(auxiliarFieldsPerSubcell);
      setLevel(level);
      setIsVirtual(isVirtual);
      setIsRemote(isRemote);
      setIsPaddingSubgrid(isPaddingSubgrid);
      setNumberOfSkippedTransfers(numberOfSkippedTransfers);
      setCurrentStateWasSend(currentStateWasSend);
      setAdjacentRanksChanged(adjacentRanksChanged);
      setSynchronizeFineGrids(synchronizeFineGrids);
      setWillCoarsen(willCoarsen);
      setSkipGridIterations(skipGridIterations);
      assertion((31 < (8 * sizeof(int))));
      
   }
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::CellDescriptionPacked::PersistentRecords::getSubdivisionFactor() const  {
      return _subdivisionFactor;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setSubdivisionFactor(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor)  {
      _subdivisionFactor = (subdivisionFactor);
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getGhostLayerWidth() const  {
      int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (0));
   int tmp = static_cast<int>(_packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (0));
   assertion(( tmp >= 0 &&  tmp <= 15));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setGhostLayerWidth(const int& ghostLayerWidth)  {
      assertion((ghostLayerWidth >= 0 && ghostLayerWidth <= 15));
   int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (0));
   _packedRecords0 = static_cast<int>(_packedRecords0 & ~mask);
   _packedRecords0 = static_cast<int>(_packedRecords0 | ghostLayerWidth << (0));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getUnknownsPerSubcell() const  {
      int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (4));
   int tmp = static_cast<int>(_packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (4));
   assertion(( tmp >= 0 &&  tmp <= 15));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setUnknownsPerSubcell(const int& unknownsPerSubcell)  {
      assertion((unknownsPerSubcell >= 0 && unknownsPerSubcell <= 15));
   int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (4));
   _packedRecords0 = static_cast<int>(_packedRecords0 & ~mask);
   _packedRecords0 = static_cast<int>(_packedRecords0 | unknownsPerSubcell << (4));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getAuxiliarFieldsPerSubcell() const  {
      int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (8));
   int tmp = static_cast<int>(_packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (8));
   assertion(( tmp >= 0 &&  tmp <= 15));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setAuxiliarFieldsPerSubcell(const int& auxiliarFieldsPerSubcell)  {
      assertion((auxiliarFieldsPerSubcell >= 0 && auxiliarFieldsPerSubcell <= 15));
   int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (8));
   _packedRecords0 = static_cast<int>(_packedRecords0 & ~mask);
   _packedRecords0 = static_cast<int>(_packedRecords0 | auxiliarFieldsPerSubcell << (8));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getLevel() const  {
      int mask =  (1 << (5)) - 1;
   mask = static_cast<int>(mask << (12));
   int tmp = static_cast<int>(_packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (12));
   assertion(( tmp >= 0 &&  tmp <= 31));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setLevel(const int& level)  {
      assertion((level >= 0 && level <= 31));
   int mask =  (1 << (5)) - 1;
   mask = static_cast<int>(mask << (12));
   _packedRecords0 = static_cast<int>(_packedRecords0 & ~mask);
   _packedRecords0 = static_cast<int>(_packedRecords0 | level << (12));
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::PersistentRecords::getIsVirtual() const  {
      int mask = 1 << (17);
   int tmp = static_cast<int>(_packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setIsVirtual(const bool& isVirtual)  {
      int mask = 1 << (17);
   _packedRecords0 = static_cast<int>( isVirtual ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::PersistentRecords::getIsRemote() const  {
      int mask = 1 << (18);
   int tmp = static_cast<int>(_packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setIsRemote(const bool& isRemote)  {
      int mask = 1 << (18);
   _packedRecords0 = static_cast<int>( isRemote ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::PersistentRecords::getIsPaddingSubgrid() const  {
      int mask = 1 << (19);
   int tmp = static_cast<int>(_packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setIsPaddingSubgrid(const bool& isPaddingSubgrid)  {
      int mask = 1 << (19);
   _packedRecords0 = static_cast<int>( isPaddingSubgrid ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getNumberOfSkippedTransfers() const  {
      int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (20));
   int tmp = static_cast<int>(_packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (20));
   assertion(( tmp >= 0 &&  tmp <= 8));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setNumberOfSkippedTransfers(const int& numberOfSkippedTransfers)  {
      assertion((numberOfSkippedTransfers >= 0 && numberOfSkippedTransfers <= 8));
   int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (20));
   _packedRecords0 = static_cast<int>(_packedRecords0 & ~mask);
   _packedRecords0 = static_cast<int>(_packedRecords0 | numberOfSkippedTransfers << (20));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getAdjacentRank() const  {
      return _adjacentRank;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setAdjacentRank(const int& adjacentRank)  {
      _adjacentRank = adjacentRank;
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getNumberOfSharedAdjacentVertices() const  {
      return _numberOfSharedAdjacentVertices;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setNumberOfSharedAdjacentVertices(const int& numberOfSharedAdjacentVertices)  {
      _numberOfSharedAdjacentVertices = numberOfSharedAdjacentVertices;
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::PersistentRecords::getCurrentStateWasSend() const  {
      int mask = 1 << (24);
   int tmp = static_cast<int>(_packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setCurrentStateWasSend(const bool& currentStateWasSend)  {
      int mask = 1 << (24);
   _packedRecords0 = static_cast<int>( currentStateWasSend ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::PersistentRecords::getAdjacentRanksChanged() const  {
      int mask = 1 << (25);
   int tmp = static_cast<int>(_packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setAdjacentRanksChanged(const bool& adjacentRanksChanged)  {
      int mask = 1 << (25);
   _packedRecords0 = static_cast<int>( adjacentRanksChanged ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::CellDescriptionPacked::PersistentRecords::getUpperOverlapByRemoteGhostlayer() const  {
      return _upperOverlapByRemoteGhostlayer;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setUpperOverlapByRemoteGhostlayer(const tarch::la::Vector<DIMENSIONS,int>& upperOverlapByRemoteGhostlayer)  {
      _upperOverlapByRemoteGhostlayer = (upperOverlapByRemoteGhostlayer);
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::CellDescriptionPacked::PersistentRecords::getLowerOverlapByRemoteGhostlayer() const  {
      return _lowerOverlapByRemoteGhostlayer;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setLowerOverlapByRemoteGhostlayer(const tarch::la::Vector<DIMENSIONS,int>& lowerOverlapByRemoteGhostlayer)  {
      _lowerOverlapByRemoteGhostlayer = (lowerOverlapByRemoteGhostlayer);
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescriptionPacked::PersistentRecords::getPosition() const  {
      return _position;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setPosition(const tarch::la::Vector<DIMENSIONS,double>& position)  {
      _position = (position);
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescriptionPacked::PersistentRecords::getSize() const  {
      return _size;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setSize(const tarch::la::Vector<DIMENSIONS,double>& size)  {
      _size = (size);
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getTime() const  {
      return _time;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setTime(const double& time)  {
      _time = time;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getTimestepSize() const  {
      return _timestepSize;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setTimestepSize(const double& timestepSize)  {
      _timestepSize = timestepSize;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getMaximumFineGridTime() const  {
      return _maximumFineGridTime;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setMaximumFineGridTime(const double& maximumFineGridTime)  {
      _maximumFineGridTime = maximumFineGridTime;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getMinimumFineGridTimestep() const  {
      return _minimumFineGridTimestep;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setMinimumFineGridTimestep(const double& minimumFineGridTimestep)  {
      _minimumFineGridTimestep = minimumFineGridTimestep;
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::PersistentRecords::getSynchronizeFineGrids() const  {
      int mask = 1 << (26);
   int tmp = static_cast<int>(_packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setSynchronizeFineGrids(const bool& synchronizeFineGrids)  {
      int mask = 1 << (26);
   _packedRecords0 = static_cast<int>( synchronizeFineGrids ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::PersistentRecords::getWillCoarsen() const  {
      int mask = 1 << (27);
   int tmp = static_cast<int>(_packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setWillCoarsen(const bool& willCoarsen)  {
      int mask = 1 << (27);
   _packedRecords0 = static_cast<int>( willCoarsen ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getMinimalNeighborTimeConstraint() const  {
      return _minimalNeighborTimeConstraint;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setMinimalNeighborTimeConstraint(const double& minimalNeighborTimeConstraint)  {
      _minimalNeighborTimeConstraint = minimalNeighborTimeConstraint;
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getConstrainingNeighborIndex() const  {
      return _constrainingNeighborIndex;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setConstrainingNeighborIndex(const int& constrainingNeighborIndex)  {
      _constrainingNeighborIndex = constrainingNeighborIndex;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getMinimalLeafNeighborTimeConstraint() const  {
      return _minimalLeafNeighborTimeConstraint;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setMinimalLeafNeighborTimeConstraint(const double& minimalLeafNeighborTimeConstraint)  {
      _minimalLeafNeighborTimeConstraint = minimalLeafNeighborTimeConstraint;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getMinimalNeighborTime() const  {
      return _minimalNeighborTime;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setMinimalNeighborTime(const double& minimalNeighborTime)  {
      _minimalNeighborTime = minimalNeighborTime;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getMaximalNeighborTimestep() const  {
      return _maximalNeighborTimestep;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setMaximalNeighborTimestep(const double& maximalNeighborTimestep)  {
      _maximalNeighborTimestep = maximalNeighborTimestep;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getEstimatedNextTimestepSize() const  {
      return _estimatedNextTimestepSize;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setEstimatedNextTimestepSize(const double& estimatedNextTimestepSize)  {
      _estimatedNextTimestepSize = estimatedNextTimestepSize;
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getSkipGridIterations() const  {
      int mask =  (1 << (3)) - 1;
   mask = static_cast<int>(mask << (28));
   int tmp = static_cast<int>(_packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (28));
   assertion(( tmp >= 0 &&  tmp <= 7));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setSkipGridIterations(const int& skipGridIterations)  {
      assertion((skipGridIterations >= 0 && skipGridIterations <= 7));
   int mask =  (1 << (3)) - 1;
   mask = static_cast<int>(mask << (28));
   _packedRecords0 = static_cast<int>(_packedRecords0 & ~mask);
   _packedRecords0 = static_cast<int>(_packedRecords0 | skipGridIterations << (28));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getAgeInGridIterations() const  {
      return _ageInGridIterations;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setAgeInGridIterations(const int& ageInGridIterations)  {
      _ageInGridIterations = ageInGridIterations;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getDemandedMeshWidth() const  {
      return _demandedMeshWidth;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setDemandedMeshWidth(const double& demandedMeshWidth)  {
      _demandedMeshWidth = demandedMeshWidth;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescriptionPacked::PersistentRecords::getRestrictionLowerBounds() const  {
      return _restrictionLowerBounds;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setRestrictionLowerBounds(const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds)  {
      _restrictionLowerBounds = (restrictionLowerBounds);
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescriptionPacked::PersistentRecords::getRestrictionUpperBounds() const  {
      return _restrictionUpperBounds;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setRestrictionUpperBounds(const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds)  {
      _restrictionUpperBounds = (restrictionUpperBounds);
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getCellDescriptionIndex() const  {
      return _cellDescriptionIndex;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setCellDescriptionIndex(const int& cellDescriptionIndex)  {
      _cellDescriptionIndex = cellDescriptionIndex;
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getUNewIndex() const  {
      return _uNewIndex;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setUNewIndex(const int& uNewIndex)  {
      _uNewIndex = uNewIndex;
   }
   
   
   peanoclaw::records::CellDescriptionPacked::CellDescriptionPacked() {
      assertion((31 < (8 * sizeof(int))));
      
   }
   
   
   peanoclaw::records::CellDescriptionPacked::CellDescriptionPacked(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._subdivisionFactor, persistentRecords.getGhostLayerWidth(), persistentRecords.getUnknownsPerSubcell(), persistentRecords.getAuxiliarFieldsPerSubcell(), persistentRecords.getLevel(), persistentRecords.getIsVirtual(), persistentRecords.getIsRemote(), persistentRecords.getIsPaddingSubgrid(), persistentRecords.getNumberOfSkippedTransfers(), persistentRecords._adjacentRank, persistentRecords._numberOfSharedAdjacentVertices, persistentRecords.getCurrentStateWasSend(), persistentRecords.getAdjacentRanksChanged(), persistentRecords._upperOverlapByRemoteGhostlayer, persistentRecords._lowerOverlapByRemoteGhostlayer, persistentRecords._position, persistentRecords._size, persistentRecords._time, persistentRecords._timestepSize, persistentRecords._maximumFineGridTime, persistentRecords._minimumFineGridTimestep, persistentRecords.getSynchronizeFineGrids(), persistentRecords.getWillCoarsen(), persistentRecords._minimalNeighborTimeConstraint, persistentRecords._constrainingNeighborIndex, persistentRecords._minimalLeafNeighborTimeConstraint, persistentRecords._minimalNeighborTime, persistentRecords._maximalNeighborTimestep, persistentRecords._estimatedNextTimestepSize, persistentRecords.getSkipGridIterations(), persistentRecords._ageInGridIterations, persistentRecords._demandedMeshWidth, persistentRecords._restrictionLowerBounds, persistentRecords._restrictionUpperBounds, persistentRecords._cellDescriptionIndex, persistentRecords._uNewIndex) {
      assertion((31 < (8 * sizeof(int))));
      
   }
   
   
   peanoclaw::records::CellDescriptionPacked::CellDescriptionPacked(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const int& level, const bool& isVirtual, const bool& isRemote, const bool& isPaddingSubgrid, const int& numberOfSkippedTransfers, const int& adjacentRank, const int& numberOfSharedAdjacentVertices, const bool& currentStateWasSend, const bool& adjacentRanksChanged, const tarch::la::Vector<DIMENSIONS,int>& upperOverlapByRemoteGhostlayer, const tarch::la::Vector<DIMENSIONS,int>& lowerOverlapByRemoteGhostlayer, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& maximumFineGridTime, const double& minimumFineGridTimestep, const bool& synchronizeFineGrids, const bool& willCoarsen, const double& minimalNeighborTimeConstraint, const int& constrainingNeighborIndex, const double& minimalLeafNeighborTimeConstraint, const double& minimalNeighborTime, const double& maximalNeighborTimestep, const double& estimatedNextTimestepSize, const int& skipGridIterations, const int& ageInGridIterations, const double& demandedMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds, const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds, const int& cellDescriptionIndex, const int& uNewIndex):
   _persistentRecords(subdivisionFactor, ghostLayerWidth, unknownsPerSubcell, auxiliarFieldsPerSubcell, level, isVirtual, isRemote, isPaddingSubgrid, numberOfSkippedTransfers, adjacentRank, numberOfSharedAdjacentVertices, currentStateWasSend, adjacentRanksChanged, upperOverlapByRemoteGhostlayer, lowerOverlapByRemoteGhostlayer, position, size, time, timestepSize, maximumFineGridTime, minimumFineGridTimestep, synchronizeFineGrids, willCoarsen, minimalNeighborTimeConstraint, constrainingNeighborIndex, minimalLeafNeighborTimeConstraint, minimalNeighborTime, maximalNeighborTimestep, estimatedNextTimestepSize, skipGridIterations, ageInGridIterations, demandedMeshWidth, restrictionLowerBounds, restrictionUpperBounds, cellDescriptionIndex, uNewIndex) {
      assertion((31 < (8 * sizeof(int))));
      
   }
   
   
   peanoclaw::records::CellDescriptionPacked::~CellDescriptionPacked() { }
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::CellDescriptionPacked::getSubdivisionFactor() const  {
      return _persistentRecords._subdivisionFactor;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setSubdivisionFactor(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor)  {
      _persistentRecords._subdivisionFactor = (subdivisionFactor);
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getSubdivisionFactor(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._subdivisionFactor[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setSubdivisionFactor(int elementIndex, const int& subdivisionFactor)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._subdivisionFactor[elementIndex]= subdivisionFactor;
      
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getGhostLayerWidth() const  {
      int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (0));
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (0));
   assertion(( tmp >= 0 &&  tmp <= 15));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setGhostLayerWidth(const int& ghostLayerWidth)  {
      assertion((ghostLayerWidth >= 0 && ghostLayerWidth <= 15));
   int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (0));
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 & ~mask);
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 | ghostLayerWidth << (0));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getUnknownsPerSubcell() const  {
      int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (4));
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (4));
   assertion(( tmp >= 0 &&  tmp <= 15));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setUnknownsPerSubcell(const int& unknownsPerSubcell)  {
      assertion((unknownsPerSubcell >= 0 && unknownsPerSubcell <= 15));
   int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (4));
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 & ~mask);
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 | unknownsPerSubcell << (4));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getAuxiliarFieldsPerSubcell() const  {
      int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (8));
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (8));
   assertion(( tmp >= 0 &&  tmp <= 15));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setAuxiliarFieldsPerSubcell(const int& auxiliarFieldsPerSubcell)  {
      assertion((auxiliarFieldsPerSubcell >= 0 && auxiliarFieldsPerSubcell <= 15));
   int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (8));
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 & ~mask);
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 | auxiliarFieldsPerSubcell << (8));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getLevel() const  {
      int mask =  (1 << (5)) - 1;
   mask = static_cast<int>(mask << (12));
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (12));
   assertion(( tmp >= 0 &&  tmp <= 31));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setLevel(const int& level)  {
      assertion((level >= 0 && level <= 31));
   int mask =  (1 << (5)) - 1;
   mask = static_cast<int>(mask << (12));
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 & ~mask);
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 | level << (12));
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::getIsVirtual() const  {
      int mask = 1 << (17);
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setIsVirtual(const bool& isVirtual)  {
      int mask = 1 << (17);
   _persistentRecords._packedRecords0 = static_cast<int>( isVirtual ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::getIsRemote() const  {
      int mask = 1 << (18);
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setIsRemote(const bool& isRemote)  {
      int mask = 1 << (18);
   _persistentRecords._packedRecords0 = static_cast<int>( isRemote ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::getIsPaddingSubgrid() const  {
      int mask = 1 << (19);
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setIsPaddingSubgrid(const bool& isPaddingSubgrid)  {
      int mask = 1 << (19);
   _persistentRecords._packedRecords0 = static_cast<int>( isPaddingSubgrid ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getNumberOfSkippedTransfers() const  {
      int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (20));
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (20));
   assertion(( tmp >= 0 &&  tmp <= 8));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setNumberOfSkippedTransfers(const int& numberOfSkippedTransfers)  {
      assertion((numberOfSkippedTransfers >= 0 && numberOfSkippedTransfers <= 8));
   int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (20));
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 & ~mask);
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 | numberOfSkippedTransfers << (20));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getAdjacentRank() const  {
      return _persistentRecords._adjacentRank;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setAdjacentRank(const int& adjacentRank)  {
      _persistentRecords._adjacentRank = adjacentRank;
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getNumberOfSharedAdjacentVertices() const  {
      return _persistentRecords._numberOfSharedAdjacentVertices;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setNumberOfSharedAdjacentVertices(const int& numberOfSharedAdjacentVertices)  {
      _persistentRecords._numberOfSharedAdjacentVertices = numberOfSharedAdjacentVertices;
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::getCurrentStateWasSend() const  {
      int mask = 1 << (24);
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setCurrentStateWasSend(const bool& currentStateWasSend)  {
      int mask = 1 << (24);
   _persistentRecords._packedRecords0 = static_cast<int>( currentStateWasSend ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::getAdjacentRanksChanged() const  {
      int mask = 1 << (25);
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setAdjacentRanksChanged(const bool& adjacentRanksChanged)  {
      int mask = 1 << (25);
   _persistentRecords._packedRecords0 = static_cast<int>( adjacentRanksChanged ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::CellDescriptionPacked::getUpperOverlapByRemoteGhostlayer() const  {
      return _persistentRecords._upperOverlapByRemoteGhostlayer;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setUpperOverlapByRemoteGhostlayer(const tarch::la::Vector<DIMENSIONS,int>& upperOverlapByRemoteGhostlayer)  {
      _persistentRecords._upperOverlapByRemoteGhostlayer = (upperOverlapByRemoteGhostlayer);
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getUpperOverlapByRemoteGhostlayer(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._upperOverlapByRemoteGhostlayer[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setUpperOverlapByRemoteGhostlayer(int elementIndex, const int& upperOverlapByRemoteGhostlayer)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._upperOverlapByRemoteGhostlayer[elementIndex]= upperOverlapByRemoteGhostlayer;
      
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::CellDescriptionPacked::getLowerOverlapByRemoteGhostlayer() const  {
      return _persistentRecords._lowerOverlapByRemoteGhostlayer;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setLowerOverlapByRemoteGhostlayer(const tarch::la::Vector<DIMENSIONS,int>& lowerOverlapByRemoteGhostlayer)  {
      _persistentRecords._lowerOverlapByRemoteGhostlayer = (lowerOverlapByRemoteGhostlayer);
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getLowerOverlapByRemoteGhostlayer(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._lowerOverlapByRemoteGhostlayer[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setLowerOverlapByRemoteGhostlayer(int elementIndex, const int& lowerOverlapByRemoteGhostlayer)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._lowerOverlapByRemoteGhostlayer[elementIndex]= lowerOverlapByRemoteGhostlayer;
      
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescriptionPacked::getPosition() const  {
      return _persistentRecords._position;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setPosition(const tarch::la::Vector<DIMENSIONS,double>& position)  {
      _persistentRecords._position = (position);
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getPosition(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._position[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setPosition(int elementIndex, const double& position)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._position[elementIndex]= position;
      
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescriptionPacked::getSize() const  {
      return _persistentRecords._size;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setSize(const tarch::la::Vector<DIMENSIONS,double>& size)  {
      _persistentRecords._size = (size);
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getSize(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._size[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setSize(int elementIndex, const double& size)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._size[elementIndex]= size;
      
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getTime() const  {
      return _persistentRecords._time;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setTime(const double& time)  {
      _persistentRecords._time = time;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getTimestepSize() const  {
      return _persistentRecords._timestepSize;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setTimestepSize(const double& timestepSize)  {
      _persistentRecords._timestepSize = timestepSize;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getMaximumFineGridTime() const  {
      return _persistentRecords._maximumFineGridTime;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setMaximumFineGridTime(const double& maximumFineGridTime)  {
      _persistentRecords._maximumFineGridTime = maximumFineGridTime;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getMinimumFineGridTimestep() const  {
      return _persistentRecords._minimumFineGridTimestep;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setMinimumFineGridTimestep(const double& minimumFineGridTimestep)  {
      _persistentRecords._minimumFineGridTimestep = minimumFineGridTimestep;
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::getSynchronizeFineGrids() const  {
      int mask = 1 << (26);
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setSynchronizeFineGrids(const bool& synchronizeFineGrids)  {
      int mask = 1 << (26);
   _persistentRecords._packedRecords0 = static_cast<int>( synchronizeFineGrids ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::getWillCoarsen() const  {
      int mask = 1 << (27);
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setWillCoarsen(const bool& willCoarsen)  {
      int mask = 1 << (27);
   _persistentRecords._packedRecords0 = static_cast<int>( willCoarsen ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getMinimalNeighborTimeConstraint() const  {
      return _persistentRecords._minimalNeighborTimeConstraint;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setMinimalNeighborTimeConstraint(const double& minimalNeighborTimeConstraint)  {
      _persistentRecords._minimalNeighborTimeConstraint = minimalNeighborTimeConstraint;
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getConstrainingNeighborIndex() const  {
      return _persistentRecords._constrainingNeighborIndex;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setConstrainingNeighborIndex(const int& constrainingNeighborIndex)  {
      _persistentRecords._constrainingNeighborIndex = constrainingNeighborIndex;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getMinimalLeafNeighborTimeConstraint() const  {
      return _persistentRecords._minimalLeafNeighborTimeConstraint;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setMinimalLeafNeighborTimeConstraint(const double& minimalLeafNeighborTimeConstraint)  {
      _persistentRecords._minimalLeafNeighborTimeConstraint = minimalLeafNeighborTimeConstraint;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getMinimalNeighborTime() const  {
      return _persistentRecords._minimalNeighborTime;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setMinimalNeighborTime(const double& minimalNeighborTime)  {
      _persistentRecords._minimalNeighborTime = minimalNeighborTime;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getMaximalNeighborTimestep() const  {
      return _persistentRecords._maximalNeighborTimestep;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setMaximalNeighborTimestep(const double& maximalNeighborTimestep)  {
      _persistentRecords._maximalNeighborTimestep = maximalNeighborTimestep;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getEstimatedNextTimestepSize() const  {
      return _persistentRecords._estimatedNextTimestepSize;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setEstimatedNextTimestepSize(const double& estimatedNextTimestepSize)  {
      _persistentRecords._estimatedNextTimestepSize = estimatedNextTimestepSize;
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getSkipGridIterations() const  {
      int mask =  (1 << (3)) - 1;
   mask = static_cast<int>(mask << (28));
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (28));
   assertion(( tmp >= 0 &&  tmp <= 7));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setSkipGridIterations(const int& skipGridIterations)  {
      assertion((skipGridIterations >= 0 && skipGridIterations <= 7));
   int mask =  (1 << (3)) - 1;
   mask = static_cast<int>(mask << (28));
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 & ~mask);
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 | skipGridIterations << (28));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getAgeInGridIterations() const  {
      return _persistentRecords._ageInGridIterations;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setAgeInGridIterations(const int& ageInGridIterations)  {
      _persistentRecords._ageInGridIterations = ageInGridIterations;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getDemandedMeshWidth() const  {
      return _persistentRecords._demandedMeshWidth;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setDemandedMeshWidth(const double& demandedMeshWidth)  {
      _persistentRecords._demandedMeshWidth = demandedMeshWidth;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescriptionPacked::getRestrictionLowerBounds() const  {
      return _persistentRecords._restrictionLowerBounds;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setRestrictionLowerBounds(const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds)  {
      _persistentRecords._restrictionLowerBounds = (restrictionLowerBounds);
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getRestrictionLowerBounds(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._restrictionLowerBounds[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setRestrictionLowerBounds(int elementIndex, const double& restrictionLowerBounds)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._restrictionLowerBounds[elementIndex]= restrictionLowerBounds;
      
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescriptionPacked::getRestrictionUpperBounds() const  {
      return _persistentRecords._restrictionUpperBounds;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setRestrictionUpperBounds(const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds)  {
      _persistentRecords._restrictionUpperBounds = (restrictionUpperBounds);
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getRestrictionUpperBounds(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._restrictionUpperBounds[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setRestrictionUpperBounds(int elementIndex, const double& restrictionUpperBounds)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._restrictionUpperBounds[elementIndex]= restrictionUpperBounds;
      
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getCellDescriptionIndex() const  {
      return _persistentRecords._cellDescriptionIndex;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setCellDescriptionIndex(const int& cellDescriptionIndex)  {
      _persistentRecords._cellDescriptionIndex = cellDescriptionIndex;
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getUNewIndex() const  {
      return _persistentRecords._uNewIndex;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setUNewIndex(const int& uNewIndex)  {
      _persistentRecords._uNewIndex = uNewIndex;
   }
   
   
   
   
   std::string peanoclaw::records::CellDescriptionPacked::toString() const {
      std::ostringstream stringstr;
      toString(stringstr);
      return stringstr.str();
   }
   
   void peanoclaw::records::CellDescriptionPacked::toString (std::ostream& out) const {
      out << "("; 
      out << "subdivisionFactor:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getSubdivisionFactor(i) << ",";
   }
   out << getSubdivisionFactor(DIMENSIONS-1) << "]";
      out << ",";
      out << "ghostLayerWidth:" << getGhostLayerWidth();
      out << ",";
      out << "unknownsPerSubcell:" << getUnknownsPerSubcell();
      out << ",";
      out << "auxiliarFieldsPerSubcell:" << getAuxiliarFieldsPerSubcell();
      out << ",";
      out << "level:" << getLevel();
      out << ",";
      out << "isVirtual:" << getIsVirtual();
      out << ",";
      out << "isRemote:" << getIsRemote();
      out << ",";
      out << "isPaddingSubgrid:" << getIsPaddingSubgrid();
      out << ",";
      out << "numberOfSkippedTransfers:" << getNumberOfSkippedTransfers();
      out << ",";
      out << "adjacentRank:" << getAdjacentRank();
      out << ",";
      out << "numberOfSharedAdjacentVertices:" << getNumberOfSharedAdjacentVertices();
      out << ",";
      out << "currentStateWasSend:" << getCurrentStateWasSend();
      out << ",";
      out << "adjacentRanksChanged:" << getAdjacentRanksChanged();
      out << ",";
      out << "upperOverlapByRemoteGhostlayer:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getUpperOverlapByRemoteGhostlayer(i) << ",";
   }
   out << getUpperOverlapByRemoteGhostlayer(DIMENSIONS-1) << "]";
      out << ",";
      out << "lowerOverlapByRemoteGhostlayer:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getLowerOverlapByRemoteGhostlayer(i) << ",";
   }
   out << getLowerOverlapByRemoteGhostlayer(DIMENSIONS-1) << "]";
      out << ",";
      out << "position:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getPosition(i) << ",";
   }
   out << getPosition(DIMENSIONS-1) << "]";
      out << ",";
      out << "size:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getSize(i) << ",";
   }
   out << getSize(DIMENSIONS-1) << "]";
      out << ",";
      out << "time:" << getTime();
      out << ",";
      out << "timestepSize:" << getTimestepSize();
      out << ",";
      out << "maximumFineGridTime:" << getMaximumFineGridTime();
      out << ",";
      out << "minimumFineGridTimestep:" << getMinimumFineGridTimestep();
      out << ",";
      out << "synchronizeFineGrids:" << getSynchronizeFineGrids();
      out << ",";
      out << "willCoarsen:" << getWillCoarsen();
      out << ",";
      out << "minimalNeighborTimeConstraint:" << getMinimalNeighborTimeConstraint();
      out << ",";
      out << "constrainingNeighborIndex:" << getConstrainingNeighborIndex();
      out << ",";
      out << "minimalLeafNeighborTimeConstraint:" << getMinimalLeafNeighborTimeConstraint();
      out << ",";
      out << "minimalNeighborTime:" << getMinimalNeighborTime();
      out << ",";
      out << "maximalNeighborTimestep:" << getMaximalNeighborTimestep();
      out << ",";
      out << "estimatedNextTimestepSize:" << getEstimatedNextTimestepSize();
      out << ",";
      out << "skipGridIterations:" << getSkipGridIterations();
      out << ",";
      out << "ageInGridIterations:" << getAgeInGridIterations();
      out << ",";
      out << "demandedMeshWidth:" << getDemandedMeshWidth();
      out << ",";
      out << "restrictionLowerBounds:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getRestrictionLowerBounds(i) << ",";
   }
   out << getRestrictionLowerBounds(DIMENSIONS-1) << "]";
      out << ",";
      out << "restrictionUpperBounds:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getRestrictionUpperBounds(i) << ",";
   }
   out << getRestrictionUpperBounds(DIMENSIONS-1) << "]";
      out << ",";
      out << "cellDescriptionIndex:" << getCellDescriptionIndex();
      out << ",";
      out << "uNewIndex:" << getUNewIndex();
      out <<  ")";
   }
   
   
   peanoclaw::records::CellDescriptionPacked::PersistentRecords peanoclaw::records::CellDescriptionPacked::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::CellDescription peanoclaw::records::CellDescriptionPacked::convert() const{
      return CellDescription(
         getSubdivisionFactor(),
         getGhostLayerWidth(),
         getUnknownsPerSubcell(),
         getAuxiliarFieldsPerSubcell(),
         getLevel(),
         getIsVirtual(),
         getIsRemote(),
         getIsPaddingSubgrid(),
         getNumberOfSkippedTransfers(),
         getAdjacentRank(),
         getNumberOfSharedAdjacentVertices(),
         getCurrentStateWasSend(),
         getAdjacentRanksChanged(),
         getUpperOverlapByRemoteGhostlayer(),
         getLowerOverlapByRemoteGhostlayer(),
         getPosition(),
         getSize(),
         getTime(),
         getTimestepSize(),
         getMaximumFineGridTime(),
         getMinimumFineGridTimestep(),
         getSynchronizeFineGrids(),
         getWillCoarsen(),
         getMinimalNeighborTimeConstraint(),
         getConstrainingNeighborIndex(),
         getMinimalLeafNeighborTimeConstraint(),
         getMinimalNeighborTime(),
         getMaximalNeighborTimestep(),
         getEstimatedNextTimestepSize(),
         getSkipGridIterations(),
         getAgeInGridIterations(),
         getDemandedMeshWidth(),
         getRestrictionLowerBounds(),
         getRestrictionUpperBounds(),
         getCellDescriptionIndex(),
         getUNewIndex()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::CellDescriptionPacked::_log( "peanoclaw::records::CellDescriptionPacked" );
      
      MPI_Datatype peanoclaw::records::CellDescriptionPacked::Datatype = 0;
      MPI_Datatype peanoclaw::records::CellDescriptionPacked::FullDatatype = 0;
      
      
      void peanoclaw::records::CellDescriptionPacked::initDatatype() {
         {
            CellDescriptionPacked dummyCellDescriptionPacked[2];
            
            const int Attributes = 22;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //subdivisionFactor
               MPI_INT,		 //upperOverlapByRemoteGhostlayer
               MPI_INT,		 //lowerOverlapByRemoteGhostlayer
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
               MPI_DOUBLE,		 //maximumFineGridTime
               MPI_DOUBLE,		 //minimumFineGridTimestep
               MPI_DOUBLE,		 //minimalNeighborTimeConstraint
               MPI_DOUBLE,		 //minimalLeafNeighborTimeConstraint
               MPI_DOUBLE,		 //minimalNeighborTime
               MPI_DOUBLE,		 //maximalNeighborTimestep
               MPI_DOUBLE,		 //estimatedNextTimestepSize
               MPI_INT,		 //ageInGridIterations
               MPI_DOUBLE,		 //demandedMeshWidth
               MPI_DOUBLE,		 //restrictionLowerBounds
               MPI_DOUBLE,		 //restrictionUpperBounds
               MPI_INT,		 //cellDescriptionIndex
               MPI_INT,		 //uNewIndex
               MPI_INT,		 //_packedRecords0
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               DIMENSIONS,		 //subdivisionFactor
               DIMENSIONS,		 //upperOverlapByRemoteGhostlayer
               DIMENSIONS,		 //lowerOverlapByRemoteGhostlayer
               DIMENSIONS,		 //position
               DIMENSIONS,		 //size
               1,		 //time
               1,		 //timestepSize
               1,		 //maximumFineGridTime
               1,		 //minimumFineGridTimestep
               1,		 //minimalNeighborTimeConstraint
               1,		 //minimalLeafNeighborTimeConstraint
               1,		 //minimalNeighborTime
               1,		 //maximalNeighborTimestep
               1,		 //estimatedNextTimestepSize
               1,		 //ageInGridIterations
               1,		 //demandedMeshWidth
               DIMENSIONS,		 //restrictionLowerBounds
               DIMENSIONS,		 //restrictionUpperBounds
               1,		 //cellDescriptionIndex
               1,		 //uNewIndex
               1,		 //_packedRecords0
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._upperOverlapByRemoteGhostlayer[0]))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._lowerOverlapByRemoteGhostlayer[0]))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._position[0]))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._size[0]))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._time))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._timestepSize))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._maximumFineGridTime))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimumFineGridTimestep))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimalNeighborTimeConstraint))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimalLeafNeighborTimeConstraint))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimalNeighborTime))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._maximalNeighborTimestep))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._estimatedNextTimestepSize))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._ageInGridIterations))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._demandedMeshWidth))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._restrictionLowerBounds[0]))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._restrictionUpperBounds[0]))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._uNewIndex))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._packedRecords0))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyCellDescriptionPacked[1]._persistentRecords._subdivisionFactor[0])), 		&disp[21] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellDescriptionPacked::Datatype );
            MPI_Type_commit( &CellDescriptionPacked::Datatype );
            
         }
         {
            CellDescriptionPacked dummyCellDescriptionPacked[2];
            
            const int Attributes = 25;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //subdivisionFactor
               MPI_INT,		 //adjacentRank
               MPI_INT,		 //numberOfSharedAdjacentVertices
               MPI_INT,		 //upperOverlapByRemoteGhostlayer
               MPI_INT,		 //lowerOverlapByRemoteGhostlayer
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
               MPI_DOUBLE,		 //maximumFineGridTime
               MPI_DOUBLE,		 //minimumFineGridTimestep
               MPI_DOUBLE,		 //minimalNeighborTimeConstraint
               MPI_INT,		 //constrainingNeighborIndex
               MPI_DOUBLE,		 //minimalLeafNeighborTimeConstraint
               MPI_DOUBLE,		 //minimalNeighborTime
               MPI_DOUBLE,		 //maximalNeighborTimestep
               MPI_DOUBLE,		 //estimatedNextTimestepSize
               MPI_INT,		 //ageInGridIterations
               MPI_DOUBLE,		 //demandedMeshWidth
               MPI_DOUBLE,		 //restrictionLowerBounds
               MPI_DOUBLE,		 //restrictionUpperBounds
               MPI_INT,		 //cellDescriptionIndex
               MPI_INT,		 //uNewIndex
               MPI_INT,		 //_packedRecords0
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               DIMENSIONS,		 //subdivisionFactor
               1,		 //adjacentRank
               1,		 //numberOfSharedAdjacentVertices
               DIMENSIONS,		 //upperOverlapByRemoteGhostlayer
               DIMENSIONS,		 //lowerOverlapByRemoteGhostlayer
               DIMENSIONS,		 //position
               DIMENSIONS,		 //size
               1,		 //time
               1,		 //timestepSize
               1,		 //maximumFineGridTime
               1,		 //minimumFineGridTimestep
               1,		 //minimalNeighborTimeConstraint
               1,		 //constrainingNeighborIndex
               1,		 //minimalLeafNeighborTimeConstraint
               1,		 //minimalNeighborTime
               1,		 //maximalNeighborTimestep
               1,		 //estimatedNextTimestepSize
               1,		 //ageInGridIterations
               1,		 //demandedMeshWidth
               DIMENSIONS,		 //restrictionLowerBounds
               DIMENSIONS,		 //restrictionUpperBounds
               1,		 //cellDescriptionIndex
               1,		 //uNewIndex
               1,		 //_packedRecords0
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._adjacentRank))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._numberOfSharedAdjacentVertices))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._upperOverlapByRemoteGhostlayer[0]))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._lowerOverlapByRemoteGhostlayer[0]))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._position[0]))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._size[0]))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._time))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._timestepSize))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._maximumFineGridTime))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimumFineGridTimestep))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimalNeighborTimeConstraint))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._constrainingNeighborIndex))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimalLeafNeighborTimeConstraint))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimalNeighborTime))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._maximalNeighborTimestep))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._estimatedNextTimestepSize))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._ageInGridIterations))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._demandedMeshWidth))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._restrictionLowerBounds[0]))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._restrictionUpperBounds[0]))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._uNewIndex))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._packedRecords0))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyCellDescriptionPacked[1]._persistentRecords._subdivisionFactor[0])), 		&disp[24] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellDescriptionPacked::FullDatatype );
            MPI_Type_commit( &CellDescriptionPacked::FullDatatype );
            
         }
         
      }
      
      
      void peanoclaw::records::CellDescriptionPacked::shutdownDatatype() {
         MPI_Type_free( &CellDescriptionPacked::Datatype );
         MPI_Type_free( &CellDescriptionPacked::FullDatatype );
         
      }
      
      void peanoclaw::records::CellDescriptionPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
         MPI_Request* sendRequestHandle = new MPI_Request();
         MPI_Status   status;
         int          flag = 0;
         int          result;
         
         clock_t      timeOutWarning   = -1;
         clock_t      timeOutShutdown  = -1;
         bool         triggeredTimeoutWarning = false;
         
         if (exchangeOnlyAttributesMarkedWithParallelise) {
            result = MPI_Isend(
               this, 1, Datatype, destination,
               tag, tarch::parallel::Node::getInstance().getCommunicator(),
               sendRequestHandle
            );
            
         }
         else {
            result = MPI_Isend(
               this, 1, FullDatatype, destination,
               tag, tarch::parallel::Node::getInstance().getCommunicator(),
               sendRequestHandle
            );
            
         }
         if  (result!=MPI_SUCCESS) {
            std::ostringstream msg;
            msg << "was not able to send message peanoclaw::records::CellDescriptionPacked "
            << toString()
            << " to node " << destination
            << ": " << tarch::parallel::MPIReturnValueToString(result);
            _log.error( "send(int)",msg.str() );
         }
         result = MPI_Test( sendRequestHandle, &flag, &status );
         while (!flag) {
            if (timeOutWarning==-1)   timeOutWarning   = tarch::parallel::Node::getInstance().getDeadlockWarningTimeStamp();
            if (timeOutShutdown==-1)  timeOutShutdown  = tarch::parallel::Node::getInstance().getDeadlockTimeOutTimeStamp();
            result = MPI_Test( sendRequestHandle, &flag, &status );
            if (result!=MPI_SUCCESS) {
               std::ostringstream msg;
               msg << "testing for finished send task for peanoclaw::records::CellDescriptionPacked "
               << toString()
               << " sent to node " << destination
               << " failed: " << tarch::parallel::MPIReturnValueToString(result);
               _log.error("send(int)", msg.str() );
            }
            
            // deadlock aspect
            if (
               tarch::parallel::Node::getInstance().isTimeOutWarningEnabled() &&
               (clock()>timeOutWarning) &&
               (!triggeredTimeoutWarning)
            ) {
               tarch::parallel::Node::getInstance().writeTimeOutWarning(
               "peanoclaw::records::CellDescriptionPacked",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::CellDescriptionPacked",
               "send(int)", destination,tag,1
               );
            }
            tarch::parallel::Node::getInstance().receiveDanglingMessages();
         }
         
         delete sendRequestHandle;
         #ifdef Debug
         _log.debug("send(int,int)", "sent " + toString() );
         #endif
         
      }
      
      
      
      void peanoclaw::records::CellDescriptionPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
         MPI_Request* sendRequestHandle = new MPI_Request();
         MPI_Status   status;
         int          flag = 0;
         int          result;
         
         clock_t      timeOutWarning   = -1;
         clock_t      timeOutShutdown  = -1;
         bool         triggeredTimeoutWarning = false;
         
         if (exchangeOnlyAttributesMarkedWithParallelise) {
            result = MPI_Irecv(
               this, 1, Datatype, source, tag,
               tarch::parallel::Node::getInstance().getCommunicator(), sendRequestHandle
            );
            
         }
         else {
            result = MPI_Irecv(
               this, 1, FullDatatype, source, tag,
               tarch::parallel::Node::getInstance().getCommunicator(), sendRequestHandle
            );
            
         }
         if ( result != MPI_SUCCESS ) {
            std::ostringstream msg;
            msg << "failed to start to receive peanoclaw::records::CellDescriptionPacked from node "
            << source << ": " << tarch::parallel::MPIReturnValueToString(result);
            _log.error( "receive(int)", msg.str() );
         }
         
         result = MPI_Test( sendRequestHandle, &flag, &status );
         while (!flag) {
            if (timeOutWarning==-1)   timeOutWarning   = tarch::parallel::Node::getInstance().getDeadlockWarningTimeStamp();
            if (timeOutShutdown==-1)  timeOutShutdown  = tarch::parallel::Node::getInstance().getDeadlockTimeOutTimeStamp();
            result = MPI_Test( sendRequestHandle, &flag, &status );
            if (result!=MPI_SUCCESS) {
               std::ostringstream msg;
               msg << "testing for finished receive task for peanoclaw::records::CellDescriptionPacked failed: "
               << tarch::parallel::MPIReturnValueToString(result);
               _log.error("receive(int)", msg.str() );
            }
            
            // deadlock aspect
            if (
               tarch::parallel::Node::getInstance().isTimeOutWarningEnabled() &&
               (clock()>timeOutWarning) &&
               (!triggeredTimeoutWarning)
            ) {
               tarch::parallel::Node::getInstance().writeTimeOutWarning(
               "peanoclaw::records::CellDescriptionPacked",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::CellDescriptionPacked",
               "receive(int)", source,tag,1
               );
            }
            tarch::parallel::Node::getInstance().receiveDanglingMessages();
         }
         
         delete sendRequestHandle;
         
         #ifdef Debug
         _log.debug("receive(int,int)", "received " + toString() ); 
         #endif
         
      }
      
      
      
      bool peanoclaw::records::CellDescriptionPacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
         MPI_Status status;
         int  flag        = 0;
         MPI_Iprobe(
            MPI_ANY_SOURCE, tag,
            tarch::parallel::Node::getInstance().getCommunicator(), &flag, &status
         );
         if (flag) {
            int  messageCounter;
            if (exchangeOnlyAttributesMarkedWithParallelise) {
               MPI_Get_count(&status, Datatype, &messageCounter);
            }
            else {
               MPI_Get_count(&status, FullDatatype, &messageCounter);
            }
            return messageCounter > 0;
         }
         else return false;
         
      }
      
      
   #endif
   
   
   
#elif !defined(Parallel)
   peanoclaw::records::CellDescription::PersistentRecords::PersistentRecords() {
      
   }
   
   
   peanoclaw::records::CellDescription::PersistentRecords::PersistentRecords(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const int& level, const bool& isVirtual, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& maximumFineGridTime, const double& minimumFineGridTimestep, const bool& synchronizeFineGrids, const bool& willCoarsen, const double& minimalNeighborTimeConstraint, const int& constrainingNeighborIndex, const double& minimalLeafNeighborTimeConstraint, const double& minimalNeighborTime, const double& maximalNeighborTimestep, const double& estimatedNextTimestepSize, const int& skipGridIterations, const int& ageInGridIterations, const double& demandedMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds, const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds, const int& cellDescriptionIndex, const int& uNewIndex):
   _subdivisionFactor(subdivisionFactor),
   _ghostLayerWidth(ghostLayerWidth),
   _unknownsPerSubcell(unknownsPerSubcell),
   _auxiliarFieldsPerSubcell(auxiliarFieldsPerSubcell),
   _level(level),
   _isVirtual(isVirtual),
   _position(position),
   _size(size),
   _time(time),
   _timestepSize(timestepSize),
   _maximumFineGridTime(maximumFineGridTime),
   _minimumFineGridTimestep(minimumFineGridTimestep),
   _synchronizeFineGrids(synchronizeFineGrids),
   _willCoarsen(willCoarsen),
   _minimalNeighborTimeConstraint(minimalNeighborTimeConstraint),
   _constrainingNeighborIndex(constrainingNeighborIndex),
   _minimalLeafNeighborTimeConstraint(minimalLeafNeighborTimeConstraint),
   _minimalNeighborTime(minimalNeighborTime),
   _maximalNeighborTimestep(maximalNeighborTimestep),
   _estimatedNextTimestepSize(estimatedNextTimestepSize),
   _skipGridIterations(skipGridIterations),
   _ageInGridIterations(ageInGridIterations),
   _demandedMeshWidth(demandedMeshWidth),
   _restrictionLowerBounds(restrictionLowerBounds),
   _restrictionUpperBounds(restrictionUpperBounds),
   _cellDescriptionIndex(cellDescriptionIndex),
   _uNewIndex(uNewIndex) {
      
   }
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::CellDescription::PersistentRecords::getSubdivisionFactor() const  {
      return _subdivisionFactor;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setSubdivisionFactor(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor)  {
      _subdivisionFactor = (subdivisionFactor);
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getGhostLayerWidth() const  {
      return _ghostLayerWidth;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setGhostLayerWidth(const int& ghostLayerWidth)  {
      _ghostLayerWidth = ghostLayerWidth;
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getUnknownsPerSubcell() const  {
      return _unknownsPerSubcell;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setUnknownsPerSubcell(const int& unknownsPerSubcell)  {
      _unknownsPerSubcell = unknownsPerSubcell;
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getAuxiliarFieldsPerSubcell() const  {
      return _auxiliarFieldsPerSubcell;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setAuxiliarFieldsPerSubcell(const int& auxiliarFieldsPerSubcell)  {
      _auxiliarFieldsPerSubcell = auxiliarFieldsPerSubcell;
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getLevel() const  {
      return _level;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setLevel(const int& level)  {
      _level = level;
   }
   
   
   
    bool peanoclaw::records::CellDescription::PersistentRecords::getIsVirtual() const  {
      return _isVirtual;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setIsVirtual(const bool& isVirtual)  {
      _isVirtual = isVirtual;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescription::PersistentRecords::getPosition() const  {
      return _position;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setPosition(const tarch::la::Vector<DIMENSIONS,double>& position)  {
      _position = (position);
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescription::PersistentRecords::getSize() const  {
      return _size;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setSize(const tarch::la::Vector<DIMENSIONS,double>& size)  {
      _size = (size);
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getTime() const  {
      return _time;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setTime(const double& time)  {
      _time = time;
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getTimestepSize() const  {
      return _timestepSize;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setTimestepSize(const double& timestepSize)  {
      _timestepSize = timestepSize;
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getMaximumFineGridTime() const  {
      return _maximumFineGridTime;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setMaximumFineGridTime(const double& maximumFineGridTime)  {
      _maximumFineGridTime = maximumFineGridTime;
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getMinimumFineGridTimestep() const  {
      return _minimumFineGridTimestep;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setMinimumFineGridTimestep(const double& minimumFineGridTimestep)  {
      _minimumFineGridTimestep = minimumFineGridTimestep;
   }
   
   
   
    bool peanoclaw::records::CellDescription::PersistentRecords::getSynchronizeFineGrids() const  {
      return _synchronizeFineGrids;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setSynchronizeFineGrids(const bool& synchronizeFineGrids)  {
      _synchronizeFineGrids = synchronizeFineGrids;
   }
   
   
   
    bool peanoclaw::records::CellDescription::PersistentRecords::getWillCoarsen() const  {
      return _willCoarsen;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setWillCoarsen(const bool& willCoarsen)  {
      _willCoarsen = willCoarsen;
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getMinimalNeighborTimeConstraint() const  {
      return _minimalNeighborTimeConstraint;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setMinimalNeighborTimeConstraint(const double& minimalNeighborTimeConstraint)  {
      _minimalNeighborTimeConstraint = minimalNeighborTimeConstraint;
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getConstrainingNeighborIndex() const  {
      return _constrainingNeighborIndex;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setConstrainingNeighborIndex(const int& constrainingNeighborIndex)  {
      _constrainingNeighborIndex = constrainingNeighborIndex;
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getMinimalLeafNeighborTimeConstraint() const  {
      return _minimalLeafNeighborTimeConstraint;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setMinimalLeafNeighborTimeConstraint(const double& minimalLeafNeighborTimeConstraint)  {
      _minimalLeafNeighborTimeConstraint = minimalLeafNeighborTimeConstraint;
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getMinimalNeighborTime() const  {
      return _minimalNeighborTime;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setMinimalNeighborTime(const double& minimalNeighborTime)  {
      _minimalNeighborTime = minimalNeighborTime;
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getMaximalNeighborTimestep() const  {
      return _maximalNeighborTimestep;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setMaximalNeighborTimestep(const double& maximalNeighborTimestep)  {
      _maximalNeighborTimestep = maximalNeighborTimestep;
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getEstimatedNextTimestepSize() const  {
      return _estimatedNextTimestepSize;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setEstimatedNextTimestepSize(const double& estimatedNextTimestepSize)  {
      _estimatedNextTimestepSize = estimatedNextTimestepSize;
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getSkipGridIterations() const  {
      return _skipGridIterations;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setSkipGridIterations(const int& skipGridIterations)  {
      _skipGridIterations = skipGridIterations;
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getAgeInGridIterations() const  {
      return _ageInGridIterations;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setAgeInGridIterations(const int& ageInGridIterations)  {
      _ageInGridIterations = ageInGridIterations;
   }
   
   
   
    double peanoclaw::records::CellDescription::PersistentRecords::getDemandedMeshWidth() const  {
      return _demandedMeshWidth;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setDemandedMeshWidth(const double& demandedMeshWidth)  {
      _demandedMeshWidth = demandedMeshWidth;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescription::PersistentRecords::getRestrictionLowerBounds() const  {
      return _restrictionLowerBounds;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setRestrictionLowerBounds(const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds)  {
      _restrictionLowerBounds = (restrictionLowerBounds);
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescription::PersistentRecords::getRestrictionUpperBounds() const  {
      return _restrictionUpperBounds;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setRestrictionUpperBounds(const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds)  {
      _restrictionUpperBounds = (restrictionUpperBounds);
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getCellDescriptionIndex() const  {
      return _cellDescriptionIndex;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setCellDescriptionIndex(const int& cellDescriptionIndex)  {
      _cellDescriptionIndex = cellDescriptionIndex;
   }
   
   
   
    int peanoclaw::records::CellDescription::PersistentRecords::getUNewIndex() const  {
      return _uNewIndex;
   }
   
   
   
    void peanoclaw::records::CellDescription::PersistentRecords::setUNewIndex(const int& uNewIndex)  {
      _uNewIndex = uNewIndex;
   }
   
   
   peanoclaw::records::CellDescription::CellDescription() {
      
   }
   
   
   peanoclaw::records::CellDescription::CellDescription(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._subdivisionFactor, persistentRecords._ghostLayerWidth, persistentRecords._unknownsPerSubcell, persistentRecords._auxiliarFieldsPerSubcell, persistentRecords._level, persistentRecords._isVirtual, persistentRecords._position, persistentRecords._size, persistentRecords._time, persistentRecords._timestepSize, persistentRecords._maximumFineGridTime, persistentRecords._minimumFineGridTimestep, persistentRecords._synchronizeFineGrids, persistentRecords._willCoarsen, persistentRecords._minimalNeighborTimeConstraint, persistentRecords._constrainingNeighborIndex, persistentRecords._minimalLeafNeighborTimeConstraint, persistentRecords._minimalNeighborTime, persistentRecords._maximalNeighborTimestep, persistentRecords._estimatedNextTimestepSize, persistentRecords._skipGridIterations, persistentRecords._ageInGridIterations, persistentRecords._demandedMeshWidth, persistentRecords._restrictionLowerBounds, persistentRecords._restrictionUpperBounds, persistentRecords._cellDescriptionIndex, persistentRecords._uNewIndex) {
      
   }
   
   
   peanoclaw::records::CellDescription::CellDescription(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const int& level, const bool& isVirtual, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& maximumFineGridTime, const double& minimumFineGridTimestep, const bool& synchronizeFineGrids, const bool& willCoarsen, const double& minimalNeighborTimeConstraint, const int& constrainingNeighborIndex, const double& minimalLeafNeighborTimeConstraint, const double& minimalNeighborTime, const double& maximalNeighborTimestep, const double& estimatedNextTimestepSize, const int& skipGridIterations, const int& ageInGridIterations, const double& demandedMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds, const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds, const int& cellDescriptionIndex, const int& uNewIndex):
   _persistentRecords(subdivisionFactor, ghostLayerWidth, unknownsPerSubcell, auxiliarFieldsPerSubcell, level, isVirtual, position, size, time, timestepSize, maximumFineGridTime, minimumFineGridTimestep, synchronizeFineGrids, willCoarsen, minimalNeighborTimeConstraint, constrainingNeighborIndex, minimalLeafNeighborTimeConstraint, minimalNeighborTime, maximalNeighborTimestep, estimatedNextTimestepSize, skipGridIterations, ageInGridIterations, demandedMeshWidth, restrictionLowerBounds, restrictionUpperBounds, cellDescriptionIndex, uNewIndex) {
      
   }
   
   
   peanoclaw::records::CellDescription::~CellDescription() { }
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::CellDescription::getSubdivisionFactor() const  {
      return _persistentRecords._subdivisionFactor;
   }
   
   
   
    void peanoclaw::records::CellDescription::setSubdivisionFactor(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor)  {
      _persistentRecords._subdivisionFactor = (subdivisionFactor);
   }
   
   
   
    int peanoclaw::records::CellDescription::getSubdivisionFactor(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._subdivisionFactor[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescription::setSubdivisionFactor(int elementIndex, const int& subdivisionFactor)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._subdivisionFactor[elementIndex]= subdivisionFactor;
      
   }
   
   
   
    int peanoclaw::records::CellDescription::getGhostLayerWidth() const  {
      return _persistentRecords._ghostLayerWidth;
   }
   
   
   
    void peanoclaw::records::CellDescription::setGhostLayerWidth(const int& ghostLayerWidth)  {
      _persistentRecords._ghostLayerWidth = ghostLayerWidth;
   }
   
   
   
    int peanoclaw::records::CellDescription::getUnknownsPerSubcell() const  {
      return _persistentRecords._unknownsPerSubcell;
   }
   
   
   
    void peanoclaw::records::CellDescription::setUnknownsPerSubcell(const int& unknownsPerSubcell)  {
      _persistentRecords._unknownsPerSubcell = unknownsPerSubcell;
   }
   
   
   
    int peanoclaw::records::CellDescription::getAuxiliarFieldsPerSubcell() const  {
      return _persistentRecords._auxiliarFieldsPerSubcell;
   }
   
   
   
    void peanoclaw::records::CellDescription::setAuxiliarFieldsPerSubcell(const int& auxiliarFieldsPerSubcell)  {
      _persistentRecords._auxiliarFieldsPerSubcell = auxiliarFieldsPerSubcell;
   }
   
   
   
    int peanoclaw::records::CellDescription::getLevel() const  {
      return _persistentRecords._level;
   }
   
   
   
    void peanoclaw::records::CellDescription::setLevel(const int& level)  {
      _persistentRecords._level = level;
   }
   
   
   
    bool peanoclaw::records::CellDescription::getIsVirtual() const  {
      return _persistentRecords._isVirtual;
   }
   
   
   
    void peanoclaw::records::CellDescription::setIsVirtual(const bool& isVirtual)  {
      _persistentRecords._isVirtual = isVirtual;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescription::getPosition() const  {
      return _persistentRecords._position;
   }
   
   
   
    void peanoclaw::records::CellDescription::setPosition(const tarch::la::Vector<DIMENSIONS,double>& position)  {
      _persistentRecords._position = (position);
   }
   
   
   
    double peanoclaw::records::CellDescription::getPosition(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._position[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescription::setPosition(int elementIndex, const double& position)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._position[elementIndex]= position;
      
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescription::getSize() const  {
      return _persistentRecords._size;
   }
   
   
   
    void peanoclaw::records::CellDescription::setSize(const tarch::la::Vector<DIMENSIONS,double>& size)  {
      _persistentRecords._size = (size);
   }
   
   
   
    double peanoclaw::records::CellDescription::getSize(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._size[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescription::setSize(int elementIndex, const double& size)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._size[elementIndex]= size;
      
   }
   
   
   
    double peanoclaw::records::CellDescription::getTime() const  {
      return _persistentRecords._time;
   }
   
   
   
    void peanoclaw::records::CellDescription::setTime(const double& time)  {
      _persistentRecords._time = time;
   }
   
   
   
    double peanoclaw::records::CellDescription::getTimestepSize() const  {
      return _persistentRecords._timestepSize;
   }
   
   
   
    void peanoclaw::records::CellDescription::setTimestepSize(const double& timestepSize)  {
      _persistentRecords._timestepSize = timestepSize;
   }
   
   
   
    double peanoclaw::records::CellDescription::getMaximumFineGridTime() const  {
      return _persistentRecords._maximumFineGridTime;
   }
   
   
   
    void peanoclaw::records::CellDescription::setMaximumFineGridTime(const double& maximumFineGridTime)  {
      _persistentRecords._maximumFineGridTime = maximumFineGridTime;
   }
   
   
   
    double peanoclaw::records::CellDescription::getMinimumFineGridTimestep() const  {
      return _persistentRecords._minimumFineGridTimestep;
   }
   
   
   
    void peanoclaw::records::CellDescription::setMinimumFineGridTimestep(const double& minimumFineGridTimestep)  {
      _persistentRecords._minimumFineGridTimestep = minimumFineGridTimestep;
   }
   
   
   
    bool peanoclaw::records::CellDescription::getSynchronizeFineGrids() const  {
      return _persistentRecords._synchronizeFineGrids;
   }
   
   
   
    void peanoclaw::records::CellDescription::setSynchronizeFineGrids(const bool& synchronizeFineGrids)  {
      _persistentRecords._synchronizeFineGrids = synchronizeFineGrids;
   }
   
   
   
    bool peanoclaw::records::CellDescription::getWillCoarsen() const  {
      return _persistentRecords._willCoarsen;
   }
   
   
   
    void peanoclaw::records::CellDescription::setWillCoarsen(const bool& willCoarsen)  {
      _persistentRecords._willCoarsen = willCoarsen;
   }
   
   
   
    double peanoclaw::records::CellDescription::getMinimalNeighborTimeConstraint() const  {
      return _persistentRecords._minimalNeighborTimeConstraint;
   }
   
   
   
    void peanoclaw::records::CellDescription::setMinimalNeighborTimeConstraint(const double& minimalNeighborTimeConstraint)  {
      _persistentRecords._minimalNeighborTimeConstraint = minimalNeighborTimeConstraint;
   }
   
   
   
    int peanoclaw::records::CellDescription::getConstrainingNeighborIndex() const  {
      return _persistentRecords._constrainingNeighborIndex;
   }
   
   
   
    void peanoclaw::records::CellDescription::setConstrainingNeighborIndex(const int& constrainingNeighborIndex)  {
      _persistentRecords._constrainingNeighborIndex = constrainingNeighborIndex;
   }
   
   
   
    double peanoclaw::records::CellDescription::getMinimalLeafNeighborTimeConstraint() const  {
      return _persistentRecords._minimalLeafNeighborTimeConstraint;
   }
   
   
   
    void peanoclaw::records::CellDescription::setMinimalLeafNeighborTimeConstraint(const double& minimalLeafNeighborTimeConstraint)  {
      _persistentRecords._minimalLeafNeighborTimeConstraint = minimalLeafNeighborTimeConstraint;
   }
   
   
   
    double peanoclaw::records::CellDescription::getMinimalNeighborTime() const  {
      return _persistentRecords._minimalNeighborTime;
   }
   
   
   
    void peanoclaw::records::CellDescription::setMinimalNeighborTime(const double& minimalNeighborTime)  {
      _persistentRecords._minimalNeighborTime = minimalNeighborTime;
   }
   
   
   
    double peanoclaw::records::CellDescription::getMaximalNeighborTimestep() const  {
      return _persistentRecords._maximalNeighborTimestep;
   }
   
   
   
    void peanoclaw::records::CellDescription::setMaximalNeighborTimestep(const double& maximalNeighborTimestep)  {
      _persistentRecords._maximalNeighborTimestep = maximalNeighborTimestep;
   }
   
   
   
    double peanoclaw::records::CellDescription::getEstimatedNextTimestepSize() const  {
      return _persistentRecords._estimatedNextTimestepSize;
   }
   
   
   
    void peanoclaw::records::CellDescription::setEstimatedNextTimestepSize(const double& estimatedNextTimestepSize)  {
      _persistentRecords._estimatedNextTimestepSize = estimatedNextTimestepSize;
   }
   
   
   
    int peanoclaw::records::CellDescription::getSkipGridIterations() const  {
      return _persistentRecords._skipGridIterations;
   }
   
   
   
    void peanoclaw::records::CellDescription::setSkipGridIterations(const int& skipGridIterations)  {
      _persistentRecords._skipGridIterations = skipGridIterations;
   }
   
   
   
    int peanoclaw::records::CellDescription::getAgeInGridIterations() const  {
      return _persistentRecords._ageInGridIterations;
   }
   
   
   
    void peanoclaw::records::CellDescription::setAgeInGridIterations(const int& ageInGridIterations)  {
      _persistentRecords._ageInGridIterations = ageInGridIterations;
   }
   
   
   
    double peanoclaw::records::CellDescription::getDemandedMeshWidth() const  {
      return _persistentRecords._demandedMeshWidth;
   }
   
   
   
    void peanoclaw::records::CellDescription::setDemandedMeshWidth(const double& demandedMeshWidth)  {
      _persistentRecords._demandedMeshWidth = demandedMeshWidth;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescription::getRestrictionLowerBounds() const  {
      return _persistentRecords._restrictionLowerBounds;
   }
   
   
   
    void peanoclaw::records::CellDescription::setRestrictionLowerBounds(const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds)  {
      _persistentRecords._restrictionLowerBounds = (restrictionLowerBounds);
   }
   
   
   
    double peanoclaw::records::CellDescription::getRestrictionLowerBounds(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._restrictionLowerBounds[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescription::setRestrictionLowerBounds(int elementIndex, const double& restrictionLowerBounds)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._restrictionLowerBounds[elementIndex]= restrictionLowerBounds;
      
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescription::getRestrictionUpperBounds() const  {
      return _persistentRecords._restrictionUpperBounds;
   }
   
   
   
    void peanoclaw::records::CellDescription::setRestrictionUpperBounds(const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds)  {
      _persistentRecords._restrictionUpperBounds = (restrictionUpperBounds);
   }
   
   
   
    double peanoclaw::records::CellDescription::getRestrictionUpperBounds(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._restrictionUpperBounds[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescription::setRestrictionUpperBounds(int elementIndex, const double& restrictionUpperBounds)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._restrictionUpperBounds[elementIndex]= restrictionUpperBounds;
      
   }
   
   
   
    int peanoclaw::records::CellDescription::getCellDescriptionIndex() const  {
      return _persistentRecords._cellDescriptionIndex;
   }
   
   
   
    void peanoclaw::records::CellDescription::setCellDescriptionIndex(const int& cellDescriptionIndex)  {
      _persistentRecords._cellDescriptionIndex = cellDescriptionIndex;
   }
   
   
   
    int peanoclaw::records::CellDescription::getUNewIndex() const  {
      return _persistentRecords._uNewIndex;
   }
   
   
   
    void peanoclaw::records::CellDescription::setUNewIndex(const int& uNewIndex)  {
      _persistentRecords._uNewIndex = uNewIndex;
   }
   
   
   
   
   std::string peanoclaw::records::CellDescription::toString() const {
      std::ostringstream stringstr;
      toString(stringstr);
      return stringstr.str();
   }
   
   void peanoclaw::records::CellDescription::toString (std::ostream& out) const {
      out << "("; 
      out << "subdivisionFactor:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getSubdivisionFactor(i) << ",";
   }
   out << getSubdivisionFactor(DIMENSIONS-1) << "]";
      out << ",";
      out << "ghostLayerWidth:" << getGhostLayerWidth();
      out << ",";
      out << "unknownsPerSubcell:" << getUnknownsPerSubcell();
      out << ",";
      out << "auxiliarFieldsPerSubcell:" << getAuxiliarFieldsPerSubcell();
      out << ",";
      out << "level:" << getLevel();
      out << ",";
      out << "isVirtual:" << getIsVirtual();
      out << ",";
      out << "position:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getPosition(i) << ",";
   }
   out << getPosition(DIMENSIONS-1) << "]";
      out << ",";
      out << "size:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getSize(i) << ",";
   }
   out << getSize(DIMENSIONS-1) << "]";
      out << ",";
      out << "time:" << getTime();
      out << ",";
      out << "timestepSize:" << getTimestepSize();
      out << ",";
      out << "maximumFineGridTime:" << getMaximumFineGridTime();
      out << ",";
      out << "minimumFineGridTimestep:" << getMinimumFineGridTimestep();
      out << ",";
      out << "synchronizeFineGrids:" << getSynchronizeFineGrids();
      out << ",";
      out << "willCoarsen:" << getWillCoarsen();
      out << ",";
      out << "minimalNeighborTimeConstraint:" << getMinimalNeighborTimeConstraint();
      out << ",";
      out << "constrainingNeighborIndex:" << getConstrainingNeighborIndex();
      out << ",";
      out << "minimalLeafNeighborTimeConstraint:" << getMinimalLeafNeighborTimeConstraint();
      out << ",";
      out << "minimalNeighborTime:" << getMinimalNeighborTime();
      out << ",";
      out << "maximalNeighborTimestep:" << getMaximalNeighborTimestep();
      out << ",";
      out << "estimatedNextTimestepSize:" << getEstimatedNextTimestepSize();
      out << ",";
      out << "skipGridIterations:" << getSkipGridIterations();
      out << ",";
      out << "ageInGridIterations:" << getAgeInGridIterations();
      out << ",";
      out << "demandedMeshWidth:" << getDemandedMeshWidth();
      out << ",";
      out << "restrictionLowerBounds:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getRestrictionLowerBounds(i) << ",";
   }
   out << getRestrictionLowerBounds(DIMENSIONS-1) << "]";
      out << ",";
      out << "restrictionUpperBounds:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getRestrictionUpperBounds(i) << ",";
   }
   out << getRestrictionUpperBounds(DIMENSIONS-1) << "]";
      out << ",";
      out << "cellDescriptionIndex:" << getCellDescriptionIndex();
      out << ",";
      out << "uNewIndex:" << getUNewIndex();
      out <<  ")";
   }
   
   
   peanoclaw::records::CellDescription::PersistentRecords peanoclaw::records::CellDescription::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::CellDescriptionPacked peanoclaw::records::CellDescription::convert() const{
      return CellDescriptionPacked(
         getSubdivisionFactor(),
         getGhostLayerWidth(),
         getUnknownsPerSubcell(),
         getAuxiliarFieldsPerSubcell(),
         getLevel(),
         getIsVirtual(),
         getPosition(),
         getSize(),
         getTime(),
         getTimestepSize(),
         getMaximumFineGridTime(),
         getMinimumFineGridTimestep(),
         getSynchronizeFineGrids(),
         getWillCoarsen(),
         getMinimalNeighborTimeConstraint(),
         getConstrainingNeighborIndex(),
         getMinimalLeafNeighborTimeConstraint(),
         getMinimalNeighborTime(),
         getMaximalNeighborTimestep(),
         getEstimatedNextTimestepSize(),
         getSkipGridIterations(),
         getAgeInGridIterations(),
         getDemandedMeshWidth(),
         getRestrictionLowerBounds(),
         getRestrictionUpperBounds(),
         getCellDescriptionIndex(),
         getUNewIndex()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::CellDescription::_log( "peanoclaw::records::CellDescription" );
      
      MPI_Datatype peanoclaw::records::CellDescription::Datatype = 0;
      MPI_Datatype peanoclaw::records::CellDescription::FullDatatype = 0;
      
      
      void peanoclaw::records::CellDescription::initDatatype() {
         {
            CellDescription dummyCellDescription[2];
            
            const int Attributes = 27;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //subdivisionFactor
               MPI_INT,		 //ghostLayerWidth
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_INT,		 //level
               MPI_CHAR,		 //isVirtual
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
               MPI_DOUBLE,		 //maximumFineGridTime
               MPI_DOUBLE,		 //minimumFineGridTimestep
               MPI_CHAR,		 //synchronizeFineGrids
               MPI_CHAR,		 //willCoarsen
               MPI_DOUBLE,		 //minimalNeighborTimeConstraint
               MPI_DOUBLE,		 //minimalLeafNeighborTimeConstraint
               MPI_DOUBLE,		 //minimalNeighborTime
               MPI_DOUBLE,		 //maximalNeighborTimestep
               MPI_DOUBLE,		 //estimatedNextTimestepSize
               MPI_INT,		 //skipGridIterations
               MPI_INT,		 //ageInGridIterations
               MPI_DOUBLE,		 //demandedMeshWidth
               MPI_DOUBLE,		 //restrictionLowerBounds
               MPI_DOUBLE,		 //restrictionUpperBounds
               MPI_INT,		 //cellDescriptionIndex
               MPI_INT,		 //uNewIndex
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               DIMENSIONS,		 //subdivisionFactor
               1,		 //ghostLayerWidth
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               1,		 //level
               1,		 //isVirtual
               DIMENSIONS,		 //position
               DIMENSIONS,		 //size
               1,		 //time
               1,		 //timestepSize
               1,		 //maximumFineGridTime
               1,		 //minimumFineGridTimestep
               1,		 //synchronizeFineGrids
               1,		 //willCoarsen
               1,		 //minimalNeighborTimeConstraint
               1,		 //minimalLeafNeighborTimeConstraint
               1,		 //minimalNeighborTime
               1,		 //maximalNeighborTimestep
               1,		 //estimatedNextTimestepSize
               1,		 //skipGridIterations
               1,		 //ageInGridIterations
               1,		 //demandedMeshWidth
               DIMENSIONS,		 //restrictionLowerBounds
               DIMENSIONS,		 //restrictionUpperBounds
               1,		 //cellDescriptionIndex
               1,		 //uNewIndex
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._ghostLayerWidth))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._unknownsPerSubcell))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._level))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._isVirtual))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._position[0]))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._size[0]))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._time))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._timestepSize))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._maximumFineGridTime))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimumFineGridTimestep))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._synchronizeFineGrids))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._willCoarsen))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalNeighborTimeConstraint))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalLeafNeighborTimeConstraint))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalNeighborTime))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._maximalNeighborTimestep))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._estimatedNextTimestepSize))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._skipGridIterations))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._ageInGridIterations))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._demandedMeshWidth))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._restrictionLowerBounds[0]))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._restrictionUpperBounds[0]))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._cellDescriptionIndex))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._uNewIndex))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyCellDescription[1]._persistentRecords._subdivisionFactor[0])), 		&disp[26] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellDescription::Datatype );
            MPI_Type_commit( &CellDescription::Datatype );
            
         }
         {
            CellDescription dummyCellDescription[2];
            
            const int Attributes = 28;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //subdivisionFactor
               MPI_INT,		 //ghostLayerWidth
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_INT,		 //level
               MPI_CHAR,		 //isVirtual
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
               MPI_DOUBLE,		 //maximumFineGridTime
               MPI_DOUBLE,		 //minimumFineGridTimestep
               MPI_CHAR,		 //synchronizeFineGrids
               MPI_CHAR,		 //willCoarsen
               MPI_DOUBLE,		 //minimalNeighborTimeConstraint
               MPI_INT,		 //constrainingNeighborIndex
               MPI_DOUBLE,		 //minimalLeafNeighborTimeConstraint
               MPI_DOUBLE,		 //minimalNeighborTime
               MPI_DOUBLE,		 //maximalNeighborTimestep
               MPI_DOUBLE,		 //estimatedNextTimestepSize
               MPI_INT,		 //skipGridIterations
               MPI_INT,		 //ageInGridIterations
               MPI_DOUBLE,		 //demandedMeshWidth
               MPI_DOUBLE,		 //restrictionLowerBounds
               MPI_DOUBLE,		 //restrictionUpperBounds
               MPI_INT,		 //cellDescriptionIndex
               MPI_INT,		 //uNewIndex
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               DIMENSIONS,		 //subdivisionFactor
               1,		 //ghostLayerWidth
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               1,		 //level
               1,		 //isVirtual
               DIMENSIONS,		 //position
               DIMENSIONS,		 //size
               1,		 //time
               1,		 //timestepSize
               1,		 //maximumFineGridTime
               1,		 //minimumFineGridTimestep
               1,		 //synchronizeFineGrids
               1,		 //willCoarsen
               1,		 //minimalNeighborTimeConstraint
               1,		 //constrainingNeighborIndex
               1,		 //minimalLeafNeighborTimeConstraint
               1,		 //minimalNeighborTime
               1,		 //maximalNeighborTimestep
               1,		 //estimatedNextTimestepSize
               1,		 //skipGridIterations
               1,		 //ageInGridIterations
               1,		 //demandedMeshWidth
               DIMENSIONS,		 //restrictionLowerBounds
               DIMENSIONS,		 //restrictionUpperBounds
               1,		 //cellDescriptionIndex
               1,		 //uNewIndex
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._ghostLayerWidth))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._unknownsPerSubcell))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._level))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._isVirtual))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._position[0]))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._size[0]))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._time))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._timestepSize))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._maximumFineGridTime))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimumFineGridTimestep))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._synchronizeFineGrids))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._willCoarsen))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalNeighborTimeConstraint))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._constrainingNeighborIndex))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalLeafNeighborTimeConstraint))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalNeighborTime))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._maximalNeighborTimestep))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._estimatedNextTimestepSize))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._skipGridIterations))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._ageInGridIterations))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._demandedMeshWidth))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._restrictionLowerBounds[0]))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._restrictionUpperBounds[0]))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._cellDescriptionIndex))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._uNewIndex))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyCellDescription[1]._persistentRecords._subdivisionFactor[0])), 		&disp[27] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellDescription::FullDatatype );
            MPI_Type_commit( &CellDescription::FullDatatype );
            
         }
         
      }
      
      
      void peanoclaw::records::CellDescription::shutdownDatatype() {
         MPI_Type_free( &CellDescription::Datatype );
         MPI_Type_free( &CellDescription::FullDatatype );
         
      }
      
      void peanoclaw::records::CellDescription::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
         MPI_Request* sendRequestHandle = new MPI_Request();
         MPI_Status   status;
         int          flag = 0;
         int          result;
         
         clock_t      timeOutWarning   = -1;
         clock_t      timeOutShutdown  = -1;
         bool         triggeredTimeoutWarning = false;
         
         if (exchangeOnlyAttributesMarkedWithParallelise) {
            result = MPI_Isend(
               this, 1, Datatype, destination,
               tag, tarch::parallel::Node::getInstance().getCommunicator(),
               sendRequestHandle
            );
            
         }
         else {
            result = MPI_Isend(
               this, 1, FullDatatype, destination,
               tag, tarch::parallel::Node::getInstance().getCommunicator(),
               sendRequestHandle
            );
            
         }
         if  (result!=MPI_SUCCESS) {
            std::ostringstream msg;
            msg << "was not able to send message peanoclaw::records::CellDescription "
            << toString()
            << " to node " << destination
            << ": " << tarch::parallel::MPIReturnValueToString(result);
            _log.error( "send(int)",msg.str() );
         }
         result = MPI_Test( sendRequestHandle, &flag, &status );
         while (!flag) {
            if (timeOutWarning==-1)   timeOutWarning   = tarch::parallel::Node::getInstance().getDeadlockWarningTimeStamp();
            if (timeOutShutdown==-1)  timeOutShutdown  = tarch::parallel::Node::getInstance().getDeadlockTimeOutTimeStamp();
            result = MPI_Test( sendRequestHandle, &flag, &status );
            if (result!=MPI_SUCCESS) {
               std::ostringstream msg;
               msg << "testing for finished send task for peanoclaw::records::CellDescription "
               << toString()
               << " sent to node " << destination
               << " failed: " << tarch::parallel::MPIReturnValueToString(result);
               _log.error("send(int)", msg.str() );
            }
            
            // deadlock aspect
            if (
               tarch::parallel::Node::getInstance().isTimeOutWarningEnabled() &&
               (clock()>timeOutWarning) &&
               (!triggeredTimeoutWarning)
            ) {
               tarch::parallel::Node::getInstance().writeTimeOutWarning(
               "peanoclaw::records::CellDescription",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::CellDescription",
               "send(int)", destination,tag,1
               );
            }
            tarch::parallel::Node::getInstance().receiveDanglingMessages();
         }
         
         delete sendRequestHandle;
         #ifdef Debug
         _log.debug("send(int,int)", "sent " + toString() );
         #endif
         
      }
      
      
      
      void peanoclaw::records::CellDescription::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
         MPI_Request* sendRequestHandle = new MPI_Request();
         MPI_Status   status;
         int          flag = 0;
         int          result;
         
         clock_t      timeOutWarning   = -1;
         clock_t      timeOutShutdown  = -1;
         bool         triggeredTimeoutWarning = false;
         
         if (exchangeOnlyAttributesMarkedWithParallelise) {
            result = MPI_Irecv(
               this, 1, Datatype, source, tag,
               tarch::parallel::Node::getInstance().getCommunicator(), sendRequestHandle
            );
            
         }
         else {
            result = MPI_Irecv(
               this, 1, FullDatatype, source, tag,
               tarch::parallel::Node::getInstance().getCommunicator(), sendRequestHandle
            );
            
         }
         if ( result != MPI_SUCCESS ) {
            std::ostringstream msg;
            msg << "failed to start to receive peanoclaw::records::CellDescription from node "
            << source << ": " << tarch::parallel::MPIReturnValueToString(result);
            _log.error( "receive(int)", msg.str() );
         }
         
         result = MPI_Test( sendRequestHandle, &flag, &status );
         while (!flag) {
            if (timeOutWarning==-1)   timeOutWarning   = tarch::parallel::Node::getInstance().getDeadlockWarningTimeStamp();
            if (timeOutShutdown==-1)  timeOutShutdown  = tarch::parallel::Node::getInstance().getDeadlockTimeOutTimeStamp();
            result = MPI_Test( sendRequestHandle, &flag, &status );
            if (result!=MPI_SUCCESS) {
               std::ostringstream msg;
               msg << "testing for finished receive task for peanoclaw::records::CellDescription failed: "
               << tarch::parallel::MPIReturnValueToString(result);
               _log.error("receive(int)", msg.str() );
            }
            
            // deadlock aspect
            if (
               tarch::parallel::Node::getInstance().isTimeOutWarningEnabled() &&
               (clock()>timeOutWarning) &&
               (!triggeredTimeoutWarning)
            ) {
               tarch::parallel::Node::getInstance().writeTimeOutWarning(
               "peanoclaw::records::CellDescription",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::CellDescription",
               "receive(int)", source,tag,1
               );
            }
            tarch::parallel::Node::getInstance().receiveDanglingMessages();
         }
         
         delete sendRequestHandle;
         
         #ifdef Debug
         _log.debug("receive(int,int)", "received " + toString() ); 
         #endif
         
      }
      
      
      
      bool peanoclaw::records::CellDescription::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
         MPI_Status status;
         int  flag        = 0;
         MPI_Iprobe(
            MPI_ANY_SOURCE, tag,
            tarch::parallel::Node::getInstance().getCommunicator(), &flag, &status
         );
         if (flag) {
            int  messageCounter;
            if (exchangeOnlyAttributesMarkedWithParallelise) {
               MPI_Get_count(&status, Datatype, &messageCounter);
            }
            else {
               MPI_Get_count(&status, FullDatatype, &messageCounter);
            }
            return messageCounter > 0;
         }
         else return false;
         
      }
      
      
   #endif
   
   
   peanoclaw::records::CellDescriptionPacked::PersistentRecords::PersistentRecords() {
      assertion((23 < (8 * sizeof(int))));
      
   }
   
   
   peanoclaw::records::CellDescriptionPacked::PersistentRecords::PersistentRecords(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const int& level, const bool& isVirtual, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& maximumFineGridTime, const double& minimumFineGridTimestep, const bool& synchronizeFineGrids, const bool& willCoarsen, const double& minimalNeighborTimeConstraint, const int& constrainingNeighborIndex, const double& minimalLeafNeighborTimeConstraint, const double& minimalNeighborTime, const double& maximalNeighborTimestep, const double& estimatedNextTimestepSize, const int& skipGridIterations, const int& ageInGridIterations, const double& demandedMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds, const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds, const int& cellDescriptionIndex, const int& uNewIndex):
   _subdivisionFactor(subdivisionFactor),
   _position(position),
   _size(size),
   _time(time),
   _timestepSize(timestepSize),
   _maximumFineGridTime(maximumFineGridTime),
   _minimumFineGridTimestep(minimumFineGridTimestep),
   _minimalNeighborTimeConstraint(minimalNeighborTimeConstraint),
   _constrainingNeighborIndex(constrainingNeighborIndex),
   _minimalLeafNeighborTimeConstraint(minimalLeafNeighborTimeConstraint),
   _minimalNeighborTime(minimalNeighborTime),
   _maximalNeighborTimestep(maximalNeighborTimestep),
   _estimatedNextTimestepSize(estimatedNextTimestepSize),
   _ageInGridIterations(ageInGridIterations),
   _demandedMeshWidth(demandedMeshWidth),
   _restrictionLowerBounds(restrictionLowerBounds),
   _restrictionUpperBounds(restrictionUpperBounds),
   _cellDescriptionIndex(cellDescriptionIndex),
   _uNewIndex(uNewIndex) {
      setGhostLayerWidth(ghostLayerWidth);
      setUnknownsPerSubcell(unknownsPerSubcell);
      setAuxiliarFieldsPerSubcell(auxiliarFieldsPerSubcell);
      setLevel(level);
      setIsVirtual(isVirtual);
      setSynchronizeFineGrids(synchronizeFineGrids);
      setWillCoarsen(willCoarsen);
      setSkipGridIterations(skipGridIterations);
      assertion((23 < (8 * sizeof(int))));
      
   }
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::CellDescriptionPacked::PersistentRecords::getSubdivisionFactor() const  {
      return _subdivisionFactor;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setSubdivisionFactor(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor)  {
      _subdivisionFactor = (subdivisionFactor);
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getGhostLayerWidth() const  {
      int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (0));
   int tmp = static_cast<int>(_packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (0));
   assertion(( tmp >= 0 &&  tmp <= 15));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setGhostLayerWidth(const int& ghostLayerWidth)  {
      assertion((ghostLayerWidth >= 0 && ghostLayerWidth <= 15));
   int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (0));
   _packedRecords0 = static_cast<int>(_packedRecords0 & ~mask);
   _packedRecords0 = static_cast<int>(_packedRecords0 | ghostLayerWidth << (0));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getUnknownsPerSubcell() const  {
      int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (4));
   int tmp = static_cast<int>(_packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (4));
   assertion(( tmp >= 0 &&  tmp <= 15));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setUnknownsPerSubcell(const int& unknownsPerSubcell)  {
      assertion((unknownsPerSubcell >= 0 && unknownsPerSubcell <= 15));
   int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (4));
   _packedRecords0 = static_cast<int>(_packedRecords0 & ~mask);
   _packedRecords0 = static_cast<int>(_packedRecords0 | unknownsPerSubcell << (4));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getAuxiliarFieldsPerSubcell() const  {
      int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (8));
   int tmp = static_cast<int>(_packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (8));
   assertion(( tmp >= 0 &&  tmp <= 15));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setAuxiliarFieldsPerSubcell(const int& auxiliarFieldsPerSubcell)  {
      assertion((auxiliarFieldsPerSubcell >= 0 && auxiliarFieldsPerSubcell <= 15));
   int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (8));
   _packedRecords0 = static_cast<int>(_packedRecords0 & ~mask);
   _packedRecords0 = static_cast<int>(_packedRecords0 | auxiliarFieldsPerSubcell << (8));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getLevel() const  {
      int mask =  (1 << (5)) - 1;
   mask = static_cast<int>(mask << (12));
   int tmp = static_cast<int>(_packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (12));
   assertion(( tmp >= 0 &&  tmp <= 31));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setLevel(const int& level)  {
      assertion((level >= 0 && level <= 31));
   int mask =  (1 << (5)) - 1;
   mask = static_cast<int>(mask << (12));
   _packedRecords0 = static_cast<int>(_packedRecords0 & ~mask);
   _packedRecords0 = static_cast<int>(_packedRecords0 | level << (12));
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::PersistentRecords::getIsVirtual() const  {
      int mask = 1 << (17);
   int tmp = static_cast<int>(_packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setIsVirtual(const bool& isVirtual)  {
      int mask = 1 << (17);
   _packedRecords0 = static_cast<int>( isVirtual ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescriptionPacked::PersistentRecords::getPosition() const  {
      return _position;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setPosition(const tarch::la::Vector<DIMENSIONS,double>& position)  {
      _position = (position);
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescriptionPacked::PersistentRecords::getSize() const  {
      return _size;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setSize(const tarch::la::Vector<DIMENSIONS,double>& size)  {
      _size = (size);
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getTime() const  {
      return _time;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setTime(const double& time)  {
      _time = time;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getTimestepSize() const  {
      return _timestepSize;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setTimestepSize(const double& timestepSize)  {
      _timestepSize = timestepSize;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getMaximumFineGridTime() const  {
      return _maximumFineGridTime;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setMaximumFineGridTime(const double& maximumFineGridTime)  {
      _maximumFineGridTime = maximumFineGridTime;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getMinimumFineGridTimestep() const  {
      return _minimumFineGridTimestep;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setMinimumFineGridTimestep(const double& minimumFineGridTimestep)  {
      _minimumFineGridTimestep = minimumFineGridTimestep;
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::PersistentRecords::getSynchronizeFineGrids() const  {
      int mask = 1 << (18);
   int tmp = static_cast<int>(_packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setSynchronizeFineGrids(const bool& synchronizeFineGrids)  {
      int mask = 1 << (18);
   _packedRecords0 = static_cast<int>( synchronizeFineGrids ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::PersistentRecords::getWillCoarsen() const  {
      int mask = 1 << (19);
   int tmp = static_cast<int>(_packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setWillCoarsen(const bool& willCoarsen)  {
      int mask = 1 << (19);
   _packedRecords0 = static_cast<int>( willCoarsen ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getMinimalNeighborTimeConstraint() const  {
      return _minimalNeighborTimeConstraint;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setMinimalNeighborTimeConstraint(const double& minimalNeighborTimeConstraint)  {
      _minimalNeighborTimeConstraint = minimalNeighborTimeConstraint;
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getConstrainingNeighborIndex() const  {
      return _constrainingNeighborIndex;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setConstrainingNeighborIndex(const int& constrainingNeighborIndex)  {
      _constrainingNeighborIndex = constrainingNeighborIndex;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getMinimalLeafNeighborTimeConstraint() const  {
      return _minimalLeafNeighborTimeConstraint;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setMinimalLeafNeighborTimeConstraint(const double& minimalLeafNeighborTimeConstraint)  {
      _minimalLeafNeighborTimeConstraint = minimalLeafNeighborTimeConstraint;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getMinimalNeighborTime() const  {
      return _minimalNeighborTime;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setMinimalNeighborTime(const double& minimalNeighborTime)  {
      _minimalNeighborTime = minimalNeighborTime;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getMaximalNeighborTimestep() const  {
      return _maximalNeighborTimestep;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setMaximalNeighborTimestep(const double& maximalNeighborTimestep)  {
      _maximalNeighborTimestep = maximalNeighborTimestep;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getEstimatedNextTimestepSize() const  {
      return _estimatedNextTimestepSize;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setEstimatedNextTimestepSize(const double& estimatedNextTimestepSize)  {
      _estimatedNextTimestepSize = estimatedNextTimestepSize;
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getSkipGridIterations() const  {
      int mask =  (1 << (3)) - 1;
   mask = static_cast<int>(mask << (20));
   int tmp = static_cast<int>(_packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (20));
   assertion(( tmp >= 0 &&  tmp <= 7));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setSkipGridIterations(const int& skipGridIterations)  {
      assertion((skipGridIterations >= 0 && skipGridIterations <= 7));
   int mask =  (1 << (3)) - 1;
   mask = static_cast<int>(mask << (20));
   _packedRecords0 = static_cast<int>(_packedRecords0 & ~mask);
   _packedRecords0 = static_cast<int>(_packedRecords0 | skipGridIterations << (20));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getAgeInGridIterations() const  {
      return _ageInGridIterations;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setAgeInGridIterations(const int& ageInGridIterations)  {
      _ageInGridIterations = ageInGridIterations;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::PersistentRecords::getDemandedMeshWidth() const  {
      return _demandedMeshWidth;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setDemandedMeshWidth(const double& demandedMeshWidth)  {
      _demandedMeshWidth = demandedMeshWidth;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescriptionPacked::PersistentRecords::getRestrictionLowerBounds() const  {
      return _restrictionLowerBounds;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setRestrictionLowerBounds(const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds)  {
      _restrictionLowerBounds = (restrictionLowerBounds);
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescriptionPacked::PersistentRecords::getRestrictionUpperBounds() const  {
      return _restrictionUpperBounds;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setRestrictionUpperBounds(const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds)  {
      _restrictionUpperBounds = (restrictionUpperBounds);
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getCellDescriptionIndex() const  {
      return _cellDescriptionIndex;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setCellDescriptionIndex(const int& cellDescriptionIndex)  {
      _cellDescriptionIndex = cellDescriptionIndex;
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::PersistentRecords::getUNewIndex() const  {
      return _uNewIndex;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::PersistentRecords::setUNewIndex(const int& uNewIndex)  {
      _uNewIndex = uNewIndex;
   }
   
   
   peanoclaw::records::CellDescriptionPacked::CellDescriptionPacked() {
      assertion((23 < (8 * sizeof(int))));
      
   }
   
   
   peanoclaw::records::CellDescriptionPacked::CellDescriptionPacked(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._subdivisionFactor, persistentRecords.getGhostLayerWidth(), persistentRecords.getUnknownsPerSubcell(), persistentRecords.getAuxiliarFieldsPerSubcell(), persistentRecords.getLevel(), persistentRecords.getIsVirtual(), persistentRecords._position, persistentRecords._size, persistentRecords._time, persistentRecords._timestepSize, persistentRecords._maximumFineGridTime, persistentRecords._minimumFineGridTimestep, persistentRecords.getSynchronizeFineGrids(), persistentRecords.getWillCoarsen(), persistentRecords._minimalNeighborTimeConstraint, persistentRecords._constrainingNeighborIndex, persistentRecords._minimalLeafNeighborTimeConstraint, persistentRecords._minimalNeighborTime, persistentRecords._maximalNeighborTimestep, persistentRecords._estimatedNextTimestepSize, persistentRecords.getSkipGridIterations(), persistentRecords._ageInGridIterations, persistentRecords._demandedMeshWidth, persistentRecords._restrictionLowerBounds, persistentRecords._restrictionUpperBounds, persistentRecords._cellDescriptionIndex, persistentRecords._uNewIndex) {
      assertion((23 < (8 * sizeof(int))));
      
   }
   
   
   peanoclaw::records::CellDescriptionPacked::CellDescriptionPacked(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const int& level, const bool& isVirtual, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& maximumFineGridTime, const double& minimumFineGridTimestep, const bool& synchronizeFineGrids, const bool& willCoarsen, const double& minimalNeighborTimeConstraint, const int& constrainingNeighborIndex, const double& minimalLeafNeighborTimeConstraint, const double& minimalNeighborTime, const double& maximalNeighborTimestep, const double& estimatedNextTimestepSize, const int& skipGridIterations, const int& ageInGridIterations, const double& demandedMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds, const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds, const int& cellDescriptionIndex, const int& uNewIndex):
   _persistentRecords(subdivisionFactor, ghostLayerWidth, unknownsPerSubcell, auxiliarFieldsPerSubcell, level, isVirtual, position, size, time, timestepSize, maximumFineGridTime, minimumFineGridTimestep, synchronizeFineGrids, willCoarsen, minimalNeighborTimeConstraint, constrainingNeighborIndex, minimalLeafNeighborTimeConstraint, minimalNeighborTime, maximalNeighborTimestep, estimatedNextTimestepSize, skipGridIterations, ageInGridIterations, demandedMeshWidth, restrictionLowerBounds, restrictionUpperBounds, cellDescriptionIndex, uNewIndex) {
      assertion((23 < (8 * sizeof(int))));
      
   }
   
   
   peanoclaw::records::CellDescriptionPacked::~CellDescriptionPacked() { }
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::CellDescriptionPacked::getSubdivisionFactor() const  {
      return _persistentRecords._subdivisionFactor;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setSubdivisionFactor(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor)  {
      _persistentRecords._subdivisionFactor = (subdivisionFactor);
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getSubdivisionFactor(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._subdivisionFactor[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setSubdivisionFactor(int elementIndex, const int& subdivisionFactor)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._subdivisionFactor[elementIndex]= subdivisionFactor;
      
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getGhostLayerWidth() const  {
      int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (0));
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (0));
   assertion(( tmp >= 0 &&  tmp <= 15));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setGhostLayerWidth(const int& ghostLayerWidth)  {
      assertion((ghostLayerWidth >= 0 && ghostLayerWidth <= 15));
   int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (0));
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 & ~mask);
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 | ghostLayerWidth << (0));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getUnknownsPerSubcell() const  {
      int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (4));
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (4));
   assertion(( tmp >= 0 &&  tmp <= 15));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setUnknownsPerSubcell(const int& unknownsPerSubcell)  {
      assertion((unknownsPerSubcell >= 0 && unknownsPerSubcell <= 15));
   int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (4));
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 & ~mask);
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 | unknownsPerSubcell << (4));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getAuxiliarFieldsPerSubcell() const  {
      int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (8));
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (8));
   assertion(( tmp >= 0 &&  tmp <= 15));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setAuxiliarFieldsPerSubcell(const int& auxiliarFieldsPerSubcell)  {
      assertion((auxiliarFieldsPerSubcell >= 0 && auxiliarFieldsPerSubcell <= 15));
   int mask =  (1 << (4)) - 1;
   mask = static_cast<int>(mask << (8));
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 & ~mask);
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 | auxiliarFieldsPerSubcell << (8));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getLevel() const  {
      int mask =  (1 << (5)) - 1;
   mask = static_cast<int>(mask << (12));
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (12));
   assertion(( tmp >= 0 &&  tmp <= 31));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setLevel(const int& level)  {
      assertion((level >= 0 && level <= 31));
   int mask =  (1 << (5)) - 1;
   mask = static_cast<int>(mask << (12));
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 & ~mask);
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 | level << (12));
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::getIsVirtual() const  {
      int mask = 1 << (17);
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setIsVirtual(const bool& isVirtual)  {
      int mask = 1 << (17);
   _persistentRecords._packedRecords0 = static_cast<int>( isVirtual ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescriptionPacked::getPosition() const  {
      return _persistentRecords._position;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setPosition(const tarch::la::Vector<DIMENSIONS,double>& position)  {
      _persistentRecords._position = (position);
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getPosition(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._position[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setPosition(int elementIndex, const double& position)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._position[elementIndex]= position;
      
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescriptionPacked::getSize() const  {
      return _persistentRecords._size;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setSize(const tarch::la::Vector<DIMENSIONS,double>& size)  {
      _persistentRecords._size = (size);
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getSize(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._size[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setSize(int elementIndex, const double& size)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._size[elementIndex]= size;
      
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getTime() const  {
      return _persistentRecords._time;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setTime(const double& time)  {
      _persistentRecords._time = time;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getTimestepSize() const  {
      return _persistentRecords._timestepSize;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setTimestepSize(const double& timestepSize)  {
      _persistentRecords._timestepSize = timestepSize;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getMaximumFineGridTime() const  {
      return _persistentRecords._maximumFineGridTime;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setMaximumFineGridTime(const double& maximumFineGridTime)  {
      _persistentRecords._maximumFineGridTime = maximumFineGridTime;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getMinimumFineGridTimestep() const  {
      return _persistentRecords._minimumFineGridTimestep;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setMinimumFineGridTimestep(const double& minimumFineGridTimestep)  {
      _persistentRecords._minimumFineGridTimestep = minimumFineGridTimestep;
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::getSynchronizeFineGrids() const  {
      int mask = 1 << (18);
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setSynchronizeFineGrids(const bool& synchronizeFineGrids)  {
      int mask = 1 << (18);
   _persistentRecords._packedRecords0 = static_cast<int>( synchronizeFineGrids ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
   }
   
   
   
    bool peanoclaw::records::CellDescriptionPacked::getWillCoarsen() const  {
      int mask = 1 << (19);
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   return (tmp != 0);
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setWillCoarsen(const bool& willCoarsen)  {
      int mask = 1 << (19);
   _persistentRecords._packedRecords0 = static_cast<int>( willCoarsen ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getMinimalNeighborTimeConstraint() const  {
      return _persistentRecords._minimalNeighborTimeConstraint;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setMinimalNeighborTimeConstraint(const double& minimalNeighborTimeConstraint)  {
      _persistentRecords._minimalNeighborTimeConstraint = minimalNeighborTimeConstraint;
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getConstrainingNeighborIndex() const  {
      return _persistentRecords._constrainingNeighborIndex;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setConstrainingNeighborIndex(const int& constrainingNeighborIndex)  {
      _persistentRecords._constrainingNeighborIndex = constrainingNeighborIndex;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getMinimalLeafNeighborTimeConstraint() const  {
      return _persistentRecords._minimalLeafNeighborTimeConstraint;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setMinimalLeafNeighborTimeConstraint(const double& minimalLeafNeighborTimeConstraint)  {
      _persistentRecords._minimalLeafNeighborTimeConstraint = minimalLeafNeighborTimeConstraint;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getMinimalNeighborTime() const  {
      return _persistentRecords._minimalNeighborTime;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setMinimalNeighborTime(const double& minimalNeighborTime)  {
      _persistentRecords._minimalNeighborTime = minimalNeighborTime;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getMaximalNeighborTimestep() const  {
      return _persistentRecords._maximalNeighborTimestep;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setMaximalNeighborTimestep(const double& maximalNeighborTimestep)  {
      _persistentRecords._maximalNeighborTimestep = maximalNeighborTimestep;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getEstimatedNextTimestepSize() const  {
      return _persistentRecords._estimatedNextTimestepSize;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setEstimatedNextTimestepSize(const double& estimatedNextTimestepSize)  {
      _persistentRecords._estimatedNextTimestepSize = estimatedNextTimestepSize;
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getSkipGridIterations() const  {
      int mask =  (1 << (3)) - 1;
   mask = static_cast<int>(mask << (20));
   int tmp = static_cast<int>(_persistentRecords._packedRecords0 & mask);
   tmp = static_cast<int>(tmp >> (20));
   assertion(( tmp >= 0 &&  tmp <= 7));
   return (int) tmp;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setSkipGridIterations(const int& skipGridIterations)  {
      assertion((skipGridIterations >= 0 && skipGridIterations <= 7));
   int mask =  (1 << (3)) - 1;
   mask = static_cast<int>(mask << (20));
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 & ~mask);
   _persistentRecords._packedRecords0 = static_cast<int>(_persistentRecords._packedRecords0 | skipGridIterations << (20));
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getAgeInGridIterations() const  {
      return _persistentRecords._ageInGridIterations;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setAgeInGridIterations(const int& ageInGridIterations)  {
      _persistentRecords._ageInGridIterations = ageInGridIterations;
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getDemandedMeshWidth() const  {
      return _persistentRecords._demandedMeshWidth;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setDemandedMeshWidth(const double& demandedMeshWidth)  {
      _persistentRecords._demandedMeshWidth = demandedMeshWidth;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescriptionPacked::getRestrictionLowerBounds() const  {
      return _persistentRecords._restrictionLowerBounds;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setRestrictionLowerBounds(const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds)  {
      _persistentRecords._restrictionLowerBounds = (restrictionLowerBounds);
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getRestrictionLowerBounds(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._restrictionLowerBounds[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setRestrictionLowerBounds(int elementIndex, const double& restrictionLowerBounds)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._restrictionLowerBounds[elementIndex]= restrictionLowerBounds;
      
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::CellDescriptionPacked::getRestrictionUpperBounds() const  {
      return _persistentRecords._restrictionUpperBounds;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setRestrictionUpperBounds(const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds)  {
      _persistentRecords._restrictionUpperBounds = (restrictionUpperBounds);
   }
   
   
   
    double peanoclaw::records::CellDescriptionPacked::getRestrictionUpperBounds(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._restrictionUpperBounds[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setRestrictionUpperBounds(int elementIndex, const double& restrictionUpperBounds)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._restrictionUpperBounds[elementIndex]= restrictionUpperBounds;
      
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getCellDescriptionIndex() const  {
      return _persistentRecords._cellDescriptionIndex;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setCellDescriptionIndex(const int& cellDescriptionIndex)  {
      _persistentRecords._cellDescriptionIndex = cellDescriptionIndex;
   }
   
   
   
    int peanoclaw::records::CellDescriptionPacked::getUNewIndex() const  {
      return _persistentRecords._uNewIndex;
   }
   
   
   
    void peanoclaw::records::CellDescriptionPacked::setUNewIndex(const int& uNewIndex)  {
      _persistentRecords._uNewIndex = uNewIndex;
   }
   
   
   
   
   std::string peanoclaw::records::CellDescriptionPacked::toString() const {
      std::ostringstream stringstr;
      toString(stringstr);
      return stringstr.str();
   }
   
   void peanoclaw::records::CellDescriptionPacked::toString (std::ostream& out) const {
      out << "("; 
      out << "subdivisionFactor:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getSubdivisionFactor(i) << ",";
   }
   out << getSubdivisionFactor(DIMENSIONS-1) << "]";
      out << ",";
      out << "ghostLayerWidth:" << getGhostLayerWidth();
      out << ",";
      out << "unknownsPerSubcell:" << getUnknownsPerSubcell();
      out << ",";
      out << "auxiliarFieldsPerSubcell:" << getAuxiliarFieldsPerSubcell();
      out << ",";
      out << "level:" << getLevel();
      out << ",";
      out << "isVirtual:" << getIsVirtual();
      out << ",";
      out << "position:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getPosition(i) << ",";
   }
   out << getPosition(DIMENSIONS-1) << "]";
      out << ",";
      out << "size:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getSize(i) << ",";
   }
   out << getSize(DIMENSIONS-1) << "]";
      out << ",";
      out << "time:" << getTime();
      out << ",";
      out << "timestepSize:" << getTimestepSize();
      out << ",";
      out << "maximumFineGridTime:" << getMaximumFineGridTime();
      out << ",";
      out << "minimumFineGridTimestep:" << getMinimumFineGridTimestep();
      out << ",";
      out << "synchronizeFineGrids:" << getSynchronizeFineGrids();
      out << ",";
      out << "willCoarsen:" << getWillCoarsen();
      out << ",";
      out << "minimalNeighborTimeConstraint:" << getMinimalNeighborTimeConstraint();
      out << ",";
      out << "constrainingNeighborIndex:" << getConstrainingNeighborIndex();
      out << ",";
      out << "minimalLeafNeighborTimeConstraint:" << getMinimalLeafNeighborTimeConstraint();
      out << ",";
      out << "minimalNeighborTime:" << getMinimalNeighborTime();
      out << ",";
      out << "maximalNeighborTimestep:" << getMaximalNeighborTimestep();
      out << ",";
      out << "estimatedNextTimestepSize:" << getEstimatedNextTimestepSize();
      out << ",";
      out << "skipGridIterations:" << getSkipGridIterations();
      out << ",";
      out << "ageInGridIterations:" << getAgeInGridIterations();
      out << ",";
      out << "demandedMeshWidth:" << getDemandedMeshWidth();
      out << ",";
      out << "restrictionLowerBounds:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getRestrictionLowerBounds(i) << ",";
   }
   out << getRestrictionLowerBounds(DIMENSIONS-1) << "]";
      out << ",";
      out << "restrictionUpperBounds:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getRestrictionUpperBounds(i) << ",";
   }
   out << getRestrictionUpperBounds(DIMENSIONS-1) << "]";
      out << ",";
      out << "cellDescriptionIndex:" << getCellDescriptionIndex();
      out << ",";
      out << "uNewIndex:" << getUNewIndex();
      out <<  ")";
   }
   
   
   peanoclaw::records::CellDescriptionPacked::PersistentRecords peanoclaw::records::CellDescriptionPacked::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::CellDescription peanoclaw::records::CellDescriptionPacked::convert() const{
      return CellDescription(
         getSubdivisionFactor(),
         getGhostLayerWidth(),
         getUnknownsPerSubcell(),
         getAuxiliarFieldsPerSubcell(),
         getLevel(),
         getIsVirtual(),
         getPosition(),
         getSize(),
         getTime(),
         getTimestepSize(),
         getMaximumFineGridTime(),
         getMinimumFineGridTimestep(),
         getSynchronizeFineGrids(),
         getWillCoarsen(),
         getMinimalNeighborTimeConstraint(),
         getConstrainingNeighborIndex(),
         getMinimalLeafNeighborTimeConstraint(),
         getMinimalNeighborTime(),
         getMaximalNeighborTimestep(),
         getEstimatedNextTimestepSize(),
         getSkipGridIterations(),
         getAgeInGridIterations(),
         getDemandedMeshWidth(),
         getRestrictionLowerBounds(),
         getRestrictionUpperBounds(),
         getCellDescriptionIndex(),
         getUNewIndex()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::CellDescriptionPacked::_log( "peanoclaw::records::CellDescriptionPacked" );
      
      MPI_Datatype peanoclaw::records::CellDescriptionPacked::Datatype = 0;
      MPI_Datatype peanoclaw::records::CellDescriptionPacked::FullDatatype = 0;
      
      
      void peanoclaw::records::CellDescriptionPacked::initDatatype() {
         {
            CellDescriptionPacked dummyCellDescriptionPacked[2];
            
            const int Attributes = 20;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //subdivisionFactor
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
               MPI_DOUBLE,		 //maximumFineGridTime
               MPI_DOUBLE,		 //minimumFineGridTimestep
               MPI_DOUBLE,		 //minimalNeighborTimeConstraint
               MPI_DOUBLE,		 //minimalLeafNeighborTimeConstraint
               MPI_DOUBLE,		 //minimalNeighborTime
               MPI_DOUBLE,		 //maximalNeighborTimestep
               MPI_DOUBLE,		 //estimatedNextTimestepSize
               MPI_INT,		 //ageInGridIterations
               MPI_DOUBLE,		 //demandedMeshWidth
               MPI_DOUBLE,		 //restrictionLowerBounds
               MPI_DOUBLE,		 //restrictionUpperBounds
               MPI_INT,		 //cellDescriptionIndex
               MPI_INT,		 //uNewIndex
               MPI_INT,		 //_packedRecords0
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               DIMENSIONS,		 //subdivisionFactor
               DIMENSIONS,		 //position
               DIMENSIONS,		 //size
               1,		 //time
               1,		 //timestepSize
               1,		 //maximumFineGridTime
               1,		 //minimumFineGridTimestep
               1,		 //minimalNeighborTimeConstraint
               1,		 //minimalLeafNeighborTimeConstraint
               1,		 //minimalNeighborTime
               1,		 //maximalNeighborTimestep
               1,		 //estimatedNextTimestepSize
               1,		 //ageInGridIterations
               1,		 //demandedMeshWidth
               DIMENSIONS,		 //restrictionLowerBounds
               DIMENSIONS,		 //restrictionUpperBounds
               1,		 //cellDescriptionIndex
               1,		 //uNewIndex
               1,		 //_packedRecords0
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._position[0]))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._size[0]))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._time))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._timestepSize))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._maximumFineGridTime))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimumFineGridTimestep))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimalNeighborTimeConstraint))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimalLeafNeighborTimeConstraint))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimalNeighborTime))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._maximalNeighborTimestep))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._estimatedNextTimestepSize))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._ageInGridIterations))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._demandedMeshWidth))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._restrictionLowerBounds[0]))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._restrictionUpperBounds[0]))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._uNewIndex))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._packedRecords0))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyCellDescriptionPacked[1]._persistentRecords._subdivisionFactor[0])), 		&disp[19] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellDescriptionPacked::Datatype );
            MPI_Type_commit( &CellDescriptionPacked::Datatype );
            
         }
         {
            CellDescriptionPacked dummyCellDescriptionPacked[2];
            
            const int Attributes = 21;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //subdivisionFactor
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
               MPI_DOUBLE,		 //maximumFineGridTime
               MPI_DOUBLE,		 //minimumFineGridTimestep
               MPI_DOUBLE,		 //minimalNeighborTimeConstraint
               MPI_INT,		 //constrainingNeighborIndex
               MPI_DOUBLE,		 //minimalLeafNeighborTimeConstraint
               MPI_DOUBLE,		 //minimalNeighborTime
               MPI_DOUBLE,		 //maximalNeighborTimestep
               MPI_DOUBLE,		 //estimatedNextTimestepSize
               MPI_INT,		 //ageInGridIterations
               MPI_DOUBLE,		 //demandedMeshWidth
               MPI_DOUBLE,		 //restrictionLowerBounds
               MPI_DOUBLE,		 //restrictionUpperBounds
               MPI_INT,		 //cellDescriptionIndex
               MPI_INT,		 //uNewIndex
               MPI_INT,		 //_packedRecords0
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               DIMENSIONS,		 //subdivisionFactor
               DIMENSIONS,		 //position
               DIMENSIONS,		 //size
               1,		 //time
               1,		 //timestepSize
               1,		 //maximumFineGridTime
               1,		 //minimumFineGridTimestep
               1,		 //minimalNeighborTimeConstraint
               1,		 //constrainingNeighborIndex
               1,		 //minimalLeafNeighborTimeConstraint
               1,		 //minimalNeighborTime
               1,		 //maximalNeighborTimestep
               1,		 //estimatedNextTimestepSize
               1,		 //ageInGridIterations
               1,		 //demandedMeshWidth
               DIMENSIONS,		 //restrictionLowerBounds
               DIMENSIONS,		 //restrictionUpperBounds
               1,		 //cellDescriptionIndex
               1,		 //uNewIndex
               1,		 //_packedRecords0
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._position[0]))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._size[0]))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._time))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._timestepSize))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._maximumFineGridTime))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimumFineGridTimestep))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimalNeighborTimeConstraint))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._constrainingNeighborIndex))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimalLeafNeighborTimeConstraint))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimalNeighborTime))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._maximalNeighborTimestep))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._estimatedNextTimestepSize))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._ageInGridIterations))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._demandedMeshWidth))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._restrictionLowerBounds[0]))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._restrictionUpperBounds[0]))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._uNewIndex))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._packedRecords0))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyCellDescriptionPacked[1]._persistentRecords._subdivisionFactor[0])), 		&disp[20] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellDescriptionPacked::FullDatatype );
            MPI_Type_commit( &CellDescriptionPacked::FullDatatype );
            
         }
         
      }
      
      
      void peanoclaw::records::CellDescriptionPacked::shutdownDatatype() {
         MPI_Type_free( &CellDescriptionPacked::Datatype );
         MPI_Type_free( &CellDescriptionPacked::FullDatatype );
         
      }
      
      void peanoclaw::records::CellDescriptionPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
         MPI_Request* sendRequestHandle = new MPI_Request();
         MPI_Status   status;
         int          flag = 0;
         int          result;
         
         clock_t      timeOutWarning   = -1;
         clock_t      timeOutShutdown  = -1;
         bool         triggeredTimeoutWarning = false;
         
         if (exchangeOnlyAttributesMarkedWithParallelise) {
            result = MPI_Isend(
               this, 1, Datatype, destination,
               tag, tarch::parallel::Node::getInstance().getCommunicator(),
               sendRequestHandle
            );
            
         }
         else {
            result = MPI_Isend(
               this, 1, FullDatatype, destination,
               tag, tarch::parallel::Node::getInstance().getCommunicator(),
               sendRequestHandle
            );
            
         }
         if  (result!=MPI_SUCCESS) {
            std::ostringstream msg;
            msg << "was not able to send message peanoclaw::records::CellDescriptionPacked "
            << toString()
            << " to node " << destination
            << ": " << tarch::parallel::MPIReturnValueToString(result);
            _log.error( "send(int)",msg.str() );
         }
         result = MPI_Test( sendRequestHandle, &flag, &status );
         while (!flag) {
            if (timeOutWarning==-1)   timeOutWarning   = tarch::parallel::Node::getInstance().getDeadlockWarningTimeStamp();
            if (timeOutShutdown==-1)  timeOutShutdown  = tarch::parallel::Node::getInstance().getDeadlockTimeOutTimeStamp();
            result = MPI_Test( sendRequestHandle, &flag, &status );
            if (result!=MPI_SUCCESS) {
               std::ostringstream msg;
               msg << "testing for finished send task for peanoclaw::records::CellDescriptionPacked "
               << toString()
               << " sent to node " << destination
               << " failed: " << tarch::parallel::MPIReturnValueToString(result);
               _log.error("send(int)", msg.str() );
            }
            
            // deadlock aspect
            if (
               tarch::parallel::Node::getInstance().isTimeOutWarningEnabled() &&
               (clock()>timeOutWarning) &&
               (!triggeredTimeoutWarning)
            ) {
               tarch::parallel::Node::getInstance().writeTimeOutWarning(
               "peanoclaw::records::CellDescriptionPacked",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::CellDescriptionPacked",
               "send(int)", destination,tag,1
               );
            }
            tarch::parallel::Node::getInstance().receiveDanglingMessages();
         }
         
         delete sendRequestHandle;
         #ifdef Debug
         _log.debug("send(int,int)", "sent " + toString() );
         #endif
         
      }
      
      
      
      void peanoclaw::records::CellDescriptionPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
         MPI_Request* sendRequestHandle = new MPI_Request();
         MPI_Status   status;
         int          flag = 0;
         int          result;
         
         clock_t      timeOutWarning   = -1;
         clock_t      timeOutShutdown  = -1;
         bool         triggeredTimeoutWarning = false;
         
         if (exchangeOnlyAttributesMarkedWithParallelise) {
            result = MPI_Irecv(
               this, 1, Datatype, source, tag,
               tarch::parallel::Node::getInstance().getCommunicator(), sendRequestHandle
            );
            
         }
         else {
            result = MPI_Irecv(
               this, 1, FullDatatype, source, tag,
               tarch::parallel::Node::getInstance().getCommunicator(), sendRequestHandle
            );
            
         }
         if ( result != MPI_SUCCESS ) {
            std::ostringstream msg;
            msg << "failed to start to receive peanoclaw::records::CellDescriptionPacked from node "
            << source << ": " << tarch::parallel::MPIReturnValueToString(result);
            _log.error( "receive(int)", msg.str() );
         }
         
         result = MPI_Test( sendRequestHandle, &flag, &status );
         while (!flag) {
            if (timeOutWarning==-1)   timeOutWarning   = tarch::parallel::Node::getInstance().getDeadlockWarningTimeStamp();
            if (timeOutShutdown==-1)  timeOutShutdown  = tarch::parallel::Node::getInstance().getDeadlockTimeOutTimeStamp();
            result = MPI_Test( sendRequestHandle, &flag, &status );
            if (result!=MPI_SUCCESS) {
               std::ostringstream msg;
               msg << "testing for finished receive task for peanoclaw::records::CellDescriptionPacked failed: "
               << tarch::parallel::MPIReturnValueToString(result);
               _log.error("receive(int)", msg.str() );
            }
            
            // deadlock aspect
            if (
               tarch::parallel::Node::getInstance().isTimeOutWarningEnabled() &&
               (clock()>timeOutWarning) &&
               (!triggeredTimeoutWarning)
            ) {
               tarch::parallel::Node::getInstance().writeTimeOutWarning(
               "peanoclaw::records::CellDescriptionPacked",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::CellDescriptionPacked",
               "receive(int)", source,tag,1
               );
            }
            tarch::parallel::Node::getInstance().receiveDanglingMessages();
         }
         
         delete sendRequestHandle;
         
         #ifdef Debug
         _log.debug("receive(int,int)", "received " + toString() ); 
         #endif
         
      }
      
      
      
      bool peanoclaw::records::CellDescriptionPacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
         MPI_Status status;
         int  flag        = 0;
         MPI_Iprobe(
            MPI_ANY_SOURCE, tag,
            tarch::parallel::Node::getInstance().getCommunicator(), &flag, &status
         );
         if (flag) {
            int  messageCounter;
            if (exchangeOnlyAttributesMarkedWithParallelise) {
               MPI_Get_count(&status, Datatype, &messageCounter);
            }
            else {
               MPI_Get_count(&status, FullDatatype, &messageCounter);
            }
            return messageCounter > 0;
         }
         else return false;
         
      }
      
      
   #endif
   
   
   

#endif


