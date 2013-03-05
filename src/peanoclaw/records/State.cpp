#include "peanoclaw/records/State.h"

#if defined(Parallel)
   peanoclaw::records::State::PersistentRecords::PersistentRecords() {
      
   }
   
   
   peanoclaw::records::State::PersistentRecords::PersistentRecords(const int& additionalLevelsForPredefinedRefinement, const bool& isInitializing, const bool& initialRefinmentTriggered, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const tarch::la::Vector<DIMENSIONS,double>& initialMinimalMeshWidth, const tarch::la::Vector<DIMENSIONS,int>& defaultSubdivisionFactor, const int& defaultGhostWidthLayer, const double& initialTimestepSize, const bool& useDimensionalSplitting, const double& globalTimestepEndTime, const bool& allPatchesEvolvedToGlobalTimestep, const tarch::la::Vector<DIMENSIONS,double>& domainOffset, const tarch::la::Vector<DIMENSIONS,double>& domainSize, const int& plotNumber, const int& subPlotNumber, const double& startMaximumGlobalTimeInterval, const double& endMaximumGlobalTimeInterval, const double& startMinimumGlobalTimeInterval, const double& endMinimumGlobalTimeInterval, const double& minimalTimestep, const double& totalNumberOfCellUpdates, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& isTraversalInverted, const LoadBalancingState& loadRebalancingState, const bool& reduceStateAndCell):
   _additionalLevelsForPredefinedRefinement(additionalLevelsForPredefinedRefinement),
   _isInitializing(isInitializing),
   _initialRefinmentTriggered(initialRefinmentTriggered),
   _unknownsPerSubcell(unknownsPerSubcell),
   _auxiliarFieldsPerSubcell(auxiliarFieldsPerSubcell),
   _initialMinimalMeshWidth(initialMinimalMeshWidth),
   _defaultSubdivisionFactor(defaultSubdivisionFactor),
   _defaultGhostWidthLayer(defaultGhostWidthLayer),
   _initialTimestepSize(initialTimestepSize),
   _useDimensionalSplitting(useDimensionalSplitting),
   _globalTimestepEndTime(globalTimestepEndTime),
   _allPatchesEvolvedToGlobalTimestep(allPatchesEvolvedToGlobalTimestep),
   _domainOffset(domainOffset),
   _domainSize(domainSize),
   _plotNumber(plotNumber),
   _subPlotNumber(subPlotNumber),
   _startMaximumGlobalTimeInterval(startMaximumGlobalTimeInterval),
   _endMaximumGlobalTimeInterval(endMaximumGlobalTimeInterval),
   _startMinimumGlobalTimeInterval(startMinimumGlobalTimeInterval),
   _endMinimumGlobalTimeInterval(endMinimumGlobalTimeInterval),
   _minimalTimestep(minimalTimestep),
   _totalNumberOfCellUpdates(totalNumberOfCellUpdates),
   _minMeshWidth(minMeshWidth),
   _maxMeshWidth(maxMeshWidth),
   _numberOfInnerVertices(numberOfInnerVertices),
   _numberOfBoundaryVertices(numberOfBoundaryVertices),
   _numberOfOuterVertices(numberOfOuterVertices),
   _numberOfInnerCells(numberOfInnerCells),
   _numberOfOuterCells(numberOfOuterCells),
   _maxLevel(maxLevel),
   _hasRefined(hasRefined),
   _hasTriggeredRefinementForNextIteration(hasTriggeredRefinementForNextIteration),
   _hasErased(hasErased),
   _hasTriggeredEraseForNextIteration(hasTriggeredEraseForNextIteration),
   _hasChangedVertexOrCellState(hasChangedVertexOrCellState),
   _isTraversalInverted(isTraversalInverted),
   _loadRebalancingState(loadRebalancingState),
   _reduceStateAndCell(reduceStateAndCell) {
      
   }
   
   peanoclaw::records::State::State() {
      
   }
   
   
   peanoclaw::records::State::State(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._additionalLevelsForPredefinedRefinement, persistentRecords._isInitializing, persistentRecords._initialRefinmentTriggered, persistentRecords._unknownsPerSubcell, persistentRecords._auxiliarFieldsPerSubcell, persistentRecords._initialMinimalMeshWidth, persistentRecords._defaultSubdivisionFactor, persistentRecords._defaultGhostWidthLayer, persistentRecords._initialTimestepSize, persistentRecords._useDimensionalSplitting, persistentRecords._globalTimestepEndTime, persistentRecords._allPatchesEvolvedToGlobalTimestep, persistentRecords._domainOffset, persistentRecords._domainSize, persistentRecords._plotNumber, persistentRecords._subPlotNumber, persistentRecords._startMaximumGlobalTimeInterval, persistentRecords._endMaximumGlobalTimeInterval, persistentRecords._startMinimumGlobalTimeInterval, persistentRecords._endMinimumGlobalTimeInterval, persistentRecords._minimalTimestep, persistentRecords._totalNumberOfCellUpdates, persistentRecords._minMeshWidth, persistentRecords._maxMeshWidth, persistentRecords._numberOfInnerVertices, persistentRecords._numberOfBoundaryVertices, persistentRecords._numberOfOuterVertices, persistentRecords._numberOfInnerCells, persistentRecords._numberOfOuterCells, persistentRecords._maxLevel, persistentRecords._hasRefined, persistentRecords._hasTriggeredRefinementForNextIteration, persistentRecords._hasErased, persistentRecords._hasTriggeredEraseForNextIteration, persistentRecords._hasChangedVertexOrCellState, persistentRecords._isTraversalInverted, persistentRecords._loadRebalancingState, persistentRecords._reduceStateAndCell) {
      
   }
   
   
   peanoclaw::records::State::State(const int& additionalLevelsForPredefinedRefinement, const bool& isInitializing, const bool& initialRefinmentTriggered, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const tarch::la::Vector<DIMENSIONS,double>& initialMinimalMeshWidth, const tarch::la::Vector<DIMENSIONS,int>& defaultSubdivisionFactor, const int& defaultGhostWidthLayer, const double& initialTimestepSize, const bool& useDimensionalSplitting, const double& globalTimestepEndTime, const bool& allPatchesEvolvedToGlobalTimestep, const tarch::la::Vector<DIMENSIONS,double>& domainOffset, const tarch::la::Vector<DIMENSIONS,double>& domainSize, const int& plotNumber, const int& subPlotNumber, const double& startMaximumGlobalTimeInterval, const double& endMaximumGlobalTimeInterval, const double& startMinimumGlobalTimeInterval, const double& endMinimumGlobalTimeInterval, const double& minimalTimestep, const double& totalNumberOfCellUpdates, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& isTraversalInverted, const LoadBalancingState& loadRebalancingState, const bool& reduceStateAndCell):
   _persistentRecords(additionalLevelsForPredefinedRefinement, isInitializing, initialRefinmentTriggered, unknownsPerSubcell, auxiliarFieldsPerSubcell, initialMinimalMeshWidth, defaultSubdivisionFactor, defaultGhostWidthLayer, initialTimestepSize, useDimensionalSplitting, globalTimestepEndTime, allPatchesEvolvedToGlobalTimestep, domainOffset, domainSize, plotNumber, subPlotNumber, startMaximumGlobalTimeInterval, endMaximumGlobalTimeInterval, startMinimumGlobalTimeInterval, endMinimumGlobalTimeInterval, minimalTimestep, totalNumberOfCellUpdates, minMeshWidth, maxMeshWidth, numberOfInnerVertices, numberOfBoundaryVertices, numberOfOuterVertices, numberOfInnerCells, numberOfOuterCells, maxLevel, hasRefined, hasTriggeredRefinementForNextIteration, hasErased, hasTriggeredEraseForNextIteration, hasChangedVertexOrCellState, isTraversalInverted, loadRebalancingState, reduceStateAndCell) {
      
   }
   
   
   peanoclaw::records::State::State(const int& additionalLevelsForPredefinedRefinement, const bool& isInitializing, const bool& initialRefinmentTriggered, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const tarch::la::Vector<DIMENSIONS,double>& initialMinimalMeshWidth, const tarch::la::Vector<DIMENSIONS,int>& defaultSubdivisionFactor, const int& defaultGhostWidthLayer, const double& initialTimestepSize, const bool& useDimensionalSplitting, const double& globalTimestepEndTime, const bool& allPatchesEvolvedToGlobalTimestep, const tarch::la::Vector<DIMENSIONS,double>& domainOffset, const tarch::la::Vector<DIMENSIONS,double>& domainSize, const int& plotNumber, const int& subPlotNumber, const double& startMaximumGlobalTimeInterval, const double& endMaximumGlobalTimeInterval, const double& startMinimumGlobalTimeInterval, const double& endMinimumGlobalTimeInterval, const double& minimalTimestep, const double& totalNumberOfCellUpdates, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& isTraversalInverted, const LoadBalancingState& loadRebalancingState, const bool& reduceStateAndCell, const bool& hasWorkerWithWorker):
   _persistentRecords(additionalLevelsForPredefinedRefinement, isInitializing, initialRefinmentTriggered, unknownsPerSubcell, auxiliarFieldsPerSubcell, initialMinimalMeshWidth, defaultSubdivisionFactor, defaultGhostWidthLayer, initialTimestepSize, useDimensionalSplitting, globalTimestepEndTime, allPatchesEvolvedToGlobalTimestep, domainOffset, domainSize, plotNumber, subPlotNumber, startMaximumGlobalTimeInterval, endMaximumGlobalTimeInterval, startMinimumGlobalTimeInterval, endMinimumGlobalTimeInterval, minimalTimestep, totalNumberOfCellUpdates, minMeshWidth, maxMeshWidth, numberOfInnerVertices, numberOfBoundaryVertices, numberOfOuterVertices, numberOfInnerCells, numberOfOuterCells, maxLevel, hasRefined, hasTriggeredRefinementForNextIteration, hasErased, hasTriggeredEraseForNextIteration, hasChangedVertexOrCellState, isTraversalInverted, loadRebalancingState, reduceStateAndCell),_hasWorkerWithWorker(hasWorkerWithWorker) {
      
   }
   
   peanoclaw::records::State::~State() { }
   
   std::string peanoclaw::records::State::toString(const LoadBalancingState& param) {
      switch (param) {
         case NoRebalancing: return "NoRebalancing";
         case ForkTriggered: return "ForkTriggered";
         case Forking: return "Forking";
         case JoinTriggered: return "JoinTriggered";
         case Joining: return "Joining";
         case JoinWithMasterTriggered: return "JoinWithMasterTriggered";
         case JoiningWithMaster: return "JoiningWithMaster";
         case HasJoinedWithMaster: return "HasJoinedWithMaster";
         case IsNewWorkerDueToForkOfExistingDomain: return "IsNewWorkerDueToForkOfExistingDomain";
      }
      return "undefined";
   }
   
   std::string peanoclaw::records::State::getLoadBalancingStateMapping() {
      return "LoadBalancingState(NoRebalancing=0,ForkTriggered=1,Forking=2,JoinTriggered=3,Joining=4,JoinWithMasterTriggered=5,JoiningWithMaster=6,HasJoinedWithMaster=7,IsNewWorkerDueToForkOfExistingDomain=8)";
   }
   
   
   std::string peanoclaw::records::State::toString() const {
      std::ostringstream stringstr;
      toString(stringstr);
      return stringstr.str();
   }
   
   void peanoclaw::records::State::toString (std::ostream& out) const {
      out << "("; 
      out << "additionalLevelsForPredefinedRefinement:" << getAdditionalLevelsForPredefinedRefinement();
      out << ",";
      out << "isInitializing:" << getIsInitializing();
      out << ",";
      out << "initialRefinmentTriggered:" << getInitialRefinmentTriggered();
      out << ",";
      out << "unknownsPerSubcell:" << getUnknownsPerSubcell();
      out << ",";
      out << "auxiliarFieldsPerSubcell:" << getAuxiliarFieldsPerSubcell();
      out << ",";
      out << "initialMinimalMeshWidth:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getInitialMinimalMeshWidth(i) << ",";
   }
   out << getInitialMinimalMeshWidth(DIMENSIONS-1) << "]";
      out << ",";
      out << "defaultSubdivisionFactor:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getDefaultSubdivisionFactor(i) << ",";
   }
   out << getDefaultSubdivisionFactor(DIMENSIONS-1) << "]";
      out << ",";
      out << "defaultGhostWidthLayer:" << getDefaultGhostWidthLayer();
      out << ",";
      out << "initialTimestepSize:" << getInitialTimestepSize();
      out << ",";
      out << "useDimensionalSplitting:" << getUseDimensionalSplitting();
      out << ",";
      out << "globalTimestepEndTime:" << getGlobalTimestepEndTime();
      out << ",";
      out << "allPatchesEvolvedToGlobalTimestep:" << getAllPatchesEvolvedToGlobalTimestep();
      out << ",";
      out << "domainOffset:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getDomainOffset(i) << ",";
   }
   out << getDomainOffset(DIMENSIONS-1) << "]";
      out << ",";
      out << "domainSize:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getDomainSize(i) << ",";
   }
   out << getDomainSize(DIMENSIONS-1) << "]";
      out << ",";
      out << "plotNumber:" << getPlotNumber();
      out << ",";
      out << "subPlotNumber:" << getSubPlotNumber();
      out << ",";
      out << "startMaximumGlobalTimeInterval:" << getStartMaximumGlobalTimeInterval();
      out << ",";
      out << "endMaximumGlobalTimeInterval:" << getEndMaximumGlobalTimeInterval();
      out << ",";
      out << "startMinimumGlobalTimeInterval:" << getStartMinimumGlobalTimeInterval();
      out << ",";
      out << "endMinimumGlobalTimeInterval:" << getEndMinimumGlobalTimeInterval();
      out << ",";
      out << "minimalTimestep:" << getMinimalTimestep();
      out << ",";
      out << "totalNumberOfCellUpdates:" << getTotalNumberOfCellUpdates();
      out << ",";
      out << "minMeshWidth:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getMinMeshWidth(i) << ",";
   }
   out << getMinMeshWidth(DIMENSIONS-1) << "]";
      out << ",";
      out << "maxMeshWidth:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getMaxMeshWidth(i) << ",";
   }
   out << getMaxMeshWidth(DIMENSIONS-1) << "]";
      out << ",";
      out << "numberOfInnerVertices:" << getNumberOfInnerVertices();
      out << ",";
      out << "numberOfBoundaryVertices:" << getNumberOfBoundaryVertices();
      out << ",";
      out << "numberOfOuterVertices:" << getNumberOfOuterVertices();
      out << ",";
      out << "numberOfInnerCells:" << getNumberOfInnerCells();
      out << ",";
      out << "numberOfOuterCells:" << getNumberOfOuterCells();
      out << ",";
      out << "maxLevel:" << getMaxLevel();
      out << ",";
      out << "hasRefined:" << getHasRefined();
      out << ",";
      out << "hasTriggeredRefinementForNextIteration:" << getHasTriggeredRefinementForNextIteration();
      out << ",";
      out << "hasErased:" << getHasErased();
      out << ",";
      out << "hasTriggeredEraseForNextIteration:" << getHasTriggeredEraseForNextIteration();
      out << ",";
      out << "hasChangedVertexOrCellState:" << getHasChangedVertexOrCellState();
      out << ",";
      out << "isTraversalInverted:" << getIsTraversalInverted();
      out << ",";
      out << "loadRebalancingState:" << toString(getLoadRebalancingState());
      out << ",";
      out << "reduceStateAndCell:" << getReduceStateAndCell();
      out << ",";
      out << "hasWorkerWithWorker:" << getHasWorkerWithWorker();
      out <<  ")";
   }
   
   
   peanoclaw::records::State::PersistentRecords peanoclaw::records::State::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::StatePacked peanoclaw::records::State::convert() const{
      return StatePacked(
         getAdditionalLevelsForPredefinedRefinement(),
         getIsInitializing(),
         getInitialRefinmentTriggered(),
         getUnknownsPerSubcell(),
         getAuxiliarFieldsPerSubcell(),
         getInitialMinimalMeshWidth(),
         getDefaultSubdivisionFactor(),
         getDefaultGhostWidthLayer(),
         getInitialTimestepSize(),
         getUseDimensionalSplitting(),
         getGlobalTimestepEndTime(),
         getAllPatchesEvolvedToGlobalTimestep(),
         getDomainOffset(),
         getDomainSize(),
         getPlotNumber(),
         getSubPlotNumber(),
         getStartMaximumGlobalTimeInterval(),
         getEndMaximumGlobalTimeInterval(),
         getStartMinimumGlobalTimeInterval(),
         getEndMinimumGlobalTimeInterval(),
         getMinimalTimestep(),
         getTotalNumberOfCellUpdates(),
         getMinMeshWidth(),
         getMaxMeshWidth(),
         getNumberOfInnerVertices(),
         getNumberOfBoundaryVertices(),
         getNumberOfOuterVertices(),
         getNumberOfInnerCells(),
         getNumberOfOuterCells(),
         getMaxLevel(),
         getHasRefined(),
         getHasTriggeredRefinementForNextIteration(),
         getHasErased(),
         getHasTriggeredEraseForNextIteration(),
         getHasChangedVertexOrCellState(),
         getIsTraversalInverted(),
         getLoadRebalancingState(),
         getReduceStateAndCell(),
         getHasWorkerWithWorker()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::State::_log( "peanoclaw::records::State" );
      
      MPI_Datatype peanoclaw::records::State::Datatype = 0;
      MPI_Datatype peanoclaw::records::State::FullDatatype = 0;
      
      
      void peanoclaw::records::State::initDatatype() {
         {
            State dummyState[2];
            
            const int Attributes = 38;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //additionalLevelsForPredefinedRefinement
               MPI_CHAR,		 //isInitializing
               MPI_CHAR,		 //initialRefinmentTriggered
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_DOUBLE,		 //initialMinimalMeshWidth
               MPI_INT,		 //defaultSubdivisionFactor
               MPI_INT,		 //defaultGhostWidthLayer
               MPI_DOUBLE,		 //initialTimestepSize
               MPI_CHAR,		 //useDimensionalSplitting
               MPI_DOUBLE,		 //globalTimestepEndTime
               MPI_CHAR,		 //allPatchesEvolvedToGlobalTimestep
               MPI_DOUBLE,		 //domainOffset
               MPI_DOUBLE,		 //domainSize
               MPI_INT,		 //plotNumber
               MPI_INT,		 //subPlotNumber
               MPI_DOUBLE,		 //startMaximumGlobalTimeInterval
               MPI_DOUBLE,		 //endMaximumGlobalTimeInterval
               MPI_DOUBLE,		 //startMinimumGlobalTimeInterval
               MPI_DOUBLE,		 //endMinimumGlobalTimeInterval
               MPI_DOUBLE,		 //minimalTimestep
               MPI_DOUBLE,		 //totalNumberOfCellUpdates
               MPI_DOUBLE,		 //minMeshWidth
               MPI_DOUBLE,		 //maxMeshWidth
               MPI_DOUBLE,		 //numberOfInnerVertices
               MPI_DOUBLE,		 //numberOfBoundaryVertices
               MPI_DOUBLE,		 //numberOfOuterVertices
               MPI_DOUBLE,		 //numberOfInnerCells
               MPI_DOUBLE,		 //numberOfOuterCells
               MPI_INT,		 //maxLevel
               MPI_CHAR,		 //hasRefined
               MPI_CHAR,		 //hasTriggeredRefinementForNextIteration
               MPI_CHAR,		 //hasErased
               MPI_CHAR,		 //hasTriggeredEraseForNextIteration
               MPI_CHAR,		 //hasChangedVertexOrCellState
               MPI_CHAR,		 //isTraversalInverted
               MPI_CHAR,		 //hasWorkerWithWorker
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //additionalLevelsForPredefinedRefinement
               1,		 //isInitializing
               1,		 //initialRefinmentTriggered
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               DIMENSIONS,		 //initialMinimalMeshWidth
               DIMENSIONS,		 //defaultSubdivisionFactor
               1,		 //defaultGhostWidthLayer
               1,		 //initialTimestepSize
               1,		 //useDimensionalSplitting
               1,		 //globalTimestepEndTime
               1,		 //allPatchesEvolvedToGlobalTimestep
               DIMENSIONS,		 //domainOffset
               DIMENSIONS,		 //domainSize
               1,		 //plotNumber
               1,		 //subPlotNumber
               1,		 //startMaximumGlobalTimeInterval
               1,		 //endMaximumGlobalTimeInterval
               1,		 //startMinimumGlobalTimeInterval
               1,		 //endMinimumGlobalTimeInterval
               1,		 //minimalTimestep
               1,		 //totalNumberOfCellUpdates
               DIMENSIONS,		 //minMeshWidth
               DIMENSIONS,		 //maxMeshWidth
               1,		 //numberOfInnerVertices
               1,		 //numberOfBoundaryVertices
               1,		 //numberOfOuterVertices
               1,		 //numberOfInnerCells
               1,		 //numberOfOuterCells
               1,		 //maxLevel
               1,		 //hasRefined
               1,		 //hasTriggeredRefinementForNextIteration
               1,		 //hasErased
               1,		 //hasTriggeredEraseForNextIteration
               1,		 //hasChangedVertexOrCellState
               1,		 //isTraversalInverted
               1,		 //hasWorkerWithWorker
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._isInitializing))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialRefinmentTriggered))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._unknownsPerSubcell))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialMinimalMeshWidth[0]))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._defaultSubdivisionFactor[0]))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._defaultGhostWidthLayer))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialTimestepSize))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._useDimensionalSplitting))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._globalTimestepEndTime))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._allPatchesEvolvedToGlobalTimestep))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._domainOffset[0]))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._domainSize[0]))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._plotNumber))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._subPlotNumber))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._startMaximumGlobalTimeInterval))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._endMaximumGlobalTimeInterval))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._startMinimumGlobalTimeInterval))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._endMinimumGlobalTimeInterval))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._minimalTimestep))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._totalNumberOfCellUpdates))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._minMeshWidth[0]))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._maxMeshWidth[0]))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerVertices))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfBoundaryVertices))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterVertices))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerCells))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterCells))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._maxLevel))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasRefined))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasTriggeredRefinementForNextIteration))), 		&disp[31] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasErased))), 		&disp[32] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasTriggeredEraseForNextIteration))), 		&disp[33] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasChangedVertexOrCellState))), 		&disp[34] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._isTraversalInverted))), 		&disp[35] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._hasWorkerWithWorker))), 		&disp[36] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[1]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[37] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &State::Datatype );
            MPI_Type_commit( &State::Datatype );
            
         }
         {
            State dummyState[2];
            
            const int Attributes = 40;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //additionalLevelsForPredefinedRefinement
               MPI_CHAR,		 //isInitializing
               MPI_CHAR,		 //initialRefinmentTriggered
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_DOUBLE,		 //initialMinimalMeshWidth
               MPI_INT,		 //defaultSubdivisionFactor
               MPI_INT,		 //defaultGhostWidthLayer
               MPI_DOUBLE,		 //initialTimestepSize
               MPI_CHAR,		 //useDimensionalSplitting
               MPI_DOUBLE,		 //globalTimestepEndTime
               MPI_CHAR,		 //allPatchesEvolvedToGlobalTimestep
               MPI_DOUBLE,		 //domainOffset
               MPI_DOUBLE,		 //domainSize
               MPI_INT,		 //plotNumber
               MPI_INT,		 //subPlotNumber
               MPI_DOUBLE,		 //startMaximumGlobalTimeInterval
               MPI_DOUBLE,		 //endMaximumGlobalTimeInterval
               MPI_DOUBLE,		 //startMinimumGlobalTimeInterval
               MPI_DOUBLE,		 //endMinimumGlobalTimeInterval
               MPI_DOUBLE,		 //minimalTimestep
               MPI_DOUBLE,		 //totalNumberOfCellUpdates
               MPI_DOUBLE,		 //minMeshWidth
               MPI_DOUBLE,		 //maxMeshWidth
               MPI_DOUBLE,		 //numberOfInnerVertices
               MPI_DOUBLE,		 //numberOfBoundaryVertices
               MPI_DOUBLE,		 //numberOfOuterVertices
               MPI_DOUBLE,		 //numberOfInnerCells
               MPI_DOUBLE,		 //numberOfOuterCells
               MPI_INT,		 //maxLevel
               MPI_CHAR,		 //hasRefined
               MPI_CHAR,		 //hasTriggeredRefinementForNextIteration
               MPI_CHAR,		 //hasErased
               MPI_CHAR,		 //hasTriggeredEraseForNextIteration
               MPI_CHAR,		 //hasChangedVertexOrCellState
               MPI_CHAR,		 //isTraversalInverted
               MPI_INT,		 //loadRebalancingState
               MPI_CHAR,		 //reduceStateAndCell
               MPI_CHAR,		 //hasWorkerWithWorker
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //additionalLevelsForPredefinedRefinement
               1,		 //isInitializing
               1,		 //initialRefinmentTriggered
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               DIMENSIONS,		 //initialMinimalMeshWidth
               DIMENSIONS,		 //defaultSubdivisionFactor
               1,		 //defaultGhostWidthLayer
               1,		 //initialTimestepSize
               1,		 //useDimensionalSplitting
               1,		 //globalTimestepEndTime
               1,		 //allPatchesEvolvedToGlobalTimestep
               DIMENSIONS,		 //domainOffset
               DIMENSIONS,		 //domainSize
               1,		 //plotNumber
               1,		 //subPlotNumber
               1,		 //startMaximumGlobalTimeInterval
               1,		 //endMaximumGlobalTimeInterval
               1,		 //startMinimumGlobalTimeInterval
               1,		 //endMinimumGlobalTimeInterval
               1,		 //minimalTimestep
               1,		 //totalNumberOfCellUpdates
               DIMENSIONS,		 //minMeshWidth
               DIMENSIONS,		 //maxMeshWidth
               1,		 //numberOfInnerVertices
               1,		 //numberOfBoundaryVertices
               1,		 //numberOfOuterVertices
               1,		 //numberOfInnerCells
               1,		 //numberOfOuterCells
               1,		 //maxLevel
               1,		 //hasRefined
               1,		 //hasTriggeredRefinementForNextIteration
               1,		 //hasErased
               1,		 //hasTriggeredEraseForNextIteration
               1,		 //hasChangedVertexOrCellState
               1,		 //isTraversalInverted
               1,		 //loadRebalancingState
               1,		 //reduceStateAndCell
               1,		 //hasWorkerWithWorker
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._isInitializing))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialRefinmentTriggered))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._unknownsPerSubcell))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialMinimalMeshWidth[0]))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._defaultSubdivisionFactor[0]))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._defaultGhostWidthLayer))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialTimestepSize))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._useDimensionalSplitting))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._globalTimestepEndTime))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._allPatchesEvolvedToGlobalTimestep))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._domainOffset[0]))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._domainSize[0]))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._plotNumber))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._subPlotNumber))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._startMaximumGlobalTimeInterval))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._endMaximumGlobalTimeInterval))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._startMinimumGlobalTimeInterval))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._endMinimumGlobalTimeInterval))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._minimalTimestep))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._totalNumberOfCellUpdates))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._minMeshWidth[0]))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._maxMeshWidth[0]))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerVertices))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfBoundaryVertices))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterVertices))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerCells))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterCells))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._maxLevel))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasRefined))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasTriggeredRefinementForNextIteration))), 		&disp[31] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasErased))), 		&disp[32] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasTriggeredEraseForNextIteration))), 		&disp[33] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasChangedVertexOrCellState))), 		&disp[34] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._isTraversalInverted))), 		&disp[35] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._loadRebalancingState))), 		&disp[36] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._reduceStateAndCell))), 		&disp[37] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._hasWorkerWithWorker))), 		&disp[38] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[1]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[39] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &State::FullDatatype );
            MPI_Type_commit( &State::FullDatatype );
            
         }
         
      }
      
      
      void peanoclaw::records::State::shutdownDatatype() {
         MPI_Type_free( &State::Datatype );
         MPI_Type_free( &State::FullDatatype );
         
      }
      
      void peanoclaw::records::State::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
         MPI_Request* sendRequestHandle = new MPI_Request();
         MPI_Status   status;
         int          flag = 0;
         int          result;
         
         clock_t      timeOutWarning   = -1;
         clock_t      timeOutShutdown  = -1;
         bool         triggeredTimeoutWarning = false;
         
         #ifdef Asserts
         _senderRank = -1;
         #endif
         
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
            msg << "was not able to send message peanoclaw::records::State "
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
               msg << "testing for finished send task for peanoclaw::records::State "
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
               "peanoclaw::records::State",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::State",
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
      
      
      
      void peanoclaw::records::State::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "failed to start to receive peanoclaw::records::State from node "
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
               msg << "testing for finished receive task for peanoclaw::records::State failed: "
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
               "peanoclaw::records::State",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::State",
               "receive(int)", source,tag,1
               );
            }
            tarch::parallel::Node::getInstance().receiveDanglingMessages();
         }
         
         delete sendRequestHandle;
         
         _senderRank = status.MPI_SOURCE;
         #ifdef Debug
         _log.debug("receive(int,int)", "received " + toString() ); 
         #endif
         
      }
      
      
      
      bool peanoclaw::records::State::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
      
      int peanoclaw::records::State::getSenderRank() const {
         assertion( _senderRank!=-1 );
         return _senderRank;
         
      }
   #endif
   
   
   peanoclaw::records::StatePacked::PersistentRecords::PersistentRecords() {
      assertion((7 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::StatePacked::PersistentRecords::PersistentRecords(const int& additionalLevelsForPredefinedRefinement, const bool& isInitializing, const bool& initialRefinmentTriggered, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const tarch::la::Vector<DIMENSIONS,double>& initialMinimalMeshWidth, const tarch::la::Vector<DIMENSIONS,int>& defaultSubdivisionFactor, const int& defaultGhostWidthLayer, const double& initialTimestepSize, const bool& useDimensionalSplitting, const double& globalTimestepEndTime, const bool& allPatchesEvolvedToGlobalTimestep, const tarch::la::Vector<DIMENSIONS,double>& domainOffset, const tarch::la::Vector<DIMENSIONS,double>& domainSize, const int& plotNumber, const int& subPlotNumber, const double& startMaximumGlobalTimeInterval, const double& endMaximumGlobalTimeInterval, const double& startMinimumGlobalTimeInterval, const double& endMinimumGlobalTimeInterval, const double& minimalTimestep, const double& totalNumberOfCellUpdates, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& isTraversalInverted, const LoadBalancingState& loadRebalancingState, const bool& reduceStateAndCell):
   _additionalLevelsForPredefinedRefinement(additionalLevelsForPredefinedRefinement),
   _isInitializing(isInitializing),
   _initialRefinmentTriggered(initialRefinmentTriggered),
   _unknownsPerSubcell(unknownsPerSubcell),
   _auxiliarFieldsPerSubcell(auxiliarFieldsPerSubcell),
   _initialMinimalMeshWidth(initialMinimalMeshWidth),
   _defaultSubdivisionFactor(defaultSubdivisionFactor),
   _defaultGhostWidthLayer(defaultGhostWidthLayer),
   _initialTimestepSize(initialTimestepSize),
   _useDimensionalSplitting(useDimensionalSplitting),
   _globalTimestepEndTime(globalTimestepEndTime),
   _domainOffset(domainOffset),
   _domainSize(domainSize),
   _plotNumber(plotNumber),
   _subPlotNumber(subPlotNumber),
   _startMaximumGlobalTimeInterval(startMaximumGlobalTimeInterval),
   _endMaximumGlobalTimeInterval(endMaximumGlobalTimeInterval),
   _startMinimumGlobalTimeInterval(startMinimumGlobalTimeInterval),
   _endMinimumGlobalTimeInterval(endMinimumGlobalTimeInterval),
   _minimalTimestep(minimalTimestep),
   _totalNumberOfCellUpdates(totalNumberOfCellUpdates),
   _minMeshWidth(minMeshWidth),
   _maxMeshWidth(maxMeshWidth),
   _numberOfInnerVertices(numberOfInnerVertices),
   _numberOfBoundaryVertices(numberOfBoundaryVertices),
   _numberOfOuterVertices(numberOfOuterVertices),
   _numberOfInnerCells(numberOfInnerCells),
   _numberOfOuterCells(numberOfOuterCells),
   _maxLevel(maxLevel),
   _isTraversalInverted(isTraversalInverted),
   _loadRebalancingState(loadRebalancingState) {
      setAllPatchesEvolvedToGlobalTimestep(allPatchesEvolvedToGlobalTimestep);
      setHasRefined(hasRefined);
      setHasTriggeredRefinementForNextIteration(hasTriggeredRefinementForNextIteration);
      setHasErased(hasErased);
      setHasTriggeredEraseForNextIteration(hasTriggeredEraseForNextIteration);
      setHasChangedVertexOrCellState(hasChangedVertexOrCellState);
      setReduceStateAndCell(reduceStateAndCell);
      assertion((7 < (8 * sizeof(short int))));
      
   }
   
   peanoclaw::records::StatePacked::StatePacked() {
      assertion((1 < (8 * sizeof(short int))));
      assertion((7 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::StatePacked::StatePacked(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._additionalLevelsForPredefinedRefinement, persistentRecords._isInitializing, persistentRecords._initialRefinmentTriggered, persistentRecords._unknownsPerSubcell, persistentRecords._auxiliarFieldsPerSubcell, persistentRecords._initialMinimalMeshWidth, persistentRecords._defaultSubdivisionFactor, persistentRecords._defaultGhostWidthLayer, persistentRecords._initialTimestepSize, persistentRecords._useDimensionalSplitting, persistentRecords._globalTimestepEndTime, persistentRecords.getAllPatchesEvolvedToGlobalTimestep(), persistentRecords._domainOffset, persistentRecords._domainSize, persistentRecords._plotNumber, persistentRecords._subPlotNumber, persistentRecords._startMaximumGlobalTimeInterval, persistentRecords._endMaximumGlobalTimeInterval, persistentRecords._startMinimumGlobalTimeInterval, persistentRecords._endMinimumGlobalTimeInterval, persistentRecords._minimalTimestep, persistentRecords._totalNumberOfCellUpdates, persistentRecords._minMeshWidth, persistentRecords._maxMeshWidth, persistentRecords._numberOfInnerVertices, persistentRecords._numberOfBoundaryVertices, persistentRecords._numberOfOuterVertices, persistentRecords._numberOfInnerCells, persistentRecords._numberOfOuterCells, persistentRecords._maxLevel, persistentRecords.getHasRefined(), persistentRecords.getHasTriggeredRefinementForNextIteration(), persistentRecords.getHasErased(), persistentRecords.getHasTriggeredEraseForNextIteration(), persistentRecords.getHasChangedVertexOrCellState(), persistentRecords._isTraversalInverted, persistentRecords._loadRebalancingState, persistentRecords.getReduceStateAndCell()) {
      assertion((1 < (8 * sizeof(short int))));
      assertion((7 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::StatePacked::StatePacked(const int& additionalLevelsForPredefinedRefinement, const bool& isInitializing, const bool& initialRefinmentTriggered, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const tarch::la::Vector<DIMENSIONS,double>& initialMinimalMeshWidth, const tarch::la::Vector<DIMENSIONS,int>& defaultSubdivisionFactor, const int& defaultGhostWidthLayer, const double& initialTimestepSize, const bool& useDimensionalSplitting, const double& globalTimestepEndTime, const bool& allPatchesEvolvedToGlobalTimestep, const tarch::la::Vector<DIMENSIONS,double>& domainOffset, const tarch::la::Vector<DIMENSIONS,double>& domainSize, const int& plotNumber, const int& subPlotNumber, const double& startMaximumGlobalTimeInterval, const double& endMaximumGlobalTimeInterval, const double& startMinimumGlobalTimeInterval, const double& endMinimumGlobalTimeInterval, const double& minimalTimestep, const double& totalNumberOfCellUpdates, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& isTraversalInverted, const LoadBalancingState& loadRebalancingState, const bool& reduceStateAndCell):
   _persistentRecords(additionalLevelsForPredefinedRefinement, isInitializing, initialRefinmentTriggered, unknownsPerSubcell, auxiliarFieldsPerSubcell, initialMinimalMeshWidth, defaultSubdivisionFactor, defaultGhostWidthLayer, initialTimestepSize, useDimensionalSplitting, globalTimestepEndTime, allPatchesEvolvedToGlobalTimestep, domainOffset, domainSize, plotNumber, subPlotNumber, startMaximumGlobalTimeInterval, endMaximumGlobalTimeInterval, startMinimumGlobalTimeInterval, endMinimumGlobalTimeInterval, minimalTimestep, totalNumberOfCellUpdates, minMeshWidth, maxMeshWidth, numberOfInnerVertices, numberOfBoundaryVertices, numberOfOuterVertices, numberOfInnerCells, numberOfOuterCells, maxLevel, hasRefined, hasTriggeredRefinementForNextIteration, hasErased, hasTriggeredEraseForNextIteration, hasChangedVertexOrCellState, isTraversalInverted, loadRebalancingState, reduceStateAndCell) {
      assertion((1 < (8 * sizeof(short int))));
      assertion((7 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::StatePacked::StatePacked(const int& additionalLevelsForPredefinedRefinement, const bool& isInitializing, const bool& initialRefinmentTriggered, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const tarch::la::Vector<DIMENSIONS,double>& initialMinimalMeshWidth, const tarch::la::Vector<DIMENSIONS,int>& defaultSubdivisionFactor, const int& defaultGhostWidthLayer, const double& initialTimestepSize, const bool& useDimensionalSplitting, const double& globalTimestepEndTime, const bool& allPatchesEvolvedToGlobalTimestep, const tarch::la::Vector<DIMENSIONS,double>& domainOffset, const tarch::la::Vector<DIMENSIONS,double>& domainSize, const int& plotNumber, const int& subPlotNumber, const double& startMaximumGlobalTimeInterval, const double& endMaximumGlobalTimeInterval, const double& startMinimumGlobalTimeInterval, const double& endMinimumGlobalTimeInterval, const double& minimalTimestep, const double& totalNumberOfCellUpdates, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& isTraversalInverted, const LoadBalancingState& loadRebalancingState, const bool& reduceStateAndCell, const bool& hasWorkerWithWorker):
   _persistentRecords(additionalLevelsForPredefinedRefinement, isInitializing, initialRefinmentTriggered, unknownsPerSubcell, auxiliarFieldsPerSubcell, initialMinimalMeshWidth, defaultSubdivisionFactor, defaultGhostWidthLayer, initialTimestepSize, useDimensionalSplitting, globalTimestepEndTime, allPatchesEvolvedToGlobalTimestep, domainOffset, domainSize, plotNumber, subPlotNumber, startMaximumGlobalTimeInterval, endMaximumGlobalTimeInterval, startMinimumGlobalTimeInterval, endMinimumGlobalTimeInterval, minimalTimestep, totalNumberOfCellUpdates, minMeshWidth, maxMeshWidth, numberOfInnerVertices, numberOfBoundaryVertices, numberOfOuterVertices, numberOfInnerCells, numberOfOuterCells, maxLevel, hasRefined, hasTriggeredRefinementForNextIteration, hasErased, hasTriggeredEraseForNextIteration, hasChangedVertexOrCellState, isTraversalInverted, loadRebalancingState, reduceStateAndCell) {
      setHasWorkerWithWorker(hasWorkerWithWorker);
      assertion((1 < (8 * sizeof(short int))));
      assertion((7 < (8 * sizeof(short int))));
      
   }
   
   peanoclaw::records::StatePacked::~StatePacked() { }
   
   std::string peanoclaw::records::StatePacked::toString(const LoadBalancingState& param) {
      return peanoclaw::records::State::toString(param);
   }
   
   std::string peanoclaw::records::StatePacked::getLoadBalancingStateMapping() {
      return peanoclaw::records::State::getLoadBalancingStateMapping();
   }
   
   
   
   std::string peanoclaw::records::StatePacked::toString() const {
      std::ostringstream stringstr;
      toString(stringstr);
      return stringstr.str();
   }
   
   void peanoclaw::records::StatePacked::toString (std::ostream& out) const {
      out << "("; 
      out << "additionalLevelsForPredefinedRefinement:" << getAdditionalLevelsForPredefinedRefinement();
      out << ",";
      out << "isInitializing:" << getIsInitializing();
      out << ",";
      out << "initialRefinmentTriggered:" << getInitialRefinmentTriggered();
      out << ",";
      out << "unknownsPerSubcell:" << getUnknownsPerSubcell();
      out << ",";
      out << "auxiliarFieldsPerSubcell:" << getAuxiliarFieldsPerSubcell();
      out << ",";
      out << "initialMinimalMeshWidth:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getInitialMinimalMeshWidth(i) << ",";
   }
   out << getInitialMinimalMeshWidth(DIMENSIONS-1) << "]";
      out << ",";
      out << "defaultSubdivisionFactor:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getDefaultSubdivisionFactor(i) << ",";
   }
   out << getDefaultSubdivisionFactor(DIMENSIONS-1) << "]";
      out << ",";
      out << "defaultGhostWidthLayer:" << getDefaultGhostWidthLayer();
      out << ",";
      out << "initialTimestepSize:" << getInitialTimestepSize();
      out << ",";
      out << "useDimensionalSplitting:" << getUseDimensionalSplitting();
      out << ",";
      out << "globalTimestepEndTime:" << getGlobalTimestepEndTime();
      out << ",";
      out << "allPatchesEvolvedToGlobalTimestep:" << getAllPatchesEvolvedToGlobalTimestep();
      out << ",";
      out << "domainOffset:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getDomainOffset(i) << ",";
   }
   out << getDomainOffset(DIMENSIONS-1) << "]";
      out << ",";
      out << "domainSize:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getDomainSize(i) << ",";
   }
   out << getDomainSize(DIMENSIONS-1) << "]";
      out << ",";
      out << "plotNumber:" << getPlotNumber();
      out << ",";
      out << "subPlotNumber:" << getSubPlotNumber();
      out << ",";
      out << "startMaximumGlobalTimeInterval:" << getStartMaximumGlobalTimeInterval();
      out << ",";
      out << "endMaximumGlobalTimeInterval:" << getEndMaximumGlobalTimeInterval();
      out << ",";
      out << "startMinimumGlobalTimeInterval:" << getStartMinimumGlobalTimeInterval();
      out << ",";
      out << "endMinimumGlobalTimeInterval:" << getEndMinimumGlobalTimeInterval();
      out << ",";
      out << "minimalTimestep:" << getMinimalTimestep();
      out << ",";
      out << "totalNumberOfCellUpdates:" << getTotalNumberOfCellUpdates();
      out << ",";
      out << "minMeshWidth:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getMinMeshWidth(i) << ",";
   }
   out << getMinMeshWidth(DIMENSIONS-1) << "]";
      out << ",";
      out << "maxMeshWidth:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getMaxMeshWidth(i) << ",";
   }
   out << getMaxMeshWidth(DIMENSIONS-1) << "]";
      out << ",";
      out << "numberOfInnerVertices:" << getNumberOfInnerVertices();
      out << ",";
      out << "numberOfBoundaryVertices:" << getNumberOfBoundaryVertices();
      out << ",";
      out << "numberOfOuterVertices:" << getNumberOfOuterVertices();
      out << ",";
      out << "numberOfInnerCells:" << getNumberOfInnerCells();
      out << ",";
      out << "numberOfOuterCells:" << getNumberOfOuterCells();
      out << ",";
      out << "maxLevel:" << getMaxLevel();
      out << ",";
      out << "hasRefined:" << getHasRefined();
      out << ",";
      out << "hasTriggeredRefinementForNextIteration:" << getHasTriggeredRefinementForNextIteration();
      out << ",";
      out << "hasErased:" << getHasErased();
      out << ",";
      out << "hasTriggeredEraseForNextIteration:" << getHasTriggeredEraseForNextIteration();
      out << ",";
      out << "hasChangedVertexOrCellState:" << getHasChangedVertexOrCellState();
      out << ",";
      out << "isTraversalInverted:" << getIsTraversalInverted();
      out << ",";
      out << "loadRebalancingState:" << toString(getLoadRebalancingState());
      out << ",";
      out << "reduceStateAndCell:" << getReduceStateAndCell();
      out << ",";
      out << "hasWorkerWithWorker:" << getHasWorkerWithWorker();
      out <<  ")";
   }
   
   
   peanoclaw::records::StatePacked::PersistentRecords peanoclaw::records::StatePacked::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::State peanoclaw::records::StatePacked::convert() const{
      return State(
         getAdditionalLevelsForPredefinedRefinement(),
         getIsInitializing(),
         getInitialRefinmentTriggered(),
         getUnknownsPerSubcell(),
         getAuxiliarFieldsPerSubcell(),
         getInitialMinimalMeshWidth(),
         getDefaultSubdivisionFactor(),
         getDefaultGhostWidthLayer(),
         getInitialTimestepSize(),
         getUseDimensionalSplitting(),
         getGlobalTimestepEndTime(),
         getAllPatchesEvolvedToGlobalTimestep(),
         getDomainOffset(),
         getDomainSize(),
         getPlotNumber(),
         getSubPlotNumber(),
         getStartMaximumGlobalTimeInterval(),
         getEndMaximumGlobalTimeInterval(),
         getStartMinimumGlobalTimeInterval(),
         getEndMinimumGlobalTimeInterval(),
         getMinimalTimestep(),
         getTotalNumberOfCellUpdates(),
         getMinMeshWidth(),
         getMaxMeshWidth(),
         getNumberOfInnerVertices(),
         getNumberOfBoundaryVertices(),
         getNumberOfOuterVertices(),
         getNumberOfInnerCells(),
         getNumberOfOuterCells(),
         getMaxLevel(),
         getHasRefined(),
         getHasTriggeredRefinementForNextIteration(),
         getHasErased(),
         getHasTriggeredEraseForNextIteration(),
         getHasChangedVertexOrCellState(),
         getIsTraversalInverted(),
         getLoadRebalancingState(),
         getReduceStateAndCell(),
         getHasWorkerWithWorker()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::StatePacked::_log( "peanoclaw::records::StatePacked" );
      
      MPI_Datatype peanoclaw::records::StatePacked::Datatype = 0;
      MPI_Datatype peanoclaw::records::StatePacked::FullDatatype = 0;
      
      
      void peanoclaw::records::StatePacked::initDatatype() {
         {
            StatePacked dummyStatePacked[2];
            
            const int Attributes = 33;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //additionalLevelsForPredefinedRefinement
               MPI_CHAR,		 //isInitializing
               MPI_CHAR,		 //initialRefinmentTriggered
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_DOUBLE,		 //initialMinimalMeshWidth
               MPI_INT,		 //defaultSubdivisionFactor
               MPI_INT,		 //defaultGhostWidthLayer
               MPI_DOUBLE,		 //initialTimestepSize
               MPI_CHAR,		 //useDimensionalSplitting
               MPI_DOUBLE,		 //globalTimestepEndTime
               MPI_DOUBLE,		 //domainOffset
               MPI_DOUBLE,		 //domainSize
               MPI_INT,		 //plotNumber
               MPI_INT,		 //subPlotNumber
               MPI_DOUBLE,		 //startMaximumGlobalTimeInterval
               MPI_DOUBLE,		 //endMaximumGlobalTimeInterval
               MPI_DOUBLE,		 //startMinimumGlobalTimeInterval
               MPI_DOUBLE,		 //endMinimumGlobalTimeInterval
               MPI_DOUBLE,		 //minimalTimestep
               MPI_DOUBLE,		 //totalNumberOfCellUpdates
               MPI_DOUBLE,		 //minMeshWidth
               MPI_DOUBLE,		 //maxMeshWidth
               MPI_DOUBLE,		 //numberOfInnerVertices
               MPI_DOUBLE,		 //numberOfBoundaryVertices
               MPI_DOUBLE,		 //numberOfOuterVertices
               MPI_DOUBLE,		 //numberOfInnerCells
               MPI_DOUBLE,		 //numberOfOuterCells
               MPI_INT,		 //maxLevel
               MPI_CHAR,		 //isTraversalInverted
               MPI_SHORT,		 //_packedRecords0
               MPI_SHORT,		 //_packedRecords0
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //additionalLevelsForPredefinedRefinement
               1,		 //isInitializing
               1,		 //initialRefinmentTriggered
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               DIMENSIONS,		 //initialMinimalMeshWidth
               DIMENSIONS,		 //defaultSubdivisionFactor
               1,		 //defaultGhostWidthLayer
               1,		 //initialTimestepSize
               1,		 //useDimensionalSplitting
               1,		 //globalTimestepEndTime
               DIMENSIONS,		 //domainOffset
               DIMENSIONS,		 //domainSize
               1,		 //plotNumber
               1,		 //subPlotNumber
               1,		 //startMaximumGlobalTimeInterval
               1,		 //endMaximumGlobalTimeInterval
               1,		 //startMinimumGlobalTimeInterval
               1,		 //endMinimumGlobalTimeInterval
               1,		 //minimalTimestep
               1,		 //totalNumberOfCellUpdates
               DIMENSIONS,		 //minMeshWidth
               DIMENSIONS,		 //maxMeshWidth
               1,		 //numberOfInnerVertices
               1,		 //numberOfBoundaryVertices
               1,		 //numberOfOuterVertices
               1,		 //numberOfInnerCells
               1,		 //numberOfOuterCells
               1,		 //maxLevel
               1,		 //isTraversalInverted
               1,		 //_packedRecords0
               1,		 //_packedRecords0
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._isInitializing))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialRefinmentTriggered))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._unknownsPerSubcell))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialMinimalMeshWidth[0]))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._defaultSubdivisionFactor[0]))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._defaultGhostWidthLayer))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialTimestepSize))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._useDimensionalSplitting))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._globalTimestepEndTime))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._domainOffset[0]))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._domainSize[0]))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._plotNumber))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._subPlotNumber))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._startMaximumGlobalTimeInterval))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._endMaximumGlobalTimeInterval))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._startMinimumGlobalTimeInterval))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._endMinimumGlobalTimeInterval))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._minimalTimestep))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._totalNumberOfCellUpdates))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._minMeshWidth[0]))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._maxMeshWidth[0]))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerVertices))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfBoundaryVertices))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterVertices))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerCells))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterCells))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._maxLevel))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._isTraversalInverted))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._packedRecords0))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._packedRecords0))), 		&disp[31] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[1]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[32] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &StatePacked::Datatype );
            MPI_Type_commit( &StatePacked::Datatype );
            
         }
         {
            StatePacked dummyStatePacked[2];
            
            const int Attributes = 34;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //additionalLevelsForPredefinedRefinement
               MPI_CHAR,		 //isInitializing
               MPI_CHAR,		 //initialRefinmentTriggered
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_DOUBLE,		 //initialMinimalMeshWidth
               MPI_INT,		 //defaultSubdivisionFactor
               MPI_INT,		 //defaultGhostWidthLayer
               MPI_DOUBLE,		 //initialTimestepSize
               MPI_CHAR,		 //useDimensionalSplitting
               MPI_DOUBLE,		 //globalTimestepEndTime
               MPI_DOUBLE,		 //domainOffset
               MPI_DOUBLE,		 //domainSize
               MPI_INT,		 //plotNumber
               MPI_INT,		 //subPlotNumber
               MPI_DOUBLE,		 //startMaximumGlobalTimeInterval
               MPI_DOUBLE,		 //endMaximumGlobalTimeInterval
               MPI_DOUBLE,		 //startMinimumGlobalTimeInterval
               MPI_DOUBLE,		 //endMinimumGlobalTimeInterval
               MPI_DOUBLE,		 //minimalTimestep
               MPI_DOUBLE,		 //totalNumberOfCellUpdates
               MPI_DOUBLE,		 //minMeshWidth
               MPI_DOUBLE,		 //maxMeshWidth
               MPI_DOUBLE,		 //numberOfInnerVertices
               MPI_DOUBLE,		 //numberOfBoundaryVertices
               MPI_DOUBLE,		 //numberOfOuterVertices
               MPI_DOUBLE,		 //numberOfInnerCells
               MPI_DOUBLE,		 //numberOfOuterCells
               MPI_INT,		 //maxLevel
               MPI_CHAR,		 //isTraversalInverted
               MPI_INT,		 //loadRebalancingState
               MPI_SHORT,		 //_packedRecords0
               MPI_SHORT,		 //_packedRecords0
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //additionalLevelsForPredefinedRefinement
               1,		 //isInitializing
               1,		 //initialRefinmentTriggered
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               DIMENSIONS,		 //initialMinimalMeshWidth
               DIMENSIONS,		 //defaultSubdivisionFactor
               1,		 //defaultGhostWidthLayer
               1,		 //initialTimestepSize
               1,		 //useDimensionalSplitting
               1,		 //globalTimestepEndTime
               DIMENSIONS,		 //domainOffset
               DIMENSIONS,		 //domainSize
               1,		 //plotNumber
               1,		 //subPlotNumber
               1,		 //startMaximumGlobalTimeInterval
               1,		 //endMaximumGlobalTimeInterval
               1,		 //startMinimumGlobalTimeInterval
               1,		 //endMinimumGlobalTimeInterval
               1,		 //minimalTimestep
               1,		 //totalNumberOfCellUpdates
               DIMENSIONS,		 //minMeshWidth
               DIMENSIONS,		 //maxMeshWidth
               1,		 //numberOfInnerVertices
               1,		 //numberOfBoundaryVertices
               1,		 //numberOfOuterVertices
               1,		 //numberOfInnerCells
               1,		 //numberOfOuterCells
               1,		 //maxLevel
               1,		 //isTraversalInverted
               1,		 //loadRebalancingState
               1,		 //_packedRecords0
               1,		 //_packedRecords0
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._isInitializing))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialRefinmentTriggered))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._unknownsPerSubcell))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialMinimalMeshWidth[0]))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._defaultSubdivisionFactor[0]))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._defaultGhostWidthLayer))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialTimestepSize))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._useDimensionalSplitting))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._globalTimestepEndTime))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._domainOffset[0]))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._domainSize[0]))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._plotNumber))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._subPlotNumber))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._startMaximumGlobalTimeInterval))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._endMaximumGlobalTimeInterval))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._startMinimumGlobalTimeInterval))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._endMinimumGlobalTimeInterval))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._minimalTimestep))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._totalNumberOfCellUpdates))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._minMeshWidth[0]))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._maxMeshWidth[0]))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerVertices))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfBoundaryVertices))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterVertices))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerCells))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterCells))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._maxLevel))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._isTraversalInverted))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._loadRebalancingState))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._packedRecords0))), 		&disp[31] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._packedRecords0))), 		&disp[32] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[1]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[33] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &StatePacked::FullDatatype );
            MPI_Type_commit( &StatePacked::FullDatatype );
            
         }
         
      }
      
      
      void peanoclaw::records::StatePacked::shutdownDatatype() {
         MPI_Type_free( &StatePacked::Datatype );
         MPI_Type_free( &StatePacked::FullDatatype );
         
      }
      
      void peanoclaw::records::StatePacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
         MPI_Request* sendRequestHandle = new MPI_Request();
         MPI_Status   status;
         int          flag = 0;
         int          result;
         
         clock_t      timeOutWarning   = -1;
         clock_t      timeOutShutdown  = -1;
         bool         triggeredTimeoutWarning = false;
         
         #ifdef Asserts
         _senderRank = -1;
         #endif
         
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
            msg << "was not able to send message peanoclaw::records::StatePacked "
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
               msg << "testing for finished send task for peanoclaw::records::StatePacked "
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
               "peanoclaw::records::StatePacked",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::StatePacked",
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
      
      
      
      void peanoclaw::records::StatePacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "failed to start to receive peanoclaw::records::StatePacked from node "
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
               msg << "testing for finished receive task for peanoclaw::records::StatePacked failed: "
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
               "peanoclaw::records::StatePacked",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::StatePacked",
               "receive(int)", source,tag,1
               );
            }
            tarch::parallel::Node::getInstance().receiveDanglingMessages();
         }
         
         delete sendRequestHandle;
         
         _senderRank = status.MPI_SOURCE;
         #ifdef Debug
         _log.debug("receive(int,int)", "received " + toString() ); 
         #endif
         
      }
      
      
      
      bool peanoclaw::records::StatePacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
      
      int peanoclaw::records::StatePacked::getSenderRank() const {
         assertion( _senderRank!=-1 );
         return _senderRank;
         
      }
   #endif
   
   
   
#elif !defined(Parallel)
   peanoclaw::records::State::PersistentRecords::PersistentRecords() {
      
   }
   
   
   peanoclaw::records::State::PersistentRecords::PersistentRecords(const int& additionalLevelsForPredefinedRefinement, const bool& isInitializing, const bool& initialRefinmentTriggered, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const tarch::la::Vector<DIMENSIONS,double>& initialMinimalMeshWidth, const tarch::la::Vector<DIMENSIONS,int>& defaultSubdivisionFactor, const int& defaultGhostWidthLayer, const double& initialTimestepSize, const bool& useDimensionalSplitting, const double& globalTimestepEndTime, const bool& allPatchesEvolvedToGlobalTimestep, const tarch::la::Vector<DIMENSIONS,double>& domainOffset, const tarch::la::Vector<DIMENSIONS,double>& domainSize, const int& plotNumber, const int& subPlotNumber, const double& startMaximumGlobalTimeInterval, const double& endMaximumGlobalTimeInterval, const double& startMinimumGlobalTimeInterval, const double& endMinimumGlobalTimeInterval, const double& minimalTimestep, const double& totalNumberOfCellUpdates, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& isTraversalInverted):
   _additionalLevelsForPredefinedRefinement(additionalLevelsForPredefinedRefinement),
   _isInitializing(isInitializing),
   _initialRefinmentTriggered(initialRefinmentTriggered),
   _unknownsPerSubcell(unknownsPerSubcell),
   _auxiliarFieldsPerSubcell(auxiliarFieldsPerSubcell),
   _initialMinimalMeshWidth(initialMinimalMeshWidth),
   _defaultSubdivisionFactor(defaultSubdivisionFactor),
   _defaultGhostWidthLayer(defaultGhostWidthLayer),
   _initialTimestepSize(initialTimestepSize),
   _useDimensionalSplitting(useDimensionalSplitting),
   _globalTimestepEndTime(globalTimestepEndTime),
   _allPatchesEvolvedToGlobalTimestep(allPatchesEvolvedToGlobalTimestep),
   _domainOffset(domainOffset),
   _domainSize(domainSize),
   _plotNumber(plotNumber),
   _subPlotNumber(subPlotNumber),
   _startMaximumGlobalTimeInterval(startMaximumGlobalTimeInterval),
   _endMaximumGlobalTimeInterval(endMaximumGlobalTimeInterval),
   _startMinimumGlobalTimeInterval(startMinimumGlobalTimeInterval),
   _endMinimumGlobalTimeInterval(endMinimumGlobalTimeInterval),
   _minimalTimestep(minimalTimestep),
   _totalNumberOfCellUpdates(totalNumberOfCellUpdates),
   _minMeshWidth(minMeshWidth),
   _maxMeshWidth(maxMeshWidth),
   _numberOfInnerVertices(numberOfInnerVertices),
   _numberOfBoundaryVertices(numberOfBoundaryVertices),
   _numberOfOuterVertices(numberOfOuterVertices),
   _numberOfInnerCells(numberOfInnerCells),
   _numberOfOuterCells(numberOfOuterCells),
   _maxLevel(maxLevel),
   _hasRefined(hasRefined),
   _hasTriggeredRefinementForNextIteration(hasTriggeredRefinementForNextIteration),
   _hasErased(hasErased),
   _hasTriggeredEraseForNextIteration(hasTriggeredEraseForNextIteration),
   _hasChangedVertexOrCellState(hasChangedVertexOrCellState),
   _isTraversalInverted(isTraversalInverted) {
      
   }
   
   peanoclaw::records::State::State() {
      
   }
   
   
   peanoclaw::records::State::State(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._additionalLevelsForPredefinedRefinement, persistentRecords._isInitializing, persistentRecords._initialRefinmentTriggered, persistentRecords._unknownsPerSubcell, persistentRecords._auxiliarFieldsPerSubcell, persistentRecords._initialMinimalMeshWidth, persistentRecords._defaultSubdivisionFactor, persistentRecords._defaultGhostWidthLayer, persistentRecords._initialTimestepSize, persistentRecords._useDimensionalSplitting, persistentRecords._globalTimestepEndTime, persistentRecords._allPatchesEvolvedToGlobalTimestep, persistentRecords._domainOffset, persistentRecords._domainSize, persistentRecords._plotNumber, persistentRecords._subPlotNumber, persistentRecords._startMaximumGlobalTimeInterval, persistentRecords._endMaximumGlobalTimeInterval, persistentRecords._startMinimumGlobalTimeInterval, persistentRecords._endMinimumGlobalTimeInterval, persistentRecords._minimalTimestep, persistentRecords._totalNumberOfCellUpdates, persistentRecords._minMeshWidth, persistentRecords._maxMeshWidth, persistentRecords._numberOfInnerVertices, persistentRecords._numberOfBoundaryVertices, persistentRecords._numberOfOuterVertices, persistentRecords._numberOfInnerCells, persistentRecords._numberOfOuterCells, persistentRecords._maxLevel, persistentRecords._hasRefined, persistentRecords._hasTriggeredRefinementForNextIteration, persistentRecords._hasErased, persistentRecords._hasTriggeredEraseForNextIteration, persistentRecords._hasChangedVertexOrCellState, persistentRecords._isTraversalInverted) {
      
   }
   
   
   peanoclaw::records::State::State(const int& additionalLevelsForPredefinedRefinement, const bool& isInitializing, const bool& initialRefinmentTriggered, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const tarch::la::Vector<DIMENSIONS,double>& initialMinimalMeshWidth, const tarch::la::Vector<DIMENSIONS,int>& defaultSubdivisionFactor, const int& defaultGhostWidthLayer, const double& initialTimestepSize, const bool& useDimensionalSplitting, const double& globalTimestepEndTime, const bool& allPatchesEvolvedToGlobalTimestep, const tarch::la::Vector<DIMENSIONS,double>& domainOffset, const tarch::la::Vector<DIMENSIONS,double>& domainSize, const int& plotNumber, const int& subPlotNumber, const double& startMaximumGlobalTimeInterval, const double& endMaximumGlobalTimeInterval, const double& startMinimumGlobalTimeInterval, const double& endMinimumGlobalTimeInterval, const double& minimalTimestep, const double& totalNumberOfCellUpdates, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& isTraversalInverted):
   _persistentRecords(additionalLevelsForPredefinedRefinement, isInitializing, initialRefinmentTriggered, unknownsPerSubcell, auxiliarFieldsPerSubcell, initialMinimalMeshWidth, defaultSubdivisionFactor, defaultGhostWidthLayer, initialTimestepSize, useDimensionalSplitting, globalTimestepEndTime, allPatchesEvolvedToGlobalTimestep, domainOffset, domainSize, plotNumber, subPlotNumber, startMaximumGlobalTimeInterval, endMaximumGlobalTimeInterval, startMinimumGlobalTimeInterval, endMinimumGlobalTimeInterval, minimalTimestep, totalNumberOfCellUpdates, minMeshWidth, maxMeshWidth, numberOfInnerVertices, numberOfBoundaryVertices, numberOfOuterVertices, numberOfInnerCells, numberOfOuterCells, maxLevel, hasRefined, hasTriggeredRefinementForNextIteration, hasErased, hasTriggeredEraseForNextIteration, hasChangedVertexOrCellState, isTraversalInverted) {
      
   }
   
   
   peanoclaw::records::State::~State() { }
   
   
   
   std::string peanoclaw::records::State::toString() const {
      std::ostringstream stringstr;
      toString(stringstr);
      return stringstr.str();
   }
   
   void peanoclaw::records::State::toString (std::ostream& out) const {
      out << "("; 
      out << "additionalLevelsForPredefinedRefinement:" << getAdditionalLevelsForPredefinedRefinement();
      out << ",";
      out << "isInitializing:" << getIsInitializing();
      out << ",";
      out << "initialRefinmentTriggered:" << getInitialRefinmentTriggered();
      out << ",";
      out << "unknownsPerSubcell:" << getUnknownsPerSubcell();
      out << ",";
      out << "auxiliarFieldsPerSubcell:" << getAuxiliarFieldsPerSubcell();
      out << ",";
      out << "initialMinimalMeshWidth:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getInitialMinimalMeshWidth(i) << ",";
   }
   out << getInitialMinimalMeshWidth(DIMENSIONS-1) << "]";
      out << ",";
      out << "defaultSubdivisionFactor:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getDefaultSubdivisionFactor(i) << ",";
   }
   out << getDefaultSubdivisionFactor(DIMENSIONS-1) << "]";
      out << ",";
      out << "defaultGhostWidthLayer:" << getDefaultGhostWidthLayer();
      out << ",";
      out << "initialTimestepSize:" << getInitialTimestepSize();
      out << ",";
      out << "useDimensionalSplitting:" << getUseDimensionalSplitting();
      out << ",";
      out << "globalTimestepEndTime:" << getGlobalTimestepEndTime();
      out << ",";
      out << "allPatchesEvolvedToGlobalTimestep:" << getAllPatchesEvolvedToGlobalTimestep();
      out << ",";
      out << "domainOffset:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getDomainOffset(i) << ",";
   }
   out << getDomainOffset(DIMENSIONS-1) << "]";
      out << ",";
      out << "domainSize:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getDomainSize(i) << ",";
   }
   out << getDomainSize(DIMENSIONS-1) << "]";
      out << ",";
      out << "plotNumber:" << getPlotNumber();
      out << ",";
      out << "subPlotNumber:" << getSubPlotNumber();
      out << ",";
      out << "startMaximumGlobalTimeInterval:" << getStartMaximumGlobalTimeInterval();
      out << ",";
      out << "endMaximumGlobalTimeInterval:" << getEndMaximumGlobalTimeInterval();
      out << ",";
      out << "startMinimumGlobalTimeInterval:" << getStartMinimumGlobalTimeInterval();
      out << ",";
      out << "endMinimumGlobalTimeInterval:" << getEndMinimumGlobalTimeInterval();
      out << ",";
      out << "minimalTimestep:" << getMinimalTimestep();
      out << ",";
      out << "totalNumberOfCellUpdates:" << getTotalNumberOfCellUpdates();
      out << ",";
      out << "minMeshWidth:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getMinMeshWidth(i) << ",";
   }
   out << getMinMeshWidth(DIMENSIONS-1) << "]";
      out << ",";
      out << "maxMeshWidth:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getMaxMeshWidth(i) << ",";
   }
   out << getMaxMeshWidth(DIMENSIONS-1) << "]";
      out << ",";
      out << "numberOfInnerVertices:" << getNumberOfInnerVertices();
      out << ",";
      out << "numberOfBoundaryVertices:" << getNumberOfBoundaryVertices();
      out << ",";
      out << "numberOfOuterVertices:" << getNumberOfOuterVertices();
      out << ",";
      out << "numberOfInnerCells:" << getNumberOfInnerCells();
      out << ",";
      out << "numberOfOuterCells:" << getNumberOfOuterCells();
      out << ",";
      out << "maxLevel:" << getMaxLevel();
      out << ",";
      out << "hasRefined:" << getHasRefined();
      out << ",";
      out << "hasTriggeredRefinementForNextIteration:" << getHasTriggeredRefinementForNextIteration();
      out << ",";
      out << "hasErased:" << getHasErased();
      out << ",";
      out << "hasTriggeredEraseForNextIteration:" << getHasTriggeredEraseForNextIteration();
      out << ",";
      out << "hasChangedVertexOrCellState:" << getHasChangedVertexOrCellState();
      out << ",";
      out << "isTraversalInverted:" << getIsTraversalInverted();
      out <<  ")";
   }
   
   
   peanoclaw::records::State::PersistentRecords peanoclaw::records::State::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::StatePacked peanoclaw::records::State::convert() const{
      return StatePacked(
         getAdditionalLevelsForPredefinedRefinement(),
         getIsInitializing(),
         getInitialRefinmentTriggered(),
         getUnknownsPerSubcell(),
         getAuxiliarFieldsPerSubcell(),
         getInitialMinimalMeshWidth(),
         getDefaultSubdivisionFactor(),
         getDefaultGhostWidthLayer(),
         getInitialTimestepSize(),
         getUseDimensionalSplitting(),
         getGlobalTimestepEndTime(),
         getAllPatchesEvolvedToGlobalTimestep(),
         getDomainOffset(),
         getDomainSize(),
         getPlotNumber(),
         getSubPlotNumber(),
         getStartMaximumGlobalTimeInterval(),
         getEndMaximumGlobalTimeInterval(),
         getStartMinimumGlobalTimeInterval(),
         getEndMinimumGlobalTimeInterval(),
         getMinimalTimestep(),
         getTotalNumberOfCellUpdates(),
         getMinMeshWidth(),
         getMaxMeshWidth(),
         getNumberOfInnerVertices(),
         getNumberOfBoundaryVertices(),
         getNumberOfOuterVertices(),
         getNumberOfInnerCells(),
         getNumberOfOuterCells(),
         getMaxLevel(),
         getHasRefined(),
         getHasTriggeredRefinementForNextIteration(),
         getHasErased(),
         getHasTriggeredEraseForNextIteration(),
         getHasChangedVertexOrCellState(),
         getIsTraversalInverted()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::State::_log( "peanoclaw::records::State" );
      
      MPI_Datatype peanoclaw::records::State::Datatype = 0;
      MPI_Datatype peanoclaw::records::State::FullDatatype = 0;
      
      
      void peanoclaw::records::State::initDatatype() {
         {
            State dummyState[2];
            
            const int Attributes = 37;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //additionalLevelsForPredefinedRefinement
               MPI_CHAR,		 //isInitializing
               MPI_CHAR,		 //initialRefinmentTriggered
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_DOUBLE,		 //initialMinimalMeshWidth
               MPI_INT,		 //defaultSubdivisionFactor
               MPI_INT,		 //defaultGhostWidthLayer
               MPI_DOUBLE,		 //initialTimestepSize
               MPI_CHAR,		 //useDimensionalSplitting
               MPI_DOUBLE,		 //globalTimestepEndTime
               MPI_CHAR,		 //allPatchesEvolvedToGlobalTimestep
               MPI_DOUBLE,		 //domainOffset
               MPI_DOUBLE,		 //domainSize
               MPI_INT,		 //plotNumber
               MPI_INT,		 //subPlotNumber
               MPI_DOUBLE,		 //startMaximumGlobalTimeInterval
               MPI_DOUBLE,		 //endMaximumGlobalTimeInterval
               MPI_DOUBLE,		 //startMinimumGlobalTimeInterval
               MPI_DOUBLE,		 //endMinimumGlobalTimeInterval
               MPI_DOUBLE,		 //minimalTimestep
               MPI_DOUBLE,		 //totalNumberOfCellUpdates
               MPI_DOUBLE,		 //minMeshWidth
               MPI_DOUBLE,		 //maxMeshWidth
               MPI_DOUBLE,		 //numberOfInnerVertices
               MPI_DOUBLE,		 //numberOfBoundaryVertices
               MPI_DOUBLE,		 //numberOfOuterVertices
               MPI_DOUBLE,		 //numberOfInnerCells
               MPI_DOUBLE,		 //numberOfOuterCells
               MPI_INT,		 //maxLevel
               MPI_CHAR,		 //hasRefined
               MPI_CHAR,		 //hasTriggeredRefinementForNextIteration
               MPI_CHAR,		 //hasErased
               MPI_CHAR,		 //hasTriggeredEraseForNextIteration
               MPI_CHAR,		 //hasChangedVertexOrCellState
               MPI_CHAR,		 //isTraversalInverted
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //additionalLevelsForPredefinedRefinement
               1,		 //isInitializing
               1,		 //initialRefinmentTriggered
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               DIMENSIONS,		 //initialMinimalMeshWidth
               DIMENSIONS,		 //defaultSubdivisionFactor
               1,		 //defaultGhostWidthLayer
               1,		 //initialTimestepSize
               1,		 //useDimensionalSplitting
               1,		 //globalTimestepEndTime
               1,		 //allPatchesEvolvedToGlobalTimestep
               DIMENSIONS,		 //domainOffset
               DIMENSIONS,		 //domainSize
               1,		 //plotNumber
               1,		 //subPlotNumber
               1,		 //startMaximumGlobalTimeInterval
               1,		 //endMaximumGlobalTimeInterval
               1,		 //startMinimumGlobalTimeInterval
               1,		 //endMinimumGlobalTimeInterval
               1,		 //minimalTimestep
               1,		 //totalNumberOfCellUpdates
               DIMENSIONS,		 //minMeshWidth
               DIMENSIONS,		 //maxMeshWidth
               1,		 //numberOfInnerVertices
               1,		 //numberOfBoundaryVertices
               1,		 //numberOfOuterVertices
               1,		 //numberOfInnerCells
               1,		 //numberOfOuterCells
               1,		 //maxLevel
               1,		 //hasRefined
               1,		 //hasTriggeredRefinementForNextIteration
               1,		 //hasErased
               1,		 //hasTriggeredEraseForNextIteration
               1,		 //hasChangedVertexOrCellState
               1,		 //isTraversalInverted
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._isInitializing))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialRefinmentTriggered))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._unknownsPerSubcell))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialMinimalMeshWidth[0]))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._defaultSubdivisionFactor[0]))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._defaultGhostWidthLayer))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialTimestepSize))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._useDimensionalSplitting))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._globalTimestepEndTime))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._allPatchesEvolvedToGlobalTimestep))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._domainOffset[0]))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._domainSize[0]))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._plotNumber))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._subPlotNumber))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._startMaximumGlobalTimeInterval))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._endMaximumGlobalTimeInterval))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._startMinimumGlobalTimeInterval))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._endMinimumGlobalTimeInterval))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._minimalTimestep))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._totalNumberOfCellUpdates))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._minMeshWidth[0]))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._maxMeshWidth[0]))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerVertices))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfBoundaryVertices))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterVertices))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerCells))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterCells))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._maxLevel))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasRefined))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasTriggeredRefinementForNextIteration))), 		&disp[31] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasErased))), 		&disp[32] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasTriggeredEraseForNextIteration))), 		&disp[33] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasChangedVertexOrCellState))), 		&disp[34] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._isTraversalInverted))), 		&disp[35] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[1]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[36] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &State::Datatype );
            MPI_Type_commit( &State::Datatype );
            
         }
         {
            State dummyState[2];
            
            const int Attributes = 37;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //additionalLevelsForPredefinedRefinement
               MPI_CHAR,		 //isInitializing
               MPI_CHAR,		 //initialRefinmentTriggered
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_DOUBLE,		 //initialMinimalMeshWidth
               MPI_INT,		 //defaultSubdivisionFactor
               MPI_INT,		 //defaultGhostWidthLayer
               MPI_DOUBLE,		 //initialTimestepSize
               MPI_CHAR,		 //useDimensionalSplitting
               MPI_DOUBLE,		 //globalTimestepEndTime
               MPI_CHAR,		 //allPatchesEvolvedToGlobalTimestep
               MPI_DOUBLE,		 //domainOffset
               MPI_DOUBLE,		 //domainSize
               MPI_INT,		 //plotNumber
               MPI_INT,		 //subPlotNumber
               MPI_DOUBLE,		 //startMaximumGlobalTimeInterval
               MPI_DOUBLE,		 //endMaximumGlobalTimeInterval
               MPI_DOUBLE,		 //startMinimumGlobalTimeInterval
               MPI_DOUBLE,		 //endMinimumGlobalTimeInterval
               MPI_DOUBLE,		 //minimalTimestep
               MPI_DOUBLE,		 //totalNumberOfCellUpdates
               MPI_DOUBLE,		 //minMeshWidth
               MPI_DOUBLE,		 //maxMeshWidth
               MPI_DOUBLE,		 //numberOfInnerVertices
               MPI_DOUBLE,		 //numberOfBoundaryVertices
               MPI_DOUBLE,		 //numberOfOuterVertices
               MPI_DOUBLE,		 //numberOfInnerCells
               MPI_DOUBLE,		 //numberOfOuterCells
               MPI_INT,		 //maxLevel
               MPI_CHAR,		 //hasRefined
               MPI_CHAR,		 //hasTriggeredRefinementForNextIteration
               MPI_CHAR,		 //hasErased
               MPI_CHAR,		 //hasTriggeredEraseForNextIteration
               MPI_CHAR,		 //hasChangedVertexOrCellState
               MPI_CHAR,		 //isTraversalInverted
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //additionalLevelsForPredefinedRefinement
               1,		 //isInitializing
               1,		 //initialRefinmentTriggered
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               DIMENSIONS,		 //initialMinimalMeshWidth
               DIMENSIONS,		 //defaultSubdivisionFactor
               1,		 //defaultGhostWidthLayer
               1,		 //initialTimestepSize
               1,		 //useDimensionalSplitting
               1,		 //globalTimestepEndTime
               1,		 //allPatchesEvolvedToGlobalTimestep
               DIMENSIONS,		 //domainOffset
               DIMENSIONS,		 //domainSize
               1,		 //plotNumber
               1,		 //subPlotNumber
               1,		 //startMaximumGlobalTimeInterval
               1,		 //endMaximumGlobalTimeInterval
               1,		 //startMinimumGlobalTimeInterval
               1,		 //endMinimumGlobalTimeInterval
               1,		 //minimalTimestep
               1,		 //totalNumberOfCellUpdates
               DIMENSIONS,		 //minMeshWidth
               DIMENSIONS,		 //maxMeshWidth
               1,		 //numberOfInnerVertices
               1,		 //numberOfBoundaryVertices
               1,		 //numberOfOuterVertices
               1,		 //numberOfInnerCells
               1,		 //numberOfOuterCells
               1,		 //maxLevel
               1,		 //hasRefined
               1,		 //hasTriggeredRefinementForNextIteration
               1,		 //hasErased
               1,		 //hasTriggeredEraseForNextIteration
               1,		 //hasChangedVertexOrCellState
               1,		 //isTraversalInverted
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._isInitializing))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialRefinmentTriggered))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._unknownsPerSubcell))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialMinimalMeshWidth[0]))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._defaultSubdivisionFactor[0]))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._defaultGhostWidthLayer))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialTimestepSize))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._useDimensionalSplitting))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._globalTimestepEndTime))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._allPatchesEvolvedToGlobalTimestep))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._domainOffset[0]))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._domainSize[0]))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._plotNumber))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._subPlotNumber))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._startMaximumGlobalTimeInterval))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._endMaximumGlobalTimeInterval))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._startMinimumGlobalTimeInterval))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._endMinimumGlobalTimeInterval))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._minimalTimestep))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._totalNumberOfCellUpdates))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._minMeshWidth[0]))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._maxMeshWidth[0]))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerVertices))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfBoundaryVertices))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterVertices))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerCells))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterCells))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._maxLevel))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasRefined))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasTriggeredRefinementForNextIteration))), 		&disp[31] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasErased))), 		&disp[32] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasTriggeredEraseForNextIteration))), 		&disp[33] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasChangedVertexOrCellState))), 		&disp[34] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._isTraversalInverted))), 		&disp[35] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[1]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[36] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &State::FullDatatype );
            MPI_Type_commit( &State::FullDatatype );
            
         }
         
      }
      
      
      void peanoclaw::records::State::shutdownDatatype() {
         MPI_Type_free( &State::Datatype );
         MPI_Type_free( &State::FullDatatype );
         
      }
      
      void peanoclaw::records::State::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
         MPI_Request* sendRequestHandle = new MPI_Request();
         MPI_Status   status;
         int          flag = 0;
         int          result;
         
         clock_t      timeOutWarning   = -1;
         clock_t      timeOutShutdown  = -1;
         bool         triggeredTimeoutWarning = false;
         
         #ifdef Asserts
         _senderRank = -1;
         #endif
         
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
            msg << "was not able to send message peanoclaw::records::State "
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
               msg << "testing for finished send task for peanoclaw::records::State "
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
               "peanoclaw::records::State",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::State",
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
      
      
      
      void peanoclaw::records::State::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "failed to start to receive peanoclaw::records::State from node "
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
               msg << "testing for finished receive task for peanoclaw::records::State failed: "
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
               "peanoclaw::records::State",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::State",
               "receive(int)", source,tag,1
               );
            }
            tarch::parallel::Node::getInstance().receiveDanglingMessages();
         }
         
         delete sendRequestHandle;
         
         _senderRank = status.MPI_SOURCE;
         #ifdef Debug
         _log.debug("receive(int,int)", "received " + toString() ); 
         #endif
         
      }
      
      
      
      bool peanoclaw::records::State::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
      
      int peanoclaw::records::State::getSenderRank() const {
         assertion( _senderRank!=-1 );
         return _senderRank;
         
      }
   #endif
   
   
   peanoclaw::records::StatePacked::PersistentRecords::PersistentRecords() {
      assertion((6 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::StatePacked::PersistentRecords::PersistentRecords(const int& additionalLevelsForPredefinedRefinement, const bool& isInitializing, const bool& initialRefinmentTriggered, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const tarch::la::Vector<DIMENSIONS,double>& initialMinimalMeshWidth, const tarch::la::Vector<DIMENSIONS,int>& defaultSubdivisionFactor, const int& defaultGhostWidthLayer, const double& initialTimestepSize, const bool& useDimensionalSplitting, const double& globalTimestepEndTime, const bool& allPatchesEvolvedToGlobalTimestep, const tarch::la::Vector<DIMENSIONS,double>& domainOffset, const tarch::la::Vector<DIMENSIONS,double>& domainSize, const int& plotNumber, const int& subPlotNumber, const double& startMaximumGlobalTimeInterval, const double& endMaximumGlobalTimeInterval, const double& startMinimumGlobalTimeInterval, const double& endMinimumGlobalTimeInterval, const double& minimalTimestep, const double& totalNumberOfCellUpdates, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& isTraversalInverted):
   _additionalLevelsForPredefinedRefinement(additionalLevelsForPredefinedRefinement),
   _isInitializing(isInitializing),
   _initialRefinmentTriggered(initialRefinmentTriggered),
   _unknownsPerSubcell(unknownsPerSubcell),
   _auxiliarFieldsPerSubcell(auxiliarFieldsPerSubcell),
   _initialMinimalMeshWidth(initialMinimalMeshWidth),
   _defaultSubdivisionFactor(defaultSubdivisionFactor),
   _defaultGhostWidthLayer(defaultGhostWidthLayer),
   _initialTimestepSize(initialTimestepSize),
   _useDimensionalSplitting(useDimensionalSplitting),
   _globalTimestepEndTime(globalTimestepEndTime),
   _domainOffset(domainOffset),
   _domainSize(domainSize),
   _plotNumber(plotNumber),
   _subPlotNumber(subPlotNumber),
   _startMaximumGlobalTimeInterval(startMaximumGlobalTimeInterval),
   _endMaximumGlobalTimeInterval(endMaximumGlobalTimeInterval),
   _startMinimumGlobalTimeInterval(startMinimumGlobalTimeInterval),
   _endMinimumGlobalTimeInterval(endMinimumGlobalTimeInterval),
   _minimalTimestep(minimalTimestep),
   _totalNumberOfCellUpdates(totalNumberOfCellUpdates),
   _minMeshWidth(minMeshWidth),
   _maxMeshWidth(maxMeshWidth),
   _numberOfInnerVertices(numberOfInnerVertices),
   _numberOfBoundaryVertices(numberOfBoundaryVertices),
   _numberOfOuterVertices(numberOfOuterVertices),
   _numberOfInnerCells(numberOfInnerCells),
   _numberOfOuterCells(numberOfOuterCells),
   _maxLevel(maxLevel),
   _isTraversalInverted(isTraversalInverted) {
      setAllPatchesEvolvedToGlobalTimestep(allPatchesEvolvedToGlobalTimestep);
      setHasRefined(hasRefined);
      setHasTriggeredRefinementForNextIteration(hasTriggeredRefinementForNextIteration);
      setHasErased(hasErased);
      setHasTriggeredEraseForNextIteration(hasTriggeredEraseForNextIteration);
      setHasChangedVertexOrCellState(hasChangedVertexOrCellState);
      assertion((6 < (8 * sizeof(short int))));
      
   }
   
   peanoclaw::records::StatePacked::StatePacked() {
      assertion((6 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::StatePacked::StatePacked(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._additionalLevelsForPredefinedRefinement, persistentRecords._isInitializing, persistentRecords._initialRefinmentTriggered, persistentRecords._unknownsPerSubcell, persistentRecords._auxiliarFieldsPerSubcell, persistentRecords._initialMinimalMeshWidth, persistentRecords._defaultSubdivisionFactor, persistentRecords._defaultGhostWidthLayer, persistentRecords._initialTimestepSize, persistentRecords._useDimensionalSplitting, persistentRecords._globalTimestepEndTime, persistentRecords.getAllPatchesEvolvedToGlobalTimestep(), persistentRecords._domainOffset, persistentRecords._domainSize, persistentRecords._plotNumber, persistentRecords._subPlotNumber, persistentRecords._startMaximumGlobalTimeInterval, persistentRecords._endMaximumGlobalTimeInterval, persistentRecords._startMinimumGlobalTimeInterval, persistentRecords._endMinimumGlobalTimeInterval, persistentRecords._minimalTimestep, persistentRecords._totalNumberOfCellUpdates, persistentRecords._minMeshWidth, persistentRecords._maxMeshWidth, persistentRecords._numberOfInnerVertices, persistentRecords._numberOfBoundaryVertices, persistentRecords._numberOfOuterVertices, persistentRecords._numberOfInnerCells, persistentRecords._numberOfOuterCells, persistentRecords._maxLevel, persistentRecords.getHasRefined(), persistentRecords.getHasTriggeredRefinementForNextIteration(), persistentRecords.getHasErased(), persistentRecords.getHasTriggeredEraseForNextIteration(), persistentRecords.getHasChangedVertexOrCellState(), persistentRecords._isTraversalInverted) {
      assertion((6 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::StatePacked::StatePacked(const int& additionalLevelsForPredefinedRefinement, const bool& isInitializing, const bool& initialRefinmentTriggered, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const tarch::la::Vector<DIMENSIONS,double>& initialMinimalMeshWidth, const tarch::la::Vector<DIMENSIONS,int>& defaultSubdivisionFactor, const int& defaultGhostWidthLayer, const double& initialTimestepSize, const bool& useDimensionalSplitting, const double& globalTimestepEndTime, const bool& allPatchesEvolvedToGlobalTimestep, const tarch::la::Vector<DIMENSIONS,double>& domainOffset, const tarch::la::Vector<DIMENSIONS,double>& domainSize, const int& plotNumber, const int& subPlotNumber, const double& startMaximumGlobalTimeInterval, const double& endMaximumGlobalTimeInterval, const double& startMinimumGlobalTimeInterval, const double& endMinimumGlobalTimeInterval, const double& minimalTimestep, const double& totalNumberOfCellUpdates, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& isTraversalInverted):
   _persistentRecords(additionalLevelsForPredefinedRefinement, isInitializing, initialRefinmentTriggered, unknownsPerSubcell, auxiliarFieldsPerSubcell, initialMinimalMeshWidth, defaultSubdivisionFactor, defaultGhostWidthLayer, initialTimestepSize, useDimensionalSplitting, globalTimestepEndTime, allPatchesEvolvedToGlobalTimestep, domainOffset, domainSize, plotNumber, subPlotNumber, startMaximumGlobalTimeInterval, endMaximumGlobalTimeInterval, startMinimumGlobalTimeInterval, endMinimumGlobalTimeInterval, minimalTimestep, totalNumberOfCellUpdates, minMeshWidth, maxMeshWidth, numberOfInnerVertices, numberOfBoundaryVertices, numberOfOuterVertices, numberOfInnerCells, numberOfOuterCells, maxLevel, hasRefined, hasTriggeredRefinementForNextIteration, hasErased, hasTriggeredEraseForNextIteration, hasChangedVertexOrCellState, isTraversalInverted) {
      assertion((6 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::StatePacked::~StatePacked() { }
   
   
   
   std::string peanoclaw::records::StatePacked::toString() const {
      std::ostringstream stringstr;
      toString(stringstr);
      return stringstr.str();
   }
   
   void peanoclaw::records::StatePacked::toString (std::ostream& out) const {
      out << "("; 
      out << "additionalLevelsForPredefinedRefinement:" << getAdditionalLevelsForPredefinedRefinement();
      out << ",";
      out << "isInitializing:" << getIsInitializing();
      out << ",";
      out << "initialRefinmentTriggered:" << getInitialRefinmentTriggered();
      out << ",";
      out << "unknownsPerSubcell:" << getUnknownsPerSubcell();
      out << ",";
      out << "auxiliarFieldsPerSubcell:" << getAuxiliarFieldsPerSubcell();
      out << ",";
      out << "initialMinimalMeshWidth:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getInitialMinimalMeshWidth(i) << ",";
   }
   out << getInitialMinimalMeshWidth(DIMENSIONS-1) << "]";
      out << ",";
      out << "defaultSubdivisionFactor:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getDefaultSubdivisionFactor(i) << ",";
   }
   out << getDefaultSubdivisionFactor(DIMENSIONS-1) << "]";
      out << ",";
      out << "defaultGhostWidthLayer:" << getDefaultGhostWidthLayer();
      out << ",";
      out << "initialTimestepSize:" << getInitialTimestepSize();
      out << ",";
      out << "useDimensionalSplitting:" << getUseDimensionalSplitting();
      out << ",";
      out << "globalTimestepEndTime:" << getGlobalTimestepEndTime();
      out << ",";
      out << "allPatchesEvolvedToGlobalTimestep:" << getAllPatchesEvolvedToGlobalTimestep();
      out << ",";
      out << "domainOffset:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getDomainOffset(i) << ",";
   }
   out << getDomainOffset(DIMENSIONS-1) << "]";
      out << ",";
      out << "domainSize:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getDomainSize(i) << ",";
   }
   out << getDomainSize(DIMENSIONS-1) << "]";
      out << ",";
      out << "plotNumber:" << getPlotNumber();
      out << ",";
      out << "subPlotNumber:" << getSubPlotNumber();
      out << ",";
      out << "startMaximumGlobalTimeInterval:" << getStartMaximumGlobalTimeInterval();
      out << ",";
      out << "endMaximumGlobalTimeInterval:" << getEndMaximumGlobalTimeInterval();
      out << ",";
      out << "startMinimumGlobalTimeInterval:" << getStartMinimumGlobalTimeInterval();
      out << ",";
      out << "endMinimumGlobalTimeInterval:" << getEndMinimumGlobalTimeInterval();
      out << ",";
      out << "minimalTimestep:" << getMinimalTimestep();
      out << ",";
      out << "totalNumberOfCellUpdates:" << getTotalNumberOfCellUpdates();
      out << ",";
      out << "minMeshWidth:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getMinMeshWidth(i) << ",";
   }
   out << getMinMeshWidth(DIMENSIONS-1) << "]";
      out << ",";
      out << "maxMeshWidth:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getMaxMeshWidth(i) << ",";
   }
   out << getMaxMeshWidth(DIMENSIONS-1) << "]";
      out << ",";
      out << "numberOfInnerVertices:" << getNumberOfInnerVertices();
      out << ",";
      out << "numberOfBoundaryVertices:" << getNumberOfBoundaryVertices();
      out << ",";
      out << "numberOfOuterVertices:" << getNumberOfOuterVertices();
      out << ",";
      out << "numberOfInnerCells:" << getNumberOfInnerCells();
      out << ",";
      out << "numberOfOuterCells:" << getNumberOfOuterCells();
      out << ",";
      out << "maxLevel:" << getMaxLevel();
      out << ",";
      out << "hasRefined:" << getHasRefined();
      out << ",";
      out << "hasTriggeredRefinementForNextIteration:" << getHasTriggeredRefinementForNextIteration();
      out << ",";
      out << "hasErased:" << getHasErased();
      out << ",";
      out << "hasTriggeredEraseForNextIteration:" << getHasTriggeredEraseForNextIteration();
      out << ",";
      out << "hasChangedVertexOrCellState:" << getHasChangedVertexOrCellState();
      out << ",";
      out << "isTraversalInverted:" << getIsTraversalInverted();
      out <<  ")";
   }
   
   
   peanoclaw::records::StatePacked::PersistentRecords peanoclaw::records::StatePacked::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::State peanoclaw::records::StatePacked::convert() const{
      return State(
         getAdditionalLevelsForPredefinedRefinement(),
         getIsInitializing(),
         getInitialRefinmentTriggered(),
         getUnknownsPerSubcell(),
         getAuxiliarFieldsPerSubcell(),
         getInitialMinimalMeshWidth(),
         getDefaultSubdivisionFactor(),
         getDefaultGhostWidthLayer(),
         getInitialTimestepSize(),
         getUseDimensionalSplitting(),
         getGlobalTimestepEndTime(),
         getAllPatchesEvolvedToGlobalTimestep(),
         getDomainOffset(),
         getDomainSize(),
         getPlotNumber(),
         getSubPlotNumber(),
         getStartMaximumGlobalTimeInterval(),
         getEndMaximumGlobalTimeInterval(),
         getStartMinimumGlobalTimeInterval(),
         getEndMinimumGlobalTimeInterval(),
         getMinimalTimestep(),
         getTotalNumberOfCellUpdates(),
         getMinMeshWidth(),
         getMaxMeshWidth(),
         getNumberOfInnerVertices(),
         getNumberOfBoundaryVertices(),
         getNumberOfOuterVertices(),
         getNumberOfInnerCells(),
         getNumberOfOuterCells(),
         getMaxLevel(),
         getHasRefined(),
         getHasTriggeredRefinementForNextIteration(),
         getHasErased(),
         getHasTriggeredEraseForNextIteration(),
         getHasChangedVertexOrCellState(),
         getIsTraversalInverted()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::StatePacked::_log( "peanoclaw::records::StatePacked" );
      
      MPI_Datatype peanoclaw::records::StatePacked::Datatype = 0;
      MPI_Datatype peanoclaw::records::StatePacked::FullDatatype = 0;
      
      
      void peanoclaw::records::StatePacked::initDatatype() {
         {
            StatePacked dummyStatePacked[2];
            
            const int Attributes = 32;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //additionalLevelsForPredefinedRefinement
               MPI_CHAR,		 //isInitializing
               MPI_CHAR,		 //initialRefinmentTriggered
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_DOUBLE,		 //initialMinimalMeshWidth
               MPI_INT,		 //defaultSubdivisionFactor
               MPI_INT,		 //defaultGhostWidthLayer
               MPI_DOUBLE,		 //initialTimestepSize
               MPI_CHAR,		 //useDimensionalSplitting
               MPI_DOUBLE,		 //globalTimestepEndTime
               MPI_DOUBLE,		 //domainOffset
               MPI_DOUBLE,		 //domainSize
               MPI_INT,		 //plotNumber
               MPI_INT,		 //subPlotNumber
               MPI_DOUBLE,		 //startMaximumGlobalTimeInterval
               MPI_DOUBLE,		 //endMaximumGlobalTimeInterval
               MPI_DOUBLE,		 //startMinimumGlobalTimeInterval
               MPI_DOUBLE,		 //endMinimumGlobalTimeInterval
               MPI_DOUBLE,		 //minimalTimestep
               MPI_DOUBLE,		 //totalNumberOfCellUpdates
               MPI_DOUBLE,		 //minMeshWidth
               MPI_DOUBLE,		 //maxMeshWidth
               MPI_DOUBLE,		 //numberOfInnerVertices
               MPI_DOUBLE,		 //numberOfBoundaryVertices
               MPI_DOUBLE,		 //numberOfOuterVertices
               MPI_DOUBLE,		 //numberOfInnerCells
               MPI_DOUBLE,		 //numberOfOuterCells
               MPI_INT,		 //maxLevel
               MPI_CHAR,		 //isTraversalInverted
               MPI_SHORT,		 //_packedRecords0
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //additionalLevelsForPredefinedRefinement
               1,		 //isInitializing
               1,		 //initialRefinmentTriggered
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               DIMENSIONS,		 //initialMinimalMeshWidth
               DIMENSIONS,		 //defaultSubdivisionFactor
               1,		 //defaultGhostWidthLayer
               1,		 //initialTimestepSize
               1,		 //useDimensionalSplitting
               1,		 //globalTimestepEndTime
               DIMENSIONS,		 //domainOffset
               DIMENSIONS,		 //domainSize
               1,		 //plotNumber
               1,		 //subPlotNumber
               1,		 //startMaximumGlobalTimeInterval
               1,		 //endMaximumGlobalTimeInterval
               1,		 //startMinimumGlobalTimeInterval
               1,		 //endMinimumGlobalTimeInterval
               1,		 //minimalTimestep
               1,		 //totalNumberOfCellUpdates
               DIMENSIONS,		 //minMeshWidth
               DIMENSIONS,		 //maxMeshWidth
               1,		 //numberOfInnerVertices
               1,		 //numberOfBoundaryVertices
               1,		 //numberOfOuterVertices
               1,		 //numberOfInnerCells
               1,		 //numberOfOuterCells
               1,		 //maxLevel
               1,		 //isTraversalInverted
               1,		 //_packedRecords0
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._isInitializing))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialRefinmentTriggered))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._unknownsPerSubcell))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialMinimalMeshWidth[0]))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._defaultSubdivisionFactor[0]))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._defaultGhostWidthLayer))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialTimestepSize))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._useDimensionalSplitting))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._globalTimestepEndTime))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._domainOffset[0]))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._domainSize[0]))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._plotNumber))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._subPlotNumber))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._startMaximumGlobalTimeInterval))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._endMaximumGlobalTimeInterval))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._startMinimumGlobalTimeInterval))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._endMinimumGlobalTimeInterval))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._minimalTimestep))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._totalNumberOfCellUpdates))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._minMeshWidth[0]))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._maxMeshWidth[0]))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerVertices))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfBoundaryVertices))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterVertices))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerCells))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterCells))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._maxLevel))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._isTraversalInverted))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._packedRecords0))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[1]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[31] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &StatePacked::Datatype );
            MPI_Type_commit( &StatePacked::Datatype );
            
         }
         {
            StatePacked dummyStatePacked[2];
            
            const int Attributes = 32;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //additionalLevelsForPredefinedRefinement
               MPI_CHAR,		 //isInitializing
               MPI_CHAR,		 //initialRefinmentTriggered
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_DOUBLE,		 //initialMinimalMeshWidth
               MPI_INT,		 //defaultSubdivisionFactor
               MPI_INT,		 //defaultGhostWidthLayer
               MPI_DOUBLE,		 //initialTimestepSize
               MPI_CHAR,		 //useDimensionalSplitting
               MPI_DOUBLE,		 //globalTimestepEndTime
               MPI_DOUBLE,		 //domainOffset
               MPI_DOUBLE,		 //domainSize
               MPI_INT,		 //plotNumber
               MPI_INT,		 //subPlotNumber
               MPI_DOUBLE,		 //startMaximumGlobalTimeInterval
               MPI_DOUBLE,		 //endMaximumGlobalTimeInterval
               MPI_DOUBLE,		 //startMinimumGlobalTimeInterval
               MPI_DOUBLE,		 //endMinimumGlobalTimeInterval
               MPI_DOUBLE,		 //minimalTimestep
               MPI_DOUBLE,		 //totalNumberOfCellUpdates
               MPI_DOUBLE,		 //minMeshWidth
               MPI_DOUBLE,		 //maxMeshWidth
               MPI_DOUBLE,		 //numberOfInnerVertices
               MPI_DOUBLE,		 //numberOfBoundaryVertices
               MPI_DOUBLE,		 //numberOfOuterVertices
               MPI_DOUBLE,		 //numberOfInnerCells
               MPI_DOUBLE,		 //numberOfOuterCells
               MPI_INT,		 //maxLevel
               MPI_CHAR,		 //isTraversalInverted
               MPI_SHORT,		 //_packedRecords0
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //additionalLevelsForPredefinedRefinement
               1,		 //isInitializing
               1,		 //initialRefinmentTriggered
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               DIMENSIONS,		 //initialMinimalMeshWidth
               DIMENSIONS,		 //defaultSubdivisionFactor
               1,		 //defaultGhostWidthLayer
               1,		 //initialTimestepSize
               1,		 //useDimensionalSplitting
               1,		 //globalTimestepEndTime
               DIMENSIONS,		 //domainOffset
               DIMENSIONS,		 //domainSize
               1,		 //plotNumber
               1,		 //subPlotNumber
               1,		 //startMaximumGlobalTimeInterval
               1,		 //endMaximumGlobalTimeInterval
               1,		 //startMinimumGlobalTimeInterval
               1,		 //endMinimumGlobalTimeInterval
               1,		 //minimalTimestep
               1,		 //totalNumberOfCellUpdates
               DIMENSIONS,		 //minMeshWidth
               DIMENSIONS,		 //maxMeshWidth
               1,		 //numberOfInnerVertices
               1,		 //numberOfBoundaryVertices
               1,		 //numberOfOuterVertices
               1,		 //numberOfInnerCells
               1,		 //numberOfOuterCells
               1,		 //maxLevel
               1,		 //isTraversalInverted
               1,		 //_packedRecords0
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._isInitializing))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialRefinmentTriggered))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._unknownsPerSubcell))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialMinimalMeshWidth[0]))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._defaultSubdivisionFactor[0]))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._defaultGhostWidthLayer))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialTimestepSize))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._useDimensionalSplitting))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._globalTimestepEndTime))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._domainOffset[0]))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._domainSize[0]))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._plotNumber))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._subPlotNumber))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._startMaximumGlobalTimeInterval))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._endMaximumGlobalTimeInterval))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._startMinimumGlobalTimeInterval))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._endMinimumGlobalTimeInterval))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._minimalTimestep))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._totalNumberOfCellUpdates))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._minMeshWidth[0]))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._maxMeshWidth[0]))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerVertices))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfBoundaryVertices))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterVertices))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerCells))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterCells))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._maxLevel))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._isTraversalInverted))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._packedRecords0))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[1]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[31] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &StatePacked::FullDatatype );
            MPI_Type_commit( &StatePacked::FullDatatype );
            
         }
         
      }
      
      
      void peanoclaw::records::StatePacked::shutdownDatatype() {
         MPI_Type_free( &StatePacked::Datatype );
         MPI_Type_free( &StatePacked::FullDatatype );
         
      }
      
      void peanoclaw::records::StatePacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
         MPI_Request* sendRequestHandle = new MPI_Request();
         MPI_Status   status;
         int          flag = 0;
         int          result;
         
         clock_t      timeOutWarning   = -1;
         clock_t      timeOutShutdown  = -1;
         bool         triggeredTimeoutWarning = false;
         
         #ifdef Asserts
         _senderRank = -1;
         #endif
         
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
            msg << "was not able to send message peanoclaw::records::StatePacked "
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
               msg << "testing for finished send task for peanoclaw::records::StatePacked "
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
               "peanoclaw::records::StatePacked",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::StatePacked",
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
      
      
      
      void peanoclaw::records::StatePacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "failed to start to receive peanoclaw::records::StatePacked from node "
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
               msg << "testing for finished receive task for peanoclaw::records::StatePacked failed: "
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
               "peanoclaw::records::StatePacked",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::StatePacked",
               "receive(int)", source,tag,1
               );
            }
            tarch::parallel::Node::getInstance().receiveDanglingMessages();
         }
         
         delete sendRequestHandle;
         
         _senderRank = status.MPI_SOURCE;
         #ifdef Debug
         _log.debug("receive(int,int)", "received " + toString() ); 
         #endif
         
      }
      
      
      
      bool peanoclaw::records::StatePacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
      
      int peanoclaw::records::StatePacked::getSenderRank() const {
         assertion( _senderRank!=-1 );
         return _senderRank;
         
      }
   #endif
   
   
   

#endif


