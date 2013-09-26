#include "peanoclaw/records/State.h"

#if defined(Parallel)
   peanoclaw::records::State::PersistentRecords::PersistentRecords() {
      
   }
   
   
   peanoclaw::records::State::PersistentRecords::PersistentRecords(const int& additionalLevelsForPredefinedRefinement, const bool& isInitializing, const bool& isRefinementCriterionEnabled, const bool& initialRefinmentTriggered, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const tarch::la::Vector<DIMENSIONS,double>& initialMaximalSubgridSize, const tarch::la::Vector<DIMENSIONS,int>& defaultSubdivisionFactor, const int& defaultGhostWidthLayer, const double& initialTimestepSize, const bool& useDimensionalSplittingOptimization, const double& globalTimestepEndTime, const bool& allPatchesEvolvedToGlobalTimestep, const tarch::la::Vector<DIMENSIONS,double>& domainOffset, const tarch::la::Vector<DIMENSIONS,double>& domainSize, const int& plotNumber, const int& subPlotNumber, const double& startMaximumGlobalTimeInterval, const double& endMaximumGlobalTimeInterval, const double& startMinimumGlobalTimeInterval, const double& endMinimumGlobalTimeInterval, const double& minimalTimestep, const double& totalNumberOfCellUpdates, const int& localHeightOfWorkerTree, const int& globalHeightOfWorkerTreeDuringLastIteration, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const double& numberOfInnerLeafVertices, const double& numberOfBoundaryLeafVertices, const double& numberOfOuterLeafVertices, const double& numberOfInnerLeafCells, const double& numberOfOuterLeafCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& isTraversalInverted, const bool& reduceStateAndCell, const bool& couldNotEraseDueToDecompositionFlag, const bool& subWorkerIsInvolvedInJoinOrFork):
   _additionalLevelsForPredefinedRefinement(additionalLevelsForPredefinedRefinement),
   _isInitializing(isInitializing),
   _isRefinementCriterionEnabled(isRefinementCriterionEnabled),
   _initialRefinmentTriggered(initialRefinmentTriggered),
   _unknownsPerSubcell(unknownsPerSubcell),
   _auxiliarFieldsPerSubcell(auxiliarFieldsPerSubcell),
   _initialMaximalSubgridSize(initialMaximalSubgridSize),
   _defaultSubdivisionFactor(defaultSubdivisionFactor),
   _defaultGhostWidthLayer(defaultGhostWidthLayer),
   _initialTimestepSize(initialTimestepSize),
   _useDimensionalSplittingOptimization(useDimensionalSplittingOptimization),
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
   _localHeightOfWorkerTree(localHeightOfWorkerTree),
   _globalHeightOfWorkerTreeDuringLastIteration(globalHeightOfWorkerTreeDuringLastIteration),
   _minMeshWidth(minMeshWidth),
   _maxMeshWidth(maxMeshWidth),
   _numberOfInnerVertices(numberOfInnerVertices),
   _numberOfBoundaryVertices(numberOfBoundaryVertices),
   _numberOfOuterVertices(numberOfOuterVertices),
   _numberOfInnerCells(numberOfInnerCells),
   _numberOfOuterCells(numberOfOuterCells),
   _numberOfInnerLeafVertices(numberOfInnerLeafVertices),
   _numberOfBoundaryLeafVertices(numberOfBoundaryLeafVertices),
   _numberOfOuterLeafVertices(numberOfOuterLeafVertices),
   _numberOfInnerLeafCells(numberOfInnerLeafCells),
   _numberOfOuterLeafCells(numberOfOuterLeafCells),
   _maxLevel(maxLevel),
   _hasRefined(hasRefined),
   _hasTriggeredRefinementForNextIteration(hasTriggeredRefinementForNextIteration),
   _hasErased(hasErased),
   _hasTriggeredEraseForNextIteration(hasTriggeredEraseForNextIteration),
   _hasChangedVertexOrCellState(hasChangedVertexOrCellState),
   _isTraversalInverted(isTraversalInverted),
   _reduceStateAndCell(reduceStateAndCell),
   _couldNotEraseDueToDecompositionFlag(couldNotEraseDueToDecompositionFlag),
   _subWorkerIsInvolvedInJoinOrFork(subWorkerIsInvolvedInJoinOrFork) {
      
   }
   
   peanoclaw::records::State::State() {
      
   }
   
   
   peanoclaw::records::State::State(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._additionalLevelsForPredefinedRefinement, persistentRecords._isInitializing, persistentRecords._isRefinementCriterionEnabled, persistentRecords._initialRefinmentTriggered, persistentRecords._unknownsPerSubcell, persistentRecords._auxiliarFieldsPerSubcell, persistentRecords._initialMaximalSubgridSize, persistentRecords._defaultSubdivisionFactor, persistentRecords._defaultGhostWidthLayer, persistentRecords._initialTimestepSize, persistentRecords._useDimensionalSplittingOptimization, persistentRecords._globalTimestepEndTime, persistentRecords._allPatchesEvolvedToGlobalTimestep, persistentRecords._domainOffset, persistentRecords._domainSize, persistentRecords._plotNumber, persistentRecords._subPlotNumber, persistentRecords._startMaximumGlobalTimeInterval, persistentRecords._endMaximumGlobalTimeInterval, persistentRecords._startMinimumGlobalTimeInterval, persistentRecords._endMinimumGlobalTimeInterval, persistentRecords._minimalTimestep, persistentRecords._totalNumberOfCellUpdates, persistentRecords._localHeightOfWorkerTree, persistentRecords._globalHeightOfWorkerTreeDuringLastIteration, persistentRecords._minMeshWidth, persistentRecords._maxMeshWidth, persistentRecords._numberOfInnerVertices, persistentRecords._numberOfBoundaryVertices, persistentRecords._numberOfOuterVertices, persistentRecords._numberOfInnerCells, persistentRecords._numberOfOuterCells, persistentRecords._numberOfInnerLeafVertices, persistentRecords._numberOfBoundaryLeafVertices, persistentRecords._numberOfOuterLeafVertices, persistentRecords._numberOfInnerLeafCells, persistentRecords._numberOfOuterLeafCells, persistentRecords._maxLevel, persistentRecords._hasRefined, persistentRecords._hasTriggeredRefinementForNextIteration, persistentRecords._hasErased, persistentRecords._hasTriggeredEraseForNextIteration, persistentRecords._hasChangedVertexOrCellState, persistentRecords._isTraversalInverted, persistentRecords._reduceStateAndCell, persistentRecords._couldNotEraseDueToDecompositionFlag, persistentRecords._subWorkerIsInvolvedInJoinOrFork) {
      
   }
   
   
   peanoclaw::records::State::State(const int& additionalLevelsForPredefinedRefinement, const bool& isInitializing, const bool& isRefinementCriterionEnabled, const bool& initialRefinmentTriggered, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const tarch::la::Vector<DIMENSIONS,double>& initialMaximalSubgridSize, const tarch::la::Vector<DIMENSIONS,int>& defaultSubdivisionFactor, const int& defaultGhostWidthLayer, const double& initialTimestepSize, const bool& useDimensionalSplittingOptimization, const double& globalTimestepEndTime, const bool& allPatchesEvolvedToGlobalTimestep, const tarch::la::Vector<DIMENSIONS,double>& domainOffset, const tarch::la::Vector<DIMENSIONS,double>& domainSize, const int& plotNumber, const int& subPlotNumber, const double& startMaximumGlobalTimeInterval, const double& endMaximumGlobalTimeInterval, const double& startMinimumGlobalTimeInterval, const double& endMinimumGlobalTimeInterval, const double& minimalTimestep, const double& totalNumberOfCellUpdates, const int& localHeightOfWorkerTree, const int& globalHeightOfWorkerTreeDuringLastIteration, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const double& numberOfInnerLeafVertices, const double& numberOfBoundaryLeafVertices, const double& numberOfOuterLeafVertices, const double& numberOfInnerLeafCells, const double& numberOfOuterLeafCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& isTraversalInverted, const bool& reduceStateAndCell, const bool& couldNotEraseDueToDecompositionFlag, const bool& subWorkerIsInvolvedInJoinOrFork):
   _persistentRecords(additionalLevelsForPredefinedRefinement, isInitializing, isRefinementCriterionEnabled, initialRefinmentTriggered, unknownsPerSubcell, auxiliarFieldsPerSubcell, initialMaximalSubgridSize, defaultSubdivisionFactor, defaultGhostWidthLayer, initialTimestepSize, useDimensionalSplittingOptimization, globalTimestepEndTime, allPatchesEvolvedToGlobalTimestep, domainOffset, domainSize, plotNumber, subPlotNumber, startMaximumGlobalTimeInterval, endMaximumGlobalTimeInterval, startMinimumGlobalTimeInterval, endMinimumGlobalTimeInterval, minimalTimestep, totalNumberOfCellUpdates, localHeightOfWorkerTree, globalHeightOfWorkerTreeDuringLastIteration, minMeshWidth, maxMeshWidth, numberOfInnerVertices, numberOfBoundaryVertices, numberOfOuterVertices, numberOfInnerCells, numberOfOuterCells, numberOfInnerLeafVertices, numberOfBoundaryLeafVertices, numberOfOuterLeafVertices, numberOfInnerLeafCells, numberOfOuterLeafCells, maxLevel, hasRefined, hasTriggeredRefinementForNextIteration, hasErased, hasTriggeredEraseForNextIteration, hasChangedVertexOrCellState, isTraversalInverted, reduceStateAndCell, couldNotEraseDueToDecompositionFlag, subWorkerIsInvolvedInJoinOrFork) {
      
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
      out << "isRefinementCriterionEnabled:" << getIsRefinementCriterionEnabled();
      out << ",";
      out << "initialRefinmentTriggered:" << getInitialRefinmentTriggered();
      out << ",";
      out << "unknownsPerSubcell:" << getUnknownsPerSubcell();
      out << ",";
      out << "auxiliarFieldsPerSubcell:" << getAuxiliarFieldsPerSubcell();
      out << ",";
      out << "initialMaximalSubgridSize:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getInitialMaximalSubgridSize(i) << ",";
   }
   out << getInitialMaximalSubgridSize(DIMENSIONS-1) << "]";
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
      out << "useDimensionalSplittingOptimization:" << getUseDimensionalSplittingOptimization();
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
      out << "localHeightOfWorkerTree:" << getLocalHeightOfWorkerTree();
      out << ",";
      out << "globalHeightOfWorkerTreeDuringLastIteration:" << getGlobalHeightOfWorkerTreeDuringLastIteration();
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
      out << "numberOfInnerLeafVertices:" << getNumberOfInnerLeafVertices();
      out << ",";
      out << "numberOfBoundaryLeafVertices:" << getNumberOfBoundaryLeafVertices();
      out << ",";
      out << "numberOfOuterLeafVertices:" << getNumberOfOuterLeafVertices();
      out << ",";
      out << "numberOfInnerLeafCells:" << getNumberOfInnerLeafCells();
      out << ",";
      out << "numberOfOuterLeafCells:" << getNumberOfOuterLeafCells();
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
      out << "reduceStateAndCell:" << getReduceStateAndCell();
      out << ",";
      out << "couldNotEraseDueToDecompositionFlag:" << getCouldNotEraseDueToDecompositionFlag();
      out << ",";
      out << "subWorkerIsInvolvedInJoinOrFork:" << getSubWorkerIsInvolvedInJoinOrFork();
      out <<  ")";
   }
   
   
   peanoclaw::records::State::PersistentRecords peanoclaw::records::State::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::StatePacked peanoclaw::records::State::convert() const{
      return StatePacked(
         getAdditionalLevelsForPredefinedRefinement(),
         getIsInitializing(),
         getIsRefinementCriterionEnabled(),
         getInitialRefinmentTriggered(),
         getUnknownsPerSubcell(),
         getAuxiliarFieldsPerSubcell(),
         getInitialMaximalSubgridSize(),
         getDefaultSubdivisionFactor(),
         getDefaultGhostWidthLayer(),
         getInitialTimestepSize(),
         getUseDimensionalSplittingOptimization(),
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
         getLocalHeightOfWorkerTree(),
         getGlobalHeightOfWorkerTreeDuringLastIteration(),
         getMinMeshWidth(),
         getMaxMeshWidth(),
         getNumberOfInnerVertices(),
         getNumberOfBoundaryVertices(),
         getNumberOfOuterVertices(),
         getNumberOfInnerCells(),
         getNumberOfOuterCells(),
         getNumberOfInnerLeafVertices(),
         getNumberOfBoundaryLeafVertices(),
         getNumberOfOuterLeafVertices(),
         getNumberOfInnerLeafCells(),
         getNumberOfOuterLeafCells(),
         getMaxLevel(),
         getHasRefined(),
         getHasTriggeredRefinementForNextIteration(),
         getHasErased(),
         getHasTriggeredEraseForNextIteration(),
         getHasChangedVertexOrCellState(),
         getIsTraversalInverted(),
         getReduceStateAndCell(),
         getCouldNotEraseDueToDecompositionFlag(),
         getSubWorkerIsInvolvedInJoinOrFork()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::State::_log( "peanoclaw::records::State" );
      
      MPI_Datatype peanoclaw::records::State::Datatype = 0;
      MPI_Datatype peanoclaw::records::State::FullDatatype = 0;
      
      
      void peanoclaw::records::State::initDatatype() {
         {
            State dummyState[2];
            
            const int Attributes = 48;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //additionalLevelsForPredefinedRefinement
               MPI_CHAR,		 //isInitializing
               MPI_CHAR,		 //isRefinementCriterionEnabled
               MPI_CHAR,		 //initialRefinmentTriggered
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_DOUBLE,		 //initialMaximalSubgridSize
               MPI_INT,		 //defaultSubdivisionFactor
               MPI_INT,		 //defaultGhostWidthLayer
               MPI_DOUBLE,		 //initialTimestepSize
               MPI_CHAR,		 //useDimensionalSplittingOptimization
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
               MPI_INT,		 //localHeightOfWorkerTree
               MPI_INT,		 //globalHeightOfWorkerTreeDuringLastIteration
               MPI_DOUBLE,		 //minMeshWidth
               MPI_DOUBLE,		 //maxMeshWidth
               MPI_DOUBLE,		 //numberOfInnerVertices
               MPI_DOUBLE,		 //numberOfBoundaryVertices
               MPI_DOUBLE,		 //numberOfOuterVertices
               MPI_DOUBLE,		 //numberOfInnerCells
               MPI_DOUBLE,		 //numberOfOuterCells
               MPI_DOUBLE,		 //numberOfInnerLeafVertices
               MPI_DOUBLE,		 //numberOfBoundaryLeafVertices
               MPI_DOUBLE,		 //numberOfOuterLeafVertices
               MPI_DOUBLE,		 //numberOfInnerLeafCells
               MPI_DOUBLE,		 //numberOfOuterLeafCells
               MPI_INT,		 //maxLevel
               MPI_CHAR,		 //hasRefined
               MPI_CHAR,		 //hasTriggeredRefinementForNextIteration
               MPI_CHAR,		 //hasErased
               MPI_CHAR,		 //hasTriggeredEraseForNextIteration
               MPI_CHAR,		 //hasChangedVertexOrCellState
               MPI_CHAR,		 //isTraversalInverted
               MPI_CHAR,		 //reduceStateAndCell
               MPI_CHAR,		 //couldNotEraseDueToDecompositionFlag
               MPI_CHAR,		 //subWorkerIsInvolvedInJoinOrFork
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //additionalLevelsForPredefinedRefinement
               1,		 //isInitializing
               1,		 //isRefinementCriterionEnabled
               1,		 //initialRefinmentTriggered
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               DIMENSIONS,		 //initialMaximalSubgridSize
               DIMENSIONS,		 //defaultSubdivisionFactor
               1,		 //defaultGhostWidthLayer
               1,		 //initialTimestepSize
               1,		 //useDimensionalSplittingOptimization
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
               1,		 //localHeightOfWorkerTree
               1,		 //globalHeightOfWorkerTreeDuringLastIteration
               DIMENSIONS,		 //minMeshWidth
               DIMENSIONS,		 //maxMeshWidth
               1,		 //numberOfInnerVertices
               1,		 //numberOfBoundaryVertices
               1,		 //numberOfOuterVertices
               1,		 //numberOfInnerCells
               1,		 //numberOfOuterCells
               1,		 //numberOfInnerLeafVertices
               1,		 //numberOfBoundaryLeafVertices
               1,		 //numberOfOuterLeafVertices
               1,		 //numberOfInnerLeafCells
               1,		 //numberOfOuterLeafCells
               1,		 //maxLevel
               1,		 //hasRefined
               1,		 //hasTriggeredRefinementForNextIteration
               1,		 //hasErased
               1,		 //hasTriggeredEraseForNextIteration
               1,		 //hasChangedVertexOrCellState
               1,		 //isTraversalInverted
               1,		 //reduceStateAndCell
               1,		 //couldNotEraseDueToDecompositionFlag
               1,		 //subWorkerIsInvolvedInJoinOrFork
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._isInitializing))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._isRefinementCriterionEnabled))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialRefinmentTriggered))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._unknownsPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialMaximalSubgridSize[0]))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._defaultSubdivisionFactor[0]))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._defaultGhostWidthLayer))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialTimestepSize))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._useDimensionalSplittingOptimization))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._globalTimestepEndTime))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._allPatchesEvolvedToGlobalTimestep))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._domainOffset[0]))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._domainSize[0]))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._plotNumber))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._subPlotNumber))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._startMaximumGlobalTimeInterval))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._endMaximumGlobalTimeInterval))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._startMinimumGlobalTimeInterval))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._endMinimumGlobalTimeInterval))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._minimalTimestep))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._totalNumberOfCellUpdates))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._localHeightOfWorkerTree))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._globalHeightOfWorkerTreeDuringLastIteration))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._minMeshWidth[0]))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._maxMeshWidth[0]))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerVertices))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfBoundaryVertices))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterVertices))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerCells))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterCells))), 		&disp[31] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerLeafVertices))), 		&disp[32] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfBoundaryLeafVertices))), 		&disp[33] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterLeafVertices))), 		&disp[34] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerLeafCells))), 		&disp[35] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterLeafCells))), 		&disp[36] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._maxLevel))), 		&disp[37] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasRefined))), 		&disp[38] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasTriggeredRefinementForNextIteration))), 		&disp[39] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasErased))), 		&disp[40] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasTriggeredEraseForNextIteration))), 		&disp[41] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasChangedVertexOrCellState))), 		&disp[42] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._isTraversalInverted))), 		&disp[43] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._reduceStateAndCell))), 		&disp[44] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._couldNotEraseDueToDecompositionFlag))), 		&disp[45] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._subWorkerIsInvolvedInJoinOrFork))), 		&disp[46] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[1]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[47] );
            
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
            
            const int Attributes = 48;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //additionalLevelsForPredefinedRefinement
               MPI_CHAR,		 //isInitializing
               MPI_CHAR,		 //isRefinementCriterionEnabled
               MPI_CHAR,		 //initialRefinmentTriggered
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_DOUBLE,		 //initialMaximalSubgridSize
               MPI_INT,		 //defaultSubdivisionFactor
               MPI_INT,		 //defaultGhostWidthLayer
               MPI_DOUBLE,		 //initialTimestepSize
               MPI_CHAR,		 //useDimensionalSplittingOptimization
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
               MPI_INT,		 //localHeightOfWorkerTree
               MPI_INT,		 //globalHeightOfWorkerTreeDuringLastIteration
               MPI_DOUBLE,		 //minMeshWidth
               MPI_DOUBLE,		 //maxMeshWidth
               MPI_DOUBLE,		 //numberOfInnerVertices
               MPI_DOUBLE,		 //numberOfBoundaryVertices
               MPI_DOUBLE,		 //numberOfOuterVertices
               MPI_DOUBLE,		 //numberOfInnerCells
               MPI_DOUBLE,		 //numberOfOuterCells
               MPI_DOUBLE,		 //numberOfInnerLeafVertices
               MPI_DOUBLE,		 //numberOfBoundaryLeafVertices
               MPI_DOUBLE,		 //numberOfOuterLeafVertices
               MPI_DOUBLE,		 //numberOfInnerLeafCells
               MPI_DOUBLE,		 //numberOfOuterLeafCells
               MPI_INT,		 //maxLevel
               MPI_CHAR,		 //hasRefined
               MPI_CHAR,		 //hasTriggeredRefinementForNextIteration
               MPI_CHAR,		 //hasErased
               MPI_CHAR,		 //hasTriggeredEraseForNextIteration
               MPI_CHAR,		 //hasChangedVertexOrCellState
               MPI_CHAR,		 //isTraversalInverted
               MPI_CHAR,		 //reduceStateAndCell
               MPI_CHAR,		 //couldNotEraseDueToDecompositionFlag
               MPI_CHAR,		 //subWorkerIsInvolvedInJoinOrFork
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //additionalLevelsForPredefinedRefinement
               1,		 //isInitializing
               1,		 //isRefinementCriterionEnabled
               1,		 //initialRefinmentTriggered
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               DIMENSIONS,		 //initialMaximalSubgridSize
               DIMENSIONS,		 //defaultSubdivisionFactor
               1,		 //defaultGhostWidthLayer
               1,		 //initialTimestepSize
               1,		 //useDimensionalSplittingOptimization
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
               1,		 //localHeightOfWorkerTree
               1,		 //globalHeightOfWorkerTreeDuringLastIteration
               DIMENSIONS,		 //minMeshWidth
               DIMENSIONS,		 //maxMeshWidth
               1,		 //numberOfInnerVertices
               1,		 //numberOfBoundaryVertices
               1,		 //numberOfOuterVertices
               1,		 //numberOfInnerCells
               1,		 //numberOfOuterCells
               1,		 //numberOfInnerLeafVertices
               1,		 //numberOfBoundaryLeafVertices
               1,		 //numberOfOuterLeafVertices
               1,		 //numberOfInnerLeafCells
               1,		 //numberOfOuterLeafCells
               1,		 //maxLevel
               1,		 //hasRefined
               1,		 //hasTriggeredRefinementForNextIteration
               1,		 //hasErased
               1,		 //hasTriggeredEraseForNextIteration
               1,		 //hasChangedVertexOrCellState
               1,		 //isTraversalInverted
               1,		 //reduceStateAndCell
               1,		 //couldNotEraseDueToDecompositionFlag
               1,		 //subWorkerIsInvolvedInJoinOrFork
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._isInitializing))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._isRefinementCriterionEnabled))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialRefinmentTriggered))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._unknownsPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialMaximalSubgridSize[0]))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._defaultSubdivisionFactor[0]))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._defaultGhostWidthLayer))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialTimestepSize))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._useDimensionalSplittingOptimization))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._globalTimestepEndTime))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._allPatchesEvolvedToGlobalTimestep))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._domainOffset[0]))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._domainSize[0]))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._plotNumber))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._subPlotNumber))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._startMaximumGlobalTimeInterval))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._endMaximumGlobalTimeInterval))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._startMinimumGlobalTimeInterval))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._endMinimumGlobalTimeInterval))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._minimalTimestep))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._totalNumberOfCellUpdates))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._localHeightOfWorkerTree))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._globalHeightOfWorkerTreeDuringLastIteration))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._minMeshWidth[0]))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._maxMeshWidth[0]))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerVertices))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfBoundaryVertices))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterVertices))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerCells))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterCells))), 		&disp[31] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerLeafVertices))), 		&disp[32] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfBoundaryLeafVertices))), 		&disp[33] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterLeafVertices))), 		&disp[34] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerLeafCells))), 		&disp[35] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterLeafCells))), 		&disp[36] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._maxLevel))), 		&disp[37] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasRefined))), 		&disp[38] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasTriggeredRefinementForNextIteration))), 		&disp[39] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasErased))), 		&disp[40] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasTriggeredEraseForNextIteration))), 		&disp[41] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasChangedVertexOrCellState))), 		&disp[42] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._isTraversalInverted))), 		&disp[43] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._reduceStateAndCell))), 		&disp[44] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._couldNotEraseDueToDecompositionFlag))), 		&disp[45] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._subWorkerIsInvolvedInJoinOrFork))), 		&disp[46] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[1]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[47] );
            
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
         
         _senderDestinationRank = destination;
         
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
         
         _senderDestinationRank = status.MPI_SOURCE;
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
         assertion( _senderDestinationRank!=-1 );
         return _senderDestinationRank;
         
      }
   #endif
   
   
   peanoclaw::records::StatePacked::PersistentRecords::PersistentRecords() {
      assertion((9 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::StatePacked::PersistentRecords::PersistentRecords(const int& additionalLevelsForPredefinedRefinement, const bool& isInitializing, const bool& isRefinementCriterionEnabled, const bool& initialRefinmentTriggered, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const tarch::la::Vector<DIMENSIONS,double>& initialMaximalSubgridSize, const tarch::la::Vector<DIMENSIONS,int>& defaultSubdivisionFactor, const int& defaultGhostWidthLayer, const double& initialTimestepSize, const bool& useDimensionalSplittingOptimization, const double& globalTimestepEndTime, const bool& allPatchesEvolvedToGlobalTimestep, const tarch::la::Vector<DIMENSIONS,double>& domainOffset, const tarch::la::Vector<DIMENSIONS,double>& domainSize, const int& plotNumber, const int& subPlotNumber, const double& startMaximumGlobalTimeInterval, const double& endMaximumGlobalTimeInterval, const double& startMinimumGlobalTimeInterval, const double& endMinimumGlobalTimeInterval, const double& minimalTimestep, const double& totalNumberOfCellUpdates, const int& localHeightOfWorkerTree, const int& globalHeightOfWorkerTreeDuringLastIteration, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const double& numberOfInnerLeafVertices, const double& numberOfBoundaryLeafVertices, const double& numberOfOuterLeafVertices, const double& numberOfInnerLeafCells, const double& numberOfOuterLeafCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& isTraversalInverted, const bool& reduceStateAndCell, const bool& couldNotEraseDueToDecompositionFlag, const bool& subWorkerIsInvolvedInJoinOrFork):
   _additionalLevelsForPredefinedRefinement(additionalLevelsForPredefinedRefinement),
   _isInitializing(isInitializing),
   _isRefinementCriterionEnabled(isRefinementCriterionEnabled),
   _initialRefinmentTriggered(initialRefinmentTriggered),
   _unknownsPerSubcell(unknownsPerSubcell),
   _auxiliarFieldsPerSubcell(auxiliarFieldsPerSubcell),
   _initialMaximalSubgridSize(initialMaximalSubgridSize),
   _defaultSubdivisionFactor(defaultSubdivisionFactor),
   _defaultGhostWidthLayer(defaultGhostWidthLayer),
   _initialTimestepSize(initialTimestepSize),
   _useDimensionalSplittingOptimization(useDimensionalSplittingOptimization),
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
   _localHeightOfWorkerTree(localHeightOfWorkerTree),
   _globalHeightOfWorkerTreeDuringLastIteration(globalHeightOfWorkerTreeDuringLastIteration),
   _minMeshWidth(minMeshWidth),
   _maxMeshWidth(maxMeshWidth),
   _numberOfInnerVertices(numberOfInnerVertices),
   _numberOfBoundaryVertices(numberOfBoundaryVertices),
   _numberOfOuterVertices(numberOfOuterVertices),
   _numberOfInnerCells(numberOfInnerCells),
   _numberOfOuterCells(numberOfOuterCells),
   _numberOfInnerLeafVertices(numberOfInnerLeafVertices),
   _numberOfBoundaryLeafVertices(numberOfBoundaryLeafVertices),
   _numberOfOuterLeafVertices(numberOfOuterLeafVertices),
   _numberOfInnerLeafCells(numberOfInnerLeafCells),
   _numberOfOuterLeafCells(numberOfOuterLeafCells),
   _maxLevel(maxLevel),
   _isTraversalInverted(isTraversalInverted) {
      setAllPatchesEvolvedToGlobalTimestep(allPatchesEvolvedToGlobalTimestep);
      setHasRefined(hasRefined);
      setHasTriggeredRefinementForNextIteration(hasTriggeredRefinementForNextIteration);
      setHasErased(hasErased);
      setHasTriggeredEraseForNextIteration(hasTriggeredEraseForNextIteration);
      setHasChangedVertexOrCellState(hasChangedVertexOrCellState);
      setReduceStateAndCell(reduceStateAndCell);
      setCouldNotEraseDueToDecompositionFlag(couldNotEraseDueToDecompositionFlag);
      setSubWorkerIsInvolvedInJoinOrFork(subWorkerIsInvolvedInJoinOrFork);
      assertion((9 < (8 * sizeof(short int))));
      
   }
   
   peanoclaw::records::StatePacked::StatePacked() {
      assertion((9 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::StatePacked::StatePacked(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._additionalLevelsForPredefinedRefinement, persistentRecords._isInitializing, persistentRecords._isRefinementCriterionEnabled, persistentRecords._initialRefinmentTriggered, persistentRecords._unknownsPerSubcell, persistentRecords._auxiliarFieldsPerSubcell, persistentRecords._initialMaximalSubgridSize, persistentRecords._defaultSubdivisionFactor, persistentRecords._defaultGhostWidthLayer, persistentRecords._initialTimestepSize, persistentRecords._useDimensionalSplittingOptimization, persistentRecords._globalTimestepEndTime, persistentRecords.getAllPatchesEvolvedToGlobalTimestep(), persistentRecords._domainOffset, persistentRecords._domainSize, persistentRecords._plotNumber, persistentRecords._subPlotNumber, persistentRecords._startMaximumGlobalTimeInterval, persistentRecords._endMaximumGlobalTimeInterval, persistentRecords._startMinimumGlobalTimeInterval, persistentRecords._endMinimumGlobalTimeInterval, persistentRecords._minimalTimestep, persistentRecords._totalNumberOfCellUpdates, persistentRecords._localHeightOfWorkerTree, persistentRecords._globalHeightOfWorkerTreeDuringLastIteration, persistentRecords._minMeshWidth, persistentRecords._maxMeshWidth, persistentRecords._numberOfInnerVertices, persistentRecords._numberOfBoundaryVertices, persistentRecords._numberOfOuterVertices, persistentRecords._numberOfInnerCells, persistentRecords._numberOfOuterCells, persistentRecords._numberOfInnerLeafVertices, persistentRecords._numberOfBoundaryLeafVertices, persistentRecords._numberOfOuterLeafVertices, persistentRecords._numberOfInnerLeafCells, persistentRecords._numberOfOuterLeafCells, persistentRecords._maxLevel, persistentRecords.getHasRefined(), persistentRecords.getHasTriggeredRefinementForNextIteration(), persistentRecords.getHasErased(), persistentRecords.getHasTriggeredEraseForNextIteration(), persistentRecords.getHasChangedVertexOrCellState(), persistentRecords._isTraversalInverted, persistentRecords.getReduceStateAndCell(), persistentRecords.getCouldNotEraseDueToDecompositionFlag(), persistentRecords.getSubWorkerIsInvolvedInJoinOrFork()) {
      assertion((9 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::StatePacked::StatePacked(const int& additionalLevelsForPredefinedRefinement, const bool& isInitializing, const bool& isRefinementCriterionEnabled, const bool& initialRefinmentTriggered, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const tarch::la::Vector<DIMENSIONS,double>& initialMaximalSubgridSize, const tarch::la::Vector<DIMENSIONS,int>& defaultSubdivisionFactor, const int& defaultGhostWidthLayer, const double& initialTimestepSize, const bool& useDimensionalSplittingOptimization, const double& globalTimestepEndTime, const bool& allPatchesEvolvedToGlobalTimestep, const tarch::la::Vector<DIMENSIONS,double>& domainOffset, const tarch::la::Vector<DIMENSIONS,double>& domainSize, const int& plotNumber, const int& subPlotNumber, const double& startMaximumGlobalTimeInterval, const double& endMaximumGlobalTimeInterval, const double& startMinimumGlobalTimeInterval, const double& endMinimumGlobalTimeInterval, const double& minimalTimestep, const double& totalNumberOfCellUpdates, const int& localHeightOfWorkerTree, const int& globalHeightOfWorkerTreeDuringLastIteration, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const double& numberOfInnerLeafVertices, const double& numberOfBoundaryLeafVertices, const double& numberOfOuterLeafVertices, const double& numberOfInnerLeafCells, const double& numberOfOuterLeafCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& isTraversalInverted, const bool& reduceStateAndCell, const bool& couldNotEraseDueToDecompositionFlag, const bool& subWorkerIsInvolvedInJoinOrFork):
   _persistentRecords(additionalLevelsForPredefinedRefinement, isInitializing, isRefinementCriterionEnabled, initialRefinmentTriggered, unknownsPerSubcell, auxiliarFieldsPerSubcell, initialMaximalSubgridSize, defaultSubdivisionFactor, defaultGhostWidthLayer, initialTimestepSize, useDimensionalSplittingOptimization, globalTimestepEndTime, allPatchesEvolvedToGlobalTimestep, domainOffset, domainSize, plotNumber, subPlotNumber, startMaximumGlobalTimeInterval, endMaximumGlobalTimeInterval, startMinimumGlobalTimeInterval, endMinimumGlobalTimeInterval, minimalTimestep, totalNumberOfCellUpdates, localHeightOfWorkerTree, globalHeightOfWorkerTreeDuringLastIteration, minMeshWidth, maxMeshWidth, numberOfInnerVertices, numberOfBoundaryVertices, numberOfOuterVertices, numberOfInnerCells, numberOfOuterCells, numberOfInnerLeafVertices, numberOfBoundaryLeafVertices, numberOfOuterLeafVertices, numberOfInnerLeafCells, numberOfOuterLeafCells, maxLevel, hasRefined, hasTriggeredRefinementForNextIteration, hasErased, hasTriggeredEraseForNextIteration, hasChangedVertexOrCellState, isTraversalInverted, reduceStateAndCell, couldNotEraseDueToDecompositionFlag, subWorkerIsInvolvedInJoinOrFork) {
      assertion((9 < (8 * sizeof(short int))));
      
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
      out << "isRefinementCriterionEnabled:" << getIsRefinementCriterionEnabled();
      out << ",";
      out << "initialRefinmentTriggered:" << getInitialRefinmentTriggered();
      out << ",";
      out << "unknownsPerSubcell:" << getUnknownsPerSubcell();
      out << ",";
      out << "auxiliarFieldsPerSubcell:" << getAuxiliarFieldsPerSubcell();
      out << ",";
      out << "initialMaximalSubgridSize:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getInitialMaximalSubgridSize(i) << ",";
   }
   out << getInitialMaximalSubgridSize(DIMENSIONS-1) << "]";
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
      out << "useDimensionalSplittingOptimization:" << getUseDimensionalSplittingOptimization();
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
      out << "localHeightOfWorkerTree:" << getLocalHeightOfWorkerTree();
      out << ",";
      out << "globalHeightOfWorkerTreeDuringLastIteration:" << getGlobalHeightOfWorkerTreeDuringLastIteration();
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
      out << "numberOfInnerLeafVertices:" << getNumberOfInnerLeafVertices();
      out << ",";
      out << "numberOfBoundaryLeafVertices:" << getNumberOfBoundaryLeafVertices();
      out << ",";
      out << "numberOfOuterLeafVertices:" << getNumberOfOuterLeafVertices();
      out << ",";
      out << "numberOfInnerLeafCells:" << getNumberOfInnerLeafCells();
      out << ",";
      out << "numberOfOuterLeafCells:" << getNumberOfOuterLeafCells();
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
      out << "reduceStateAndCell:" << getReduceStateAndCell();
      out << ",";
      out << "couldNotEraseDueToDecompositionFlag:" << getCouldNotEraseDueToDecompositionFlag();
      out << ",";
      out << "subWorkerIsInvolvedInJoinOrFork:" << getSubWorkerIsInvolvedInJoinOrFork();
      out <<  ")";
   }
   
   
   peanoclaw::records::StatePacked::PersistentRecords peanoclaw::records::StatePacked::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::State peanoclaw::records::StatePacked::convert() const{
      return State(
         getAdditionalLevelsForPredefinedRefinement(),
         getIsInitializing(),
         getIsRefinementCriterionEnabled(),
         getInitialRefinmentTriggered(),
         getUnknownsPerSubcell(),
         getAuxiliarFieldsPerSubcell(),
         getInitialMaximalSubgridSize(),
         getDefaultSubdivisionFactor(),
         getDefaultGhostWidthLayer(),
         getInitialTimestepSize(),
         getUseDimensionalSplittingOptimization(),
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
         getLocalHeightOfWorkerTree(),
         getGlobalHeightOfWorkerTreeDuringLastIteration(),
         getMinMeshWidth(),
         getMaxMeshWidth(),
         getNumberOfInnerVertices(),
         getNumberOfBoundaryVertices(),
         getNumberOfOuterVertices(),
         getNumberOfInnerCells(),
         getNumberOfOuterCells(),
         getNumberOfInnerLeafVertices(),
         getNumberOfBoundaryLeafVertices(),
         getNumberOfOuterLeafVertices(),
         getNumberOfInnerLeafCells(),
         getNumberOfOuterLeafCells(),
         getMaxLevel(),
         getHasRefined(),
         getHasTriggeredRefinementForNextIteration(),
         getHasErased(),
         getHasTriggeredEraseForNextIteration(),
         getHasChangedVertexOrCellState(),
         getIsTraversalInverted(),
         getReduceStateAndCell(),
         getCouldNotEraseDueToDecompositionFlag(),
         getSubWorkerIsInvolvedInJoinOrFork()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::StatePacked::_log( "peanoclaw::records::StatePacked" );
      
      MPI_Datatype peanoclaw::records::StatePacked::Datatype = 0;
      MPI_Datatype peanoclaw::records::StatePacked::FullDatatype = 0;
      
      
      void peanoclaw::records::StatePacked::initDatatype() {
         {
            StatePacked dummyStatePacked[2];
            
            const int Attributes = 40;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //additionalLevelsForPredefinedRefinement
               MPI_CHAR,		 //isInitializing
               MPI_CHAR,		 //isRefinementCriterionEnabled
               MPI_CHAR,		 //initialRefinmentTriggered
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_DOUBLE,		 //initialMaximalSubgridSize
               MPI_INT,		 //defaultSubdivisionFactor
               MPI_INT,		 //defaultGhostWidthLayer
               MPI_DOUBLE,		 //initialTimestepSize
               MPI_CHAR,		 //useDimensionalSplittingOptimization
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
               MPI_INT,		 //localHeightOfWorkerTree
               MPI_INT,		 //globalHeightOfWorkerTreeDuringLastIteration
               MPI_DOUBLE,		 //minMeshWidth
               MPI_DOUBLE,		 //maxMeshWidth
               MPI_DOUBLE,		 //numberOfInnerVertices
               MPI_DOUBLE,		 //numberOfBoundaryVertices
               MPI_DOUBLE,		 //numberOfOuterVertices
               MPI_DOUBLE,		 //numberOfInnerCells
               MPI_DOUBLE,		 //numberOfOuterCells
               MPI_DOUBLE,		 //numberOfInnerLeafVertices
               MPI_DOUBLE,		 //numberOfBoundaryLeafVertices
               MPI_DOUBLE,		 //numberOfOuterLeafVertices
               MPI_DOUBLE,		 //numberOfInnerLeafCells
               MPI_DOUBLE,		 //numberOfOuterLeafCells
               MPI_INT,		 //maxLevel
               MPI_CHAR,		 //isTraversalInverted
               MPI_SHORT,		 //_packedRecords0
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //additionalLevelsForPredefinedRefinement
               1,		 //isInitializing
               1,		 //isRefinementCriterionEnabled
               1,		 //initialRefinmentTriggered
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               DIMENSIONS,		 //initialMaximalSubgridSize
               DIMENSIONS,		 //defaultSubdivisionFactor
               1,		 //defaultGhostWidthLayer
               1,		 //initialTimestepSize
               1,		 //useDimensionalSplittingOptimization
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
               1,		 //localHeightOfWorkerTree
               1,		 //globalHeightOfWorkerTreeDuringLastIteration
               DIMENSIONS,		 //minMeshWidth
               DIMENSIONS,		 //maxMeshWidth
               1,		 //numberOfInnerVertices
               1,		 //numberOfBoundaryVertices
               1,		 //numberOfOuterVertices
               1,		 //numberOfInnerCells
               1,		 //numberOfOuterCells
               1,		 //numberOfInnerLeafVertices
               1,		 //numberOfBoundaryLeafVertices
               1,		 //numberOfOuterLeafVertices
               1,		 //numberOfInnerLeafCells
               1,		 //numberOfOuterLeafCells
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
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._isRefinementCriterionEnabled))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialRefinmentTriggered))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._unknownsPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialMaximalSubgridSize[0]))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._defaultSubdivisionFactor[0]))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._defaultGhostWidthLayer))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialTimestepSize))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._useDimensionalSplittingOptimization))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._globalTimestepEndTime))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._domainOffset[0]))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._domainSize[0]))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._plotNumber))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._subPlotNumber))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._startMaximumGlobalTimeInterval))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._endMaximumGlobalTimeInterval))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._startMinimumGlobalTimeInterval))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._endMinimumGlobalTimeInterval))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._minimalTimestep))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._totalNumberOfCellUpdates))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._localHeightOfWorkerTree))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._globalHeightOfWorkerTreeDuringLastIteration))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._minMeshWidth[0]))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._maxMeshWidth[0]))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerVertices))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfBoundaryVertices))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterVertices))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerCells))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterCells))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerLeafVertices))), 		&disp[31] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfBoundaryLeafVertices))), 		&disp[32] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterLeafVertices))), 		&disp[33] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerLeafCells))), 		&disp[34] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterLeafCells))), 		&disp[35] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._maxLevel))), 		&disp[36] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._isTraversalInverted))), 		&disp[37] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._packedRecords0))), 		&disp[38] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[1]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[39] );
            
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
            
            const int Attributes = 40;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //additionalLevelsForPredefinedRefinement
               MPI_CHAR,		 //isInitializing
               MPI_CHAR,		 //isRefinementCriterionEnabled
               MPI_CHAR,		 //initialRefinmentTriggered
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_DOUBLE,		 //initialMaximalSubgridSize
               MPI_INT,		 //defaultSubdivisionFactor
               MPI_INT,		 //defaultGhostWidthLayer
               MPI_DOUBLE,		 //initialTimestepSize
               MPI_CHAR,		 //useDimensionalSplittingOptimization
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
               MPI_INT,		 //localHeightOfWorkerTree
               MPI_INT,		 //globalHeightOfWorkerTreeDuringLastIteration
               MPI_DOUBLE,		 //minMeshWidth
               MPI_DOUBLE,		 //maxMeshWidth
               MPI_DOUBLE,		 //numberOfInnerVertices
               MPI_DOUBLE,		 //numberOfBoundaryVertices
               MPI_DOUBLE,		 //numberOfOuterVertices
               MPI_DOUBLE,		 //numberOfInnerCells
               MPI_DOUBLE,		 //numberOfOuterCells
               MPI_DOUBLE,		 //numberOfInnerLeafVertices
               MPI_DOUBLE,		 //numberOfBoundaryLeafVertices
               MPI_DOUBLE,		 //numberOfOuterLeafVertices
               MPI_DOUBLE,		 //numberOfInnerLeafCells
               MPI_DOUBLE,		 //numberOfOuterLeafCells
               MPI_INT,		 //maxLevel
               MPI_CHAR,		 //isTraversalInverted
               MPI_SHORT,		 //_packedRecords0
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //additionalLevelsForPredefinedRefinement
               1,		 //isInitializing
               1,		 //isRefinementCriterionEnabled
               1,		 //initialRefinmentTriggered
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               DIMENSIONS,		 //initialMaximalSubgridSize
               DIMENSIONS,		 //defaultSubdivisionFactor
               1,		 //defaultGhostWidthLayer
               1,		 //initialTimestepSize
               1,		 //useDimensionalSplittingOptimization
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
               1,		 //localHeightOfWorkerTree
               1,		 //globalHeightOfWorkerTreeDuringLastIteration
               DIMENSIONS,		 //minMeshWidth
               DIMENSIONS,		 //maxMeshWidth
               1,		 //numberOfInnerVertices
               1,		 //numberOfBoundaryVertices
               1,		 //numberOfOuterVertices
               1,		 //numberOfInnerCells
               1,		 //numberOfOuterCells
               1,		 //numberOfInnerLeafVertices
               1,		 //numberOfBoundaryLeafVertices
               1,		 //numberOfOuterLeafVertices
               1,		 //numberOfInnerLeafCells
               1,		 //numberOfOuterLeafCells
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
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._isRefinementCriterionEnabled))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialRefinmentTriggered))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._unknownsPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialMaximalSubgridSize[0]))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._defaultSubdivisionFactor[0]))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._defaultGhostWidthLayer))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialTimestepSize))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._useDimensionalSplittingOptimization))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._globalTimestepEndTime))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._domainOffset[0]))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._domainSize[0]))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._plotNumber))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._subPlotNumber))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._startMaximumGlobalTimeInterval))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._endMaximumGlobalTimeInterval))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._startMinimumGlobalTimeInterval))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._endMinimumGlobalTimeInterval))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._minimalTimestep))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._totalNumberOfCellUpdates))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._localHeightOfWorkerTree))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._globalHeightOfWorkerTreeDuringLastIteration))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._minMeshWidth[0]))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._maxMeshWidth[0]))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerVertices))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfBoundaryVertices))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterVertices))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerCells))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterCells))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerLeafVertices))), 		&disp[31] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfBoundaryLeafVertices))), 		&disp[32] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterLeafVertices))), 		&disp[33] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerLeafCells))), 		&disp[34] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterLeafCells))), 		&disp[35] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._maxLevel))), 		&disp[36] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._isTraversalInverted))), 		&disp[37] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._packedRecords0))), 		&disp[38] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[1]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[39] );
            
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
         
         _senderDestinationRank = destination;
         
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
         
         _senderDestinationRank = status.MPI_SOURCE;
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
         assertion( _senderDestinationRank!=-1 );
         return _senderDestinationRank;
         
      }
   #endif
   
   
   
#elif !defined(Parallel)
   peanoclaw::records::State::PersistentRecords::PersistentRecords() {
      
   }
   
   
   peanoclaw::records::State::PersistentRecords::PersistentRecords(const int& additionalLevelsForPredefinedRefinement, const bool& isInitializing, const bool& isRefinementCriterionEnabled, const bool& initialRefinmentTriggered, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const tarch::la::Vector<DIMENSIONS,double>& initialMaximalSubgridSize, const tarch::la::Vector<DIMENSIONS,int>& defaultSubdivisionFactor, const int& defaultGhostWidthLayer, const double& initialTimestepSize, const bool& useDimensionalSplittingOptimization, const double& globalTimestepEndTime, const bool& allPatchesEvolvedToGlobalTimestep, const tarch::la::Vector<DIMENSIONS,double>& domainOffset, const tarch::la::Vector<DIMENSIONS,double>& domainSize, const int& plotNumber, const int& subPlotNumber, const double& startMaximumGlobalTimeInterval, const double& endMaximumGlobalTimeInterval, const double& startMinimumGlobalTimeInterval, const double& endMinimumGlobalTimeInterval, const double& minimalTimestep, const double& totalNumberOfCellUpdates, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const double& numberOfInnerLeafVertices, const double& numberOfBoundaryLeafVertices, const double& numberOfOuterLeafVertices, const double& numberOfInnerLeafCells, const double& numberOfOuterLeafCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& isTraversalInverted):
   _additionalLevelsForPredefinedRefinement(additionalLevelsForPredefinedRefinement),
   _isInitializing(isInitializing),
   _isRefinementCriterionEnabled(isRefinementCriterionEnabled),
   _initialRefinmentTriggered(initialRefinmentTriggered),
   _unknownsPerSubcell(unknownsPerSubcell),
   _auxiliarFieldsPerSubcell(auxiliarFieldsPerSubcell),
   _initialMaximalSubgridSize(initialMaximalSubgridSize),
   _defaultSubdivisionFactor(defaultSubdivisionFactor),
   _defaultGhostWidthLayer(defaultGhostWidthLayer),
   _initialTimestepSize(initialTimestepSize),
   _useDimensionalSplittingOptimization(useDimensionalSplittingOptimization),
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
   _numberOfInnerLeafVertices(numberOfInnerLeafVertices),
   _numberOfBoundaryLeafVertices(numberOfBoundaryLeafVertices),
   _numberOfOuterLeafVertices(numberOfOuterLeafVertices),
   _numberOfInnerLeafCells(numberOfInnerLeafCells),
   _numberOfOuterLeafCells(numberOfOuterLeafCells),
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
   _persistentRecords(persistentRecords._additionalLevelsForPredefinedRefinement, persistentRecords._isInitializing, persistentRecords._isRefinementCriterionEnabled, persistentRecords._initialRefinmentTriggered, persistentRecords._unknownsPerSubcell, persistentRecords._auxiliarFieldsPerSubcell, persistentRecords._initialMaximalSubgridSize, persistentRecords._defaultSubdivisionFactor, persistentRecords._defaultGhostWidthLayer, persistentRecords._initialTimestepSize, persistentRecords._useDimensionalSplittingOptimization, persistentRecords._globalTimestepEndTime, persistentRecords._allPatchesEvolvedToGlobalTimestep, persistentRecords._domainOffset, persistentRecords._domainSize, persistentRecords._plotNumber, persistentRecords._subPlotNumber, persistentRecords._startMaximumGlobalTimeInterval, persistentRecords._endMaximumGlobalTimeInterval, persistentRecords._startMinimumGlobalTimeInterval, persistentRecords._endMinimumGlobalTimeInterval, persistentRecords._minimalTimestep, persistentRecords._totalNumberOfCellUpdates, persistentRecords._minMeshWidth, persistentRecords._maxMeshWidth, persistentRecords._numberOfInnerVertices, persistentRecords._numberOfBoundaryVertices, persistentRecords._numberOfOuterVertices, persistentRecords._numberOfInnerCells, persistentRecords._numberOfOuterCells, persistentRecords._numberOfInnerLeafVertices, persistentRecords._numberOfBoundaryLeafVertices, persistentRecords._numberOfOuterLeafVertices, persistentRecords._numberOfInnerLeafCells, persistentRecords._numberOfOuterLeafCells, persistentRecords._maxLevel, persistentRecords._hasRefined, persistentRecords._hasTriggeredRefinementForNextIteration, persistentRecords._hasErased, persistentRecords._hasTriggeredEraseForNextIteration, persistentRecords._hasChangedVertexOrCellState, persistentRecords._isTraversalInverted) {
      
   }
   
   
   peanoclaw::records::State::State(const int& additionalLevelsForPredefinedRefinement, const bool& isInitializing, const bool& isRefinementCriterionEnabled, const bool& initialRefinmentTriggered, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const tarch::la::Vector<DIMENSIONS,double>& initialMaximalSubgridSize, const tarch::la::Vector<DIMENSIONS,int>& defaultSubdivisionFactor, const int& defaultGhostWidthLayer, const double& initialTimestepSize, const bool& useDimensionalSplittingOptimization, const double& globalTimestepEndTime, const bool& allPatchesEvolvedToGlobalTimestep, const tarch::la::Vector<DIMENSIONS,double>& domainOffset, const tarch::la::Vector<DIMENSIONS,double>& domainSize, const int& plotNumber, const int& subPlotNumber, const double& startMaximumGlobalTimeInterval, const double& endMaximumGlobalTimeInterval, const double& startMinimumGlobalTimeInterval, const double& endMinimumGlobalTimeInterval, const double& minimalTimestep, const double& totalNumberOfCellUpdates, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const double& numberOfInnerLeafVertices, const double& numberOfBoundaryLeafVertices, const double& numberOfOuterLeafVertices, const double& numberOfInnerLeafCells, const double& numberOfOuterLeafCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& isTraversalInverted):
   _persistentRecords(additionalLevelsForPredefinedRefinement, isInitializing, isRefinementCriterionEnabled, initialRefinmentTriggered, unknownsPerSubcell, auxiliarFieldsPerSubcell, initialMaximalSubgridSize, defaultSubdivisionFactor, defaultGhostWidthLayer, initialTimestepSize, useDimensionalSplittingOptimization, globalTimestepEndTime, allPatchesEvolvedToGlobalTimestep, domainOffset, domainSize, plotNumber, subPlotNumber, startMaximumGlobalTimeInterval, endMaximumGlobalTimeInterval, startMinimumGlobalTimeInterval, endMinimumGlobalTimeInterval, minimalTimestep, totalNumberOfCellUpdates, minMeshWidth, maxMeshWidth, numberOfInnerVertices, numberOfBoundaryVertices, numberOfOuterVertices, numberOfInnerCells, numberOfOuterCells, numberOfInnerLeafVertices, numberOfBoundaryLeafVertices, numberOfOuterLeafVertices, numberOfInnerLeafCells, numberOfOuterLeafCells, maxLevel, hasRefined, hasTriggeredRefinementForNextIteration, hasErased, hasTriggeredEraseForNextIteration, hasChangedVertexOrCellState, isTraversalInverted) {
      
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
      out << "isRefinementCriterionEnabled:" << getIsRefinementCriterionEnabled();
      out << ",";
      out << "initialRefinmentTriggered:" << getInitialRefinmentTriggered();
      out << ",";
      out << "unknownsPerSubcell:" << getUnknownsPerSubcell();
      out << ",";
      out << "auxiliarFieldsPerSubcell:" << getAuxiliarFieldsPerSubcell();
      out << ",";
      out << "initialMaximalSubgridSize:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getInitialMaximalSubgridSize(i) << ",";
   }
   out << getInitialMaximalSubgridSize(DIMENSIONS-1) << "]";
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
      out << "useDimensionalSplittingOptimization:" << getUseDimensionalSplittingOptimization();
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
      out << "numberOfInnerLeafVertices:" << getNumberOfInnerLeafVertices();
      out << ",";
      out << "numberOfBoundaryLeafVertices:" << getNumberOfBoundaryLeafVertices();
      out << ",";
      out << "numberOfOuterLeafVertices:" << getNumberOfOuterLeafVertices();
      out << ",";
      out << "numberOfInnerLeafCells:" << getNumberOfInnerLeafCells();
      out << ",";
      out << "numberOfOuterLeafCells:" << getNumberOfOuterLeafCells();
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
         getIsRefinementCriterionEnabled(),
         getInitialRefinmentTriggered(),
         getUnknownsPerSubcell(),
         getAuxiliarFieldsPerSubcell(),
         getInitialMaximalSubgridSize(),
         getDefaultSubdivisionFactor(),
         getDefaultGhostWidthLayer(),
         getInitialTimestepSize(),
         getUseDimensionalSplittingOptimization(),
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
         getNumberOfInnerLeafVertices(),
         getNumberOfBoundaryLeafVertices(),
         getNumberOfOuterLeafVertices(),
         getNumberOfInnerLeafCells(),
         getNumberOfOuterLeafCells(),
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
            
            const int Attributes = 43;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //additionalLevelsForPredefinedRefinement
               MPI_CHAR,		 //isInitializing
               MPI_CHAR,		 //isRefinementCriterionEnabled
               MPI_CHAR,		 //initialRefinmentTriggered
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_DOUBLE,		 //initialMaximalSubgridSize
               MPI_INT,		 //defaultSubdivisionFactor
               MPI_INT,		 //defaultGhostWidthLayer
               MPI_DOUBLE,		 //initialTimestepSize
               MPI_CHAR,		 //useDimensionalSplittingOptimization
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
               MPI_DOUBLE,		 //numberOfInnerLeafVertices
               MPI_DOUBLE,		 //numberOfBoundaryLeafVertices
               MPI_DOUBLE,		 //numberOfOuterLeafVertices
               MPI_DOUBLE,		 //numberOfInnerLeafCells
               MPI_DOUBLE,		 //numberOfOuterLeafCells
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
               1,		 //isRefinementCriterionEnabled
               1,		 //initialRefinmentTriggered
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               DIMENSIONS,		 //initialMaximalSubgridSize
               DIMENSIONS,		 //defaultSubdivisionFactor
               1,		 //defaultGhostWidthLayer
               1,		 //initialTimestepSize
               1,		 //useDimensionalSplittingOptimization
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
               1,		 //numberOfInnerLeafVertices
               1,		 //numberOfBoundaryLeafVertices
               1,		 //numberOfOuterLeafVertices
               1,		 //numberOfInnerLeafCells
               1,		 //numberOfOuterLeafCells
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
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._isRefinementCriterionEnabled))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialRefinmentTriggered))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._unknownsPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialMaximalSubgridSize[0]))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._defaultSubdivisionFactor[0]))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._defaultGhostWidthLayer))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialTimestepSize))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._useDimensionalSplittingOptimization))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._globalTimestepEndTime))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._allPatchesEvolvedToGlobalTimestep))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._domainOffset[0]))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._domainSize[0]))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._plotNumber))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._subPlotNumber))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._startMaximumGlobalTimeInterval))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._endMaximumGlobalTimeInterval))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._startMinimumGlobalTimeInterval))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._endMinimumGlobalTimeInterval))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._minimalTimestep))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._totalNumberOfCellUpdates))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._minMeshWidth[0]))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._maxMeshWidth[0]))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerVertices))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfBoundaryVertices))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterVertices))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerCells))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterCells))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerLeafVertices))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfBoundaryLeafVertices))), 		&disp[31] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterLeafVertices))), 		&disp[32] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerLeafCells))), 		&disp[33] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterLeafCells))), 		&disp[34] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._maxLevel))), 		&disp[35] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasRefined))), 		&disp[36] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasTriggeredRefinementForNextIteration))), 		&disp[37] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasErased))), 		&disp[38] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasTriggeredEraseForNextIteration))), 		&disp[39] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasChangedVertexOrCellState))), 		&disp[40] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._isTraversalInverted))), 		&disp[41] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[1]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[42] );
            
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
            
            const int Attributes = 43;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //additionalLevelsForPredefinedRefinement
               MPI_CHAR,		 //isInitializing
               MPI_CHAR,		 //isRefinementCriterionEnabled
               MPI_CHAR,		 //initialRefinmentTriggered
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_DOUBLE,		 //initialMaximalSubgridSize
               MPI_INT,		 //defaultSubdivisionFactor
               MPI_INT,		 //defaultGhostWidthLayer
               MPI_DOUBLE,		 //initialTimestepSize
               MPI_CHAR,		 //useDimensionalSplittingOptimization
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
               MPI_DOUBLE,		 //numberOfInnerLeafVertices
               MPI_DOUBLE,		 //numberOfBoundaryLeafVertices
               MPI_DOUBLE,		 //numberOfOuterLeafVertices
               MPI_DOUBLE,		 //numberOfInnerLeafCells
               MPI_DOUBLE,		 //numberOfOuterLeafCells
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
               1,		 //isRefinementCriterionEnabled
               1,		 //initialRefinmentTriggered
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               DIMENSIONS,		 //initialMaximalSubgridSize
               DIMENSIONS,		 //defaultSubdivisionFactor
               1,		 //defaultGhostWidthLayer
               1,		 //initialTimestepSize
               1,		 //useDimensionalSplittingOptimization
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
               1,		 //numberOfInnerLeafVertices
               1,		 //numberOfBoundaryLeafVertices
               1,		 //numberOfOuterLeafVertices
               1,		 //numberOfInnerLeafCells
               1,		 //numberOfOuterLeafCells
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
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._isRefinementCriterionEnabled))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialRefinmentTriggered))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._unknownsPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialMaximalSubgridSize[0]))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._defaultSubdivisionFactor[0]))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._defaultGhostWidthLayer))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._initialTimestepSize))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._useDimensionalSplittingOptimization))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._globalTimestepEndTime))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._allPatchesEvolvedToGlobalTimestep))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._domainOffset[0]))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._domainSize[0]))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._plotNumber))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._subPlotNumber))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._startMaximumGlobalTimeInterval))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._endMaximumGlobalTimeInterval))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._startMinimumGlobalTimeInterval))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._endMinimumGlobalTimeInterval))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._minimalTimestep))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._totalNumberOfCellUpdates))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._minMeshWidth[0]))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._maxMeshWidth[0]))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerVertices))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfBoundaryVertices))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterVertices))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerCells))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterCells))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerLeafVertices))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfBoundaryLeafVertices))), 		&disp[31] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterLeafVertices))), 		&disp[32] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfInnerLeafCells))), 		&disp[33] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._numberOfOuterLeafCells))), 		&disp[34] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._maxLevel))), 		&disp[35] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasRefined))), 		&disp[36] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasTriggeredRefinementForNextIteration))), 		&disp[37] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasErased))), 		&disp[38] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasTriggeredEraseForNextIteration))), 		&disp[39] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._hasChangedVertexOrCellState))), 		&disp[40] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[0]._persistentRecords._isTraversalInverted))), 		&disp[41] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyState[1]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[42] );
            
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
         
         _senderDestinationRank = destination;
         
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
         
         _senderDestinationRank = status.MPI_SOURCE;
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
         assertion( _senderDestinationRank!=-1 );
         return _senderDestinationRank;
         
      }
   #endif
   
   
   peanoclaw::records::StatePacked::PersistentRecords::PersistentRecords() {
      assertion((6 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::StatePacked::PersistentRecords::PersistentRecords(const int& additionalLevelsForPredefinedRefinement, const bool& isInitializing, const bool& isRefinementCriterionEnabled, const bool& initialRefinmentTriggered, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const tarch::la::Vector<DIMENSIONS,double>& initialMaximalSubgridSize, const tarch::la::Vector<DIMENSIONS,int>& defaultSubdivisionFactor, const int& defaultGhostWidthLayer, const double& initialTimestepSize, const bool& useDimensionalSplittingOptimization, const double& globalTimestepEndTime, const bool& allPatchesEvolvedToGlobalTimestep, const tarch::la::Vector<DIMENSIONS,double>& domainOffset, const tarch::la::Vector<DIMENSIONS,double>& domainSize, const int& plotNumber, const int& subPlotNumber, const double& startMaximumGlobalTimeInterval, const double& endMaximumGlobalTimeInterval, const double& startMinimumGlobalTimeInterval, const double& endMinimumGlobalTimeInterval, const double& minimalTimestep, const double& totalNumberOfCellUpdates, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const double& numberOfInnerLeafVertices, const double& numberOfBoundaryLeafVertices, const double& numberOfOuterLeafVertices, const double& numberOfInnerLeafCells, const double& numberOfOuterLeafCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& isTraversalInverted):
   _additionalLevelsForPredefinedRefinement(additionalLevelsForPredefinedRefinement),
   _isInitializing(isInitializing),
   _isRefinementCriterionEnabled(isRefinementCriterionEnabled),
   _initialRefinmentTriggered(initialRefinmentTriggered),
   _unknownsPerSubcell(unknownsPerSubcell),
   _auxiliarFieldsPerSubcell(auxiliarFieldsPerSubcell),
   _initialMaximalSubgridSize(initialMaximalSubgridSize),
   _defaultSubdivisionFactor(defaultSubdivisionFactor),
   _defaultGhostWidthLayer(defaultGhostWidthLayer),
   _initialTimestepSize(initialTimestepSize),
   _useDimensionalSplittingOptimization(useDimensionalSplittingOptimization),
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
   _numberOfInnerLeafVertices(numberOfInnerLeafVertices),
   _numberOfBoundaryLeafVertices(numberOfBoundaryLeafVertices),
   _numberOfOuterLeafVertices(numberOfOuterLeafVertices),
   _numberOfInnerLeafCells(numberOfInnerLeafCells),
   _numberOfOuterLeafCells(numberOfOuterLeafCells),
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
   _persistentRecords(persistentRecords._additionalLevelsForPredefinedRefinement, persistentRecords._isInitializing, persistentRecords._isRefinementCriterionEnabled, persistentRecords._initialRefinmentTriggered, persistentRecords._unknownsPerSubcell, persistentRecords._auxiliarFieldsPerSubcell, persistentRecords._initialMaximalSubgridSize, persistentRecords._defaultSubdivisionFactor, persistentRecords._defaultGhostWidthLayer, persistentRecords._initialTimestepSize, persistentRecords._useDimensionalSplittingOptimization, persistentRecords._globalTimestepEndTime, persistentRecords.getAllPatchesEvolvedToGlobalTimestep(), persistentRecords._domainOffset, persistentRecords._domainSize, persistentRecords._plotNumber, persistentRecords._subPlotNumber, persistentRecords._startMaximumGlobalTimeInterval, persistentRecords._endMaximumGlobalTimeInterval, persistentRecords._startMinimumGlobalTimeInterval, persistentRecords._endMinimumGlobalTimeInterval, persistentRecords._minimalTimestep, persistentRecords._totalNumberOfCellUpdates, persistentRecords._minMeshWidth, persistentRecords._maxMeshWidth, persistentRecords._numberOfInnerVertices, persistentRecords._numberOfBoundaryVertices, persistentRecords._numberOfOuterVertices, persistentRecords._numberOfInnerCells, persistentRecords._numberOfOuterCells, persistentRecords._numberOfInnerLeafVertices, persistentRecords._numberOfBoundaryLeafVertices, persistentRecords._numberOfOuterLeafVertices, persistentRecords._numberOfInnerLeafCells, persistentRecords._numberOfOuterLeafCells, persistentRecords._maxLevel, persistentRecords.getHasRefined(), persistentRecords.getHasTriggeredRefinementForNextIteration(), persistentRecords.getHasErased(), persistentRecords.getHasTriggeredEraseForNextIteration(), persistentRecords.getHasChangedVertexOrCellState(), persistentRecords._isTraversalInverted) {
      assertion((6 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::StatePacked::StatePacked(const int& additionalLevelsForPredefinedRefinement, const bool& isInitializing, const bool& isRefinementCriterionEnabled, const bool& initialRefinmentTriggered, const int& unknownsPerSubcell, const int& auxiliarFieldsPerSubcell, const tarch::la::Vector<DIMENSIONS,double>& initialMaximalSubgridSize, const tarch::la::Vector<DIMENSIONS,int>& defaultSubdivisionFactor, const int& defaultGhostWidthLayer, const double& initialTimestepSize, const bool& useDimensionalSplittingOptimization, const double& globalTimestepEndTime, const bool& allPatchesEvolvedToGlobalTimestep, const tarch::la::Vector<DIMENSIONS,double>& domainOffset, const tarch::la::Vector<DIMENSIONS,double>& domainSize, const int& plotNumber, const int& subPlotNumber, const double& startMaximumGlobalTimeInterval, const double& endMaximumGlobalTimeInterval, const double& startMinimumGlobalTimeInterval, const double& endMinimumGlobalTimeInterval, const double& minimalTimestep, const double& totalNumberOfCellUpdates, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const double& numberOfInnerLeafVertices, const double& numberOfBoundaryLeafVertices, const double& numberOfOuterLeafVertices, const double& numberOfInnerLeafCells, const double& numberOfOuterLeafCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& isTraversalInverted):
   _persistentRecords(additionalLevelsForPredefinedRefinement, isInitializing, isRefinementCriterionEnabled, initialRefinmentTriggered, unknownsPerSubcell, auxiliarFieldsPerSubcell, initialMaximalSubgridSize, defaultSubdivisionFactor, defaultGhostWidthLayer, initialTimestepSize, useDimensionalSplittingOptimization, globalTimestepEndTime, allPatchesEvolvedToGlobalTimestep, domainOffset, domainSize, plotNumber, subPlotNumber, startMaximumGlobalTimeInterval, endMaximumGlobalTimeInterval, startMinimumGlobalTimeInterval, endMinimumGlobalTimeInterval, minimalTimestep, totalNumberOfCellUpdates, minMeshWidth, maxMeshWidth, numberOfInnerVertices, numberOfBoundaryVertices, numberOfOuterVertices, numberOfInnerCells, numberOfOuterCells, numberOfInnerLeafVertices, numberOfBoundaryLeafVertices, numberOfOuterLeafVertices, numberOfInnerLeafCells, numberOfOuterLeafCells, maxLevel, hasRefined, hasTriggeredRefinementForNextIteration, hasErased, hasTriggeredEraseForNextIteration, hasChangedVertexOrCellState, isTraversalInverted) {
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
      out << "isRefinementCriterionEnabled:" << getIsRefinementCriterionEnabled();
      out << ",";
      out << "initialRefinmentTriggered:" << getInitialRefinmentTriggered();
      out << ",";
      out << "unknownsPerSubcell:" << getUnknownsPerSubcell();
      out << ",";
      out << "auxiliarFieldsPerSubcell:" << getAuxiliarFieldsPerSubcell();
      out << ",";
      out << "initialMaximalSubgridSize:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getInitialMaximalSubgridSize(i) << ",";
   }
   out << getInitialMaximalSubgridSize(DIMENSIONS-1) << "]";
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
      out << "useDimensionalSplittingOptimization:" << getUseDimensionalSplittingOptimization();
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
      out << "numberOfInnerLeafVertices:" << getNumberOfInnerLeafVertices();
      out << ",";
      out << "numberOfBoundaryLeafVertices:" << getNumberOfBoundaryLeafVertices();
      out << ",";
      out << "numberOfOuterLeafVertices:" << getNumberOfOuterLeafVertices();
      out << ",";
      out << "numberOfInnerLeafCells:" << getNumberOfInnerLeafCells();
      out << ",";
      out << "numberOfOuterLeafCells:" << getNumberOfOuterLeafCells();
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
         getIsRefinementCriterionEnabled(),
         getInitialRefinmentTriggered(),
         getUnknownsPerSubcell(),
         getAuxiliarFieldsPerSubcell(),
         getInitialMaximalSubgridSize(),
         getDefaultSubdivisionFactor(),
         getDefaultGhostWidthLayer(),
         getInitialTimestepSize(),
         getUseDimensionalSplittingOptimization(),
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
         getNumberOfInnerLeafVertices(),
         getNumberOfBoundaryLeafVertices(),
         getNumberOfOuterLeafVertices(),
         getNumberOfInnerLeafCells(),
         getNumberOfOuterLeafCells(),
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
            
            const int Attributes = 38;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //additionalLevelsForPredefinedRefinement
               MPI_CHAR,		 //isInitializing
               MPI_CHAR,		 //isRefinementCriterionEnabled
               MPI_CHAR,		 //initialRefinmentTriggered
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_DOUBLE,		 //initialMaximalSubgridSize
               MPI_INT,		 //defaultSubdivisionFactor
               MPI_INT,		 //defaultGhostWidthLayer
               MPI_DOUBLE,		 //initialTimestepSize
               MPI_CHAR,		 //useDimensionalSplittingOptimization
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
               MPI_DOUBLE,		 //numberOfInnerLeafVertices
               MPI_DOUBLE,		 //numberOfBoundaryLeafVertices
               MPI_DOUBLE,		 //numberOfOuterLeafVertices
               MPI_DOUBLE,		 //numberOfInnerLeafCells
               MPI_DOUBLE,		 //numberOfOuterLeafCells
               MPI_INT,		 //maxLevel
               MPI_CHAR,		 //isTraversalInverted
               MPI_SHORT,		 //_packedRecords0
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //additionalLevelsForPredefinedRefinement
               1,		 //isInitializing
               1,		 //isRefinementCriterionEnabled
               1,		 //initialRefinmentTriggered
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               DIMENSIONS,		 //initialMaximalSubgridSize
               DIMENSIONS,		 //defaultSubdivisionFactor
               1,		 //defaultGhostWidthLayer
               1,		 //initialTimestepSize
               1,		 //useDimensionalSplittingOptimization
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
               1,		 //numberOfInnerLeafVertices
               1,		 //numberOfBoundaryLeafVertices
               1,		 //numberOfOuterLeafVertices
               1,		 //numberOfInnerLeafCells
               1,		 //numberOfOuterLeafCells
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
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._isRefinementCriterionEnabled))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialRefinmentTriggered))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._unknownsPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialMaximalSubgridSize[0]))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._defaultSubdivisionFactor[0]))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._defaultGhostWidthLayer))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialTimestepSize))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._useDimensionalSplittingOptimization))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._globalTimestepEndTime))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._domainOffset[0]))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._domainSize[0]))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._plotNumber))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._subPlotNumber))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._startMaximumGlobalTimeInterval))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._endMaximumGlobalTimeInterval))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._startMinimumGlobalTimeInterval))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._endMinimumGlobalTimeInterval))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._minimalTimestep))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._totalNumberOfCellUpdates))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._minMeshWidth[0]))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._maxMeshWidth[0]))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerVertices))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfBoundaryVertices))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterVertices))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerCells))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterCells))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerLeafVertices))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfBoundaryLeafVertices))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterLeafVertices))), 		&disp[31] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerLeafCells))), 		&disp[32] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterLeafCells))), 		&disp[33] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._maxLevel))), 		&disp[34] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._isTraversalInverted))), 		&disp[35] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._packedRecords0))), 		&disp[36] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[1]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[37] );
            
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
            
            const int Attributes = 38;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //additionalLevelsForPredefinedRefinement
               MPI_CHAR,		 //isInitializing
               MPI_CHAR,		 //isRefinementCriterionEnabled
               MPI_CHAR,		 //initialRefinmentTriggered
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //auxiliarFieldsPerSubcell
               MPI_DOUBLE,		 //initialMaximalSubgridSize
               MPI_INT,		 //defaultSubdivisionFactor
               MPI_INT,		 //defaultGhostWidthLayer
               MPI_DOUBLE,		 //initialTimestepSize
               MPI_CHAR,		 //useDimensionalSplittingOptimization
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
               MPI_DOUBLE,		 //numberOfInnerLeafVertices
               MPI_DOUBLE,		 //numberOfBoundaryLeafVertices
               MPI_DOUBLE,		 //numberOfOuterLeafVertices
               MPI_DOUBLE,		 //numberOfInnerLeafCells
               MPI_DOUBLE,		 //numberOfOuterLeafCells
               MPI_INT,		 //maxLevel
               MPI_CHAR,		 //isTraversalInverted
               MPI_SHORT,		 //_packedRecords0
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //additionalLevelsForPredefinedRefinement
               1,		 //isInitializing
               1,		 //isRefinementCriterionEnabled
               1,		 //initialRefinmentTriggered
               1,		 //unknownsPerSubcell
               1,		 //auxiliarFieldsPerSubcell
               DIMENSIONS,		 //initialMaximalSubgridSize
               DIMENSIONS,		 //defaultSubdivisionFactor
               1,		 //defaultGhostWidthLayer
               1,		 //initialTimestepSize
               1,		 //useDimensionalSplittingOptimization
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
               1,		 //numberOfInnerLeafVertices
               1,		 //numberOfBoundaryLeafVertices
               1,		 //numberOfOuterLeafVertices
               1,		 //numberOfInnerLeafCells
               1,		 //numberOfOuterLeafCells
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
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._isRefinementCriterionEnabled))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialRefinmentTriggered))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._unknownsPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._auxiliarFieldsPerSubcell))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialMaximalSubgridSize[0]))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._defaultSubdivisionFactor[0]))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._defaultGhostWidthLayer))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._initialTimestepSize))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._useDimensionalSplittingOptimization))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._globalTimestepEndTime))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._domainOffset[0]))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._domainSize[0]))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._plotNumber))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._subPlotNumber))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._startMaximumGlobalTimeInterval))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._endMaximumGlobalTimeInterval))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._startMinimumGlobalTimeInterval))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._endMinimumGlobalTimeInterval))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._minimalTimestep))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._totalNumberOfCellUpdates))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._minMeshWidth[0]))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._maxMeshWidth[0]))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerVertices))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfBoundaryVertices))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterVertices))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerCells))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterCells))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerLeafVertices))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfBoundaryLeafVertices))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterLeafVertices))), 		&disp[31] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfInnerLeafCells))), 		&disp[32] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._numberOfOuterLeafCells))), 		&disp[33] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._maxLevel))), 		&disp[34] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._isTraversalInverted))), 		&disp[35] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[0]._persistentRecords._packedRecords0))), 		&disp[36] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyStatePacked[1]._persistentRecords._additionalLevelsForPredefinedRefinement))), 		&disp[37] );
            
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
         
         _senderDestinationRank = destination;
         
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
         
         _senderDestinationRank = status.MPI_SOURCE;
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
         assertion( _senderDestinationRank!=-1 );
         return _senderDestinationRank;
         
      }
   #endif
   
   
   

#endif


