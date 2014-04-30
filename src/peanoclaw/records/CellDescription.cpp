#include "peanoclaw/records/CellDescription.h"

#if defined(Parallel)
   peanoclaw::records::CellDescription::PersistentRecords::PersistentRecords() {
      
   }
   
   
   peanoclaw::records::CellDescription::PersistentRecords::PersistentRecords(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostlayerWidth, const int& unknownsPerSubcell, const int& numberOfParametersWithoutGhostlayerPerSubcell, const int& numberOfParametersWithGhostlayerPerSubcell, const int& level, const bool& isVirtual, const bool& isRemote, const bool& isPaddingSubgrid, const tarch::la::Vector<THREE_POWER_D_MINUS_ONE,int>& numberOfTransfersToBeSkipped, const tarch::la::Vector<THREE_POWER_D_MINUS_ONE,int>& numberOfSharedAdjacentVertices, const bool& currentStateWasSent, const bool& markStateAsSentInNextIteration, const bool& adjacentRanksChanged, const tarch::la::Vector<THREE_POWER_D_MINUS_ONE,int>& adjacentRanks, const tarch::la::Vector<THREE_POWER_D_MINUS_ONE,int>& overlapByRemoteGhostlayer, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& maximumFineGridTime, const double& minimumFineGridTimestep, const bool& synchronizeFineGrids, const bool& willCoarsen, const double& minimalNeighborTimeConstraint, const int& constrainingNeighborIndex, const double& minimalLeafNeighborTimeConstraint, const double& minimalNeighborTime, const double& maximalNeighborTimestep, const double& estimatedNextTimestepSize, const int& skipGridIterations, const int& ageInGridIterations, const tarch::la::Vector<DIMENSIONS,double>& demandedMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds, const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds, const int& cellDescriptionIndex, const int& uIndex):
   _subdivisionFactor(subdivisionFactor),
   _ghostlayerWidth(ghostlayerWidth),
   _unknownsPerSubcell(unknownsPerSubcell),
   _numberOfParametersWithoutGhostlayerPerSubcell(numberOfParametersWithoutGhostlayerPerSubcell),
   _numberOfParametersWithGhostlayerPerSubcell(numberOfParametersWithGhostlayerPerSubcell),
   _level(level),
   _isVirtual(isVirtual),
   _isRemote(isRemote),
   _isPaddingSubgrid(isPaddingSubgrid),
   _numberOfTransfersToBeSkipped(numberOfTransfersToBeSkipped),
   _numberOfSharedAdjacentVertices(numberOfSharedAdjacentVertices),
   _currentStateWasSent(currentStateWasSent),
   _markStateAsSentInNextIteration(markStateAsSentInNextIteration),
   _adjacentRanksChanged(adjacentRanksChanged),
   _adjacentRanks(adjacentRanks),
   _overlapByRemoteGhostlayer(overlapByRemoteGhostlayer),
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
   _uIndex(uIndex) {
      
   }
   
   peanoclaw::records::CellDescription::CellDescription() {
      
   }
   
   
   peanoclaw::records::CellDescription::CellDescription(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._subdivisionFactor, persistentRecords._ghostlayerWidth, persistentRecords._unknownsPerSubcell, persistentRecords._numberOfParametersWithoutGhostlayerPerSubcell, persistentRecords._numberOfParametersWithGhostlayerPerSubcell, persistentRecords._level, persistentRecords._isVirtual, persistentRecords._isRemote, persistentRecords._isPaddingSubgrid, persistentRecords._numberOfTransfersToBeSkipped, persistentRecords._numberOfSharedAdjacentVertices, persistentRecords._currentStateWasSent, persistentRecords._markStateAsSentInNextIteration, persistentRecords._adjacentRanksChanged, persistentRecords._adjacentRanks, persistentRecords._overlapByRemoteGhostlayer, persistentRecords._position, persistentRecords._size, persistentRecords._time, persistentRecords._timestepSize, persistentRecords._maximumFineGridTime, persistentRecords._minimumFineGridTimestep, persistentRecords._synchronizeFineGrids, persistentRecords._willCoarsen, persistentRecords._minimalNeighborTimeConstraint, persistentRecords._constrainingNeighborIndex, persistentRecords._minimalLeafNeighborTimeConstraint, persistentRecords._minimalNeighborTime, persistentRecords._maximalNeighborTimestep, persistentRecords._estimatedNextTimestepSize, persistentRecords._skipGridIterations, persistentRecords._ageInGridIterations, persistentRecords._demandedMeshWidth, persistentRecords._restrictionLowerBounds, persistentRecords._restrictionUpperBounds, persistentRecords._cellDescriptionIndex, persistentRecords._uIndex) {
      
   }
   
   
   peanoclaw::records::CellDescription::CellDescription(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostlayerWidth, const int& unknownsPerSubcell, const int& numberOfParametersWithoutGhostlayerPerSubcell, const int& numberOfParametersWithGhostlayerPerSubcell, const int& level, const bool& isVirtual, const bool& isRemote, const bool& isPaddingSubgrid, const tarch::la::Vector<THREE_POWER_D_MINUS_ONE,int>& numberOfTransfersToBeSkipped, const tarch::la::Vector<THREE_POWER_D_MINUS_ONE,int>& numberOfSharedAdjacentVertices, const bool& currentStateWasSent, const bool& markStateAsSentInNextIteration, const bool& adjacentRanksChanged, const tarch::la::Vector<THREE_POWER_D_MINUS_ONE,int>& adjacentRanks, const tarch::la::Vector<THREE_POWER_D_MINUS_ONE,int>& overlapByRemoteGhostlayer, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& maximumFineGridTime, const double& minimumFineGridTimestep, const bool& synchronizeFineGrids, const bool& willCoarsen, const double& minimalNeighborTimeConstraint, const int& constrainingNeighborIndex, const double& minimalLeafNeighborTimeConstraint, const double& minimalNeighborTime, const double& maximalNeighborTimestep, const double& estimatedNextTimestepSize, const int& skipGridIterations, const int& ageInGridIterations, const tarch::la::Vector<DIMENSIONS,double>& demandedMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds, const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds, const int& cellDescriptionIndex, const int& uIndex):
   _persistentRecords(subdivisionFactor, ghostlayerWidth, unknownsPerSubcell, numberOfParametersWithoutGhostlayerPerSubcell, numberOfParametersWithGhostlayerPerSubcell, level, isVirtual, isRemote, isPaddingSubgrid, numberOfTransfersToBeSkipped, numberOfSharedAdjacentVertices, currentStateWasSent, markStateAsSentInNextIteration, adjacentRanksChanged, adjacentRanks, overlapByRemoteGhostlayer, position, size, time, timestepSize, maximumFineGridTime, minimumFineGridTimestep, synchronizeFineGrids, willCoarsen, minimalNeighborTimeConstraint, constrainingNeighborIndex, minimalLeafNeighborTimeConstraint, minimalNeighborTime, maximalNeighborTimestep, estimatedNextTimestepSize, skipGridIterations, ageInGridIterations, demandedMeshWidth, restrictionLowerBounds, restrictionUpperBounds, cellDescriptionIndex, uIndex) {
      
   }
   
   
   peanoclaw::records::CellDescription::~CellDescription() { }
   
   
   
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
      out << "ghostlayerWidth:" << getGhostlayerWidth();
      out << ",";
      out << "unknownsPerSubcell:" << getUnknownsPerSubcell();
      out << ",";
      out << "numberOfParametersWithoutGhostlayerPerSubcell:" << getNumberOfParametersWithoutGhostlayerPerSubcell();
      out << ",";
      out << "numberOfParametersWithGhostlayerPerSubcell:" << getNumberOfParametersWithGhostlayerPerSubcell();
      out << ",";
      out << "level:" << getLevel();
      out << ",";
      out << "isVirtual:" << getIsVirtual();
      out << ",";
      out << "isRemote:" << getIsRemote();
      out << ",";
      out << "isPaddingSubgrid:" << getIsPaddingSubgrid();
      out << ",";
      out << "numberOfTransfersToBeSkipped:[";
   for (int i = 0; i < THREE_POWER_D_MINUS_ONE-1; i++) {
      out << getNumberOfTransfersToBeSkipped(i) << ",";
   }
   out << getNumberOfTransfersToBeSkipped(THREE_POWER_D_MINUS_ONE-1) << "]";
      out << ",";
      out << "numberOfSharedAdjacentVertices:[";
   for (int i = 0; i < THREE_POWER_D_MINUS_ONE-1; i++) {
      out << getNumberOfSharedAdjacentVertices(i) << ",";
   }
   out << getNumberOfSharedAdjacentVertices(THREE_POWER_D_MINUS_ONE-1) << "]";
      out << ",";
      out << "currentStateWasSent:" << getCurrentStateWasSent();
      out << ",";
      out << "markStateAsSentInNextIteration:" << getMarkStateAsSentInNextIteration();
      out << ",";
      out << "adjacentRanksChanged:" << getAdjacentRanksChanged();
      out << ",";
      out << "adjacentRanks:[";
   for (int i = 0; i < THREE_POWER_D_MINUS_ONE-1; i++) {
      out << getAdjacentRanks(i) << ",";
   }
   out << getAdjacentRanks(THREE_POWER_D_MINUS_ONE-1) << "]";
      out << ",";
      out << "overlapByRemoteGhostlayer:[";
   for (int i = 0; i < THREE_POWER_D_MINUS_ONE-1; i++) {
      out << getOverlapByRemoteGhostlayer(i) << ",";
   }
   out << getOverlapByRemoteGhostlayer(THREE_POWER_D_MINUS_ONE-1) << "]";
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
      out << "demandedMeshWidth:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getDemandedMeshWidth(i) << ",";
   }
   out << getDemandedMeshWidth(DIMENSIONS-1) << "]";
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
      out << "uIndex:" << getUIndex();
      out <<  ")";
   }
   
   
   peanoclaw::records::CellDescription::PersistentRecords peanoclaw::records::CellDescription::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::CellDescriptionPacked peanoclaw::records::CellDescription::convert() const{
      return CellDescriptionPacked(
         getSubdivisionFactor(),
         getGhostlayerWidth(),
         getUnknownsPerSubcell(),
         getNumberOfParametersWithoutGhostlayerPerSubcell(),
         getNumberOfParametersWithGhostlayerPerSubcell(),
         getLevel(),
         getIsVirtual(),
         getIsRemote(),
         getIsPaddingSubgrid(),
         getNumberOfTransfersToBeSkipped(),
         getNumberOfSharedAdjacentVertices(),
         getCurrentStateWasSent(),
         getMarkStateAsSentInNextIteration(),
         getAdjacentRanksChanged(),
         getAdjacentRanks(),
         getOverlapByRemoteGhostlayer(),
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
         getUIndex()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::CellDescription::_log( "peanoclaw::records::CellDescription" );
      
      MPI_Datatype peanoclaw::records::CellDescription::Datatype = 0;
      MPI_Datatype peanoclaw::records::CellDescription::FullDatatype = 0;
      
      
      void peanoclaw::records::CellDescription::initDatatype() {
         {
            CellDescription dummyCellDescription[2];
            
            const int Attributes = 33;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //subdivisionFactor
               MPI_INT,		 //ghostlayerWidth
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //numberOfParametersWithoutGhostlayerPerSubcell
               MPI_INT,		 //numberOfParametersWithGhostlayerPerSubcell
               MPI_INT,		 //level
               MPI_CHAR,		 //isVirtual
               MPI_CHAR,		 //isRemote
               MPI_CHAR,		 //isPaddingSubgrid
               MPI_INT,		 //numberOfTransfersToBeSkipped
               MPI_INT,		 //adjacentRanks
               MPI_INT,		 //overlapByRemoteGhostlayer
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
               MPI_INT,		 //uIndex
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               DIMENSIONS,		 //subdivisionFactor
               1,		 //ghostlayerWidth
               1,		 //unknownsPerSubcell
               1,		 //numberOfParametersWithoutGhostlayerPerSubcell
               1,		 //numberOfParametersWithGhostlayerPerSubcell
               1,		 //level
               1,		 //isVirtual
               1,		 //isRemote
               1,		 //isPaddingSubgrid
               THREE_POWER_D_MINUS_ONE,		 //numberOfTransfersToBeSkipped
               THREE_POWER_D_MINUS_ONE,		 //adjacentRanks
               THREE_POWER_D_MINUS_ONE,		 //overlapByRemoteGhostlayer
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
               DIMENSIONS,		 //demandedMeshWidth
               DIMENSIONS,		 //restrictionLowerBounds
               DIMENSIONS,		 //restrictionUpperBounds
               1,		 //cellDescriptionIndex
               1,		 //uIndex
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._ghostlayerWidth))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._unknownsPerSubcell))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._numberOfParametersWithoutGhostlayerPerSubcell))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._numberOfParametersWithGhostlayerPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._level))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._isVirtual))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._isRemote))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._isPaddingSubgrid))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._numberOfTransfersToBeSkipped[0]))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._adjacentRanks[0]))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._overlapByRemoteGhostlayer[0]))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._position[0]))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._size[0]))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._time))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._timestepSize))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._maximumFineGridTime))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimumFineGridTimestep))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._synchronizeFineGrids))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._willCoarsen))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalNeighborTimeConstraint))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalLeafNeighborTimeConstraint))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalNeighborTime))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._maximalNeighborTimestep))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._estimatedNextTimestepSize))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._skipGridIterations))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._ageInGridIterations))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._demandedMeshWidth[0]))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._restrictionLowerBounds[0]))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._restrictionUpperBounds[0]))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._cellDescriptionIndex))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._uIndex))), 		&disp[31] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyCellDescription[1]._persistentRecords._subdivisionFactor[0])), 		&disp[32] );
            
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
            
            const int Attributes = 38;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //subdivisionFactor
               MPI_INT,		 //ghostlayerWidth
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //numberOfParametersWithoutGhostlayerPerSubcell
               MPI_INT,		 //numberOfParametersWithGhostlayerPerSubcell
               MPI_INT,		 //level
               MPI_CHAR,		 //isVirtual
               MPI_CHAR,		 //isRemote
               MPI_CHAR,		 //isPaddingSubgrid
               MPI_INT,		 //numberOfTransfersToBeSkipped
               MPI_INT,		 //numberOfSharedAdjacentVertices
               MPI_CHAR,		 //currentStateWasSent
               MPI_CHAR,		 //markStateAsSentInNextIteration
               MPI_CHAR,		 //adjacentRanksChanged
               MPI_INT,		 //adjacentRanks
               MPI_INT,		 //overlapByRemoteGhostlayer
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
               MPI_INT,		 //uIndex
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               DIMENSIONS,		 //subdivisionFactor
               1,		 //ghostlayerWidth
               1,		 //unknownsPerSubcell
               1,		 //numberOfParametersWithoutGhostlayerPerSubcell
               1,		 //numberOfParametersWithGhostlayerPerSubcell
               1,		 //level
               1,		 //isVirtual
               1,		 //isRemote
               1,		 //isPaddingSubgrid
               THREE_POWER_D_MINUS_ONE,		 //numberOfTransfersToBeSkipped
               THREE_POWER_D_MINUS_ONE,		 //numberOfSharedAdjacentVertices
               1,		 //currentStateWasSent
               1,		 //markStateAsSentInNextIteration
               1,		 //adjacentRanksChanged
               THREE_POWER_D_MINUS_ONE,		 //adjacentRanks
               THREE_POWER_D_MINUS_ONE,		 //overlapByRemoteGhostlayer
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
               DIMENSIONS,		 //demandedMeshWidth
               DIMENSIONS,		 //restrictionLowerBounds
               DIMENSIONS,		 //restrictionUpperBounds
               1,		 //cellDescriptionIndex
               1,		 //uIndex
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._ghostlayerWidth))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._unknownsPerSubcell))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._numberOfParametersWithoutGhostlayerPerSubcell))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._numberOfParametersWithGhostlayerPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._level))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._isVirtual))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._isRemote))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._isPaddingSubgrid))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._numberOfTransfersToBeSkipped[0]))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._numberOfSharedAdjacentVertices[0]))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._currentStateWasSent))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._markStateAsSentInNextIteration))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._adjacentRanksChanged))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._adjacentRanks[0]))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._overlapByRemoteGhostlayer[0]))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._position[0]))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._size[0]))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._time))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._timestepSize))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._maximumFineGridTime))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimumFineGridTimestep))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._synchronizeFineGrids))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._willCoarsen))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalNeighborTimeConstraint))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._constrainingNeighborIndex))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalLeafNeighborTimeConstraint))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalNeighborTime))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._maximalNeighborTimestep))), 		&disp[28] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._estimatedNextTimestepSize))), 		&disp[29] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._skipGridIterations))), 		&disp[30] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._ageInGridIterations))), 		&disp[31] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._demandedMeshWidth[0]))), 		&disp[32] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._restrictionLowerBounds[0]))), 		&disp[33] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._restrictionUpperBounds[0]))), 		&disp[34] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._cellDescriptionIndex))), 		&disp[35] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._uIndex))), 		&disp[36] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyCellDescription[1]._persistentRecords._subdivisionFactor[0])), 		&disp[37] );
            
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
      
      void peanoclaw::records::CellDescription::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
         if (communicateBlocking) {
         
            const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
            if  (result!=MPI_SUCCESS) {
               std::ostringstream msg;
               msg << "was not able to send message peanoclaw::records::CellDescription "
               << toString()
               << " to node " << destination
               << ": " << tarch::parallel::MPIReturnValueToString(result);
               _log.error( "send(int)",msg.str() );
            }
            
         }
         else {
         
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
         
      }
      
      
      
      void peanoclaw::records::CellDescription::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
         if (communicateBlocking) {
         
            MPI_Status  status;
            const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
            if ( result != MPI_SUCCESS ) {
               std::ostringstream msg;
               msg << "failed to start to receive peanoclaw::records::CellDescription from node "
               << source << ": " << tarch::parallel::MPIReturnValueToString(result);
               _log.error( "receive(int)", msg.str() );
            }
            
         }
         else {
         
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
      assertion((30 < (8 * sizeof(int))));
      
   }
   
   
   peanoclaw::records::CellDescriptionPacked::PersistentRecords::PersistentRecords(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostlayerWidth, const int& unknownsPerSubcell, const int& numberOfParametersWithoutGhostlayerPerSubcell, const int& numberOfParametersWithGhostlayerPerSubcell, const int& level, const bool& isVirtual, const bool& isRemote, const bool& isPaddingSubgrid, const tarch::la::Vector<THREE_POWER_D_MINUS_ONE,int>& numberOfTransfersToBeSkipped, const tarch::la::Vector<THREE_POWER_D_MINUS_ONE,int>& numberOfSharedAdjacentVertices, const bool& currentStateWasSent, const bool& markStateAsSentInNextIteration, const bool& adjacentRanksChanged, const tarch::la::Vector<THREE_POWER_D_MINUS_ONE,int>& adjacentRanks, const tarch::la::Vector<THREE_POWER_D_MINUS_ONE,int>& overlapByRemoteGhostlayer, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& maximumFineGridTime, const double& minimumFineGridTimestep, const bool& synchronizeFineGrids, const bool& willCoarsen, const double& minimalNeighborTimeConstraint, const int& constrainingNeighborIndex, const double& minimalLeafNeighborTimeConstraint, const double& minimalNeighborTime, const double& maximalNeighborTimestep, const double& estimatedNextTimestepSize, const int& skipGridIterations, const int& ageInGridIterations, const tarch::la::Vector<DIMENSIONS,double>& demandedMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds, const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds, const int& cellDescriptionIndex, const int& uIndex):
   _subdivisionFactor(subdivisionFactor),
   _numberOfTransfersToBeSkipped(numberOfTransfersToBeSkipped),
   _numberOfSharedAdjacentVertices(numberOfSharedAdjacentVertices),
   _adjacentRanks(adjacentRanks),
   _overlapByRemoteGhostlayer(overlapByRemoteGhostlayer),
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
   _uIndex(uIndex) {
      setGhostlayerWidth(ghostlayerWidth);
      setUnknownsPerSubcell(unknownsPerSubcell);
      setNumberOfParametersWithoutGhostlayerPerSubcell(numberOfParametersWithoutGhostlayerPerSubcell);
      setNumberOfParametersWithGhostlayerPerSubcell(numberOfParametersWithGhostlayerPerSubcell);
      setLevel(level);
      setIsVirtual(isVirtual);
      setIsRemote(isRemote);
      setIsPaddingSubgrid(isPaddingSubgrid);
      setCurrentStateWasSent(currentStateWasSent);
      setMarkStateAsSentInNextIteration(markStateAsSentInNextIteration);
      setAdjacentRanksChanged(adjacentRanksChanged);
      setSynchronizeFineGrids(synchronizeFineGrids);
      setWillCoarsen(willCoarsen);
      setSkipGridIterations(skipGridIterations);
      assertion((30 < (8 * sizeof(int))));
      
   }
   
   peanoclaw::records::CellDescriptionPacked::CellDescriptionPacked() {
      assertion((30 < (8 * sizeof(int))));
      
   }
   
   
   peanoclaw::records::CellDescriptionPacked::CellDescriptionPacked(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._subdivisionFactor, persistentRecords.getGhostlayerWidth(), persistentRecords.getUnknownsPerSubcell(), persistentRecords.getNumberOfParametersWithoutGhostlayerPerSubcell(), persistentRecords.getNumberOfParametersWithGhostlayerPerSubcell(), persistentRecords.getLevel(), persistentRecords.getIsVirtual(), persistentRecords.getIsRemote(), persistentRecords.getIsPaddingSubgrid(), persistentRecords._numberOfTransfersToBeSkipped, persistentRecords._numberOfSharedAdjacentVertices, persistentRecords.getCurrentStateWasSent(), persistentRecords.getMarkStateAsSentInNextIteration(), persistentRecords.getAdjacentRanksChanged(), persistentRecords._adjacentRanks, persistentRecords._overlapByRemoteGhostlayer, persistentRecords._position, persistentRecords._size, persistentRecords._time, persistentRecords._timestepSize, persistentRecords._maximumFineGridTime, persistentRecords._minimumFineGridTimestep, persistentRecords.getSynchronizeFineGrids(), persistentRecords.getWillCoarsen(), persistentRecords._minimalNeighborTimeConstraint, persistentRecords._constrainingNeighborIndex, persistentRecords._minimalLeafNeighborTimeConstraint, persistentRecords._minimalNeighborTime, persistentRecords._maximalNeighborTimestep, persistentRecords._estimatedNextTimestepSize, persistentRecords.getSkipGridIterations(), persistentRecords._ageInGridIterations, persistentRecords._demandedMeshWidth, persistentRecords._restrictionLowerBounds, persistentRecords._restrictionUpperBounds, persistentRecords._cellDescriptionIndex, persistentRecords._uIndex) {
      assertion((30 < (8 * sizeof(int))));
      
   }
   
   
   peanoclaw::records::CellDescriptionPacked::CellDescriptionPacked(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostlayerWidth, const int& unknownsPerSubcell, const int& numberOfParametersWithoutGhostlayerPerSubcell, const int& numberOfParametersWithGhostlayerPerSubcell, const int& level, const bool& isVirtual, const bool& isRemote, const bool& isPaddingSubgrid, const tarch::la::Vector<THREE_POWER_D_MINUS_ONE,int>& numberOfTransfersToBeSkipped, const tarch::la::Vector<THREE_POWER_D_MINUS_ONE,int>& numberOfSharedAdjacentVertices, const bool& currentStateWasSent, const bool& markStateAsSentInNextIteration, const bool& adjacentRanksChanged, const tarch::la::Vector<THREE_POWER_D_MINUS_ONE,int>& adjacentRanks, const tarch::la::Vector<THREE_POWER_D_MINUS_ONE,int>& overlapByRemoteGhostlayer, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& maximumFineGridTime, const double& minimumFineGridTimestep, const bool& synchronizeFineGrids, const bool& willCoarsen, const double& minimalNeighborTimeConstraint, const int& constrainingNeighborIndex, const double& minimalLeafNeighborTimeConstraint, const double& minimalNeighborTime, const double& maximalNeighborTimestep, const double& estimatedNextTimestepSize, const int& skipGridIterations, const int& ageInGridIterations, const tarch::la::Vector<DIMENSIONS,double>& demandedMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds, const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds, const int& cellDescriptionIndex, const int& uIndex):
   _persistentRecords(subdivisionFactor, ghostlayerWidth, unknownsPerSubcell, numberOfParametersWithoutGhostlayerPerSubcell, numberOfParametersWithGhostlayerPerSubcell, level, isVirtual, isRemote, isPaddingSubgrid, numberOfTransfersToBeSkipped, numberOfSharedAdjacentVertices, currentStateWasSent, markStateAsSentInNextIteration, adjacentRanksChanged, adjacentRanks, overlapByRemoteGhostlayer, position, size, time, timestepSize, maximumFineGridTime, minimumFineGridTimestep, synchronizeFineGrids, willCoarsen, minimalNeighborTimeConstraint, constrainingNeighborIndex, minimalLeafNeighborTimeConstraint, minimalNeighborTime, maximalNeighborTimestep, estimatedNextTimestepSize, skipGridIterations, ageInGridIterations, demandedMeshWidth, restrictionLowerBounds, restrictionUpperBounds, cellDescriptionIndex, uIndex) {
      assertion((30 < (8 * sizeof(int))));
      
   }
   
   
   peanoclaw::records::CellDescriptionPacked::~CellDescriptionPacked() { }
   
   
   
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
      out << "ghostlayerWidth:" << getGhostlayerWidth();
      out << ",";
      out << "unknownsPerSubcell:" << getUnknownsPerSubcell();
      out << ",";
      out << "numberOfParametersWithoutGhostlayerPerSubcell:" << getNumberOfParametersWithoutGhostlayerPerSubcell();
      out << ",";
      out << "numberOfParametersWithGhostlayerPerSubcell:" << getNumberOfParametersWithGhostlayerPerSubcell();
      out << ",";
      out << "level:" << getLevel();
      out << ",";
      out << "isVirtual:" << getIsVirtual();
      out << ",";
      out << "isRemote:" << getIsRemote();
      out << ",";
      out << "isPaddingSubgrid:" << getIsPaddingSubgrid();
      out << ",";
      out << "numberOfTransfersToBeSkipped:[";
   for (int i = 0; i < THREE_POWER_D_MINUS_ONE-1; i++) {
      out << getNumberOfTransfersToBeSkipped(i) << ",";
   }
   out << getNumberOfTransfersToBeSkipped(THREE_POWER_D_MINUS_ONE-1) << "]";
      out << ",";
      out << "numberOfSharedAdjacentVertices:[";
   for (int i = 0; i < THREE_POWER_D_MINUS_ONE-1; i++) {
      out << getNumberOfSharedAdjacentVertices(i) << ",";
   }
   out << getNumberOfSharedAdjacentVertices(THREE_POWER_D_MINUS_ONE-1) << "]";
      out << ",";
      out << "currentStateWasSent:" << getCurrentStateWasSent();
      out << ",";
      out << "markStateAsSentInNextIteration:" << getMarkStateAsSentInNextIteration();
      out << ",";
      out << "adjacentRanksChanged:" << getAdjacentRanksChanged();
      out << ",";
      out << "adjacentRanks:[";
   for (int i = 0; i < THREE_POWER_D_MINUS_ONE-1; i++) {
      out << getAdjacentRanks(i) << ",";
   }
   out << getAdjacentRanks(THREE_POWER_D_MINUS_ONE-1) << "]";
      out << ",";
      out << "overlapByRemoteGhostlayer:[";
   for (int i = 0; i < THREE_POWER_D_MINUS_ONE-1; i++) {
      out << getOverlapByRemoteGhostlayer(i) << ",";
   }
   out << getOverlapByRemoteGhostlayer(THREE_POWER_D_MINUS_ONE-1) << "]";
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
      out << "demandedMeshWidth:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getDemandedMeshWidth(i) << ",";
   }
   out << getDemandedMeshWidth(DIMENSIONS-1) << "]";
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
      out << "uIndex:" << getUIndex();
      out <<  ")";
   }
   
   
   peanoclaw::records::CellDescriptionPacked::PersistentRecords peanoclaw::records::CellDescriptionPacked::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::CellDescription peanoclaw::records::CellDescriptionPacked::convert() const{
      return CellDescription(
         getSubdivisionFactor(),
         getGhostlayerWidth(),
         getUnknownsPerSubcell(),
         getNumberOfParametersWithoutGhostlayerPerSubcell(),
         getNumberOfParametersWithGhostlayerPerSubcell(),
         getLevel(),
         getIsVirtual(),
         getIsRemote(),
         getIsPaddingSubgrid(),
         getNumberOfTransfersToBeSkipped(),
         getNumberOfSharedAdjacentVertices(),
         getCurrentStateWasSent(),
         getMarkStateAsSentInNextIteration(),
         getAdjacentRanksChanged(),
         getAdjacentRanks(),
         getOverlapByRemoteGhostlayer(),
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
         getUIndex()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::CellDescriptionPacked::_log( "peanoclaw::records::CellDescriptionPacked" );
      
      MPI_Datatype peanoclaw::records::CellDescriptionPacked::Datatype = 0;
      MPI_Datatype peanoclaw::records::CellDescriptionPacked::FullDatatype = 0;
      
      
      void peanoclaw::records::CellDescriptionPacked::initDatatype() {
         {
            CellDescriptionPacked dummyCellDescriptionPacked[2];
            
            const int Attributes = 23;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //subdivisionFactor
               MPI_INT,		 //numberOfTransfersToBeSkipped
               MPI_INT,		 //adjacentRanks
               MPI_INT,		 //overlapByRemoteGhostlayer
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
               MPI_INT,		 //uIndex
               MPI_INT,		 //_packedRecords0
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               DIMENSIONS,		 //subdivisionFactor
               THREE_POWER_D_MINUS_ONE,		 //numberOfTransfersToBeSkipped
               THREE_POWER_D_MINUS_ONE,		 //adjacentRanks
               THREE_POWER_D_MINUS_ONE,		 //overlapByRemoteGhostlayer
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
               DIMENSIONS,		 //demandedMeshWidth
               DIMENSIONS,		 //restrictionLowerBounds
               DIMENSIONS,		 //restrictionUpperBounds
               1,		 //cellDescriptionIndex
               1,		 //uIndex
               1,		 //_packedRecords0
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._numberOfTransfersToBeSkipped[0]))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._adjacentRanks[0]))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._overlapByRemoteGhostlayer[0]))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._position[0]))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._size[0]))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._time))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._timestepSize))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._maximumFineGridTime))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimumFineGridTimestep))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimalNeighborTimeConstraint))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimalLeafNeighborTimeConstraint))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._minimalNeighborTime))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._maximalNeighborTimestep))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._estimatedNextTimestepSize))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._ageInGridIterations))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._demandedMeshWidth[0]))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._restrictionLowerBounds[0]))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._restrictionUpperBounds[0]))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._uIndex))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._packedRecords0))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyCellDescriptionPacked[1]._persistentRecords._subdivisionFactor[0])), 		&disp[22] );
            
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
               MPI_INT,		 //numberOfTransfersToBeSkipped
               MPI_INT,		 //numberOfSharedAdjacentVertices
               MPI_INT,		 //adjacentRanks
               MPI_INT,		 //overlapByRemoteGhostlayer
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
               MPI_INT,		 //uIndex
               MPI_INT,		 //_packedRecords0
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               DIMENSIONS,		 //subdivisionFactor
               THREE_POWER_D_MINUS_ONE,		 //numberOfTransfersToBeSkipped
               THREE_POWER_D_MINUS_ONE,		 //numberOfSharedAdjacentVertices
               THREE_POWER_D_MINUS_ONE,		 //adjacentRanks
               THREE_POWER_D_MINUS_ONE,		 //overlapByRemoteGhostlayer
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
               DIMENSIONS,		 //demandedMeshWidth
               DIMENSIONS,		 //restrictionLowerBounds
               DIMENSIONS,		 //restrictionUpperBounds
               1,		 //cellDescriptionIndex
               1,		 //uIndex
               1,		 //_packedRecords0
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._numberOfTransfersToBeSkipped[0]))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._numberOfSharedAdjacentVertices[0]))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._adjacentRanks[0]))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._overlapByRemoteGhostlayer[0]))), 		&disp[4] );
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
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._demandedMeshWidth[0]))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._restrictionLowerBounds[0]))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._restrictionUpperBounds[0]))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._uIndex))), 		&disp[22] );
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
      
      void peanoclaw::records::CellDescriptionPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
         if (communicateBlocking) {
         
            const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
            if  (result!=MPI_SUCCESS) {
               std::ostringstream msg;
               msg << "was not able to send message peanoclaw::records::CellDescriptionPacked "
               << toString()
               << " to node " << destination
               << ": " << tarch::parallel::MPIReturnValueToString(result);
               _log.error( "send(int)",msg.str() );
            }
            
         }
         else {
         
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
         
      }
      
      
      
      void peanoclaw::records::CellDescriptionPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
         if (communicateBlocking) {
         
            MPI_Status  status;
            const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
            if ( result != MPI_SUCCESS ) {
               std::ostringstream msg;
               msg << "failed to start to receive peanoclaw::records::CellDescriptionPacked from node "
               << source << ": " << tarch::parallel::MPIReturnValueToString(result);
               _log.error( "receive(int)", msg.str() );
            }
            
         }
         else {
         
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
   
   
   peanoclaw::records::CellDescription::PersistentRecords::PersistentRecords(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostlayerWidth, const int& unknownsPerSubcell, const int& numberOfParametersWithoutGhostlayerPerSubcell, const int& numberOfParametersWithGhostlayerPerSubcell, const int& level, const bool& isVirtual, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& maximumFineGridTime, const double& minimumFineGridTimestep, const bool& synchronizeFineGrids, const bool& willCoarsen, const double& minimalNeighborTimeConstraint, const int& constrainingNeighborIndex, const double& minimalLeafNeighborTimeConstraint, const double& minimalNeighborTime, const double& maximalNeighborTimestep, const double& estimatedNextTimestepSize, const int& skipGridIterations, const int& ageInGridIterations, const tarch::la::Vector<DIMENSIONS,double>& demandedMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds, const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds, const int& cellDescriptionIndex, const int& uIndex):
   _subdivisionFactor(subdivisionFactor),
   _ghostlayerWidth(ghostlayerWidth),
   _unknownsPerSubcell(unknownsPerSubcell),
   _numberOfParametersWithoutGhostlayerPerSubcell(numberOfParametersWithoutGhostlayerPerSubcell),
   _numberOfParametersWithGhostlayerPerSubcell(numberOfParametersWithGhostlayerPerSubcell),
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
   _uIndex(uIndex) {
      
   }
   
   peanoclaw::records::CellDescription::CellDescription() {
      
   }
   
   
   peanoclaw::records::CellDescription::CellDescription(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._subdivisionFactor, persistentRecords._ghostlayerWidth, persistentRecords._unknownsPerSubcell, persistentRecords._numberOfParametersWithoutGhostlayerPerSubcell, persistentRecords._numberOfParametersWithGhostlayerPerSubcell, persistentRecords._level, persistentRecords._isVirtual, persistentRecords._position, persistentRecords._size, persistentRecords._time, persistentRecords._timestepSize, persistentRecords._maximumFineGridTime, persistentRecords._minimumFineGridTimestep, persistentRecords._synchronizeFineGrids, persistentRecords._willCoarsen, persistentRecords._minimalNeighborTimeConstraint, persistentRecords._constrainingNeighborIndex, persistentRecords._minimalLeafNeighborTimeConstraint, persistentRecords._minimalNeighborTime, persistentRecords._maximalNeighborTimestep, persistentRecords._estimatedNextTimestepSize, persistentRecords._skipGridIterations, persistentRecords._ageInGridIterations, persistentRecords._demandedMeshWidth, persistentRecords._restrictionLowerBounds, persistentRecords._restrictionUpperBounds, persistentRecords._cellDescriptionIndex, persistentRecords._uIndex) {
      
   }
   
   
   peanoclaw::records::CellDescription::CellDescription(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostlayerWidth, const int& unknownsPerSubcell, const int& numberOfParametersWithoutGhostlayerPerSubcell, const int& numberOfParametersWithGhostlayerPerSubcell, const int& level, const bool& isVirtual, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& maximumFineGridTime, const double& minimumFineGridTimestep, const bool& synchronizeFineGrids, const bool& willCoarsen, const double& minimalNeighborTimeConstraint, const int& constrainingNeighborIndex, const double& minimalLeafNeighborTimeConstraint, const double& minimalNeighborTime, const double& maximalNeighborTimestep, const double& estimatedNextTimestepSize, const int& skipGridIterations, const int& ageInGridIterations, const tarch::la::Vector<DIMENSIONS,double>& demandedMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds, const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds, const int& cellDescriptionIndex, const int& uIndex):
   _persistentRecords(subdivisionFactor, ghostlayerWidth, unknownsPerSubcell, numberOfParametersWithoutGhostlayerPerSubcell, numberOfParametersWithGhostlayerPerSubcell, level, isVirtual, position, size, time, timestepSize, maximumFineGridTime, minimumFineGridTimestep, synchronizeFineGrids, willCoarsen, minimalNeighborTimeConstraint, constrainingNeighborIndex, minimalLeafNeighborTimeConstraint, minimalNeighborTime, maximalNeighborTimestep, estimatedNextTimestepSize, skipGridIterations, ageInGridIterations, demandedMeshWidth, restrictionLowerBounds, restrictionUpperBounds, cellDescriptionIndex, uIndex) {
      
   }
   
   
   peanoclaw::records::CellDescription::~CellDescription() { }
   
   
   
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
      out << "ghostlayerWidth:" << getGhostlayerWidth();
      out << ",";
      out << "unknownsPerSubcell:" << getUnknownsPerSubcell();
      out << ",";
      out << "numberOfParametersWithoutGhostlayerPerSubcell:" << getNumberOfParametersWithoutGhostlayerPerSubcell();
      out << ",";
      out << "numberOfParametersWithGhostlayerPerSubcell:" << getNumberOfParametersWithGhostlayerPerSubcell();
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
      out << "demandedMeshWidth:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getDemandedMeshWidth(i) << ",";
   }
   out << getDemandedMeshWidth(DIMENSIONS-1) << "]";
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
      out << "uIndex:" << getUIndex();
      out <<  ")";
   }
   
   
   peanoclaw::records::CellDescription::PersistentRecords peanoclaw::records::CellDescription::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::CellDescriptionPacked peanoclaw::records::CellDescription::convert() const{
      return CellDescriptionPacked(
         getSubdivisionFactor(),
         getGhostlayerWidth(),
         getUnknownsPerSubcell(),
         getNumberOfParametersWithoutGhostlayerPerSubcell(),
         getNumberOfParametersWithGhostlayerPerSubcell(),
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
         getUIndex()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::CellDescription::_log( "peanoclaw::records::CellDescription" );
      
      MPI_Datatype peanoclaw::records::CellDescription::Datatype = 0;
      MPI_Datatype peanoclaw::records::CellDescription::FullDatatype = 0;
      
      
      void peanoclaw::records::CellDescription::initDatatype() {
         {
            CellDescription dummyCellDescription[2];
            
            const int Attributes = 28;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //subdivisionFactor
               MPI_INT,		 //ghostlayerWidth
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //numberOfParametersWithoutGhostlayerPerSubcell
               MPI_INT,		 //numberOfParametersWithGhostlayerPerSubcell
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
               MPI_INT,		 //uIndex
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               DIMENSIONS,		 //subdivisionFactor
               1,		 //ghostlayerWidth
               1,		 //unknownsPerSubcell
               1,		 //numberOfParametersWithoutGhostlayerPerSubcell
               1,		 //numberOfParametersWithGhostlayerPerSubcell
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
               DIMENSIONS,		 //demandedMeshWidth
               DIMENSIONS,		 //restrictionLowerBounds
               DIMENSIONS,		 //restrictionUpperBounds
               1,		 //cellDescriptionIndex
               1,		 //uIndex
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._ghostlayerWidth))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._unknownsPerSubcell))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._numberOfParametersWithoutGhostlayerPerSubcell))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._numberOfParametersWithGhostlayerPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._level))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._isVirtual))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._position[0]))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._size[0]))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._time))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._timestepSize))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._maximumFineGridTime))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimumFineGridTimestep))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._synchronizeFineGrids))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._willCoarsen))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalNeighborTimeConstraint))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalLeafNeighborTimeConstraint))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalNeighborTime))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._maximalNeighborTimestep))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._estimatedNextTimestepSize))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._skipGridIterations))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._ageInGridIterations))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._demandedMeshWidth[0]))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._restrictionLowerBounds[0]))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._restrictionUpperBounds[0]))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._cellDescriptionIndex))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._uIndex))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyCellDescription[1]._persistentRecords._subdivisionFactor[0])), 		&disp[27] );
            
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
            
            const int Attributes = 29;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //subdivisionFactor
               MPI_INT,		 //ghostlayerWidth
               MPI_INT,		 //unknownsPerSubcell
               MPI_INT,		 //numberOfParametersWithoutGhostlayerPerSubcell
               MPI_INT,		 //numberOfParametersWithGhostlayerPerSubcell
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
               MPI_INT,		 //uIndex
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               DIMENSIONS,		 //subdivisionFactor
               1,		 //ghostlayerWidth
               1,		 //unknownsPerSubcell
               1,		 //numberOfParametersWithoutGhostlayerPerSubcell
               1,		 //numberOfParametersWithGhostlayerPerSubcell
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
               DIMENSIONS,		 //demandedMeshWidth
               DIMENSIONS,		 //restrictionLowerBounds
               DIMENSIONS,		 //restrictionUpperBounds
               1,		 //cellDescriptionIndex
               1,		 //uIndex
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._ghostlayerWidth))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._unknownsPerSubcell))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._numberOfParametersWithoutGhostlayerPerSubcell))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._numberOfParametersWithGhostlayerPerSubcell))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._level))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._isVirtual))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._position[0]))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._size[0]))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._time))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._timestepSize))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._maximumFineGridTime))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimumFineGridTimestep))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._synchronizeFineGrids))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._willCoarsen))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalNeighborTimeConstraint))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._constrainingNeighborIndex))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalLeafNeighborTimeConstraint))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._minimalNeighborTime))), 		&disp[18] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._maximalNeighborTimestep))), 		&disp[19] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._estimatedNextTimestepSize))), 		&disp[20] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._skipGridIterations))), 		&disp[21] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._ageInGridIterations))), 		&disp[22] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._demandedMeshWidth[0]))), 		&disp[23] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._restrictionLowerBounds[0]))), 		&disp[24] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._restrictionUpperBounds[0]))), 		&disp[25] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._cellDescriptionIndex))), 		&disp[26] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescription[0]._persistentRecords._uIndex))), 		&disp[27] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyCellDescription[1]._persistentRecords._subdivisionFactor[0])), 		&disp[28] );
            
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
      
      void peanoclaw::records::CellDescription::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
         if (communicateBlocking) {
         
            const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
            if  (result!=MPI_SUCCESS) {
               std::ostringstream msg;
               msg << "was not able to send message peanoclaw::records::CellDescription "
               << toString()
               << " to node " << destination
               << ": " << tarch::parallel::MPIReturnValueToString(result);
               _log.error( "send(int)",msg.str() );
            }
            
         }
         else {
         
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
         
      }
      
      
      
      void peanoclaw::records::CellDescription::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
         if (communicateBlocking) {
         
            MPI_Status  status;
            const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
            if ( result != MPI_SUCCESS ) {
               std::ostringstream msg;
               msg << "failed to start to receive peanoclaw::records::CellDescription from node "
               << source << ": " << tarch::parallel::MPIReturnValueToString(result);
               _log.error( "receive(int)", msg.str() );
            }
            
         }
         else {
         
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
      assertion((25 < (8 * sizeof(int))));
      
   }
   
   
   peanoclaw::records::CellDescriptionPacked::PersistentRecords::PersistentRecords(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostlayerWidth, const int& unknownsPerSubcell, const int& numberOfParametersWithoutGhostlayerPerSubcell, const int& numberOfParametersWithGhostlayerPerSubcell, const int& level, const bool& isVirtual, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& maximumFineGridTime, const double& minimumFineGridTimestep, const bool& synchronizeFineGrids, const bool& willCoarsen, const double& minimalNeighborTimeConstraint, const int& constrainingNeighborIndex, const double& minimalLeafNeighborTimeConstraint, const double& minimalNeighborTime, const double& maximalNeighborTimestep, const double& estimatedNextTimestepSize, const int& skipGridIterations, const int& ageInGridIterations, const tarch::la::Vector<DIMENSIONS,double>& demandedMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds, const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds, const int& cellDescriptionIndex, const int& uIndex):
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
   _uIndex(uIndex) {
      setGhostlayerWidth(ghostlayerWidth);
      setUnknownsPerSubcell(unknownsPerSubcell);
      setNumberOfParametersWithoutGhostlayerPerSubcell(numberOfParametersWithoutGhostlayerPerSubcell);
      setNumberOfParametersWithGhostlayerPerSubcell(numberOfParametersWithGhostlayerPerSubcell);
      setLevel(level);
      setIsVirtual(isVirtual);
      setSynchronizeFineGrids(synchronizeFineGrids);
      setWillCoarsen(willCoarsen);
      setSkipGridIterations(skipGridIterations);
      assertion((25 < (8 * sizeof(int))));
      
   }
   
   peanoclaw::records::CellDescriptionPacked::CellDescriptionPacked() {
      assertion((25 < (8 * sizeof(int))));
      
   }
   
   
   peanoclaw::records::CellDescriptionPacked::CellDescriptionPacked(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._subdivisionFactor, persistentRecords.getGhostlayerWidth(), persistentRecords.getUnknownsPerSubcell(), persistentRecords.getNumberOfParametersWithoutGhostlayerPerSubcell(), persistentRecords.getNumberOfParametersWithGhostlayerPerSubcell(), persistentRecords.getLevel(), persistentRecords.getIsVirtual(), persistentRecords._position, persistentRecords._size, persistentRecords._time, persistentRecords._timestepSize, persistentRecords._maximumFineGridTime, persistentRecords._minimumFineGridTimestep, persistentRecords.getSynchronizeFineGrids(), persistentRecords.getWillCoarsen(), persistentRecords._minimalNeighborTimeConstraint, persistentRecords._constrainingNeighborIndex, persistentRecords._minimalLeafNeighborTimeConstraint, persistentRecords._minimalNeighborTime, persistentRecords._maximalNeighborTimestep, persistentRecords._estimatedNextTimestepSize, persistentRecords.getSkipGridIterations(), persistentRecords._ageInGridIterations, persistentRecords._demandedMeshWidth, persistentRecords._restrictionLowerBounds, persistentRecords._restrictionUpperBounds, persistentRecords._cellDescriptionIndex, persistentRecords._uIndex) {
      assertion((25 < (8 * sizeof(int))));
      
   }
   
   
   peanoclaw::records::CellDescriptionPacked::CellDescriptionPacked(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostlayerWidth, const int& unknownsPerSubcell, const int& numberOfParametersWithoutGhostlayerPerSubcell, const int& numberOfParametersWithGhostlayerPerSubcell, const int& level, const bool& isVirtual, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& maximumFineGridTime, const double& minimumFineGridTimestep, const bool& synchronizeFineGrids, const bool& willCoarsen, const double& minimalNeighborTimeConstraint, const int& constrainingNeighborIndex, const double& minimalLeafNeighborTimeConstraint, const double& minimalNeighborTime, const double& maximalNeighborTimestep, const double& estimatedNextTimestepSize, const int& skipGridIterations, const int& ageInGridIterations, const tarch::la::Vector<DIMENSIONS,double>& demandedMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& restrictionLowerBounds, const tarch::la::Vector<DIMENSIONS,double>& restrictionUpperBounds, const int& cellDescriptionIndex, const int& uIndex):
   _persistentRecords(subdivisionFactor, ghostlayerWidth, unknownsPerSubcell, numberOfParametersWithoutGhostlayerPerSubcell, numberOfParametersWithGhostlayerPerSubcell, level, isVirtual, position, size, time, timestepSize, maximumFineGridTime, minimumFineGridTimestep, synchronizeFineGrids, willCoarsen, minimalNeighborTimeConstraint, constrainingNeighborIndex, minimalLeafNeighborTimeConstraint, minimalNeighborTime, maximalNeighborTimestep, estimatedNextTimestepSize, skipGridIterations, ageInGridIterations, demandedMeshWidth, restrictionLowerBounds, restrictionUpperBounds, cellDescriptionIndex, uIndex) {
      assertion((25 < (8 * sizeof(int))));
      
   }
   
   
   peanoclaw::records::CellDescriptionPacked::~CellDescriptionPacked() { }
   
   
   
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
      out << "ghostlayerWidth:" << getGhostlayerWidth();
      out << ",";
      out << "unknownsPerSubcell:" << getUnknownsPerSubcell();
      out << ",";
      out << "numberOfParametersWithoutGhostlayerPerSubcell:" << getNumberOfParametersWithoutGhostlayerPerSubcell();
      out << ",";
      out << "numberOfParametersWithGhostlayerPerSubcell:" << getNumberOfParametersWithGhostlayerPerSubcell();
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
      out << "demandedMeshWidth:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getDemandedMeshWidth(i) << ",";
   }
   out << getDemandedMeshWidth(DIMENSIONS-1) << "]";
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
      out << "uIndex:" << getUIndex();
      out <<  ")";
   }
   
   
   peanoclaw::records::CellDescriptionPacked::PersistentRecords peanoclaw::records::CellDescriptionPacked::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::CellDescription peanoclaw::records::CellDescriptionPacked::convert() const{
      return CellDescription(
         getSubdivisionFactor(),
         getGhostlayerWidth(),
         getUnknownsPerSubcell(),
         getNumberOfParametersWithoutGhostlayerPerSubcell(),
         getNumberOfParametersWithGhostlayerPerSubcell(),
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
         getUIndex()
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
               MPI_INT,		 //uIndex
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
               DIMENSIONS,		 //demandedMeshWidth
               DIMENSIONS,		 //restrictionLowerBounds
               DIMENSIONS,		 //restrictionUpperBounds
               1,		 //cellDescriptionIndex
               1,		 //uIndex
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
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._demandedMeshWidth[0]))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._restrictionLowerBounds[0]))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._restrictionUpperBounds[0]))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._uIndex))), 		&disp[17] );
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
               MPI_INT,		 //uIndex
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
               DIMENSIONS,		 //demandedMeshWidth
               DIMENSIONS,		 //restrictionLowerBounds
               DIMENSIONS,		 //restrictionUpperBounds
               1,		 //cellDescriptionIndex
               1,		 //uIndex
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
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._demandedMeshWidth[0]))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._restrictionLowerBounds[0]))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._restrictionUpperBounds[0]))), 		&disp[16] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[17] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellDescriptionPacked[0]._persistentRecords._uIndex))), 		&disp[18] );
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
      
      void peanoclaw::records::CellDescriptionPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
         if (communicateBlocking) {
         
            const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
            if  (result!=MPI_SUCCESS) {
               std::ostringstream msg;
               msg << "was not able to send message peanoclaw::records::CellDescriptionPacked "
               << toString()
               << " to node " << destination
               << ": " << tarch::parallel::MPIReturnValueToString(result);
               _log.error( "send(int)",msg.str() );
            }
            
         }
         else {
         
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
         
      }
      
      
      
      void peanoclaw::records::CellDescriptionPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
         if (communicateBlocking) {
         
            MPI_Status  status;
            const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
            if ( result != MPI_SUCCESS ) {
               std::ostringstream msg;
               msg << "failed to start to receive peanoclaw::records::CellDescriptionPacked from node "
               << source << ": " << tarch::parallel::MPIReturnValueToString(result);
               _log.error( "receive(int)", msg.str() );
            }
            
         }
         else {
         
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


