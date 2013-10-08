#include "peanoclaw/records/PatchDescription.h"

#if defined(Parallel)
   peanoclaw::records::PatchDescription::PersistentRecords::PersistentRecords() {
      
   }
   
   
   peanoclaw::records::PatchDescription::PersistentRecords::PersistentRecords(const bool& isReferenced, const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks, const int& rank, const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& level, const bool& isVirtual, const bool& isRemote, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& hash, const int& skipGridIterations, const double& demandedMeshWidth, const int& cellDescriptionIndex):
   _isReferenced(isReferenced),
   _adjacentRanks(adjacentRanks),
   _rank(rank),
   _subdivisionFactor(subdivisionFactor),
   _ghostLayerWidth(ghostLayerWidth),
   _level(level),
   _isVirtual(isVirtual),
   _isRemote(isRemote),
   _position(position),
   _size(size),
   _time(time),
   _timestepSize(timestepSize),
   _hash(hash),
   _skipGridIterations(skipGridIterations),
   _demandedMeshWidth(demandedMeshWidth),
   _cellDescriptionIndex(cellDescriptionIndex) {
      
   }
   
   
    bool peanoclaw::records::PatchDescription::PersistentRecords::getIsReferenced() const  {
      return _isReferenced;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setIsReferenced(const bool& isReferenced)  {
      _isReferenced = isReferenced;
   }
   
   
   
    tarch::la::Vector<THREE_POWER_D,int> peanoclaw::records::PatchDescription::PersistentRecords::getAdjacentRanks() const  {
      return _adjacentRanks;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setAdjacentRanks(const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks)  {
      _adjacentRanks = (adjacentRanks);
   }
   
   
   
    int peanoclaw::records::PatchDescription::PersistentRecords::getRank() const  {
      return _rank;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setRank(const int& rank)  {
      _rank = rank;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::PatchDescription::PersistentRecords::getSubdivisionFactor() const  {
      return _subdivisionFactor;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setSubdivisionFactor(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor)  {
      _subdivisionFactor = (subdivisionFactor);
   }
   
   
   
    int peanoclaw::records::PatchDescription::PersistentRecords::getGhostLayerWidth() const  {
      return _ghostLayerWidth;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setGhostLayerWidth(const int& ghostLayerWidth)  {
      _ghostLayerWidth = ghostLayerWidth;
   }
   
   
   
    int peanoclaw::records::PatchDescription::PersistentRecords::getLevel() const  {
      return _level;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setLevel(const int& level)  {
      _level = level;
   }
   
   
   
    bool peanoclaw::records::PatchDescription::PersistentRecords::getIsVirtual() const  {
      return _isVirtual;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setIsVirtual(const bool& isVirtual)  {
      _isVirtual = isVirtual;
   }
   
   
   
    bool peanoclaw::records::PatchDescription::PersistentRecords::getIsRemote() const  {
      return _isRemote;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setIsRemote(const bool& isRemote)  {
      _isRemote = isRemote;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::PatchDescription::PersistentRecords::getPosition() const  {
      return _position;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setPosition(const tarch::la::Vector<DIMENSIONS,double>& position)  {
      _position = (position);
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::PatchDescription::PersistentRecords::getSize() const  {
      return _size;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setSize(const tarch::la::Vector<DIMENSIONS,double>& size)  {
      _size = (size);
   }
   
   
   
    double peanoclaw::records::PatchDescription::PersistentRecords::getTime() const  {
      return _time;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setTime(const double& time)  {
      _time = time;
   }
   
   
   
    double peanoclaw::records::PatchDescription::PersistentRecords::getTimestepSize() const  {
      return _timestepSize;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setTimestepSize(const double& timestepSize)  {
      _timestepSize = timestepSize;
   }
   
   
   
    double peanoclaw::records::PatchDescription::PersistentRecords::getHash() const  {
      return _hash;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setHash(const double& hash)  {
      _hash = hash;
   }
   
   
   
    int peanoclaw::records::PatchDescription::PersistentRecords::getSkipGridIterations() const  {
      return _skipGridIterations;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setSkipGridIterations(const int& skipGridIterations)  {
      _skipGridIterations = skipGridIterations;
   }
   
   
   
    double peanoclaw::records::PatchDescription::PersistentRecords::getDemandedMeshWidth() const  {
      return _demandedMeshWidth;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setDemandedMeshWidth(const double& demandedMeshWidth)  {
      _demandedMeshWidth = demandedMeshWidth;
   }
   
   
   
    int peanoclaw::records::PatchDescription::PersistentRecords::getCellDescriptionIndex() const  {
      return _cellDescriptionIndex;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setCellDescriptionIndex(const int& cellDescriptionIndex)  {
      _cellDescriptionIndex = cellDescriptionIndex;
   }
   
   
   peanoclaw::records::PatchDescription::PatchDescription() {
      
   }
   
   
   peanoclaw::records::PatchDescription::PatchDescription(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._isReferenced, persistentRecords._adjacentRanks, persistentRecords._rank, persistentRecords._subdivisionFactor, persistentRecords._ghostLayerWidth, persistentRecords._level, persistentRecords._isVirtual, persistentRecords._isRemote, persistentRecords._position, persistentRecords._size, persistentRecords._time, persistentRecords._timestepSize, persistentRecords._hash, persistentRecords._skipGridIterations, persistentRecords._demandedMeshWidth, persistentRecords._cellDescriptionIndex) {
      
   }
   
   
   peanoclaw::records::PatchDescription::PatchDescription(const bool& isReferenced, const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks, const int& rank, const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& level, const bool& isVirtual, const bool& isRemote, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& hash, const int& skipGridIterations, const double& demandedMeshWidth, const int& cellDescriptionIndex):
   _persistentRecords(isReferenced, adjacentRanks, rank, subdivisionFactor, ghostLayerWidth, level, isVirtual, isRemote, position, size, time, timestepSize, hash, skipGridIterations, demandedMeshWidth, cellDescriptionIndex) {
      
   }
   
   
   peanoclaw::records::PatchDescription::~PatchDescription() { }
   
   
    bool peanoclaw::records::PatchDescription::getIsReferenced() const  {
      return _persistentRecords._isReferenced;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setIsReferenced(const bool& isReferenced)  {
      _persistentRecords._isReferenced = isReferenced;
   }
   
   
   
    tarch::la::Vector<THREE_POWER_D,int> peanoclaw::records::PatchDescription::getAdjacentRanks() const  {
      return _persistentRecords._adjacentRanks;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setAdjacentRanks(const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks)  {
      _persistentRecords._adjacentRanks = (adjacentRanks);
   }
   
   
   
    int peanoclaw::records::PatchDescription::getAdjacentRanks(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<THREE_POWER_D);
      return _persistentRecords._adjacentRanks[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::PatchDescription::setAdjacentRanks(int elementIndex, const int& adjacentRanks)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<THREE_POWER_D);
      _persistentRecords._adjacentRanks[elementIndex]= adjacentRanks;
      
   }
   
   
   
    int peanoclaw::records::PatchDescription::getRank() const  {
      return _persistentRecords._rank;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setRank(const int& rank)  {
      _persistentRecords._rank = rank;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::PatchDescription::getSubdivisionFactor() const  {
      return _persistentRecords._subdivisionFactor;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setSubdivisionFactor(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor)  {
      _persistentRecords._subdivisionFactor = (subdivisionFactor);
   }
   
   
   
    int peanoclaw::records::PatchDescription::getSubdivisionFactor(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._subdivisionFactor[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::PatchDescription::setSubdivisionFactor(int elementIndex, const int& subdivisionFactor)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._subdivisionFactor[elementIndex]= subdivisionFactor;
      
   }
   
   
   
    int peanoclaw::records::PatchDescription::getGhostLayerWidth() const  {
      return _persistentRecords._ghostLayerWidth;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setGhostLayerWidth(const int& ghostLayerWidth)  {
      _persistentRecords._ghostLayerWidth = ghostLayerWidth;
   }
   
   
   
    int peanoclaw::records::PatchDescription::getLevel() const  {
      return _persistentRecords._level;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setLevel(const int& level)  {
      _persistentRecords._level = level;
   }
   
   
   
    bool peanoclaw::records::PatchDescription::getIsVirtual() const  {
      return _persistentRecords._isVirtual;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setIsVirtual(const bool& isVirtual)  {
      _persistentRecords._isVirtual = isVirtual;
   }
   
   
   
    bool peanoclaw::records::PatchDescription::getIsRemote() const  {
      return _persistentRecords._isRemote;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setIsRemote(const bool& isRemote)  {
      _persistentRecords._isRemote = isRemote;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::PatchDescription::getPosition() const  {
      return _persistentRecords._position;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setPosition(const tarch::la::Vector<DIMENSIONS,double>& position)  {
      _persistentRecords._position = (position);
   }
   
   
   
    double peanoclaw::records::PatchDescription::getPosition(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._position[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::PatchDescription::setPosition(int elementIndex, const double& position)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._position[elementIndex]= position;
      
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::PatchDescription::getSize() const  {
      return _persistentRecords._size;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setSize(const tarch::la::Vector<DIMENSIONS,double>& size)  {
      _persistentRecords._size = (size);
   }
   
   
   
    double peanoclaw::records::PatchDescription::getSize(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._size[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::PatchDescription::setSize(int elementIndex, const double& size)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._size[elementIndex]= size;
      
   }
   
   
   
    double peanoclaw::records::PatchDescription::getTime() const  {
      return _persistentRecords._time;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setTime(const double& time)  {
      _persistentRecords._time = time;
   }
   
   
   
    double peanoclaw::records::PatchDescription::getTimestepSize() const  {
      return _persistentRecords._timestepSize;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setTimestepSize(const double& timestepSize)  {
      _persistentRecords._timestepSize = timestepSize;
   }
   
   
   
    double peanoclaw::records::PatchDescription::getHash() const  {
      return _persistentRecords._hash;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setHash(const double& hash)  {
      _persistentRecords._hash = hash;
   }
   
   
   
    int peanoclaw::records::PatchDescription::getSkipGridIterations() const  {
      return _persistentRecords._skipGridIterations;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setSkipGridIterations(const int& skipGridIterations)  {
      _persistentRecords._skipGridIterations = skipGridIterations;
   }
   
   
   
    double peanoclaw::records::PatchDescription::getDemandedMeshWidth() const  {
      return _persistentRecords._demandedMeshWidth;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setDemandedMeshWidth(const double& demandedMeshWidth)  {
      _persistentRecords._demandedMeshWidth = demandedMeshWidth;
   }
   
   
   
    int peanoclaw::records::PatchDescription::getCellDescriptionIndex() const  {
      return _persistentRecords._cellDescriptionIndex;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setCellDescriptionIndex(const int& cellDescriptionIndex)  {
      _persistentRecords._cellDescriptionIndex = cellDescriptionIndex;
   }
   
   
   
   
   std::string peanoclaw::records::PatchDescription::toString() const {
      std::ostringstream stringstr;
      toString(stringstr);
      return stringstr.str();
   }
   
   void peanoclaw::records::PatchDescription::toString (std::ostream& out) const {
      out << "("; 
      out << "isReferenced:" << getIsReferenced();
      out << ",";
      out << "adjacentRanks:[";
   for (int i = 0; i < THREE_POWER_D-1; i++) {
      out << getAdjacentRanks(i) << ",";
   }
   out << getAdjacentRanks(THREE_POWER_D-1) << "]";
      out << ",";
      out << "rank:" << getRank();
      out << ",";
      out << "subdivisionFactor:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getSubdivisionFactor(i) << ",";
   }
   out << getSubdivisionFactor(DIMENSIONS-1) << "]";
      out << ",";
      out << "ghostLayerWidth:" << getGhostLayerWidth();
      out << ",";
      out << "level:" << getLevel();
      out << ",";
      out << "isVirtual:" << getIsVirtual();
      out << ",";
      out << "isRemote:" << getIsRemote();
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
      out << "hash:" << getHash();
      out << ",";
      out << "skipGridIterations:" << getSkipGridIterations();
      out << ",";
      out << "demandedMeshWidth:" << getDemandedMeshWidth();
      out << ",";
      out << "cellDescriptionIndex:" << getCellDescriptionIndex();
      out <<  ")";
   }
   
   
   peanoclaw::records::PatchDescription::PersistentRecords peanoclaw::records::PatchDescription::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::PatchDescriptionPacked peanoclaw::records::PatchDescription::convert() const{
      return PatchDescriptionPacked(
         getIsReferenced(),
         getAdjacentRanks(),
         getRank(),
         getSubdivisionFactor(),
         getGhostLayerWidth(),
         getLevel(),
         getIsVirtual(),
         getIsRemote(),
         getPosition(),
         getSize(),
         getTime(),
         getTimestepSize(),
         getHash(),
         getSkipGridIterations(),
         getDemandedMeshWidth(),
         getCellDescriptionIndex()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::PatchDescription::_log( "peanoclaw::records::PatchDescription" );
      
      MPI_Datatype peanoclaw::records::PatchDescription::Datatype = 0;
      MPI_Datatype peanoclaw::records::PatchDescription::FullDatatype = 0;
      
      
      void peanoclaw::records::PatchDescription::initDatatype() {
         {
            PatchDescription dummyPatchDescription[2];
            
            const int Attributes = 17;
            MPI_Datatype subtypes[Attributes] = {
               MPI_CHAR,		 //isReferenced
               MPI_INT,		 //adjacentRanks
               MPI_INT,		 //rank
               MPI_INT,		 //subdivisionFactor
               MPI_INT,		 //ghostLayerWidth
               MPI_INT,		 //level
               MPI_CHAR,		 //isVirtual
               MPI_CHAR,		 //isRemote
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
               MPI_DOUBLE,		 //hash
               MPI_INT,		 //skipGridIterations
               MPI_DOUBLE,		 //demandedMeshWidth
               MPI_INT,		 //cellDescriptionIndex
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //isReferenced
               THREE_POWER_D,		 //adjacentRanks
               1,		 //rank
               DIMENSIONS,		 //subdivisionFactor
               1,		 //ghostLayerWidth
               1,		 //level
               1,		 //isVirtual
               1,		 //isRemote
               DIMENSIONS,		 //position
               DIMENSIONS,		 //size
               1,		 //time
               1,		 //timestepSize
               1,		 //hash
               1,		 //skipGridIterations
               1,		 //demandedMeshWidth
               1,		 //cellDescriptionIndex
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._isReferenced))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._adjacentRanks[0]))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._rank))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._ghostLayerWidth))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._level))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._isVirtual))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._isRemote))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._position[0]))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._size[0]))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._time))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._timestepSize))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._hash))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._skipGridIterations))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._demandedMeshWidth))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._cellDescriptionIndex))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[1]._persistentRecords._isReferenced))), 		&disp[16] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &PatchDescription::Datatype );
            MPI_Type_commit( &PatchDescription::Datatype );
            
         }
         {
            PatchDescription dummyPatchDescription[2];
            
            const int Attributes = 17;
            MPI_Datatype subtypes[Attributes] = {
               MPI_CHAR,		 //isReferenced
               MPI_INT,		 //adjacentRanks
               MPI_INT,		 //rank
               MPI_INT,		 //subdivisionFactor
               MPI_INT,		 //ghostLayerWidth
               MPI_INT,		 //level
               MPI_CHAR,		 //isVirtual
               MPI_CHAR,		 //isRemote
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
               MPI_DOUBLE,		 //hash
               MPI_INT,		 //skipGridIterations
               MPI_DOUBLE,		 //demandedMeshWidth
               MPI_INT,		 //cellDescriptionIndex
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //isReferenced
               THREE_POWER_D,		 //adjacentRanks
               1,		 //rank
               DIMENSIONS,		 //subdivisionFactor
               1,		 //ghostLayerWidth
               1,		 //level
               1,		 //isVirtual
               1,		 //isRemote
               DIMENSIONS,		 //position
               DIMENSIONS,		 //size
               1,		 //time
               1,		 //timestepSize
               1,		 //hash
               1,		 //skipGridIterations
               1,		 //demandedMeshWidth
               1,		 //cellDescriptionIndex
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._isReferenced))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._adjacentRanks[0]))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._rank))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._ghostLayerWidth))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._level))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._isVirtual))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._isRemote))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._position[0]))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._size[0]))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._time))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._timestepSize))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._hash))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._skipGridIterations))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._demandedMeshWidth))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._cellDescriptionIndex))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[1]._persistentRecords._isReferenced))), 		&disp[16] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &PatchDescription::FullDatatype );
            MPI_Type_commit( &PatchDescription::FullDatatype );
            
         }
         
      }
      
      
      void peanoclaw::records::PatchDescription::shutdownDatatype() {
         MPI_Type_free( &PatchDescription::Datatype );
         MPI_Type_free( &PatchDescription::FullDatatype );
         
      }
      
      void peanoclaw::records::PatchDescription::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "was not able to send message peanoclaw::records::PatchDescription "
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
               msg << "testing for finished send task for peanoclaw::records::PatchDescription "
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
               "peanoclaw::records::PatchDescription",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::PatchDescription",
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
      
      
      
      void peanoclaw::records::PatchDescription::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "failed to start to receive peanoclaw::records::PatchDescription from node "
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
               msg << "testing for finished receive task for peanoclaw::records::PatchDescription failed: "
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
               "peanoclaw::records::PatchDescription",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::PatchDescription",
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
      
      
      
      bool peanoclaw::records::PatchDescription::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
   
   
   peanoclaw::records::PatchDescriptionPacked::PersistentRecords::PersistentRecords() {
      
   }
   
   
   peanoclaw::records::PatchDescriptionPacked::PersistentRecords::PersistentRecords(const bool& isReferenced, const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks, const int& rank, const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& level, const bool& isVirtual, const bool& isRemote, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& hash, const int& skipGridIterations, const double& demandedMeshWidth, const int& cellDescriptionIndex):
   _isReferenced(isReferenced),
   _adjacentRanks(adjacentRanks),
   _rank(rank),
   _subdivisionFactor(subdivisionFactor),
   _ghostLayerWidth(ghostLayerWidth),
   _level(level),
   _isVirtual(isVirtual),
   _isRemote(isRemote),
   _position(position),
   _size(size),
   _time(time),
   _timestepSize(timestepSize),
   _hash(hash),
   _skipGridIterations(skipGridIterations),
   _demandedMeshWidth(demandedMeshWidth),
   _cellDescriptionIndex(cellDescriptionIndex) {
      
   }
   
   
    bool peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getIsReferenced() const  {
      return _isReferenced;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setIsReferenced(const bool& isReferenced)  {
      _isReferenced = isReferenced;
   }
   
   
   
    tarch::la::Vector<THREE_POWER_D,int> peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getAdjacentRanks() const  {
      return _adjacentRanks;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setAdjacentRanks(const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks)  {
      _adjacentRanks = (adjacentRanks);
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getRank() const  {
      return _rank;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setRank(const int& rank)  {
      _rank = rank;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getSubdivisionFactor() const  {
      return _subdivisionFactor;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setSubdivisionFactor(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor)  {
      _subdivisionFactor = (subdivisionFactor);
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getGhostLayerWidth() const  {
      return _ghostLayerWidth;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setGhostLayerWidth(const int& ghostLayerWidth)  {
      _ghostLayerWidth = ghostLayerWidth;
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getLevel() const  {
      return _level;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setLevel(const int& level)  {
      _level = level;
   }
   
   
   
    bool peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getIsVirtual() const  {
      return _isVirtual;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setIsVirtual(const bool& isVirtual)  {
      _isVirtual = isVirtual;
   }
   
   
   
    bool peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getIsRemote() const  {
      return _isRemote;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setIsRemote(const bool& isRemote)  {
      _isRemote = isRemote;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getPosition() const  {
      return _position;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setPosition(const tarch::la::Vector<DIMENSIONS,double>& position)  {
      _position = (position);
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getSize() const  {
      return _size;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setSize(const tarch::la::Vector<DIMENSIONS,double>& size)  {
      _size = (size);
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getTime() const  {
      return _time;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setTime(const double& time)  {
      _time = time;
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getTimestepSize() const  {
      return _timestepSize;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setTimestepSize(const double& timestepSize)  {
      _timestepSize = timestepSize;
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getHash() const  {
      return _hash;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setHash(const double& hash)  {
      _hash = hash;
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getSkipGridIterations() const  {
      return _skipGridIterations;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setSkipGridIterations(const int& skipGridIterations)  {
      _skipGridIterations = skipGridIterations;
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getDemandedMeshWidth() const  {
      return _demandedMeshWidth;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setDemandedMeshWidth(const double& demandedMeshWidth)  {
      _demandedMeshWidth = demandedMeshWidth;
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getCellDescriptionIndex() const  {
      return _cellDescriptionIndex;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setCellDescriptionIndex(const int& cellDescriptionIndex)  {
      _cellDescriptionIndex = cellDescriptionIndex;
   }
   
   
   peanoclaw::records::PatchDescriptionPacked::PatchDescriptionPacked() {
      
   }
   
   
   peanoclaw::records::PatchDescriptionPacked::PatchDescriptionPacked(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._isReferenced, persistentRecords._adjacentRanks, persistentRecords._rank, persistentRecords._subdivisionFactor, persistentRecords._ghostLayerWidth, persistentRecords._level, persistentRecords._isVirtual, persistentRecords._isRemote, persistentRecords._position, persistentRecords._size, persistentRecords._time, persistentRecords._timestepSize, persistentRecords._hash, persistentRecords._skipGridIterations, persistentRecords._demandedMeshWidth, persistentRecords._cellDescriptionIndex) {
      
   }
   
   
   peanoclaw::records::PatchDescriptionPacked::PatchDescriptionPacked(const bool& isReferenced, const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks, const int& rank, const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& level, const bool& isVirtual, const bool& isRemote, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& hash, const int& skipGridIterations, const double& demandedMeshWidth, const int& cellDescriptionIndex):
   _persistentRecords(isReferenced, adjacentRanks, rank, subdivisionFactor, ghostLayerWidth, level, isVirtual, isRemote, position, size, time, timestepSize, hash, skipGridIterations, demandedMeshWidth, cellDescriptionIndex) {
      
   }
   
   
   peanoclaw::records::PatchDescriptionPacked::~PatchDescriptionPacked() { }
   
   
    bool peanoclaw::records::PatchDescriptionPacked::getIsReferenced() const  {
      return _persistentRecords._isReferenced;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setIsReferenced(const bool& isReferenced)  {
      _persistentRecords._isReferenced = isReferenced;
   }
   
   
   
    tarch::la::Vector<THREE_POWER_D,int> peanoclaw::records::PatchDescriptionPacked::getAdjacentRanks() const  {
      return _persistentRecords._adjacentRanks;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setAdjacentRanks(const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks)  {
      _persistentRecords._adjacentRanks = (adjacentRanks);
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::getAdjacentRanks(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<THREE_POWER_D);
      return _persistentRecords._adjacentRanks[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setAdjacentRanks(int elementIndex, const int& adjacentRanks)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<THREE_POWER_D);
      _persistentRecords._adjacentRanks[elementIndex]= adjacentRanks;
      
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::getRank() const  {
      return _persistentRecords._rank;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setRank(const int& rank)  {
      _persistentRecords._rank = rank;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::PatchDescriptionPacked::getSubdivisionFactor() const  {
      return _persistentRecords._subdivisionFactor;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setSubdivisionFactor(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor)  {
      _persistentRecords._subdivisionFactor = (subdivisionFactor);
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::getSubdivisionFactor(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._subdivisionFactor[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setSubdivisionFactor(int elementIndex, const int& subdivisionFactor)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._subdivisionFactor[elementIndex]= subdivisionFactor;
      
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::getGhostLayerWidth() const  {
      return _persistentRecords._ghostLayerWidth;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setGhostLayerWidth(const int& ghostLayerWidth)  {
      _persistentRecords._ghostLayerWidth = ghostLayerWidth;
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::getLevel() const  {
      return _persistentRecords._level;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setLevel(const int& level)  {
      _persistentRecords._level = level;
   }
   
   
   
    bool peanoclaw::records::PatchDescriptionPacked::getIsVirtual() const  {
      return _persistentRecords._isVirtual;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setIsVirtual(const bool& isVirtual)  {
      _persistentRecords._isVirtual = isVirtual;
   }
   
   
   
    bool peanoclaw::records::PatchDescriptionPacked::getIsRemote() const  {
      return _persistentRecords._isRemote;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setIsRemote(const bool& isRemote)  {
      _persistentRecords._isRemote = isRemote;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::PatchDescriptionPacked::getPosition() const  {
      return _persistentRecords._position;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setPosition(const tarch::la::Vector<DIMENSIONS,double>& position)  {
      _persistentRecords._position = (position);
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::getPosition(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._position[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setPosition(int elementIndex, const double& position)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._position[elementIndex]= position;
      
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::PatchDescriptionPacked::getSize() const  {
      return _persistentRecords._size;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setSize(const tarch::la::Vector<DIMENSIONS,double>& size)  {
      _persistentRecords._size = (size);
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::getSize(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._size[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setSize(int elementIndex, const double& size)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._size[elementIndex]= size;
      
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::getTime() const  {
      return _persistentRecords._time;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setTime(const double& time)  {
      _persistentRecords._time = time;
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::getTimestepSize() const  {
      return _persistentRecords._timestepSize;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setTimestepSize(const double& timestepSize)  {
      _persistentRecords._timestepSize = timestepSize;
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::getHash() const  {
      return _persistentRecords._hash;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setHash(const double& hash)  {
      _persistentRecords._hash = hash;
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::getSkipGridIterations() const  {
      return _persistentRecords._skipGridIterations;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setSkipGridIterations(const int& skipGridIterations)  {
      _persistentRecords._skipGridIterations = skipGridIterations;
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::getDemandedMeshWidth() const  {
      return _persistentRecords._demandedMeshWidth;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setDemandedMeshWidth(const double& demandedMeshWidth)  {
      _persistentRecords._demandedMeshWidth = demandedMeshWidth;
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::getCellDescriptionIndex() const  {
      return _persistentRecords._cellDescriptionIndex;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setCellDescriptionIndex(const int& cellDescriptionIndex)  {
      _persistentRecords._cellDescriptionIndex = cellDescriptionIndex;
   }
   
   
   
   
   std::string peanoclaw::records::PatchDescriptionPacked::toString() const {
      std::ostringstream stringstr;
      toString(stringstr);
      return stringstr.str();
   }
   
   void peanoclaw::records::PatchDescriptionPacked::toString (std::ostream& out) const {
      out << "("; 
      out << "isReferenced:" << getIsReferenced();
      out << ",";
      out << "adjacentRanks:[";
   for (int i = 0; i < THREE_POWER_D-1; i++) {
      out << getAdjacentRanks(i) << ",";
   }
   out << getAdjacentRanks(THREE_POWER_D-1) << "]";
      out << ",";
      out << "rank:" << getRank();
      out << ",";
      out << "subdivisionFactor:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getSubdivisionFactor(i) << ",";
   }
   out << getSubdivisionFactor(DIMENSIONS-1) << "]";
      out << ",";
      out << "ghostLayerWidth:" << getGhostLayerWidth();
      out << ",";
      out << "level:" << getLevel();
      out << ",";
      out << "isVirtual:" << getIsVirtual();
      out << ",";
      out << "isRemote:" << getIsRemote();
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
      out << "hash:" << getHash();
      out << ",";
      out << "skipGridIterations:" << getSkipGridIterations();
      out << ",";
      out << "demandedMeshWidth:" << getDemandedMeshWidth();
      out << ",";
      out << "cellDescriptionIndex:" << getCellDescriptionIndex();
      out <<  ")";
   }
   
   
   peanoclaw::records::PatchDescriptionPacked::PersistentRecords peanoclaw::records::PatchDescriptionPacked::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::PatchDescription peanoclaw::records::PatchDescriptionPacked::convert() const{
      return PatchDescription(
         getIsReferenced(),
         getAdjacentRanks(),
         getRank(),
         getSubdivisionFactor(),
         getGhostLayerWidth(),
         getLevel(),
         getIsVirtual(),
         getIsRemote(),
         getPosition(),
         getSize(),
         getTime(),
         getTimestepSize(),
         getHash(),
         getSkipGridIterations(),
         getDemandedMeshWidth(),
         getCellDescriptionIndex()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::PatchDescriptionPacked::_log( "peanoclaw::records::PatchDescriptionPacked" );
      
      MPI_Datatype peanoclaw::records::PatchDescriptionPacked::Datatype = 0;
      MPI_Datatype peanoclaw::records::PatchDescriptionPacked::FullDatatype = 0;
      
      
      void peanoclaw::records::PatchDescriptionPacked::initDatatype() {
         {
            PatchDescriptionPacked dummyPatchDescriptionPacked[2];
            
            const int Attributes = 17;
            MPI_Datatype subtypes[Attributes] = {
               MPI_CHAR,		 //isReferenced
               MPI_INT,		 //adjacentRanks
               MPI_INT,		 //rank
               MPI_INT,		 //subdivisionFactor
               MPI_INT,		 //ghostLayerWidth
               MPI_INT,		 //level
               MPI_CHAR,		 //isVirtual
               MPI_CHAR,		 //isRemote
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
               MPI_DOUBLE,		 //hash
               MPI_INT,		 //skipGridIterations
               MPI_DOUBLE,		 //demandedMeshWidth
               MPI_INT,		 //cellDescriptionIndex
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //isReferenced
               THREE_POWER_D,		 //adjacentRanks
               1,		 //rank
               DIMENSIONS,		 //subdivisionFactor
               1,		 //ghostLayerWidth
               1,		 //level
               1,		 //isVirtual
               1,		 //isRemote
               DIMENSIONS,		 //position
               DIMENSIONS,		 //size
               1,		 //time
               1,		 //timestepSize
               1,		 //hash
               1,		 //skipGridIterations
               1,		 //demandedMeshWidth
               1,		 //cellDescriptionIndex
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._isReferenced))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._adjacentRanks[0]))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._rank))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._ghostLayerWidth))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._level))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._isVirtual))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._isRemote))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._position[0]))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._size[0]))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._time))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._timestepSize))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._hash))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._skipGridIterations))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._demandedMeshWidth))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[1]._persistentRecords._isReferenced))), 		&disp[16] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &PatchDescriptionPacked::Datatype );
            MPI_Type_commit( &PatchDescriptionPacked::Datatype );
            
         }
         {
            PatchDescriptionPacked dummyPatchDescriptionPacked[2];
            
            const int Attributes = 17;
            MPI_Datatype subtypes[Attributes] = {
               MPI_CHAR,		 //isReferenced
               MPI_INT,		 //adjacentRanks
               MPI_INT,		 //rank
               MPI_INT,		 //subdivisionFactor
               MPI_INT,		 //ghostLayerWidth
               MPI_INT,		 //level
               MPI_CHAR,		 //isVirtual
               MPI_CHAR,		 //isRemote
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
               MPI_DOUBLE,		 //hash
               MPI_INT,		 //skipGridIterations
               MPI_DOUBLE,		 //demandedMeshWidth
               MPI_INT,		 //cellDescriptionIndex
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //isReferenced
               THREE_POWER_D,		 //adjacentRanks
               1,		 //rank
               DIMENSIONS,		 //subdivisionFactor
               1,		 //ghostLayerWidth
               1,		 //level
               1,		 //isVirtual
               1,		 //isRemote
               DIMENSIONS,		 //position
               DIMENSIONS,		 //size
               1,		 //time
               1,		 //timestepSize
               1,		 //hash
               1,		 //skipGridIterations
               1,		 //demandedMeshWidth
               1,		 //cellDescriptionIndex
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._isReferenced))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._adjacentRanks[0]))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._rank))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._ghostLayerWidth))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._level))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._isVirtual))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._isRemote))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._position[0]))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._size[0]))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._time))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._timestepSize))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._hash))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._skipGridIterations))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._demandedMeshWidth))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[1]._persistentRecords._isReferenced))), 		&disp[16] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &PatchDescriptionPacked::FullDatatype );
            MPI_Type_commit( &PatchDescriptionPacked::FullDatatype );
            
         }
         
      }
      
      
      void peanoclaw::records::PatchDescriptionPacked::shutdownDatatype() {
         MPI_Type_free( &PatchDescriptionPacked::Datatype );
         MPI_Type_free( &PatchDescriptionPacked::FullDatatype );
         
      }
      
      void peanoclaw::records::PatchDescriptionPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "was not able to send message peanoclaw::records::PatchDescriptionPacked "
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
               msg << "testing for finished send task for peanoclaw::records::PatchDescriptionPacked "
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
               "peanoclaw::records::PatchDescriptionPacked",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::PatchDescriptionPacked",
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
      
      
      
      void peanoclaw::records::PatchDescriptionPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "failed to start to receive peanoclaw::records::PatchDescriptionPacked from node "
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
               msg << "testing for finished receive task for peanoclaw::records::PatchDescriptionPacked failed: "
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
               "peanoclaw::records::PatchDescriptionPacked",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::PatchDescriptionPacked",
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
      
      
      
      bool peanoclaw::records::PatchDescriptionPacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
   peanoclaw::records::PatchDescription::PersistentRecords::PersistentRecords() {
      
   }
   
   
   peanoclaw::records::PatchDescription::PersistentRecords::PersistentRecords(const bool& isReferenced, const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks, const int& rank, const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& level, const bool& isVirtual, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& hash, const int& skipGridIterations, const double& demandedMeshWidth, const int& cellDescriptionIndex):
   _isReferenced(isReferenced),
   _adjacentRanks(adjacentRanks),
   _rank(rank),
   _subdivisionFactor(subdivisionFactor),
   _ghostLayerWidth(ghostLayerWidth),
   _level(level),
   _isVirtual(isVirtual),
   _position(position),
   _size(size),
   _time(time),
   _timestepSize(timestepSize),
   _hash(hash),
   _skipGridIterations(skipGridIterations),
   _demandedMeshWidth(demandedMeshWidth),
   _cellDescriptionIndex(cellDescriptionIndex) {
      
   }
   
   
    bool peanoclaw::records::PatchDescription::PersistentRecords::getIsReferenced() const  {
      return _isReferenced;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setIsReferenced(const bool& isReferenced)  {
      _isReferenced = isReferenced;
   }
   
   
   
    tarch::la::Vector<THREE_POWER_D,int> peanoclaw::records::PatchDescription::PersistentRecords::getAdjacentRanks() const  {
      return _adjacentRanks;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setAdjacentRanks(const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks)  {
      _adjacentRanks = (adjacentRanks);
   }
   
   
   
    int peanoclaw::records::PatchDescription::PersistentRecords::getRank() const  {
      return _rank;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setRank(const int& rank)  {
      _rank = rank;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::PatchDescription::PersistentRecords::getSubdivisionFactor() const  {
      return _subdivisionFactor;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setSubdivisionFactor(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor)  {
      _subdivisionFactor = (subdivisionFactor);
   }
   
   
   
    int peanoclaw::records::PatchDescription::PersistentRecords::getGhostLayerWidth() const  {
      return _ghostLayerWidth;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setGhostLayerWidth(const int& ghostLayerWidth)  {
      _ghostLayerWidth = ghostLayerWidth;
   }
   
   
   
    int peanoclaw::records::PatchDescription::PersistentRecords::getLevel() const  {
      return _level;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setLevel(const int& level)  {
      _level = level;
   }
   
   
   
    bool peanoclaw::records::PatchDescription::PersistentRecords::getIsVirtual() const  {
      return _isVirtual;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setIsVirtual(const bool& isVirtual)  {
      _isVirtual = isVirtual;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::PatchDescription::PersistentRecords::getPosition() const  {
      return _position;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setPosition(const tarch::la::Vector<DIMENSIONS,double>& position)  {
      _position = (position);
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::PatchDescription::PersistentRecords::getSize() const  {
      return _size;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setSize(const tarch::la::Vector<DIMENSIONS,double>& size)  {
      _size = (size);
   }
   
   
   
    double peanoclaw::records::PatchDescription::PersistentRecords::getTime() const  {
      return _time;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setTime(const double& time)  {
      _time = time;
   }
   
   
   
    double peanoclaw::records::PatchDescription::PersistentRecords::getTimestepSize() const  {
      return _timestepSize;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setTimestepSize(const double& timestepSize)  {
      _timestepSize = timestepSize;
   }
   
   
   
    double peanoclaw::records::PatchDescription::PersistentRecords::getHash() const  {
      return _hash;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setHash(const double& hash)  {
      _hash = hash;
   }
   
   
   
    int peanoclaw::records::PatchDescription::PersistentRecords::getSkipGridIterations() const  {
      return _skipGridIterations;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setSkipGridIterations(const int& skipGridIterations)  {
      _skipGridIterations = skipGridIterations;
   }
   
   
   
    double peanoclaw::records::PatchDescription::PersistentRecords::getDemandedMeshWidth() const  {
      return _demandedMeshWidth;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setDemandedMeshWidth(const double& demandedMeshWidth)  {
      _demandedMeshWidth = demandedMeshWidth;
   }
   
   
   
    int peanoclaw::records::PatchDescription::PersistentRecords::getCellDescriptionIndex() const  {
      return _cellDescriptionIndex;
   }
   
   
   
    void peanoclaw::records::PatchDescription::PersistentRecords::setCellDescriptionIndex(const int& cellDescriptionIndex)  {
      _cellDescriptionIndex = cellDescriptionIndex;
   }
   
   
   peanoclaw::records::PatchDescription::PatchDescription() {
      
   }
   
   
   peanoclaw::records::PatchDescription::PatchDescription(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._isReferenced, persistentRecords._adjacentRanks, persistentRecords._rank, persistentRecords._subdivisionFactor, persistentRecords._ghostLayerWidth, persistentRecords._level, persistentRecords._isVirtual, persistentRecords._position, persistentRecords._size, persistentRecords._time, persistentRecords._timestepSize, persistentRecords._hash, persistentRecords._skipGridIterations, persistentRecords._demandedMeshWidth, persistentRecords._cellDescriptionIndex) {
      
   }
   
   
   peanoclaw::records::PatchDescription::PatchDescription(const bool& isReferenced, const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks, const int& rank, const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& level, const bool& isVirtual, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& hash, const int& skipGridIterations, const double& demandedMeshWidth, const int& cellDescriptionIndex):
   _persistentRecords(isReferenced, adjacentRanks, rank, subdivisionFactor, ghostLayerWidth, level, isVirtual, position, size, time, timestepSize, hash, skipGridIterations, demandedMeshWidth, cellDescriptionIndex) {
      
   }
   
   
   peanoclaw::records::PatchDescription::~PatchDescription() { }
   
   
    bool peanoclaw::records::PatchDescription::getIsReferenced() const  {
      return _persistentRecords._isReferenced;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setIsReferenced(const bool& isReferenced)  {
      _persistentRecords._isReferenced = isReferenced;
   }
   
   
   
    tarch::la::Vector<THREE_POWER_D,int> peanoclaw::records::PatchDescription::getAdjacentRanks() const  {
      return _persistentRecords._adjacentRanks;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setAdjacentRanks(const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks)  {
      _persistentRecords._adjacentRanks = (adjacentRanks);
   }
   
   
   
    int peanoclaw::records::PatchDescription::getAdjacentRanks(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<THREE_POWER_D);
      return _persistentRecords._adjacentRanks[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::PatchDescription::setAdjacentRanks(int elementIndex, const int& adjacentRanks)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<THREE_POWER_D);
      _persistentRecords._adjacentRanks[elementIndex]= adjacentRanks;
      
   }
   
   
   
    int peanoclaw::records::PatchDescription::getRank() const  {
      return _persistentRecords._rank;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setRank(const int& rank)  {
      _persistentRecords._rank = rank;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::PatchDescription::getSubdivisionFactor() const  {
      return _persistentRecords._subdivisionFactor;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setSubdivisionFactor(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor)  {
      _persistentRecords._subdivisionFactor = (subdivisionFactor);
   }
   
   
   
    int peanoclaw::records::PatchDescription::getSubdivisionFactor(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._subdivisionFactor[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::PatchDescription::setSubdivisionFactor(int elementIndex, const int& subdivisionFactor)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._subdivisionFactor[elementIndex]= subdivisionFactor;
      
   }
   
   
   
    int peanoclaw::records::PatchDescription::getGhostLayerWidth() const  {
      return _persistentRecords._ghostLayerWidth;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setGhostLayerWidth(const int& ghostLayerWidth)  {
      _persistentRecords._ghostLayerWidth = ghostLayerWidth;
   }
   
   
   
    int peanoclaw::records::PatchDescription::getLevel() const  {
      return _persistentRecords._level;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setLevel(const int& level)  {
      _persistentRecords._level = level;
   }
   
   
   
    bool peanoclaw::records::PatchDescription::getIsVirtual() const  {
      return _persistentRecords._isVirtual;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setIsVirtual(const bool& isVirtual)  {
      _persistentRecords._isVirtual = isVirtual;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::PatchDescription::getPosition() const  {
      return _persistentRecords._position;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setPosition(const tarch::la::Vector<DIMENSIONS,double>& position)  {
      _persistentRecords._position = (position);
   }
   
   
   
    double peanoclaw::records::PatchDescription::getPosition(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._position[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::PatchDescription::setPosition(int elementIndex, const double& position)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._position[elementIndex]= position;
      
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::PatchDescription::getSize() const  {
      return _persistentRecords._size;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setSize(const tarch::la::Vector<DIMENSIONS,double>& size)  {
      _persistentRecords._size = (size);
   }
   
   
   
    double peanoclaw::records::PatchDescription::getSize(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._size[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::PatchDescription::setSize(int elementIndex, const double& size)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._size[elementIndex]= size;
      
   }
   
   
   
    double peanoclaw::records::PatchDescription::getTime() const  {
      return _persistentRecords._time;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setTime(const double& time)  {
      _persistentRecords._time = time;
   }
   
   
   
    double peanoclaw::records::PatchDescription::getTimestepSize() const  {
      return _persistentRecords._timestepSize;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setTimestepSize(const double& timestepSize)  {
      _persistentRecords._timestepSize = timestepSize;
   }
   
   
   
    double peanoclaw::records::PatchDescription::getHash() const  {
      return _persistentRecords._hash;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setHash(const double& hash)  {
      _persistentRecords._hash = hash;
   }
   
   
   
    int peanoclaw::records::PatchDescription::getSkipGridIterations() const  {
      return _persistentRecords._skipGridIterations;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setSkipGridIterations(const int& skipGridIterations)  {
      _persistentRecords._skipGridIterations = skipGridIterations;
   }
   
   
   
    double peanoclaw::records::PatchDescription::getDemandedMeshWidth() const  {
      return _persistentRecords._demandedMeshWidth;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setDemandedMeshWidth(const double& demandedMeshWidth)  {
      _persistentRecords._demandedMeshWidth = demandedMeshWidth;
   }
   
   
   
    int peanoclaw::records::PatchDescription::getCellDescriptionIndex() const  {
      return _persistentRecords._cellDescriptionIndex;
   }
   
   
   
    void peanoclaw::records::PatchDescription::setCellDescriptionIndex(const int& cellDescriptionIndex)  {
      _persistentRecords._cellDescriptionIndex = cellDescriptionIndex;
   }
   
   
   
   
   std::string peanoclaw::records::PatchDescription::toString() const {
      std::ostringstream stringstr;
      toString(stringstr);
      return stringstr.str();
   }
   
   void peanoclaw::records::PatchDescription::toString (std::ostream& out) const {
      out << "("; 
      out << "isReferenced:" << getIsReferenced();
      out << ",";
      out << "adjacentRanks:[";
   for (int i = 0; i < THREE_POWER_D-1; i++) {
      out << getAdjacentRanks(i) << ",";
   }
   out << getAdjacentRanks(THREE_POWER_D-1) << "]";
      out << ",";
      out << "rank:" << getRank();
      out << ",";
      out << "subdivisionFactor:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getSubdivisionFactor(i) << ",";
   }
   out << getSubdivisionFactor(DIMENSIONS-1) << "]";
      out << ",";
      out << "ghostLayerWidth:" << getGhostLayerWidth();
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
      out << "hash:" << getHash();
      out << ",";
      out << "skipGridIterations:" << getSkipGridIterations();
      out << ",";
      out << "demandedMeshWidth:" << getDemandedMeshWidth();
      out << ",";
      out << "cellDescriptionIndex:" << getCellDescriptionIndex();
      out <<  ")";
   }
   
   
   peanoclaw::records::PatchDescription::PersistentRecords peanoclaw::records::PatchDescription::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::PatchDescriptionPacked peanoclaw::records::PatchDescription::convert() const{
      return PatchDescriptionPacked(
         getIsReferenced(),
         getAdjacentRanks(),
         getRank(),
         getSubdivisionFactor(),
         getGhostLayerWidth(),
         getLevel(),
         getIsVirtual(),
         getPosition(),
         getSize(),
         getTime(),
         getTimestepSize(),
         getHash(),
         getSkipGridIterations(),
         getDemandedMeshWidth(),
         getCellDescriptionIndex()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::PatchDescription::_log( "peanoclaw::records::PatchDescription" );
      
      MPI_Datatype peanoclaw::records::PatchDescription::Datatype = 0;
      MPI_Datatype peanoclaw::records::PatchDescription::FullDatatype = 0;
      
      
      void peanoclaw::records::PatchDescription::initDatatype() {
         {
            PatchDescription dummyPatchDescription[2];
            
            const int Attributes = 16;
            MPI_Datatype subtypes[Attributes] = {
               MPI_CHAR,		 //isReferenced
               MPI_INT,		 //adjacentRanks
               MPI_INT,		 //rank
               MPI_INT,		 //subdivisionFactor
               MPI_INT,		 //ghostLayerWidth
               MPI_INT,		 //level
               MPI_CHAR,		 //isVirtual
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
               MPI_DOUBLE,		 //hash
               MPI_INT,		 //skipGridIterations
               MPI_DOUBLE,		 //demandedMeshWidth
               MPI_INT,		 //cellDescriptionIndex
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //isReferenced
               THREE_POWER_D,		 //adjacentRanks
               1,		 //rank
               DIMENSIONS,		 //subdivisionFactor
               1,		 //ghostLayerWidth
               1,		 //level
               1,		 //isVirtual
               DIMENSIONS,		 //position
               DIMENSIONS,		 //size
               1,		 //time
               1,		 //timestepSize
               1,		 //hash
               1,		 //skipGridIterations
               1,		 //demandedMeshWidth
               1,		 //cellDescriptionIndex
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._isReferenced))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._adjacentRanks[0]))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._rank))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._ghostLayerWidth))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._level))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._isVirtual))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._position[0]))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._size[0]))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._time))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._timestepSize))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._hash))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._skipGridIterations))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._demandedMeshWidth))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._cellDescriptionIndex))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[1]._persistentRecords._isReferenced))), 		&disp[15] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &PatchDescription::Datatype );
            MPI_Type_commit( &PatchDescription::Datatype );
            
         }
         {
            PatchDescription dummyPatchDescription[2];
            
            const int Attributes = 16;
            MPI_Datatype subtypes[Attributes] = {
               MPI_CHAR,		 //isReferenced
               MPI_INT,		 //adjacentRanks
               MPI_INT,		 //rank
               MPI_INT,		 //subdivisionFactor
               MPI_INT,		 //ghostLayerWidth
               MPI_INT,		 //level
               MPI_CHAR,		 //isVirtual
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
               MPI_DOUBLE,		 //hash
               MPI_INT,		 //skipGridIterations
               MPI_DOUBLE,		 //demandedMeshWidth
               MPI_INT,		 //cellDescriptionIndex
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //isReferenced
               THREE_POWER_D,		 //adjacentRanks
               1,		 //rank
               DIMENSIONS,		 //subdivisionFactor
               1,		 //ghostLayerWidth
               1,		 //level
               1,		 //isVirtual
               DIMENSIONS,		 //position
               DIMENSIONS,		 //size
               1,		 //time
               1,		 //timestepSize
               1,		 //hash
               1,		 //skipGridIterations
               1,		 //demandedMeshWidth
               1,		 //cellDescriptionIndex
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._isReferenced))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._adjacentRanks[0]))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._rank))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._ghostLayerWidth))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._level))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._isVirtual))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._position[0]))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._size[0]))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._time))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._timestepSize))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._hash))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._skipGridIterations))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._demandedMeshWidth))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._cellDescriptionIndex))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[1]._persistentRecords._isReferenced))), 		&disp[15] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &PatchDescription::FullDatatype );
            MPI_Type_commit( &PatchDescription::FullDatatype );
            
         }
         
      }
      
      
      void peanoclaw::records::PatchDescription::shutdownDatatype() {
         MPI_Type_free( &PatchDescription::Datatype );
         MPI_Type_free( &PatchDescription::FullDatatype );
         
      }
      
      void peanoclaw::records::PatchDescription::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "was not able to send message peanoclaw::records::PatchDescription "
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
               msg << "testing for finished send task for peanoclaw::records::PatchDescription "
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
               "peanoclaw::records::PatchDescription",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::PatchDescription",
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
      
      
      
      void peanoclaw::records::PatchDescription::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "failed to start to receive peanoclaw::records::PatchDescription from node "
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
               msg << "testing for finished receive task for peanoclaw::records::PatchDescription failed: "
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
               "peanoclaw::records::PatchDescription",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::PatchDescription",
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
      
      
      
      bool peanoclaw::records::PatchDescription::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
   
   
   peanoclaw::records::PatchDescriptionPacked::PersistentRecords::PersistentRecords() {
      
   }
   
   
   peanoclaw::records::PatchDescriptionPacked::PersistentRecords::PersistentRecords(const bool& isReferenced, const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks, const int& rank, const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& level, const bool& isVirtual, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& hash, const int& skipGridIterations, const double& demandedMeshWidth, const int& cellDescriptionIndex):
   _isReferenced(isReferenced),
   _adjacentRanks(adjacentRanks),
   _rank(rank),
   _subdivisionFactor(subdivisionFactor),
   _ghostLayerWidth(ghostLayerWidth),
   _level(level),
   _isVirtual(isVirtual),
   _position(position),
   _size(size),
   _time(time),
   _timestepSize(timestepSize),
   _hash(hash),
   _skipGridIterations(skipGridIterations),
   _demandedMeshWidth(demandedMeshWidth),
   _cellDescriptionIndex(cellDescriptionIndex) {
      
   }
   
   
    bool peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getIsReferenced() const  {
      return _isReferenced;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setIsReferenced(const bool& isReferenced)  {
      _isReferenced = isReferenced;
   }
   
   
   
    tarch::la::Vector<THREE_POWER_D,int> peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getAdjacentRanks() const  {
      return _adjacentRanks;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setAdjacentRanks(const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks)  {
      _adjacentRanks = (adjacentRanks);
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getRank() const  {
      return _rank;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setRank(const int& rank)  {
      _rank = rank;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getSubdivisionFactor() const  {
      return _subdivisionFactor;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setSubdivisionFactor(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor)  {
      _subdivisionFactor = (subdivisionFactor);
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getGhostLayerWidth() const  {
      return _ghostLayerWidth;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setGhostLayerWidth(const int& ghostLayerWidth)  {
      _ghostLayerWidth = ghostLayerWidth;
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getLevel() const  {
      return _level;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setLevel(const int& level)  {
      _level = level;
   }
   
   
   
    bool peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getIsVirtual() const  {
      return _isVirtual;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setIsVirtual(const bool& isVirtual)  {
      _isVirtual = isVirtual;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getPosition() const  {
      return _position;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setPosition(const tarch::la::Vector<DIMENSIONS,double>& position)  {
      _position = (position);
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getSize() const  {
      return _size;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setSize(const tarch::la::Vector<DIMENSIONS,double>& size)  {
      _size = (size);
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getTime() const  {
      return _time;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setTime(const double& time)  {
      _time = time;
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getTimestepSize() const  {
      return _timestepSize;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setTimestepSize(const double& timestepSize)  {
      _timestepSize = timestepSize;
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getHash() const  {
      return _hash;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setHash(const double& hash)  {
      _hash = hash;
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getSkipGridIterations() const  {
      return _skipGridIterations;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setSkipGridIterations(const int& skipGridIterations)  {
      _skipGridIterations = skipGridIterations;
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getDemandedMeshWidth() const  {
      return _demandedMeshWidth;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setDemandedMeshWidth(const double& demandedMeshWidth)  {
      _demandedMeshWidth = demandedMeshWidth;
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::PersistentRecords::getCellDescriptionIndex() const  {
      return _cellDescriptionIndex;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::PersistentRecords::setCellDescriptionIndex(const int& cellDescriptionIndex)  {
      _cellDescriptionIndex = cellDescriptionIndex;
   }
   
   
   peanoclaw::records::PatchDescriptionPacked::PatchDescriptionPacked() {
      
   }
   
   
   peanoclaw::records::PatchDescriptionPacked::PatchDescriptionPacked(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._isReferenced, persistentRecords._adjacentRanks, persistentRecords._rank, persistentRecords._subdivisionFactor, persistentRecords._ghostLayerWidth, persistentRecords._level, persistentRecords._isVirtual, persistentRecords._position, persistentRecords._size, persistentRecords._time, persistentRecords._timestepSize, persistentRecords._hash, persistentRecords._skipGridIterations, persistentRecords._demandedMeshWidth, persistentRecords._cellDescriptionIndex) {
      
   }
   
   
   peanoclaw::records::PatchDescriptionPacked::PatchDescriptionPacked(const bool& isReferenced, const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks, const int& rank, const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& level, const bool& isVirtual, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const double& hash, const int& skipGridIterations, const double& demandedMeshWidth, const int& cellDescriptionIndex):
   _persistentRecords(isReferenced, adjacentRanks, rank, subdivisionFactor, ghostLayerWidth, level, isVirtual, position, size, time, timestepSize, hash, skipGridIterations, demandedMeshWidth, cellDescriptionIndex) {
      
   }
   
   
   peanoclaw::records::PatchDescriptionPacked::~PatchDescriptionPacked() { }
   
   
    bool peanoclaw::records::PatchDescriptionPacked::getIsReferenced() const  {
      return _persistentRecords._isReferenced;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setIsReferenced(const bool& isReferenced)  {
      _persistentRecords._isReferenced = isReferenced;
   }
   
   
   
    tarch::la::Vector<THREE_POWER_D,int> peanoclaw::records::PatchDescriptionPacked::getAdjacentRanks() const  {
      return _persistentRecords._adjacentRanks;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setAdjacentRanks(const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks)  {
      _persistentRecords._adjacentRanks = (adjacentRanks);
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::getAdjacentRanks(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<THREE_POWER_D);
      return _persistentRecords._adjacentRanks[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setAdjacentRanks(int elementIndex, const int& adjacentRanks)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<THREE_POWER_D);
      _persistentRecords._adjacentRanks[elementIndex]= adjacentRanks;
      
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::getRank() const  {
      return _persistentRecords._rank;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setRank(const int& rank)  {
      _persistentRecords._rank = rank;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,int> peanoclaw::records::PatchDescriptionPacked::getSubdivisionFactor() const  {
      return _persistentRecords._subdivisionFactor;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setSubdivisionFactor(const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor)  {
      _persistentRecords._subdivisionFactor = (subdivisionFactor);
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::getSubdivisionFactor(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._subdivisionFactor[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setSubdivisionFactor(int elementIndex, const int& subdivisionFactor)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._subdivisionFactor[elementIndex]= subdivisionFactor;
      
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::getGhostLayerWidth() const  {
      return _persistentRecords._ghostLayerWidth;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setGhostLayerWidth(const int& ghostLayerWidth)  {
      _persistentRecords._ghostLayerWidth = ghostLayerWidth;
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::getLevel() const  {
      return _persistentRecords._level;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setLevel(const int& level)  {
      _persistentRecords._level = level;
   }
   
   
   
    bool peanoclaw::records::PatchDescriptionPacked::getIsVirtual() const  {
      return _persistentRecords._isVirtual;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setIsVirtual(const bool& isVirtual)  {
      _persistentRecords._isVirtual = isVirtual;
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::PatchDescriptionPacked::getPosition() const  {
      return _persistentRecords._position;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setPosition(const tarch::la::Vector<DIMENSIONS,double>& position)  {
      _persistentRecords._position = (position);
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::getPosition(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._position[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setPosition(int elementIndex, const double& position)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._position[elementIndex]= position;
      
   }
   
   
   
    tarch::la::Vector<DIMENSIONS,double> peanoclaw::records::PatchDescriptionPacked::getSize() const  {
      return _persistentRecords._size;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setSize(const tarch::la::Vector<DIMENSIONS,double>& size)  {
      _persistentRecords._size = (size);
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::getSize(int elementIndex) const  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      return _persistentRecords._size[elementIndex];
      
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setSize(int elementIndex, const double& size)  {
      assertion(elementIndex>=0);
      assertion(elementIndex<DIMENSIONS);
      _persistentRecords._size[elementIndex]= size;
      
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::getTime() const  {
      return _persistentRecords._time;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setTime(const double& time)  {
      _persistentRecords._time = time;
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::getTimestepSize() const  {
      return _persistentRecords._timestepSize;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setTimestepSize(const double& timestepSize)  {
      _persistentRecords._timestepSize = timestepSize;
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::getHash() const  {
      return _persistentRecords._hash;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setHash(const double& hash)  {
      _persistentRecords._hash = hash;
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::getSkipGridIterations() const  {
      return _persistentRecords._skipGridIterations;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setSkipGridIterations(const int& skipGridIterations)  {
      _persistentRecords._skipGridIterations = skipGridIterations;
   }
   
   
   
    double peanoclaw::records::PatchDescriptionPacked::getDemandedMeshWidth() const  {
      return _persistentRecords._demandedMeshWidth;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setDemandedMeshWidth(const double& demandedMeshWidth)  {
      _persistentRecords._demandedMeshWidth = demandedMeshWidth;
   }
   
   
   
    int peanoclaw::records::PatchDescriptionPacked::getCellDescriptionIndex() const  {
      return _persistentRecords._cellDescriptionIndex;
   }
   
   
   
    void peanoclaw::records::PatchDescriptionPacked::setCellDescriptionIndex(const int& cellDescriptionIndex)  {
      _persistentRecords._cellDescriptionIndex = cellDescriptionIndex;
   }
   
   
   
   
   std::string peanoclaw::records::PatchDescriptionPacked::toString() const {
      std::ostringstream stringstr;
      toString(stringstr);
      return stringstr.str();
   }
   
   void peanoclaw::records::PatchDescriptionPacked::toString (std::ostream& out) const {
      out << "("; 
      out << "isReferenced:" << getIsReferenced();
      out << ",";
      out << "adjacentRanks:[";
   for (int i = 0; i < THREE_POWER_D-1; i++) {
      out << getAdjacentRanks(i) << ",";
   }
   out << getAdjacentRanks(THREE_POWER_D-1) << "]";
      out << ",";
      out << "rank:" << getRank();
      out << ",";
      out << "subdivisionFactor:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getSubdivisionFactor(i) << ",";
   }
   out << getSubdivisionFactor(DIMENSIONS-1) << "]";
      out << ",";
      out << "ghostLayerWidth:" << getGhostLayerWidth();
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
      out << "hash:" << getHash();
      out << ",";
      out << "skipGridIterations:" << getSkipGridIterations();
      out << ",";
      out << "demandedMeshWidth:" << getDemandedMeshWidth();
      out << ",";
      out << "cellDescriptionIndex:" << getCellDescriptionIndex();
      out <<  ")";
   }
   
   
   peanoclaw::records::PatchDescriptionPacked::PersistentRecords peanoclaw::records::PatchDescriptionPacked::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::PatchDescription peanoclaw::records::PatchDescriptionPacked::convert() const{
      return PatchDescription(
         getIsReferenced(),
         getAdjacentRanks(),
         getRank(),
         getSubdivisionFactor(),
         getGhostLayerWidth(),
         getLevel(),
         getIsVirtual(),
         getPosition(),
         getSize(),
         getTime(),
         getTimestepSize(),
         getHash(),
         getSkipGridIterations(),
         getDemandedMeshWidth(),
         getCellDescriptionIndex()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::PatchDescriptionPacked::_log( "peanoclaw::records::PatchDescriptionPacked" );
      
      MPI_Datatype peanoclaw::records::PatchDescriptionPacked::Datatype = 0;
      MPI_Datatype peanoclaw::records::PatchDescriptionPacked::FullDatatype = 0;
      
      
      void peanoclaw::records::PatchDescriptionPacked::initDatatype() {
         {
            PatchDescriptionPacked dummyPatchDescriptionPacked[2];
            
            const int Attributes = 16;
            MPI_Datatype subtypes[Attributes] = {
               MPI_CHAR,		 //isReferenced
               MPI_INT,		 //adjacentRanks
               MPI_INT,		 //rank
               MPI_INT,		 //subdivisionFactor
               MPI_INT,		 //ghostLayerWidth
               MPI_INT,		 //level
               MPI_CHAR,		 //isVirtual
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
               MPI_DOUBLE,		 //hash
               MPI_INT,		 //skipGridIterations
               MPI_DOUBLE,		 //demandedMeshWidth
               MPI_INT,		 //cellDescriptionIndex
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //isReferenced
               THREE_POWER_D,		 //adjacentRanks
               1,		 //rank
               DIMENSIONS,		 //subdivisionFactor
               1,		 //ghostLayerWidth
               1,		 //level
               1,		 //isVirtual
               DIMENSIONS,		 //position
               DIMENSIONS,		 //size
               1,		 //time
               1,		 //timestepSize
               1,		 //hash
               1,		 //skipGridIterations
               1,		 //demandedMeshWidth
               1,		 //cellDescriptionIndex
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._isReferenced))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._adjacentRanks[0]))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._rank))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._ghostLayerWidth))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._level))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._isVirtual))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._position[0]))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._size[0]))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._time))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._timestepSize))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._hash))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._skipGridIterations))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._demandedMeshWidth))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[1]._persistentRecords._isReferenced))), 		&disp[15] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &PatchDescriptionPacked::Datatype );
            MPI_Type_commit( &PatchDescriptionPacked::Datatype );
            
         }
         {
            PatchDescriptionPacked dummyPatchDescriptionPacked[2];
            
            const int Attributes = 16;
            MPI_Datatype subtypes[Attributes] = {
               MPI_CHAR,		 //isReferenced
               MPI_INT,		 //adjacentRanks
               MPI_INT,		 //rank
               MPI_INT,		 //subdivisionFactor
               MPI_INT,		 //ghostLayerWidth
               MPI_INT,		 //level
               MPI_CHAR,		 //isVirtual
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
               MPI_DOUBLE,		 //hash
               MPI_INT,		 //skipGridIterations
               MPI_DOUBLE,		 //demandedMeshWidth
               MPI_INT,		 //cellDescriptionIndex
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //isReferenced
               THREE_POWER_D,		 //adjacentRanks
               1,		 //rank
               DIMENSIONS,		 //subdivisionFactor
               1,		 //ghostLayerWidth
               1,		 //level
               1,		 //isVirtual
               DIMENSIONS,		 //position
               DIMENSIONS,		 //size
               1,		 //time
               1,		 //timestepSize
               1,		 //hash
               1,		 //skipGridIterations
               1,		 //demandedMeshWidth
               1,		 //cellDescriptionIndex
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._isReferenced))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._adjacentRanks[0]))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._rank))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._subdivisionFactor[0]))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._ghostLayerWidth))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._level))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._isVirtual))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._position[0]))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._size[0]))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._time))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._timestepSize))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._hash))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._skipGridIterations))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._demandedMeshWidth))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[1]._persistentRecords._isReferenced))), 		&disp[15] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &PatchDescriptionPacked::FullDatatype );
            MPI_Type_commit( &PatchDescriptionPacked::FullDatatype );
            
         }
         
      }
      
      
      void peanoclaw::records::PatchDescriptionPacked::shutdownDatatype() {
         MPI_Type_free( &PatchDescriptionPacked::Datatype );
         MPI_Type_free( &PatchDescriptionPacked::FullDatatype );
         
      }
      
      void peanoclaw::records::PatchDescriptionPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "was not able to send message peanoclaw::records::PatchDescriptionPacked "
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
               msg << "testing for finished send task for peanoclaw::records::PatchDescriptionPacked "
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
               "peanoclaw::records::PatchDescriptionPacked",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::PatchDescriptionPacked",
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
      
      
      
      void peanoclaw::records::PatchDescriptionPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "failed to start to receive peanoclaw::records::PatchDescriptionPacked from node "
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
               msg << "testing for finished receive task for peanoclaw::records::PatchDescriptionPacked failed: "
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
               "peanoclaw::records::PatchDescriptionPacked",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::PatchDescriptionPacked",
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
      
      
      
      bool peanoclaw::records::PatchDescriptionPacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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


