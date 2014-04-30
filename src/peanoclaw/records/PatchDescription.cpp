#include "peanoclaw/records/PatchDescription.h"

#if defined(Parallel)
   peanoclaw::records::PatchDescription::PersistentRecords::PersistentRecords() {
      
   }
   
   
   peanoclaw::records::PatchDescription::PersistentRecords::PersistentRecords(const bool& isReferenced, const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks, const int& rank, const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& level, const bool& isVirtual, const bool& isRemote, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const int& skipGridIterations, const double& demandedMeshWidth, const int& cellDescriptionIndex):
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
   _skipGridIterations(skipGridIterations),
   _demandedMeshWidth(demandedMeshWidth),
   _cellDescriptionIndex(cellDescriptionIndex) {
      
   }
   
   peanoclaw::records::PatchDescription::PatchDescription() {
      
   }
   
   
   peanoclaw::records::PatchDescription::PatchDescription(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._isReferenced, persistentRecords._adjacentRanks, persistentRecords._rank, persistentRecords._subdivisionFactor, persistentRecords._ghostLayerWidth, persistentRecords._level, persistentRecords._isVirtual, persistentRecords._isRemote, persistentRecords._position, persistentRecords._size, persistentRecords._time, persistentRecords._timestepSize, persistentRecords._skipGridIterations, persistentRecords._demandedMeshWidth, persistentRecords._cellDescriptionIndex) {
      
   }
   
   
   peanoclaw::records::PatchDescription::PatchDescription(const bool& isReferenced, const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks, const int& rank, const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& level, const bool& isVirtual, const bool& isRemote, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const int& skipGridIterations, const double& demandedMeshWidth, const int& cellDescriptionIndex):
   _persistentRecords(isReferenced, adjacentRanks, rank, subdivisionFactor, ghostLayerWidth, level, isVirtual, isRemote, position, size, time, timestepSize, skipGridIterations, demandedMeshWidth, cellDescriptionIndex) {
      
   }
   
   
   peanoclaw::records::PatchDescription::~PatchDescription() { }
   
   
   
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
               MPI_CHAR,		 //isRemote
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
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
               MPI_CHAR,		 //isRemote
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
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
      
      void peanoclaw::records::PatchDescription::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
         if (communicateBlocking) {
         
            const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
            if  (result!=MPI_SUCCESS) {
               std::ostringstream msg;
               msg << "was not able to send message peanoclaw::records::PatchDescription "
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
         
      }
      
      
      
      void peanoclaw::records::PatchDescription::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
         if (communicateBlocking) {
         
            MPI_Status  status;
            const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
            if ( result != MPI_SUCCESS ) {
               std::ostringstream msg;
               msg << "failed to start to receive peanoclaw::records::PatchDescription from node "
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
   
   
   peanoclaw::records::PatchDescriptionPacked::PersistentRecords::PersistentRecords(const bool& isReferenced, const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks, const int& rank, const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& level, const bool& isVirtual, const bool& isRemote, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const int& skipGridIterations, const double& demandedMeshWidth, const int& cellDescriptionIndex):
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
   _skipGridIterations(skipGridIterations),
   _demandedMeshWidth(demandedMeshWidth),
   _cellDescriptionIndex(cellDescriptionIndex) {
      
   }
   
   peanoclaw::records::PatchDescriptionPacked::PatchDescriptionPacked() {
      
   }
   
   
   peanoclaw::records::PatchDescriptionPacked::PatchDescriptionPacked(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._isReferenced, persistentRecords._adjacentRanks, persistentRecords._rank, persistentRecords._subdivisionFactor, persistentRecords._ghostLayerWidth, persistentRecords._level, persistentRecords._isVirtual, persistentRecords._isRemote, persistentRecords._position, persistentRecords._size, persistentRecords._time, persistentRecords._timestepSize, persistentRecords._skipGridIterations, persistentRecords._demandedMeshWidth, persistentRecords._cellDescriptionIndex) {
      
   }
   
   
   peanoclaw::records::PatchDescriptionPacked::PatchDescriptionPacked(const bool& isReferenced, const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks, const int& rank, const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& level, const bool& isVirtual, const bool& isRemote, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const int& skipGridIterations, const double& demandedMeshWidth, const int& cellDescriptionIndex):
   _persistentRecords(isReferenced, adjacentRanks, rank, subdivisionFactor, ghostLayerWidth, level, isVirtual, isRemote, position, size, time, timestepSize, skipGridIterations, demandedMeshWidth, cellDescriptionIndex) {
      
   }
   
   
   peanoclaw::records::PatchDescriptionPacked::~PatchDescriptionPacked() { }
   
   
   
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
               MPI_CHAR,		 //isRemote
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
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
               MPI_CHAR,		 //isRemote
               MPI_DOUBLE,		 //position
               MPI_DOUBLE,		 //size
               MPI_DOUBLE,		 //time
               MPI_DOUBLE,		 //timestepSize
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
      
      void peanoclaw::records::PatchDescriptionPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
         if (communicateBlocking) {
         
            const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
            if  (result!=MPI_SUCCESS) {
               std::ostringstream msg;
               msg << "was not able to send message peanoclaw::records::PatchDescriptionPacked "
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
         
      }
      
      
      
      void peanoclaw::records::PatchDescriptionPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
         if (communicateBlocking) {
         
            MPI_Status  status;
            const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
            if ( result != MPI_SUCCESS ) {
               std::ostringstream msg;
               msg << "failed to start to receive peanoclaw::records::PatchDescriptionPacked from node "
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
   
   
   peanoclaw::records::PatchDescription::PersistentRecords::PersistentRecords(const bool& isReferenced, const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks, const int& rank, const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& level, const bool& isVirtual, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const int& skipGridIterations, const double& demandedMeshWidth, const int& cellDescriptionIndex):
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
   _skipGridIterations(skipGridIterations),
   _demandedMeshWidth(demandedMeshWidth),
   _cellDescriptionIndex(cellDescriptionIndex) {
      
   }
   
   peanoclaw::records::PatchDescription::PatchDescription() {
      
   }
   
   
   peanoclaw::records::PatchDescription::PatchDescription(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._isReferenced, persistentRecords._adjacentRanks, persistentRecords._rank, persistentRecords._subdivisionFactor, persistentRecords._ghostLayerWidth, persistentRecords._level, persistentRecords._isVirtual, persistentRecords._position, persistentRecords._size, persistentRecords._time, persistentRecords._timestepSize, persistentRecords._skipGridIterations, persistentRecords._demandedMeshWidth, persistentRecords._cellDescriptionIndex) {
      
   }
   
   
   peanoclaw::records::PatchDescription::PatchDescription(const bool& isReferenced, const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks, const int& rank, const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& level, const bool& isVirtual, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const int& skipGridIterations, const double& demandedMeshWidth, const int& cellDescriptionIndex):
   _persistentRecords(isReferenced, adjacentRanks, rank, subdivisionFactor, ghostLayerWidth, level, isVirtual, position, size, time, timestepSize, skipGridIterations, demandedMeshWidth, cellDescriptionIndex) {
      
   }
   
   
   peanoclaw::records::PatchDescription::~PatchDescription() { }
   
   
   
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
            
            const int Attributes = 15;
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
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._skipGridIterations))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._demandedMeshWidth))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._cellDescriptionIndex))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[1]._persistentRecords._isReferenced))), 		&disp[14] );
            
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
            
            const int Attributes = 15;
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
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._skipGridIterations))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._demandedMeshWidth))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[0]._persistentRecords._cellDescriptionIndex))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescription[1]._persistentRecords._isReferenced))), 		&disp[14] );
            
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
      
      void peanoclaw::records::PatchDescription::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
         if (communicateBlocking) {
         
            const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
            if  (result!=MPI_SUCCESS) {
               std::ostringstream msg;
               msg << "was not able to send message peanoclaw::records::PatchDescription "
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
         
      }
      
      
      
      void peanoclaw::records::PatchDescription::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
         if (communicateBlocking) {
         
            MPI_Status  status;
            const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
            if ( result != MPI_SUCCESS ) {
               std::ostringstream msg;
               msg << "failed to start to receive peanoclaw::records::PatchDescription from node "
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
   
   
   peanoclaw::records::PatchDescriptionPacked::PersistentRecords::PersistentRecords(const bool& isReferenced, const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks, const int& rank, const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& level, const bool& isVirtual, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const int& skipGridIterations, const double& demandedMeshWidth, const int& cellDescriptionIndex):
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
   _skipGridIterations(skipGridIterations),
   _demandedMeshWidth(demandedMeshWidth),
   _cellDescriptionIndex(cellDescriptionIndex) {
      
   }
   
   peanoclaw::records::PatchDescriptionPacked::PatchDescriptionPacked() {
      
   }
   
   
   peanoclaw::records::PatchDescriptionPacked::PatchDescriptionPacked(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._isReferenced, persistentRecords._adjacentRanks, persistentRecords._rank, persistentRecords._subdivisionFactor, persistentRecords._ghostLayerWidth, persistentRecords._level, persistentRecords._isVirtual, persistentRecords._position, persistentRecords._size, persistentRecords._time, persistentRecords._timestepSize, persistentRecords._skipGridIterations, persistentRecords._demandedMeshWidth, persistentRecords._cellDescriptionIndex) {
      
   }
   
   
   peanoclaw::records::PatchDescriptionPacked::PatchDescriptionPacked(const bool& isReferenced, const tarch::la::Vector<THREE_POWER_D,int>& adjacentRanks, const int& rank, const tarch::la::Vector<DIMENSIONS,int>& subdivisionFactor, const int& ghostLayerWidth, const int& level, const bool& isVirtual, const tarch::la::Vector<DIMENSIONS,double>& position, const tarch::la::Vector<DIMENSIONS,double>& size, const double& time, const double& timestepSize, const int& skipGridIterations, const double& demandedMeshWidth, const int& cellDescriptionIndex):
   _persistentRecords(isReferenced, adjacentRanks, rank, subdivisionFactor, ghostLayerWidth, level, isVirtual, position, size, time, timestepSize, skipGridIterations, demandedMeshWidth, cellDescriptionIndex) {
      
   }
   
   
   peanoclaw::records::PatchDescriptionPacked::~PatchDescriptionPacked() { }
   
   
   
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
            
            const int Attributes = 15;
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
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._skipGridIterations))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._demandedMeshWidth))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[1]._persistentRecords._isReferenced))), 		&disp[14] );
            
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
            
            const int Attributes = 15;
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
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._skipGridIterations))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._demandedMeshWidth))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyPatchDescriptionPacked[1]._persistentRecords._isReferenced))), 		&disp[14] );
            
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
      
      void peanoclaw::records::PatchDescriptionPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
         if (communicateBlocking) {
         
            const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
            if  (result!=MPI_SUCCESS) {
               std::ostringstream msg;
               msg << "was not able to send message peanoclaw::records::PatchDescriptionPacked "
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
         
      }
      
      
      
      void peanoclaw::records::PatchDescriptionPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
         if (communicateBlocking) {
         
            MPI_Status  status;
            const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
            if ( result != MPI_SUCCESS ) {
               std::ostringstream msg;
               msg << "failed to start to receive peanoclaw::records::PatchDescriptionPacked from node "
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


