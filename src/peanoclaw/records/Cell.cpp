#include "peanoclaw/records/Cell.h"

#if !defined(Debug) && !defined(Parallel) && defined(SharedMemoryParallelisation)
   peanoclaw::records::Cell::PersistentRecords::PersistentRecords() {
      
   }
   
   
   peanoclaw::records::Cell::PersistentRecords::PersistentRecords(const int& cellDescriptionIndex, const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream):
   _cellDescriptionIndex(cellDescriptionIndex),
   _isInside(isInside),
   _state(state),
   _evenFlags(evenFlags),
   _accessNumber(accessNumber),
   _numberOfLoadsFromInputStream(numberOfLoadsFromInputStream),
   _numberOfStoresToOutputStream(numberOfStoresToOutputStream) {
      
   }
   
   peanoclaw::records::Cell::Cell() {
      
   }
   
   
   peanoclaw::records::Cell::Cell(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._cellDescriptionIndex, persistentRecords._isInside, persistentRecords._state, persistentRecords._evenFlags, persistentRecords._accessNumber, persistentRecords._numberOfLoadsFromInputStream, persistentRecords._numberOfStoresToOutputStream) {
      
   }
   
   
   peanoclaw::records::Cell::Cell(const int& cellDescriptionIndex, const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream):
   _persistentRecords(cellDescriptionIndex, isInside, state, evenFlags, accessNumber, numberOfLoadsFromInputStream, numberOfStoresToOutputStream) {
      
   }
   
   
   peanoclaw::records::Cell::~Cell() { }
   
   std::string peanoclaw::records::Cell::toString(const State& param) {
      switch (param) {
         case Leaf: return "Leaf";
         case Refined: return "Refined";
         case Root: return "Root";
      }
      return "undefined";
   }
   
   std::string peanoclaw::records::Cell::getStateMapping() {
      return "State(Leaf=0,Refined=1,Root=2)";
   }
   
   
   std::string peanoclaw::records::Cell::toString() const {
      std::ostringstream stringstr;
      toString(stringstr);
      return stringstr.str();
   }
   
   void peanoclaw::records::Cell::toString (std::ostream& out) const {
      out << "("; 
      out << "cellDescriptionIndex:" << getCellDescriptionIndex();
      out << ",";
      out << "isInside:" << getIsInside();
      out << ",";
      out << "state:" << toString(getState());
      out << ",";
      out << "evenFlags:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getEvenFlags(i) << ",";
   }
   out << getEvenFlags(DIMENSIONS-1) << "]";
      out << ",";
      out << "accessNumber:[";
   for (int i = 0; i < DIMENSIONS_TIMES_TWO-1; i++) {
      out << getAccessNumber(i) << ",";
   }
   out << getAccessNumber(DIMENSIONS_TIMES_TWO-1) << "]";
      out << ",";
      out << "numberOfLoadsFromInputStream:" << getNumberOfLoadsFromInputStream();
      out << ",";
      out << "numberOfStoresToOutputStream:" << getNumberOfStoresToOutputStream();
      out <<  ")";
   }
   
   
   peanoclaw::records::Cell::PersistentRecords peanoclaw::records::Cell::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::CellPacked peanoclaw::records::Cell::convert() const{
      return CellPacked(
         getCellDescriptionIndex(),
         getIsInside(),
         getState(),
         getEvenFlags(),
         getAccessNumber(),
         getNumberOfLoadsFromInputStream(),
         getNumberOfStoresToOutputStream()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::Cell::_log( "peanoclaw::records::Cell" );
      
      MPI_Datatype peanoclaw::records::Cell::Datatype = 0;
      MPI_Datatype peanoclaw::records::Cell::FullDatatype = 0;
      
      
      void peanoclaw::records::Cell::initDatatype() {
         {
            Cell dummyCell[2];
            
            const int Attributes = 4;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //cellDescriptionIndex
               MPI_CHAR,		 //isInside
               MPI_INT,		 //state
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //cellDescriptionIndex
               1,		 //isInside
               1,		 //state
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._isInside))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._state))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[1]._persistentRecords._cellDescriptionIndex))), 		&disp[3] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Cell::Datatype );
            MPI_Type_commit( &Cell::Datatype );
            
         }
         {
            Cell dummyCell[2];
            
            const int Attributes = 8;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //cellDescriptionIndex
               MPI_CHAR,		 //isInside
               MPI_INT,		 //state
               MPI_INT,		 //evenFlags
               MPI_SHORT,		 //accessNumber
               MPI_INT,		 //numberOfLoadsFromInputStream
               MPI_INT,		 //numberOfStoresToOutputStream
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //cellDescriptionIndex
               1,		 //isInside
               1,		 //state
               DIMENSIONS,		 //evenFlags
               DIMENSIONS_TIMES_TWO,		 //accessNumber
               1,		 //numberOfLoadsFromInputStream
               1,		 //numberOfStoresToOutputStream
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._isInside))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._state))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._evenFlags))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._accessNumber[0]))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._numberOfLoadsFromInputStream))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._numberOfStoresToOutputStream))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[1]._persistentRecords._cellDescriptionIndex))), 		&disp[7] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Cell::FullDatatype );
            MPI_Type_commit( &Cell::FullDatatype );
            
         }
         
      }
      
      
      void peanoclaw::records::Cell::shutdownDatatype() {
         MPI_Type_free( &Cell::Datatype );
         MPI_Type_free( &Cell::FullDatatype );
         
      }
      
      void peanoclaw::records::Cell::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "was not able to send message peanoclaw::records::Cell "
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
               msg << "testing for finished send task for peanoclaw::records::Cell "
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
               "peanoclaw::records::Cell",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::Cell",
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
      
      
      
      void peanoclaw::records::Cell::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "failed to start to receive peanoclaw::records::Cell from node "
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
               msg << "testing for finished receive task for peanoclaw::records::Cell failed: "
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
               "peanoclaw::records::Cell",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::Cell",
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
      
      
      
      bool peanoclaw::records::Cell::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
      
      int peanoclaw::records::Cell::getSenderRank() const {
         assertion( _senderDestinationRank!=-1 );
         return _senderDestinationRank;
         
      }
   #endif
   
   
   peanoclaw::records::CellPacked::PersistentRecords::PersistentRecords() {
      assertion((DIMENSIONS+3 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::CellPacked::PersistentRecords::PersistentRecords(const int& cellDescriptionIndex, const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream):
   _cellDescriptionIndex(cellDescriptionIndex),
   _accessNumber(accessNumber),
   _numberOfLoadsFromInputStream(numberOfLoadsFromInputStream),
   _numberOfStoresToOutputStream(numberOfStoresToOutputStream) {
      setIsInside(isInside);
      setState(state);
      setEvenFlags(evenFlags);
      assertion((DIMENSIONS+3 < (8 * sizeof(short int))));
      
   }
   
   peanoclaw::records::CellPacked::CellPacked() {
      assertion((DIMENSIONS+3 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::CellPacked::CellPacked(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._cellDescriptionIndex, persistentRecords.getIsInside(), persistentRecords.getState(), persistentRecords.getEvenFlags(), persistentRecords._accessNumber, persistentRecords._numberOfLoadsFromInputStream, persistentRecords._numberOfStoresToOutputStream) {
      assertion((DIMENSIONS+3 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::CellPacked::CellPacked(const int& cellDescriptionIndex, const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream):
   _persistentRecords(cellDescriptionIndex, isInside, state, evenFlags, accessNumber, numberOfLoadsFromInputStream, numberOfStoresToOutputStream) {
      assertion((DIMENSIONS+3 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::CellPacked::~CellPacked() { }
   
   std::string peanoclaw::records::CellPacked::toString(const State& param) {
      return peanoclaw::records::Cell::toString(param);
   }
   
   std::string peanoclaw::records::CellPacked::getStateMapping() {
      return peanoclaw::records::Cell::getStateMapping();
   }
   
   
   
   std::string peanoclaw::records::CellPacked::toString() const {
      std::ostringstream stringstr;
      toString(stringstr);
      return stringstr.str();
   }
   
   void peanoclaw::records::CellPacked::toString (std::ostream& out) const {
      out << "("; 
      out << "cellDescriptionIndex:" << getCellDescriptionIndex();
      out << ",";
      out << "isInside:" << getIsInside();
      out << ",";
      out << "state:" << toString(getState());
      out << ",";
      out << "evenFlags:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getEvenFlags(i) << ",";
   }
   out << getEvenFlags(DIMENSIONS-1) << "]";
      out << ",";
      out << "accessNumber:[";
   for (int i = 0; i < DIMENSIONS_TIMES_TWO-1; i++) {
      out << getAccessNumber(i) << ",";
   }
   out << getAccessNumber(DIMENSIONS_TIMES_TWO-1) << "]";
      out << ",";
      out << "numberOfLoadsFromInputStream:" << getNumberOfLoadsFromInputStream();
      out << ",";
      out << "numberOfStoresToOutputStream:" << getNumberOfStoresToOutputStream();
      out <<  ")";
   }
   
   
   peanoclaw::records::CellPacked::PersistentRecords peanoclaw::records::CellPacked::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::Cell peanoclaw::records::CellPacked::convert() const{
      return Cell(
         getCellDescriptionIndex(),
         getIsInside(),
         getState(),
         getEvenFlags(),
         getAccessNumber(),
         getNumberOfLoadsFromInputStream(),
         getNumberOfStoresToOutputStream()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::CellPacked::_log( "peanoclaw::records::CellPacked" );
      
      MPI_Datatype peanoclaw::records::CellPacked::Datatype = 0;
      MPI_Datatype peanoclaw::records::CellPacked::FullDatatype = 0;
      
      
      void peanoclaw::records::CellPacked::initDatatype() {
         {
            CellPacked dummyCellPacked[2];
            
            const int Attributes = 3;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //cellDescriptionIndex
               MPI_SHORT,		 //_packedRecords0
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //cellDescriptionIndex
               1,		 //_packedRecords0
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._packedRecords0))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[1]._persistentRecords._cellDescriptionIndex))), 		&disp[2] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellPacked::Datatype );
            MPI_Type_commit( &CellPacked::Datatype );
            
         }
         {
            CellPacked dummyCellPacked[2];
            
            const int Attributes = 6;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //cellDescriptionIndex
               MPI_SHORT,		 //accessNumber
               MPI_INT,		 //numberOfLoadsFromInputStream
               MPI_INT,		 //numberOfStoresToOutputStream
               MPI_SHORT,		 //_packedRecords0
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //cellDescriptionIndex
               DIMENSIONS_TIMES_TWO,		 //accessNumber
               1,		 //numberOfLoadsFromInputStream
               1,		 //numberOfStoresToOutputStream
               1,		 //_packedRecords0
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._accessNumber[0]))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._numberOfLoadsFromInputStream))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._numberOfStoresToOutputStream))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._packedRecords0))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[1]._persistentRecords._cellDescriptionIndex))), 		&disp[5] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellPacked::FullDatatype );
            MPI_Type_commit( &CellPacked::FullDatatype );
            
         }
         
      }
      
      
      void peanoclaw::records::CellPacked::shutdownDatatype() {
         MPI_Type_free( &CellPacked::Datatype );
         MPI_Type_free( &CellPacked::FullDatatype );
         
      }
      
      void peanoclaw::records::CellPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "was not able to send message peanoclaw::records::CellPacked "
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
               msg << "testing for finished send task for peanoclaw::records::CellPacked "
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
               "peanoclaw::records::CellPacked",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::CellPacked",
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
      
      
      
      void peanoclaw::records::CellPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "failed to start to receive peanoclaw::records::CellPacked from node "
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
               msg << "testing for finished receive task for peanoclaw::records::CellPacked failed: "
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
               "peanoclaw::records::CellPacked",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::CellPacked",
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
      
      
      
      bool peanoclaw::records::CellPacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
      
      int peanoclaw::records::CellPacked::getSenderRank() const {
         assertion( _senderDestinationRank!=-1 );
         return _senderDestinationRank;
         
      }
   #endif
   
   
   
#elif !defined(Parallel) && defined(Debug) && !defined(SharedMemoryParallelisation)
   peanoclaw::records::Cell::PersistentRecords::PersistentRecords() {
      
   }
   
   
   peanoclaw::records::Cell::PersistentRecords::PersistentRecords(const int& cellDescriptionIndex, const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber):
   _cellDescriptionIndex(cellDescriptionIndex),
   _isInside(isInside),
   _state(state),
   _level(level),
   _evenFlags(evenFlags),
   _accessNumber(accessNumber) {
      
   }
   
   peanoclaw::records::Cell::Cell() {
      
   }
   
   
   peanoclaw::records::Cell::Cell(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._cellDescriptionIndex, persistentRecords._isInside, persistentRecords._state, persistentRecords._level, persistentRecords._evenFlags, persistentRecords._accessNumber) {
      
   }
   
   
   peanoclaw::records::Cell::Cell(const int& cellDescriptionIndex, const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber):
   _persistentRecords(cellDescriptionIndex, isInside, state, level, evenFlags, accessNumber) {
      
   }
   
   
   peanoclaw::records::Cell::~Cell() { }
   
   std::string peanoclaw::records::Cell::toString(const State& param) {
      switch (param) {
         case Leaf: return "Leaf";
         case Refined: return "Refined";
         case Root: return "Root";
      }
      return "undefined";
   }
   
   std::string peanoclaw::records::Cell::getStateMapping() {
      return "State(Leaf=0,Refined=1,Root=2)";
   }
   
   
   std::string peanoclaw::records::Cell::toString() const {
      std::ostringstream stringstr;
      toString(stringstr);
      return stringstr.str();
   }
   
   void peanoclaw::records::Cell::toString (std::ostream& out) const {
      out << "("; 
      out << "cellDescriptionIndex:" << getCellDescriptionIndex();
      out << ",";
      out << "isInside:" << getIsInside();
      out << ",";
      out << "state:" << toString(getState());
      out << ",";
      out << "level:" << getLevel();
      out << ",";
      out << "evenFlags:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getEvenFlags(i) << ",";
   }
   out << getEvenFlags(DIMENSIONS-1) << "]";
      out << ",";
      out << "accessNumber:[";
   for (int i = 0; i < DIMENSIONS_TIMES_TWO-1; i++) {
      out << getAccessNumber(i) << ",";
   }
   out << getAccessNumber(DIMENSIONS_TIMES_TWO-1) << "]";
      out <<  ")";
   }
   
   
   peanoclaw::records::Cell::PersistentRecords peanoclaw::records::Cell::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::CellPacked peanoclaw::records::Cell::convert() const{
      return CellPacked(
         getCellDescriptionIndex(),
         getIsInside(),
         getState(),
         getLevel(),
         getEvenFlags(),
         getAccessNumber()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::Cell::_log( "peanoclaw::records::Cell" );
      
      MPI_Datatype peanoclaw::records::Cell::Datatype = 0;
      MPI_Datatype peanoclaw::records::Cell::FullDatatype = 0;
      
      
      void peanoclaw::records::Cell::initDatatype() {
         {
            Cell dummyCell[2];
            
            const int Attributes = 5;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //cellDescriptionIndex
               MPI_CHAR,		 //isInside
               MPI_INT,		 //state
               MPI_INT,		 //level
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //cellDescriptionIndex
               1,		 //isInside
               1,		 //state
               1,		 //level
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._isInside))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._state))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._level))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[1]._persistentRecords._cellDescriptionIndex))), 		&disp[4] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Cell::Datatype );
            MPI_Type_commit( &Cell::Datatype );
            
         }
         {
            Cell dummyCell[2];
            
            const int Attributes = 7;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //cellDescriptionIndex
               MPI_CHAR,		 //isInside
               MPI_INT,		 //state
               MPI_INT,		 //level
               MPI_INT,		 //evenFlags
               MPI_SHORT,		 //accessNumber
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //cellDescriptionIndex
               1,		 //isInside
               1,		 //state
               1,		 //level
               DIMENSIONS,		 //evenFlags
               DIMENSIONS_TIMES_TWO,		 //accessNumber
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._isInside))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._state))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._level))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._evenFlags))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._accessNumber[0]))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[1]._persistentRecords._cellDescriptionIndex))), 		&disp[6] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Cell::FullDatatype );
            MPI_Type_commit( &Cell::FullDatatype );
            
         }
         
      }
      
      
      void peanoclaw::records::Cell::shutdownDatatype() {
         MPI_Type_free( &Cell::Datatype );
         MPI_Type_free( &Cell::FullDatatype );
         
      }
      
      void peanoclaw::records::Cell::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "was not able to send message peanoclaw::records::Cell "
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
               msg << "testing for finished send task for peanoclaw::records::Cell "
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
               "peanoclaw::records::Cell",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::Cell",
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
      
      
      
      void peanoclaw::records::Cell::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "failed to start to receive peanoclaw::records::Cell from node "
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
               msg << "testing for finished receive task for peanoclaw::records::Cell failed: "
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
               "peanoclaw::records::Cell",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::Cell",
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
      
      
      
      bool peanoclaw::records::Cell::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
      
      int peanoclaw::records::Cell::getSenderRank() const {
         assertion( _senderDestinationRank!=-1 );
         return _senderDestinationRank;
         
      }
   #endif
   
   
   peanoclaw::records::CellPacked::PersistentRecords::PersistentRecords() {
      assertion((DIMENSIONS+3 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::CellPacked::PersistentRecords::PersistentRecords(const int& cellDescriptionIndex, const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber):
   _cellDescriptionIndex(cellDescriptionIndex),
   _level(level),
   _accessNumber(accessNumber) {
      setIsInside(isInside);
      setState(state);
      setEvenFlags(evenFlags);
      assertion((DIMENSIONS+3 < (8 * sizeof(short int))));
      
   }
   
   peanoclaw::records::CellPacked::CellPacked() {
      assertion((DIMENSIONS+3 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::CellPacked::CellPacked(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._cellDescriptionIndex, persistentRecords.getIsInside(), persistentRecords.getState(), persistentRecords._level, persistentRecords.getEvenFlags(), persistentRecords._accessNumber) {
      assertion((DIMENSIONS+3 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::CellPacked::CellPacked(const int& cellDescriptionIndex, const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber):
   _persistentRecords(cellDescriptionIndex, isInside, state, level, evenFlags, accessNumber) {
      assertion((DIMENSIONS+3 < (8 * sizeof(short int))));
      
   }
   
   
   peanoclaw::records::CellPacked::~CellPacked() { }
   
   std::string peanoclaw::records::CellPacked::toString(const State& param) {
      return peanoclaw::records::Cell::toString(param);
   }
   
   std::string peanoclaw::records::CellPacked::getStateMapping() {
      return peanoclaw::records::Cell::getStateMapping();
   }
   
   
   
   std::string peanoclaw::records::CellPacked::toString() const {
      std::ostringstream stringstr;
      toString(stringstr);
      return stringstr.str();
   }
   
   void peanoclaw::records::CellPacked::toString (std::ostream& out) const {
      out << "("; 
      out << "cellDescriptionIndex:" << getCellDescriptionIndex();
      out << ",";
      out << "isInside:" << getIsInside();
      out << ",";
      out << "state:" << toString(getState());
      out << ",";
      out << "level:" << getLevel();
      out << ",";
      out << "evenFlags:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getEvenFlags(i) << ",";
   }
   out << getEvenFlags(DIMENSIONS-1) << "]";
      out << ",";
      out << "accessNumber:[";
   for (int i = 0; i < DIMENSIONS_TIMES_TWO-1; i++) {
      out << getAccessNumber(i) << ",";
   }
   out << getAccessNumber(DIMENSIONS_TIMES_TWO-1) << "]";
      out <<  ")";
   }
   
   
   peanoclaw::records::CellPacked::PersistentRecords peanoclaw::records::CellPacked::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::Cell peanoclaw::records::CellPacked::convert() const{
      return Cell(
         getCellDescriptionIndex(),
         getIsInside(),
         getState(),
         getLevel(),
         getEvenFlags(),
         getAccessNumber()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::CellPacked::_log( "peanoclaw::records::CellPacked" );
      
      MPI_Datatype peanoclaw::records::CellPacked::Datatype = 0;
      MPI_Datatype peanoclaw::records::CellPacked::FullDatatype = 0;
      
      
      void peanoclaw::records::CellPacked::initDatatype() {
         {
            CellPacked dummyCellPacked[2];
            
            const int Attributes = 4;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //cellDescriptionIndex
               MPI_INT,		 //level
               MPI_SHORT,		 //_packedRecords0
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //cellDescriptionIndex
               1,		 //level
               1,		 //_packedRecords0
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._level))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._packedRecords0))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[1]._persistentRecords._cellDescriptionIndex))), 		&disp[3] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellPacked::Datatype );
            MPI_Type_commit( &CellPacked::Datatype );
            
         }
         {
            CellPacked dummyCellPacked[2];
            
            const int Attributes = 5;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //cellDescriptionIndex
               MPI_INT,		 //level
               MPI_SHORT,		 //accessNumber
               MPI_SHORT,		 //_packedRecords0
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               1,		 //cellDescriptionIndex
               1,		 //level
               DIMENSIONS_TIMES_TWO,		 //accessNumber
               1,		 //_packedRecords0
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._level))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._accessNumber[0]))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._packedRecords0))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[1]._persistentRecords._cellDescriptionIndex))), 		&disp[4] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellPacked::FullDatatype );
            MPI_Type_commit( &CellPacked::FullDatatype );
            
         }
         
      }
      
      
      void peanoclaw::records::CellPacked::shutdownDatatype() {
         MPI_Type_free( &CellPacked::Datatype );
         MPI_Type_free( &CellPacked::FullDatatype );
         
      }
      
      void peanoclaw::records::CellPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "was not able to send message peanoclaw::records::CellPacked "
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
               msg << "testing for finished send task for peanoclaw::records::CellPacked "
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
               "peanoclaw::records::CellPacked",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::CellPacked",
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
      
      
      
      void peanoclaw::records::CellPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
            msg << "failed to start to receive peanoclaw::records::CellPacked from node "
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
               msg << "testing for finished receive task for peanoclaw::records::CellPacked failed: "
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
               "peanoclaw::records::CellPacked",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::CellPacked",
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
      
      
      
      bool peanoclaw::records::CellPacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
      
      int peanoclaw::records::CellPacked::getSenderRank() const {
         assertion( _senderDestinationRank!=-1 );
         return _senderDestinationRank;
         
      }
   #endif
   
   
   

#elif defined(Parallel) && !defined(Debug) && !defined(SharedMemoryParallelisation)
peanoclaw::records::Cell::PersistentRecords::PersistentRecords() {
   
}


peanoclaw::records::Cell::PersistentRecords::PersistentRecords(const int& cellDescriptionIndex, const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const double& nodeWorkload, const double& localWorkload, const double& totalWorkload, const bool& cellIsAForkCandidate):
_cellDescriptionIndex(cellDescriptionIndex),
_isInside(isInside),
_state(state),
_evenFlags(evenFlags),
_accessNumber(accessNumber),
_responsibleRank(responsibleRank),
_subtreeHoldsWorker(subtreeHoldsWorker),
_nodeWorkload(nodeWorkload),
_localWorkload(localWorkload),
_totalWorkload(totalWorkload),
_cellIsAForkCandidate(cellIsAForkCandidate) {
   
}

peanoclaw::records::Cell::Cell() {
   
}


peanoclaw::records::Cell::Cell(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._cellDescriptionIndex, persistentRecords._isInside, persistentRecords._state, persistentRecords._evenFlags, persistentRecords._accessNumber, persistentRecords._responsibleRank, persistentRecords._subtreeHoldsWorker, persistentRecords._nodeWorkload, persistentRecords._localWorkload, persistentRecords._totalWorkload, persistentRecords._cellIsAForkCandidate) {
   
}


peanoclaw::records::Cell::Cell(const int& cellDescriptionIndex, const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const double& nodeWorkload, const double& localWorkload, const double& totalWorkload, const bool& cellIsAForkCandidate):
_persistentRecords(cellDescriptionIndex, isInside, state, evenFlags, accessNumber, responsibleRank, subtreeHoldsWorker, nodeWorkload, localWorkload, totalWorkload, cellIsAForkCandidate) {
   
}


peanoclaw::records::Cell::~Cell() { }

std::string peanoclaw::records::Cell::toString(const State& param) {
   switch (param) {
      case Leaf: return "Leaf";
      case Refined: return "Refined";
      case Root: return "Root";
   }
   return "undefined";
}

std::string peanoclaw::records::Cell::getStateMapping() {
   return "State(Leaf=0,Refined=1,Root=2)";
}


std::string peanoclaw::records::Cell::toString() const {
   std::ostringstream stringstr;
   toString(stringstr);
   return stringstr.str();
}

void peanoclaw::records::Cell::toString (std::ostream& out) const {
   out << "("; 
   out << "cellDescriptionIndex:" << getCellDescriptionIndex();
   out << ",";
   out << "isInside:" << getIsInside();
   out << ",";
   out << "state:" << toString(getState());
   out << ",";
   out << "evenFlags:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getEvenFlags(i) << ",";
   }
   out << getEvenFlags(DIMENSIONS-1) << "]";
   out << ",";
   out << "accessNumber:[";
   for (int i = 0; i < DIMENSIONS_TIMES_TWO-1; i++) {
      out << getAccessNumber(i) << ",";
   }
   out << getAccessNumber(DIMENSIONS_TIMES_TWO-1) << "]";
   out << ",";
   out << "responsibleRank:" << getResponsibleRank();
   out << ",";
   out << "subtreeHoldsWorker:" << getSubtreeHoldsWorker();
   out << ",";
   out << "nodeWorkload:" << getNodeWorkload();
   out << ",";
   out << "localWorkload:" << getLocalWorkload();
   out << ",";
   out << "totalWorkload:" << getTotalWorkload();
   out << ",";
   out << "cellIsAForkCandidate:" << getCellIsAForkCandidate();
   out <<  ")";
}


peanoclaw::records::Cell::PersistentRecords peanoclaw::records::Cell::getPersistentRecords() const {
   return _persistentRecords;
}

peanoclaw::records::CellPacked peanoclaw::records::Cell::convert() const{
   return CellPacked(
      getCellDescriptionIndex(),
      getIsInside(),
      getState(),
      getEvenFlags(),
      getAccessNumber(),
      getResponsibleRank(),
      getSubtreeHoldsWorker(),
      getNodeWorkload(),
      getLocalWorkload(),
      getTotalWorkload(),
      getCellIsAForkCandidate()
   );
}

#ifdef Parallel
   tarch::logging::Log peanoclaw::records::Cell::_log( "peanoclaw::records::Cell" );
   
   MPI_Datatype peanoclaw::records::Cell::Datatype = 0;
   MPI_Datatype peanoclaw::records::Cell::FullDatatype = 0;
   
   
   void peanoclaw::records::Cell::initDatatype() {
      {
         Cell dummyCell[2];
         
         const int Attributes = 8;
         MPI_Datatype subtypes[Attributes] = {
            MPI_INT,		 //cellDescriptionIndex
            MPI_CHAR,		 //isInside
            MPI_INT,		 //state
            MPI_CHAR,		 //subtreeHoldsWorker
            MPI_DOUBLE,		 //nodeWorkload
            MPI_DOUBLE,		 //localWorkload
            MPI_DOUBLE,		 //totalWorkload
            MPI_UB		 // end/displacement flag
         };
         
         int blocklen[Attributes] = {
            1,		 //cellDescriptionIndex
            1,		 //isInside
            1,		 //state
            1,		 //subtreeHoldsWorker
            1,		 //nodeWorkload
            1,		 //localWorkload
            1,		 //totalWorkload
            1		 // end/displacement flag
         };
         
         MPI_Aint     disp[Attributes];
         
         MPI_Aint base;
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]))), &base);
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._isInside))), 		&disp[1] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._state))), 		&disp[2] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._subtreeHoldsWorker))), 		&disp[3] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._nodeWorkload))), 		&disp[4] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._localWorkload))), 		&disp[5] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._totalWorkload))), 		&disp[6] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[1]._persistentRecords._cellDescriptionIndex))), 		&disp[7] );
         
         for (int i=1; i<Attributes; i++) {
            assertion1( disp[i] > disp[i-1], i );
         }
         for (int i=0; i<Attributes; i++) {
            disp[i] -= base;
         }
         MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Cell::Datatype );
         MPI_Type_commit( &Cell::Datatype );
         
      }
      {
         Cell dummyCell[2];
         
         const int Attributes = 12;
         MPI_Datatype subtypes[Attributes] = {
            MPI_INT,		 //cellDescriptionIndex
            MPI_CHAR,		 //isInside
            MPI_INT,		 //state
            MPI_INT,		 //evenFlags
            MPI_SHORT,		 //accessNumber
            MPI_INT,		 //responsibleRank
            MPI_CHAR,		 //subtreeHoldsWorker
            MPI_DOUBLE,		 //nodeWorkload
            MPI_DOUBLE,		 //localWorkload
            MPI_DOUBLE,		 //totalWorkload
            MPI_CHAR,		 //cellIsAForkCandidate
            MPI_UB		 // end/displacement flag
         };
         
         int blocklen[Attributes] = {
            1,		 //cellDescriptionIndex
            1,		 //isInside
            1,		 //state
            DIMENSIONS,		 //evenFlags
            DIMENSIONS_TIMES_TWO,		 //accessNumber
            1,		 //responsibleRank
            1,		 //subtreeHoldsWorker
            1,		 //nodeWorkload
            1,		 //localWorkload
            1,		 //totalWorkload
            1,		 //cellIsAForkCandidate
            1		 // end/displacement flag
         };
         
         MPI_Aint     disp[Attributes];
         
         MPI_Aint base;
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]))), &base);
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._isInside))), 		&disp[1] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._state))), 		&disp[2] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._evenFlags))), 		&disp[3] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._accessNumber[0]))), 		&disp[4] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._responsibleRank))), 		&disp[5] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._subtreeHoldsWorker))), 		&disp[6] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._nodeWorkload))), 		&disp[7] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._localWorkload))), 		&disp[8] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._totalWorkload))), 		&disp[9] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellIsAForkCandidate))), 		&disp[10] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[1]._persistentRecords._cellDescriptionIndex))), 		&disp[11] );
         
         for (int i=1; i<Attributes; i++) {
            assertion1( disp[i] > disp[i-1], i );
         }
         for (int i=0; i<Attributes; i++) {
            disp[i] -= base;
         }
         MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Cell::FullDatatype );
         MPI_Type_commit( &Cell::FullDatatype );
         
      }
      
   }
   
   
   void peanoclaw::records::Cell::shutdownDatatype() {
      MPI_Type_free( &Cell::Datatype );
      MPI_Type_free( &Cell::FullDatatype );
      
   }
   
   void peanoclaw::records::Cell::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
         msg << "was not able to send message peanoclaw::records::Cell "
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
            msg << "testing for finished send task for peanoclaw::records::Cell "
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
            "peanoclaw::records::Cell",
            "send(int)", destination,tag,1
            );
            triggeredTimeoutWarning = true;
         }
         if (
            tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
            (clock()>timeOutShutdown)
         ) {
            tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
            "peanoclaw::records::Cell",
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
   
   
   
   void peanoclaw::records::Cell::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
         msg << "failed to start to receive peanoclaw::records::Cell from node "
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
            msg << "testing for finished receive task for peanoclaw::records::Cell failed: "
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
            "peanoclaw::records::Cell",
            "receive(int)", source,tag,1
            );
            triggeredTimeoutWarning = true;
         }
         if (
            tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
            (clock()>timeOutShutdown)
         ) {
            tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
            "peanoclaw::records::Cell",
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
   
   
   
   bool peanoclaw::records::Cell::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
   
   int peanoclaw::records::Cell::getSenderRank() const {
      assertion( _senderDestinationRank!=-1 );
      return _senderDestinationRank;
      
   }
#endif


peanoclaw::records::CellPacked::PersistentRecords::PersistentRecords() {
   assertion((DIMENSIONS+4 < (8 * sizeof(short int))));
   
}


peanoclaw::records::CellPacked::PersistentRecords::PersistentRecords(const int& cellDescriptionIndex, const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const double& nodeWorkload, const double& localWorkload, const double& totalWorkload, const bool& cellIsAForkCandidate):
_cellDescriptionIndex(cellDescriptionIndex),
_accessNumber(accessNumber),
_responsibleRank(responsibleRank),
_subtreeHoldsWorker(subtreeHoldsWorker),
_nodeWorkload(nodeWorkload),
_localWorkload(localWorkload),
_totalWorkload(totalWorkload) {
   setIsInside(isInside);
   setState(state);
   setEvenFlags(evenFlags);
   setCellIsAForkCandidate(cellIsAForkCandidate);
   assertion((DIMENSIONS+4 < (8 * sizeof(short int))));
   
}

peanoclaw::records::CellPacked::CellPacked() {
   assertion((DIMENSIONS+4 < (8 * sizeof(short int))));
   
}


peanoclaw::records::CellPacked::CellPacked(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._cellDescriptionIndex, persistentRecords.getIsInside(), persistentRecords.getState(), persistentRecords.getEvenFlags(), persistentRecords._accessNumber, persistentRecords._responsibleRank, persistentRecords._subtreeHoldsWorker, persistentRecords._nodeWorkload, persistentRecords._localWorkload, persistentRecords._totalWorkload, persistentRecords.getCellIsAForkCandidate()) {
   assertion((DIMENSIONS+4 < (8 * sizeof(short int))));
   
}


peanoclaw::records::CellPacked::CellPacked(const int& cellDescriptionIndex, const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const double& nodeWorkload, const double& localWorkload, const double& totalWorkload, const bool& cellIsAForkCandidate):
_persistentRecords(cellDescriptionIndex, isInside, state, evenFlags, accessNumber, responsibleRank, subtreeHoldsWorker, nodeWorkload, localWorkload, totalWorkload, cellIsAForkCandidate) {
   assertion((DIMENSIONS+4 < (8 * sizeof(short int))));
   
}


peanoclaw::records::CellPacked::~CellPacked() { }

std::string peanoclaw::records::CellPacked::toString(const State& param) {
   return peanoclaw::records::Cell::toString(param);
}

std::string peanoclaw::records::CellPacked::getStateMapping() {
   return peanoclaw::records::Cell::getStateMapping();
}



std::string peanoclaw::records::CellPacked::toString() const {
   std::ostringstream stringstr;
   toString(stringstr);
   return stringstr.str();
}

void peanoclaw::records::CellPacked::toString (std::ostream& out) const {
   out << "("; 
   out << "cellDescriptionIndex:" << getCellDescriptionIndex();
   out << ",";
   out << "isInside:" << getIsInside();
   out << ",";
   out << "state:" << toString(getState());
   out << ",";
   out << "evenFlags:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getEvenFlags(i) << ",";
   }
   out << getEvenFlags(DIMENSIONS-1) << "]";
   out << ",";
   out << "accessNumber:[";
   for (int i = 0; i < DIMENSIONS_TIMES_TWO-1; i++) {
      out << getAccessNumber(i) << ",";
   }
   out << getAccessNumber(DIMENSIONS_TIMES_TWO-1) << "]";
   out << ",";
   out << "responsibleRank:" << getResponsibleRank();
   out << ",";
   out << "subtreeHoldsWorker:" << getSubtreeHoldsWorker();
   out << ",";
   out << "nodeWorkload:" << getNodeWorkload();
   out << ",";
   out << "localWorkload:" << getLocalWorkload();
   out << ",";
   out << "totalWorkload:" << getTotalWorkload();
   out << ",";
   out << "cellIsAForkCandidate:" << getCellIsAForkCandidate();
   out <<  ")";
}


peanoclaw::records::CellPacked::PersistentRecords peanoclaw::records::CellPacked::getPersistentRecords() const {
   return _persistentRecords;
}

peanoclaw::records::Cell peanoclaw::records::CellPacked::convert() const{
   return Cell(
      getCellDescriptionIndex(),
      getIsInside(),
      getState(),
      getEvenFlags(),
      getAccessNumber(),
      getResponsibleRank(),
      getSubtreeHoldsWorker(),
      getNodeWorkload(),
      getLocalWorkload(),
      getTotalWorkload(),
      getCellIsAForkCandidate()
   );
}

#ifdef Parallel
   tarch::logging::Log peanoclaw::records::CellPacked::_log( "peanoclaw::records::CellPacked" );
   
   MPI_Datatype peanoclaw::records::CellPacked::Datatype = 0;
   MPI_Datatype peanoclaw::records::CellPacked::FullDatatype = 0;
   
   
   void peanoclaw::records::CellPacked::initDatatype() {
      {
         CellPacked dummyCellPacked[2];
         
         const int Attributes = 7;
         MPI_Datatype subtypes[Attributes] = {
            MPI_INT,		 //cellDescriptionIndex
            MPI_CHAR,		 //subtreeHoldsWorker
            MPI_DOUBLE,		 //nodeWorkload
            MPI_DOUBLE,		 //localWorkload
            MPI_DOUBLE,		 //totalWorkload
            MPI_SHORT,		 //_packedRecords0
            MPI_UB		 // end/displacement flag
         };
         
         int blocklen[Attributes] = {
            1,		 //cellDescriptionIndex
            1,		 //subtreeHoldsWorker
            1,		 //nodeWorkload
            1,		 //localWorkload
            1,		 //totalWorkload
            1,		 //_packedRecords0
            1		 // end/displacement flag
         };
         
         MPI_Aint     disp[Attributes];
         
         MPI_Aint base;
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]))), &base);
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._subtreeHoldsWorker))), 		&disp[1] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._nodeWorkload))), 		&disp[2] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._localWorkload))), 		&disp[3] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._totalWorkload))), 		&disp[4] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._packedRecords0))), 		&disp[5] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[1]._persistentRecords._cellDescriptionIndex))), 		&disp[6] );
         
         for (int i=1; i<Attributes; i++) {
            assertion1( disp[i] > disp[i-1], i );
         }
         for (int i=0; i<Attributes; i++) {
            disp[i] -= base;
         }
         MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellPacked::Datatype );
         MPI_Type_commit( &CellPacked::Datatype );
         
      }
      {
         CellPacked dummyCellPacked[2];
         
         const int Attributes = 9;
         MPI_Datatype subtypes[Attributes] = {
            MPI_INT,		 //cellDescriptionIndex
            MPI_SHORT,		 //accessNumber
            MPI_INT,		 //responsibleRank
            MPI_CHAR,		 //subtreeHoldsWorker
            MPI_DOUBLE,		 //nodeWorkload
            MPI_DOUBLE,		 //localWorkload
            MPI_DOUBLE,		 //totalWorkload
            MPI_SHORT,		 //_packedRecords0
            MPI_UB		 // end/displacement flag
         };
         
         int blocklen[Attributes] = {
            1,		 //cellDescriptionIndex
            DIMENSIONS_TIMES_TWO,		 //accessNumber
            1,		 //responsibleRank
            1,		 //subtreeHoldsWorker
            1,		 //nodeWorkload
            1,		 //localWorkload
            1,		 //totalWorkload
            1,		 //_packedRecords0
            1		 // end/displacement flag
         };
         
         MPI_Aint     disp[Attributes];
         
         MPI_Aint base;
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]))), &base);
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._accessNumber[0]))), 		&disp[1] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._responsibleRank))), 		&disp[2] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._subtreeHoldsWorker))), 		&disp[3] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._nodeWorkload))), 		&disp[4] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._localWorkload))), 		&disp[5] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._totalWorkload))), 		&disp[6] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._packedRecords0))), 		&disp[7] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[1]._persistentRecords._cellDescriptionIndex))), 		&disp[8] );
         
         for (int i=1; i<Attributes; i++) {
            assertion1( disp[i] > disp[i-1], i );
         }
         for (int i=0; i<Attributes; i++) {
            disp[i] -= base;
         }
         MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellPacked::FullDatatype );
         MPI_Type_commit( &CellPacked::FullDatatype );
         
      }
      
   }
   
   
   void peanoclaw::records::CellPacked::shutdownDatatype() {
      MPI_Type_free( &CellPacked::Datatype );
      MPI_Type_free( &CellPacked::FullDatatype );
      
   }
   
   void peanoclaw::records::CellPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
         msg << "was not able to send message peanoclaw::records::CellPacked "
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
            msg << "testing for finished send task for peanoclaw::records::CellPacked "
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
            "peanoclaw::records::CellPacked",
            "send(int)", destination,tag,1
            );
            triggeredTimeoutWarning = true;
         }
         if (
            tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
            (clock()>timeOutShutdown)
         ) {
            tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
            "peanoclaw::records::CellPacked",
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
   
   
   
   void peanoclaw::records::CellPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
         msg << "failed to start to receive peanoclaw::records::CellPacked from node "
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
            msg << "testing for finished receive task for peanoclaw::records::CellPacked failed: "
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
            "peanoclaw::records::CellPacked",
            "receive(int)", source,tag,1
            );
            triggeredTimeoutWarning = true;
         }
         if (
            tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
            (clock()>timeOutShutdown)
         ) {
            tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
            "peanoclaw::records::CellPacked",
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
   
   
   
   bool peanoclaw::records::CellPacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
   
   int peanoclaw::records::CellPacked::getSenderRank() const {
      assertion( _senderDestinationRank!=-1 );
      return _senderDestinationRank;
      
   }
#endif




#elif !defined(Debug) && !defined(Parallel) && !defined(SharedMemoryParallelisation)
peanoclaw::records::Cell::PersistentRecords::PersistentRecords() {

}


peanoclaw::records::Cell::PersistentRecords::PersistentRecords(const int& cellDescriptionIndex, const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber):
_cellDescriptionIndex(cellDescriptionIndex),
_isInside(isInside),
_state(state),
_evenFlags(evenFlags),
_accessNumber(accessNumber) {

}

peanoclaw::records::Cell::Cell() {

}


peanoclaw::records::Cell::Cell(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._cellDescriptionIndex, persistentRecords._isInside, persistentRecords._state, persistentRecords._evenFlags, persistentRecords._accessNumber) {

}


peanoclaw::records::Cell::Cell(const int& cellDescriptionIndex, const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber):
_persistentRecords(cellDescriptionIndex, isInside, state, evenFlags, accessNumber) {

}


peanoclaw::records::Cell::~Cell() { }

std::string peanoclaw::records::Cell::toString(const State& param) {
switch (param) {
   case Leaf: return "Leaf";
   case Refined: return "Refined";
   case Root: return "Root";
}
return "undefined";
}

std::string peanoclaw::records::Cell::getStateMapping() {
return "State(Leaf=0,Refined=1,Root=2)";
}


std::string peanoclaw::records::Cell::toString() const {
std::ostringstream stringstr;
toString(stringstr);
return stringstr.str();
}

void peanoclaw::records::Cell::toString (std::ostream& out) const {
out << "("; 
out << "cellDescriptionIndex:" << getCellDescriptionIndex();
out << ",";
out << "isInside:" << getIsInside();
out << ",";
out << "state:" << toString(getState());
out << ",";
out << "evenFlags:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getEvenFlags(i) << ",";
   }
   out << getEvenFlags(DIMENSIONS-1) << "]";
out << ",";
out << "accessNumber:[";
   for (int i = 0; i < DIMENSIONS_TIMES_TWO-1; i++) {
      out << getAccessNumber(i) << ",";
   }
   out << getAccessNumber(DIMENSIONS_TIMES_TWO-1) << "]";
out <<  ")";
}


peanoclaw::records::Cell::PersistentRecords peanoclaw::records::Cell::getPersistentRecords() const {
return _persistentRecords;
}

peanoclaw::records::CellPacked peanoclaw::records::Cell::convert() const{
return CellPacked(
   getCellDescriptionIndex(),
   getIsInside(),
   getState(),
   getEvenFlags(),
   getAccessNumber()
);
}

#ifdef Parallel
tarch::logging::Log peanoclaw::records::Cell::_log( "peanoclaw::records::Cell" );

MPI_Datatype peanoclaw::records::Cell::Datatype = 0;
MPI_Datatype peanoclaw::records::Cell::FullDatatype = 0;


void peanoclaw::records::Cell::initDatatype() {
   {
      Cell dummyCell[2];
      
      const int Attributes = 4;
      MPI_Datatype subtypes[Attributes] = {
         MPI_INT,		 //cellDescriptionIndex
         MPI_CHAR,		 //isInside
         MPI_INT,		 //state
         MPI_UB		 // end/displacement flag
      };
      
      int blocklen[Attributes] = {
         1,		 //cellDescriptionIndex
         1,		 //isInside
         1,		 //state
         1		 // end/displacement flag
      };
      
      MPI_Aint     disp[Attributes];
      
      MPI_Aint base;
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]))), &base);
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._isInside))), 		&disp[1] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._state))), 		&disp[2] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[1]._persistentRecords._cellDescriptionIndex))), 		&disp[3] );
      
      for (int i=1; i<Attributes; i++) {
         assertion1( disp[i] > disp[i-1], i );
      }
      for (int i=0; i<Attributes; i++) {
         disp[i] -= base;
      }
      MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Cell::Datatype );
      MPI_Type_commit( &Cell::Datatype );
      
   }
   {
      Cell dummyCell[2];
      
      const int Attributes = 6;
      MPI_Datatype subtypes[Attributes] = {
         MPI_INT,		 //cellDescriptionIndex
         MPI_CHAR,		 //isInside
         MPI_INT,		 //state
         MPI_INT,		 //evenFlags
         MPI_SHORT,		 //accessNumber
         MPI_UB		 // end/displacement flag
      };
      
      int blocklen[Attributes] = {
         1,		 //cellDescriptionIndex
         1,		 //isInside
         1,		 //state
         DIMENSIONS,		 //evenFlags
         DIMENSIONS_TIMES_TWO,		 //accessNumber
         1		 // end/displacement flag
      };
      
      MPI_Aint     disp[Attributes];
      
      MPI_Aint base;
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]))), &base);
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._isInside))), 		&disp[1] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._state))), 		&disp[2] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._evenFlags))), 		&disp[3] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._accessNumber[0]))), 		&disp[4] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[1]._persistentRecords._cellDescriptionIndex))), 		&disp[5] );
      
      for (int i=1; i<Attributes; i++) {
         assertion1( disp[i] > disp[i-1], i );
      }
      for (int i=0; i<Attributes; i++) {
         disp[i] -= base;
      }
      MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Cell::FullDatatype );
      MPI_Type_commit( &Cell::FullDatatype );
      
   }
   
}


void peanoclaw::records::Cell::shutdownDatatype() {
   MPI_Type_free( &Cell::Datatype );
   MPI_Type_free( &Cell::FullDatatype );
   
}

void peanoclaw::records::Cell::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
      msg << "was not able to send message peanoclaw::records::Cell "
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
         msg << "testing for finished send task for peanoclaw::records::Cell "
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
         "peanoclaw::records::Cell",
         "send(int)", destination,tag,1
         );
         triggeredTimeoutWarning = true;
      }
      if (
         tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
         (clock()>timeOutShutdown)
      ) {
         tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
         "peanoclaw::records::Cell",
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



void peanoclaw::records::Cell::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
      msg << "failed to start to receive peanoclaw::records::Cell from node "
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
         msg << "testing for finished receive task for peanoclaw::records::Cell failed: "
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
         "peanoclaw::records::Cell",
         "receive(int)", source,tag,1
         );
         triggeredTimeoutWarning = true;
      }
      if (
         tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
         (clock()>timeOutShutdown)
      ) {
         tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
         "peanoclaw::records::Cell",
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



bool peanoclaw::records::Cell::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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

int peanoclaw::records::Cell::getSenderRank() const {
   assertion( _senderDestinationRank!=-1 );
   return _senderDestinationRank;
   
}
#endif


peanoclaw::records::CellPacked::PersistentRecords::PersistentRecords() {
assertion((DIMENSIONS+3 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::PersistentRecords::PersistentRecords(const int& cellDescriptionIndex, const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber):
_cellDescriptionIndex(cellDescriptionIndex),
_accessNumber(accessNumber) {
setIsInside(isInside);
setState(state);
setEvenFlags(evenFlags);
assertion((DIMENSIONS+3 < (8 * sizeof(short int))));

}

peanoclaw::records::CellPacked::CellPacked() {
assertion((DIMENSIONS+3 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::CellPacked(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._cellDescriptionIndex, persistentRecords.getIsInside(), persistentRecords.getState(), persistentRecords.getEvenFlags(), persistentRecords._accessNumber) {
assertion((DIMENSIONS+3 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::CellPacked(const int& cellDescriptionIndex, const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber):
_persistentRecords(cellDescriptionIndex, isInside, state, evenFlags, accessNumber) {
assertion((DIMENSIONS+3 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::~CellPacked() { }

std::string peanoclaw::records::CellPacked::toString(const State& param) {
return peanoclaw::records::Cell::toString(param);
}

std::string peanoclaw::records::CellPacked::getStateMapping() {
return peanoclaw::records::Cell::getStateMapping();
}



std::string peanoclaw::records::CellPacked::toString() const {
std::ostringstream stringstr;
toString(stringstr);
return stringstr.str();
}

void peanoclaw::records::CellPacked::toString (std::ostream& out) const {
out << "("; 
out << "cellDescriptionIndex:" << getCellDescriptionIndex();
out << ",";
out << "isInside:" << getIsInside();
out << ",";
out << "state:" << toString(getState());
out << ",";
out << "evenFlags:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getEvenFlags(i) << ",";
   }
   out << getEvenFlags(DIMENSIONS-1) << "]";
out << ",";
out << "accessNumber:[";
   for (int i = 0; i < DIMENSIONS_TIMES_TWO-1; i++) {
      out << getAccessNumber(i) << ",";
   }
   out << getAccessNumber(DIMENSIONS_TIMES_TWO-1) << "]";
out <<  ")";
}


peanoclaw::records::CellPacked::PersistentRecords peanoclaw::records::CellPacked::getPersistentRecords() const {
return _persistentRecords;
}

peanoclaw::records::Cell peanoclaw::records::CellPacked::convert() const{
return Cell(
   getCellDescriptionIndex(),
   getIsInside(),
   getState(),
   getEvenFlags(),
   getAccessNumber()
);
}

#ifdef Parallel
tarch::logging::Log peanoclaw::records::CellPacked::_log( "peanoclaw::records::CellPacked" );

MPI_Datatype peanoclaw::records::CellPacked::Datatype = 0;
MPI_Datatype peanoclaw::records::CellPacked::FullDatatype = 0;


void peanoclaw::records::CellPacked::initDatatype() {
   {
      CellPacked dummyCellPacked[2];
      
      const int Attributes = 3;
      MPI_Datatype subtypes[Attributes] = {
         MPI_INT,		 //cellDescriptionIndex
         MPI_SHORT,		 //_packedRecords0
         MPI_UB		 // end/displacement flag
      };
      
      int blocklen[Attributes] = {
         1,		 //cellDescriptionIndex
         1,		 //_packedRecords0
         1		 // end/displacement flag
      };
      
      MPI_Aint     disp[Attributes];
      
      MPI_Aint base;
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]))), &base);
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._packedRecords0))), 		&disp[1] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[1]._persistentRecords._cellDescriptionIndex))), 		&disp[2] );
      
      for (int i=1; i<Attributes; i++) {
         assertion1( disp[i] > disp[i-1], i );
      }
      for (int i=0; i<Attributes; i++) {
         disp[i] -= base;
      }
      MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellPacked::Datatype );
      MPI_Type_commit( &CellPacked::Datatype );
      
   }
   {
      CellPacked dummyCellPacked[2];
      
      const int Attributes = 4;
      MPI_Datatype subtypes[Attributes] = {
         MPI_INT,		 //cellDescriptionIndex
         MPI_SHORT,		 //accessNumber
         MPI_SHORT,		 //_packedRecords0
         MPI_UB		 // end/displacement flag
      };
      
      int blocklen[Attributes] = {
         1,		 //cellDescriptionIndex
         DIMENSIONS_TIMES_TWO,		 //accessNumber
         1,		 //_packedRecords0
         1		 // end/displacement flag
      };
      
      MPI_Aint     disp[Attributes];
      
      MPI_Aint base;
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]))), &base);
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._accessNumber[0]))), 		&disp[1] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._packedRecords0))), 		&disp[2] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[1]._persistentRecords._cellDescriptionIndex))), 		&disp[3] );
      
      for (int i=1; i<Attributes; i++) {
         assertion1( disp[i] > disp[i-1], i );
      }
      for (int i=0; i<Attributes; i++) {
         disp[i] -= base;
      }
      MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellPacked::FullDatatype );
      MPI_Type_commit( &CellPacked::FullDatatype );
      
   }
   
}


void peanoclaw::records::CellPacked::shutdownDatatype() {
   MPI_Type_free( &CellPacked::Datatype );
   MPI_Type_free( &CellPacked::FullDatatype );
   
}

void peanoclaw::records::CellPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
      msg << "was not able to send message peanoclaw::records::CellPacked "
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
         msg << "testing for finished send task for peanoclaw::records::CellPacked "
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
         "peanoclaw::records::CellPacked",
         "send(int)", destination,tag,1
         );
         triggeredTimeoutWarning = true;
      }
      if (
         tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
         (clock()>timeOutShutdown)
      ) {
         tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
         "peanoclaw::records::CellPacked",
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



void peanoclaw::records::CellPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
      msg << "failed to start to receive peanoclaw::records::CellPacked from node "
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
         msg << "testing for finished receive task for peanoclaw::records::CellPacked failed: "
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
         "peanoclaw::records::CellPacked",
         "receive(int)", source,tag,1
         );
         triggeredTimeoutWarning = true;
      }
      if (
         tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
         (clock()>timeOutShutdown)
      ) {
         tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
         "peanoclaw::records::CellPacked",
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



bool peanoclaw::records::CellPacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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

int peanoclaw::records::CellPacked::getSenderRank() const {
   assertion( _senderDestinationRank!=-1 );
   return _senderDestinationRank;
   
}
#endif




#elif defined(Parallel) && defined(SharedMemoryParallelisation) && defined(Debug)
peanoclaw::records::Cell::PersistentRecords::PersistentRecords() {

}


peanoclaw::records::Cell::PersistentRecords::PersistentRecords(const int& cellDescriptionIndex, const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const double& nodeWorkload, const double& localWorkload, const double& totalWorkload, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream):
_cellDescriptionIndex(cellDescriptionIndex),
_isInside(isInside),
_state(state),
_level(level),
_evenFlags(evenFlags),
_accessNumber(accessNumber),
_responsibleRank(responsibleRank),
_subtreeHoldsWorker(subtreeHoldsWorker),
_nodeWorkload(nodeWorkload),
_localWorkload(localWorkload),
_totalWorkload(totalWorkload),
_cellIsAForkCandidate(cellIsAForkCandidate),
_numberOfLoadsFromInputStream(numberOfLoadsFromInputStream),
_numberOfStoresToOutputStream(numberOfStoresToOutputStream) {

}

peanoclaw::records::Cell::Cell() {

}


peanoclaw::records::Cell::Cell(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._cellDescriptionIndex, persistentRecords._isInside, persistentRecords._state, persistentRecords._level, persistentRecords._evenFlags, persistentRecords._accessNumber, persistentRecords._responsibleRank, persistentRecords._subtreeHoldsWorker, persistentRecords._nodeWorkload, persistentRecords._localWorkload, persistentRecords._totalWorkload, persistentRecords._cellIsAForkCandidate, persistentRecords._numberOfLoadsFromInputStream, persistentRecords._numberOfStoresToOutputStream) {

}


peanoclaw::records::Cell::Cell(const int& cellDescriptionIndex, const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const double& nodeWorkload, const double& localWorkload, const double& totalWorkload, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream):
_persistentRecords(cellDescriptionIndex, isInside, state, level, evenFlags, accessNumber, responsibleRank, subtreeHoldsWorker, nodeWorkload, localWorkload, totalWorkload, cellIsAForkCandidate, numberOfLoadsFromInputStream, numberOfStoresToOutputStream) {

}


peanoclaw::records::Cell::~Cell() { }

std::string peanoclaw::records::Cell::toString(const State& param) {
switch (param) {
case Leaf: return "Leaf";
case Refined: return "Refined";
case Root: return "Root";
}
return "undefined";
}

std::string peanoclaw::records::Cell::getStateMapping() {
return "State(Leaf=0,Refined=1,Root=2)";
}


std::string peanoclaw::records::Cell::toString() const {
std::ostringstream stringstr;
toString(stringstr);
return stringstr.str();
}

void peanoclaw::records::Cell::toString (std::ostream& out) const {
out << "("; 
out << "cellDescriptionIndex:" << getCellDescriptionIndex();
out << ",";
out << "isInside:" << getIsInside();
out << ",";
out << "state:" << toString(getState());
out << ",";
out << "level:" << getLevel();
out << ",";
out << "evenFlags:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getEvenFlags(i) << ",";
   }
   out << getEvenFlags(DIMENSIONS-1) << "]";
out << ",";
out << "accessNumber:[";
   for (int i = 0; i < DIMENSIONS_TIMES_TWO-1; i++) {
      out << getAccessNumber(i) << ",";
   }
   out << getAccessNumber(DIMENSIONS_TIMES_TWO-1) << "]";
out << ",";
out << "responsibleRank:" << getResponsibleRank();
out << ",";
out << "subtreeHoldsWorker:" << getSubtreeHoldsWorker();
out << ",";
out << "nodeWorkload:" << getNodeWorkload();
out << ",";
out << "localWorkload:" << getLocalWorkload();
out << ",";
out << "totalWorkload:" << getTotalWorkload();
out << ",";
out << "cellIsAForkCandidate:" << getCellIsAForkCandidate();
out << ",";
out << "numberOfLoadsFromInputStream:" << getNumberOfLoadsFromInputStream();
out << ",";
out << "numberOfStoresToOutputStream:" << getNumberOfStoresToOutputStream();
out <<  ")";
}


peanoclaw::records::Cell::PersistentRecords peanoclaw::records::Cell::getPersistentRecords() const {
return _persistentRecords;
}

peanoclaw::records::CellPacked peanoclaw::records::Cell::convert() const{
return CellPacked(
getCellDescriptionIndex(),
getIsInside(),
getState(),
getLevel(),
getEvenFlags(),
getAccessNumber(),
getResponsibleRank(),
getSubtreeHoldsWorker(),
getNodeWorkload(),
getLocalWorkload(),
getTotalWorkload(),
getCellIsAForkCandidate(),
getNumberOfLoadsFromInputStream(),
getNumberOfStoresToOutputStream()
);
}

#ifdef Parallel
tarch::logging::Log peanoclaw::records::Cell::_log( "peanoclaw::records::Cell" );

MPI_Datatype peanoclaw::records::Cell::Datatype = 0;
MPI_Datatype peanoclaw::records::Cell::FullDatatype = 0;


void peanoclaw::records::Cell::initDatatype() {
{
   Cell dummyCell[2];
   
   const int Attributes = 9;
   MPI_Datatype subtypes[Attributes] = {
      MPI_INT,		 //cellDescriptionIndex
      MPI_CHAR,		 //isInside
      MPI_INT,		 //state
      MPI_INT,		 //level
      MPI_CHAR,		 //subtreeHoldsWorker
      MPI_DOUBLE,		 //nodeWorkload
      MPI_DOUBLE,		 //localWorkload
      MPI_DOUBLE,		 //totalWorkload
      MPI_UB		 // end/displacement flag
   };
   
   int blocklen[Attributes] = {
      1,		 //cellDescriptionIndex
      1,		 //isInside
      1,		 //state
      1,		 //level
      1,		 //subtreeHoldsWorker
      1,		 //nodeWorkload
      1,		 //localWorkload
      1,		 //totalWorkload
      1		 // end/displacement flag
   };
   
   MPI_Aint     disp[Attributes];
   
   MPI_Aint base;
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]))), &base);
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._isInside))), 		&disp[1] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._state))), 		&disp[2] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._level))), 		&disp[3] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._subtreeHoldsWorker))), 		&disp[4] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._nodeWorkload))), 		&disp[5] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._localWorkload))), 		&disp[6] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._totalWorkload))), 		&disp[7] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[1]._persistentRecords._cellDescriptionIndex))), 		&disp[8] );
   
   for (int i=1; i<Attributes; i++) {
      assertion1( disp[i] > disp[i-1], i );
   }
   for (int i=0; i<Attributes; i++) {
      disp[i] -= base;
   }
   MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Cell::Datatype );
   MPI_Type_commit( &Cell::Datatype );
   
}
{
   Cell dummyCell[2];
   
   const int Attributes = 15;
   MPI_Datatype subtypes[Attributes] = {
      MPI_INT,		 //cellDescriptionIndex
      MPI_CHAR,		 //isInside
      MPI_INT,		 //state
      MPI_INT,		 //level
      MPI_INT,		 //evenFlags
      MPI_SHORT,		 //accessNumber
      MPI_INT,		 //responsibleRank
      MPI_CHAR,		 //subtreeHoldsWorker
      MPI_DOUBLE,		 //nodeWorkload
      MPI_DOUBLE,		 //localWorkload
      MPI_DOUBLE,		 //totalWorkload
      MPI_CHAR,		 //cellIsAForkCandidate
      MPI_INT,		 //numberOfLoadsFromInputStream
      MPI_INT,		 //numberOfStoresToOutputStream
      MPI_UB		 // end/displacement flag
   };
   
   int blocklen[Attributes] = {
      1,		 //cellDescriptionIndex
      1,		 //isInside
      1,		 //state
      1,		 //level
      DIMENSIONS,		 //evenFlags
      DIMENSIONS_TIMES_TWO,		 //accessNumber
      1,		 //responsibleRank
      1,		 //subtreeHoldsWorker
      1,		 //nodeWorkload
      1,		 //localWorkload
      1,		 //totalWorkload
      1,		 //cellIsAForkCandidate
      1,		 //numberOfLoadsFromInputStream
      1,		 //numberOfStoresToOutputStream
      1		 // end/displacement flag
   };
   
   MPI_Aint     disp[Attributes];
   
   MPI_Aint base;
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]))), &base);
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._isInside))), 		&disp[1] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._state))), 		&disp[2] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._level))), 		&disp[3] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._evenFlags))), 		&disp[4] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._accessNumber[0]))), 		&disp[5] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._responsibleRank))), 		&disp[6] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._subtreeHoldsWorker))), 		&disp[7] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._nodeWorkload))), 		&disp[8] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._localWorkload))), 		&disp[9] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._totalWorkload))), 		&disp[10] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellIsAForkCandidate))), 		&disp[11] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._numberOfLoadsFromInputStream))), 		&disp[12] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._numberOfStoresToOutputStream))), 		&disp[13] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[1]._persistentRecords._cellDescriptionIndex))), 		&disp[14] );
   
   for (int i=1; i<Attributes; i++) {
      assertion1( disp[i] > disp[i-1], i );
   }
   for (int i=0; i<Attributes; i++) {
      disp[i] -= base;
   }
   MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Cell::FullDatatype );
   MPI_Type_commit( &Cell::FullDatatype );
   
}

}


void peanoclaw::records::Cell::shutdownDatatype() {
MPI_Type_free( &Cell::Datatype );
MPI_Type_free( &Cell::FullDatatype );

}

void peanoclaw::records::Cell::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
   msg << "was not able to send message peanoclaw::records::Cell "
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
      msg << "testing for finished send task for peanoclaw::records::Cell "
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
      "peanoclaw::records::Cell",
      "send(int)", destination,tag,1
      );
      triggeredTimeoutWarning = true;
   }
   if (
      tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
      (clock()>timeOutShutdown)
   ) {
      tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
      "peanoclaw::records::Cell",
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



void peanoclaw::records::Cell::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
   msg << "failed to start to receive peanoclaw::records::Cell from node "
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
      msg << "testing for finished receive task for peanoclaw::records::Cell failed: "
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
      "peanoclaw::records::Cell",
      "receive(int)", source,tag,1
      );
      triggeredTimeoutWarning = true;
   }
   if (
      tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
      (clock()>timeOutShutdown)
   ) {
      tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
      "peanoclaw::records::Cell",
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



bool peanoclaw::records::Cell::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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

int peanoclaw::records::Cell::getSenderRank() const {
assertion( _senderDestinationRank!=-1 );
return _senderDestinationRank;

}
#endif


peanoclaw::records::CellPacked::PersistentRecords::PersistentRecords() {
assertion((DIMENSIONS+4 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::PersistentRecords::PersistentRecords(const int& cellDescriptionIndex, const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const double& nodeWorkload, const double& localWorkload, const double& totalWorkload, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream):
_cellDescriptionIndex(cellDescriptionIndex),
_level(level),
_accessNumber(accessNumber),
_responsibleRank(responsibleRank),
_subtreeHoldsWorker(subtreeHoldsWorker),
_nodeWorkload(nodeWorkload),
_localWorkload(localWorkload),
_totalWorkload(totalWorkload),
_numberOfLoadsFromInputStream(numberOfLoadsFromInputStream),
_numberOfStoresToOutputStream(numberOfStoresToOutputStream) {
setIsInside(isInside);
setState(state);
setEvenFlags(evenFlags);
setCellIsAForkCandidate(cellIsAForkCandidate);
assertion((DIMENSIONS+4 < (8 * sizeof(short int))));

}

peanoclaw::records::CellPacked::CellPacked() {
assertion((DIMENSIONS+4 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::CellPacked(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._cellDescriptionIndex, persistentRecords.getIsInside(), persistentRecords.getState(), persistentRecords._level, persistentRecords.getEvenFlags(), persistentRecords._accessNumber, persistentRecords._responsibleRank, persistentRecords._subtreeHoldsWorker, persistentRecords._nodeWorkload, persistentRecords._localWorkload, persistentRecords._totalWorkload, persistentRecords.getCellIsAForkCandidate(), persistentRecords._numberOfLoadsFromInputStream, persistentRecords._numberOfStoresToOutputStream) {
assertion((DIMENSIONS+4 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::CellPacked(const int& cellDescriptionIndex, const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const double& nodeWorkload, const double& localWorkload, const double& totalWorkload, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream):
_persistentRecords(cellDescriptionIndex, isInside, state, level, evenFlags, accessNumber, responsibleRank, subtreeHoldsWorker, nodeWorkload, localWorkload, totalWorkload, cellIsAForkCandidate, numberOfLoadsFromInputStream, numberOfStoresToOutputStream) {
assertion((DIMENSIONS+4 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::~CellPacked() { }

std::string peanoclaw::records::CellPacked::toString(const State& param) {
return peanoclaw::records::Cell::toString(param);
}

std::string peanoclaw::records::CellPacked::getStateMapping() {
return peanoclaw::records::Cell::getStateMapping();
}



std::string peanoclaw::records::CellPacked::toString() const {
std::ostringstream stringstr;
toString(stringstr);
return stringstr.str();
}

void peanoclaw::records::CellPacked::toString (std::ostream& out) const {
out << "("; 
out << "cellDescriptionIndex:" << getCellDescriptionIndex();
out << ",";
out << "isInside:" << getIsInside();
out << ",";
out << "state:" << toString(getState());
out << ",";
out << "level:" << getLevel();
out << ",";
out << "evenFlags:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getEvenFlags(i) << ",";
   }
   out << getEvenFlags(DIMENSIONS-1) << "]";
out << ",";
out << "accessNumber:[";
   for (int i = 0; i < DIMENSIONS_TIMES_TWO-1; i++) {
      out << getAccessNumber(i) << ",";
   }
   out << getAccessNumber(DIMENSIONS_TIMES_TWO-1) << "]";
out << ",";
out << "responsibleRank:" << getResponsibleRank();
out << ",";
out << "subtreeHoldsWorker:" << getSubtreeHoldsWorker();
out << ",";
out << "nodeWorkload:" << getNodeWorkload();
out << ",";
out << "localWorkload:" << getLocalWorkload();
out << ",";
out << "totalWorkload:" << getTotalWorkload();
out << ",";
out << "cellIsAForkCandidate:" << getCellIsAForkCandidate();
out << ",";
out << "numberOfLoadsFromInputStream:" << getNumberOfLoadsFromInputStream();
out << ",";
out << "numberOfStoresToOutputStream:" << getNumberOfStoresToOutputStream();
out <<  ")";
}


peanoclaw::records::CellPacked::PersistentRecords peanoclaw::records::CellPacked::getPersistentRecords() const {
return _persistentRecords;
}

peanoclaw::records::Cell peanoclaw::records::CellPacked::convert() const{
return Cell(
getCellDescriptionIndex(),
getIsInside(),
getState(),
getLevel(),
getEvenFlags(),
getAccessNumber(),
getResponsibleRank(),
getSubtreeHoldsWorker(),
getNodeWorkload(),
getLocalWorkload(),
getTotalWorkload(),
getCellIsAForkCandidate(),
getNumberOfLoadsFromInputStream(),
getNumberOfStoresToOutputStream()
);
}

#ifdef Parallel
tarch::logging::Log peanoclaw::records::CellPacked::_log( "peanoclaw::records::CellPacked" );

MPI_Datatype peanoclaw::records::CellPacked::Datatype = 0;
MPI_Datatype peanoclaw::records::CellPacked::FullDatatype = 0;


void peanoclaw::records::CellPacked::initDatatype() {
{
   CellPacked dummyCellPacked[2];
   
   const int Attributes = 8;
   MPI_Datatype subtypes[Attributes] = {
      MPI_INT,		 //cellDescriptionIndex
      MPI_INT,		 //level
      MPI_CHAR,		 //subtreeHoldsWorker
      MPI_DOUBLE,		 //nodeWorkload
      MPI_DOUBLE,		 //localWorkload
      MPI_DOUBLE,		 //totalWorkload
      MPI_SHORT,		 //_packedRecords0
      MPI_UB		 // end/displacement flag
   };
   
   int blocklen[Attributes] = {
      1,		 //cellDescriptionIndex
      1,		 //level
      1,		 //subtreeHoldsWorker
      1,		 //nodeWorkload
      1,		 //localWorkload
      1,		 //totalWorkload
      1,		 //_packedRecords0
      1		 // end/displacement flag
   };
   
   MPI_Aint     disp[Attributes];
   
   MPI_Aint base;
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]))), &base);
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._level))), 		&disp[1] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._subtreeHoldsWorker))), 		&disp[2] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._nodeWorkload))), 		&disp[3] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._localWorkload))), 		&disp[4] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._totalWorkload))), 		&disp[5] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._packedRecords0))), 		&disp[6] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[1]._persistentRecords._cellDescriptionIndex))), 		&disp[7] );
   
   for (int i=1; i<Attributes; i++) {
      assertion1( disp[i] > disp[i-1], i );
   }
   for (int i=0; i<Attributes; i++) {
      disp[i] -= base;
   }
   MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellPacked::Datatype );
   MPI_Type_commit( &CellPacked::Datatype );
   
}
{
   CellPacked dummyCellPacked[2];
   
   const int Attributes = 12;
   MPI_Datatype subtypes[Attributes] = {
      MPI_INT,		 //cellDescriptionIndex
      MPI_INT,		 //level
      MPI_SHORT,		 //accessNumber
      MPI_INT,		 //responsibleRank
      MPI_CHAR,		 //subtreeHoldsWorker
      MPI_DOUBLE,		 //nodeWorkload
      MPI_DOUBLE,		 //localWorkload
      MPI_DOUBLE,		 //totalWorkload
      MPI_INT,		 //numberOfLoadsFromInputStream
      MPI_INT,		 //numberOfStoresToOutputStream
      MPI_SHORT,		 //_packedRecords0
      MPI_UB		 // end/displacement flag
   };
   
   int blocklen[Attributes] = {
      1,		 //cellDescriptionIndex
      1,		 //level
      DIMENSIONS_TIMES_TWO,		 //accessNumber
      1,		 //responsibleRank
      1,		 //subtreeHoldsWorker
      1,		 //nodeWorkload
      1,		 //localWorkload
      1,		 //totalWorkload
      1,		 //numberOfLoadsFromInputStream
      1,		 //numberOfStoresToOutputStream
      1,		 //_packedRecords0
      1		 // end/displacement flag
   };
   
   MPI_Aint     disp[Attributes];
   
   MPI_Aint base;
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]))), &base);
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._level))), 		&disp[1] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._accessNumber[0]))), 		&disp[2] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._responsibleRank))), 		&disp[3] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._subtreeHoldsWorker))), 		&disp[4] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._nodeWorkload))), 		&disp[5] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._localWorkload))), 		&disp[6] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._totalWorkload))), 		&disp[7] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._numberOfLoadsFromInputStream))), 		&disp[8] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._numberOfStoresToOutputStream))), 		&disp[9] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._packedRecords0))), 		&disp[10] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[1]._persistentRecords._cellDescriptionIndex))), 		&disp[11] );
   
   for (int i=1; i<Attributes; i++) {
      assertion1( disp[i] > disp[i-1], i );
   }
   for (int i=0; i<Attributes; i++) {
      disp[i] -= base;
   }
   MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellPacked::FullDatatype );
   MPI_Type_commit( &CellPacked::FullDatatype );
   
}

}


void peanoclaw::records::CellPacked::shutdownDatatype() {
MPI_Type_free( &CellPacked::Datatype );
MPI_Type_free( &CellPacked::FullDatatype );

}

void peanoclaw::records::CellPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
   msg << "was not able to send message peanoclaw::records::CellPacked "
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
      msg << "testing for finished send task for peanoclaw::records::CellPacked "
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
      "peanoclaw::records::CellPacked",
      "send(int)", destination,tag,1
      );
      triggeredTimeoutWarning = true;
   }
   if (
      tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
      (clock()>timeOutShutdown)
   ) {
      tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
      "peanoclaw::records::CellPacked",
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



void peanoclaw::records::CellPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
   msg << "failed to start to receive peanoclaw::records::CellPacked from node "
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
      msg << "testing for finished receive task for peanoclaw::records::CellPacked failed: "
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
      "peanoclaw::records::CellPacked",
      "receive(int)", source,tag,1
      );
      triggeredTimeoutWarning = true;
   }
   if (
      tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
      (clock()>timeOutShutdown)
   ) {
      tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
      "peanoclaw::records::CellPacked",
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



bool peanoclaw::records::CellPacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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

int peanoclaw::records::CellPacked::getSenderRank() const {
assertion( _senderDestinationRank!=-1 );
return _senderDestinationRank;

}
#endif




#elif defined(Parallel) && defined(Debug) && !defined(SharedMemoryParallelisation)
peanoclaw::records::Cell::PersistentRecords::PersistentRecords() {

}


peanoclaw::records::Cell::PersistentRecords::PersistentRecords(const int& cellDescriptionIndex, const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const double& nodeWorkload, const double& localWorkload, const double& totalWorkload, const bool& cellIsAForkCandidate):
_cellDescriptionIndex(cellDescriptionIndex),
_isInside(isInside),
_state(state),
_level(level),
_evenFlags(evenFlags),
_accessNumber(accessNumber),
_responsibleRank(responsibleRank),
_subtreeHoldsWorker(subtreeHoldsWorker),
_nodeWorkload(nodeWorkload),
_localWorkload(localWorkload),
_totalWorkload(totalWorkload),
_cellIsAForkCandidate(cellIsAForkCandidate) {

}

peanoclaw::records::Cell::Cell() {

}


peanoclaw::records::Cell::Cell(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._cellDescriptionIndex, persistentRecords._isInside, persistentRecords._state, persistentRecords._level, persistentRecords._evenFlags, persistentRecords._accessNumber, persistentRecords._responsibleRank, persistentRecords._subtreeHoldsWorker, persistentRecords._nodeWorkload, persistentRecords._localWorkload, persistentRecords._totalWorkload, persistentRecords._cellIsAForkCandidate) {

}


peanoclaw::records::Cell::Cell(const int& cellDescriptionIndex, const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const double& nodeWorkload, const double& localWorkload, const double& totalWorkload, const bool& cellIsAForkCandidate):
_persistentRecords(cellDescriptionIndex, isInside, state, level, evenFlags, accessNumber, responsibleRank, subtreeHoldsWorker, nodeWorkload, localWorkload, totalWorkload, cellIsAForkCandidate) {

}


peanoclaw::records::Cell::~Cell() { }

std::string peanoclaw::records::Cell::toString(const State& param) {
switch (param) {
case Leaf: return "Leaf";
case Refined: return "Refined";
case Root: return "Root";
}
return "undefined";
}

std::string peanoclaw::records::Cell::getStateMapping() {
return "State(Leaf=0,Refined=1,Root=2)";
}


std::string peanoclaw::records::Cell::toString() const {
std::ostringstream stringstr;
toString(stringstr);
return stringstr.str();
}

void peanoclaw::records::Cell::toString (std::ostream& out) const {
out << "("; 
out << "cellDescriptionIndex:" << getCellDescriptionIndex();
out << ",";
out << "isInside:" << getIsInside();
out << ",";
out << "state:" << toString(getState());
out << ",";
out << "level:" << getLevel();
out << ",";
out << "evenFlags:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getEvenFlags(i) << ",";
   }
   out << getEvenFlags(DIMENSIONS-1) << "]";
out << ",";
out << "accessNumber:[";
   for (int i = 0; i < DIMENSIONS_TIMES_TWO-1; i++) {
      out << getAccessNumber(i) << ",";
   }
   out << getAccessNumber(DIMENSIONS_TIMES_TWO-1) << "]";
out << ",";
out << "responsibleRank:" << getResponsibleRank();
out << ",";
out << "subtreeHoldsWorker:" << getSubtreeHoldsWorker();
out << ",";
out << "nodeWorkload:" << getNodeWorkload();
out << ",";
out << "localWorkload:" << getLocalWorkload();
out << ",";
out << "totalWorkload:" << getTotalWorkload();
out << ",";
out << "cellIsAForkCandidate:" << getCellIsAForkCandidate();
out <<  ")";
}


peanoclaw::records::Cell::PersistentRecords peanoclaw::records::Cell::getPersistentRecords() const {
return _persistentRecords;
}

peanoclaw::records::CellPacked peanoclaw::records::Cell::convert() const{
return CellPacked(
getCellDescriptionIndex(),
getIsInside(),
getState(),
getLevel(),
getEvenFlags(),
getAccessNumber(),
getResponsibleRank(),
getSubtreeHoldsWorker(),
getNodeWorkload(),
getLocalWorkload(),
getTotalWorkload(),
getCellIsAForkCandidate()
);
}

#ifdef Parallel
tarch::logging::Log peanoclaw::records::Cell::_log( "peanoclaw::records::Cell" );

MPI_Datatype peanoclaw::records::Cell::Datatype = 0;
MPI_Datatype peanoclaw::records::Cell::FullDatatype = 0;


void peanoclaw::records::Cell::initDatatype() {
{
Cell dummyCell[2];

const int Attributes = 9;
MPI_Datatype subtypes[Attributes] = {
   MPI_INT,		 //cellDescriptionIndex
   MPI_CHAR,		 //isInside
   MPI_INT,		 //state
   MPI_INT,		 //level
   MPI_CHAR,		 //subtreeHoldsWorker
   MPI_DOUBLE,		 //nodeWorkload
   MPI_DOUBLE,		 //localWorkload
   MPI_DOUBLE,		 //totalWorkload
   MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
   1,		 //cellDescriptionIndex
   1,		 //isInside
   1,		 //state
   1,		 //level
   1,		 //subtreeHoldsWorker
   1,		 //nodeWorkload
   1,		 //localWorkload
   1,		 //totalWorkload
   1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._isInside))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._state))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._level))), 		&disp[3] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._subtreeHoldsWorker))), 		&disp[4] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._nodeWorkload))), 		&disp[5] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._localWorkload))), 		&disp[6] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._totalWorkload))), 		&disp[7] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[1]._persistentRecords._cellDescriptionIndex))), 		&disp[8] );

for (int i=1; i<Attributes; i++) {
   assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
   disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Cell::Datatype );
MPI_Type_commit( &Cell::Datatype );

}
{
Cell dummyCell[2];

const int Attributes = 13;
MPI_Datatype subtypes[Attributes] = {
   MPI_INT,		 //cellDescriptionIndex
   MPI_CHAR,		 //isInside
   MPI_INT,		 //state
   MPI_INT,		 //level
   MPI_INT,		 //evenFlags
   MPI_SHORT,		 //accessNumber
   MPI_INT,		 //responsibleRank
   MPI_CHAR,		 //subtreeHoldsWorker
   MPI_DOUBLE,		 //nodeWorkload
   MPI_DOUBLE,		 //localWorkload
   MPI_DOUBLE,		 //totalWorkload
   MPI_CHAR,		 //cellIsAForkCandidate
   MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
   1,		 //cellDescriptionIndex
   1,		 //isInside
   1,		 //state
   1,		 //level
   DIMENSIONS,		 //evenFlags
   DIMENSIONS_TIMES_TWO,		 //accessNumber
   1,		 //responsibleRank
   1,		 //subtreeHoldsWorker
   1,		 //nodeWorkload
   1,		 //localWorkload
   1,		 //totalWorkload
   1,		 //cellIsAForkCandidate
   1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._isInside))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._state))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._level))), 		&disp[3] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._evenFlags))), 		&disp[4] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._accessNumber[0]))), 		&disp[5] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._responsibleRank))), 		&disp[6] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._subtreeHoldsWorker))), 		&disp[7] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._nodeWorkload))), 		&disp[8] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._localWorkload))), 		&disp[9] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._totalWorkload))), 		&disp[10] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellIsAForkCandidate))), 		&disp[11] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[1]._persistentRecords._cellDescriptionIndex))), 		&disp[12] );

for (int i=1; i<Attributes; i++) {
   assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
   disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Cell::FullDatatype );
MPI_Type_commit( &Cell::FullDatatype );

}

}


void peanoclaw::records::Cell::shutdownDatatype() {
MPI_Type_free( &Cell::Datatype );
MPI_Type_free( &Cell::FullDatatype );

}

void peanoclaw::records::Cell::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
msg << "was not able to send message peanoclaw::records::Cell "
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
   msg << "testing for finished send task for peanoclaw::records::Cell "
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
   "peanoclaw::records::Cell",
   "send(int)", destination,tag,1
   );
   triggeredTimeoutWarning = true;
}
if (
   tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
   (clock()>timeOutShutdown)
) {
   tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
   "peanoclaw::records::Cell",
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



void peanoclaw::records::Cell::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
msg << "failed to start to receive peanoclaw::records::Cell from node "
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
   msg << "testing for finished receive task for peanoclaw::records::Cell failed: "
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
   "peanoclaw::records::Cell",
   "receive(int)", source,tag,1
   );
   triggeredTimeoutWarning = true;
}
if (
   tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
   (clock()>timeOutShutdown)
) {
   tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
   "peanoclaw::records::Cell",
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



bool peanoclaw::records::Cell::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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

int peanoclaw::records::Cell::getSenderRank() const {
assertion( _senderDestinationRank!=-1 );
return _senderDestinationRank;

}
#endif


peanoclaw::records::CellPacked::PersistentRecords::PersistentRecords() {
assertion((DIMENSIONS+4 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::PersistentRecords::PersistentRecords(const int& cellDescriptionIndex, const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const double& nodeWorkload, const double& localWorkload, const double& totalWorkload, const bool& cellIsAForkCandidate):
_cellDescriptionIndex(cellDescriptionIndex),
_level(level),
_accessNumber(accessNumber),
_responsibleRank(responsibleRank),
_subtreeHoldsWorker(subtreeHoldsWorker),
_nodeWorkload(nodeWorkload),
_localWorkload(localWorkload),
_totalWorkload(totalWorkload) {
setIsInside(isInside);
setState(state);
setEvenFlags(evenFlags);
setCellIsAForkCandidate(cellIsAForkCandidate);
assertion((DIMENSIONS+4 < (8 * sizeof(short int))));

}

peanoclaw::records::CellPacked::CellPacked() {
assertion((DIMENSIONS+4 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::CellPacked(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._cellDescriptionIndex, persistentRecords.getIsInside(), persistentRecords.getState(), persistentRecords._level, persistentRecords.getEvenFlags(), persistentRecords._accessNumber, persistentRecords._responsibleRank, persistentRecords._subtreeHoldsWorker, persistentRecords._nodeWorkload, persistentRecords._localWorkload, persistentRecords._totalWorkload, persistentRecords.getCellIsAForkCandidate()) {
assertion((DIMENSIONS+4 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::CellPacked(const int& cellDescriptionIndex, const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const double& nodeWorkload, const double& localWorkload, const double& totalWorkload, const bool& cellIsAForkCandidate):
_persistentRecords(cellDescriptionIndex, isInside, state, level, evenFlags, accessNumber, responsibleRank, subtreeHoldsWorker, nodeWorkload, localWorkload, totalWorkload, cellIsAForkCandidate) {
assertion((DIMENSIONS+4 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::~CellPacked() { }

std::string peanoclaw::records::CellPacked::toString(const State& param) {
return peanoclaw::records::Cell::toString(param);
}

std::string peanoclaw::records::CellPacked::getStateMapping() {
return peanoclaw::records::Cell::getStateMapping();
}



std::string peanoclaw::records::CellPacked::toString() const {
std::ostringstream stringstr;
toString(stringstr);
return stringstr.str();
}

void peanoclaw::records::CellPacked::toString (std::ostream& out) const {
out << "("; 
out << "cellDescriptionIndex:" << getCellDescriptionIndex();
out << ",";
out << "isInside:" << getIsInside();
out << ",";
out << "state:" << toString(getState());
out << ",";
out << "level:" << getLevel();
out << ",";
out << "evenFlags:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getEvenFlags(i) << ",";
   }
   out << getEvenFlags(DIMENSIONS-1) << "]";
out << ",";
out << "accessNumber:[";
   for (int i = 0; i < DIMENSIONS_TIMES_TWO-1; i++) {
      out << getAccessNumber(i) << ",";
   }
   out << getAccessNumber(DIMENSIONS_TIMES_TWO-1) << "]";
out << ",";
out << "responsibleRank:" << getResponsibleRank();
out << ",";
out << "subtreeHoldsWorker:" << getSubtreeHoldsWorker();
out << ",";
out << "nodeWorkload:" << getNodeWorkload();
out << ",";
out << "localWorkload:" << getLocalWorkload();
out << ",";
out << "totalWorkload:" << getTotalWorkload();
out << ",";
out << "cellIsAForkCandidate:" << getCellIsAForkCandidate();
out <<  ")";
}


peanoclaw::records::CellPacked::PersistentRecords peanoclaw::records::CellPacked::getPersistentRecords() const {
return _persistentRecords;
}

peanoclaw::records::Cell peanoclaw::records::CellPacked::convert() const{
return Cell(
getCellDescriptionIndex(),
getIsInside(),
getState(),
getLevel(),
getEvenFlags(),
getAccessNumber(),
getResponsibleRank(),
getSubtreeHoldsWorker(),
getNodeWorkload(),
getLocalWorkload(),
getTotalWorkload(),
getCellIsAForkCandidate()
);
}

#ifdef Parallel
tarch::logging::Log peanoclaw::records::CellPacked::_log( "peanoclaw::records::CellPacked" );

MPI_Datatype peanoclaw::records::CellPacked::Datatype = 0;
MPI_Datatype peanoclaw::records::CellPacked::FullDatatype = 0;


void peanoclaw::records::CellPacked::initDatatype() {
{
CellPacked dummyCellPacked[2];

const int Attributes = 8;
MPI_Datatype subtypes[Attributes] = {
   MPI_INT,		 //cellDescriptionIndex
   MPI_INT,		 //level
   MPI_CHAR,		 //subtreeHoldsWorker
   MPI_DOUBLE,		 //nodeWorkload
   MPI_DOUBLE,		 //localWorkload
   MPI_DOUBLE,		 //totalWorkload
   MPI_SHORT,		 //_packedRecords0
   MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
   1,		 //cellDescriptionIndex
   1,		 //level
   1,		 //subtreeHoldsWorker
   1,		 //nodeWorkload
   1,		 //localWorkload
   1,		 //totalWorkload
   1,		 //_packedRecords0
   1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._level))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._subtreeHoldsWorker))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._nodeWorkload))), 		&disp[3] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._localWorkload))), 		&disp[4] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._totalWorkload))), 		&disp[5] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._packedRecords0))), 		&disp[6] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[1]._persistentRecords._cellDescriptionIndex))), 		&disp[7] );

for (int i=1; i<Attributes; i++) {
   assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
   disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellPacked::Datatype );
MPI_Type_commit( &CellPacked::Datatype );

}
{
CellPacked dummyCellPacked[2];

const int Attributes = 10;
MPI_Datatype subtypes[Attributes] = {
   MPI_INT,		 //cellDescriptionIndex
   MPI_INT,		 //level
   MPI_SHORT,		 //accessNumber
   MPI_INT,		 //responsibleRank
   MPI_CHAR,		 //subtreeHoldsWorker
   MPI_DOUBLE,		 //nodeWorkload
   MPI_DOUBLE,		 //localWorkload
   MPI_DOUBLE,		 //totalWorkload
   MPI_SHORT,		 //_packedRecords0
   MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
   1,		 //cellDescriptionIndex
   1,		 //level
   DIMENSIONS_TIMES_TWO,		 //accessNumber
   1,		 //responsibleRank
   1,		 //subtreeHoldsWorker
   1,		 //nodeWorkload
   1,		 //localWorkload
   1,		 //totalWorkload
   1,		 //_packedRecords0
   1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._level))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._accessNumber[0]))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._responsibleRank))), 		&disp[3] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._subtreeHoldsWorker))), 		&disp[4] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._nodeWorkload))), 		&disp[5] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._localWorkload))), 		&disp[6] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._totalWorkload))), 		&disp[7] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._packedRecords0))), 		&disp[8] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[1]._persistentRecords._cellDescriptionIndex))), 		&disp[9] );

for (int i=1; i<Attributes; i++) {
   assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
   disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellPacked::FullDatatype );
MPI_Type_commit( &CellPacked::FullDatatype );

}

}


void peanoclaw::records::CellPacked::shutdownDatatype() {
MPI_Type_free( &CellPacked::Datatype );
MPI_Type_free( &CellPacked::FullDatatype );

}

void peanoclaw::records::CellPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
msg << "was not able to send message peanoclaw::records::CellPacked "
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
   msg << "testing for finished send task for peanoclaw::records::CellPacked "
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
   "peanoclaw::records::CellPacked",
   "send(int)", destination,tag,1
   );
   triggeredTimeoutWarning = true;
}
if (
   tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
   (clock()>timeOutShutdown)
) {
   tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
   "peanoclaw::records::CellPacked",
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



void peanoclaw::records::CellPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
msg << "failed to start to receive peanoclaw::records::CellPacked from node "
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
   msg << "testing for finished receive task for peanoclaw::records::CellPacked failed: "
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
   "peanoclaw::records::CellPacked",
   "receive(int)", source,tag,1
   );
   triggeredTimeoutWarning = true;
}
if (
   tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
   (clock()>timeOutShutdown)
) {
   tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
   "peanoclaw::records::CellPacked",
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



bool peanoclaw::records::CellPacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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

int peanoclaw::records::CellPacked::getSenderRank() const {
assertion( _senderDestinationRank!=-1 );
return _senderDestinationRank;

}
#endif




#elif defined(Parallel) && !defined(Debug) && defined(SharedMemoryParallelisation)
peanoclaw::records::Cell::PersistentRecords::PersistentRecords() {

}


peanoclaw::records::Cell::PersistentRecords::PersistentRecords(const int& cellDescriptionIndex, const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const double& nodeWorkload, const double& localWorkload, const double& totalWorkload, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream):
_cellDescriptionIndex(cellDescriptionIndex),
_isInside(isInside),
_state(state),
_evenFlags(evenFlags),
_accessNumber(accessNumber),
_responsibleRank(responsibleRank),
_subtreeHoldsWorker(subtreeHoldsWorker),
_nodeWorkload(nodeWorkload),
_localWorkload(localWorkload),
_totalWorkload(totalWorkload),
_cellIsAForkCandidate(cellIsAForkCandidate),
_numberOfLoadsFromInputStream(numberOfLoadsFromInputStream),
_numberOfStoresToOutputStream(numberOfStoresToOutputStream) {

}

peanoclaw::records::Cell::Cell() {

}


peanoclaw::records::Cell::Cell(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._cellDescriptionIndex, persistentRecords._isInside, persistentRecords._state, persistentRecords._evenFlags, persistentRecords._accessNumber, persistentRecords._responsibleRank, persistentRecords._subtreeHoldsWorker, persistentRecords._nodeWorkload, persistentRecords._localWorkload, persistentRecords._totalWorkload, persistentRecords._cellIsAForkCandidate, persistentRecords._numberOfLoadsFromInputStream, persistentRecords._numberOfStoresToOutputStream) {

}


peanoclaw::records::Cell::Cell(const int& cellDescriptionIndex, const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const double& nodeWorkload, const double& localWorkload, const double& totalWorkload, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream):
_persistentRecords(cellDescriptionIndex, isInside, state, evenFlags, accessNumber, responsibleRank, subtreeHoldsWorker, nodeWorkload, localWorkload, totalWorkload, cellIsAForkCandidate, numberOfLoadsFromInputStream, numberOfStoresToOutputStream) {

}


peanoclaw::records::Cell::~Cell() { }

std::string peanoclaw::records::Cell::toString(const State& param) {
switch (param) {
case Leaf: return "Leaf";
case Refined: return "Refined";
case Root: return "Root";
}
return "undefined";
}

std::string peanoclaw::records::Cell::getStateMapping() {
return "State(Leaf=0,Refined=1,Root=2)";
}


std::string peanoclaw::records::Cell::toString() const {
std::ostringstream stringstr;
toString(stringstr);
return stringstr.str();
}

void peanoclaw::records::Cell::toString (std::ostream& out) const {
out << "("; 
out << "cellDescriptionIndex:" << getCellDescriptionIndex();
out << ",";
out << "isInside:" << getIsInside();
out << ",";
out << "state:" << toString(getState());
out << ",";
out << "evenFlags:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getEvenFlags(i) << ",";
   }
   out << getEvenFlags(DIMENSIONS-1) << "]";
out << ",";
out << "accessNumber:[";
   for (int i = 0; i < DIMENSIONS_TIMES_TWO-1; i++) {
      out << getAccessNumber(i) << ",";
   }
   out << getAccessNumber(DIMENSIONS_TIMES_TWO-1) << "]";
out << ",";
out << "responsibleRank:" << getResponsibleRank();
out << ",";
out << "subtreeHoldsWorker:" << getSubtreeHoldsWorker();
out << ",";
out << "nodeWorkload:" << getNodeWorkload();
out << ",";
out << "localWorkload:" << getLocalWorkload();
out << ",";
out << "totalWorkload:" << getTotalWorkload();
out << ",";
out << "cellIsAForkCandidate:" << getCellIsAForkCandidate();
out << ",";
out << "numberOfLoadsFromInputStream:" << getNumberOfLoadsFromInputStream();
out << ",";
out << "numberOfStoresToOutputStream:" << getNumberOfStoresToOutputStream();
out <<  ")";
}


peanoclaw::records::Cell::PersistentRecords peanoclaw::records::Cell::getPersistentRecords() const {
return _persistentRecords;
}

peanoclaw::records::CellPacked peanoclaw::records::Cell::convert() const{
return CellPacked(
getCellDescriptionIndex(),
getIsInside(),
getState(),
getEvenFlags(),
getAccessNumber(),
getResponsibleRank(),
getSubtreeHoldsWorker(),
getNodeWorkload(),
getLocalWorkload(),
getTotalWorkload(),
getCellIsAForkCandidate(),
getNumberOfLoadsFromInputStream(),
getNumberOfStoresToOutputStream()
);
}

#ifdef Parallel
tarch::logging::Log peanoclaw::records::Cell::_log( "peanoclaw::records::Cell" );

MPI_Datatype peanoclaw::records::Cell::Datatype = 0;
MPI_Datatype peanoclaw::records::Cell::FullDatatype = 0;


void peanoclaw::records::Cell::initDatatype() {
{
Cell dummyCell[2];

const int Attributes = 8;
MPI_Datatype subtypes[Attributes] = {
MPI_INT,		 //cellDescriptionIndex
MPI_CHAR,		 //isInside
MPI_INT,		 //state
MPI_CHAR,		 //subtreeHoldsWorker
MPI_DOUBLE,		 //nodeWorkload
MPI_DOUBLE,		 //localWorkload
MPI_DOUBLE,		 //totalWorkload
MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
1,		 //cellDescriptionIndex
1,		 //isInside
1,		 //state
1,		 //subtreeHoldsWorker
1,		 //nodeWorkload
1,		 //localWorkload
1,		 //totalWorkload
1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._isInside))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._state))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._subtreeHoldsWorker))), 		&disp[3] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._nodeWorkload))), 		&disp[4] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._localWorkload))), 		&disp[5] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._totalWorkload))), 		&disp[6] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[1]._persistentRecords._cellDescriptionIndex))), 		&disp[7] );

for (int i=1; i<Attributes; i++) {
assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Cell::Datatype );
MPI_Type_commit( &Cell::Datatype );

}
{
Cell dummyCell[2];

const int Attributes = 14;
MPI_Datatype subtypes[Attributes] = {
MPI_INT,		 //cellDescriptionIndex
MPI_CHAR,		 //isInside
MPI_INT,		 //state
MPI_INT,		 //evenFlags
MPI_SHORT,		 //accessNumber
MPI_INT,		 //responsibleRank
MPI_CHAR,		 //subtreeHoldsWorker
MPI_DOUBLE,		 //nodeWorkload
MPI_DOUBLE,		 //localWorkload
MPI_DOUBLE,		 //totalWorkload
MPI_CHAR,		 //cellIsAForkCandidate
MPI_INT,		 //numberOfLoadsFromInputStream
MPI_INT,		 //numberOfStoresToOutputStream
MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
1,		 //cellDescriptionIndex
1,		 //isInside
1,		 //state
DIMENSIONS,		 //evenFlags
DIMENSIONS_TIMES_TWO,		 //accessNumber
1,		 //responsibleRank
1,		 //subtreeHoldsWorker
1,		 //nodeWorkload
1,		 //localWorkload
1,		 //totalWorkload
1,		 //cellIsAForkCandidate
1,		 //numberOfLoadsFromInputStream
1,		 //numberOfStoresToOutputStream
1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._isInside))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._state))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._evenFlags))), 		&disp[3] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._accessNumber[0]))), 		&disp[4] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._responsibleRank))), 		&disp[5] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._subtreeHoldsWorker))), 		&disp[6] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._nodeWorkload))), 		&disp[7] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._localWorkload))), 		&disp[8] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._totalWorkload))), 		&disp[9] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellIsAForkCandidate))), 		&disp[10] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._numberOfLoadsFromInputStream))), 		&disp[11] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._numberOfStoresToOutputStream))), 		&disp[12] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[1]._persistentRecords._cellDescriptionIndex))), 		&disp[13] );

for (int i=1; i<Attributes; i++) {
assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Cell::FullDatatype );
MPI_Type_commit( &Cell::FullDatatype );

}

}


void peanoclaw::records::Cell::shutdownDatatype() {
MPI_Type_free( &Cell::Datatype );
MPI_Type_free( &Cell::FullDatatype );

}

void peanoclaw::records::Cell::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
msg << "was not able to send message peanoclaw::records::Cell "
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
msg << "testing for finished send task for peanoclaw::records::Cell "
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
"peanoclaw::records::Cell",
"send(int)", destination,tag,1
);
triggeredTimeoutWarning = true;
}
if (
tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
(clock()>timeOutShutdown)
) {
tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
"peanoclaw::records::Cell",
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



void peanoclaw::records::Cell::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
msg << "failed to start to receive peanoclaw::records::Cell from node "
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
msg << "testing for finished receive task for peanoclaw::records::Cell failed: "
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
"peanoclaw::records::Cell",
"receive(int)", source,tag,1
);
triggeredTimeoutWarning = true;
}
if (
tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
(clock()>timeOutShutdown)
) {
tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
"peanoclaw::records::Cell",
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



bool peanoclaw::records::Cell::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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

int peanoclaw::records::Cell::getSenderRank() const {
assertion( _senderDestinationRank!=-1 );
return _senderDestinationRank;

}
#endif


peanoclaw::records::CellPacked::PersistentRecords::PersistentRecords() {
assertion((DIMENSIONS+4 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::PersistentRecords::PersistentRecords(const int& cellDescriptionIndex, const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const double& nodeWorkload, const double& localWorkload, const double& totalWorkload, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream):
_cellDescriptionIndex(cellDescriptionIndex),
_accessNumber(accessNumber),
_responsibleRank(responsibleRank),
_subtreeHoldsWorker(subtreeHoldsWorker),
_nodeWorkload(nodeWorkload),
_localWorkload(localWorkload),
_totalWorkload(totalWorkload),
_numberOfLoadsFromInputStream(numberOfLoadsFromInputStream),
_numberOfStoresToOutputStream(numberOfStoresToOutputStream) {
setIsInside(isInside);
setState(state);
setEvenFlags(evenFlags);
setCellIsAForkCandidate(cellIsAForkCandidate);
assertion((DIMENSIONS+4 < (8 * sizeof(short int))));

}

peanoclaw::records::CellPacked::CellPacked() {
assertion((DIMENSIONS+4 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::CellPacked(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._cellDescriptionIndex, persistentRecords.getIsInside(), persistentRecords.getState(), persistentRecords.getEvenFlags(), persistentRecords._accessNumber, persistentRecords._responsibleRank, persistentRecords._subtreeHoldsWorker, persistentRecords._nodeWorkload, persistentRecords._localWorkload, persistentRecords._totalWorkload, persistentRecords.getCellIsAForkCandidate(), persistentRecords._numberOfLoadsFromInputStream, persistentRecords._numberOfStoresToOutputStream) {
assertion((DIMENSIONS+4 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::CellPacked(const int& cellDescriptionIndex, const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const double& nodeWorkload, const double& localWorkload, const double& totalWorkload, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream):
_persistentRecords(cellDescriptionIndex, isInside, state, evenFlags, accessNumber, responsibleRank, subtreeHoldsWorker, nodeWorkload, localWorkload, totalWorkload, cellIsAForkCandidate, numberOfLoadsFromInputStream, numberOfStoresToOutputStream) {
assertion((DIMENSIONS+4 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::~CellPacked() { }

std::string peanoclaw::records::CellPacked::toString(const State& param) {
return peanoclaw::records::Cell::toString(param);
}

std::string peanoclaw::records::CellPacked::getStateMapping() {
return peanoclaw::records::Cell::getStateMapping();
}



std::string peanoclaw::records::CellPacked::toString() const {
std::ostringstream stringstr;
toString(stringstr);
return stringstr.str();
}

void peanoclaw::records::CellPacked::toString (std::ostream& out) const {
out << "("; 
out << "cellDescriptionIndex:" << getCellDescriptionIndex();
out << ",";
out << "isInside:" << getIsInside();
out << ",";
out << "state:" << toString(getState());
out << ",";
out << "evenFlags:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getEvenFlags(i) << ",";
   }
   out << getEvenFlags(DIMENSIONS-1) << "]";
out << ",";
out << "accessNumber:[";
   for (int i = 0; i < DIMENSIONS_TIMES_TWO-1; i++) {
      out << getAccessNumber(i) << ",";
   }
   out << getAccessNumber(DIMENSIONS_TIMES_TWO-1) << "]";
out << ",";
out << "responsibleRank:" << getResponsibleRank();
out << ",";
out << "subtreeHoldsWorker:" << getSubtreeHoldsWorker();
out << ",";
out << "nodeWorkload:" << getNodeWorkload();
out << ",";
out << "localWorkload:" << getLocalWorkload();
out << ",";
out << "totalWorkload:" << getTotalWorkload();
out << ",";
out << "cellIsAForkCandidate:" << getCellIsAForkCandidate();
out << ",";
out << "numberOfLoadsFromInputStream:" << getNumberOfLoadsFromInputStream();
out << ",";
out << "numberOfStoresToOutputStream:" << getNumberOfStoresToOutputStream();
out <<  ")";
}


peanoclaw::records::CellPacked::PersistentRecords peanoclaw::records::CellPacked::getPersistentRecords() const {
return _persistentRecords;
}

peanoclaw::records::Cell peanoclaw::records::CellPacked::convert() const{
return Cell(
getCellDescriptionIndex(),
getIsInside(),
getState(),
getEvenFlags(),
getAccessNumber(),
getResponsibleRank(),
getSubtreeHoldsWorker(),
getNodeWorkload(),
getLocalWorkload(),
getTotalWorkload(),
getCellIsAForkCandidate(),
getNumberOfLoadsFromInputStream(),
getNumberOfStoresToOutputStream()
);
}

#ifdef Parallel
tarch::logging::Log peanoclaw::records::CellPacked::_log( "peanoclaw::records::CellPacked" );

MPI_Datatype peanoclaw::records::CellPacked::Datatype = 0;
MPI_Datatype peanoclaw::records::CellPacked::FullDatatype = 0;


void peanoclaw::records::CellPacked::initDatatype() {
{
CellPacked dummyCellPacked[2];

const int Attributes = 7;
MPI_Datatype subtypes[Attributes] = {
MPI_INT,		 //cellDescriptionIndex
MPI_CHAR,		 //subtreeHoldsWorker
MPI_DOUBLE,		 //nodeWorkload
MPI_DOUBLE,		 //localWorkload
MPI_DOUBLE,		 //totalWorkload
MPI_SHORT,		 //_packedRecords0
MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
1,		 //cellDescriptionIndex
1,		 //subtreeHoldsWorker
1,		 //nodeWorkload
1,		 //localWorkload
1,		 //totalWorkload
1,		 //_packedRecords0
1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._subtreeHoldsWorker))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._nodeWorkload))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._localWorkload))), 		&disp[3] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._totalWorkload))), 		&disp[4] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._packedRecords0))), 		&disp[5] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[1]._persistentRecords._cellDescriptionIndex))), 		&disp[6] );

for (int i=1; i<Attributes; i++) {
assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellPacked::Datatype );
MPI_Type_commit( &CellPacked::Datatype );

}
{
CellPacked dummyCellPacked[2];

const int Attributes = 11;
MPI_Datatype subtypes[Attributes] = {
MPI_INT,		 //cellDescriptionIndex
MPI_SHORT,		 //accessNumber
MPI_INT,		 //responsibleRank
MPI_CHAR,		 //subtreeHoldsWorker
MPI_DOUBLE,		 //nodeWorkload
MPI_DOUBLE,		 //localWorkload
MPI_DOUBLE,		 //totalWorkload
MPI_INT,		 //numberOfLoadsFromInputStream
MPI_INT,		 //numberOfStoresToOutputStream
MPI_SHORT,		 //_packedRecords0
MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
1,		 //cellDescriptionIndex
DIMENSIONS_TIMES_TWO,		 //accessNumber
1,		 //responsibleRank
1,		 //subtreeHoldsWorker
1,		 //nodeWorkload
1,		 //localWorkload
1,		 //totalWorkload
1,		 //numberOfLoadsFromInputStream
1,		 //numberOfStoresToOutputStream
1,		 //_packedRecords0
1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._accessNumber[0]))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._responsibleRank))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._subtreeHoldsWorker))), 		&disp[3] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._nodeWorkload))), 		&disp[4] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._localWorkload))), 		&disp[5] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._totalWorkload))), 		&disp[6] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._numberOfLoadsFromInputStream))), 		&disp[7] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._numberOfStoresToOutputStream))), 		&disp[8] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._packedRecords0))), 		&disp[9] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[1]._persistentRecords._cellDescriptionIndex))), 		&disp[10] );

for (int i=1; i<Attributes; i++) {
assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellPacked::FullDatatype );
MPI_Type_commit( &CellPacked::FullDatatype );

}

}


void peanoclaw::records::CellPacked::shutdownDatatype() {
MPI_Type_free( &CellPacked::Datatype );
MPI_Type_free( &CellPacked::FullDatatype );

}

void peanoclaw::records::CellPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
msg << "was not able to send message peanoclaw::records::CellPacked "
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
msg << "testing for finished send task for peanoclaw::records::CellPacked "
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
"peanoclaw::records::CellPacked",
"send(int)", destination,tag,1
);
triggeredTimeoutWarning = true;
}
if (
tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
(clock()>timeOutShutdown)
) {
tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
"peanoclaw::records::CellPacked",
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



void peanoclaw::records::CellPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
msg << "failed to start to receive peanoclaw::records::CellPacked from node "
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
msg << "testing for finished receive task for peanoclaw::records::CellPacked failed: "
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
"peanoclaw::records::CellPacked",
"receive(int)", source,tag,1
);
triggeredTimeoutWarning = true;
}
if (
tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
(clock()>timeOutShutdown)
) {
tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
"peanoclaw::records::CellPacked",
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



bool peanoclaw::records::CellPacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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

int peanoclaw::records::CellPacked::getSenderRank() const {
assertion( _senderDestinationRank!=-1 );
return _senderDestinationRank;

}
#endif




#elif !defined(Parallel) && defined(SharedMemoryParallelisation) && defined(Debug)
peanoclaw::records::Cell::PersistentRecords::PersistentRecords() {

}


peanoclaw::records::Cell::PersistentRecords::PersistentRecords(const int& cellDescriptionIndex, const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream):
_cellDescriptionIndex(cellDescriptionIndex),
_isInside(isInside),
_state(state),
_level(level),
_evenFlags(evenFlags),
_accessNumber(accessNumber),
_numberOfLoadsFromInputStream(numberOfLoadsFromInputStream),
_numberOfStoresToOutputStream(numberOfStoresToOutputStream) {

}

peanoclaw::records::Cell::Cell() {

}


peanoclaw::records::Cell::Cell(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._cellDescriptionIndex, persistentRecords._isInside, persistentRecords._state, persistentRecords._level, persistentRecords._evenFlags, persistentRecords._accessNumber, persistentRecords._numberOfLoadsFromInputStream, persistentRecords._numberOfStoresToOutputStream) {

}


peanoclaw::records::Cell::Cell(const int& cellDescriptionIndex, const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream):
_persistentRecords(cellDescriptionIndex, isInside, state, level, evenFlags, accessNumber, numberOfLoadsFromInputStream, numberOfStoresToOutputStream) {

}


peanoclaw::records::Cell::~Cell() { }

std::string peanoclaw::records::Cell::toString(const State& param) {
switch (param) {
case Leaf: return "Leaf";
case Refined: return "Refined";
case Root: return "Root";
}
return "undefined";
}

std::string peanoclaw::records::Cell::getStateMapping() {
return "State(Leaf=0,Refined=1,Root=2)";
}


std::string peanoclaw::records::Cell::toString() const {
std::ostringstream stringstr;
toString(stringstr);
return stringstr.str();
}

void peanoclaw::records::Cell::toString (std::ostream& out) const {
out << "("; 
out << "cellDescriptionIndex:" << getCellDescriptionIndex();
out << ",";
out << "isInside:" << getIsInside();
out << ",";
out << "state:" << toString(getState());
out << ",";
out << "level:" << getLevel();
out << ",";
out << "evenFlags:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getEvenFlags(i) << ",";
   }
   out << getEvenFlags(DIMENSIONS-1) << "]";
out << ",";
out << "accessNumber:[";
   for (int i = 0; i < DIMENSIONS_TIMES_TWO-1; i++) {
      out << getAccessNumber(i) << ",";
   }
   out << getAccessNumber(DIMENSIONS_TIMES_TWO-1) << "]";
out << ",";
out << "numberOfLoadsFromInputStream:" << getNumberOfLoadsFromInputStream();
out << ",";
out << "numberOfStoresToOutputStream:" << getNumberOfStoresToOutputStream();
out <<  ")";
}


peanoclaw::records::Cell::PersistentRecords peanoclaw::records::Cell::getPersistentRecords() const {
return _persistentRecords;
}

peanoclaw::records::CellPacked peanoclaw::records::Cell::convert() const{
return CellPacked(
getCellDescriptionIndex(),
getIsInside(),
getState(),
getLevel(),
getEvenFlags(),
getAccessNumber(),
getNumberOfLoadsFromInputStream(),
getNumberOfStoresToOutputStream()
);
}

#ifdef Parallel
tarch::logging::Log peanoclaw::records::Cell::_log( "peanoclaw::records::Cell" );

MPI_Datatype peanoclaw::records::Cell::Datatype = 0;
MPI_Datatype peanoclaw::records::Cell::FullDatatype = 0;


void peanoclaw::records::Cell::initDatatype() {
{
Cell dummyCell[2];

const int Attributes = 5;
MPI_Datatype subtypes[Attributes] = {
MPI_INT,		 //cellDescriptionIndex
MPI_CHAR,		 //isInside
MPI_INT,		 //state
MPI_INT,		 //level
MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
1,		 //cellDescriptionIndex
1,		 //isInside
1,		 //state
1,		 //level
1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._isInside))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._state))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._level))), 		&disp[3] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[1]._persistentRecords._cellDescriptionIndex))), 		&disp[4] );

for (int i=1; i<Attributes; i++) {
assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Cell::Datatype );
MPI_Type_commit( &Cell::Datatype );

}
{
Cell dummyCell[2];

const int Attributes = 9;
MPI_Datatype subtypes[Attributes] = {
MPI_INT,		 //cellDescriptionIndex
MPI_CHAR,		 //isInside
MPI_INT,		 //state
MPI_INT,		 //level
MPI_INT,		 //evenFlags
MPI_SHORT,		 //accessNumber
MPI_INT,		 //numberOfLoadsFromInputStream
MPI_INT,		 //numberOfStoresToOutputStream
MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
1,		 //cellDescriptionIndex
1,		 //isInside
1,		 //state
1,		 //level
DIMENSIONS,		 //evenFlags
DIMENSIONS_TIMES_TWO,		 //accessNumber
1,		 //numberOfLoadsFromInputStream
1,		 //numberOfStoresToOutputStream
1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._isInside))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._state))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._level))), 		&disp[3] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._evenFlags))), 		&disp[4] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._accessNumber[0]))), 		&disp[5] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._numberOfLoadsFromInputStream))), 		&disp[6] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[0]._persistentRecords._numberOfStoresToOutputStream))), 		&disp[7] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCell[1]._persistentRecords._cellDescriptionIndex))), 		&disp[8] );

for (int i=1; i<Attributes; i++) {
assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Cell::FullDatatype );
MPI_Type_commit( &Cell::FullDatatype );

}

}


void peanoclaw::records::Cell::shutdownDatatype() {
MPI_Type_free( &Cell::Datatype );
MPI_Type_free( &Cell::FullDatatype );

}

void peanoclaw::records::Cell::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
msg << "was not able to send message peanoclaw::records::Cell "
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
msg << "testing for finished send task for peanoclaw::records::Cell "
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
"peanoclaw::records::Cell",
"send(int)", destination,tag,1
);
triggeredTimeoutWarning = true;
}
if (
tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
(clock()>timeOutShutdown)
) {
tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
"peanoclaw::records::Cell",
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



void peanoclaw::records::Cell::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
msg << "failed to start to receive peanoclaw::records::Cell from node "
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
msg << "testing for finished receive task for peanoclaw::records::Cell failed: "
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
"peanoclaw::records::Cell",
"receive(int)", source,tag,1
);
triggeredTimeoutWarning = true;
}
if (
tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
(clock()>timeOutShutdown)
) {
tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
"peanoclaw::records::Cell",
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



bool peanoclaw::records::Cell::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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

int peanoclaw::records::Cell::getSenderRank() const {
assertion( _senderDestinationRank!=-1 );
return _senderDestinationRank;

}
#endif


peanoclaw::records::CellPacked::PersistentRecords::PersistentRecords() {
assertion((DIMENSIONS+3 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::PersistentRecords::PersistentRecords(const int& cellDescriptionIndex, const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream):
_cellDescriptionIndex(cellDescriptionIndex),
_level(level),
_accessNumber(accessNumber),
_numberOfLoadsFromInputStream(numberOfLoadsFromInputStream),
_numberOfStoresToOutputStream(numberOfStoresToOutputStream) {
setIsInside(isInside);
setState(state);
setEvenFlags(evenFlags);
assertion((DIMENSIONS+3 < (8 * sizeof(short int))));

}

peanoclaw::records::CellPacked::CellPacked() {
assertion((DIMENSIONS+3 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::CellPacked(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._cellDescriptionIndex, persistentRecords.getIsInside(), persistentRecords.getState(), persistentRecords._level, persistentRecords.getEvenFlags(), persistentRecords._accessNumber, persistentRecords._numberOfLoadsFromInputStream, persistentRecords._numberOfStoresToOutputStream) {
assertion((DIMENSIONS+3 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::CellPacked(const int& cellDescriptionIndex, const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream):
_persistentRecords(cellDescriptionIndex, isInside, state, level, evenFlags, accessNumber, numberOfLoadsFromInputStream, numberOfStoresToOutputStream) {
assertion((DIMENSIONS+3 < (8 * sizeof(short int))));

}


peanoclaw::records::CellPacked::~CellPacked() { }

std::string peanoclaw::records::CellPacked::toString(const State& param) {
return peanoclaw::records::Cell::toString(param);
}

std::string peanoclaw::records::CellPacked::getStateMapping() {
return peanoclaw::records::Cell::getStateMapping();
}



std::string peanoclaw::records::CellPacked::toString() const {
std::ostringstream stringstr;
toString(stringstr);
return stringstr.str();
}

void peanoclaw::records::CellPacked::toString (std::ostream& out) const {
out << "("; 
out << "cellDescriptionIndex:" << getCellDescriptionIndex();
out << ",";
out << "isInside:" << getIsInside();
out << ",";
out << "state:" << toString(getState());
out << ",";
out << "level:" << getLevel();
out << ",";
out << "evenFlags:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getEvenFlags(i) << ",";
   }
   out << getEvenFlags(DIMENSIONS-1) << "]";
out << ",";
out << "accessNumber:[";
   for (int i = 0; i < DIMENSIONS_TIMES_TWO-1; i++) {
      out << getAccessNumber(i) << ",";
   }
   out << getAccessNumber(DIMENSIONS_TIMES_TWO-1) << "]";
out << ",";
out << "numberOfLoadsFromInputStream:" << getNumberOfLoadsFromInputStream();
out << ",";
out << "numberOfStoresToOutputStream:" << getNumberOfStoresToOutputStream();
out <<  ")";
}


peanoclaw::records::CellPacked::PersistentRecords peanoclaw::records::CellPacked::getPersistentRecords() const {
return _persistentRecords;
}

peanoclaw::records::Cell peanoclaw::records::CellPacked::convert() const{
return Cell(
getCellDescriptionIndex(),
getIsInside(),
getState(),
getLevel(),
getEvenFlags(),
getAccessNumber(),
getNumberOfLoadsFromInputStream(),
getNumberOfStoresToOutputStream()
);
}

#ifdef Parallel
tarch::logging::Log peanoclaw::records::CellPacked::_log( "peanoclaw::records::CellPacked" );

MPI_Datatype peanoclaw::records::CellPacked::Datatype = 0;
MPI_Datatype peanoclaw::records::CellPacked::FullDatatype = 0;


void peanoclaw::records::CellPacked::initDatatype() {
{
CellPacked dummyCellPacked[2];

const int Attributes = 4;
MPI_Datatype subtypes[Attributes] = {
MPI_INT,		 //cellDescriptionIndex
MPI_INT,		 //level
MPI_SHORT,		 //_packedRecords0
MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
1,		 //cellDescriptionIndex
1,		 //level
1,		 //_packedRecords0
1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._level))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._packedRecords0))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[1]._persistentRecords._cellDescriptionIndex))), 		&disp[3] );

for (int i=1; i<Attributes; i++) {
assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellPacked::Datatype );
MPI_Type_commit( &CellPacked::Datatype );

}
{
CellPacked dummyCellPacked[2];

const int Attributes = 7;
MPI_Datatype subtypes[Attributes] = {
MPI_INT,		 //cellDescriptionIndex
MPI_INT,		 //level
MPI_SHORT,		 //accessNumber
MPI_INT,		 //numberOfLoadsFromInputStream
MPI_INT,		 //numberOfStoresToOutputStream
MPI_SHORT,		 //_packedRecords0
MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
1,		 //cellDescriptionIndex
1,		 //level
DIMENSIONS_TIMES_TWO,		 //accessNumber
1,		 //numberOfLoadsFromInputStream
1,		 //numberOfStoresToOutputStream
1,		 //_packedRecords0
1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._cellDescriptionIndex))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._level))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._accessNumber[0]))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._numberOfLoadsFromInputStream))), 		&disp[3] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._numberOfStoresToOutputStream))), 		&disp[4] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[0]._persistentRecords._packedRecords0))), 		&disp[5] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyCellPacked[1]._persistentRecords._cellDescriptionIndex))), 		&disp[6] );

for (int i=1; i<Attributes; i++) {
assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &CellPacked::FullDatatype );
MPI_Type_commit( &CellPacked::FullDatatype );

}

}


void peanoclaw::records::CellPacked::shutdownDatatype() {
MPI_Type_free( &CellPacked::Datatype );
MPI_Type_free( &CellPacked::FullDatatype );

}

void peanoclaw::records::CellPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
msg << "was not able to send message peanoclaw::records::CellPacked "
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
msg << "testing for finished send task for peanoclaw::records::CellPacked "
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
"peanoclaw::records::CellPacked",
"send(int)", destination,tag,1
);
triggeredTimeoutWarning = true;
}
if (
tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
(clock()>timeOutShutdown)
) {
tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
"peanoclaw::records::CellPacked",
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



void peanoclaw::records::CellPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
msg << "failed to start to receive peanoclaw::records::CellPacked from node "
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
msg << "testing for finished receive task for peanoclaw::records::CellPacked failed: "
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
"peanoclaw::records::CellPacked",
"receive(int)", source,tag,1
);
triggeredTimeoutWarning = true;
}
if (
tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
(clock()>timeOutShutdown)
) {
tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
"peanoclaw::records::CellPacked",
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



bool peanoclaw::records::CellPacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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

int peanoclaw::records::CellPacked::getSenderRank() const {
assertion( _senderDestinationRank!=-1 );
return _senderDestinationRank;

}
#endif




#endif


