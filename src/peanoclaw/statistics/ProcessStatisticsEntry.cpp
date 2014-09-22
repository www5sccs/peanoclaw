#include "peanoclaw/statistics/ProcessStatisticsEntry.h"

peanoclaw::statistics::ProcessStatisticsEntry::PersistentRecords::PersistentRecords() {
   
}


peanoclaw::statistics::ProcessStatisticsEntry::PersistentRecords::PersistentRecords(const int& rank, const int& numberOfCellUpdates, const int& processorHashCode):
_rank(rank),
_numberOfCellUpdates(numberOfCellUpdates),
_processorHashCode(processorHashCode) {
   
}

peanoclaw::statistics::ProcessStatisticsEntry::ProcessStatisticsEntry() {
   
}


peanoclaw::statistics::ProcessStatisticsEntry::ProcessStatisticsEntry(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._rank, persistentRecords._numberOfCellUpdates, persistentRecords._processorHashCode) {
   
}


peanoclaw::statistics::ProcessStatisticsEntry::ProcessStatisticsEntry(const int& rank, const int& numberOfCellUpdates, const int& processorHashCode):
_persistentRecords(rank, numberOfCellUpdates, processorHashCode) {
   
}


peanoclaw::statistics::ProcessStatisticsEntry::~ProcessStatisticsEntry() { }



std::string peanoclaw::statistics::ProcessStatisticsEntry::toString() const {
   std::ostringstream stringstr;
   toString(stringstr);
   return stringstr.str();
}

void peanoclaw::statistics::ProcessStatisticsEntry::toString (std::ostream& out) const {
   out << "("; 
   out << "rank:" << getRank();
   out << ",";
   out << "numberOfCellUpdates:" << getNumberOfCellUpdates();
   out << ",";
   out << "processorHashCode:" << getProcessorHashCode();
   out <<  ")";
}


peanoclaw::statistics::ProcessStatisticsEntry::PersistentRecords peanoclaw::statistics::ProcessStatisticsEntry::getPersistentRecords() const {
   return _persistentRecords;
}

peanoclaw::statistics::ProcessStatisticsEntryPacked peanoclaw::statistics::ProcessStatisticsEntry::convert() const{
   return ProcessStatisticsEntryPacked(
      getRank(),
      getNumberOfCellUpdates(),
      getProcessorHashCode()
   );
}

#ifdef Parallel
   tarch::logging::Log peanoclaw::statistics::ProcessStatisticsEntry::_log( "peanoclaw::statistics::ProcessStatisticsEntry" );
   
   MPI_Datatype peanoclaw::statistics::ProcessStatisticsEntry::Datatype = 0;
   MPI_Datatype peanoclaw::statistics::ProcessStatisticsEntry::FullDatatype = 0;
   
   
   void peanoclaw::statistics::ProcessStatisticsEntry::initDatatype() {
      {
         ProcessStatisticsEntry dummyProcessStatisticsEntry[2];
         
         const int Attributes = 4;
         MPI_Datatype subtypes[Attributes] = {
            MPI_INT,		 //rank
            MPI_INT,		 //numberOfCellUpdates
            MPI_INT,		 //processorHashCode
            MPI_UB		 // end/displacement flag
         };
         
         int blocklen[Attributes] = {
            1,		 //rank
            1,		 //numberOfCellUpdates
            1,		 //processorHashCode
            1		 // end/displacement flag
         };
         
         MPI_Aint     disp[Attributes];
         
         MPI_Aint base;
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntry[0]))), &base);
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntry[0]._persistentRecords._rank))), 		&disp[0] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntry[0]._persistentRecords._numberOfCellUpdates))), 		&disp[1] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntry[0]._persistentRecords._processorHashCode))), 		&disp[2] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntry[1]._persistentRecords._rank))), 		&disp[3] );
         
         for (int i=1; i<Attributes; i++) {
            assertion1( disp[i] > disp[i-1], i );
         }
         for (int i=0; i<Attributes; i++) {
            disp[i] -= base;
         }
         MPI_Type_struct( Attributes, blocklen, disp, subtypes, &ProcessStatisticsEntry::Datatype );
         MPI_Type_commit( &ProcessStatisticsEntry::Datatype );
         
      }
      {
         ProcessStatisticsEntry dummyProcessStatisticsEntry[2];
         
         const int Attributes = 4;
         MPI_Datatype subtypes[Attributes] = {
            MPI_INT,		 //rank
            MPI_INT,		 //numberOfCellUpdates
            MPI_INT,		 //processorHashCode
            MPI_UB		 // end/displacement flag
         };
         
         int blocklen[Attributes] = {
            1,		 //rank
            1,		 //numberOfCellUpdates
            1,		 //processorHashCode
            1		 // end/displacement flag
         };
         
         MPI_Aint     disp[Attributes];
         
         MPI_Aint base;
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntry[0]))), &base);
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntry[0]._persistentRecords._rank))), 		&disp[0] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntry[0]._persistentRecords._numberOfCellUpdates))), 		&disp[1] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntry[0]._persistentRecords._processorHashCode))), 		&disp[2] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntry[1]._persistentRecords._rank))), 		&disp[3] );
         
         for (int i=1; i<Attributes; i++) {
            assertion1( disp[i] > disp[i-1], i );
         }
         for (int i=0; i<Attributes; i++) {
            disp[i] -= base;
         }
         MPI_Type_struct( Attributes, blocklen, disp, subtypes, &ProcessStatisticsEntry::FullDatatype );
         MPI_Type_commit( &ProcessStatisticsEntry::FullDatatype );
         
      }
      
   }
   
   
   void peanoclaw::statistics::ProcessStatisticsEntry::shutdownDatatype() {
      MPI_Type_free( &ProcessStatisticsEntry::Datatype );
      MPI_Type_free( &ProcessStatisticsEntry::FullDatatype );
      
   }
   
   void peanoclaw::statistics::ProcessStatisticsEntry::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
      if (communicateBlocking) {
      
         const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
         if  (result!=MPI_SUCCESS) {
            std::ostringstream msg;
            msg << "was not able to send message peanoclaw::statistics::ProcessStatisticsEntry "
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
            msg << "was not able to send message peanoclaw::statistics::ProcessStatisticsEntry "
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
               msg << "testing for finished send task for peanoclaw::statistics::ProcessStatisticsEntry "
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
               "peanoclaw::statistics::ProcessStatisticsEntry",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::statistics::ProcessStatisticsEntry",
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
   
   
   
   void peanoclaw::statistics::ProcessStatisticsEntry::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
      if (communicateBlocking) {
      
         MPI_Status  status;
         const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
         if ( result != MPI_SUCCESS ) {
            std::ostringstream msg;
            msg << "failed to start to receive peanoclaw::statistics::ProcessStatisticsEntry from node "
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
            msg << "failed to start to receive peanoclaw::statistics::ProcessStatisticsEntry from node "
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
               msg << "testing for finished receive task for peanoclaw::statistics::ProcessStatisticsEntry failed: "
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
               "peanoclaw::statistics::ProcessStatisticsEntry",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::statistics::ProcessStatisticsEntry",
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
   
   
   
   bool peanoclaw::statistics::ProcessStatisticsEntry::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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


peanoclaw::statistics::ProcessStatisticsEntryPacked::PersistentRecords::PersistentRecords() {
   
}


peanoclaw::statistics::ProcessStatisticsEntryPacked::PersistentRecords::PersistentRecords(const int& rank, const int& numberOfCellUpdates, const int& processorHashCode):
_rank(rank),
_numberOfCellUpdates(numberOfCellUpdates),
_processorHashCode(processorHashCode) {
   
}

peanoclaw::statistics::ProcessStatisticsEntryPacked::ProcessStatisticsEntryPacked() {
   
}


peanoclaw::statistics::ProcessStatisticsEntryPacked::ProcessStatisticsEntryPacked(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._rank, persistentRecords._numberOfCellUpdates, persistentRecords._processorHashCode) {
   
}


peanoclaw::statistics::ProcessStatisticsEntryPacked::ProcessStatisticsEntryPacked(const int& rank, const int& numberOfCellUpdates, const int& processorHashCode):
_persistentRecords(rank, numberOfCellUpdates, processorHashCode) {
   
}


peanoclaw::statistics::ProcessStatisticsEntryPacked::~ProcessStatisticsEntryPacked() { }



std::string peanoclaw::statistics::ProcessStatisticsEntryPacked::toString() const {
   std::ostringstream stringstr;
   toString(stringstr);
   return stringstr.str();
}

void peanoclaw::statistics::ProcessStatisticsEntryPacked::toString (std::ostream& out) const {
   out << "("; 
   out << "rank:" << getRank();
   out << ",";
   out << "numberOfCellUpdates:" << getNumberOfCellUpdates();
   out << ",";
   out << "processorHashCode:" << getProcessorHashCode();
   out <<  ")";
}


peanoclaw::statistics::ProcessStatisticsEntryPacked::PersistentRecords peanoclaw::statistics::ProcessStatisticsEntryPacked::getPersistentRecords() const {
   return _persistentRecords;
}

peanoclaw::statistics::ProcessStatisticsEntry peanoclaw::statistics::ProcessStatisticsEntryPacked::convert() const{
   return ProcessStatisticsEntry(
      getRank(),
      getNumberOfCellUpdates(),
      getProcessorHashCode()
   );
}

#ifdef Parallel
   tarch::logging::Log peanoclaw::statistics::ProcessStatisticsEntryPacked::_log( "peanoclaw::statistics::ProcessStatisticsEntryPacked" );
   
   MPI_Datatype peanoclaw::statistics::ProcessStatisticsEntryPacked::Datatype = 0;
   MPI_Datatype peanoclaw::statistics::ProcessStatisticsEntryPacked::FullDatatype = 0;
   
   
   void peanoclaw::statistics::ProcessStatisticsEntryPacked::initDatatype() {
      {
         ProcessStatisticsEntryPacked dummyProcessStatisticsEntryPacked[2];
         
         const int Attributes = 4;
         MPI_Datatype subtypes[Attributes] = {
            MPI_INT,		 //rank
            MPI_INT,		 //numberOfCellUpdates
            MPI_INT,		 //processorHashCode
            MPI_UB		 // end/displacement flag
         };
         
         int blocklen[Attributes] = {
            1,		 //rank
            1,		 //numberOfCellUpdates
            1,		 //processorHashCode
            1		 // end/displacement flag
         };
         
         MPI_Aint     disp[Attributes];
         
         MPI_Aint base;
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntryPacked[0]))), &base);
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntryPacked[0]._persistentRecords._rank))), 		&disp[0] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntryPacked[0]._persistentRecords._numberOfCellUpdates))), 		&disp[1] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntryPacked[0]._persistentRecords._processorHashCode))), 		&disp[2] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntryPacked[1]._persistentRecords._rank))), 		&disp[3] );
         
         for (int i=1; i<Attributes; i++) {
            assertion1( disp[i] > disp[i-1], i );
         }
         for (int i=0; i<Attributes; i++) {
            disp[i] -= base;
         }
         MPI_Type_struct( Attributes, blocklen, disp, subtypes, &ProcessStatisticsEntryPacked::Datatype );
         MPI_Type_commit( &ProcessStatisticsEntryPacked::Datatype );
         
      }
      {
         ProcessStatisticsEntryPacked dummyProcessStatisticsEntryPacked[2];
         
         const int Attributes = 4;
         MPI_Datatype subtypes[Attributes] = {
            MPI_INT,		 //rank
            MPI_INT,		 //numberOfCellUpdates
            MPI_INT,		 //processorHashCode
            MPI_UB		 // end/displacement flag
         };
         
         int blocklen[Attributes] = {
            1,		 //rank
            1,		 //numberOfCellUpdates
            1,		 //processorHashCode
            1		 // end/displacement flag
         };
         
         MPI_Aint     disp[Attributes];
         
         MPI_Aint base;
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntryPacked[0]))), &base);
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntryPacked[0]._persistentRecords._rank))), 		&disp[0] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntryPacked[0]._persistentRecords._numberOfCellUpdates))), 		&disp[1] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntryPacked[0]._persistentRecords._processorHashCode))), 		&disp[2] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyProcessStatisticsEntryPacked[1]._persistentRecords._rank))), 		&disp[3] );
         
         for (int i=1; i<Attributes; i++) {
            assertion1( disp[i] > disp[i-1], i );
         }
         for (int i=0; i<Attributes; i++) {
            disp[i] -= base;
         }
         MPI_Type_struct( Attributes, blocklen, disp, subtypes, &ProcessStatisticsEntryPacked::FullDatatype );
         MPI_Type_commit( &ProcessStatisticsEntryPacked::FullDatatype );
         
      }
      
   }
   
   
   void peanoclaw::statistics::ProcessStatisticsEntryPacked::shutdownDatatype() {
      MPI_Type_free( &ProcessStatisticsEntryPacked::Datatype );
      MPI_Type_free( &ProcessStatisticsEntryPacked::FullDatatype );
      
   }
   
   void peanoclaw::statistics::ProcessStatisticsEntryPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
      if (communicateBlocking) {
      
         const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
         if  (result!=MPI_SUCCESS) {
            std::ostringstream msg;
            msg << "was not able to send message peanoclaw::statistics::ProcessStatisticsEntryPacked "
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
            msg << "was not able to send message peanoclaw::statistics::ProcessStatisticsEntryPacked "
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
               msg << "testing for finished send task for peanoclaw::statistics::ProcessStatisticsEntryPacked "
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
               "peanoclaw::statistics::ProcessStatisticsEntryPacked",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::statistics::ProcessStatisticsEntryPacked",
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
   
   
   
   void peanoclaw::statistics::ProcessStatisticsEntryPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
      if (communicateBlocking) {
      
         MPI_Status  status;
         const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
         if ( result != MPI_SUCCESS ) {
            std::ostringstream msg;
            msg << "failed to start to receive peanoclaw::statistics::ProcessStatisticsEntryPacked from node "
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
            msg << "failed to start to receive peanoclaw::statistics::ProcessStatisticsEntryPacked from node "
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
               msg << "testing for finished receive task for peanoclaw::statistics::ProcessStatisticsEntryPacked failed: "
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
               "peanoclaw::statistics::ProcessStatisticsEntryPacked",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::statistics::ProcessStatisticsEntryPacked",
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
   
   
   
   bool peanoclaw::statistics::ProcessStatisticsEntryPacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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



