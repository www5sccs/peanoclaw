#include "records/Data.h"

peanoclaw::records::Data::PersistentRecords::PersistentRecords() {
   
}


peanoclaw::records::Data::PersistentRecords::PersistentRecords(const double& u):
_u(u) {
   
}


double peanoclaw::records::Data::PersistentRecords::getU() const {
   return _u;
}



void peanoclaw::records::Data::PersistentRecords::setU(const double& u) {
   _u = u;
}


peanoclaw::records::Data::Data() {
   
}


peanoclaw::records::Data::Data(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._u) {
   
}


peanoclaw::records::Data::Data(const double& u):
_persistentRecords(u) {
   
}


//peanoclaw::records::Data::~Data() { }


//double peanoclaw::records::Data::getU() const {
//   return _persistentRecords._u;
//}



//void peanoclaw::records::Data::setU(const double& u) {
//  _persistentRecords._u = u;
//}




std::string peanoclaw::records::Data::toString() const {
   std::ostringstream stringstr;
   toString(stringstr);
   return stringstr.str();
}

void peanoclaw::records::Data::toString (std::ostream& out) const {
   out << "("; 
   out << "u:" << getU();
   out <<  ")";
}


peanoclaw::records::Data::PersistentRecords peanoclaw::records::Data::getPersistentRecords() const {
   return _persistentRecords;
}

peanoclaw::records::DataPacked peanoclaw::records::Data::convert() const{
   return DataPacked(
      getU()
   );
}
 
tarch::logging::Log peanoclaw::records::Data::_log( "peanoclaw::records::Data" );

#if 0
#ifdef Parallel
   #include "tarch/parallel/Node.h" 
   
   MPI_Datatype peanoclaw::records::Data::Datatype = 0;
   
   
   void peanoclaw::records::Data::initDatatype() {
      const int Attributes = 2;
      MPI_Datatype subtypes[Attributes] = {
         MPI_DOUBLE,		 //u
         MPI_UB		 // end/displacement flag
      };
      
      int blocklen[Attributes] = {
         1,		 //u
         1		 // end/displacement flag
      };
      
      MPI_Aint     disp[Attributes];
      Data dummyData[2];
      
      MPI_Aint base;
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyData[0]))), &base);
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyData[0]._persistentRecords._u))), 		&disp[0] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyData[1]._persistentRecords._u))), 		&disp[1] );
      
      for (int i=1; i<Attributes; i++) {
         assertion1( disp[i] > disp[i-1], i );
      }
      for (int i=0; i<Attributes; i++) {
         disp[i] -= base;
      }
      MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Data::Datatype );
      MPI_Type_commit( &Data::Datatype );
      
   }
   
   
   void peanoclaw::records::Data::shutdownDatatype() {
      MPI_Type_free( &Data::Datatype );
      
   }
   
   void peanoclaw::records::Data::send(int destination, int tag) {
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
      
      result = MPI_Isend(
         this, 1, Datatype, destination,
         tag, tarch::parallel::Node::getInstance().getCommunicator(),
         sendRequestHandle
      );
      if  (result!=MPI_SUCCESS) {
         std::ostringstream msg;
         msg << "was not able to send message peanoclaw::records::Data "
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
            msg << "testing for finished send task for peanoclaw::records::Data "
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
            "peanoclaw::records::Data",
            "send(int)", destination, tag, 1
            );
            triggeredTimeoutWarning = true;
         }
         if (
            tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
            (clock()>timeOutShutdown)
         ) {
            tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
            "peanoclaw::records::Data",
            "send(int)", destination, tag, 1
            );
         }
         tarch::parallel::Node::getInstance().receiveDanglingMessages();
      }
      
      delete sendRequestHandle;
      #ifdef Debug
      _log.debug("send(int,int)", "sent " + toString() );
      #endif
      
   }
   
   
   
   void peanoclaw::records::Data::receive(int source, int tag) {
      MPI_Request* sendRequestHandle = new MPI_Request();
      MPI_Status   status;
      int          flag = 0;
      int          result;
      
      clock_t      timeOutWarning   = -1;
      clock_t      timeOutShutdown  = -1;
      bool         triggeredTimeoutWarning = false;
      
      result = MPI_Irecv(
         this, 1, Datatype, source, tag,
         tarch::parallel::Node::getInstance().getCommunicator(), sendRequestHandle
      );
      if ( result != MPI_SUCCESS ) {
         std::ostringstream msg;
         msg << "failed to start to receive peanoclaw::records::Data from node "
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
            msg << "testing for finished receive task for peanoclaw::records::Data failed: "
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
            "peanoclaw::records::Data",
            "receive(int)", source, tag, 1
            );
            triggeredTimeoutWarning = true;
         }
         if (
            tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
            (clock()>timeOutShutdown)
         ) {
            tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
            "peanoclaw::records::Data",
            "receive(int)", source, tag, 1
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
   
   
   
   bool peanoclaw::records::Data::isMessageInQueue(int tag) {
      MPI_Status status;
      int  flag        = 0;
      MPI_Iprobe(
         MPI_ANY_SOURCE, tag,
         tarch::parallel::Node::getInstance().getCommunicator(), &flag, &status
      );
      if (flag) {
         int  messageCounter;
         MPI_Get_count(&status, Datatype, &messageCounter);
         return messageCounter > 0;
      }
      else return false;
      
   }
   
   int peanoclaw::records::Data::getSenderRank() const {
      assertion( _senderRank!=-1 );
      return _senderRank;
      
   }
#endif
#endif


peanoclaw::records::DataPacked::PersistentRecords::PersistentRecords() {
   
}


peanoclaw::records::DataPacked::PersistentRecords::PersistentRecords(const double& u):
_u(u) {
   
}


double peanoclaw::records::DataPacked::PersistentRecords::getU() const {
   return _u;
}



void peanoclaw::records::DataPacked::PersistentRecords::setU(const double& u) {
   _u = u;
}


peanoclaw::records::DataPacked::DataPacked() {
   
}


peanoclaw::records::DataPacked::DataPacked(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._u) {
   
}


peanoclaw::records::DataPacked::DataPacked(const double& u):
_persistentRecords(u) {
   
}


peanoclaw::records::DataPacked::~DataPacked() { }


double peanoclaw::records::DataPacked::getU() const {
   return _persistentRecords._u;
}



void peanoclaw::records::DataPacked::setU(const double& u) {
   _persistentRecords._u = u;
}




std::string peanoclaw::records::DataPacked::toString() const {
   std::ostringstream stringstr;
   toString(stringstr);
   return stringstr.str();
}

void peanoclaw::records::DataPacked::toString (std::ostream& out) const {
   out << "("; 
   out << "u:" << getU();
   out <<  ")";
}


peanoclaw::records::DataPacked::PersistentRecords peanoclaw::records::DataPacked::getPersistentRecords() const {
   return _persistentRecords;
}

peanoclaw::records::Data peanoclaw::records::DataPacked::convert() const{
   return Data(
      getU()
   );
}

#ifdef Parallel
   #include "tarch/parallel/Node.h" 
   
   tarch::logging::Log peanoclaw::records::DataPacked::_log( "peanoclaw::records::DataPacked" );
   
   MPI_Datatype peanoclaw::records::DataPacked::Datatype = 0;
   
   
   void peanoclaw::records::DataPacked::initDatatype() {
      const int Attributes = 2;
      MPI_Datatype subtypes[Attributes] = {
         MPI_DOUBLE,		 //u
         MPI_UB		 // end/displacement flag
      };
      
      int blocklen[Attributes] = {
         1,		 //u
         1		 // end/displacement flag
      };
      
      MPI_Aint     disp[Attributes];
      DataPacked dummyDataPacked[2];
      
      MPI_Aint base;
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyDataPacked[0]))), &base);
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyDataPacked[0]._persistentRecords._u))), 		&disp[0] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyDataPacked[1]._persistentRecords._u))), 		&disp[1] );
      
      for (int i=1; i<Attributes; i++) {
         assertion1( disp[i] > disp[i-1], i );
      }
      for (int i=0; i<Attributes; i++) {
         disp[i] -= base;
      }
      MPI_Type_struct( Attributes, blocklen, disp, subtypes, &DataPacked::Datatype );
      MPI_Type_commit( &DataPacked::Datatype );
      
   }
   
   
   void peanoclaw::records::DataPacked::shutdownDatatype() {
      MPI_Type_free( &DataPacked::Datatype );
      
   }
   
   void peanoclaw::records::DataPacked::send(int destination, int tag) {
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
      
      result = MPI_Isend(
         this, 1, Datatype, destination,
         tag, tarch::parallel::Node::getInstance().getCommunicator(),
         sendRequestHandle
      );
      if  (result!=MPI_SUCCESS) {
         std::ostringstream msg;
         msg << "was not able to send message peanoclaw::records::DataPacked "
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
            msg << "testing for finished send task for peanoclaw::records::DataPacked "
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
            "peanoclaw::records::DataPacked",
            "send(int)", destination, tag, 1
            );
            triggeredTimeoutWarning = true;
         }
         if (
            tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
            (clock()>timeOutShutdown)
         ) {
            tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
            "peanoclaw::records::DataPacked",
            "send(int)", destination, tag, 1
            );
         }
         tarch::parallel::Node::getInstance().receiveDanglingMessages();
      }
      
      delete sendRequestHandle;
      #ifdef Debug
      _log.debug("send(int,int)", "sent " + toString() );
      #endif
      
   }
   
   
   
   void peanoclaw::records::DataPacked::receive(int source, int tag) {
      MPI_Request* sendRequestHandle = new MPI_Request();
      MPI_Status   status;
      int          flag = 0;
      int          result;
      
      clock_t      timeOutWarning   = -1;
      clock_t      timeOutShutdown  = -1;
      bool         triggeredTimeoutWarning = false;
      
      result = MPI_Irecv(
         this, 1, Datatype, source, tag,
         tarch::parallel::Node::getInstance().getCommunicator(), sendRequestHandle
      );
      if ( result != MPI_SUCCESS ) {
         std::ostringstream msg;
         msg << "failed to start to receive peanoclaw::records::DataPacked from node "
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
            msg << "testing for finished receive task for peanoclaw::records::DataPacked failed: "
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
            "peanoclaw::records::DataPacked",
            "receive(int)", source, tag, 1
            );
            triggeredTimeoutWarning = true;
         }
         if (
            tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
            (clock()>timeOutShutdown)
         ) {
            tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
            "peanoclaw::records::DataPacked",
            "receive(int)", source, tag, 1
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
   
   
   
   bool peanoclaw::records::DataPacked::isMessageInQueue(int tag) {
      MPI_Status status;
      int  flag        = 0;
      MPI_Iprobe(
         MPI_ANY_SOURCE, tag,
         tarch::parallel::Node::getInstance().getCommunicator(), &flag, &status
      );
      if (flag) {
         int  messageCounter;
         MPI_Get_count(&status, Datatype, &messageCounter);
         return messageCounter > 0;
      }
      else return false;
      
   }
   
   int peanoclaw::records::DataPacked::getSenderRank() const {
      assertion( _senderRank!=-1 );
      return _senderRank;
      
   }
#endif



