#include "peanoclaw/records/RepositoryState.h"

peanoclaw::records::RepositoryState::PersistentRecords::PersistentRecords() {
   
}


peanoclaw::records::RepositoryState::PersistentRecords::PersistentRecords(const Action& action, const int& numberOfIterations):
_action(action),
_numberOfIterations(numberOfIterations) {
   
}

peanoclaw::records::RepositoryState::RepositoryState() {
   
}


peanoclaw::records::RepositoryState::RepositoryState(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._action, persistentRecords._numberOfIterations) {
   
}


peanoclaw::records::RepositoryState::RepositoryState(const Action& action, const int& numberOfIterations):
_persistentRecords(action, numberOfIterations) {
   
}


peanoclaw::records::RepositoryState::~RepositoryState() { }

std::string peanoclaw::records::RepositoryState::toString(const Action& param) {
   switch (param) {
      case WriteCheckpoint: return "WriteCheckpoint";
      case ReadCheckpoint: return "ReadCheckpoint";
      case Terminate: return "Terminate";
      case RunOnAllNodes: return "RunOnAllNodes";
      case UseAdapterInitialiseGrid: return "UseAdapterInitialiseGrid";
      case UseAdapterInitialiseAndValidateGrid: return "UseAdapterInitialiseAndValidateGrid";
      case UseAdapterPlot: return "UseAdapterPlot";
      case UseAdapterPlotAndValidateGrid: return "UseAdapterPlotAndValidateGrid";
      case UseAdapterRemesh: return "UseAdapterRemesh";
      case UseAdapterSolveTimestep: return "UseAdapterSolveTimestep";
      case UseAdapterSolveTimestepAndValidateGrid: return "UseAdapterSolveTimestepAndValidateGrid";
      case UseAdapterSolveTimestepAndPlot: return "UseAdapterSolveTimestepAndPlot";
      case UseAdapterSolveTimestepAndPlotAndValidateGrid: return "UseAdapterSolveTimestepAndPlotAndValidateGrid";
      case UseAdapterGatherCurrentSolution: return "UseAdapterGatherCurrentSolution";
      case UseAdapterGatherCurrentSolutionAndValidateGrid: return "UseAdapterGatherCurrentSolutionAndValidateGrid";
      case UseAdapterCleanup: return "UseAdapterCleanup";
      case NumberOfAdapters: return "NumberOfAdapters";
   }
   return "undefined";
}

std::string peanoclaw::records::RepositoryState::getActionMapping() {
   return "Action(WriteCheckpoint=0,ReadCheckpoint=1,Terminate=2,RunOnAllNodes=3,UseAdapterInitialiseGrid=4,UseAdapterInitialiseAndValidateGrid=5,UseAdapterPlot=6,UseAdapterPlotAndValidateGrid=7,UseAdapterRemesh=8,UseAdapterSolveTimestep=9,UseAdapterSolveTimestepAndValidateGrid=10,UseAdapterSolveTimestepAndPlot=11,UseAdapterSolveTimestepAndPlotAndValidateGrid=12,UseAdapterGatherCurrentSolution=13,UseAdapterGatherCurrentSolutionAndValidateGrid=14,UseAdapterCleanup=15,NumberOfAdapters=16)";
}


std::string peanoclaw::records::RepositoryState::toString() const {
   std::ostringstream stringstr;
   toString(stringstr);
   return stringstr.str();
}

void peanoclaw::records::RepositoryState::toString (std::ostream& out) const {
   out << "("; 
   out << "action:" << toString(getAction());
   out << ",";
   out << "numberOfIterations:" << getNumberOfIterations();
   out <<  ")";
}


peanoclaw::records::RepositoryState::PersistentRecords peanoclaw::records::RepositoryState::getPersistentRecords() const {
   return _persistentRecords;
}

peanoclaw::records::RepositoryStatePacked peanoclaw::records::RepositoryState::convert() const{
   return RepositoryStatePacked(
      getAction(),
      getNumberOfIterations()
   );
}

#ifdef Parallel
   tarch::logging::Log peanoclaw::records::RepositoryState::_log( "peanoclaw::records::RepositoryState" );
   
   MPI_Datatype peanoclaw::records::RepositoryState::Datatype = 0;
   MPI_Datatype peanoclaw::records::RepositoryState::FullDatatype = 0;
   
   
   void peanoclaw::records::RepositoryState::initDatatype() {
      {
         RepositoryState dummyRepositoryState[2];
         
         const int Attributes = 3;
         MPI_Datatype subtypes[Attributes] = {
            MPI_INT,		 //action
            MPI_INT,		 //numberOfIterations
            MPI_UB		 // end/displacement flag
         };
         
         int blocklen[Attributes] = {
            1,		 //action
            1,		 //numberOfIterations
            1		 // end/displacement flag
         };
         
         MPI_Aint     disp[Attributes];
         
         MPI_Aint base;
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyRepositoryState[0]))), &base);
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyRepositoryState[0]._persistentRecords._action))), 		&disp[0] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyRepositoryState[0]._persistentRecords._numberOfIterations))), 		&disp[1] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyRepositoryState[1]._persistentRecords._action))), 		&disp[2] );
         
         for (int i=1; i<Attributes; i++) {
            assertion1( disp[i] > disp[i-1], i );
         }
         for (int i=0; i<Attributes; i++) {
            disp[i] -= base;
         }
         MPI_Type_struct( Attributes, blocklen, disp, subtypes, &RepositoryState::Datatype );
         MPI_Type_commit( &RepositoryState::Datatype );
         
      }
      {
         RepositoryState dummyRepositoryState[2];
         
         const int Attributes = 3;
         MPI_Datatype subtypes[Attributes] = {
            MPI_INT,		 //action
            MPI_INT,		 //numberOfIterations
            MPI_UB		 // end/displacement flag
         };
         
         int blocklen[Attributes] = {
            1,		 //action
            1,		 //numberOfIterations
            1		 // end/displacement flag
         };
         
         MPI_Aint     disp[Attributes];
         
         MPI_Aint base;
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyRepositoryState[0]))), &base);
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyRepositoryState[0]._persistentRecords._action))), 		&disp[0] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyRepositoryState[0]._persistentRecords._numberOfIterations))), 		&disp[1] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyRepositoryState[1]._persistentRecords._action))), 		&disp[2] );
         
         for (int i=1; i<Attributes; i++) {
            assertion1( disp[i] > disp[i-1], i );
         }
         for (int i=0; i<Attributes; i++) {
            disp[i] -= base;
         }
         MPI_Type_struct( Attributes, blocklen, disp, subtypes, &RepositoryState::FullDatatype );
         MPI_Type_commit( &RepositoryState::FullDatatype );
         
      }
      
   }
   
   
   void peanoclaw::records::RepositoryState::shutdownDatatype() {
      MPI_Type_free( &RepositoryState::Datatype );
      MPI_Type_free( &RepositoryState::FullDatatype );
      
   }
   
   void peanoclaw::records::RepositoryState::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
      _senderDestinationRank = destination;
      
      if (communicateBlocking) {
      
         const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
         if  (result!=MPI_SUCCESS) {
            std::ostringstream msg;
            msg << "was not able to send message peanoclaw::records::RepositoryState "
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
         msg << "was not able to send message peanoclaw::records::RepositoryState "
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
            msg << "testing for finished send task for peanoclaw::records::RepositoryState "
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
            "peanoclaw::records::RepositoryState",
            "send(int)", destination,tag,1
            );
            triggeredTimeoutWarning = true;
         }
         if (
            tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
            (clock()>timeOutShutdown)
         ) {
            tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
            "peanoclaw::records::RepositoryState",
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



void peanoclaw::records::RepositoryState::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
   if (communicateBlocking) {
   
      MPI_Status  status;
      const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
      _senderDestinationRank = status.MPI_SOURCE;
      if ( result != MPI_SUCCESS ) {
         std::ostringstream msg;
         msg << "failed to start to receive peanoclaw::records::RepositoryState from node "
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
         msg << "failed to start to receive peanoclaw::records::RepositoryState from node "
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
            msg << "testing for finished receive task for peanoclaw::records::RepositoryState failed: "
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
            "peanoclaw::records::RepositoryState",
            "receive(int)", source,tag,1
            );
            triggeredTimeoutWarning = true;
         }
         if (
            tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
            (clock()>timeOutShutdown)
         ) {
            tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
            "peanoclaw::records::RepositoryState",
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
   
}



bool peanoclaw::records::RepositoryState::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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

int peanoclaw::records::RepositoryState::getSenderRank() const {
   assertion( _senderDestinationRank!=-1 );
   return _senderDestinationRank;
   
}
#endif


peanoclaw::records::RepositoryStatePacked::PersistentRecords::PersistentRecords() {

}


peanoclaw::records::RepositoryStatePacked::PersistentRecords::PersistentRecords(const Action& action, const int& numberOfIterations):
_action(action),
_numberOfIterations(numberOfIterations) {

}

peanoclaw::records::RepositoryStatePacked::RepositoryStatePacked() {

}


peanoclaw::records::RepositoryStatePacked::RepositoryStatePacked(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._action, persistentRecords._numberOfIterations) {

}


peanoclaw::records::RepositoryStatePacked::RepositoryStatePacked(const Action& action, const int& numberOfIterations):
_persistentRecords(action, numberOfIterations) {

}


peanoclaw::records::RepositoryStatePacked::~RepositoryStatePacked() { }

std::string peanoclaw::records::RepositoryStatePacked::toString(const Action& param) {
return peanoclaw::records::RepositoryState::toString(param);
}

std::string peanoclaw::records::RepositoryStatePacked::getActionMapping() {
return peanoclaw::records::RepositoryState::getActionMapping();
}



std::string peanoclaw::records::RepositoryStatePacked::toString() const {
std::ostringstream stringstr;
toString(stringstr);
return stringstr.str();
}

void peanoclaw::records::RepositoryStatePacked::toString (std::ostream& out) const {
out << "("; 
out << "action:" << toString(getAction());
out << ",";
out << "numberOfIterations:" << getNumberOfIterations();
out <<  ")";
}


peanoclaw::records::RepositoryStatePacked::PersistentRecords peanoclaw::records::RepositoryStatePacked::getPersistentRecords() const {
return _persistentRecords;
}

peanoclaw::records::RepositoryState peanoclaw::records::RepositoryStatePacked::convert() const{
return RepositoryState(
   getAction(),
   getNumberOfIterations()
);
}

#ifdef Parallel
tarch::logging::Log peanoclaw::records::RepositoryStatePacked::_log( "peanoclaw::records::RepositoryStatePacked" );

MPI_Datatype peanoclaw::records::RepositoryStatePacked::Datatype = 0;
MPI_Datatype peanoclaw::records::RepositoryStatePacked::FullDatatype = 0;


void peanoclaw::records::RepositoryStatePacked::initDatatype() {
   {
      RepositoryStatePacked dummyRepositoryStatePacked[2];
      
      const int Attributes = 3;
      MPI_Datatype subtypes[Attributes] = {
         MPI_INT,		 //action
         MPI_INT,		 //numberOfIterations
         MPI_UB		 // end/displacement flag
      };
      
      int blocklen[Attributes] = {
         1,		 //action
         1,		 //numberOfIterations
         1		 // end/displacement flag
      };
      
      MPI_Aint     disp[Attributes];
      
      MPI_Aint base;
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyRepositoryStatePacked[0]))), &base);
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyRepositoryStatePacked[0]._persistentRecords._action))), 		&disp[0] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyRepositoryStatePacked[0]._persistentRecords._numberOfIterations))), 		&disp[1] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyRepositoryStatePacked[1]._persistentRecords._action))), 		&disp[2] );
      
      for (int i=1; i<Attributes; i++) {
         assertion1( disp[i] > disp[i-1], i );
      }
      for (int i=0; i<Attributes; i++) {
         disp[i] -= base;
      }
      MPI_Type_struct( Attributes, blocklen, disp, subtypes, &RepositoryStatePacked::Datatype );
      MPI_Type_commit( &RepositoryStatePacked::Datatype );
      
   }
   {
      RepositoryStatePacked dummyRepositoryStatePacked[2];
      
      const int Attributes = 3;
      MPI_Datatype subtypes[Attributes] = {
         MPI_INT,		 //action
         MPI_INT,		 //numberOfIterations
         MPI_UB		 // end/displacement flag
      };
      
      int blocklen[Attributes] = {
         1,		 //action
         1,		 //numberOfIterations
         1		 // end/displacement flag
      };
      
      MPI_Aint     disp[Attributes];
      
      MPI_Aint base;
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyRepositoryStatePacked[0]))), &base);
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyRepositoryStatePacked[0]._persistentRecords._action))), 		&disp[0] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyRepositoryStatePacked[0]._persistentRecords._numberOfIterations))), 		&disp[1] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyRepositoryStatePacked[1]._persistentRecords._action))), 		&disp[2] );
      
      for (int i=1; i<Attributes; i++) {
         assertion1( disp[i] > disp[i-1], i );
      }
      for (int i=0; i<Attributes; i++) {
         disp[i] -= base;
      }
      MPI_Type_struct( Attributes, blocklen, disp, subtypes, &RepositoryStatePacked::FullDatatype );
      MPI_Type_commit( &RepositoryStatePacked::FullDatatype );
      
   }
   
}


void peanoclaw::records::RepositoryStatePacked::shutdownDatatype() {
   MPI_Type_free( &RepositoryStatePacked::Datatype );
   MPI_Type_free( &RepositoryStatePacked::FullDatatype );
   
}

void peanoclaw::records::RepositoryStatePacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
   _senderDestinationRank = destination;
   
   if (communicateBlocking) {
   
      const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
      if  (result!=MPI_SUCCESS) {
         std::ostringstream msg;
         msg << "was not able to send message peanoclaw::records::RepositoryStatePacked "
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
      msg << "was not able to send message peanoclaw::records::RepositoryStatePacked "
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
         msg << "testing for finished send task for peanoclaw::records::RepositoryStatePacked "
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
         "peanoclaw::records::RepositoryStatePacked",
         "send(int)", destination,tag,1
         );
         triggeredTimeoutWarning = true;
      }
      if (
         tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
         (clock()>timeOutShutdown)
      ) {
         tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
         "peanoclaw::records::RepositoryStatePacked",
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



void peanoclaw::records::RepositoryStatePacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
if (communicateBlocking) {

   MPI_Status  status;
   const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
   _senderDestinationRank = status.MPI_SOURCE;
   if ( result != MPI_SUCCESS ) {
      std::ostringstream msg;
      msg << "failed to start to receive peanoclaw::records::RepositoryStatePacked from node "
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
      msg << "failed to start to receive peanoclaw::records::RepositoryStatePacked from node "
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
         msg << "testing for finished receive task for peanoclaw::records::RepositoryStatePacked failed: "
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
         "peanoclaw::records::RepositoryStatePacked",
         "receive(int)", source,tag,1
         );
         triggeredTimeoutWarning = true;
      }
      if (
         tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
         (clock()>timeOutShutdown)
      ) {
         tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
         "peanoclaw::records::RepositoryStatePacked",
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

}



bool peanoclaw::records::RepositoryStatePacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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

int peanoclaw::records::RepositoryStatePacked::getSenderRank() const {
assertion( _senderDestinationRank!=-1 );
return _senderDestinationRank;

}
#endif



