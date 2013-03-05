#include "peanoclaw/records/VertexDescription.h"

peanoclaw::records::VertexDescription::PersistentRecords::PersistentRecords() {
   
}


peanoclaw::records::VertexDescription::PersistentRecords::PersistentRecords(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const IterationParity& lastUpdateIterationParity, const bool& touched):
_indicesOfAdjacentCellDescriptions(indicesOfAdjacentCellDescriptions),
_lastUpdateIterationParity(lastUpdateIterationParity),
_touched(touched) {
   
}


tarch::la::Vector<TWO_POWER_D,int> peanoclaw::records::VertexDescription::PersistentRecords::getIndicesOfAdjacentCellDescriptions() const {
   return _indicesOfAdjacentCellDescriptions;
}



void peanoclaw::records::VertexDescription::PersistentRecords::setIndicesOfAdjacentCellDescriptions(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions) {
   _indicesOfAdjacentCellDescriptions = (indicesOfAdjacentCellDescriptions);
}



peanoclaw::records::VertexDescription::IterationParity peanoclaw::records::VertexDescription::PersistentRecords::getLastUpdateIterationParity() const {
   return _lastUpdateIterationParity;
}



void peanoclaw::records::VertexDescription::PersistentRecords::setLastUpdateIterationParity(const IterationParity& lastUpdateIterationParity) {
   _lastUpdateIterationParity = lastUpdateIterationParity;
}



bool peanoclaw::records::VertexDescription::PersistentRecords::getTouched() const {
   return _touched;
}



void peanoclaw::records::VertexDescription::PersistentRecords::setTouched(const bool& touched) {
   _touched = touched;
}


peanoclaw::records::VertexDescription::VertexDescription() {
   
}


peanoclaw::records::VertexDescription::VertexDescription(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._indicesOfAdjacentCellDescriptions, persistentRecords._lastUpdateIterationParity, persistentRecords._touched) {
   
}


peanoclaw::records::VertexDescription::VertexDescription(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const IterationParity& lastUpdateIterationParity, const bool& touched):
_persistentRecords(indicesOfAdjacentCellDescriptions, lastUpdateIterationParity, touched) {
   
}


peanoclaw::records::VertexDescription::~VertexDescription() { }


tarch::la::Vector<TWO_POWER_D,int> peanoclaw::records::VertexDescription::getIndicesOfAdjacentCellDescriptions() const {
   return _persistentRecords._indicesOfAdjacentCellDescriptions;
}



void peanoclaw::records::VertexDescription::setIndicesOfAdjacentCellDescriptions(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions) {
   _persistentRecords._indicesOfAdjacentCellDescriptions = (indicesOfAdjacentCellDescriptions);
}



int peanoclaw::records::VertexDescription::getIndicesOfAdjacentCellDescriptions(int elementIndex) const {
   assertion(elementIndex>=0);
   assertion(elementIndex<TWO_POWER_D);
   return _persistentRecords._indicesOfAdjacentCellDescriptions[elementIndex];
   
}



void peanoclaw::records::VertexDescription::setIndicesOfAdjacentCellDescriptions(int elementIndex, const int& indicesOfAdjacentCellDescriptions) {
   assertion(elementIndex>=0);
   assertion(elementIndex<TWO_POWER_D);
   _persistentRecords._indicesOfAdjacentCellDescriptions[elementIndex]= indicesOfAdjacentCellDescriptions;
   
}



peanoclaw::records::VertexDescription::IterationParity peanoclaw::records::VertexDescription::getLastUpdateIterationParity() const {
   return _persistentRecords._lastUpdateIterationParity;
}



void peanoclaw::records::VertexDescription::setLastUpdateIterationParity(const IterationParity& lastUpdateIterationParity) {
   _persistentRecords._lastUpdateIterationParity = lastUpdateIterationParity;
}



bool peanoclaw::records::VertexDescription::getTouched() const {
   return _persistentRecords._touched;
}



void peanoclaw::records::VertexDescription::setTouched(const bool& touched) {
   _persistentRecords._touched = touched;
}


std::string peanoclaw::records::VertexDescription::toString(const IterationParity& param) {
   switch (param) {
      case EVEN: return "EVEN";
      case ODD: return "ODD";
   }
   return "undefined";
}

std::string peanoclaw::records::VertexDescription::getIterationParityMapping() {
   return "IterationParity(EVEN=0,ODD=1)";
}


std::string peanoclaw::records::VertexDescription::toString() const {
   std::ostringstream stringstr;
   toString(stringstr);
   return stringstr.str();
}

void peanoclaw::records::VertexDescription::toString (std::ostream& out) const {
   out << "("; 
   out << "indicesOfAdjacentCellDescriptions:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getIndicesOfAdjacentCellDescriptions(i) << ",";
   }
   out << getIndicesOfAdjacentCellDescriptions(TWO_POWER_D-1) << "]";
   out << ",";
   out << "lastUpdateIterationParity:" << toString(getLastUpdateIterationParity());
   out << ",";
   out << "touched:" << getTouched();
   out <<  ")";
}


peanoclaw::records::VertexDescription::PersistentRecords peanoclaw::records::VertexDescription::getPersistentRecords() const {
   return _persistentRecords;
}

peanoclaw::records::VertexDescriptionPacked peanoclaw::records::VertexDescription::convert() const{
   return VertexDescriptionPacked(
      getIndicesOfAdjacentCellDescriptions(),
      getLastUpdateIterationParity(),
      getTouched()
   );
}

#ifdef Parallel
   #include "tarch/parallel/Node.h" 
   
   tarch::logging::Log peanoclaw::records::VertexDescription::_log( "peanoclaw::records::VertexDescription" );
   
   MPI_Datatype peanoclaw::records::VertexDescription::Datatype = 0;
   
   
   void peanoclaw::records::VertexDescription::initDatatype() {
      const int Attributes = 4;
      MPI_Datatype subtypes[Attributes] = {
         MPI_INT,		 //indicesOfAdjacentCellDescriptions
         MPI_INT,		 //lastUpdateIterationParity
         MPI_CHAR,		 //touched
         MPI_UB		 // end/displacement flag
      };
      
      int blocklen[Attributes] = {
         TWO_POWER_D,		 //indicesOfAdjacentCellDescriptions
         1,		 //lastUpdateIterationParity
         1,		 //touched
         1		 // end/displacement flag
      };
      
      MPI_Aint     disp[Attributes];
      VertexDescription dummyVertexDescription[2];
      
      MPI_Aint base;
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexDescription[0]))), &base);
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexDescription[0]._persistentRecords._indicesOfAdjacentCellDescriptions[0]))), 		&disp[0] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexDescription[0]._persistentRecords._lastUpdateIterationParity))), 		&disp[1] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexDescription[0]._persistentRecords._touched))), 		&disp[2] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyVertexDescription[1]._persistentRecords._indicesOfAdjacentCellDescriptions[0])), 		&disp[3] );
      
      for (int i=1; i<Attributes; i++) {
         assertion1( disp[i] > disp[i-1], i );
      }
      for (int i=0; i<Attributes; i++) {
         disp[i] -= base;
      }
      MPI_Type_struct( Attributes, blocklen, disp, subtypes, &VertexDescription::Datatype );
      MPI_Type_commit( &VertexDescription::Datatype );
      
   }
   
   
   void peanoclaw::records::VertexDescription::shutdownDatatype() {
      MPI_Type_free( &VertexDescription::Datatype );
      
   }
   
   void peanoclaw::records::VertexDescription::send(int destination, int tag) {
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
         msg << "was not able to send message peanoclaw::records::VertexDescription "
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
            msg << "testing for finished send task for peanoclaw::records::VertexDescription "
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
            "peanoclaw::records::VertexDescription",
            "send(int)", destination, tag, 1
            );
            triggeredTimeoutWarning = true;
         }
         if (
            tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
            (clock()>timeOutShutdown)
         ) {
            tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
            "peanoclaw::records::VertexDescription",
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
   
   
   
   void peanoclaw::records::VertexDescription::receive(int source, int tag) {
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
         msg << "failed to start to receive peanoclaw::records::VertexDescription from node "
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
            msg << "testing for finished receive task for peanoclaw::records::VertexDescription failed: "
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
            "peanoclaw::records::VertexDescription",
            "receive(int)", source, tag, 1
            );
            triggeredTimeoutWarning = true;
         }
         if (
            tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
            (clock()>timeOutShutdown)
         ) {
            tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
            "peanoclaw::records::VertexDescription",
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
   
   
   
   bool peanoclaw::records::VertexDescription::isMessageInQueue(int tag) {
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
   
   int peanoclaw::records::VertexDescription::getSenderRank() const {
      assertion( _senderRank!=-1 );
      return _senderRank;
      
   }
#endif


peanoclaw::records::VertexDescriptionPacked::PersistentRecords::PersistentRecords() {
   
}


peanoclaw::records::VertexDescriptionPacked::PersistentRecords::PersistentRecords(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const IterationParity& lastUpdateIterationParity, const bool& touched):
_indicesOfAdjacentCellDescriptions(indicesOfAdjacentCellDescriptions),
_lastUpdateIterationParity(lastUpdateIterationParity),
_touched(touched) {
   
}


tarch::la::Vector<TWO_POWER_D,int> peanoclaw::records::VertexDescriptionPacked::PersistentRecords::getIndicesOfAdjacentCellDescriptions() const {
   return _indicesOfAdjacentCellDescriptions;
}



void peanoclaw::records::VertexDescriptionPacked::PersistentRecords::setIndicesOfAdjacentCellDescriptions(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions) {
   _indicesOfAdjacentCellDescriptions = (indicesOfAdjacentCellDescriptions);
}



peanoclaw::records::VertexDescription::IterationParity peanoclaw::records::VertexDescriptionPacked::PersistentRecords::getLastUpdateIterationParity() const {
   return _lastUpdateIterationParity;
}



void peanoclaw::records::VertexDescriptionPacked::PersistentRecords::setLastUpdateIterationParity(const IterationParity& lastUpdateIterationParity) {
   _lastUpdateIterationParity = lastUpdateIterationParity;
}



bool peanoclaw::records::VertexDescriptionPacked::PersistentRecords::getTouched() const {
   return _touched;
}



void peanoclaw::records::VertexDescriptionPacked::PersistentRecords::setTouched(const bool& touched) {
   _touched = touched;
}


peanoclaw::records::VertexDescriptionPacked::VertexDescriptionPacked() {
   
}


peanoclaw::records::VertexDescriptionPacked::VertexDescriptionPacked(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._indicesOfAdjacentCellDescriptions, persistentRecords._lastUpdateIterationParity, persistentRecords._touched) {
   
}


peanoclaw::records::VertexDescriptionPacked::VertexDescriptionPacked(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const IterationParity& lastUpdateIterationParity, const bool& touched):
_persistentRecords(indicesOfAdjacentCellDescriptions, lastUpdateIterationParity, touched) {
   
}


peanoclaw::records::VertexDescriptionPacked::~VertexDescriptionPacked() { }


tarch::la::Vector<TWO_POWER_D,int> peanoclaw::records::VertexDescriptionPacked::getIndicesOfAdjacentCellDescriptions() const {
   return _persistentRecords._indicesOfAdjacentCellDescriptions;
}



void peanoclaw::records::VertexDescriptionPacked::setIndicesOfAdjacentCellDescriptions(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions) {
   _persistentRecords._indicesOfAdjacentCellDescriptions = (indicesOfAdjacentCellDescriptions);
}



int peanoclaw::records::VertexDescriptionPacked::getIndicesOfAdjacentCellDescriptions(int elementIndex) const {
   assertion(elementIndex>=0);
   assertion(elementIndex<TWO_POWER_D);
   return _persistentRecords._indicesOfAdjacentCellDescriptions[elementIndex];
   
}



void peanoclaw::records::VertexDescriptionPacked::setIndicesOfAdjacentCellDescriptions(int elementIndex, const int& indicesOfAdjacentCellDescriptions) {
   assertion(elementIndex>=0);
   assertion(elementIndex<TWO_POWER_D);
   _persistentRecords._indicesOfAdjacentCellDescriptions[elementIndex]= indicesOfAdjacentCellDescriptions;
   
}



peanoclaw::records::VertexDescription::IterationParity peanoclaw::records::VertexDescriptionPacked::getLastUpdateIterationParity() const {
   return _persistentRecords._lastUpdateIterationParity;
}



void peanoclaw::records::VertexDescriptionPacked::setLastUpdateIterationParity(const IterationParity& lastUpdateIterationParity) {
   _persistentRecords._lastUpdateIterationParity = lastUpdateIterationParity;
}



bool peanoclaw::records::VertexDescriptionPacked::getTouched() const {
   return _persistentRecords._touched;
}



void peanoclaw::records::VertexDescriptionPacked::setTouched(const bool& touched) {
   _persistentRecords._touched = touched;
}


std::string peanoclaw::records::VertexDescriptionPacked::toString(const IterationParity& param) {
   return peanoclaw::records::VertexDescription::toString(param);
}

std::string peanoclaw::records::VertexDescriptionPacked::getIterationParityMapping() {
   return peanoclaw::records::VertexDescription::getIterationParityMapping();
}



std::string peanoclaw::records::VertexDescriptionPacked::toString() const {
   std::ostringstream stringstr;
   toString(stringstr);
   return stringstr.str();
}

void peanoclaw::records::VertexDescriptionPacked::toString (std::ostream& out) const {
   out << "("; 
   out << "indicesOfAdjacentCellDescriptions:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getIndicesOfAdjacentCellDescriptions(i) << ",";
   }
   out << getIndicesOfAdjacentCellDescriptions(TWO_POWER_D-1) << "]";
   out << ",";
   out << "lastUpdateIterationParity:" << toString(getLastUpdateIterationParity());
   out << ",";
   out << "touched:" << getTouched();
   out <<  ")";
}


peanoclaw::records::VertexDescriptionPacked::PersistentRecords peanoclaw::records::VertexDescriptionPacked::getPersistentRecords() const {
   return _persistentRecords;
}

peanoclaw::records::VertexDescription peanoclaw::records::VertexDescriptionPacked::convert() const{
   return VertexDescription(
      getIndicesOfAdjacentCellDescriptions(),
      getLastUpdateIterationParity(),
      getTouched()
   );
}

#ifdef Parallel
   #include "tarch/parallel/Node.h" 
   
   tarch::logging::Log peanoclaw::records::VertexDescriptionPacked::_log( "peanoclaw::records::VertexDescriptionPacked" );
   
   MPI_Datatype peanoclaw::records::VertexDescriptionPacked::Datatype = 0;
   
   
   void peanoclaw::records::VertexDescriptionPacked::initDatatype() {
      const int Attributes = 4;
      MPI_Datatype subtypes[Attributes] = {
         MPI_INT,		 //indicesOfAdjacentCellDescriptions
         MPI_INT,		 //lastUpdateIterationParity
         MPI_CHAR,		 //touched
         MPI_UB		 // end/displacement flag
      };
      
      int blocklen[Attributes] = {
         TWO_POWER_D,		 //indicesOfAdjacentCellDescriptions
         1,		 //lastUpdateIterationParity
         1,		 //touched
         1		 // end/displacement flag
      };
      
      MPI_Aint     disp[Attributes];
      VertexDescriptionPacked dummyVertexDescriptionPacked[2];
      
      MPI_Aint base;
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexDescriptionPacked[0]))), &base);
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexDescriptionPacked[0]._persistentRecords._indicesOfAdjacentCellDescriptions[0]))), 		&disp[0] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexDescriptionPacked[0]._persistentRecords._lastUpdateIterationParity))), 		&disp[1] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexDescriptionPacked[0]._persistentRecords._touched))), 		&disp[2] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyVertexDescriptionPacked[1]._persistentRecords._indicesOfAdjacentCellDescriptions[0])), 		&disp[3] );
      
      for (int i=1; i<Attributes; i++) {
         assertion1( disp[i] > disp[i-1], i );
      }
      for (int i=0; i<Attributes; i++) {
         disp[i] -= base;
      }
      MPI_Type_struct( Attributes, blocklen, disp, subtypes, &VertexDescriptionPacked::Datatype );
      MPI_Type_commit( &VertexDescriptionPacked::Datatype );
      
   }
   
   
   void peanoclaw::records::VertexDescriptionPacked::shutdownDatatype() {
      MPI_Type_free( &VertexDescriptionPacked::Datatype );
      
   }
   
   void peanoclaw::records::VertexDescriptionPacked::send(int destination, int tag) {
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
         msg << "was not able to send message peanoclaw::records::VertexDescriptionPacked "
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
            msg << "testing for finished send task for peanoclaw::records::VertexDescriptionPacked "
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
            "peanoclaw::records::VertexDescriptionPacked",
            "send(int)", destination, tag, 1
            );
            triggeredTimeoutWarning = true;
         }
         if (
            tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
            (clock()>timeOutShutdown)
         ) {
            tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
            "peanoclaw::records::VertexDescriptionPacked",
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
   
   
   
   void peanoclaw::records::VertexDescriptionPacked::receive(int source, int tag) {
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
         msg << "failed to start to receive peanoclaw::records::VertexDescriptionPacked from node "
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
            msg << "testing for finished receive task for peanoclaw::records::VertexDescriptionPacked failed: "
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
            "peanoclaw::records::VertexDescriptionPacked",
            "receive(int)", source, tag, 1
            );
            triggeredTimeoutWarning = true;
         }
         if (
            tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
            (clock()>timeOutShutdown)
         ) {
            tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
            "peanoclaw::records::VertexDescriptionPacked",
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
   
   
   
   bool peanoclaw::records::VertexDescriptionPacked::isMessageInQueue(int tag) {
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
   
   int peanoclaw::records::VertexDescriptionPacked::getSenderRank() const {
      assertion( _senderRank!=-1 );
      return _senderRank;
      
   }
#endif



