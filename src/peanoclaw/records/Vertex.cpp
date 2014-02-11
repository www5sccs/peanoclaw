#include "peanoclaw/records/Vertex.h"

#if defined(Parallel) && defined(Asserts)
   peanoclaw::records::Vertex::PersistentRecords::PersistentRecords() {
      
   }
   
   
   peanoclaw::records::Vertex::PersistentRecords::PersistentRecords(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank):
   _indicesOfAdjacentCellDescriptions(indicesOfAdjacentCellDescriptions),
   _adjacentSubcellsEraseVeto(adjacentSubcellsEraseVeto),
   _adjacentRanksInFormerIteration(adjacentRanksInFormerIteration),
   _adjacentRanksChanged(adjacentRanksChanged),
   _shouldRefine(shouldRefine),
   _ageInGridIterations(ageInGridIterations),
   _isHangingNode(isHangingNode),
   _refinementControl(refinementControl),
   _adjacentCellsHeight(adjacentCellsHeight),
   _insideOutsideDomain(insideOutsideDomain),
   _x(x),
   _level(level),
   _adjacentRanks(adjacentRanks),
   _adjacentSubtreeForksIntoOtherRank(adjacentSubtreeForksIntoOtherRank) {
      
   }
   
   peanoclaw::records::Vertex::Vertex() {
      
   }
   
   
   peanoclaw::records::Vertex::Vertex(const PersistentRecords& persistentRecords):
   _persistentRecords(persistentRecords._indicesOfAdjacentCellDescriptions, persistentRecords._adjacentSubcellsEraseVeto, persistentRecords._adjacentRanksInFormerIteration, persistentRecords._adjacentRanksChanged, persistentRecords._shouldRefine, persistentRecords._ageInGridIterations, persistentRecords._isHangingNode, persistentRecords._refinementControl, persistentRecords._adjacentCellsHeight, persistentRecords._insideOutsideDomain, persistentRecords._x, persistentRecords._level, persistentRecords._adjacentRanks, persistentRecords._adjacentSubtreeForksIntoOtherRank) {
      
   }
   
   
   peanoclaw::records::Vertex::Vertex(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank):
   _persistentRecords(indicesOfAdjacentCellDescriptions, adjacentSubcellsEraseVeto, adjacentRanksInFormerIteration, adjacentRanksChanged, shouldRefine, ageInGridIterations, isHangingNode, refinementControl, adjacentCellsHeight, insideOutsideDomain, x, level, adjacentRanks, adjacentSubtreeForksIntoOtherRank) {
      
   }
   
   
   peanoclaw::records::Vertex::Vertex(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank):
   _persistentRecords(indicesOfAdjacentCellDescriptions, adjacentSubcellsEraseVeto, adjacentRanksInFormerIteration, adjacentRanksChanged, shouldRefine, ageInGridIterations, isHangingNode, refinementControl, adjacentCellsHeight, insideOutsideDomain, x, level, adjacentRanks, adjacentSubtreeForksIntoOtherRank),_adjacentCellsHeightOfPreviousIteration(adjacentCellsHeightOfPreviousIteration),
   _numberOfAdjacentRefinedCells(numberOfAdjacentRefinedCells) {
      
   }
   
   peanoclaw::records::Vertex::~Vertex() { }
   
   std::string peanoclaw::records::Vertex::toString(const InsideOutsideDomain& param) {
      switch (param) {
         case Inside: return "Inside";
         case Boundary: return "Boundary";
         case Outside: return "Outside";
      }
      return "undefined";
   }
   
   std::string peanoclaw::records::Vertex::getInsideOutsideDomainMapping() {
      return "InsideOutsideDomain(Inside=0,Boundary=1,Outside=2)";
   }
   std::string peanoclaw::records::Vertex::toString(const RefinementControl& param) {
      switch (param) {
         case Unrefined: return "Unrefined";
         case Refined: return "Refined";
         case RefinementTriggered: return "RefinementTriggered";
         case Refining: return "Refining";
         case EraseTriggered: return "EraseTriggered";
         case Erasing: return "Erasing";
         case RefineDueToJoinThoughWorkerIsAlreadyErasing: return "RefineDueToJoinThoughWorkerIsAlreadyErasing";
      }
      return "undefined";
   }
   
   std::string peanoclaw::records::Vertex::getRefinementControlMapping() {
      return "RefinementControl(Unrefined=0,Refined=1,RefinementTriggered=2,Refining=3,EraseTriggered=4,Erasing=5,RefineDueToJoinThoughWorkerIsAlreadyErasing=6)";
   }
   
   
   std::string peanoclaw::records::Vertex::toString() const {
      std::ostringstream stringstr;
      toString(stringstr);
      return stringstr.str();
   }
   
   void peanoclaw::records::Vertex::toString (std::ostream& out) const {
      out << "("; 
      out << "indicesOfAdjacentCellDescriptions:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getIndicesOfAdjacentCellDescriptions(i) << ",";
   }
   out << getIndicesOfAdjacentCellDescriptions(TWO_POWER_D-1) << "]";
      out << ",";
      out << "adjacentSubcellsEraseVeto:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentSubcellsEraseVeto(i) << ",";
   }
   out << getAdjacentSubcellsEraseVeto(TWO_POWER_D-1) << "]";
      out << ",";
      out << "adjacentRanksInFormerIteration:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentRanksInFormerIteration(i) << ",";
   }
   out << getAdjacentRanksInFormerIteration(TWO_POWER_D-1) << "]";
      out << ",";
      out << "adjacentRanksChanged:" << getAdjacentRanksChanged();
      out << ",";
      out << "shouldRefine:" << getShouldRefine();
      out << ",";
      out << "ageInGridIterations:" << getAgeInGridIterations();
      out << ",";
      out << "isHangingNode:" << getIsHangingNode();
      out << ",";
      out << "refinementControl:" << toString(getRefinementControl());
      out << ",";
      out << "adjacentCellsHeight:" << getAdjacentCellsHeight();
      out << ",";
      out << "adjacentCellsHeightOfPreviousIteration:" << getAdjacentCellsHeightOfPreviousIteration();
      out << ",";
      out << "numberOfAdjacentRefinedCells:" << getNumberOfAdjacentRefinedCells();
      out << ",";
      out << "insideOutsideDomain:" << toString(getInsideOutsideDomain());
      out << ",";
      out << "x:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getX(i) << ",";
   }
   out << getX(DIMENSIONS-1) << "]";
      out << ",";
      out << "level:" << getLevel();
      out << ",";
      out << "adjacentRanks:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentRanks(i) << ",";
   }
   out << getAdjacentRanks(TWO_POWER_D-1) << "]";
      out << ",";
      out << "adjacentSubtreeForksIntoOtherRank:" << getAdjacentSubtreeForksIntoOtherRank();
      out <<  ")";
   }
   
   
   peanoclaw::records::Vertex::PersistentRecords peanoclaw::records::Vertex::getPersistentRecords() const {
      return _persistentRecords;
   }
   
   peanoclaw::records::VertexPacked peanoclaw::records::Vertex::convert() const{
      return VertexPacked(
         getIndicesOfAdjacentCellDescriptions(),
         getAdjacentSubcellsEraseVeto(),
         getAdjacentRanksInFormerIteration(),
         getAdjacentRanksChanged(),
         getShouldRefine(),
         getAgeInGridIterations(),
         getIsHangingNode(),
         getRefinementControl(),
         getAdjacentCellsHeight(),
         getAdjacentCellsHeightOfPreviousIteration(),
         getNumberOfAdjacentRefinedCells(),
         getInsideOutsideDomain(),
         getX(),
         getLevel(),
         getAdjacentRanks(),
         getAdjacentSubtreeForksIntoOtherRank()
      );
   }
   
   #ifdef Parallel
      tarch::logging::Log peanoclaw::records::Vertex::_log( "peanoclaw::records::Vertex" );
      
      MPI_Datatype peanoclaw::records::Vertex::Datatype = 0;
      MPI_Datatype peanoclaw::records::Vertex::FullDatatype = 0;
      
      
      void peanoclaw::records::Vertex::initDatatype() {
         {
            Vertex dummyVertex[2];
            
            const int Attributes = 14;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //indicesOfAdjacentCellDescriptions
               MPI_INT,		 //adjacentSubcellsEraseVeto
               MPI_CHAR,		 //adjacentRanksChanged
               MPI_CHAR,		 //shouldRefine
               MPI_INT,		 //ageInGridIterations
               MPI_CHAR,		 //isHangingNode
               MPI_INT,		 //refinementControl
               MPI_INT,		 //insideOutsideDomain
               MPI_DOUBLE,		 //x
               MPI_INT,		 //level
               MPI_INT,		 //adjacentRanks
               MPI_CHAR,		 //adjacentSubtreeForksIntoOtherRank
               MPI_INT,		 //numberOfAdjacentRefinedCells
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               TWO_POWER_D,		 //indicesOfAdjacentCellDescriptions
               TWO_POWER_D,		 //adjacentSubcellsEraseVeto
               1,		 //adjacentRanksChanged
               1,		 //shouldRefine
               1,		 //ageInGridIterations
               1,		 //isHangingNode
               1,		 //refinementControl
               1,		 //insideOutsideDomain
               DIMENSIONS,		 //x
               1,		 //level
               TWO_POWER_D,		 //adjacentRanks
               1,		 //adjacentSubtreeForksIntoOtherRank
               1,		 //numberOfAdjacentRefinedCells
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._indicesOfAdjacentCellDescriptions[0]))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentSubcellsEraseVeto))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentRanksChanged))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._shouldRefine))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._ageInGridIterations))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._isHangingNode))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._refinementControl))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._insideOutsideDomain))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._x[0]))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._level))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentRanks[0]))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentSubtreeForksIntoOtherRank))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._numberOfAdjacentRefinedCells))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyVertex[1]._persistentRecords._indicesOfAdjacentCellDescriptions[0])), 		&disp[13] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Vertex::Datatype );
            MPI_Type_commit( &Vertex::Datatype );
            
         }
         {
            Vertex dummyVertex[2];
            
            const int Attributes = 17;
            MPI_Datatype subtypes[Attributes] = {
               MPI_INT,		 //indicesOfAdjacentCellDescriptions
               MPI_INT,		 //adjacentSubcellsEraseVeto
               MPI_INT,		 //adjacentRanksInFormerIteration
               MPI_CHAR,		 //adjacentRanksChanged
               MPI_CHAR,		 //shouldRefine
               MPI_INT,		 //ageInGridIterations
               MPI_CHAR,		 //isHangingNode
               MPI_INT,		 //refinementControl
               MPI_INT,		 //adjacentCellsHeight
               MPI_INT,		 //insideOutsideDomain
               MPI_DOUBLE,		 //x
               MPI_INT,		 //level
               MPI_INT,		 //adjacentRanks
               MPI_CHAR,		 //adjacentSubtreeForksIntoOtherRank
               MPI_INT,		 //adjacentCellsHeightOfPreviousIteration
               MPI_INT,		 //numberOfAdjacentRefinedCells
               MPI_UB		 // end/displacement flag
            };
            
            int blocklen[Attributes] = {
               TWO_POWER_D,		 //indicesOfAdjacentCellDescriptions
               TWO_POWER_D,		 //adjacentSubcellsEraseVeto
               TWO_POWER_D,		 //adjacentRanksInFormerIteration
               1,		 //adjacentRanksChanged
               1,		 //shouldRefine
               1,		 //ageInGridIterations
               1,		 //isHangingNode
               1,		 //refinementControl
               1,		 //adjacentCellsHeight
               1,		 //insideOutsideDomain
               DIMENSIONS,		 //x
               1,		 //level
               TWO_POWER_D,		 //adjacentRanks
               1,		 //adjacentSubtreeForksIntoOtherRank
               1,		 //adjacentCellsHeightOfPreviousIteration
               1,		 //numberOfAdjacentRefinedCells
               1		 // end/displacement flag
            };
            
            MPI_Aint     disp[Attributes];
            
            MPI_Aint base;
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]))), &base);
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._indicesOfAdjacentCellDescriptions[0]))), 		&disp[0] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentSubcellsEraseVeto))), 		&disp[1] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentRanksInFormerIteration[0]))), 		&disp[2] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentRanksChanged))), 		&disp[3] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._shouldRefine))), 		&disp[4] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._ageInGridIterations))), 		&disp[5] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._isHangingNode))), 		&disp[6] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._refinementControl))), 		&disp[7] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentCellsHeight))), 		&disp[8] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._insideOutsideDomain))), 		&disp[9] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._x[0]))), 		&disp[10] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._level))), 		&disp[11] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentRanks[0]))), 		&disp[12] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentSubtreeForksIntoOtherRank))), 		&disp[13] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._adjacentCellsHeightOfPreviousIteration))), 		&disp[14] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._numberOfAdjacentRefinedCells))), 		&disp[15] );
            MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyVertex[1]._persistentRecords._indicesOfAdjacentCellDescriptions[0])), 		&disp[16] );
            
            for (int i=1; i<Attributes; i++) {
               assertion1( disp[i] > disp[i-1], i );
            }
            for (int i=0; i<Attributes; i++) {
               disp[i] -= base;
            }
            MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Vertex::FullDatatype );
            MPI_Type_commit( &Vertex::FullDatatype );
            
         }
         
      }
      
      
      void peanoclaw::records::Vertex::shutdownDatatype() {
         MPI_Type_free( &Vertex::Datatype );
         MPI_Type_free( &Vertex::FullDatatype );
         
      }
      
      void peanoclaw::records::Vertex::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
         _senderDestinationRank = destination;
         
         if (communicateBlocking) {
         
            const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
            if  (result!=MPI_SUCCESS) {
               std::ostringstream msg;
               msg << "was not able to send message peanoclaw::records::Vertex "
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
            msg << "was not able to send message peanoclaw::records::Vertex "
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
               msg << "testing for finished send task for peanoclaw::records::Vertex "
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
               "peanoclaw::records::Vertex",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::Vertex",
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
   
   
   
   void peanoclaw::records::Vertex::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
      if (communicateBlocking) {
      
         MPI_Status  status;
         const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
         _senderDestinationRank = status.MPI_SOURCE;
         if ( result != MPI_SUCCESS ) {
            std::ostringstream msg;
            msg << "failed to start to receive peanoclaw::records::Vertex from node "
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
            msg << "failed to start to receive peanoclaw::records::Vertex from node "
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
               msg << "testing for finished receive task for peanoclaw::records::Vertex failed: "
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
               "peanoclaw::records::Vertex",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::records::Vertex",
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
   
   
   
   bool peanoclaw::records::Vertex::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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
   
   int peanoclaw::records::Vertex::getSenderRank() const {
      assertion( _senderDestinationRank!=-1 );
      return _senderDestinationRank;
      
   }
#endif


peanoclaw::records::VertexPacked::PersistentRecords::PersistentRecords() {
   assertion((TWO_POWER_D+9 < (8 * sizeof(int))));
   
}


peanoclaw::records::VertexPacked::PersistentRecords::PersistentRecords(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank):
_indicesOfAdjacentCellDescriptions(indicesOfAdjacentCellDescriptions),
_adjacentRanksInFormerIteration(adjacentRanksInFormerIteration),
_ageInGridIterations(ageInGridIterations),
_adjacentCellsHeight(adjacentCellsHeight),
_x(x),
_level(level),
_adjacentRanks(adjacentRanks) {
   setAdjacentSubcellsEraseVeto(adjacentSubcellsEraseVeto);
   setAdjacentRanksChanged(adjacentRanksChanged);
   setShouldRefine(shouldRefine);
   setIsHangingNode(isHangingNode);
   setRefinementControl(refinementControl);
   setInsideOutsideDomain(insideOutsideDomain);
   setAdjacentSubtreeForksIntoOtherRank(adjacentSubtreeForksIntoOtherRank);
   assertion((TWO_POWER_D+9 < (8 * sizeof(int))));
   
}

peanoclaw::records::VertexPacked::VertexPacked() {
   assertion((TWO_POWER_D+9 < (8 * sizeof(int))));
   
}


peanoclaw::records::VertexPacked::VertexPacked(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._indicesOfAdjacentCellDescriptions, persistentRecords.getAdjacentSubcellsEraseVeto(), persistentRecords._adjacentRanksInFormerIteration, persistentRecords.getAdjacentRanksChanged(), persistentRecords.getShouldRefine(), persistentRecords._ageInGridIterations, persistentRecords.getIsHangingNode(), persistentRecords.getRefinementControl(), persistentRecords._adjacentCellsHeight, persistentRecords.getInsideOutsideDomain(), persistentRecords._x, persistentRecords._level, persistentRecords._adjacentRanks, persistentRecords.getAdjacentSubtreeForksIntoOtherRank()) {
   assertion((TWO_POWER_D+9 < (8 * sizeof(int))));
   
}


peanoclaw::records::VertexPacked::VertexPacked(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank):
_persistentRecords(indicesOfAdjacentCellDescriptions, adjacentSubcellsEraseVeto, adjacentRanksInFormerIteration, adjacentRanksChanged, shouldRefine, ageInGridIterations, isHangingNode, refinementControl, adjacentCellsHeight, insideOutsideDomain, x, level, adjacentRanks, adjacentSubtreeForksIntoOtherRank) {
   assertion((TWO_POWER_D+9 < (8 * sizeof(int))));
   
}


peanoclaw::records::VertexPacked::VertexPacked(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank):
_persistentRecords(indicesOfAdjacentCellDescriptions, adjacentSubcellsEraseVeto, adjacentRanksInFormerIteration, adjacentRanksChanged, shouldRefine, ageInGridIterations, isHangingNode, refinementControl, adjacentCellsHeight, insideOutsideDomain, x, level, adjacentRanks, adjacentSubtreeForksIntoOtherRank),_adjacentCellsHeightOfPreviousIteration(adjacentCellsHeightOfPreviousIteration),
_numberOfAdjacentRefinedCells(numberOfAdjacentRefinedCells) {
   assertion((TWO_POWER_D+9 < (8 * sizeof(int))));
   
}

peanoclaw::records::VertexPacked::~VertexPacked() { }

std::string peanoclaw::records::VertexPacked::toString(const InsideOutsideDomain& param) {
   return peanoclaw::records::Vertex::toString(param);
}

std::string peanoclaw::records::VertexPacked::getInsideOutsideDomainMapping() {
   return peanoclaw::records::Vertex::getInsideOutsideDomainMapping();
}

std::string peanoclaw::records::VertexPacked::toString(const RefinementControl& param) {
   return peanoclaw::records::Vertex::toString(param);
}

std::string peanoclaw::records::VertexPacked::getRefinementControlMapping() {
   return peanoclaw::records::Vertex::getRefinementControlMapping();
}



std::string peanoclaw::records::VertexPacked::toString() const {
   std::ostringstream stringstr;
   toString(stringstr);
   return stringstr.str();
}

void peanoclaw::records::VertexPacked::toString (std::ostream& out) const {
   out << "("; 
   out << "indicesOfAdjacentCellDescriptions:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getIndicesOfAdjacentCellDescriptions(i) << ",";
   }
   out << getIndicesOfAdjacentCellDescriptions(TWO_POWER_D-1) << "]";
   out << ",";
   out << "adjacentSubcellsEraseVeto:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentSubcellsEraseVeto(i) << ",";
   }
   out << getAdjacentSubcellsEraseVeto(TWO_POWER_D-1) << "]";
   out << ",";
   out << "adjacentRanksInFormerIteration:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentRanksInFormerIteration(i) << ",";
   }
   out << getAdjacentRanksInFormerIteration(TWO_POWER_D-1) << "]";
   out << ",";
   out << "adjacentRanksChanged:" << getAdjacentRanksChanged();
   out << ",";
   out << "shouldRefine:" << getShouldRefine();
   out << ",";
   out << "ageInGridIterations:" << getAgeInGridIterations();
   out << ",";
   out << "isHangingNode:" << getIsHangingNode();
   out << ",";
   out << "refinementControl:" << toString(getRefinementControl());
   out << ",";
   out << "adjacentCellsHeight:" << getAdjacentCellsHeight();
   out << ",";
   out << "adjacentCellsHeightOfPreviousIteration:" << getAdjacentCellsHeightOfPreviousIteration();
   out << ",";
   out << "numberOfAdjacentRefinedCells:" << getNumberOfAdjacentRefinedCells();
   out << ",";
   out << "insideOutsideDomain:" << toString(getInsideOutsideDomain());
   out << ",";
   out << "x:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getX(i) << ",";
   }
   out << getX(DIMENSIONS-1) << "]";
   out << ",";
   out << "level:" << getLevel();
   out << ",";
   out << "adjacentRanks:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentRanks(i) << ",";
   }
   out << getAdjacentRanks(TWO_POWER_D-1) << "]";
   out << ",";
   out << "adjacentSubtreeForksIntoOtherRank:" << getAdjacentSubtreeForksIntoOtherRank();
   out <<  ")";
}


peanoclaw::records::VertexPacked::PersistentRecords peanoclaw::records::VertexPacked::getPersistentRecords() const {
   return _persistentRecords;
}

peanoclaw::records::Vertex peanoclaw::records::VertexPacked::convert() const{
   return Vertex(
      getIndicesOfAdjacentCellDescriptions(),
      getAdjacentSubcellsEraseVeto(),
      getAdjacentRanksInFormerIteration(),
      getAdjacentRanksChanged(),
      getShouldRefine(),
      getAgeInGridIterations(),
      getIsHangingNode(),
      getRefinementControl(),
      getAdjacentCellsHeight(),
      getAdjacentCellsHeightOfPreviousIteration(),
      getNumberOfAdjacentRefinedCells(),
      getInsideOutsideDomain(),
      getX(),
      getLevel(),
      getAdjacentRanks(),
      getAdjacentSubtreeForksIntoOtherRank()
   );
}

#ifdef Parallel
   tarch::logging::Log peanoclaw::records::VertexPacked::_log( "peanoclaw::records::VertexPacked" );
   
   MPI_Datatype peanoclaw::records::VertexPacked::Datatype = 0;
   MPI_Datatype peanoclaw::records::VertexPacked::FullDatatype = 0;
   
   
   void peanoclaw::records::VertexPacked::initDatatype() {
      {
         VertexPacked dummyVertexPacked[2];
         
         const int Attributes = 8;
         MPI_Datatype subtypes[Attributes] = {
            MPI_INT,		 //indicesOfAdjacentCellDescriptions
            MPI_INT,		 //ageInGridIterations
            MPI_DOUBLE,		 //x
            MPI_INT,		 //level
            MPI_INT,		 //adjacentRanks
            MPI_INT,		 //_packedRecords0
            MPI_INT,		 //numberOfAdjacentRefinedCells
            MPI_UB		 // end/displacement flag
         };
         
         int blocklen[Attributes] = {
            TWO_POWER_D,		 //indicesOfAdjacentCellDescriptions
            1,		 //ageInGridIterations
            DIMENSIONS,		 //x
            1,		 //level
            TWO_POWER_D,		 //adjacentRanks
            1,		 //_packedRecords0
            1,		 //numberOfAdjacentRefinedCells
            1		 // end/displacement flag
         };
         
         MPI_Aint     disp[Attributes];
         
         MPI_Aint base;
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]))), &base);
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._indicesOfAdjacentCellDescriptions[0]))), 		&disp[0] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._ageInGridIterations))), 		&disp[1] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._x[0]))), 		&disp[2] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._level))), 		&disp[3] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._adjacentRanks[0]))), 		&disp[4] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._packedRecords0))), 		&disp[5] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._numberOfAdjacentRefinedCells))), 		&disp[6] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyVertexPacked[1]._persistentRecords._indicesOfAdjacentCellDescriptions[0])), 		&disp[7] );
         
         for (int i=1; i<Attributes; i++) {
            assertion1( disp[i] > disp[i-1], i );
         }
         for (int i=0; i<Attributes; i++) {
            disp[i] -= base;
         }
         MPI_Type_struct( Attributes, blocklen, disp, subtypes, &VertexPacked::Datatype );
         MPI_Type_commit( &VertexPacked::Datatype );
         
      }
      {
         VertexPacked dummyVertexPacked[2];
         
         const int Attributes = 11;
         MPI_Datatype subtypes[Attributes] = {
            MPI_INT,		 //indicesOfAdjacentCellDescriptions
            MPI_INT,		 //adjacentRanksInFormerIteration
            MPI_INT,		 //ageInGridIterations
            MPI_INT,		 //adjacentCellsHeight
            MPI_DOUBLE,		 //x
            MPI_INT,		 //level
            MPI_INT,		 //adjacentRanks
            MPI_INT,		 //_packedRecords0
            MPI_INT,		 //adjacentCellsHeightOfPreviousIteration
            MPI_INT,		 //numberOfAdjacentRefinedCells
            MPI_UB		 // end/displacement flag
         };
         
         int blocklen[Attributes] = {
            TWO_POWER_D,		 //indicesOfAdjacentCellDescriptions
            TWO_POWER_D,		 //adjacentRanksInFormerIteration
            1,		 //ageInGridIterations
            1,		 //adjacentCellsHeight
            DIMENSIONS,		 //x
            1,		 //level
            TWO_POWER_D,		 //adjacentRanks
            1,		 //_packedRecords0
            1,		 //adjacentCellsHeightOfPreviousIteration
            1,		 //numberOfAdjacentRefinedCells
            1		 // end/displacement flag
         };
         
         MPI_Aint     disp[Attributes];
         
         MPI_Aint base;
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]))), &base);
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._indicesOfAdjacentCellDescriptions[0]))), 		&disp[0] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._adjacentRanksInFormerIteration[0]))), 		&disp[1] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._ageInGridIterations))), 		&disp[2] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._adjacentCellsHeight))), 		&disp[3] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._x[0]))), 		&disp[4] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._level))), 		&disp[5] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._adjacentRanks[0]))), 		&disp[6] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._packedRecords0))), 		&disp[7] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._adjacentCellsHeightOfPreviousIteration))), 		&disp[8] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._numberOfAdjacentRefinedCells))), 		&disp[9] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyVertexPacked[1]._persistentRecords._indicesOfAdjacentCellDescriptions[0])), 		&disp[10] );
         
         for (int i=1; i<Attributes; i++) {
            assertion1( disp[i] > disp[i-1], i );
         }
         for (int i=0; i<Attributes; i++) {
            disp[i] -= base;
         }
         MPI_Type_struct( Attributes, blocklen, disp, subtypes, &VertexPacked::FullDatatype );
         MPI_Type_commit( &VertexPacked::FullDatatype );
         
      }
      
   }
   
   
   void peanoclaw::records::VertexPacked::shutdownDatatype() {
      MPI_Type_free( &VertexPacked::Datatype );
      MPI_Type_free( &VertexPacked::FullDatatype );
      
   }
   
   void peanoclaw::records::VertexPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
      _senderDestinationRank = destination;
      
      if (communicateBlocking) {
      
         const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
         if  (result!=MPI_SUCCESS) {
            std::ostringstream msg;
            msg << "was not able to send message peanoclaw::records::VertexPacked "
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
         msg << "was not able to send message peanoclaw::records::VertexPacked "
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
            msg << "testing for finished send task for peanoclaw::records::VertexPacked "
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
            "peanoclaw::records::VertexPacked",
            "send(int)", destination,tag,1
            );
            triggeredTimeoutWarning = true;
         }
         if (
            tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
            (clock()>timeOutShutdown)
         ) {
            tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
            "peanoclaw::records::VertexPacked",
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



void peanoclaw::records::VertexPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
   if (communicateBlocking) {
   
      MPI_Status  status;
      const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
      _senderDestinationRank = status.MPI_SOURCE;
      if ( result != MPI_SUCCESS ) {
         std::ostringstream msg;
         msg << "failed to start to receive peanoclaw::records::VertexPacked from node "
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
         msg << "failed to start to receive peanoclaw::records::VertexPacked from node "
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
            msg << "testing for finished receive task for peanoclaw::records::VertexPacked failed: "
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
            "peanoclaw::records::VertexPacked",
            "receive(int)", source,tag,1
            );
            triggeredTimeoutWarning = true;
         }
         if (
            tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
            (clock()>timeOutShutdown)
         ) {
            tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
            "peanoclaw::records::VertexPacked",
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



bool peanoclaw::records::VertexPacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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

int peanoclaw::records::VertexPacked::getSenderRank() const {
   assertion( _senderDestinationRank!=-1 );
   return _senderDestinationRank;
   
}
#endif



#elif !defined(Parallel) && !defined(Asserts)
peanoclaw::records::Vertex::PersistentRecords::PersistentRecords() {

}


peanoclaw::records::Vertex::PersistentRecords::PersistentRecords(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain):
_indicesOfAdjacentCellDescriptions(indicesOfAdjacentCellDescriptions),
_adjacentSubcellsEraseVeto(adjacentSubcellsEraseVeto),
_adjacentRanksInFormerIteration(adjacentRanksInFormerIteration),
_adjacentRanksChanged(adjacentRanksChanged),
_shouldRefine(shouldRefine),
_ageInGridIterations(ageInGridIterations),
_isHangingNode(isHangingNode),
_refinementControl(refinementControl),
_adjacentCellsHeight(adjacentCellsHeight),
_insideOutsideDomain(insideOutsideDomain) {

}

peanoclaw::records::Vertex::Vertex() {

}


peanoclaw::records::Vertex::Vertex(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._indicesOfAdjacentCellDescriptions, persistentRecords._adjacentSubcellsEraseVeto, persistentRecords._adjacentRanksInFormerIteration, persistentRecords._adjacentRanksChanged, persistentRecords._shouldRefine, persistentRecords._ageInGridIterations, persistentRecords._isHangingNode, persistentRecords._refinementControl, persistentRecords._adjacentCellsHeight, persistentRecords._insideOutsideDomain) {

}


peanoclaw::records::Vertex::Vertex(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain):
_persistentRecords(indicesOfAdjacentCellDescriptions, adjacentSubcellsEraseVeto, adjacentRanksInFormerIteration, adjacentRanksChanged, shouldRefine, ageInGridIterations, isHangingNode, refinementControl, adjacentCellsHeight, insideOutsideDomain) {

}


peanoclaw::records::Vertex::Vertex(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain):
_persistentRecords(indicesOfAdjacentCellDescriptions, adjacentSubcellsEraseVeto, adjacentRanksInFormerIteration, adjacentRanksChanged, shouldRefine, ageInGridIterations, isHangingNode, refinementControl, adjacentCellsHeight, insideOutsideDomain),_adjacentCellsHeightOfPreviousIteration(adjacentCellsHeightOfPreviousIteration),
_numberOfAdjacentRefinedCells(numberOfAdjacentRefinedCells) {

}

peanoclaw::records::Vertex::~Vertex() { }

std::string peanoclaw::records::Vertex::toString(const InsideOutsideDomain& param) {
switch (param) {
   case Inside: return "Inside";
   case Boundary: return "Boundary";
   case Outside: return "Outside";
}
return "undefined";
}

std::string peanoclaw::records::Vertex::getInsideOutsideDomainMapping() {
return "InsideOutsideDomain(Inside=0,Boundary=1,Outside=2)";
}
std::string peanoclaw::records::Vertex::toString(const RefinementControl& param) {
switch (param) {
   case Unrefined: return "Unrefined";
   case Refined: return "Refined";
   case RefinementTriggered: return "RefinementTriggered";
   case Refining: return "Refining";
   case EraseTriggered: return "EraseTriggered";
   case Erasing: return "Erasing";
   case RefineDueToJoinThoughWorkerIsAlreadyErasing: return "RefineDueToJoinThoughWorkerIsAlreadyErasing";
}
return "undefined";
}

std::string peanoclaw::records::Vertex::getRefinementControlMapping() {
return "RefinementControl(Unrefined=0,Refined=1,RefinementTriggered=2,Refining=3,EraseTriggered=4,Erasing=5,RefineDueToJoinThoughWorkerIsAlreadyErasing=6)";
}


std::string peanoclaw::records::Vertex::toString() const {
std::ostringstream stringstr;
toString(stringstr);
return stringstr.str();
}

void peanoclaw::records::Vertex::toString (std::ostream& out) const {
out << "("; 
out << "indicesOfAdjacentCellDescriptions:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getIndicesOfAdjacentCellDescriptions(i) << ",";
   }
   out << getIndicesOfAdjacentCellDescriptions(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentSubcellsEraseVeto:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentSubcellsEraseVeto(i) << ",";
   }
   out << getAdjacentSubcellsEraseVeto(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentRanksInFormerIteration:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentRanksInFormerIteration(i) << ",";
   }
   out << getAdjacentRanksInFormerIteration(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentRanksChanged:" << getAdjacentRanksChanged();
out << ",";
out << "shouldRefine:" << getShouldRefine();
out << ",";
out << "ageInGridIterations:" << getAgeInGridIterations();
out << ",";
out << "isHangingNode:" << getIsHangingNode();
out << ",";
out << "refinementControl:" << toString(getRefinementControl());
out << ",";
out << "adjacentCellsHeight:" << getAdjacentCellsHeight();
out << ",";
out << "adjacentCellsHeightOfPreviousIteration:" << getAdjacentCellsHeightOfPreviousIteration();
out << ",";
out << "numberOfAdjacentRefinedCells:" << getNumberOfAdjacentRefinedCells();
out << ",";
out << "insideOutsideDomain:" << toString(getInsideOutsideDomain());
out <<  ")";
}


peanoclaw::records::Vertex::PersistentRecords peanoclaw::records::Vertex::getPersistentRecords() const {
return _persistentRecords;
}

peanoclaw::records::VertexPacked peanoclaw::records::Vertex::convert() const{
return VertexPacked(
   getIndicesOfAdjacentCellDescriptions(),
   getAdjacentSubcellsEraseVeto(),
   getAdjacentRanksInFormerIteration(),
   getAdjacentRanksChanged(),
   getShouldRefine(),
   getAgeInGridIterations(),
   getIsHangingNode(),
   getRefinementControl(),
   getAdjacentCellsHeight(),
   getAdjacentCellsHeightOfPreviousIteration(),
   getNumberOfAdjacentRefinedCells(),
   getInsideOutsideDomain()
);
}

#ifdef Parallel
tarch::logging::Log peanoclaw::records::Vertex::_log( "peanoclaw::records::Vertex" );

MPI_Datatype peanoclaw::records::Vertex::Datatype = 0;
MPI_Datatype peanoclaw::records::Vertex::FullDatatype = 0;


void peanoclaw::records::Vertex::initDatatype() {
   {
      Vertex dummyVertex[2];
      
      const int Attributes = 9;
      MPI_Datatype subtypes[Attributes] = {
         MPI_INT,		 //indicesOfAdjacentCellDescriptions
         MPI_INT,		 //adjacentSubcellsEraseVeto
         MPI_CHAR,		 //adjacentRanksChanged
         MPI_CHAR,		 //shouldRefine
         MPI_INT,		 //ageInGridIterations
         MPI_CHAR,		 //isHangingNode
         MPI_INT,		 //refinementControl
         MPI_INT,		 //numberOfAdjacentRefinedCells
         MPI_UB		 // end/displacement flag
      };
      
      int blocklen[Attributes] = {
         TWO_POWER_D,		 //indicesOfAdjacentCellDescriptions
         TWO_POWER_D,		 //adjacentSubcellsEraseVeto
         1,		 //adjacentRanksChanged
         1,		 //shouldRefine
         1,		 //ageInGridIterations
         1,		 //isHangingNode
         1,		 //refinementControl
         1,		 //numberOfAdjacentRefinedCells
         1		 // end/displacement flag
      };
      
      MPI_Aint     disp[Attributes];
      
      MPI_Aint base;
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]))), &base);
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._indicesOfAdjacentCellDescriptions[0]))), 		&disp[0] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentSubcellsEraseVeto))), 		&disp[1] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentRanksChanged))), 		&disp[2] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._shouldRefine))), 		&disp[3] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._ageInGridIterations))), 		&disp[4] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._isHangingNode))), 		&disp[5] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._refinementControl))), 		&disp[6] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._numberOfAdjacentRefinedCells))), 		&disp[7] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyVertex[1]._persistentRecords._indicesOfAdjacentCellDescriptions[0])), 		&disp[8] );
      
      for (int i=1; i<Attributes; i++) {
         assertion1( disp[i] > disp[i-1], i );
      }
      for (int i=0; i<Attributes; i++) {
         disp[i] -= base;
      }
      MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Vertex::Datatype );
      MPI_Type_commit( &Vertex::Datatype );
      
   }
   {
      Vertex dummyVertex[2];
      
      const int Attributes = 13;
      MPI_Datatype subtypes[Attributes] = {
         MPI_INT,		 //indicesOfAdjacentCellDescriptions
         MPI_INT,		 //adjacentSubcellsEraseVeto
         MPI_INT,		 //adjacentRanksInFormerIteration
         MPI_CHAR,		 //adjacentRanksChanged
         MPI_CHAR,		 //shouldRefine
         MPI_INT,		 //ageInGridIterations
         MPI_CHAR,		 //isHangingNode
         MPI_INT,		 //refinementControl
         MPI_INT,		 //adjacentCellsHeight
         MPI_INT,		 //insideOutsideDomain
         MPI_INT,		 //adjacentCellsHeightOfPreviousIteration
         MPI_INT,		 //numberOfAdjacentRefinedCells
         MPI_UB		 // end/displacement flag
      };
      
      int blocklen[Attributes] = {
         TWO_POWER_D,		 //indicesOfAdjacentCellDescriptions
         TWO_POWER_D,		 //adjacentSubcellsEraseVeto
         TWO_POWER_D,		 //adjacentRanksInFormerIteration
         1,		 //adjacentRanksChanged
         1,		 //shouldRefine
         1,		 //ageInGridIterations
         1,		 //isHangingNode
         1,		 //refinementControl
         1,		 //adjacentCellsHeight
         1,		 //insideOutsideDomain
         1,		 //adjacentCellsHeightOfPreviousIteration
         1,		 //numberOfAdjacentRefinedCells
         1		 // end/displacement flag
      };
      
      MPI_Aint     disp[Attributes];
      
      MPI_Aint base;
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]))), &base);
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._indicesOfAdjacentCellDescriptions[0]))), 		&disp[0] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentSubcellsEraseVeto))), 		&disp[1] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentRanksInFormerIteration[0]))), 		&disp[2] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentRanksChanged))), 		&disp[3] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._shouldRefine))), 		&disp[4] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._ageInGridIterations))), 		&disp[5] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._isHangingNode))), 		&disp[6] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._refinementControl))), 		&disp[7] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentCellsHeight))), 		&disp[8] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._insideOutsideDomain))), 		&disp[9] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._adjacentCellsHeightOfPreviousIteration))), 		&disp[10] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._numberOfAdjacentRefinedCells))), 		&disp[11] );
      MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyVertex[1]._persistentRecords._indicesOfAdjacentCellDescriptions[0])), 		&disp[12] );
      
      for (int i=1; i<Attributes; i++) {
         assertion1( disp[i] > disp[i-1], i );
      }
      for (int i=0; i<Attributes; i++) {
         disp[i] -= base;
      }
      MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Vertex::FullDatatype );
      MPI_Type_commit( &Vertex::FullDatatype );
      
   }
   
}


void peanoclaw::records::Vertex::shutdownDatatype() {
   MPI_Type_free( &Vertex::Datatype );
   MPI_Type_free( &Vertex::FullDatatype );
   
}

void peanoclaw::records::Vertex::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
   _senderDestinationRank = destination;
   
   if (communicateBlocking) {
   
      const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
      if  (result!=MPI_SUCCESS) {
         std::ostringstream msg;
         msg << "was not able to send message peanoclaw::records::Vertex "
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
      msg << "was not able to send message peanoclaw::records::Vertex "
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
         msg << "testing for finished send task for peanoclaw::records::Vertex "
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
         "peanoclaw::records::Vertex",
         "send(int)", destination,tag,1
         );
         triggeredTimeoutWarning = true;
      }
      if (
         tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
         (clock()>timeOutShutdown)
      ) {
         tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
         "peanoclaw::records::Vertex",
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



void peanoclaw::records::Vertex::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
if (communicateBlocking) {

   MPI_Status  status;
   const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
   _senderDestinationRank = status.MPI_SOURCE;
   if ( result != MPI_SUCCESS ) {
      std::ostringstream msg;
      msg << "failed to start to receive peanoclaw::records::Vertex from node "
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
      msg << "failed to start to receive peanoclaw::records::Vertex from node "
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
         msg << "testing for finished receive task for peanoclaw::records::Vertex failed: "
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
         "peanoclaw::records::Vertex",
         "receive(int)", source,tag,1
         );
         triggeredTimeoutWarning = true;
      }
      if (
         tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
         (clock()>timeOutShutdown)
      ) {
         tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
         "peanoclaw::records::Vertex",
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



bool peanoclaw::records::Vertex::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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

int peanoclaw::records::Vertex::getSenderRank() const {
assertion( _senderDestinationRank!=-1 );
return _senderDestinationRank;

}
#endif


peanoclaw::records::VertexPacked::PersistentRecords::PersistentRecords() {
assertion((TWO_POWER_D+8 < (8 * sizeof(int))));

}


peanoclaw::records::VertexPacked::PersistentRecords::PersistentRecords(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain):
_indicesOfAdjacentCellDescriptions(indicesOfAdjacentCellDescriptions),
_adjacentRanksInFormerIteration(adjacentRanksInFormerIteration),
_ageInGridIterations(ageInGridIterations),
_adjacentCellsHeight(adjacentCellsHeight) {
setAdjacentSubcellsEraseVeto(adjacentSubcellsEraseVeto);
setAdjacentRanksChanged(adjacentRanksChanged);
setShouldRefine(shouldRefine);
setIsHangingNode(isHangingNode);
setRefinementControl(refinementControl);
setInsideOutsideDomain(insideOutsideDomain);
assertion((TWO_POWER_D+8 < (8 * sizeof(int))));

}

peanoclaw::records::VertexPacked::VertexPacked() {
assertion((TWO_POWER_D+8 < (8 * sizeof(int))));

}


peanoclaw::records::VertexPacked::VertexPacked(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._indicesOfAdjacentCellDescriptions, persistentRecords.getAdjacentSubcellsEraseVeto(), persistentRecords._adjacentRanksInFormerIteration, persistentRecords.getAdjacentRanksChanged(), persistentRecords.getShouldRefine(), persistentRecords._ageInGridIterations, persistentRecords.getIsHangingNode(), persistentRecords.getRefinementControl(), persistentRecords._adjacentCellsHeight, persistentRecords.getInsideOutsideDomain()) {
assertion((TWO_POWER_D+8 < (8 * sizeof(int))));

}


peanoclaw::records::VertexPacked::VertexPacked(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain):
_persistentRecords(indicesOfAdjacentCellDescriptions, adjacentSubcellsEraseVeto, adjacentRanksInFormerIteration, adjacentRanksChanged, shouldRefine, ageInGridIterations, isHangingNode, refinementControl, adjacentCellsHeight, insideOutsideDomain) {
assertion((TWO_POWER_D+8 < (8 * sizeof(int))));

}


peanoclaw::records::VertexPacked::VertexPacked(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain):
_persistentRecords(indicesOfAdjacentCellDescriptions, adjacentSubcellsEraseVeto, adjacentRanksInFormerIteration, adjacentRanksChanged, shouldRefine, ageInGridIterations, isHangingNode, refinementControl, adjacentCellsHeight, insideOutsideDomain),_adjacentCellsHeightOfPreviousIteration(adjacentCellsHeightOfPreviousIteration),
_numberOfAdjacentRefinedCells(numberOfAdjacentRefinedCells) {
assertion((TWO_POWER_D+8 < (8 * sizeof(int))));

}

peanoclaw::records::VertexPacked::~VertexPacked() { }

std::string peanoclaw::records::VertexPacked::toString(const InsideOutsideDomain& param) {
return peanoclaw::records::Vertex::toString(param);
}

std::string peanoclaw::records::VertexPacked::getInsideOutsideDomainMapping() {
return peanoclaw::records::Vertex::getInsideOutsideDomainMapping();
}

std::string peanoclaw::records::VertexPacked::toString(const RefinementControl& param) {
return peanoclaw::records::Vertex::toString(param);
}

std::string peanoclaw::records::VertexPacked::getRefinementControlMapping() {
return peanoclaw::records::Vertex::getRefinementControlMapping();
}



std::string peanoclaw::records::VertexPacked::toString() const {
std::ostringstream stringstr;
toString(stringstr);
return stringstr.str();
}

void peanoclaw::records::VertexPacked::toString (std::ostream& out) const {
out << "("; 
out << "indicesOfAdjacentCellDescriptions:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getIndicesOfAdjacentCellDescriptions(i) << ",";
   }
   out << getIndicesOfAdjacentCellDescriptions(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentSubcellsEraseVeto:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentSubcellsEraseVeto(i) << ",";
   }
   out << getAdjacentSubcellsEraseVeto(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentRanksInFormerIteration:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentRanksInFormerIteration(i) << ",";
   }
   out << getAdjacentRanksInFormerIteration(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentRanksChanged:" << getAdjacentRanksChanged();
out << ",";
out << "shouldRefine:" << getShouldRefine();
out << ",";
out << "ageInGridIterations:" << getAgeInGridIterations();
out << ",";
out << "isHangingNode:" << getIsHangingNode();
out << ",";
out << "refinementControl:" << toString(getRefinementControl());
out << ",";
out << "adjacentCellsHeight:" << getAdjacentCellsHeight();
out << ",";
out << "adjacentCellsHeightOfPreviousIteration:" << getAdjacentCellsHeightOfPreviousIteration();
out << ",";
out << "numberOfAdjacentRefinedCells:" << getNumberOfAdjacentRefinedCells();
out << ",";
out << "insideOutsideDomain:" << toString(getInsideOutsideDomain());
out <<  ")";
}


peanoclaw::records::VertexPacked::PersistentRecords peanoclaw::records::VertexPacked::getPersistentRecords() const {
return _persistentRecords;
}

peanoclaw::records::Vertex peanoclaw::records::VertexPacked::convert() const{
return Vertex(
getIndicesOfAdjacentCellDescriptions(),
getAdjacentSubcellsEraseVeto(),
getAdjacentRanksInFormerIteration(),
getAdjacentRanksChanged(),
getShouldRefine(),
getAgeInGridIterations(),
getIsHangingNode(),
getRefinementControl(),
getAdjacentCellsHeight(),
getAdjacentCellsHeightOfPreviousIteration(),
getNumberOfAdjacentRefinedCells(),
getInsideOutsideDomain()
);
}

#ifdef Parallel
tarch::logging::Log peanoclaw::records::VertexPacked::_log( "peanoclaw::records::VertexPacked" );

MPI_Datatype peanoclaw::records::VertexPacked::Datatype = 0;
MPI_Datatype peanoclaw::records::VertexPacked::FullDatatype = 0;


void peanoclaw::records::VertexPacked::initDatatype() {
{
   VertexPacked dummyVertexPacked[2];
   
   const int Attributes = 5;
   MPI_Datatype subtypes[Attributes] = {
      MPI_INT,		 //indicesOfAdjacentCellDescriptions
      MPI_INT,		 //ageInGridIterations
      MPI_INT,		 //_packedRecords0
      MPI_INT,		 //numberOfAdjacentRefinedCells
      MPI_UB		 // end/displacement flag
   };
   
   int blocklen[Attributes] = {
      TWO_POWER_D,		 //indicesOfAdjacentCellDescriptions
      1,		 //ageInGridIterations
      1,		 //_packedRecords0
      1,		 //numberOfAdjacentRefinedCells
      1		 // end/displacement flag
   };
   
   MPI_Aint     disp[Attributes];
   
   MPI_Aint base;
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]))), &base);
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._indicesOfAdjacentCellDescriptions[0]))), 		&disp[0] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._ageInGridIterations))), 		&disp[1] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._packedRecords0))), 		&disp[2] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._numberOfAdjacentRefinedCells))), 		&disp[3] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyVertexPacked[1]._persistentRecords._indicesOfAdjacentCellDescriptions[0])), 		&disp[4] );
   
   for (int i=1; i<Attributes; i++) {
      assertion1( disp[i] > disp[i-1], i );
   }
   for (int i=0; i<Attributes; i++) {
      disp[i] -= base;
   }
   MPI_Type_struct( Attributes, blocklen, disp, subtypes, &VertexPacked::Datatype );
   MPI_Type_commit( &VertexPacked::Datatype );
   
}
{
   VertexPacked dummyVertexPacked[2];
   
   const int Attributes = 8;
   MPI_Datatype subtypes[Attributes] = {
      MPI_INT,		 //indicesOfAdjacentCellDescriptions
      MPI_INT,		 //adjacentRanksInFormerIteration
      MPI_INT,		 //ageInGridIterations
      MPI_INT,		 //adjacentCellsHeight
      MPI_INT,		 //_packedRecords0
      MPI_INT,		 //adjacentCellsHeightOfPreviousIteration
      MPI_INT,		 //numberOfAdjacentRefinedCells
      MPI_UB		 // end/displacement flag
   };
   
   int blocklen[Attributes] = {
      TWO_POWER_D,		 //indicesOfAdjacentCellDescriptions
      TWO_POWER_D,		 //adjacentRanksInFormerIteration
      1,		 //ageInGridIterations
      1,		 //adjacentCellsHeight
      1,		 //_packedRecords0
      1,		 //adjacentCellsHeightOfPreviousIteration
      1,		 //numberOfAdjacentRefinedCells
      1		 // end/displacement flag
   };
   
   MPI_Aint     disp[Attributes];
   
   MPI_Aint base;
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]))), &base);
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._indicesOfAdjacentCellDescriptions[0]))), 		&disp[0] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._adjacentRanksInFormerIteration[0]))), 		&disp[1] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._ageInGridIterations))), 		&disp[2] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._adjacentCellsHeight))), 		&disp[3] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._packedRecords0))), 		&disp[4] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._adjacentCellsHeightOfPreviousIteration))), 		&disp[5] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._numberOfAdjacentRefinedCells))), 		&disp[6] );
   MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyVertexPacked[1]._persistentRecords._indicesOfAdjacentCellDescriptions[0])), 		&disp[7] );
   
   for (int i=1; i<Attributes; i++) {
      assertion1( disp[i] > disp[i-1], i );
   }
   for (int i=0; i<Attributes; i++) {
      disp[i] -= base;
   }
   MPI_Type_struct( Attributes, blocklen, disp, subtypes, &VertexPacked::FullDatatype );
   MPI_Type_commit( &VertexPacked::FullDatatype );
   
}

}


void peanoclaw::records::VertexPacked::shutdownDatatype() {
MPI_Type_free( &VertexPacked::Datatype );
MPI_Type_free( &VertexPacked::FullDatatype );

}

void peanoclaw::records::VertexPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
_senderDestinationRank = destination;

if (communicateBlocking) {

   const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
   if  (result!=MPI_SUCCESS) {
      std::ostringstream msg;
      msg << "was not able to send message peanoclaw::records::VertexPacked "
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
   msg << "was not able to send message peanoclaw::records::VertexPacked "
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
      msg << "testing for finished send task for peanoclaw::records::VertexPacked "
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
      "peanoclaw::records::VertexPacked",
      "send(int)", destination,tag,1
      );
      triggeredTimeoutWarning = true;
   }
   if (
      tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
      (clock()>timeOutShutdown)
   ) {
      tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
      "peanoclaw::records::VertexPacked",
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



void peanoclaw::records::VertexPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
if (communicateBlocking) {

MPI_Status  status;
const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
_senderDestinationRank = status.MPI_SOURCE;
if ( result != MPI_SUCCESS ) {
   std::ostringstream msg;
   msg << "failed to start to receive peanoclaw::records::VertexPacked from node "
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
   msg << "failed to start to receive peanoclaw::records::VertexPacked from node "
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
      msg << "testing for finished receive task for peanoclaw::records::VertexPacked failed: "
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
      "peanoclaw::records::VertexPacked",
      "receive(int)", source,tag,1
      );
      triggeredTimeoutWarning = true;
   }
   if (
      tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
      (clock()>timeOutShutdown)
   ) {
      tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
      "peanoclaw::records::VertexPacked",
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



bool peanoclaw::records::VertexPacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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

int peanoclaw::records::VertexPacked::getSenderRank() const {
assertion( _senderDestinationRank!=-1 );
return _senderDestinationRank;

}
#endif




#elif !defined(Parallel) && defined(Asserts)
peanoclaw::records::Vertex::PersistentRecords::PersistentRecords() {

}


peanoclaw::records::Vertex::PersistentRecords::PersistentRecords(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level):
_indicesOfAdjacentCellDescriptions(indicesOfAdjacentCellDescriptions),
_adjacentSubcellsEraseVeto(adjacentSubcellsEraseVeto),
_adjacentRanksInFormerIteration(adjacentRanksInFormerIteration),
_adjacentRanksChanged(adjacentRanksChanged),
_shouldRefine(shouldRefine),
_ageInGridIterations(ageInGridIterations),
_isHangingNode(isHangingNode),
_refinementControl(refinementControl),
_adjacentCellsHeight(adjacentCellsHeight),
_insideOutsideDomain(insideOutsideDomain),
_x(x),
_level(level) {

}

peanoclaw::records::Vertex::Vertex() {

}


peanoclaw::records::Vertex::Vertex(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._indicesOfAdjacentCellDescriptions, persistentRecords._adjacentSubcellsEraseVeto, persistentRecords._adjacentRanksInFormerIteration, persistentRecords._adjacentRanksChanged, persistentRecords._shouldRefine, persistentRecords._ageInGridIterations, persistentRecords._isHangingNode, persistentRecords._refinementControl, persistentRecords._adjacentCellsHeight, persistentRecords._insideOutsideDomain, persistentRecords._x, persistentRecords._level) {

}


peanoclaw::records::Vertex::Vertex(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level):
_persistentRecords(indicesOfAdjacentCellDescriptions, adjacentSubcellsEraseVeto, adjacentRanksInFormerIteration, adjacentRanksChanged, shouldRefine, ageInGridIterations, isHangingNode, refinementControl, adjacentCellsHeight, insideOutsideDomain, x, level) {

}


peanoclaw::records::Vertex::Vertex(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level):
_persistentRecords(indicesOfAdjacentCellDescriptions, adjacentSubcellsEraseVeto, adjacentRanksInFormerIteration, adjacentRanksChanged, shouldRefine, ageInGridIterations, isHangingNode, refinementControl, adjacentCellsHeight, insideOutsideDomain, x, level),_adjacentCellsHeightOfPreviousIteration(adjacentCellsHeightOfPreviousIteration),
_numberOfAdjacentRefinedCells(numberOfAdjacentRefinedCells) {

}

peanoclaw::records::Vertex::~Vertex() { }

std::string peanoclaw::records::Vertex::toString(const InsideOutsideDomain& param) {
switch (param) {
case Inside: return "Inside";
case Boundary: return "Boundary";
case Outside: return "Outside";
}
return "undefined";
}

std::string peanoclaw::records::Vertex::getInsideOutsideDomainMapping() {
return "InsideOutsideDomain(Inside=0,Boundary=1,Outside=2)";
}
std::string peanoclaw::records::Vertex::toString(const RefinementControl& param) {
switch (param) {
case Unrefined: return "Unrefined";
case Refined: return "Refined";
case RefinementTriggered: return "RefinementTriggered";
case Refining: return "Refining";
case EraseTriggered: return "EraseTriggered";
case Erasing: return "Erasing";
case RefineDueToJoinThoughWorkerIsAlreadyErasing: return "RefineDueToJoinThoughWorkerIsAlreadyErasing";
}
return "undefined";
}

std::string peanoclaw::records::Vertex::getRefinementControlMapping() {
return "RefinementControl(Unrefined=0,Refined=1,RefinementTriggered=2,Refining=3,EraseTriggered=4,Erasing=5,RefineDueToJoinThoughWorkerIsAlreadyErasing=6)";
}


std::string peanoclaw::records::Vertex::toString() const {
std::ostringstream stringstr;
toString(stringstr);
return stringstr.str();
}

void peanoclaw::records::Vertex::toString (std::ostream& out) const {
out << "("; 
out << "indicesOfAdjacentCellDescriptions:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getIndicesOfAdjacentCellDescriptions(i) << ",";
   }
   out << getIndicesOfAdjacentCellDescriptions(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentSubcellsEraseVeto:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentSubcellsEraseVeto(i) << ",";
   }
   out << getAdjacentSubcellsEraseVeto(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentRanksInFormerIteration:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentRanksInFormerIteration(i) << ",";
   }
   out << getAdjacentRanksInFormerIteration(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentRanksChanged:" << getAdjacentRanksChanged();
out << ",";
out << "shouldRefine:" << getShouldRefine();
out << ",";
out << "ageInGridIterations:" << getAgeInGridIterations();
out << ",";
out << "isHangingNode:" << getIsHangingNode();
out << ",";
out << "refinementControl:" << toString(getRefinementControl());
out << ",";
out << "adjacentCellsHeight:" << getAdjacentCellsHeight();
out << ",";
out << "adjacentCellsHeightOfPreviousIteration:" << getAdjacentCellsHeightOfPreviousIteration();
out << ",";
out << "numberOfAdjacentRefinedCells:" << getNumberOfAdjacentRefinedCells();
out << ",";
out << "insideOutsideDomain:" << toString(getInsideOutsideDomain());
out << ",";
out << "x:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getX(i) << ",";
   }
   out << getX(DIMENSIONS-1) << "]";
out << ",";
out << "level:" << getLevel();
out <<  ")";
}


peanoclaw::records::Vertex::PersistentRecords peanoclaw::records::Vertex::getPersistentRecords() const {
return _persistentRecords;
}

peanoclaw::records::VertexPacked peanoclaw::records::Vertex::convert() const{
return VertexPacked(
getIndicesOfAdjacentCellDescriptions(),
getAdjacentSubcellsEraseVeto(),
getAdjacentRanksInFormerIteration(),
getAdjacentRanksChanged(),
getShouldRefine(),
getAgeInGridIterations(),
getIsHangingNode(),
getRefinementControl(),
getAdjacentCellsHeight(),
getAdjacentCellsHeightOfPreviousIteration(),
getNumberOfAdjacentRefinedCells(),
getInsideOutsideDomain(),
getX(),
getLevel()
);
}

#ifdef Parallel
tarch::logging::Log peanoclaw::records::Vertex::_log( "peanoclaw::records::Vertex" );

MPI_Datatype peanoclaw::records::Vertex::Datatype = 0;
MPI_Datatype peanoclaw::records::Vertex::FullDatatype = 0;


void peanoclaw::records::Vertex::initDatatype() {
{
Vertex dummyVertex[2];

const int Attributes = 12;
MPI_Datatype subtypes[Attributes] = {
MPI_INT,		 //indicesOfAdjacentCellDescriptions
MPI_INT,		 //adjacentSubcellsEraseVeto
MPI_CHAR,		 //adjacentRanksChanged
MPI_CHAR,		 //shouldRefine
MPI_INT,		 //ageInGridIterations
MPI_CHAR,		 //isHangingNode
MPI_INT,		 //refinementControl
MPI_INT,		 //insideOutsideDomain
MPI_DOUBLE,		 //x
MPI_INT,		 //level
MPI_INT,		 //numberOfAdjacentRefinedCells
MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
TWO_POWER_D,		 //indicesOfAdjacentCellDescriptions
TWO_POWER_D,		 //adjacentSubcellsEraseVeto
1,		 //adjacentRanksChanged
1,		 //shouldRefine
1,		 //ageInGridIterations
1,		 //isHangingNode
1,		 //refinementControl
1,		 //insideOutsideDomain
DIMENSIONS,		 //x
1,		 //level
1,		 //numberOfAdjacentRefinedCells
1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._indicesOfAdjacentCellDescriptions[0]))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentSubcellsEraseVeto))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentRanksChanged))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._shouldRefine))), 		&disp[3] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._ageInGridIterations))), 		&disp[4] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._isHangingNode))), 		&disp[5] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._refinementControl))), 		&disp[6] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._insideOutsideDomain))), 		&disp[7] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._x[0]))), 		&disp[8] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._level))), 		&disp[9] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._numberOfAdjacentRefinedCells))), 		&disp[10] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyVertex[1]._persistentRecords._indicesOfAdjacentCellDescriptions[0])), 		&disp[11] );

for (int i=1; i<Attributes; i++) {
assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Vertex::Datatype );
MPI_Type_commit( &Vertex::Datatype );

}
{
Vertex dummyVertex[2];

const int Attributes = 15;
MPI_Datatype subtypes[Attributes] = {
MPI_INT,		 //indicesOfAdjacentCellDescriptions
MPI_INT,		 //adjacentSubcellsEraseVeto
MPI_INT,		 //adjacentRanksInFormerIteration
MPI_CHAR,		 //adjacentRanksChanged
MPI_CHAR,		 //shouldRefine
MPI_INT,		 //ageInGridIterations
MPI_CHAR,		 //isHangingNode
MPI_INT,		 //refinementControl
MPI_INT,		 //adjacentCellsHeight
MPI_INT,		 //insideOutsideDomain
MPI_DOUBLE,		 //x
MPI_INT,		 //level
MPI_INT,		 //adjacentCellsHeightOfPreviousIteration
MPI_INT,		 //numberOfAdjacentRefinedCells
MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
TWO_POWER_D,		 //indicesOfAdjacentCellDescriptions
TWO_POWER_D,		 //adjacentSubcellsEraseVeto
TWO_POWER_D,		 //adjacentRanksInFormerIteration
1,		 //adjacentRanksChanged
1,		 //shouldRefine
1,		 //ageInGridIterations
1,		 //isHangingNode
1,		 //refinementControl
1,		 //adjacentCellsHeight
1,		 //insideOutsideDomain
DIMENSIONS,		 //x
1,		 //level
1,		 //adjacentCellsHeightOfPreviousIteration
1,		 //numberOfAdjacentRefinedCells
1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._indicesOfAdjacentCellDescriptions[0]))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentSubcellsEraseVeto))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentRanksInFormerIteration[0]))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentRanksChanged))), 		&disp[3] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._shouldRefine))), 		&disp[4] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._ageInGridIterations))), 		&disp[5] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._isHangingNode))), 		&disp[6] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._refinementControl))), 		&disp[7] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentCellsHeight))), 		&disp[8] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._insideOutsideDomain))), 		&disp[9] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._x[0]))), 		&disp[10] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._level))), 		&disp[11] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._adjacentCellsHeightOfPreviousIteration))), 		&disp[12] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._numberOfAdjacentRefinedCells))), 		&disp[13] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyVertex[1]._persistentRecords._indicesOfAdjacentCellDescriptions[0])), 		&disp[14] );

for (int i=1; i<Attributes; i++) {
assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Vertex::FullDatatype );
MPI_Type_commit( &Vertex::FullDatatype );

}

}


void peanoclaw::records::Vertex::shutdownDatatype() {
MPI_Type_free( &Vertex::Datatype );
MPI_Type_free( &Vertex::FullDatatype );

}

void peanoclaw::records::Vertex::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
_senderDestinationRank = destination;

if (communicateBlocking) {

const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
if  (result!=MPI_SUCCESS) {
std::ostringstream msg;
msg << "was not able to send message peanoclaw::records::Vertex "
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
msg << "was not able to send message peanoclaw::records::Vertex "
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
msg << "testing for finished send task for peanoclaw::records::Vertex "
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
"peanoclaw::records::Vertex",
"send(int)", destination,tag,1
);
triggeredTimeoutWarning = true;
}
if (
tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
(clock()>timeOutShutdown)
) {
tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
"peanoclaw::records::Vertex",
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



void peanoclaw::records::Vertex::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
if (communicateBlocking) {

MPI_Status  status;
const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
_senderDestinationRank = status.MPI_SOURCE;
if ( result != MPI_SUCCESS ) {
std::ostringstream msg;
msg << "failed to start to receive peanoclaw::records::Vertex from node "
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
msg << "failed to start to receive peanoclaw::records::Vertex from node "
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
msg << "testing for finished receive task for peanoclaw::records::Vertex failed: "
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
"peanoclaw::records::Vertex",
"receive(int)", source,tag,1
);
triggeredTimeoutWarning = true;
}
if (
tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
(clock()>timeOutShutdown)
) {
tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
"peanoclaw::records::Vertex",
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



bool peanoclaw::records::Vertex::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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

int peanoclaw::records::Vertex::getSenderRank() const {
assertion( _senderDestinationRank!=-1 );
return _senderDestinationRank;

}
#endif


peanoclaw::records::VertexPacked::PersistentRecords::PersistentRecords() {
assertion((TWO_POWER_D+8 < (8 * sizeof(int))));

}


peanoclaw::records::VertexPacked::PersistentRecords::PersistentRecords(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level):
_indicesOfAdjacentCellDescriptions(indicesOfAdjacentCellDescriptions),
_adjacentRanksInFormerIteration(adjacentRanksInFormerIteration),
_ageInGridIterations(ageInGridIterations),
_adjacentCellsHeight(adjacentCellsHeight),
_x(x),
_level(level) {
setAdjacentSubcellsEraseVeto(adjacentSubcellsEraseVeto);
setAdjacentRanksChanged(adjacentRanksChanged);
setShouldRefine(shouldRefine);
setIsHangingNode(isHangingNode);
setRefinementControl(refinementControl);
setInsideOutsideDomain(insideOutsideDomain);
assertion((TWO_POWER_D+8 < (8 * sizeof(int))));

}

peanoclaw::records::VertexPacked::VertexPacked() {
assertion((TWO_POWER_D+8 < (8 * sizeof(int))));

}


peanoclaw::records::VertexPacked::VertexPacked(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._indicesOfAdjacentCellDescriptions, persistentRecords.getAdjacentSubcellsEraseVeto(), persistentRecords._adjacentRanksInFormerIteration, persistentRecords.getAdjacentRanksChanged(), persistentRecords.getShouldRefine(), persistentRecords._ageInGridIterations, persistentRecords.getIsHangingNode(), persistentRecords.getRefinementControl(), persistentRecords._adjacentCellsHeight, persistentRecords.getInsideOutsideDomain(), persistentRecords._x, persistentRecords._level) {
assertion((TWO_POWER_D+8 < (8 * sizeof(int))));

}


peanoclaw::records::VertexPacked::VertexPacked(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level):
_persistentRecords(indicesOfAdjacentCellDescriptions, adjacentSubcellsEraseVeto, adjacentRanksInFormerIteration, adjacentRanksChanged, shouldRefine, ageInGridIterations, isHangingNode, refinementControl, adjacentCellsHeight, insideOutsideDomain, x, level) {
assertion((TWO_POWER_D+8 < (8 * sizeof(int))));

}


peanoclaw::records::VertexPacked::VertexPacked(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level):
_persistentRecords(indicesOfAdjacentCellDescriptions, adjacentSubcellsEraseVeto, adjacentRanksInFormerIteration, adjacentRanksChanged, shouldRefine, ageInGridIterations, isHangingNode, refinementControl, adjacentCellsHeight, insideOutsideDomain, x, level),_adjacentCellsHeightOfPreviousIteration(adjacentCellsHeightOfPreviousIteration),
_numberOfAdjacentRefinedCells(numberOfAdjacentRefinedCells) {
assertion((TWO_POWER_D+8 < (8 * sizeof(int))));

}

peanoclaw::records::VertexPacked::~VertexPacked() { }

std::string peanoclaw::records::VertexPacked::toString(const InsideOutsideDomain& param) {
return peanoclaw::records::Vertex::toString(param);
}

std::string peanoclaw::records::VertexPacked::getInsideOutsideDomainMapping() {
return peanoclaw::records::Vertex::getInsideOutsideDomainMapping();
}

std::string peanoclaw::records::VertexPacked::toString(const RefinementControl& param) {
return peanoclaw::records::Vertex::toString(param);
}

std::string peanoclaw::records::VertexPacked::getRefinementControlMapping() {
return peanoclaw::records::Vertex::getRefinementControlMapping();
}



std::string peanoclaw::records::VertexPacked::toString() const {
std::ostringstream stringstr;
toString(stringstr);
return stringstr.str();
}

void peanoclaw::records::VertexPacked::toString (std::ostream& out) const {
out << "("; 
out << "indicesOfAdjacentCellDescriptions:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getIndicesOfAdjacentCellDescriptions(i) << ",";
   }
   out << getIndicesOfAdjacentCellDescriptions(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentSubcellsEraseVeto:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentSubcellsEraseVeto(i) << ",";
   }
   out << getAdjacentSubcellsEraseVeto(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentRanksInFormerIteration:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentRanksInFormerIteration(i) << ",";
   }
   out << getAdjacentRanksInFormerIteration(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentRanksChanged:" << getAdjacentRanksChanged();
out << ",";
out << "shouldRefine:" << getShouldRefine();
out << ",";
out << "ageInGridIterations:" << getAgeInGridIterations();
out << ",";
out << "isHangingNode:" << getIsHangingNode();
out << ",";
out << "refinementControl:" << toString(getRefinementControl());
out << ",";
out << "adjacentCellsHeight:" << getAdjacentCellsHeight();
out << ",";
out << "adjacentCellsHeightOfPreviousIteration:" << getAdjacentCellsHeightOfPreviousIteration();
out << ",";
out << "numberOfAdjacentRefinedCells:" << getNumberOfAdjacentRefinedCells();
out << ",";
out << "insideOutsideDomain:" << toString(getInsideOutsideDomain());
out << ",";
out << "x:[";
   for (int i = 0; i < DIMENSIONS-1; i++) {
      out << getX(i) << ",";
   }
   out << getX(DIMENSIONS-1) << "]";
out << ",";
out << "level:" << getLevel();
out <<  ")";
}


peanoclaw::records::VertexPacked::PersistentRecords peanoclaw::records::VertexPacked::getPersistentRecords() const {
return _persistentRecords;
}

peanoclaw::records::Vertex peanoclaw::records::VertexPacked::convert() const{
return Vertex(
getIndicesOfAdjacentCellDescriptions(),
getAdjacentSubcellsEraseVeto(),
getAdjacentRanksInFormerIteration(),
getAdjacentRanksChanged(),
getShouldRefine(),
getAgeInGridIterations(),
getIsHangingNode(),
getRefinementControl(),
getAdjacentCellsHeight(),
getAdjacentCellsHeightOfPreviousIteration(),
getNumberOfAdjacentRefinedCells(),
getInsideOutsideDomain(),
getX(),
getLevel()
);
}

#ifdef Parallel
tarch::logging::Log peanoclaw::records::VertexPacked::_log( "peanoclaw::records::VertexPacked" );

MPI_Datatype peanoclaw::records::VertexPacked::Datatype = 0;
MPI_Datatype peanoclaw::records::VertexPacked::FullDatatype = 0;


void peanoclaw::records::VertexPacked::initDatatype() {
{
VertexPacked dummyVertexPacked[2];

const int Attributes = 7;
MPI_Datatype subtypes[Attributes] = {
MPI_INT,		 //indicesOfAdjacentCellDescriptions
MPI_INT,		 //ageInGridIterations
MPI_DOUBLE,		 //x
MPI_INT,		 //level
MPI_INT,		 //_packedRecords0
MPI_INT,		 //numberOfAdjacentRefinedCells
MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
TWO_POWER_D,		 //indicesOfAdjacentCellDescriptions
1,		 //ageInGridIterations
DIMENSIONS,		 //x
1,		 //level
1,		 //_packedRecords0
1,		 //numberOfAdjacentRefinedCells
1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._indicesOfAdjacentCellDescriptions[0]))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._ageInGridIterations))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._x[0]))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._level))), 		&disp[3] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._packedRecords0))), 		&disp[4] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._numberOfAdjacentRefinedCells))), 		&disp[5] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyVertexPacked[1]._persistentRecords._indicesOfAdjacentCellDescriptions[0])), 		&disp[6] );

for (int i=1; i<Attributes; i++) {
assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &VertexPacked::Datatype );
MPI_Type_commit( &VertexPacked::Datatype );

}
{
VertexPacked dummyVertexPacked[2];

const int Attributes = 10;
MPI_Datatype subtypes[Attributes] = {
MPI_INT,		 //indicesOfAdjacentCellDescriptions
MPI_INT,		 //adjacentRanksInFormerIteration
MPI_INT,		 //ageInGridIterations
MPI_INT,		 //adjacentCellsHeight
MPI_DOUBLE,		 //x
MPI_INT,		 //level
MPI_INT,		 //_packedRecords0
MPI_INT,		 //adjacentCellsHeightOfPreviousIteration
MPI_INT,		 //numberOfAdjacentRefinedCells
MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
TWO_POWER_D,		 //indicesOfAdjacentCellDescriptions
TWO_POWER_D,		 //adjacentRanksInFormerIteration
1,		 //ageInGridIterations
1,		 //adjacentCellsHeight
DIMENSIONS,		 //x
1,		 //level
1,		 //_packedRecords0
1,		 //adjacentCellsHeightOfPreviousIteration
1,		 //numberOfAdjacentRefinedCells
1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._indicesOfAdjacentCellDescriptions[0]))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._adjacentRanksInFormerIteration[0]))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._ageInGridIterations))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._adjacentCellsHeight))), 		&disp[3] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._x[0]))), 		&disp[4] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._level))), 		&disp[5] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._packedRecords0))), 		&disp[6] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._adjacentCellsHeightOfPreviousIteration))), 		&disp[7] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._numberOfAdjacentRefinedCells))), 		&disp[8] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyVertexPacked[1]._persistentRecords._indicesOfAdjacentCellDescriptions[0])), 		&disp[9] );

for (int i=1; i<Attributes; i++) {
assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &VertexPacked::FullDatatype );
MPI_Type_commit( &VertexPacked::FullDatatype );

}

}


void peanoclaw::records::VertexPacked::shutdownDatatype() {
MPI_Type_free( &VertexPacked::Datatype );
MPI_Type_free( &VertexPacked::FullDatatype );

}

void peanoclaw::records::VertexPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
_senderDestinationRank = destination;

if (communicateBlocking) {

const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
if  (result!=MPI_SUCCESS) {
std::ostringstream msg;
msg << "was not able to send message peanoclaw::records::VertexPacked "
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
msg << "was not able to send message peanoclaw::records::VertexPacked "
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
msg << "testing for finished send task for peanoclaw::records::VertexPacked "
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
"peanoclaw::records::VertexPacked",
"send(int)", destination,tag,1
);
triggeredTimeoutWarning = true;
}
if (
tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
(clock()>timeOutShutdown)
) {
tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
"peanoclaw::records::VertexPacked",
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



void peanoclaw::records::VertexPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
if (communicateBlocking) {

MPI_Status  status;
const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
_senderDestinationRank = status.MPI_SOURCE;
if ( result != MPI_SUCCESS ) {
std::ostringstream msg;
msg << "failed to start to receive peanoclaw::records::VertexPacked from node "
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
msg << "failed to start to receive peanoclaw::records::VertexPacked from node "
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
msg << "testing for finished receive task for peanoclaw::records::VertexPacked failed: "
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
"peanoclaw::records::VertexPacked",
"receive(int)", source,tag,1
);
triggeredTimeoutWarning = true;
}
if (
tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
(clock()>timeOutShutdown)
) {
tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
"peanoclaw::records::VertexPacked",
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



bool peanoclaw::records::VertexPacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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

int peanoclaw::records::VertexPacked::getSenderRank() const {
assertion( _senderDestinationRank!=-1 );
return _senderDestinationRank;

}
#endif




#elif defined(Parallel) && !defined(Asserts)
peanoclaw::records::Vertex::PersistentRecords::PersistentRecords() {

}


peanoclaw::records::Vertex::PersistentRecords::PersistentRecords(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank):
_indicesOfAdjacentCellDescriptions(indicesOfAdjacentCellDescriptions),
_adjacentSubcellsEraseVeto(adjacentSubcellsEraseVeto),
_adjacentRanksInFormerIteration(adjacentRanksInFormerIteration),
_adjacentRanksChanged(adjacentRanksChanged),
_shouldRefine(shouldRefine),
_ageInGridIterations(ageInGridIterations),
_isHangingNode(isHangingNode),
_refinementControl(refinementControl),
_adjacentCellsHeight(adjacentCellsHeight),
_insideOutsideDomain(insideOutsideDomain),
_adjacentRanks(adjacentRanks),
_adjacentSubtreeForksIntoOtherRank(adjacentSubtreeForksIntoOtherRank) {

}

peanoclaw::records::Vertex::Vertex() {

}


peanoclaw::records::Vertex::Vertex(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._indicesOfAdjacentCellDescriptions, persistentRecords._adjacentSubcellsEraseVeto, persistentRecords._adjacentRanksInFormerIteration, persistentRecords._adjacentRanksChanged, persistentRecords._shouldRefine, persistentRecords._ageInGridIterations, persistentRecords._isHangingNode, persistentRecords._refinementControl, persistentRecords._adjacentCellsHeight, persistentRecords._insideOutsideDomain, persistentRecords._adjacentRanks, persistentRecords._adjacentSubtreeForksIntoOtherRank) {

}


peanoclaw::records::Vertex::Vertex(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank):
_persistentRecords(indicesOfAdjacentCellDescriptions, adjacentSubcellsEraseVeto, adjacentRanksInFormerIteration, adjacentRanksChanged, shouldRefine, ageInGridIterations, isHangingNode, refinementControl, adjacentCellsHeight, insideOutsideDomain, adjacentRanks, adjacentSubtreeForksIntoOtherRank) {

}


peanoclaw::records::Vertex::Vertex(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank):
_persistentRecords(indicesOfAdjacentCellDescriptions, adjacentSubcellsEraseVeto, adjacentRanksInFormerIteration, adjacentRanksChanged, shouldRefine, ageInGridIterations, isHangingNode, refinementControl, adjacentCellsHeight, insideOutsideDomain, adjacentRanks, adjacentSubtreeForksIntoOtherRank),_adjacentCellsHeightOfPreviousIteration(adjacentCellsHeightOfPreviousIteration),
_numberOfAdjacentRefinedCells(numberOfAdjacentRefinedCells) {

}

peanoclaw::records::Vertex::~Vertex() { }

std::string peanoclaw::records::Vertex::toString(const InsideOutsideDomain& param) {
switch (param) {
case Inside: return "Inside";
case Boundary: return "Boundary";
case Outside: return "Outside";
}
return "undefined";
}

std::string peanoclaw::records::Vertex::getInsideOutsideDomainMapping() {
return "InsideOutsideDomain(Inside=0,Boundary=1,Outside=2)";
}
std::string peanoclaw::records::Vertex::toString(const RefinementControl& param) {
switch (param) {
case Unrefined: return "Unrefined";
case Refined: return "Refined";
case RefinementTriggered: return "RefinementTriggered";
case Refining: return "Refining";
case EraseTriggered: return "EraseTriggered";
case Erasing: return "Erasing";
case RefineDueToJoinThoughWorkerIsAlreadyErasing: return "RefineDueToJoinThoughWorkerIsAlreadyErasing";
}
return "undefined";
}

std::string peanoclaw::records::Vertex::getRefinementControlMapping() {
return "RefinementControl(Unrefined=0,Refined=1,RefinementTriggered=2,Refining=3,EraseTriggered=4,Erasing=5,RefineDueToJoinThoughWorkerIsAlreadyErasing=6)";
}


std::string peanoclaw::records::Vertex::toString() const {
std::ostringstream stringstr;
toString(stringstr);
return stringstr.str();
}

void peanoclaw::records::Vertex::toString (std::ostream& out) const {
out << "("; 
out << "indicesOfAdjacentCellDescriptions:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getIndicesOfAdjacentCellDescriptions(i) << ",";
   }
   out << getIndicesOfAdjacentCellDescriptions(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentSubcellsEraseVeto:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentSubcellsEraseVeto(i) << ",";
   }
   out << getAdjacentSubcellsEraseVeto(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentRanksInFormerIteration:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentRanksInFormerIteration(i) << ",";
   }
   out << getAdjacentRanksInFormerIteration(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentRanksChanged:" << getAdjacentRanksChanged();
out << ",";
out << "shouldRefine:" << getShouldRefine();
out << ",";
out << "ageInGridIterations:" << getAgeInGridIterations();
out << ",";
out << "isHangingNode:" << getIsHangingNode();
out << ",";
out << "refinementControl:" << toString(getRefinementControl());
out << ",";
out << "adjacentCellsHeight:" << getAdjacentCellsHeight();
out << ",";
out << "adjacentCellsHeightOfPreviousIteration:" << getAdjacentCellsHeightOfPreviousIteration();
out << ",";
out << "numberOfAdjacentRefinedCells:" << getNumberOfAdjacentRefinedCells();
out << ",";
out << "insideOutsideDomain:" << toString(getInsideOutsideDomain());
out << ",";
out << "adjacentRanks:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentRanks(i) << ",";
   }
   out << getAdjacentRanks(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentSubtreeForksIntoOtherRank:" << getAdjacentSubtreeForksIntoOtherRank();
out <<  ")";
}


peanoclaw::records::Vertex::PersistentRecords peanoclaw::records::Vertex::getPersistentRecords() const {
return _persistentRecords;
}

peanoclaw::records::VertexPacked peanoclaw::records::Vertex::convert() const{
return VertexPacked(
getIndicesOfAdjacentCellDescriptions(),
getAdjacentSubcellsEraseVeto(),
getAdjacentRanksInFormerIteration(),
getAdjacentRanksChanged(),
getShouldRefine(),
getAgeInGridIterations(),
getIsHangingNode(),
getRefinementControl(),
getAdjacentCellsHeight(),
getAdjacentCellsHeightOfPreviousIteration(),
getNumberOfAdjacentRefinedCells(),
getInsideOutsideDomain(),
getAdjacentRanks(),
getAdjacentSubtreeForksIntoOtherRank()
);
}

#ifdef Parallel
tarch::logging::Log peanoclaw::records::Vertex::_log( "peanoclaw::records::Vertex" );

MPI_Datatype peanoclaw::records::Vertex::Datatype = 0;
MPI_Datatype peanoclaw::records::Vertex::FullDatatype = 0;


void peanoclaw::records::Vertex::initDatatype() {
{
Vertex dummyVertex[2];

const int Attributes = 11;
MPI_Datatype subtypes[Attributes] = {
MPI_INT,		 //indicesOfAdjacentCellDescriptions
MPI_INT,		 //adjacentSubcellsEraseVeto
MPI_CHAR,		 //adjacentRanksChanged
MPI_CHAR,		 //shouldRefine
MPI_INT,		 //ageInGridIterations
MPI_CHAR,		 //isHangingNode
MPI_INT,		 //refinementControl
MPI_INT,		 //adjacentRanks
MPI_CHAR,		 //adjacentSubtreeForksIntoOtherRank
MPI_INT,		 //numberOfAdjacentRefinedCells
MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
TWO_POWER_D,		 //indicesOfAdjacentCellDescriptions
TWO_POWER_D,		 //adjacentSubcellsEraseVeto
1,		 //adjacentRanksChanged
1,		 //shouldRefine
1,		 //ageInGridIterations
1,		 //isHangingNode
1,		 //refinementControl
TWO_POWER_D,		 //adjacentRanks
1,		 //adjacentSubtreeForksIntoOtherRank
1,		 //numberOfAdjacentRefinedCells
1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._indicesOfAdjacentCellDescriptions[0]))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentSubcellsEraseVeto))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentRanksChanged))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._shouldRefine))), 		&disp[3] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._ageInGridIterations))), 		&disp[4] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._isHangingNode))), 		&disp[5] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._refinementControl))), 		&disp[6] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentRanks[0]))), 		&disp[7] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentSubtreeForksIntoOtherRank))), 		&disp[8] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._numberOfAdjacentRefinedCells))), 		&disp[9] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyVertex[1]._persistentRecords._indicesOfAdjacentCellDescriptions[0])), 		&disp[10] );

for (int i=1; i<Attributes; i++) {
assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Vertex::Datatype );
MPI_Type_commit( &Vertex::Datatype );

}
{
Vertex dummyVertex[2];

const int Attributes = 15;
MPI_Datatype subtypes[Attributes] = {
MPI_INT,		 //indicesOfAdjacentCellDescriptions
MPI_INT,		 //adjacentSubcellsEraseVeto
MPI_INT,		 //adjacentRanksInFormerIteration
MPI_CHAR,		 //adjacentRanksChanged
MPI_CHAR,		 //shouldRefine
MPI_INT,		 //ageInGridIterations
MPI_CHAR,		 //isHangingNode
MPI_INT,		 //refinementControl
MPI_INT,		 //adjacentCellsHeight
MPI_INT,		 //insideOutsideDomain
MPI_INT,		 //adjacentRanks
MPI_CHAR,		 //adjacentSubtreeForksIntoOtherRank
MPI_INT,		 //adjacentCellsHeightOfPreviousIteration
MPI_INT,		 //numberOfAdjacentRefinedCells
MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
TWO_POWER_D,		 //indicesOfAdjacentCellDescriptions
TWO_POWER_D,		 //adjacentSubcellsEraseVeto
TWO_POWER_D,		 //adjacentRanksInFormerIteration
1,		 //adjacentRanksChanged
1,		 //shouldRefine
1,		 //ageInGridIterations
1,		 //isHangingNode
1,		 //refinementControl
1,		 //adjacentCellsHeight
1,		 //insideOutsideDomain
TWO_POWER_D,		 //adjacentRanks
1,		 //adjacentSubtreeForksIntoOtherRank
1,		 //adjacentCellsHeightOfPreviousIteration
1,		 //numberOfAdjacentRefinedCells
1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._indicesOfAdjacentCellDescriptions[0]))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentSubcellsEraseVeto))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentRanksInFormerIteration[0]))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentRanksChanged))), 		&disp[3] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._shouldRefine))), 		&disp[4] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._ageInGridIterations))), 		&disp[5] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._isHangingNode))), 		&disp[6] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._refinementControl))), 		&disp[7] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentCellsHeight))), 		&disp[8] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._insideOutsideDomain))), 		&disp[9] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentRanks[0]))), 		&disp[10] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._persistentRecords._adjacentSubtreeForksIntoOtherRank))), 		&disp[11] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._adjacentCellsHeightOfPreviousIteration))), 		&disp[12] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertex[0]._numberOfAdjacentRefinedCells))), 		&disp[13] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyVertex[1]._persistentRecords._indicesOfAdjacentCellDescriptions[0])), 		&disp[14] );

for (int i=1; i<Attributes; i++) {
assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &Vertex::FullDatatype );
MPI_Type_commit( &Vertex::FullDatatype );

}

}


void peanoclaw::records::Vertex::shutdownDatatype() {
MPI_Type_free( &Vertex::Datatype );
MPI_Type_free( &Vertex::FullDatatype );

}

void peanoclaw::records::Vertex::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
_senderDestinationRank = destination;

if (communicateBlocking) {

const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
if  (result!=MPI_SUCCESS) {
std::ostringstream msg;
msg << "was not able to send message peanoclaw::records::Vertex "
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
msg << "was not able to send message peanoclaw::records::Vertex "
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
msg << "testing for finished send task for peanoclaw::records::Vertex "
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
"peanoclaw::records::Vertex",
"send(int)", destination,tag,1
);
triggeredTimeoutWarning = true;
}
if (
tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
(clock()>timeOutShutdown)
) {
tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
"peanoclaw::records::Vertex",
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



void peanoclaw::records::Vertex::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
if (communicateBlocking) {

MPI_Status  status;
const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
_senderDestinationRank = status.MPI_SOURCE;
if ( result != MPI_SUCCESS ) {
std::ostringstream msg;
msg << "failed to start to receive peanoclaw::records::Vertex from node "
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
msg << "failed to start to receive peanoclaw::records::Vertex from node "
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
msg << "testing for finished receive task for peanoclaw::records::Vertex failed: "
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
"peanoclaw::records::Vertex",
"receive(int)", source,tag,1
);
triggeredTimeoutWarning = true;
}
if (
tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
(clock()>timeOutShutdown)
) {
tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
"peanoclaw::records::Vertex",
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



bool peanoclaw::records::Vertex::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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

int peanoclaw::records::Vertex::getSenderRank() const {
assertion( _senderDestinationRank!=-1 );
return _senderDestinationRank;

}
#endif


peanoclaw::records::VertexPacked::PersistentRecords::PersistentRecords() {
assertion((TWO_POWER_D+9 < (8 * sizeof(int))));

}


peanoclaw::records::VertexPacked::PersistentRecords::PersistentRecords(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank):
_indicesOfAdjacentCellDescriptions(indicesOfAdjacentCellDescriptions),
_adjacentRanksInFormerIteration(adjacentRanksInFormerIteration),
_ageInGridIterations(ageInGridIterations),
_adjacentCellsHeight(adjacentCellsHeight),
_adjacentRanks(adjacentRanks) {
setAdjacentSubcellsEraseVeto(adjacentSubcellsEraseVeto);
setAdjacentRanksChanged(adjacentRanksChanged);
setShouldRefine(shouldRefine);
setIsHangingNode(isHangingNode);
setRefinementControl(refinementControl);
setInsideOutsideDomain(insideOutsideDomain);
setAdjacentSubtreeForksIntoOtherRank(adjacentSubtreeForksIntoOtherRank);
assertion((TWO_POWER_D+9 < (8 * sizeof(int))));

}

peanoclaw::records::VertexPacked::VertexPacked() {
assertion((TWO_POWER_D+9 < (8 * sizeof(int))));

}


peanoclaw::records::VertexPacked::VertexPacked(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._indicesOfAdjacentCellDescriptions, persistentRecords.getAdjacentSubcellsEraseVeto(), persistentRecords._adjacentRanksInFormerIteration, persistentRecords.getAdjacentRanksChanged(), persistentRecords.getShouldRefine(), persistentRecords._ageInGridIterations, persistentRecords.getIsHangingNode(), persistentRecords.getRefinementControl(), persistentRecords._adjacentCellsHeight, persistentRecords.getInsideOutsideDomain(), persistentRecords._adjacentRanks, persistentRecords.getAdjacentSubtreeForksIntoOtherRank()) {
assertion((TWO_POWER_D+9 < (8 * sizeof(int))));

}


peanoclaw::records::VertexPacked::VertexPacked(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank):
_persistentRecords(indicesOfAdjacentCellDescriptions, adjacentSubcellsEraseVeto, adjacentRanksInFormerIteration, adjacentRanksChanged, shouldRefine, ageInGridIterations, isHangingNode, refinementControl, adjacentCellsHeight, insideOutsideDomain, adjacentRanks, adjacentSubtreeForksIntoOtherRank) {
assertion((TWO_POWER_D+9 < (8 * sizeof(int))));

}


peanoclaw::records::VertexPacked::VertexPacked(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const std::bitset<TWO_POWER_D>& adjacentSubcellsEraseVeto, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanksInFormerIteration, const bool& adjacentRanksChanged, const bool& shouldRefine, const int& ageInGridIterations, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank):
_persistentRecords(indicesOfAdjacentCellDescriptions, adjacentSubcellsEraseVeto, adjacentRanksInFormerIteration, adjacentRanksChanged, shouldRefine, ageInGridIterations, isHangingNode, refinementControl, adjacentCellsHeight, insideOutsideDomain, adjacentRanks, adjacentSubtreeForksIntoOtherRank),_adjacentCellsHeightOfPreviousIteration(adjacentCellsHeightOfPreviousIteration),
_numberOfAdjacentRefinedCells(numberOfAdjacentRefinedCells) {
assertion((TWO_POWER_D+9 < (8 * sizeof(int))));

}

peanoclaw::records::VertexPacked::~VertexPacked() { }

std::string peanoclaw::records::VertexPacked::toString(const InsideOutsideDomain& param) {
return peanoclaw::records::Vertex::toString(param);
}

std::string peanoclaw::records::VertexPacked::getInsideOutsideDomainMapping() {
return peanoclaw::records::Vertex::getInsideOutsideDomainMapping();
}

std::string peanoclaw::records::VertexPacked::toString(const RefinementControl& param) {
return peanoclaw::records::Vertex::toString(param);
}

std::string peanoclaw::records::VertexPacked::getRefinementControlMapping() {
return peanoclaw::records::Vertex::getRefinementControlMapping();
}



std::string peanoclaw::records::VertexPacked::toString() const {
std::ostringstream stringstr;
toString(stringstr);
return stringstr.str();
}

void peanoclaw::records::VertexPacked::toString (std::ostream& out) const {
out << "("; 
out << "indicesOfAdjacentCellDescriptions:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getIndicesOfAdjacentCellDescriptions(i) << ",";
   }
   out << getIndicesOfAdjacentCellDescriptions(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentSubcellsEraseVeto:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentSubcellsEraseVeto(i) << ",";
   }
   out << getAdjacentSubcellsEraseVeto(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentRanksInFormerIteration:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentRanksInFormerIteration(i) << ",";
   }
   out << getAdjacentRanksInFormerIteration(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentRanksChanged:" << getAdjacentRanksChanged();
out << ",";
out << "shouldRefine:" << getShouldRefine();
out << ",";
out << "ageInGridIterations:" << getAgeInGridIterations();
out << ",";
out << "isHangingNode:" << getIsHangingNode();
out << ",";
out << "refinementControl:" << toString(getRefinementControl());
out << ",";
out << "adjacentCellsHeight:" << getAdjacentCellsHeight();
out << ",";
out << "adjacentCellsHeightOfPreviousIteration:" << getAdjacentCellsHeightOfPreviousIteration();
out << ",";
out << "numberOfAdjacentRefinedCells:" << getNumberOfAdjacentRefinedCells();
out << ",";
out << "insideOutsideDomain:" << toString(getInsideOutsideDomain());
out << ",";
out << "adjacentRanks:[";
   for (int i = 0; i < TWO_POWER_D-1; i++) {
      out << getAdjacentRanks(i) << ",";
   }
   out << getAdjacentRanks(TWO_POWER_D-1) << "]";
out << ",";
out << "adjacentSubtreeForksIntoOtherRank:" << getAdjacentSubtreeForksIntoOtherRank();
out <<  ")";
}


peanoclaw::records::VertexPacked::PersistentRecords peanoclaw::records::VertexPacked::getPersistentRecords() const {
return _persistentRecords;
}

peanoclaw::records::Vertex peanoclaw::records::VertexPacked::convert() const{
return Vertex(
getIndicesOfAdjacentCellDescriptions(),
getAdjacentSubcellsEraseVeto(),
getAdjacentRanksInFormerIteration(),
getAdjacentRanksChanged(),
getShouldRefine(),
getAgeInGridIterations(),
getIsHangingNode(),
getRefinementControl(),
getAdjacentCellsHeight(),
getAdjacentCellsHeightOfPreviousIteration(),
getNumberOfAdjacentRefinedCells(),
getInsideOutsideDomain(),
getAdjacentRanks(),
getAdjacentSubtreeForksIntoOtherRank()
);
}

#ifdef Parallel
tarch::logging::Log peanoclaw::records::VertexPacked::_log( "peanoclaw::records::VertexPacked" );

MPI_Datatype peanoclaw::records::VertexPacked::Datatype = 0;
MPI_Datatype peanoclaw::records::VertexPacked::FullDatatype = 0;


void peanoclaw::records::VertexPacked::initDatatype() {
{
VertexPacked dummyVertexPacked[2];

const int Attributes = 6;
MPI_Datatype subtypes[Attributes] = {
MPI_INT,		 //indicesOfAdjacentCellDescriptions
MPI_INT,		 //ageInGridIterations
MPI_INT,		 //adjacentRanks
MPI_INT,		 //_packedRecords0
MPI_INT,		 //numberOfAdjacentRefinedCells
MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
TWO_POWER_D,		 //indicesOfAdjacentCellDescriptions
1,		 //ageInGridIterations
TWO_POWER_D,		 //adjacentRanks
1,		 //_packedRecords0
1,		 //numberOfAdjacentRefinedCells
1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._indicesOfAdjacentCellDescriptions[0]))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._ageInGridIterations))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._adjacentRanks[0]))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._packedRecords0))), 		&disp[3] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._numberOfAdjacentRefinedCells))), 		&disp[4] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyVertexPacked[1]._persistentRecords._indicesOfAdjacentCellDescriptions[0])), 		&disp[5] );

for (int i=1; i<Attributes; i++) {
assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &VertexPacked::Datatype );
MPI_Type_commit( &VertexPacked::Datatype );

}
{
VertexPacked dummyVertexPacked[2];

const int Attributes = 9;
MPI_Datatype subtypes[Attributes] = {
MPI_INT,		 //indicesOfAdjacentCellDescriptions
MPI_INT,		 //adjacentRanksInFormerIteration
MPI_INT,		 //ageInGridIterations
MPI_INT,		 //adjacentCellsHeight
MPI_INT,		 //adjacentRanks
MPI_INT,		 //_packedRecords0
MPI_INT,		 //adjacentCellsHeightOfPreviousIteration
MPI_INT,		 //numberOfAdjacentRefinedCells
MPI_UB		 // end/displacement flag
};

int blocklen[Attributes] = {
TWO_POWER_D,		 //indicesOfAdjacentCellDescriptions
TWO_POWER_D,		 //adjacentRanksInFormerIteration
1,		 //ageInGridIterations
1,		 //adjacentCellsHeight
TWO_POWER_D,		 //adjacentRanks
1,		 //_packedRecords0
1,		 //adjacentCellsHeightOfPreviousIteration
1,		 //numberOfAdjacentRefinedCells
1		 // end/displacement flag
};

MPI_Aint     disp[Attributes];

MPI_Aint base;
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]))), &base);
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._indicesOfAdjacentCellDescriptions[0]))), 		&disp[0] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._adjacentRanksInFormerIteration[0]))), 		&disp[1] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._ageInGridIterations))), 		&disp[2] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._adjacentCellsHeight))), 		&disp[3] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._adjacentRanks[0]))), 		&disp[4] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._persistentRecords._packedRecords0))), 		&disp[5] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._adjacentCellsHeightOfPreviousIteration))), 		&disp[6] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyVertexPacked[0]._numberOfAdjacentRefinedCells))), 		&disp[7] );
MPI_Address( const_cast<void*>(static_cast<const void*>(&dummyVertexPacked[1]._persistentRecords._indicesOfAdjacentCellDescriptions[0])), 		&disp[8] );

for (int i=1; i<Attributes; i++) {
assertion1( disp[i] > disp[i-1], i );
}
for (int i=0; i<Attributes; i++) {
disp[i] -= base;
}
MPI_Type_struct( Attributes, blocklen, disp, subtypes, &VertexPacked::FullDatatype );
MPI_Type_commit( &VertexPacked::FullDatatype );

}

}


void peanoclaw::records::VertexPacked::shutdownDatatype() {
MPI_Type_free( &VertexPacked::Datatype );
MPI_Type_free( &VertexPacked::FullDatatype );

}

void peanoclaw::records::VertexPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
_senderDestinationRank = destination;

if (communicateBlocking) {

const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
if  (result!=MPI_SUCCESS) {
std::ostringstream msg;
msg << "was not able to send message peanoclaw::records::VertexPacked "
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
msg << "was not able to send message peanoclaw::records::VertexPacked "
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
msg << "testing for finished send task for peanoclaw::records::VertexPacked "
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
"peanoclaw::records::VertexPacked",
"send(int)", destination,tag,1
);
triggeredTimeoutWarning = true;
}
if (
tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
(clock()>timeOutShutdown)
) {
tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
"peanoclaw::records::VertexPacked",
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



void peanoclaw::records::VertexPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
if (communicateBlocking) {

MPI_Status  status;
const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
_senderDestinationRank = status.MPI_SOURCE;
if ( result != MPI_SUCCESS ) {
std::ostringstream msg;
msg << "failed to start to receive peanoclaw::records::VertexPacked from node "
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
msg << "failed to start to receive peanoclaw::records::VertexPacked from node "
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
msg << "testing for finished receive task for peanoclaw::records::VertexPacked failed: "
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
"peanoclaw::records::VertexPacked",
"receive(int)", source,tag,1
);
triggeredTimeoutWarning = true;
}
if (
tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
(clock()>timeOutShutdown)
) {
tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
"peanoclaw::records::VertexPacked",
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



bool peanoclaw::records::VertexPacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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

int peanoclaw::records::VertexPacked::getSenderRank() const {
assertion( _senderDestinationRank!=-1 );
return _senderDestinationRank;

}
#endif




#endif


