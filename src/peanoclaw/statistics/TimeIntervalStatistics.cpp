#include "peanoclaw/statistics/TimeIntervalStatistics.h"

peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::PersistentRecords() {
   
}


peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::PersistentRecords(const int& minimalPatchIndex, const int& minimalPatchParentIndex, const double& minimalPatchTime, const double& startMaximumLocalTimeInterval, const double& endMaximumLocalTimeInterval, const double& startMinimumLocalTimeInterval, const double& endMinimumLocalTimeInterval, const double& minimalTimestep, const bool& allPatchesEvolvedToGlobalTimestep, const double& averageGlobalTimeInterval, const double& globalTimestepEndTime, const bool& minimalPatchBlockedDueToCoarsening, const bool& minimalPatchBlockedDueToGlobalTimestep):
_minimalPatchIndex(minimalPatchIndex),
_minimalPatchParentIndex(minimalPatchParentIndex),
_minimalPatchTime(minimalPatchTime),
_startMaximumLocalTimeInterval(startMaximumLocalTimeInterval),
_endMaximumLocalTimeInterval(endMaximumLocalTimeInterval),
_startMinimumLocalTimeInterval(startMinimumLocalTimeInterval),
_endMinimumLocalTimeInterval(endMinimumLocalTimeInterval),
_minimalTimestep(minimalTimestep),
_allPatchesEvolvedToGlobalTimestep(allPatchesEvolvedToGlobalTimestep),
_averageGlobalTimeInterval(averageGlobalTimeInterval),
_globalTimestepEndTime(globalTimestepEndTime),
_minimalPatchBlockedDueToCoarsening(minimalPatchBlockedDueToCoarsening),
_minimalPatchBlockedDueToGlobalTimestep(minimalPatchBlockedDueToGlobalTimestep) {
   
}


 int peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::getMinimalPatchIndex() const  {
   return _minimalPatchIndex;
}



 void peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::setMinimalPatchIndex(const int& minimalPatchIndex)  {
   _minimalPatchIndex = minimalPatchIndex;
}



 int peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::getMinimalPatchParentIndex() const  {
   return _minimalPatchParentIndex;
}



 void peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::setMinimalPatchParentIndex(const int& minimalPatchParentIndex)  {
   _minimalPatchParentIndex = minimalPatchParentIndex;
}



 double peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::getMinimalPatchTime() const  {
   return _minimalPatchTime;
}



 void peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::setMinimalPatchTime(const double& minimalPatchTime)  {
   _minimalPatchTime = minimalPatchTime;
}



 double peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::getStartMaximumLocalTimeInterval() const  {
   return _startMaximumLocalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::setStartMaximumLocalTimeInterval(const double& startMaximumLocalTimeInterval)  {
   _startMaximumLocalTimeInterval = startMaximumLocalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::getEndMaximumLocalTimeInterval() const  {
   return _endMaximumLocalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::setEndMaximumLocalTimeInterval(const double& endMaximumLocalTimeInterval)  {
   _endMaximumLocalTimeInterval = endMaximumLocalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::getStartMinimumLocalTimeInterval() const  {
   return _startMinimumLocalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::setStartMinimumLocalTimeInterval(const double& startMinimumLocalTimeInterval)  {
   _startMinimumLocalTimeInterval = startMinimumLocalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::getEndMinimumLocalTimeInterval() const  {
   return _endMinimumLocalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::setEndMinimumLocalTimeInterval(const double& endMinimumLocalTimeInterval)  {
   _endMinimumLocalTimeInterval = endMinimumLocalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::getMinimalTimestep() const  {
   return _minimalTimestep;
}



 void peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::setMinimalTimestep(const double& minimalTimestep)  {
   _minimalTimestep = minimalTimestep;
}



 bool peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::getAllPatchesEvolvedToGlobalTimestep() const  {
   return _allPatchesEvolvedToGlobalTimestep;
}



 void peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::setAllPatchesEvolvedToGlobalTimestep(const bool& allPatchesEvolvedToGlobalTimestep)  {
   _allPatchesEvolvedToGlobalTimestep = allPatchesEvolvedToGlobalTimestep;
}



 double peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::getAverageGlobalTimeInterval() const  {
   return _averageGlobalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::setAverageGlobalTimeInterval(const double& averageGlobalTimeInterval)  {
   _averageGlobalTimeInterval = averageGlobalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::getGlobalTimestepEndTime() const  {
   return _globalTimestepEndTime;
}



 void peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::setGlobalTimestepEndTime(const double& globalTimestepEndTime)  {
   _globalTimestepEndTime = globalTimestepEndTime;
}



 bool peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::getMinimalPatchBlockedDueToCoarsening() const  {
   return _minimalPatchBlockedDueToCoarsening;
}



 void peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::setMinimalPatchBlockedDueToCoarsening(const bool& minimalPatchBlockedDueToCoarsening)  {
   _minimalPatchBlockedDueToCoarsening = minimalPatchBlockedDueToCoarsening;
}



 bool peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::getMinimalPatchBlockedDueToGlobalTimestep() const  {
   return _minimalPatchBlockedDueToGlobalTimestep;
}



 void peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords::setMinimalPatchBlockedDueToGlobalTimestep(const bool& minimalPatchBlockedDueToGlobalTimestep)  {
   _minimalPatchBlockedDueToGlobalTimestep = minimalPatchBlockedDueToGlobalTimestep;
}


peanoclaw::statistics::TimeIntervalStatistics::TimeIntervalStatistics() {
   
}


peanoclaw::statistics::TimeIntervalStatistics::TimeIntervalStatistics(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._minimalPatchIndex, persistentRecords._minimalPatchParentIndex, persistentRecords._minimalPatchTime, persistentRecords._startMaximumLocalTimeInterval, persistentRecords._endMaximumLocalTimeInterval, persistentRecords._startMinimumLocalTimeInterval, persistentRecords._endMinimumLocalTimeInterval, persistentRecords._minimalTimestep, persistentRecords._allPatchesEvolvedToGlobalTimestep, persistentRecords._averageGlobalTimeInterval, persistentRecords._globalTimestepEndTime, persistentRecords._minimalPatchBlockedDueToCoarsening, persistentRecords._minimalPatchBlockedDueToGlobalTimestep) {
   
}


peanoclaw::statistics::TimeIntervalStatistics::TimeIntervalStatistics(const int& minimalPatchIndex, const int& minimalPatchParentIndex, const double& minimalPatchTime, const double& startMaximumLocalTimeInterval, const double& endMaximumLocalTimeInterval, const double& startMinimumLocalTimeInterval, const double& endMinimumLocalTimeInterval, const double& minimalTimestep, const bool& allPatchesEvolvedToGlobalTimestep, const double& averageGlobalTimeInterval, const double& globalTimestepEndTime, const bool& minimalPatchBlockedDueToCoarsening, const bool& minimalPatchBlockedDueToGlobalTimestep):
_persistentRecords(minimalPatchIndex, minimalPatchParentIndex, minimalPatchTime, startMaximumLocalTimeInterval, endMaximumLocalTimeInterval, startMinimumLocalTimeInterval, endMinimumLocalTimeInterval, minimalTimestep, allPatchesEvolvedToGlobalTimestep, averageGlobalTimeInterval, globalTimestepEndTime, minimalPatchBlockedDueToCoarsening, minimalPatchBlockedDueToGlobalTimestep) {
   
}


peanoclaw::statistics::TimeIntervalStatistics::~TimeIntervalStatistics() { }


 int peanoclaw::statistics::TimeIntervalStatistics::getMinimalPatchIndex() const  {
   return _persistentRecords._minimalPatchIndex;
}



 void peanoclaw::statistics::TimeIntervalStatistics::setMinimalPatchIndex(const int& minimalPatchIndex)  {
   _persistentRecords._minimalPatchIndex = minimalPatchIndex;
}



 int peanoclaw::statistics::TimeIntervalStatistics::getMinimalPatchParentIndex() const  {
   return _persistentRecords._minimalPatchParentIndex;
}



 void peanoclaw::statistics::TimeIntervalStatistics::setMinimalPatchParentIndex(const int& minimalPatchParentIndex)  {
   _persistentRecords._minimalPatchParentIndex = minimalPatchParentIndex;
}



 double peanoclaw::statistics::TimeIntervalStatistics::getMinimalPatchTime() const  {
   return _persistentRecords._minimalPatchTime;
}



 void peanoclaw::statistics::TimeIntervalStatistics::setMinimalPatchTime(const double& minimalPatchTime)  {
   _persistentRecords._minimalPatchTime = minimalPatchTime;
}



 double peanoclaw::statistics::TimeIntervalStatistics::getStartMaximumLocalTimeInterval() const  {
   return _persistentRecords._startMaximumLocalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatistics::setStartMaximumLocalTimeInterval(const double& startMaximumLocalTimeInterval)  {
   _persistentRecords._startMaximumLocalTimeInterval = startMaximumLocalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatistics::getEndMaximumLocalTimeInterval() const  {
   return _persistentRecords._endMaximumLocalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatistics::setEndMaximumLocalTimeInterval(const double& endMaximumLocalTimeInterval)  {
   _persistentRecords._endMaximumLocalTimeInterval = endMaximumLocalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatistics::getStartMinimumLocalTimeInterval() const  {
   return _persistentRecords._startMinimumLocalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatistics::setStartMinimumLocalTimeInterval(const double& startMinimumLocalTimeInterval)  {
   _persistentRecords._startMinimumLocalTimeInterval = startMinimumLocalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatistics::getEndMinimumLocalTimeInterval() const  {
   return _persistentRecords._endMinimumLocalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatistics::setEndMinimumLocalTimeInterval(const double& endMinimumLocalTimeInterval)  {
   _persistentRecords._endMinimumLocalTimeInterval = endMinimumLocalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatistics::getMinimalTimestep() const  {
   return _persistentRecords._minimalTimestep;
}



 void peanoclaw::statistics::TimeIntervalStatistics::setMinimalTimestep(const double& minimalTimestep)  {
   _persistentRecords._minimalTimestep = minimalTimestep;
}



 bool peanoclaw::statistics::TimeIntervalStatistics::getAllPatchesEvolvedToGlobalTimestep() const  {
   return _persistentRecords._allPatchesEvolvedToGlobalTimestep;
}



 void peanoclaw::statistics::TimeIntervalStatistics::setAllPatchesEvolvedToGlobalTimestep(const bool& allPatchesEvolvedToGlobalTimestep)  {
   _persistentRecords._allPatchesEvolvedToGlobalTimestep = allPatchesEvolvedToGlobalTimestep;
}



 double peanoclaw::statistics::TimeIntervalStatistics::getAverageGlobalTimeInterval() const  {
   return _persistentRecords._averageGlobalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatistics::setAverageGlobalTimeInterval(const double& averageGlobalTimeInterval)  {
   _persistentRecords._averageGlobalTimeInterval = averageGlobalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatistics::getGlobalTimestepEndTime() const  {
   return _persistentRecords._globalTimestepEndTime;
}



 void peanoclaw::statistics::TimeIntervalStatistics::setGlobalTimestepEndTime(const double& globalTimestepEndTime)  {
   _persistentRecords._globalTimestepEndTime = globalTimestepEndTime;
}



 bool peanoclaw::statistics::TimeIntervalStatistics::getMinimalPatchBlockedDueToCoarsening() const  {
   return _persistentRecords._minimalPatchBlockedDueToCoarsening;
}



 void peanoclaw::statistics::TimeIntervalStatistics::setMinimalPatchBlockedDueToCoarsening(const bool& minimalPatchBlockedDueToCoarsening)  {
   _persistentRecords._minimalPatchBlockedDueToCoarsening = minimalPatchBlockedDueToCoarsening;
}



 bool peanoclaw::statistics::TimeIntervalStatistics::getMinimalPatchBlockedDueToGlobalTimestep() const  {
   return _persistentRecords._minimalPatchBlockedDueToGlobalTimestep;
}



 void peanoclaw::statistics::TimeIntervalStatistics::setMinimalPatchBlockedDueToGlobalTimestep(const bool& minimalPatchBlockedDueToGlobalTimestep)  {
   _persistentRecords._minimalPatchBlockedDueToGlobalTimestep = minimalPatchBlockedDueToGlobalTimestep;
}




std::string peanoclaw::statistics::TimeIntervalStatistics::toString() const {
   std::ostringstream stringstr;
   toString(stringstr);
   return stringstr.str();
}

void peanoclaw::statistics::TimeIntervalStatistics::toString (std::ostream& out) const {
   out << "("; 
   out << "minimalPatchIndex:" << getMinimalPatchIndex();
   out << ",";
   out << "minimalPatchParentIndex:" << getMinimalPatchParentIndex();
   out << ",";
   out << "minimalPatchTime:" << getMinimalPatchTime();
   out << ",";
   out << "startMaximumLocalTimeInterval:" << getStartMaximumLocalTimeInterval();
   out << ",";
   out << "endMaximumLocalTimeInterval:" << getEndMaximumLocalTimeInterval();
   out << ",";
   out << "startMinimumLocalTimeInterval:" << getStartMinimumLocalTimeInterval();
   out << ",";
   out << "endMinimumLocalTimeInterval:" << getEndMinimumLocalTimeInterval();
   out << ",";
   out << "minimalTimestep:" << getMinimalTimestep();
   out << ",";
   out << "allPatchesEvolvedToGlobalTimestep:" << getAllPatchesEvolvedToGlobalTimestep();
   out << ",";
   out << "averageGlobalTimeInterval:" << getAverageGlobalTimeInterval();
   out << ",";
   out << "globalTimestepEndTime:" << getGlobalTimestepEndTime();
   out << ",";
   out << "minimalPatchBlockedDueToCoarsening:" << getMinimalPatchBlockedDueToCoarsening();
   out << ",";
   out << "minimalPatchBlockedDueToGlobalTimestep:" << getMinimalPatchBlockedDueToGlobalTimestep();
   out <<  ")";
}


peanoclaw::statistics::TimeIntervalStatistics::PersistentRecords peanoclaw::statistics::TimeIntervalStatistics::getPersistentRecords() const {
   return _persistentRecords;
}

peanoclaw::statistics::TimeIntervalStatisticsPacked peanoclaw::statistics::TimeIntervalStatistics::convert() const{
   return TimeIntervalStatisticsPacked(
      getMinimalPatchIndex(),
      getMinimalPatchParentIndex(),
      getMinimalPatchTime(),
      getStartMaximumLocalTimeInterval(),
      getEndMaximumLocalTimeInterval(),
      getStartMinimumLocalTimeInterval(),
      getEndMinimumLocalTimeInterval(),
      getMinimalTimestep(),
      getAllPatchesEvolvedToGlobalTimestep(),
      getAverageGlobalTimeInterval(),
      getGlobalTimestepEndTime(),
      getMinimalPatchBlockedDueToCoarsening(),
      getMinimalPatchBlockedDueToGlobalTimestep()
   );
}

#ifdef Parallel
   tarch::logging::Log peanoclaw::statistics::TimeIntervalStatistics::_log( "peanoclaw::statistics::TimeIntervalStatistics" );
   
   MPI_Datatype peanoclaw::statistics::TimeIntervalStatistics::Datatype = 0;
   MPI_Datatype peanoclaw::statistics::TimeIntervalStatistics::FullDatatype = 0;
   
   
   void peanoclaw::statistics::TimeIntervalStatistics::initDatatype() {
      {
         TimeIntervalStatistics dummyTimeIntervalStatistics[2];
         
         const int Attributes = 14;
         MPI_Datatype subtypes[Attributes] = {
            MPI_INT,		 //minimalPatchIndex
            MPI_INT,		 //minimalPatchParentIndex
            MPI_DOUBLE,		 //minimalPatchTime
            MPI_DOUBLE,		 //startMaximumLocalTimeInterval
            MPI_DOUBLE,		 //endMaximumLocalTimeInterval
            MPI_DOUBLE,		 //startMinimumLocalTimeInterval
            MPI_DOUBLE,		 //endMinimumLocalTimeInterval
            MPI_DOUBLE,		 //minimalTimestep
            MPI_CHAR,		 //allPatchesEvolvedToGlobalTimestep
            MPI_DOUBLE,		 //averageGlobalTimeInterval
            MPI_DOUBLE,		 //globalTimestepEndTime
            MPI_CHAR,		 //minimalPatchBlockedDueToCoarsening
            MPI_CHAR,		 //minimalPatchBlockedDueToGlobalTimestep
            MPI_UB		 // end/displacement flag
         };
         
         int blocklen[Attributes] = {
            1,		 //minimalPatchIndex
            1,		 //minimalPatchParentIndex
            1,		 //minimalPatchTime
            1,		 //startMaximumLocalTimeInterval
            1,		 //endMaximumLocalTimeInterval
            1,		 //startMinimumLocalTimeInterval
            1,		 //endMinimumLocalTimeInterval
            1,		 //minimalTimestep
            1,		 //allPatchesEvolvedToGlobalTimestep
            1,		 //averageGlobalTimeInterval
            1,		 //globalTimestepEndTime
            1,		 //minimalPatchBlockedDueToCoarsening
            1,		 //minimalPatchBlockedDueToGlobalTimestep
            1		 // end/displacement flag
         };
         
         MPI_Aint     disp[Attributes];
         
         MPI_Aint base;
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]))), &base);
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._minimalPatchIndex))), 		&disp[0] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._minimalPatchParentIndex))), 		&disp[1] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._minimalPatchTime))), 		&disp[2] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._startMaximumLocalTimeInterval))), 		&disp[3] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._endMaximumLocalTimeInterval))), 		&disp[4] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._startMinimumLocalTimeInterval))), 		&disp[5] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._endMinimumLocalTimeInterval))), 		&disp[6] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._minimalTimestep))), 		&disp[7] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._allPatchesEvolvedToGlobalTimestep))), 		&disp[8] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._averageGlobalTimeInterval))), 		&disp[9] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._globalTimestepEndTime))), 		&disp[10] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._minimalPatchBlockedDueToCoarsening))), 		&disp[11] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._minimalPatchBlockedDueToGlobalTimestep))), 		&disp[12] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[1]._persistentRecords._minimalPatchIndex))), 		&disp[13] );
         
         for (int i=1; i<Attributes; i++) {
            assertion1( disp[i] > disp[i-1], i );
         }
         for (int i=0; i<Attributes; i++) {
            disp[i] -= base;
         }
         MPI_Type_struct( Attributes, blocklen, disp, subtypes, &TimeIntervalStatistics::Datatype );
         MPI_Type_commit( &TimeIntervalStatistics::Datatype );
         
      }
      {
         TimeIntervalStatistics dummyTimeIntervalStatistics[2];
         
         const int Attributes = 14;
         MPI_Datatype subtypes[Attributes] = {
            MPI_INT,		 //minimalPatchIndex
            MPI_INT,		 //minimalPatchParentIndex
            MPI_DOUBLE,		 //minimalPatchTime
            MPI_DOUBLE,		 //startMaximumLocalTimeInterval
            MPI_DOUBLE,		 //endMaximumLocalTimeInterval
            MPI_DOUBLE,		 //startMinimumLocalTimeInterval
            MPI_DOUBLE,		 //endMinimumLocalTimeInterval
            MPI_DOUBLE,		 //minimalTimestep
            MPI_CHAR,		 //allPatchesEvolvedToGlobalTimestep
            MPI_DOUBLE,		 //averageGlobalTimeInterval
            MPI_DOUBLE,		 //globalTimestepEndTime
            MPI_CHAR,		 //minimalPatchBlockedDueToCoarsening
            MPI_CHAR,		 //minimalPatchBlockedDueToGlobalTimestep
            MPI_UB		 // end/displacement flag
         };
         
         int blocklen[Attributes] = {
            1,		 //minimalPatchIndex
            1,		 //minimalPatchParentIndex
            1,		 //minimalPatchTime
            1,		 //startMaximumLocalTimeInterval
            1,		 //endMaximumLocalTimeInterval
            1,		 //startMinimumLocalTimeInterval
            1,		 //endMinimumLocalTimeInterval
            1,		 //minimalTimestep
            1,		 //allPatchesEvolvedToGlobalTimestep
            1,		 //averageGlobalTimeInterval
            1,		 //globalTimestepEndTime
            1,		 //minimalPatchBlockedDueToCoarsening
            1,		 //minimalPatchBlockedDueToGlobalTimestep
            1		 // end/displacement flag
         };
         
         MPI_Aint     disp[Attributes];
         
         MPI_Aint base;
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]))), &base);
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._minimalPatchIndex))), 		&disp[0] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._minimalPatchParentIndex))), 		&disp[1] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._minimalPatchTime))), 		&disp[2] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._startMaximumLocalTimeInterval))), 		&disp[3] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._endMaximumLocalTimeInterval))), 		&disp[4] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._startMinimumLocalTimeInterval))), 		&disp[5] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._endMinimumLocalTimeInterval))), 		&disp[6] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._minimalTimestep))), 		&disp[7] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._allPatchesEvolvedToGlobalTimestep))), 		&disp[8] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._averageGlobalTimeInterval))), 		&disp[9] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._globalTimestepEndTime))), 		&disp[10] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._minimalPatchBlockedDueToCoarsening))), 		&disp[11] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[0]._persistentRecords._minimalPatchBlockedDueToGlobalTimestep))), 		&disp[12] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatistics[1]._persistentRecords._minimalPatchIndex))), 		&disp[13] );
         
         for (int i=1; i<Attributes; i++) {
            assertion1( disp[i] > disp[i-1], i );
         }
         for (int i=0; i<Attributes; i++) {
            disp[i] -= base;
         }
         MPI_Type_struct( Attributes, blocklen, disp, subtypes, &TimeIntervalStatistics::FullDatatype );
         MPI_Type_commit( &TimeIntervalStatistics::FullDatatype );
         
      }
      
   }
   
   
   void peanoclaw::statistics::TimeIntervalStatistics::shutdownDatatype() {
      MPI_Type_free( &TimeIntervalStatistics::Datatype );
      MPI_Type_free( &TimeIntervalStatistics::FullDatatype );
      
   }
   
   void peanoclaw::statistics::TimeIntervalStatistics::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
      if (communicateBlocking) {
      
         const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
         if  (result!=MPI_SUCCESS) {
            std::ostringstream msg;
            msg << "was not able to send message peanoclaw::statistics::TimeIntervalStatistics "
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
            msg << "was not able to send message peanoclaw::statistics::TimeIntervalStatistics "
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
               msg << "testing for finished send task for peanoclaw::statistics::TimeIntervalStatistics "
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
               "peanoclaw::statistics::TimeIntervalStatistics",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::statistics::TimeIntervalStatistics",
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
   
   
   
   void peanoclaw::statistics::TimeIntervalStatistics::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
      if (communicateBlocking) {
      
         MPI_Status  status;
         const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
         if ( result != MPI_SUCCESS ) {
            std::ostringstream msg;
            msg << "failed to start to receive peanoclaw::statistics::TimeIntervalStatistics from node "
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
            msg << "failed to start to receive peanoclaw::statistics::TimeIntervalStatistics from node "
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
               msg << "testing for finished receive task for peanoclaw::statistics::TimeIntervalStatistics failed: "
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
               "peanoclaw::statistics::TimeIntervalStatistics",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::statistics::TimeIntervalStatistics",
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
   
   
   
   bool peanoclaw::statistics::TimeIntervalStatistics::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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


peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::PersistentRecords() {
   assertion((3 < (8 * sizeof(short int))));
   
}


peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::PersistentRecords(const int& minimalPatchIndex, const int& minimalPatchParentIndex, const double& minimalPatchTime, const double& startMaximumLocalTimeInterval, const double& endMaximumLocalTimeInterval, const double& startMinimumLocalTimeInterval, const double& endMinimumLocalTimeInterval, const double& minimalTimestep, const bool& allPatchesEvolvedToGlobalTimestep, const double& averageGlobalTimeInterval, const double& globalTimestepEndTime, const bool& minimalPatchBlockedDueToCoarsening, const bool& minimalPatchBlockedDueToGlobalTimestep):
_minimalPatchIndex(minimalPatchIndex),
_minimalPatchParentIndex(minimalPatchParentIndex),
_minimalPatchTime(minimalPatchTime),
_startMaximumLocalTimeInterval(startMaximumLocalTimeInterval),
_endMaximumLocalTimeInterval(endMaximumLocalTimeInterval),
_startMinimumLocalTimeInterval(startMinimumLocalTimeInterval),
_endMinimumLocalTimeInterval(endMinimumLocalTimeInterval),
_minimalTimestep(minimalTimestep),
_averageGlobalTimeInterval(averageGlobalTimeInterval),
_globalTimestepEndTime(globalTimestepEndTime) {
   setAllPatchesEvolvedToGlobalTimestep(allPatchesEvolvedToGlobalTimestep);
   setMinimalPatchBlockedDueToCoarsening(minimalPatchBlockedDueToCoarsening);
   setMinimalPatchBlockedDueToGlobalTimestep(minimalPatchBlockedDueToGlobalTimestep);
   assertion((3 < (8 * sizeof(short int))));
   
}


 int peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::getMinimalPatchIndex() const  {
   return _minimalPatchIndex;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::setMinimalPatchIndex(const int& minimalPatchIndex)  {
   _minimalPatchIndex = minimalPatchIndex;
}



 int peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::getMinimalPatchParentIndex() const  {
   return _minimalPatchParentIndex;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::setMinimalPatchParentIndex(const int& minimalPatchParentIndex)  {
   _minimalPatchParentIndex = minimalPatchParentIndex;
}



 double peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::getMinimalPatchTime() const  {
   return _minimalPatchTime;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::setMinimalPatchTime(const double& minimalPatchTime)  {
   _minimalPatchTime = minimalPatchTime;
}



 double peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::getStartMaximumLocalTimeInterval() const  {
   return _startMaximumLocalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::setStartMaximumLocalTimeInterval(const double& startMaximumLocalTimeInterval)  {
   _startMaximumLocalTimeInterval = startMaximumLocalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::getEndMaximumLocalTimeInterval() const  {
   return _endMaximumLocalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::setEndMaximumLocalTimeInterval(const double& endMaximumLocalTimeInterval)  {
   _endMaximumLocalTimeInterval = endMaximumLocalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::getStartMinimumLocalTimeInterval() const  {
   return _startMinimumLocalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::setStartMinimumLocalTimeInterval(const double& startMinimumLocalTimeInterval)  {
   _startMinimumLocalTimeInterval = startMinimumLocalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::getEndMinimumLocalTimeInterval() const  {
   return _endMinimumLocalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::setEndMinimumLocalTimeInterval(const double& endMinimumLocalTimeInterval)  {
   _endMinimumLocalTimeInterval = endMinimumLocalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::getMinimalTimestep() const  {
   return _minimalTimestep;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::setMinimalTimestep(const double& minimalTimestep)  {
   _minimalTimestep = minimalTimestep;
}



 bool peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::getAllPatchesEvolvedToGlobalTimestep() const  {
   short int mask = 1 << (0);
   short int tmp = static_cast<short int>(_packedRecords0 & mask);
   return (tmp != 0);
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::setAllPatchesEvolvedToGlobalTimestep(const bool& allPatchesEvolvedToGlobalTimestep)  {
   short int mask = 1 << (0);
   _packedRecords0 = static_cast<short int>( allPatchesEvolvedToGlobalTimestep ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



 double peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::getAverageGlobalTimeInterval() const  {
   return _averageGlobalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::setAverageGlobalTimeInterval(const double& averageGlobalTimeInterval)  {
   _averageGlobalTimeInterval = averageGlobalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::getGlobalTimestepEndTime() const  {
   return _globalTimestepEndTime;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::setGlobalTimestepEndTime(const double& globalTimestepEndTime)  {
   _globalTimestepEndTime = globalTimestepEndTime;
}



 bool peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::getMinimalPatchBlockedDueToCoarsening() const  {
   short int mask = 1 << (1);
   short int tmp = static_cast<short int>(_packedRecords0 & mask);
   return (tmp != 0);
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::setMinimalPatchBlockedDueToCoarsening(const bool& minimalPatchBlockedDueToCoarsening)  {
   short int mask = 1 << (1);
   _packedRecords0 = static_cast<short int>( minimalPatchBlockedDueToCoarsening ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



 bool peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::getMinimalPatchBlockedDueToGlobalTimestep() const  {
   short int mask = 1 << (2);
   short int tmp = static_cast<short int>(_packedRecords0 & mask);
   return (tmp != 0);
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords::setMinimalPatchBlockedDueToGlobalTimestep(const bool& minimalPatchBlockedDueToGlobalTimestep)  {
   short int mask = 1 << (2);
   _packedRecords0 = static_cast<short int>( minimalPatchBlockedDueToGlobalTimestep ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}


peanoclaw::statistics::TimeIntervalStatisticsPacked::TimeIntervalStatisticsPacked() {
   assertion((3 < (8 * sizeof(short int))));
   
}


peanoclaw::statistics::TimeIntervalStatisticsPacked::TimeIntervalStatisticsPacked(const PersistentRecords& persistentRecords):
_persistentRecords(persistentRecords._minimalPatchIndex, persistentRecords._minimalPatchParentIndex, persistentRecords._minimalPatchTime, persistentRecords._startMaximumLocalTimeInterval, persistentRecords._endMaximumLocalTimeInterval, persistentRecords._startMinimumLocalTimeInterval, persistentRecords._endMinimumLocalTimeInterval, persistentRecords._minimalTimestep, persistentRecords.getAllPatchesEvolvedToGlobalTimestep(), persistentRecords._averageGlobalTimeInterval, persistentRecords._globalTimestepEndTime, persistentRecords.getMinimalPatchBlockedDueToCoarsening(), persistentRecords.getMinimalPatchBlockedDueToGlobalTimestep()) {
   assertion((3 < (8 * sizeof(short int))));
   
}


peanoclaw::statistics::TimeIntervalStatisticsPacked::TimeIntervalStatisticsPacked(const int& minimalPatchIndex, const int& minimalPatchParentIndex, const double& minimalPatchTime, const double& startMaximumLocalTimeInterval, const double& endMaximumLocalTimeInterval, const double& startMinimumLocalTimeInterval, const double& endMinimumLocalTimeInterval, const double& minimalTimestep, const bool& allPatchesEvolvedToGlobalTimestep, const double& averageGlobalTimeInterval, const double& globalTimestepEndTime, const bool& minimalPatchBlockedDueToCoarsening, const bool& minimalPatchBlockedDueToGlobalTimestep):
_persistentRecords(minimalPatchIndex, minimalPatchParentIndex, minimalPatchTime, startMaximumLocalTimeInterval, endMaximumLocalTimeInterval, startMinimumLocalTimeInterval, endMinimumLocalTimeInterval, minimalTimestep, allPatchesEvolvedToGlobalTimestep, averageGlobalTimeInterval, globalTimestepEndTime, minimalPatchBlockedDueToCoarsening, minimalPatchBlockedDueToGlobalTimestep) {
   assertion((3 < (8 * sizeof(short int))));
   
}


peanoclaw::statistics::TimeIntervalStatisticsPacked::~TimeIntervalStatisticsPacked() { }


 int peanoclaw::statistics::TimeIntervalStatisticsPacked::getMinimalPatchIndex() const  {
   return _persistentRecords._minimalPatchIndex;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::setMinimalPatchIndex(const int& minimalPatchIndex)  {
   _persistentRecords._minimalPatchIndex = minimalPatchIndex;
}



 int peanoclaw::statistics::TimeIntervalStatisticsPacked::getMinimalPatchParentIndex() const  {
   return _persistentRecords._minimalPatchParentIndex;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::setMinimalPatchParentIndex(const int& minimalPatchParentIndex)  {
   _persistentRecords._minimalPatchParentIndex = minimalPatchParentIndex;
}



 double peanoclaw::statistics::TimeIntervalStatisticsPacked::getMinimalPatchTime() const  {
   return _persistentRecords._minimalPatchTime;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::setMinimalPatchTime(const double& minimalPatchTime)  {
   _persistentRecords._minimalPatchTime = minimalPatchTime;
}



 double peanoclaw::statistics::TimeIntervalStatisticsPacked::getStartMaximumLocalTimeInterval() const  {
   return _persistentRecords._startMaximumLocalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::setStartMaximumLocalTimeInterval(const double& startMaximumLocalTimeInterval)  {
   _persistentRecords._startMaximumLocalTimeInterval = startMaximumLocalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatisticsPacked::getEndMaximumLocalTimeInterval() const  {
   return _persistentRecords._endMaximumLocalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::setEndMaximumLocalTimeInterval(const double& endMaximumLocalTimeInterval)  {
   _persistentRecords._endMaximumLocalTimeInterval = endMaximumLocalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatisticsPacked::getStartMinimumLocalTimeInterval() const  {
   return _persistentRecords._startMinimumLocalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::setStartMinimumLocalTimeInterval(const double& startMinimumLocalTimeInterval)  {
   _persistentRecords._startMinimumLocalTimeInterval = startMinimumLocalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatisticsPacked::getEndMinimumLocalTimeInterval() const  {
   return _persistentRecords._endMinimumLocalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::setEndMinimumLocalTimeInterval(const double& endMinimumLocalTimeInterval)  {
   _persistentRecords._endMinimumLocalTimeInterval = endMinimumLocalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatisticsPacked::getMinimalTimestep() const  {
   return _persistentRecords._minimalTimestep;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::setMinimalTimestep(const double& minimalTimestep)  {
   _persistentRecords._minimalTimestep = minimalTimestep;
}



 bool peanoclaw::statistics::TimeIntervalStatisticsPacked::getAllPatchesEvolvedToGlobalTimestep() const  {
   short int mask = 1 << (0);
   short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
   return (tmp != 0);
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::setAllPatchesEvolvedToGlobalTimestep(const bool& allPatchesEvolvedToGlobalTimestep)  {
   short int mask = 1 << (0);
   _persistentRecords._packedRecords0 = static_cast<short int>( allPatchesEvolvedToGlobalTimestep ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



 double peanoclaw::statistics::TimeIntervalStatisticsPacked::getAverageGlobalTimeInterval() const  {
   return _persistentRecords._averageGlobalTimeInterval;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::setAverageGlobalTimeInterval(const double& averageGlobalTimeInterval)  {
   _persistentRecords._averageGlobalTimeInterval = averageGlobalTimeInterval;
}



 double peanoclaw::statistics::TimeIntervalStatisticsPacked::getGlobalTimestepEndTime() const  {
   return _persistentRecords._globalTimestepEndTime;
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::setGlobalTimestepEndTime(const double& globalTimestepEndTime)  {
   _persistentRecords._globalTimestepEndTime = globalTimestepEndTime;
}



 bool peanoclaw::statistics::TimeIntervalStatisticsPacked::getMinimalPatchBlockedDueToCoarsening() const  {
   short int mask = 1 << (1);
   short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
   return (tmp != 0);
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::setMinimalPatchBlockedDueToCoarsening(const bool& minimalPatchBlockedDueToCoarsening)  {
   short int mask = 1 << (1);
   _persistentRecords._packedRecords0 = static_cast<short int>( minimalPatchBlockedDueToCoarsening ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



 bool peanoclaw::statistics::TimeIntervalStatisticsPacked::getMinimalPatchBlockedDueToGlobalTimestep() const  {
   short int mask = 1 << (2);
   short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
   return (tmp != 0);
}



 void peanoclaw::statistics::TimeIntervalStatisticsPacked::setMinimalPatchBlockedDueToGlobalTimestep(const bool& minimalPatchBlockedDueToGlobalTimestep)  {
   short int mask = 1 << (2);
   _persistentRecords._packedRecords0 = static_cast<short int>( minimalPatchBlockedDueToGlobalTimestep ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}




std::string peanoclaw::statistics::TimeIntervalStatisticsPacked::toString() const {
   std::ostringstream stringstr;
   toString(stringstr);
   return stringstr.str();
}

void peanoclaw::statistics::TimeIntervalStatisticsPacked::toString (std::ostream& out) const {
   out << "("; 
   out << "minimalPatchIndex:" << getMinimalPatchIndex();
   out << ",";
   out << "minimalPatchParentIndex:" << getMinimalPatchParentIndex();
   out << ",";
   out << "minimalPatchTime:" << getMinimalPatchTime();
   out << ",";
   out << "startMaximumLocalTimeInterval:" << getStartMaximumLocalTimeInterval();
   out << ",";
   out << "endMaximumLocalTimeInterval:" << getEndMaximumLocalTimeInterval();
   out << ",";
   out << "startMinimumLocalTimeInterval:" << getStartMinimumLocalTimeInterval();
   out << ",";
   out << "endMinimumLocalTimeInterval:" << getEndMinimumLocalTimeInterval();
   out << ",";
   out << "minimalTimestep:" << getMinimalTimestep();
   out << ",";
   out << "allPatchesEvolvedToGlobalTimestep:" << getAllPatchesEvolvedToGlobalTimestep();
   out << ",";
   out << "averageGlobalTimeInterval:" << getAverageGlobalTimeInterval();
   out << ",";
   out << "globalTimestepEndTime:" << getGlobalTimestepEndTime();
   out << ",";
   out << "minimalPatchBlockedDueToCoarsening:" << getMinimalPatchBlockedDueToCoarsening();
   out << ",";
   out << "minimalPatchBlockedDueToGlobalTimestep:" << getMinimalPatchBlockedDueToGlobalTimestep();
   out <<  ")";
}


peanoclaw::statistics::TimeIntervalStatisticsPacked::PersistentRecords peanoclaw::statistics::TimeIntervalStatisticsPacked::getPersistentRecords() const {
   return _persistentRecords;
}

peanoclaw::statistics::TimeIntervalStatistics peanoclaw::statistics::TimeIntervalStatisticsPacked::convert() const{
   return TimeIntervalStatistics(
      getMinimalPatchIndex(),
      getMinimalPatchParentIndex(),
      getMinimalPatchTime(),
      getStartMaximumLocalTimeInterval(),
      getEndMaximumLocalTimeInterval(),
      getStartMinimumLocalTimeInterval(),
      getEndMinimumLocalTimeInterval(),
      getMinimalTimestep(),
      getAllPatchesEvolvedToGlobalTimestep(),
      getAverageGlobalTimeInterval(),
      getGlobalTimestepEndTime(),
      getMinimalPatchBlockedDueToCoarsening(),
      getMinimalPatchBlockedDueToGlobalTimestep()
   );
}

#ifdef Parallel
   tarch::logging::Log peanoclaw::statistics::TimeIntervalStatisticsPacked::_log( "peanoclaw::statistics::TimeIntervalStatisticsPacked" );
   
   MPI_Datatype peanoclaw::statistics::TimeIntervalStatisticsPacked::Datatype = 0;
   MPI_Datatype peanoclaw::statistics::TimeIntervalStatisticsPacked::FullDatatype = 0;
   
   
   void peanoclaw::statistics::TimeIntervalStatisticsPacked::initDatatype() {
      {
         TimeIntervalStatisticsPacked dummyTimeIntervalStatisticsPacked[2];
         
         const int Attributes = 12;
         MPI_Datatype subtypes[Attributes] = {
            MPI_INT,		 //minimalPatchIndex
            MPI_INT,		 //minimalPatchParentIndex
            MPI_DOUBLE,		 //minimalPatchTime
            MPI_DOUBLE,		 //startMaximumLocalTimeInterval
            MPI_DOUBLE,		 //endMaximumLocalTimeInterval
            MPI_DOUBLE,		 //startMinimumLocalTimeInterval
            MPI_DOUBLE,		 //endMinimumLocalTimeInterval
            MPI_DOUBLE,		 //minimalTimestep
            MPI_DOUBLE,		 //averageGlobalTimeInterval
            MPI_DOUBLE,		 //globalTimestepEndTime
            MPI_SHORT,		 //_packedRecords0
            MPI_UB		 // end/displacement flag
         };
         
         int blocklen[Attributes] = {
            1,		 //minimalPatchIndex
            1,		 //minimalPatchParentIndex
            1,		 //minimalPatchTime
            1,		 //startMaximumLocalTimeInterval
            1,		 //endMaximumLocalTimeInterval
            1,		 //startMinimumLocalTimeInterval
            1,		 //endMinimumLocalTimeInterval
            1,		 //minimalTimestep
            1,		 //averageGlobalTimeInterval
            1,		 //globalTimestepEndTime
            1,		 //_packedRecords0
            1		 // end/displacement flag
         };
         
         MPI_Aint     disp[Attributes];
         
         MPI_Aint base;
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]))), &base);
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._minimalPatchIndex))), 		&disp[0] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._minimalPatchParentIndex))), 		&disp[1] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._minimalPatchTime))), 		&disp[2] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._startMaximumLocalTimeInterval))), 		&disp[3] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._endMaximumLocalTimeInterval))), 		&disp[4] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._startMinimumLocalTimeInterval))), 		&disp[5] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._endMinimumLocalTimeInterval))), 		&disp[6] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._minimalTimestep))), 		&disp[7] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._averageGlobalTimeInterval))), 		&disp[8] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._globalTimestepEndTime))), 		&disp[9] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._packedRecords0))), 		&disp[10] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[1]._persistentRecords._minimalPatchIndex))), 		&disp[11] );
         
         for (int i=1; i<Attributes; i++) {
            assertion1( disp[i] > disp[i-1], i );
         }
         for (int i=0; i<Attributes; i++) {
            disp[i] -= base;
         }
         MPI_Type_struct( Attributes, blocklen, disp, subtypes, &TimeIntervalStatisticsPacked::Datatype );
         MPI_Type_commit( &TimeIntervalStatisticsPacked::Datatype );
         
      }
      {
         TimeIntervalStatisticsPacked dummyTimeIntervalStatisticsPacked[2];
         
         const int Attributes = 12;
         MPI_Datatype subtypes[Attributes] = {
            MPI_INT,		 //minimalPatchIndex
            MPI_INT,		 //minimalPatchParentIndex
            MPI_DOUBLE,		 //minimalPatchTime
            MPI_DOUBLE,		 //startMaximumLocalTimeInterval
            MPI_DOUBLE,		 //endMaximumLocalTimeInterval
            MPI_DOUBLE,		 //startMinimumLocalTimeInterval
            MPI_DOUBLE,		 //endMinimumLocalTimeInterval
            MPI_DOUBLE,		 //minimalTimestep
            MPI_DOUBLE,		 //averageGlobalTimeInterval
            MPI_DOUBLE,		 //globalTimestepEndTime
            MPI_SHORT,		 //_packedRecords0
            MPI_UB		 // end/displacement flag
         };
         
         int blocklen[Attributes] = {
            1,		 //minimalPatchIndex
            1,		 //minimalPatchParentIndex
            1,		 //minimalPatchTime
            1,		 //startMaximumLocalTimeInterval
            1,		 //endMaximumLocalTimeInterval
            1,		 //startMinimumLocalTimeInterval
            1,		 //endMinimumLocalTimeInterval
            1,		 //minimalTimestep
            1,		 //averageGlobalTimeInterval
            1,		 //globalTimestepEndTime
            1,		 //_packedRecords0
            1		 // end/displacement flag
         };
         
         MPI_Aint     disp[Attributes];
         
         MPI_Aint base;
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]))), &base);
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._minimalPatchIndex))), 		&disp[0] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._minimalPatchParentIndex))), 		&disp[1] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._minimalPatchTime))), 		&disp[2] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._startMaximumLocalTimeInterval))), 		&disp[3] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._endMaximumLocalTimeInterval))), 		&disp[4] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._startMinimumLocalTimeInterval))), 		&disp[5] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._endMinimumLocalTimeInterval))), 		&disp[6] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._minimalTimestep))), 		&disp[7] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._averageGlobalTimeInterval))), 		&disp[8] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._globalTimestepEndTime))), 		&disp[9] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[0]._persistentRecords._packedRecords0))), 		&disp[10] );
         MPI_Address( const_cast<void*>(static_cast<const void*>(&(dummyTimeIntervalStatisticsPacked[1]._persistentRecords._minimalPatchIndex))), 		&disp[11] );
         
         for (int i=1; i<Attributes; i++) {
            assertion1( disp[i] > disp[i-1], i );
         }
         for (int i=0; i<Attributes; i++) {
            disp[i] -= base;
         }
         MPI_Type_struct( Attributes, blocklen, disp, subtypes, &TimeIntervalStatisticsPacked::FullDatatype );
         MPI_Type_commit( &TimeIntervalStatisticsPacked::FullDatatype );
         
      }
      
   }
   
   
   void peanoclaw::statistics::TimeIntervalStatisticsPacked::shutdownDatatype() {
      MPI_Type_free( &TimeIntervalStatisticsPacked::Datatype );
      MPI_Type_free( &TimeIntervalStatisticsPacked::FullDatatype );
      
   }
   
   void peanoclaw::statistics::TimeIntervalStatisticsPacked::send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
      if (communicateBlocking) {
      
         const int result = MPI_Send(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, destination, tag, tarch::parallel::Node::getInstance().getCommunicator());
         if  (result!=MPI_SUCCESS) {
            std::ostringstream msg;
            msg << "was not able to send message peanoclaw::statistics::TimeIntervalStatisticsPacked "
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
            msg << "was not able to send message peanoclaw::statistics::TimeIntervalStatisticsPacked "
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
               msg << "testing for finished send task for peanoclaw::statistics::TimeIntervalStatisticsPacked "
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
               "peanoclaw::statistics::TimeIntervalStatisticsPacked",
               "send(int)", destination,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::statistics::TimeIntervalStatisticsPacked",
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
   
   
   
   void peanoclaw::statistics::TimeIntervalStatisticsPacked::receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking) {
      if (communicateBlocking) {
      
         MPI_Status  status;
         const int   result = MPI_Recv(this, 1, exchangeOnlyAttributesMarkedWithParallelise ? Datatype : FullDatatype, source, tag, tarch::parallel::Node::getInstance().getCommunicator(), &status);
         if ( result != MPI_SUCCESS ) {
            std::ostringstream msg;
            msg << "failed to start to receive peanoclaw::statistics::TimeIntervalStatisticsPacked from node "
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
            msg << "failed to start to receive peanoclaw::statistics::TimeIntervalStatisticsPacked from node "
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
               msg << "testing for finished receive task for peanoclaw::statistics::TimeIntervalStatisticsPacked failed: "
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
               "peanoclaw::statistics::TimeIntervalStatisticsPacked",
               "receive(int)", source,tag,1
               );
               triggeredTimeoutWarning = true;
            }
            if (
               tarch::parallel::Node::getInstance().isTimeOutDeadlockEnabled() &&
               (clock()>timeOutShutdown)
            ) {
               tarch::parallel::Node::getInstance().triggerDeadlockTimeOut(
               "peanoclaw::statistics::TimeIntervalStatisticsPacked",
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
   
   
   
   bool peanoclaw::statistics::TimeIntervalStatisticsPacked::isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise) {
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



