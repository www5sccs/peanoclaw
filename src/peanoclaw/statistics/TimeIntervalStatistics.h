#ifndef _PEANOCLAW_STATISTICS_TIMEINTERVALSTATISTICS_H
#define _PEANOCLAW_STATISTICS_TIMEINTERVALSTATISTICS_H

#include "peano/utils/Globals.h"
#include "tarch/compiler/CompilerSpecificSettings.h"
#include "peano/utils/PeanoOptimisations.h"
#ifdef Parallel
	#include "tarch/parallel/Node.h"
#endif
#ifdef Parallel
	#include <mpi.h>
#endif
#include "tarch/logging/Log.h"
#include "tarch/la/Vector.h"
#include <bitset>
#include <complex>
#include <string>
#include <iostream>

namespace peanoclaw {
   namespace statistics {
      class TimeIntervalStatistics;
      class TimeIntervalStatisticsPacked;
   }
}

/**
 * @author This class is generated by DaStGen
 * 		   DataStructureGenerator (DaStGen)
 * 		   2007-2009 Wolfgang Eckhardt
 * 		   2012      Tobias Weinzierl
 *
 * 		   build date: 09-02-2014 14:40
 *
 * @date   12/06/2014 07:41
 */
class peanoclaw::statistics::TimeIntervalStatistics { 
   
   public:
      
      typedef peanoclaw::statistics::TimeIntervalStatisticsPacked Packed;
      
      struct PersistentRecords {
         int _minimalPatchIndex;
         int _minimalPatchParentIndex;
         double _minimalPatchTime;
         double _startMaximumLocalTimeInterval;
         double _endMaximumLocalTimeInterval;
         double _startMinimumLocalTimeInterval;
         double _endMinimumLocalTimeInterval;
         double _minimalTimestep;
         bool _allPatchesEvolvedToGlobalTimestep;
         double _averageGlobalTimeInterval;
         double _globalTimestepEndTime;
         bool _minimalPatchBlockedDueToCoarsening;
         bool _minimalPatchBlockedDueToGlobalTimestep;
         /**
          * Generated
          */
         PersistentRecords();
         
         /**
          * Generated
          */
         PersistentRecords(const int& minimalPatchIndex, const int& minimalPatchParentIndex, const double& minimalPatchTime, const double& startMaximumLocalTimeInterval, const double& endMaximumLocalTimeInterval, const double& startMinimumLocalTimeInterval, const double& endMinimumLocalTimeInterval, const double& minimalTimestep, const bool& allPatchesEvolvedToGlobalTimestep, const double& averageGlobalTimeInterval, const double& globalTimestepEndTime, const bool& minimalPatchBlockedDueToCoarsening, const bool& minimalPatchBlockedDueToGlobalTimestep);
         
         
         inline int getMinimalPatchIndex() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _minimalPatchIndex;
         }
         
         
         
         inline void setMinimalPatchIndex(const int& minimalPatchIndex) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _minimalPatchIndex = minimalPatchIndex;
         }
         
         
         
         inline int getMinimalPatchParentIndex() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _minimalPatchParentIndex;
         }
         
         
         
         inline void setMinimalPatchParentIndex(const int& minimalPatchParentIndex) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _minimalPatchParentIndex = minimalPatchParentIndex;
         }
         
         
         
         inline double getMinimalPatchTime() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _minimalPatchTime;
         }
         
         
         
         inline void setMinimalPatchTime(const double& minimalPatchTime) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _minimalPatchTime = minimalPatchTime;
         }
         
         
         
         inline double getStartMaximumLocalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _startMaximumLocalTimeInterval;
         }
         
         
         
         inline void setStartMaximumLocalTimeInterval(const double& startMaximumLocalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _startMaximumLocalTimeInterval = startMaximumLocalTimeInterval;
         }
         
         
         
         inline double getEndMaximumLocalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _endMaximumLocalTimeInterval;
         }
         
         
         
         inline void setEndMaximumLocalTimeInterval(const double& endMaximumLocalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _endMaximumLocalTimeInterval = endMaximumLocalTimeInterval;
         }
         
         
         
         inline double getStartMinimumLocalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _startMinimumLocalTimeInterval;
         }
         
         
         
         inline void setStartMinimumLocalTimeInterval(const double& startMinimumLocalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _startMinimumLocalTimeInterval = startMinimumLocalTimeInterval;
         }
         
         
         
         inline double getEndMinimumLocalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _endMinimumLocalTimeInterval;
         }
         
         
         
         inline void setEndMinimumLocalTimeInterval(const double& endMinimumLocalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _endMinimumLocalTimeInterval = endMinimumLocalTimeInterval;
         }
         
         
         
         inline double getMinimalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _minimalTimestep;
         }
         
         
         
         inline void setMinimalTimestep(const double& minimalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _minimalTimestep = minimalTimestep;
         }
         
         
         
         inline bool getAllPatchesEvolvedToGlobalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _allPatchesEvolvedToGlobalTimestep;
         }
         
         
         
         inline void setAllPatchesEvolvedToGlobalTimestep(const bool& allPatchesEvolvedToGlobalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _allPatchesEvolvedToGlobalTimestep = allPatchesEvolvedToGlobalTimestep;
         }
         
         
         
         inline double getAverageGlobalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _averageGlobalTimeInterval;
         }
         
         
         
         inline void setAverageGlobalTimeInterval(const double& averageGlobalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _averageGlobalTimeInterval = averageGlobalTimeInterval;
         }
         
         
         
         inline double getGlobalTimestepEndTime() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _globalTimestepEndTime;
         }
         
         
         
         inline void setGlobalTimestepEndTime(const double& globalTimestepEndTime) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _globalTimestepEndTime = globalTimestepEndTime;
         }
         
         
         
         inline bool getMinimalPatchBlockedDueToCoarsening() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _minimalPatchBlockedDueToCoarsening;
         }
         
         
         
         inline void setMinimalPatchBlockedDueToCoarsening(const bool& minimalPatchBlockedDueToCoarsening) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _minimalPatchBlockedDueToCoarsening = minimalPatchBlockedDueToCoarsening;
         }
         
         
         
         inline bool getMinimalPatchBlockedDueToGlobalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _minimalPatchBlockedDueToGlobalTimestep;
         }
         
         
         
         inline void setMinimalPatchBlockedDueToGlobalTimestep(const bool& minimalPatchBlockedDueToGlobalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _minimalPatchBlockedDueToGlobalTimestep = minimalPatchBlockedDueToGlobalTimestep;
         }
         
         
         
      };
      
   private: 
      PersistentRecords _persistentRecords;
      
   public:
      /**
       * Generated
       */
      TimeIntervalStatistics();
      
      /**
       * Generated
       */
      TimeIntervalStatistics(const PersistentRecords& persistentRecords);
      
      /**
       * Generated
       */
      TimeIntervalStatistics(const int& minimalPatchIndex, const int& minimalPatchParentIndex, const double& minimalPatchTime, const double& startMaximumLocalTimeInterval, const double& endMaximumLocalTimeInterval, const double& startMinimumLocalTimeInterval, const double& endMinimumLocalTimeInterval, const double& minimalTimestep, const bool& allPatchesEvolvedToGlobalTimestep, const double& averageGlobalTimeInterval, const double& globalTimestepEndTime, const bool& minimalPatchBlockedDueToCoarsening, const bool& minimalPatchBlockedDueToGlobalTimestep);
      
      /**
       * Generated
       */
      ~TimeIntervalStatistics();
      
      
      inline int getMinimalPatchIndex() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._minimalPatchIndex;
      }
      
      
      
      inline void setMinimalPatchIndex(const int& minimalPatchIndex) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._minimalPatchIndex = minimalPatchIndex;
      }
      
      
      
      inline int getMinimalPatchParentIndex() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._minimalPatchParentIndex;
      }
      
      
      
      inline void setMinimalPatchParentIndex(const int& minimalPatchParentIndex) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._minimalPatchParentIndex = minimalPatchParentIndex;
      }
      
      
      
      inline double getMinimalPatchTime() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._minimalPatchTime;
      }
      
      
      
      inline void setMinimalPatchTime(const double& minimalPatchTime) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._minimalPatchTime = minimalPatchTime;
      }
      
      
      
      inline double getStartMaximumLocalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._startMaximumLocalTimeInterval;
      }
      
      
      
      inline void setStartMaximumLocalTimeInterval(const double& startMaximumLocalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._startMaximumLocalTimeInterval = startMaximumLocalTimeInterval;
      }
      
      
      
      inline double getEndMaximumLocalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._endMaximumLocalTimeInterval;
      }
      
      
      
      inline void setEndMaximumLocalTimeInterval(const double& endMaximumLocalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._endMaximumLocalTimeInterval = endMaximumLocalTimeInterval;
      }
      
      
      
      inline double getStartMinimumLocalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._startMinimumLocalTimeInterval;
      }
      
      
      
      inline void setStartMinimumLocalTimeInterval(const double& startMinimumLocalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._startMinimumLocalTimeInterval = startMinimumLocalTimeInterval;
      }
      
      
      
      inline double getEndMinimumLocalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._endMinimumLocalTimeInterval;
      }
      
      
      
      inline void setEndMinimumLocalTimeInterval(const double& endMinimumLocalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._endMinimumLocalTimeInterval = endMinimumLocalTimeInterval;
      }
      
      
      
      inline double getMinimalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._minimalTimestep;
      }
      
      
      
      inline void setMinimalTimestep(const double& minimalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._minimalTimestep = minimalTimestep;
      }
      
      
      
      inline bool getAllPatchesEvolvedToGlobalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._allPatchesEvolvedToGlobalTimestep;
      }
      
      
      
      inline void setAllPatchesEvolvedToGlobalTimestep(const bool& allPatchesEvolvedToGlobalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._allPatchesEvolvedToGlobalTimestep = allPatchesEvolvedToGlobalTimestep;
      }
      
      
      
      inline double getAverageGlobalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._averageGlobalTimeInterval;
      }
      
      
      
      inline void setAverageGlobalTimeInterval(const double& averageGlobalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._averageGlobalTimeInterval = averageGlobalTimeInterval;
      }
      
      
      
      inline double getGlobalTimestepEndTime() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._globalTimestepEndTime;
      }
      
      
      
      inline void setGlobalTimestepEndTime(const double& globalTimestepEndTime) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._globalTimestepEndTime = globalTimestepEndTime;
      }
      
      
      
      inline bool getMinimalPatchBlockedDueToCoarsening() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._minimalPatchBlockedDueToCoarsening;
      }
      
      
      
      inline void setMinimalPatchBlockedDueToCoarsening(const bool& minimalPatchBlockedDueToCoarsening) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._minimalPatchBlockedDueToCoarsening = minimalPatchBlockedDueToCoarsening;
      }
      
      
      
      inline bool getMinimalPatchBlockedDueToGlobalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._minimalPatchBlockedDueToGlobalTimestep;
      }
      
      
      
      inline void setMinimalPatchBlockedDueToGlobalTimestep(const bool& minimalPatchBlockedDueToGlobalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._minimalPatchBlockedDueToGlobalTimestep = minimalPatchBlockedDueToGlobalTimestep;
      }
      
      
      /**
       * Generated
       */
      std::string toString() const;
      
      /**
       * Generated
       */
      void toString(std::ostream& out) const;
      
      
      PersistentRecords getPersistentRecords() const;
      /**
       * Generated
       */
      TimeIntervalStatisticsPacked convert() const;
      
      
   #ifdef Parallel
      protected:
         static tarch::logging::Log _log;
         
      public:
         
         /**
          * Global that represents the mpi datatype.
          * There are two variants: Datatype identifies only those attributes marked with
          * parallelise. FullDatatype instead identifies the whole record with all fields.
          */
         static MPI_Datatype Datatype;
         static MPI_Datatype FullDatatype;
         
         /**
          * Initializes the data type for the mpi operations. Has to be called
          * before the very first send or receive operation is called.
          */
         static void initDatatype();
         
         static void shutdownDatatype();
         
         void send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking);
         
         void receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking);
         
         static bool isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise);
         
         #endif
            
         };
         
         /**
          * @author This class is generated by DaStGen
          * 		   DataStructureGenerator (DaStGen)
          * 		   2007-2009 Wolfgang Eckhardt
          * 		   2012      Tobias Weinzierl
          *
          * 		   build date: 09-02-2014 14:40
          *
          * @date   12/06/2014 07:41
          */
         class peanoclaw::statistics::TimeIntervalStatisticsPacked { 
            
            public:
               
               struct PersistentRecords {
                  int _minimalPatchIndex;
                  int _minimalPatchParentIndex;
                  double _minimalPatchTime;
                  double _startMaximumLocalTimeInterval;
                  double _endMaximumLocalTimeInterval;
                  double _startMinimumLocalTimeInterval;
                  double _endMinimumLocalTimeInterval;
                  double _minimalTimestep;
                  double _averageGlobalTimeInterval;
                  double _globalTimestepEndTime;
                  
                  /** mapping of records:
                  || Member 	|| startbit 	|| length
                   |  allPatchesEvolvedToGlobalTimestep	| startbit 0	| #bits 1
                   |  minimalPatchBlockedDueToCoarsening	| startbit 1	| #bits 1
                   |  minimalPatchBlockedDueToGlobalTimestep	| startbit 2	| #bits 1
                   */
                  short int _packedRecords0;
                  
                  /**
                   * Generated
                   */
                  PersistentRecords();
                  
                  /**
                   * Generated
                   */
                  PersistentRecords(const int& minimalPatchIndex, const int& minimalPatchParentIndex, const double& minimalPatchTime, const double& startMaximumLocalTimeInterval, const double& endMaximumLocalTimeInterval, const double& startMinimumLocalTimeInterval, const double& endMinimumLocalTimeInterval, const double& minimalTimestep, const bool& allPatchesEvolvedToGlobalTimestep, const double& averageGlobalTimeInterval, const double& globalTimestepEndTime, const bool& minimalPatchBlockedDueToCoarsening, const bool& minimalPatchBlockedDueToGlobalTimestep);
                  
                  
                  inline int getMinimalPatchIndex() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _minimalPatchIndex;
                  }
                  
                  
                  
                  inline void setMinimalPatchIndex(const int& minimalPatchIndex) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _minimalPatchIndex = minimalPatchIndex;
                  }
                  
                  
                  
                  inline int getMinimalPatchParentIndex() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _minimalPatchParentIndex;
                  }
                  
                  
                  
                  inline void setMinimalPatchParentIndex(const int& minimalPatchParentIndex) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _minimalPatchParentIndex = minimalPatchParentIndex;
                  }
                  
                  
                  
                  inline double getMinimalPatchTime() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _minimalPatchTime;
                  }
                  
                  
                  
                  inline void setMinimalPatchTime(const double& minimalPatchTime) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _minimalPatchTime = minimalPatchTime;
                  }
                  
                  
                  
                  inline double getStartMaximumLocalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _startMaximumLocalTimeInterval;
                  }
                  
                  
                  
                  inline void setStartMaximumLocalTimeInterval(const double& startMaximumLocalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _startMaximumLocalTimeInterval = startMaximumLocalTimeInterval;
                  }
                  
                  
                  
                  inline double getEndMaximumLocalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _endMaximumLocalTimeInterval;
                  }
                  
                  
                  
                  inline void setEndMaximumLocalTimeInterval(const double& endMaximumLocalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _endMaximumLocalTimeInterval = endMaximumLocalTimeInterval;
                  }
                  
                  
                  
                  inline double getStartMinimumLocalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _startMinimumLocalTimeInterval;
                  }
                  
                  
                  
                  inline void setStartMinimumLocalTimeInterval(const double& startMinimumLocalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _startMinimumLocalTimeInterval = startMinimumLocalTimeInterval;
                  }
                  
                  
                  
                  inline double getEndMinimumLocalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _endMinimumLocalTimeInterval;
                  }
                  
                  
                  
                  inline void setEndMinimumLocalTimeInterval(const double& endMinimumLocalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _endMinimumLocalTimeInterval = endMinimumLocalTimeInterval;
                  }
                  
                  
                  
                  inline double getMinimalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _minimalTimestep;
                  }
                  
                  
                  
                  inline void setMinimalTimestep(const double& minimalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _minimalTimestep = minimalTimestep;
                  }
                  
                  
                  
                  inline bool getAllPatchesEvolvedToGlobalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     short int mask = 1 << (0);
   short int tmp = static_cast<short int>(_packedRecords0 & mask);
   return (tmp != 0);
                  }
                  
                  
                  
                  inline void setAllPatchesEvolvedToGlobalTimestep(const bool& allPatchesEvolvedToGlobalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     short int mask = 1 << (0);
   _packedRecords0 = static_cast<short int>( allPatchesEvolvedToGlobalTimestep ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
                  }
                  
                  
                  
                  inline double getAverageGlobalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _averageGlobalTimeInterval;
                  }
                  
                  
                  
                  inline void setAverageGlobalTimeInterval(const double& averageGlobalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _averageGlobalTimeInterval = averageGlobalTimeInterval;
                  }
                  
                  
                  
                  inline double getGlobalTimestepEndTime() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _globalTimestepEndTime;
                  }
                  
                  
                  
                  inline void setGlobalTimestepEndTime(const double& globalTimestepEndTime) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _globalTimestepEndTime = globalTimestepEndTime;
                  }
                  
                  
                  
                  inline bool getMinimalPatchBlockedDueToCoarsening() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     short int mask = 1 << (1);
   short int tmp = static_cast<short int>(_packedRecords0 & mask);
   return (tmp != 0);
                  }
                  
                  
                  
                  inline void setMinimalPatchBlockedDueToCoarsening(const bool& minimalPatchBlockedDueToCoarsening) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     short int mask = 1 << (1);
   _packedRecords0 = static_cast<short int>( minimalPatchBlockedDueToCoarsening ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
                  }
                  
                  
                  
                  inline bool getMinimalPatchBlockedDueToGlobalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     short int mask = 1 << (2);
   short int tmp = static_cast<short int>(_packedRecords0 & mask);
   return (tmp != 0);
                  }
                  
                  
                  
                  inline void setMinimalPatchBlockedDueToGlobalTimestep(const bool& minimalPatchBlockedDueToGlobalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     short int mask = 1 << (2);
   _packedRecords0 = static_cast<short int>( minimalPatchBlockedDueToGlobalTimestep ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
                  }
                  
                  
                  
               };
               
            private: 
               PersistentRecords _persistentRecords;
               
            public:
               /**
                * Generated
                */
               TimeIntervalStatisticsPacked();
               
               /**
                * Generated
                */
               TimeIntervalStatisticsPacked(const PersistentRecords& persistentRecords);
               
               /**
                * Generated
                */
               TimeIntervalStatisticsPacked(const int& minimalPatchIndex, const int& minimalPatchParentIndex, const double& minimalPatchTime, const double& startMaximumLocalTimeInterval, const double& endMaximumLocalTimeInterval, const double& startMinimumLocalTimeInterval, const double& endMinimumLocalTimeInterval, const double& minimalTimestep, const bool& allPatchesEvolvedToGlobalTimestep, const double& averageGlobalTimeInterval, const double& globalTimestepEndTime, const bool& minimalPatchBlockedDueToCoarsening, const bool& minimalPatchBlockedDueToGlobalTimestep);
               
               /**
                * Generated
                */
               ~TimeIntervalStatisticsPacked();
               
               
               inline int getMinimalPatchIndex() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._minimalPatchIndex;
               }
               
               
               
               inline void setMinimalPatchIndex(const int& minimalPatchIndex) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._minimalPatchIndex = minimalPatchIndex;
               }
               
               
               
               inline int getMinimalPatchParentIndex() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._minimalPatchParentIndex;
               }
               
               
               
               inline void setMinimalPatchParentIndex(const int& minimalPatchParentIndex) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._minimalPatchParentIndex = minimalPatchParentIndex;
               }
               
               
               
               inline double getMinimalPatchTime() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._minimalPatchTime;
               }
               
               
               
               inline void setMinimalPatchTime(const double& minimalPatchTime) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._minimalPatchTime = minimalPatchTime;
               }
               
               
               
               inline double getStartMaximumLocalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._startMaximumLocalTimeInterval;
               }
               
               
               
               inline void setStartMaximumLocalTimeInterval(const double& startMaximumLocalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._startMaximumLocalTimeInterval = startMaximumLocalTimeInterval;
               }
               
               
               
               inline double getEndMaximumLocalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._endMaximumLocalTimeInterval;
               }
               
               
               
               inline void setEndMaximumLocalTimeInterval(const double& endMaximumLocalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._endMaximumLocalTimeInterval = endMaximumLocalTimeInterval;
               }
               
               
               
               inline double getStartMinimumLocalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._startMinimumLocalTimeInterval;
               }
               
               
               
               inline void setStartMinimumLocalTimeInterval(const double& startMinimumLocalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._startMinimumLocalTimeInterval = startMinimumLocalTimeInterval;
               }
               
               
               
               inline double getEndMinimumLocalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._endMinimumLocalTimeInterval;
               }
               
               
               
               inline void setEndMinimumLocalTimeInterval(const double& endMinimumLocalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._endMinimumLocalTimeInterval = endMinimumLocalTimeInterval;
               }
               
               
               
               inline double getMinimalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._minimalTimestep;
               }
               
               
               
               inline void setMinimalTimestep(const double& minimalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._minimalTimestep = minimalTimestep;
               }
               
               
               
               inline bool getAllPatchesEvolvedToGlobalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  short int mask = 1 << (0);
   short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
   return (tmp != 0);
               }
               
               
               
               inline void setAllPatchesEvolvedToGlobalTimestep(const bool& allPatchesEvolvedToGlobalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  short int mask = 1 << (0);
   _persistentRecords._packedRecords0 = static_cast<short int>( allPatchesEvolvedToGlobalTimestep ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
               }
               
               
               
               inline double getAverageGlobalTimeInterval() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._averageGlobalTimeInterval;
               }
               
               
               
               inline void setAverageGlobalTimeInterval(const double& averageGlobalTimeInterval) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._averageGlobalTimeInterval = averageGlobalTimeInterval;
               }
               
               
               
               inline double getGlobalTimestepEndTime() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._globalTimestepEndTime;
               }
               
               
               
               inline void setGlobalTimestepEndTime(const double& globalTimestepEndTime) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._globalTimestepEndTime = globalTimestepEndTime;
               }
               
               
               
               inline bool getMinimalPatchBlockedDueToCoarsening() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  short int mask = 1 << (1);
   short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
   return (tmp != 0);
               }
               
               
               
               inline void setMinimalPatchBlockedDueToCoarsening(const bool& minimalPatchBlockedDueToCoarsening) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  short int mask = 1 << (1);
   _persistentRecords._packedRecords0 = static_cast<short int>( minimalPatchBlockedDueToCoarsening ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
               }
               
               
               
               inline bool getMinimalPatchBlockedDueToGlobalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  short int mask = 1 << (2);
   short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
   return (tmp != 0);
               }
               
               
               
               inline void setMinimalPatchBlockedDueToGlobalTimestep(const bool& minimalPatchBlockedDueToGlobalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  short int mask = 1 << (2);
   _persistentRecords._packedRecords0 = static_cast<short int>( minimalPatchBlockedDueToGlobalTimestep ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
               }
               
               
               /**
                * Generated
                */
               std::string toString() const;
               
               /**
                * Generated
                */
               void toString(std::ostream& out) const;
               
               
               PersistentRecords getPersistentRecords() const;
               /**
                * Generated
                */
               TimeIntervalStatistics convert() const;
               
               
            #ifdef Parallel
               protected:
                  static tarch::logging::Log _log;
                  
               public:
                  
                  /**
                   * Global that represents the mpi datatype.
                   * There are two variants: Datatype identifies only those attributes marked with
                   * parallelise. FullDatatype instead identifies the whole record with all fields.
                   */
                  static MPI_Datatype Datatype;
                  static MPI_Datatype FullDatatype;
                  
                  /**
                   * Initializes the data type for the mpi operations. Has to be called
                   * before the very first send or receive operation is called.
                   */
                  static void initDatatype();
                  
                  static void shutdownDatatype();
                  
                  void send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking);
                  
                  void receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, bool communicateBlocking);
                  
                  static bool isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise);
                  
                  #endif
                     
                  };
                  
                  #endif
                  
