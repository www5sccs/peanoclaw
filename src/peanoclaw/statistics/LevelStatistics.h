#ifndef _PEANOCLAW_STATISTICS_LEVELSTATISTICS_H
#define _PEANOCLAW_STATISTICS_LEVELSTATISTICS_H

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
      class LevelStatistics;
      class LevelStatisticsPacked;
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
class peanoclaw::statistics::LevelStatistics { 
   
   public:
      
      typedef peanoclaw::statistics::LevelStatisticsPacked Packed;
      
      struct PersistentRecords {
         double _area;
         int _level;
         double _numberOfPatches;
         double _numberOfCells;
         double _numberOfCellUpdates;
         double _createdPatches;
         double _destroyedPatches;
         double _patchesBlockedDueToNeighbors;
         double _patchesBlockedDueToGlobalTimestep;
         double _patchesSkippingIteration;
         double _patchesCoarsening;
         int _estimatedNumberOfRemainingIterationsToGlobalTimestep;
         /**
          * Generated
          */
         PersistentRecords();
         
         /**
          * Generated
          */
         PersistentRecords(const double& area, const int& level, const double& numberOfPatches, const double& numberOfCells, const double& numberOfCellUpdates, const double& createdPatches, const double& destroyedPatches, const double& patchesBlockedDueToNeighbors, const double& patchesBlockedDueToGlobalTimestep, const double& patchesSkippingIteration, const double& patchesCoarsening, const int& estimatedNumberOfRemainingIterationsToGlobalTimestep);
         
         
         inline double getArea() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _area;
         }
         
         
         
         inline void setArea(const double& area) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _area = area;
         }
         
         
         
         inline int getLevel() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _level;
         }
         
         
         
         inline void setLevel(const int& level) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _level = level;
         }
         
         
         
         inline double getNumberOfPatches() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _numberOfPatches;
         }
         
         
         
         inline void setNumberOfPatches(const double& numberOfPatches) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _numberOfPatches = numberOfPatches;
         }
         
         
         
         inline double getNumberOfCells() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _numberOfCells;
         }
         
         
         
         inline void setNumberOfCells(const double& numberOfCells) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _numberOfCells = numberOfCells;
         }
         
         
         
         inline double getNumberOfCellUpdates() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _numberOfCellUpdates;
         }
         
         
         
         inline void setNumberOfCellUpdates(const double& numberOfCellUpdates) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _numberOfCellUpdates = numberOfCellUpdates;
         }
         
         
         
         inline double getCreatedPatches() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _createdPatches;
         }
         
         
         
         inline void setCreatedPatches(const double& createdPatches) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _createdPatches = createdPatches;
         }
         
         
         
         inline double getDestroyedPatches() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _destroyedPatches;
         }
         
         
         
         inline void setDestroyedPatches(const double& destroyedPatches) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _destroyedPatches = destroyedPatches;
         }
         
         
         
         inline double getPatchesBlockedDueToNeighbors() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _patchesBlockedDueToNeighbors;
         }
         
         
         
         inline void setPatchesBlockedDueToNeighbors(const double& patchesBlockedDueToNeighbors) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _patchesBlockedDueToNeighbors = patchesBlockedDueToNeighbors;
         }
         
         
         
         inline double getPatchesBlockedDueToGlobalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _patchesBlockedDueToGlobalTimestep;
         }
         
         
         
         inline void setPatchesBlockedDueToGlobalTimestep(const double& patchesBlockedDueToGlobalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _patchesBlockedDueToGlobalTimestep = patchesBlockedDueToGlobalTimestep;
         }
         
         
         
         inline double getPatchesSkippingIteration() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _patchesSkippingIteration;
         }
         
         
         
         inline void setPatchesSkippingIteration(const double& patchesSkippingIteration) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _patchesSkippingIteration = patchesSkippingIteration;
         }
         
         
         
         inline double getPatchesCoarsening() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _patchesCoarsening;
         }
         
         
         
         inline void setPatchesCoarsening(const double& patchesCoarsening) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _patchesCoarsening = patchesCoarsening;
         }
         
         
         
         inline int getEstimatedNumberOfRemainingIterationsToGlobalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _estimatedNumberOfRemainingIterationsToGlobalTimestep;
         }
         
         
         
         inline void setEstimatedNumberOfRemainingIterationsToGlobalTimestep(const int& estimatedNumberOfRemainingIterationsToGlobalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _estimatedNumberOfRemainingIterationsToGlobalTimestep = estimatedNumberOfRemainingIterationsToGlobalTimestep;
         }
         
         
         
      };
      
   private: 
      PersistentRecords _persistentRecords;
      
   public:
      /**
       * Generated
       */
      LevelStatistics();
      
      /**
       * Generated
       */
      LevelStatistics(const PersistentRecords& persistentRecords);
      
      /**
       * Generated
       */
      LevelStatistics(const double& area, const int& level, const double& numberOfPatches, const double& numberOfCells, const double& numberOfCellUpdates, const double& createdPatches, const double& destroyedPatches, const double& patchesBlockedDueToNeighbors, const double& patchesBlockedDueToGlobalTimestep, const double& patchesSkippingIteration, const double& patchesCoarsening, const int& estimatedNumberOfRemainingIterationsToGlobalTimestep);
      
      /**
       * Generated
       */
      ~LevelStatistics();
      
      
      inline double getArea() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._area;
      }
      
      
      
      inline void setArea(const double& area) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._area = area;
      }
      
      
      
      inline int getLevel() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._level;
      }
      
      
      
      inline void setLevel(const int& level) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._level = level;
      }
      
      
      
      inline double getNumberOfPatches() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._numberOfPatches;
      }
      
      
      
      inline void setNumberOfPatches(const double& numberOfPatches) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._numberOfPatches = numberOfPatches;
      }
      
      
      
      inline double getNumberOfCells() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._numberOfCells;
      }
      
      
      
      inline void setNumberOfCells(const double& numberOfCells) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._numberOfCells = numberOfCells;
      }
      
      
      
      inline double getNumberOfCellUpdates() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._numberOfCellUpdates;
      }
      
      
      
      inline void setNumberOfCellUpdates(const double& numberOfCellUpdates) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._numberOfCellUpdates = numberOfCellUpdates;
      }
      
      
      
      inline double getCreatedPatches() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._createdPatches;
      }
      
      
      
      inline void setCreatedPatches(const double& createdPatches) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._createdPatches = createdPatches;
      }
      
      
      
      inline double getDestroyedPatches() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._destroyedPatches;
      }
      
      
      
      inline void setDestroyedPatches(const double& destroyedPatches) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._destroyedPatches = destroyedPatches;
      }
      
      
      
      inline double getPatchesBlockedDueToNeighbors() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._patchesBlockedDueToNeighbors;
      }
      
      
      
      inline void setPatchesBlockedDueToNeighbors(const double& patchesBlockedDueToNeighbors) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._patchesBlockedDueToNeighbors = patchesBlockedDueToNeighbors;
      }
      
      
      
      inline double getPatchesBlockedDueToGlobalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._patchesBlockedDueToGlobalTimestep;
      }
      
      
      
      inline void setPatchesBlockedDueToGlobalTimestep(const double& patchesBlockedDueToGlobalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._patchesBlockedDueToGlobalTimestep = patchesBlockedDueToGlobalTimestep;
      }
      
      
      
      inline double getPatchesSkippingIteration() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._patchesSkippingIteration;
      }
      
      
      
      inline void setPatchesSkippingIteration(const double& patchesSkippingIteration) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._patchesSkippingIteration = patchesSkippingIteration;
      }
      
      
      
      inline double getPatchesCoarsening() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._patchesCoarsening;
      }
      
      
      
      inline void setPatchesCoarsening(const double& patchesCoarsening) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._patchesCoarsening = patchesCoarsening;
      }
      
      
      
      inline int getEstimatedNumberOfRemainingIterationsToGlobalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._estimatedNumberOfRemainingIterationsToGlobalTimestep;
      }
      
      
      
      inline void setEstimatedNumberOfRemainingIterationsToGlobalTimestep(const int& estimatedNumberOfRemainingIterationsToGlobalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._estimatedNumberOfRemainingIterationsToGlobalTimestep = estimatedNumberOfRemainingIterationsToGlobalTimestep;
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
      LevelStatisticsPacked convert() const;
      
      
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
         class peanoclaw::statistics::LevelStatisticsPacked { 
            
            public:
               
               struct PersistentRecords {
                  double _area;
                  int _level;
                  double _numberOfPatches;
                  double _numberOfCells;
                  double _numberOfCellUpdates;
                  double _createdPatches;
                  double _destroyedPatches;
                  double _patchesBlockedDueToNeighbors;
                  double _patchesBlockedDueToGlobalTimestep;
                  double _patchesSkippingIteration;
                  double _patchesCoarsening;
                  int _estimatedNumberOfRemainingIterationsToGlobalTimestep;
                  /**
                   * Generated
                   */
                  PersistentRecords();
                  
                  /**
                   * Generated
                   */
                  PersistentRecords(const double& area, const int& level, const double& numberOfPatches, const double& numberOfCells, const double& numberOfCellUpdates, const double& createdPatches, const double& destroyedPatches, const double& patchesBlockedDueToNeighbors, const double& patchesBlockedDueToGlobalTimestep, const double& patchesSkippingIteration, const double& patchesCoarsening, const int& estimatedNumberOfRemainingIterationsToGlobalTimestep);
                  
                  
                  inline double getArea() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _area;
                  }
                  
                  
                  
                  inline void setArea(const double& area) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _area = area;
                  }
                  
                  
                  
                  inline int getLevel() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _level;
                  }
                  
                  
                  
                  inline void setLevel(const int& level) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _level = level;
                  }
                  
                  
                  
                  inline double getNumberOfPatches() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _numberOfPatches;
                  }
                  
                  
                  
                  inline void setNumberOfPatches(const double& numberOfPatches) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _numberOfPatches = numberOfPatches;
                  }
                  
                  
                  
                  inline double getNumberOfCells() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _numberOfCells;
                  }
                  
                  
                  
                  inline void setNumberOfCells(const double& numberOfCells) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _numberOfCells = numberOfCells;
                  }
                  
                  
                  
                  inline double getNumberOfCellUpdates() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _numberOfCellUpdates;
                  }
                  
                  
                  
                  inline void setNumberOfCellUpdates(const double& numberOfCellUpdates) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _numberOfCellUpdates = numberOfCellUpdates;
                  }
                  
                  
                  
                  inline double getCreatedPatches() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _createdPatches;
                  }
                  
                  
                  
                  inline void setCreatedPatches(const double& createdPatches) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _createdPatches = createdPatches;
                  }
                  
                  
                  
                  inline double getDestroyedPatches() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _destroyedPatches;
                  }
                  
                  
                  
                  inline void setDestroyedPatches(const double& destroyedPatches) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _destroyedPatches = destroyedPatches;
                  }
                  
                  
                  
                  inline double getPatchesBlockedDueToNeighbors() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _patchesBlockedDueToNeighbors;
                  }
                  
                  
                  
                  inline void setPatchesBlockedDueToNeighbors(const double& patchesBlockedDueToNeighbors) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _patchesBlockedDueToNeighbors = patchesBlockedDueToNeighbors;
                  }
                  
                  
                  
                  inline double getPatchesBlockedDueToGlobalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _patchesBlockedDueToGlobalTimestep;
                  }
                  
                  
                  
                  inline void setPatchesBlockedDueToGlobalTimestep(const double& patchesBlockedDueToGlobalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _patchesBlockedDueToGlobalTimestep = patchesBlockedDueToGlobalTimestep;
                  }
                  
                  
                  
                  inline double getPatchesSkippingIteration() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _patchesSkippingIteration;
                  }
                  
                  
                  
                  inline void setPatchesSkippingIteration(const double& patchesSkippingIteration) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _patchesSkippingIteration = patchesSkippingIteration;
                  }
                  
                  
                  
                  inline double getPatchesCoarsening() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _patchesCoarsening;
                  }
                  
                  
                  
                  inline void setPatchesCoarsening(const double& patchesCoarsening) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _patchesCoarsening = patchesCoarsening;
                  }
                  
                  
                  
                  inline int getEstimatedNumberOfRemainingIterationsToGlobalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _estimatedNumberOfRemainingIterationsToGlobalTimestep;
                  }
                  
                  
                  
                  inline void setEstimatedNumberOfRemainingIterationsToGlobalTimestep(const int& estimatedNumberOfRemainingIterationsToGlobalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _estimatedNumberOfRemainingIterationsToGlobalTimestep = estimatedNumberOfRemainingIterationsToGlobalTimestep;
                  }
                  
                  
                  
               };
               
            private: 
               PersistentRecords _persistentRecords;
               
            public:
               /**
                * Generated
                */
               LevelStatisticsPacked();
               
               /**
                * Generated
                */
               LevelStatisticsPacked(const PersistentRecords& persistentRecords);
               
               /**
                * Generated
                */
               LevelStatisticsPacked(const double& area, const int& level, const double& numberOfPatches, const double& numberOfCells, const double& numberOfCellUpdates, const double& createdPatches, const double& destroyedPatches, const double& patchesBlockedDueToNeighbors, const double& patchesBlockedDueToGlobalTimestep, const double& patchesSkippingIteration, const double& patchesCoarsening, const int& estimatedNumberOfRemainingIterationsToGlobalTimestep);
               
               /**
                * Generated
                */
               ~LevelStatisticsPacked();
               
               
               inline double getArea() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._area;
               }
               
               
               
               inline void setArea(const double& area) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._area = area;
               }
               
               
               
               inline int getLevel() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._level;
               }
               
               
               
               inline void setLevel(const int& level) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._level = level;
               }
               
               
               
               inline double getNumberOfPatches() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._numberOfPatches;
               }
               
               
               
               inline void setNumberOfPatches(const double& numberOfPatches) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._numberOfPatches = numberOfPatches;
               }
               
               
               
               inline double getNumberOfCells() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._numberOfCells;
               }
               
               
               
               inline void setNumberOfCells(const double& numberOfCells) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._numberOfCells = numberOfCells;
               }
               
               
               
               inline double getNumberOfCellUpdates() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._numberOfCellUpdates;
               }
               
               
               
               inline void setNumberOfCellUpdates(const double& numberOfCellUpdates) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._numberOfCellUpdates = numberOfCellUpdates;
               }
               
               
               
               inline double getCreatedPatches() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._createdPatches;
               }
               
               
               
               inline void setCreatedPatches(const double& createdPatches) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._createdPatches = createdPatches;
               }
               
               
               
               inline double getDestroyedPatches() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._destroyedPatches;
               }
               
               
               
               inline void setDestroyedPatches(const double& destroyedPatches) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._destroyedPatches = destroyedPatches;
               }
               
               
               
               inline double getPatchesBlockedDueToNeighbors() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._patchesBlockedDueToNeighbors;
               }
               
               
               
               inline void setPatchesBlockedDueToNeighbors(const double& patchesBlockedDueToNeighbors) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._patchesBlockedDueToNeighbors = patchesBlockedDueToNeighbors;
               }
               
               
               
               inline double getPatchesBlockedDueToGlobalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._patchesBlockedDueToGlobalTimestep;
               }
               
               
               
               inline void setPatchesBlockedDueToGlobalTimestep(const double& patchesBlockedDueToGlobalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._patchesBlockedDueToGlobalTimestep = patchesBlockedDueToGlobalTimestep;
               }
               
               
               
               inline double getPatchesSkippingIteration() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._patchesSkippingIteration;
               }
               
               
               
               inline void setPatchesSkippingIteration(const double& patchesSkippingIteration) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._patchesSkippingIteration = patchesSkippingIteration;
               }
               
               
               
               inline double getPatchesCoarsening() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._patchesCoarsening;
               }
               
               
               
               inline void setPatchesCoarsening(const double& patchesCoarsening) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._patchesCoarsening = patchesCoarsening;
               }
               
               
               
               inline int getEstimatedNumberOfRemainingIterationsToGlobalTimestep() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._estimatedNumberOfRemainingIterationsToGlobalTimestep;
               }
               
               
               
               inline void setEstimatedNumberOfRemainingIterationsToGlobalTimestep(const int& estimatedNumberOfRemainingIterationsToGlobalTimestep) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._estimatedNumberOfRemainingIterationsToGlobalTimestep = estimatedNumberOfRemainingIterationsToGlobalTimestep;
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
               LevelStatistics convert() const;
               
               
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
                  
