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
 * 		   build date: 22-10-2013 20:59
 *
 * @date   21/11/2013 14:28
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
         /**
          * Generated
          */
         PersistentRecords();
         
         /**
          * Generated
          */
         PersistentRecords(const double& area, const int& level, const double& numberOfPatches, const double& numberOfCells, const double& numberOfCellUpdates, const double& createdPatches, const double& destroyedPatches, const double& patchesBlockedDueToNeighbors, const double& patchesBlockedDueToGlobalTimestep, const double& patchesSkippingIteration, const double& patchesCoarsening);
         
         /**
          * Generated
          */
          double getArea() const ;
         
         /**
          * Generated
          */
          void setArea(const double& area) ;
         
         /**
          * Generated
          */
          int getLevel() const ;
         
         /**
          * Generated
          */
          void setLevel(const int& level) ;
         
         /**
          * Generated
          */
          double getNumberOfPatches() const ;
         
         /**
          * Generated
          */
          void setNumberOfPatches(const double& numberOfPatches) ;
         
         /**
          * Generated
          */
          double getNumberOfCells() const ;
         
         /**
          * Generated
          */
          void setNumberOfCells(const double& numberOfCells) ;
         
         /**
          * Generated
          */
          double getNumberOfCellUpdates() const ;
         
         /**
          * Generated
          */
          void setNumberOfCellUpdates(const double& numberOfCellUpdates) ;
         
         /**
          * Generated
          */
          double getCreatedPatches() const ;
         
         /**
          * Generated
          */
          void setCreatedPatches(const double& createdPatches) ;
         
         /**
          * Generated
          */
          double getDestroyedPatches() const ;
         
         /**
          * Generated
          */
          void setDestroyedPatches(const double& destroyedPatches) ;
         
         /**
          * Generated
          */
          double getPatchesBlockedDueToNeighbors() const ;
         
         /**
          * Generated
          */
          void setPatchesBlockedDueToNeighbors(const double& patchesBlockedDueToNeighbors) ;
         
         /**
          * Generated
          */
          double getPatchesBlockedDueToGlobalTimestep() const ;
         
         /**
          * Generated
          */
          void setPatchesBlockedDueToGlobalTimestep(const double& patchesBlockedDueToGlobalTimestep) ;
         
         /**
          * Generated
          */
          double getPatchesSkippingIteration() const ;
         
         /**
          * Generated
          */
          void setPatchesSkippingIteration(const double& patchesSkippingIteration) ;
         
         /**
          * Generated
          */
          double getPatchesCoarsening() const ;
         
         /**
          * Generated
          */
          void setPatchesCoarsening(const double& patchesCoarsening) ;
         
         
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
      LevelStatistics(const double& area, const int& level, const double& numberOfPatches, const double& numberOfCells, const double& numberOfCellUpdates, const double& createdPatches, const double& destroyedPatches, const double& patchesBlockedDueToNeighbors, const double& patchesBlockedDueToGlobalTimestep, const double& patchesSkippingIteration, const double& patchesCoarsening);
      
      /**
       * Generated
       */
      ~LevelStatistics();
      
      /**
       * Generated
       */
       double getArea() const ;
      
      /**
       * Generated
       */
       void setArea(const double& area) ;
      
      /**
       * Generated
       */
       int getLevel() const ;
      
      /**
       * Generated
       */
       void setLevel(const int& level) ;
      
      /**
       * Generated
       */
       double getNumberOfPatches() const ;
      
      /**
       * Generated
       */
       void setNumberOfPatches(const double& numberOfPatches) ;
      
      /**
       * Generated
       */
       double getNumberOfCells() const ;
      
      /**
       * Generated
       */
       void setNumberOfCells(const double& numberOfCells) ;
      
      /**
       * Generated
       */
       double getNumberOfCellUpdates() const ;
      
      /**
       * Generated
       */
       void setNumberOfCellUpdates(const double& numberOfCellUpdates) ;
      
      /**
       * Generated
       */
       double getCreatedPatches() const ;
      
      /**
       * Generated
       */
       void setCreatedPatches(const double& createdPatches) ;
      
      /**
       * Generated
       */
       double getDestroyedPatches() const ;
      
      /**
       * Generated
       */
       void setDestroyedPatches(const double& destroyedPatches) ;
      
      /**
       * Generated
       */
       double getPatchesBlockedDueToNeighbors() const ;
      
      /**
       * Generated
       */
       void setPatchesBlockedDueToNeighbors(const double& patchesBlockedDueToNeighbors) ;
      
      /**
       * Generated
       */
       double getPatchesBlockedDueToGlobalTimestep() const ;
      
      /**
       * Generated
       */
       void setPatchesBlockedDueToGlobalTimestep(const double& patchesBlockedDueToGlobalTimestep) ;
      
      /**
       * Generated
       */
       double getPatchesSkippingIteration() const ;
      
      /**
       * Generated
       */
       void setPatchesSkippingIteration(const double& patchesSkippingIteration) ;
      
      /**
       * Generated
       */
       double getPatchesCoarsening() const ;
      
      /**
       * Generated
       */
       void setPatchesCoarsening(const double& patchesCoarsening) ;
      
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
         
         void send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise);
         
         void receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise);
         
         static bool isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise);
         
         #endif
            
         };
         
         /**
          * @author This class is generated by DaStGen
          * 		   DataStructureGenerator (DaStGen)
          * 		   2007-2009 Wolfgang Eckhardt
          * 		   2012      Tobias Weinzierl
          *
          * 		   build date: 22-10-2013 20:59
          *
          * @date   21/11/2013 14:28
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
                  /**
                   * Generated
                   */
                  PersistentRecords();
                  
                  /**
                   * Generated
                   */
                  PersistentRecords(const double& area, const int& level, const double& numberOfPatches, const double& numberOfCells, const double& numberOfCellUpdates, const double& createdPatches, const double& destroyedPatches, const double& patchesBlockedDueToNeighbors, const double& patchesBlockedDueToGlobalTimestep, const double& patchesSkippingIteration, const double& patchesCoarsening);
                  
                  /**
                   * Generated
                   */
                   double getArea() const ;
                  
                  /**
                   * Generated
                   */
                   void setArea(const double& area) ;
                  
                  /**
                   * Generated
                   */
                   int getLevel() const ;
                  
                  /**
                   * Generated
                   */
                   void setLevel(const int& level) ;
                  
                  /**
                   * Generated
                   */
                   double getNumberOfPatches() const ;
                  
                  /**
                   * Generated
                   */
                   void setNumberOfPatches(const double& numberOfPatches) ;
                  
                  /**
                   * Generated
                   */
                   double getNumberOfCells() const ;
                  
                  /**
                   * Generated
                   */
                   void setNumberOfCells(const double& numberOfCells) ;
                  
                  /**
                   * Generated
                   */
                   double getNumberOfCellUpdates() const ;
                  
                  /**
                   * Generated
                   */
                   void setNumberOfCellUpdates(const double& numberOfCellUpdates) ;
                  
                  /**
                   * Generated
                   */
                   double getCreatedPatches() const ;
                  
                  /**
                   * Generated
                   */
                   void setCreatedPatches(const double& createdPatches) ;
                  
                  /**
                   * Generated
                   */
                   double getDestroyedPatches() const ;
                  
                  /**
                   * Generated
                   */
                   void setDestroyedPatches(const double& destroyedPatches) ;
                  
                  /**
                   * Generated
                   */
                   double getPatchesBlockedDueToNeighbors() const ;
                  
                  /**
                   * Generated
                   */
                   void setPatchesBlockedDueToNeighbors(const double& patchesBlockedDueToNeighbors) ;
                  
                  /**
                   * Generated
                   */
                   double getPatchesBlockedDueToGlobalTimestep() const ;
                  
                  /**
                   * Generated
                   */
                   void setPatchesBlockedDueToGlobalTimestep(const double& patchesBlockedDueToGlobalTimestep) ;
                  
                  /**
                   * Generated
                   */
                   double getPatchesSkippingIteration() const ;
                  
                  /**
                   * Generated
                   */
                   void setPatchesSkippingIteration(const double& patchesSkippingIteration) ;
                  
                  /**
                   * Generated
                   */
                   double getPatchesCoarsening() const ;
                  
                  /**
                   * Generated
                   */
                   void setPatchesCoarsening(const double& patchesCoarsening) ;
                  
                  
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
               LevelStatisticsPacked(const double& area, const int& level, const double& numberOfPatches, const double& numberOfCells, const double& numberOfCellUpdates, const double& createdPatches, const double& destroyedPatches, const double& patchesBlockedDueToNeighbors, const double& patchesBlockedDueToGlobalTimestep, const double& patchesSkippingIteration, const double& patchesCoarsening);
               
               /**
                * Generated
                */
               ~LevelStatisticsPacked();
               
               /**
                * Generated
                */
                double getArea() const ;
               
               /**
                * Generated
                */
                void setArea(const double& area) ;
               
               /**
                * Generated
                */
                int getLevel() const ;
               
               /**
                * Generated
                */
                void setLevel(const int& level) ;
               
               /**
                * Generated
                */
                double getNumberOfPatches() const ;
               
               /**
                * Generated
                */
                void setNumberOfPatches(const double& numberOfPatches) ;
               
               /**
                * Generated
                */
                double getNumberOfCells() const ;
               
               /**
                * Generated
                */
                void setNumberOfCells(const double& numberOfCells) ;
               
               /**
                * Generated
                */
                double getNumberOfCellUpdates() const ;
               
               /**
                * Generated
                */
                void setNumberOfCellUpdates(const double& numberOfCellUpdates) ;
               
               /**
                * Generated
                */
                double getCreatedPatches() const ;
               
               /**
                * Generated
                */
                void setCreatedPatches(const double& createdPatches) ;
               
               /**
                * Generated
                */
                double getDestroyedPatches() const ;
               
               /**
                * Generated
                */
                void setDestroyedPatches(const double& destroyedPatches) ;
               
               /**
                * Generated
                */
                double getPatchesBlockedDueToNeighbors() const ;
               
               /**
                * Generated
                */
                void setPatchesBlockedDueToNeighbors(const double& patchesBlockedDueToNeighbors) ;
               
               /**
                * Generated
                */
                double getPatchesBlockedDueToGlobalTimestep() const ;
               
               /**
                * Generated
                */
                void setPatchesBlockedDueToGlobalTimestep(const double& patchesBlockedDueToGlobalTimestep) ;
               
               /**
                * Generated
                */
                double getPatchesSkippingIteration() const ;
               
               /**
                * Generated
                */
                void setPatchesSkippingIteration(const double& patchesSkippingIteration) ;
               
               /**
                * Generated
                */
                double getPatchesCoarsening() const ;
               
               /**
                * Generated
                */
                void setPatchesCoarsening(const double& patchesCoarsening) ;
               
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
                  
                  void send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise);
                  
                  void receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise);
                  
                  static bool isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise);
                  
                  #endif
                     
                  };
                  
                  #endif
                  
