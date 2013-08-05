#ifndef _PEANOCLAW_RECORDS_DATA_H
#define _PEANOCLAW_RECORDS_DATA_H

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
   namespace records {
      class Data;
      class DataPacked;
   }
}

/**
 * @author This class is generated by DaStGen
 * 		   DataStructureGenerator (DaStGen)
 * 		   2007-2009 Wolfgang Eckhardt
 * 		   2012      Tobias Weinzierl
 *
 * 		   build date: 12-04-2013 09:18
 *
 * @date   05/08/2013 08:10
 */
class peanoclaw::records::Data { 
   
   public:
      
      typedef peanoclaw::records::DataPacked Packed;
      
      struct PersistentRecords {
         double _u;
         /**
          * Generated
          */
         PersistentRecords();
         
         /**
          * Generated
          */
         PersistentRecords(const double& u);
         
         /**
          * Generated
          */
          double getU() const ;
         
         /**
          * Generated
          */
          void setU(const double& u) ;
         
         
      };
      
   private: 
      public:

      PersistentRecords _persistentRecords;
      private:

      
   public:
      /**
       * Generated
       */
      Data();
      
      /**
       * Generated
       */
      Data(const PersistentRecords& persistentRecords);
      
      /**
       * Generated
       */
      Data(const double& u);
      
      /**
       * Generated
       */
      ~Data();
      
      /**
       * Generated
       */
       double getU() const ;
      
      /**
       * Generated
       */
       void setU(const double& u) ;
      
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
      DataPacked convert() const;
      
      
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
          * 		   build date: 12-04-2013 09:18
          *
          * @date   05/08/2013 08:10
          */
         class peanoclaw::records::DataPacked { 
            
            public:
               
               struct PersistentRecords {
                  double _u;
                  /**
                   * Generated
                   */
                  PersistentRecords();
                  
                  /**
                   * Generated
                   */
                  PersistentRecords(const double& u);
                  
                  /**
                   * Generated
                   */
                   double getU() const ;
                  
                  /**
                   * Generated
                   */
                   void setU(const double& u) ;
                  
                  
               };
               
            private: 
               PersistentRecords _persistentRecords;
               
            public:
               /**
                * Generated
                */
               DataPacked();
               
               /**
                * Generated
                */
               DataPacked(const PersistentRecords& persistentRecords);
               
               /**
                * Generated
                */
               DataPacked(const double& u);
               
               /**
                * Generated
                */
               ~DataPacked();
               
               /**
                * Generated
                */
                double getU() const ;
               
               /**
                * Generated
                */
                void setU(const double& u) ;
               
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
               Data convert() const;
               
               
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
                  
