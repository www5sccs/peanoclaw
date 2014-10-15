#ifndef _PEANOCLAW_RECORDS_FLOATDATA_H
#define _PEANOCLAW_RECORDS_FLOATDATA_H

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
      class FloatData;
      class FloatDataPacked;
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
 * @date   15/10/2014 11:03
 */
class peanoclaw::records::FloatData { 
   
   public:
      
      typedef peanoclaw::records::FloatDataPacked Packed;
      
      struct PersistentRecords {
         float _u;
         /**
          * Generated
          */
         PersistentRecords();
         
         /**
          * Generated
          */
         PersistentRecords(const float& u);
         
         
         inline float getU() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            return _u;
         }
         
         
         
         inline void setU(const float& u) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
            _u = u;
         }
         
         
         
      };
      
   private: 
      public:

      PersistentRecords _persistentRecords;
      private:

      
   public:
      /**
       * Generated
       */
      FloatData();
      
      /**
       * Generated
       */
      FloatData(const PersistentRecords& persistentRecords);
      
      /**
       * Generated
       */
      FloatData(const float& u);
      
      /**
       * Generated
       */
      ~FloatData();
      
      
      inline float getU() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         return _persistentRecords._u;
      }
      
      
      
      inline void setU(const float& u) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
         _persistentRecords._u = u;
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
      FloatDataPacked convert() const;
      
      
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
          * @date   15/10/2014 11:03
          */
         class peanoclaw::records::FloatDataPacked { 
            
            public:
               
               struct PersistentRecords {
                  float _u;
                  /**
                   * Generated
                   */
                  PersistentRecords();
                  
                  /**
                   * Generated
                   */
                  PersistentRecords(const float& u);
                  
                  
                  inline float getU() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     return _u;
                  }
                  
                  
                  
                  inline void setU(const float& u) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                     _u = u;
                  }
                  
                  
                  
               };
               
            private: 
               PersistentRecords _persistentRecords;
               
            public:
               /**
                * Generated
                */
               FloatDataPacked();
               
               /**
                * Generated
                */
               FloatDataPacked(const PersistentRecords& persistentRecords);
               
               /**
                * Generated
                */
               FloatDataPacked(const float& u);
               
               /**
                * Generated
                */
               ~FloatDataPacked();
               
               
               inline float getU() const 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  return _persistentRecords._u;
               }
               
               
               
               inline void setU(const float& u) 
 #ifdef UseManualInlining
 __attribute__((always_inline))
 #endif 
 {
                  _persistentRecords._u = u;
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
               FloatData convert() const;
               
               
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
                  
