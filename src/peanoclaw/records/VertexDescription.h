#ifndef _PEANOCLAW_RECORDS_VERTEXDESCRIPTION_H
#define _PEANOCLAW_RECORDS_VERTEXDESCRIPTION_H

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
   namespace records {
      class VertexDescription;
      class VertexDescriptionPacked;
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
 * @date   23/11/2013 18:01
 */
class peanoclaw::records::VertexDescription { 
   
   public:
      
      typedef peanoclaw::records::VertexDescriptionPacked Packed;
      
      enum IterationParity {
         EVEN = 0, ODD = 1
      };
      
      struct PersistentRecords {
         tarch::la::Vector<TWO_POWER_D,int> _indicesOfAdjacentCellDescriptions;
         bool _touched;
         /**
          * Generated
          */
         PersistentRecords();
         
         /**
          * Generated
          */
         PersistentRecords(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const bool& touched);
         
          tarch::la::Vector<TWO_POWER_D,int> getIndicesOfAdjacentCellDescriptions() const ;
         
          void setIndicesOfAdjacentCellDescriptions(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions) ;
         
         /**
          * Generated
          */
          bool getTouched() const ;
         
         /**
          * Generated
          */
          void setTouched(const bool& touched) ;
         
         
      };
      
   private: 
      PersistentRecords _persistentRecords;
      
   public:
      /**
       * Generated
       */
      VertexDescription();
      
      /**
       * Generated
       */
      VertexDescription(const PersistentRecords& persistentRecords);
      
      /**
       * Generated
       */
      VertexDescription(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const bool& touched);
      
      /**
       * Generated
       */
      ~VertexDescription();
      
       tarch::la::Vector<TWO_POWER_D,int> getIndicesOfAdjacentCellDescriptions() const ;
      
       void setIndicesOfAdjacentCellDescriptions(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions) ;
      
       int getIndicesOfAdjacentCellDescriptions(int elementIndex) const ;
      
       void setIndicesOfAdjacentCellDescriptions(int elementIndex, const int& indicesOfAdjacentCellDescriptions) ;
      
      /**
       * Generated
       */
       bool getTouched() const ;
      
      /**
       * Generated
       */
       void setTouched(const bool& touched) ;
      
      /**
       * Generated
       */
      static std::string toString(const IterationParity& param);
      
      /**
       * Generated
       */
      static std::string getIterationParityMapping();
      
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
      VertexDescriptionPacked convert() const;
      
      
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
          * @date   23/11/2013 18:01
          */
         class peanoclaw::records::VertexDescriptionPacked { 
            
            public:
               
               typedef peanoclaw::records::VertexDescription::IterationParity IterationParity;
               
               struct PersistentRecords {
                  tarch::la::Vector<TWO_POWER_D,int> _indicesOfAdjacentCellDescriptions;
                  bool _touched;
                  /**
                   * Generated
                   */
                  PersistentRecords();
                  
                  /**
                   * Generated
                   */
                  PersistentRecords(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const bool& touched);
                  
                   tarch::la::Vector<TWO_POWER_D,int> getIndicesOfAdjacentCellDescriptions() const ;
                  
                   void setIndicesOfAdjacentCellDescriptions(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions) ;
                  
                  /**
                   * Generated
                   */
                   bool getTouched() const ;
                  
                  /**
                   * Generated
                   */
                   void setTouched(const bool& touched) ;
                  
                  
               };
               
            private: 
               PersistentRecords _persistentRecords;
               
            public:
               /**
                * Generated
                */
               VertexDescriptionPacked();
               
               /**
                * Generated
                */
               VertexDescriptionPacked(const PersistentRecords& persistentRecords);
               
               /**
                * Generated
                */
               VertexDescriptionPacked(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions, const bool& touched);
               
               /**
                * Generated
                */
               ~VertexDescriptionPacked();
               
                tarch::la::Vector<TWO_POWER_D,int> getIndicesOfAdjacentCellDescriptions() const ;
               
                void setIndicesOfAdjacentCellDescriptions(const tarch::la::Vector<TWO_POWER_D,int>& indicesOfAdjacentCellDescriptions) ;
               
                int getIndicesOfAdjacentCellDescriptions(int elementIndex) const ;
               
                void setIndicesOfAdjacentCellDescriptions(int elementIndex, const int& indicesOfAdjacentCellDescriptions) ;
               
               /**
                * Generated
                */
                bool getTouched() const ;
               
               /**
                * Generated
                */
                void setTouched(const bool& touched) ;
               
               /**
                * Generated
                */
               static std::string toString(const IterationParity& param);
               
               /**
                * Generated
                */
               static std::string getIterationParityMapping();
               
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
               VertexDescription convert() const;
               
               
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
                  
