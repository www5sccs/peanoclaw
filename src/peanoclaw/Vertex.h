// This file is part of the Peano project. For conditions of distribution and 
// use, please see the copyright notice at www.peano-framework.org
#ifndef _PEANOCLAW_VERTEX_H_ 
#define _PEANOCLAW_VERTEX_H_


#include "records/Vertex.h"
#include "peano/grid/Vertex.h"
#include "peano/grid/VertexEnumerator.h"
#include "peano/utils/Globals.h"

#include "records/CellDescription.h"
#include "records/Data.h"

namespace peanoclaw {
  class Numerics;
  class Vertex;
}


/**
 * Blueprint for grid vertex.
 * 
 * This file has originally been created by the PDT and may be manually extended to 
 * the needs of your application. We do not recommend to remove anything!
 */
class peanoclaw::Vertex: public peano::grid::Vertex< peanoclaw::records::Vertex > { 
  private: 
    typedef class peano::grid::Vertex< peanoclaw::records::Vertex >  Base;
    typedef class peanoclaw::records::CellDescription CellDescription;
    typedef class peanoclaw::records::Data Data;

  public:
    /**
     * Default Constructor
     *
     * This constructor is required by the framework's data container. Do not 
     * remove it.
     */
    Vertex();
    
    /**
     * This constructor should not set any attributes. It is used by the 
     * traversal algorithm whenever it allocates an array whose elements 
     * will be overwritten later anyway.  
     */
    Vertex(const Base::DoNotCallStandardConstructor&);
    
    /**
     * Constructor
     *
     * This constructor is required by the framework's data container. Do not 
     * remove it. It is kind of a copy constructor that converts an object which 
     * comprises solely persistent attributes into a full attribute. This very 
     * functionality is implemented within the super type, i.e. this constructor 
     * has to invoke the correponsing super type's constructor and not the super 
     * type standard constructor.
     */
    Vertex(const Base::PersistentVertex& argument);
    
    /**
     * Returns the index to the heap data storing the vertex description.
     */
    int getVertexDescriptionIndex() const;

//    void setAdjacentUNewIndex(int cellIndex, int patchIndex);
//
//    int getAdjacentUNewIndex(int cellIndex) const;
//
//    void setAdjacentUOldIndex(int cellIndex, int patchIndex);
//
//    int getAdjacentUOldIndex(int cellIndex) const;

    void setAdjacentCellDescriptionIndex(
      int cellIndex,
      int cellDescriptionIndex
    );

    /**
     * @see getAdjacentCellDescriptionIndexInPeanoOrder
     */
    void setAdjacentCellDescriptionIndexInPeanoOrder(
      int cellIndex,
      int cellDescriptionIndex
    );

    int getAdjacentCellDescriptionIndex(int cellIndex) const;

    /**
     * PeanoClaw stores the adjacent cells in inverse-z-order, i.e.
     *
     * 1          0
     *    vertex
     * 3          2
     *
     * while Peano stores in z-order, i.e.
     * 2          3
     *    vertex
     * 0          1
     *
     * This method grants access to the adjacent cell descriptions while
     * iterating through the adjacent cells in the same way like for the
     * adjacent ranks.
     */
    int getAdjacentCellDescriptionIndexInPeanoOrder(int cellIndex) const;

    /**
     * Fills the ghostlayers of all adjacent patches.
     */
    void fillAdjacentGhostLayers(
      int level,
      bool useDimensionalSplitting,
      peanoclaw::Numerics& numerics,
      int destinationPatch = -1
    ) const;

    /**
     * Applies the coarse grid correction on all adjacent patches.
     */
    void applyCoarseGridCorrection(
      peanoclaw::Numerics& numerics
    ) const;

    void setShouldRefine(bool shouldRefine);

    bool shouldRefine() const;

    /**
     * Resets the flags indicating the subcells that
     * should be erased.
     */
    void resetSubcellsEraseVeto();

    /**
     * States that a subcell in the given quadrant/octant must
     * not be erased.
     */
    void setSubcellEraseVeto(int cellIndex);

    /**
     * Returns wether all adjacent subcells should be erased.
     */
    bool shouldErase() const;
};


#endif
