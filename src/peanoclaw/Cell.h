// This file is part of the Peano project. For conditions of distribution and 
// use, please see the copyright notice at www.peano-framework.org
#ifndef _PEANOCLAW_CELL_H_ 
#define _PEANOCLAW_CELL_H_


#include "peanoclaw/records/Cell.h"
#include "peano/grid/Cell.h"


namespace peanoclaw { 
  class Cell;
  namespace records {
    class CellDescription;
  }
}


/**
 * Blueprint for cell.
 * 
 * This file has originally been created by the PDT and may be manually extended to 
 * the needs of your application. We do not recommend to remove anything!
 */
class peanoclaw::Cell: public peano::grid::Cell< peanoclaw::records::Cell > { 
  private: 
    typedef class peano::grid::Cell< peanoclaw::records::Cell >  Base;
    typedef class peanoclaw::records::CellDescription CellDescription;

  public:
    /**
     * Default Constructor
     *
     * This constructor is required by the framework's data container. Do not 
     * remove it.
     */
    Cell();

    /**
     * This constructor should not set any attributes. It is used by the 
     * traversal algorithm whenever it allocates an array whose elements 
     * will be overwritten later anyway.  
     */
    Cell(const Base::DoNotCallStandardConstructor&);

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
    Cell(const Base::PersistentCell& argument);

    void setCellDescriptionIndex(int index);

    int getCellDescriptionIndex() const;

    /**
     * Returns whether the subgrid represented by
     * this cell is valid. This corresponds to whether
     * the cell-description index is valid.
     */
    bool holdsSubgrid() const;
};


#endif
