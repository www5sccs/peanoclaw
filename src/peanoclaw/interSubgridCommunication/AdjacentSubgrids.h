/*
 * AdjacentSubgrids.h
 *
 *  Created on: Jul 24, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ADJACENTSUBGRIDS_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ADJACENTSUBGRIDS_H_

#include "peanoclaw/Vertex.h"
#include "peanoclaw/records/VertexDescription.h"

#include "tarch/la/Vector.h"
#include "tarch/la/VectorCompare.h"

#include <map>


#define DIMENSIONS_PLUS_ONE (DIMENSIONS+1)

namespace peanoclaw {
  namespace interSubgridCommunication {
    class AdjacentSubgrids;
  }
}

/**
 * Encapsulation of the adjacent subgrids/peano cells stored within
 * a vertex. This class provides the functionality to keep this
 * adjacency information consistent.
 */
class peanoclaw::interSubgridCommunication::AdjacentSubgrids {

  public:
    typedef peanoclaw::records::VertexDescription VertexDescription;
    typedef std::map<tarch::la::Vector<DIMENSIONS_PLUS_ONE,double>, VertexDescription, tarch::la::VectorCompare<DIMENSIONS_PLUS_ONE> > VertexMap;

  private:
    peanoclaw::Vertex&                    _vertex;
    VertexMap&                            _vertexMap;
    tarch::la::Vector<DIMENSIONS, double> _position;
    int                                   _level;

  public:
    AdjacentSubgrids(
      peanoclaw::Vertex& vertex,
      VertexMap& vertexMap,
      tarch::la::Vector<DIMENSIONS, double> position,
      int level
    );

    /**
     * Called when a subgrid adjacent to the corresponding vertex of
     * this AdjacentSubgrids object has been created.
     *
     * @param cellDescriptionIndex The global index referencing the cell
     * description of the newly created subgrid.
     * @param subgridIndex The local index that describes the position of the
     * created subgrid with respect to the current vertex.
     */
    void createdAdjacentSubgrid(int cellDescriptionIndex, int subgridIndex);

    /**
     * Called when a persistent vertex is destroyed, since it may be turned
     * to a hanging vertex.
     */
    void convertPersistentToHangingVertex();
};

#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_ADJACENTSUBGRIDS_H_ */
