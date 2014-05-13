/*
 * SubgridLevelContainer.cpp
 *
 *  Created on: May 9, 2014
 *      Author: kristof
 */
#include "peanoclaw/grid/SubgridLevelContainer.h"

#include "peanoclaw/Cell.h"

#include "peano/utils/Loop.h"

peanoclaw::grid::SubgridLevelContainer::SubgridLevelContainer()
 : _index(0) {
}

void peanoclaw::grid::SubgridLevelContainer::setFirstLevel(
  peanoclaw::Cell&                     cell,
  peanoclaw::Vertex * const            vertices,
  const peano::grid::VertexEnumerator& verticesEnumerator
) {
  _firstLevelSubgrid = peanoclaw::Patch(cell.getCellDescriptionIndex());

  cell.setSubgrid(_firstLevelSubgrid);

  for(int i = 0; i < TWO_POWER_D; i++) {
    vertices[verticesEnumerator(i)].setAdjacentSubgrid(i, _firstLevelSubgrid);
  }
}

void peanoclaw::grid::SubgridLevelContainer::addNewLevel(
  peanoclaw::Cell * const cells,
  peanoclaw::Vertex * const vertices,
  const peano::grid::VertexEnumerator& verticesEnumerator
) {
  assertion(_index < 100);
  _index++;

  //TODO unterweg debug
//  for(int i = 0; i < FOUR_POWER_D; i++) {
//    std::cout << "vertex" << i << ": ";
//    for(int j = 0; j < TWO_POWER_D; j++) {
//      std::cout << vertices[verticesEnumerator(i)].getAdjacentCellDescriptionIndex(j) << " ";
//    }
//    std::cout << std::endl;
//  }
//  std::cout << "--------------------" << std::endl;
//  //for(int i = 0; i < THREE_POWER_D; i++) {
//  dfor3(cellIndex)
//    std::cout << "cell " << cellIndex << " index=" << cells[verticesEnumerator.cell(cellIndex)].getCellDescriptionIndex() << std::endl;
//  enddforx

  int subgridIndexScalar = 0;
  dfor(subgridIndex, 5) {

    //TODO unterweg debug
//    std::cout << "\nsubgridIndex=" << subgridIndex << std::endl;

    int setCellDescriptionIndex = -1;

    //Create subgrid based on cell
    tarch::la::Vector<DIMENSIONS, int> cellIndex = subgridIndex-tarch::la::Vector<DIMENSIONS,int>(1);
    if(tarch::la::allGreaterEquals(cellIndex, tarch::la::Vector<DIMENSIONS,int>(0))
            && tarch::la::allGreater(tarch::la::Vector<DIMENSIONS,int>(3), cellIndex)) {
      setCellDescriptionIndex = cells[verticesEnumerator.cell(cellIndex)].getCellDescriptionIndex();
      _subgrids[_index*FIVE_POWER_D + subgridIndexScalar] = peanoclaw::Patch(setCellDescriptionIndex);
      cells[verticesEnumerator.cell(cellIndex)].setSubgrid(_subgrids[_index*FIVE_POWER_D + subgridIndexScalar]);

      //TODO unterweg debug
//      std::cout << "\tSubgrid from cell[" << cellIndex << "]: " << setCellDescriptionIndex << std::endl;
    }


    int localVertexIndexScalar = 0;
    //Loop over adjacent vertices for current subgrid
    dfor(localVertexIndex, 2) {
      tarch::la::Vector<DIMENSIONS,int> vertexIndex = subgridIndex + localVertexIndex;

      //TODO unterweg debug
//      std::cout << "\tvertex: " << localVertexIndex;

      if(tarch::la::allGreater(vertexIndex, tarch::la::Vector<DIMENSIONS,int>(0))
        && tarch::la::allGreater(tarch::la::Vector<DIMENSIONS,int>(5), vertexIndex)) {

        //Get current vertex
        tarch::la::Vector<DIMENSIONS,int> reducedVertexIndex = vertexIndex - tarch::la::Vector<DIMENSIONS,int>(1);
        int vertexIndex = peano::utils::dLinearisedWithoutLookup(reducedVertexIndex, 4);
        Vertex& vertex = vertices[verticesEnumerator(vertexIndex)];

        int cellDescriptionIndex = vertex.getAdjacentCellDescriptionIndex(localVertexIndexScalar);

        //TODO unterweg debug
//        std::cout << " (ok), index=" << cellDescriptionIndex << std::endl;

        if(cellDescriptionIndex != -1) {
          assertion4(
            cellDescriptionIndex == setCellDescriptionIndex || setCellDescriptionIndex == -1,
            cellDescriptionIndex,
            setCellDescriptionIndex,
            subgridIndex,
            localVertexIndex
          );

          //Create subgrid based on vertex
          if(setCellDescriptionIndex == -1) {
            assertion1(!tarch::la::allGreaterEquals(subgridIndex, 1) || tarch::la::oneGreater(subgridIndex, 3), subgridIndex);
            setCellDescriptionIndex = cellDescriptionIndex;
            _subgrids[_index*FIVE_POWER_D + subgridIndexScalar] = peanoclaw::Patch(cellDescriptionIndex);

            //TODO unterweg debug
//            std::cout << "\t\tsubgrid from vertex: " << setCellDescriptionIndex << std::endl;
          }

          vertex.setAdjacentSubgrid(localVertexIndexScalar, _subgrids[_index*FIVE_POWER_D + subgridIndexScalar]);
        }
      } else {
        //TODO unterweg debug
//        std::cout << std::endl;
      }

      localVertexIndexScalar++;
    }

    subgridIndexScalar++;
  }
}

void peanoclaw::grid::SubgridLevelContainer::removeCurrentLevel() {
  assertion(_index > 0);
  _index--;
}


int peanoclaw::grid::SubgridLevelContainer::getCurrentLevel() const {
  return _index;
}



