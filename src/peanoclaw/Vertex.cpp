#include "Vertex.h"
#include "peano/utils/Loop.h"
#include "peano/grid/Checkpoint.h"
#include "peanoclaw/Heap.h"

#include "Patch.h"
#include "Numerics.h"
#include "interSubgridCommunication/GhostLayerCompositor.h"

tarch::logging::Log peanoclaw::Vertex::_log("peanoclaw::Vertex");

peanoclaw::Vertex::Vertex():
  Base() {
  _vertexData.setIndicesOfAdjacentCellDescriptions(-1);
  _vertexData.setWasCreatedInThisIteration(true);
}


peanoclaw::Vertex::Vertex(const Base::DoNotCallStandardConstructor& value):
  Base(value) { 
  // Please do not insert anything here
}


peanoclaw::Vertex::Vertex(const Base::PersistentVertex& argument):
  Base(argument) {
  // @todo Insert your code here
}

void peanoclaw::Vertex::setAdjacentCellDescriptionIndex(int cellIndex, int cellDescriptionIndex) {
  _vertexData.setIndicesOfAdjacentCellDescriptions(cellIndex, cellDescriptionIndex);
}

void peanoclaw::Vertex::setAdjacentCellDescriptionIndexInPeanoOrder(
  int cellIndex,
  int cellDescriptionIndex
) {
  setAdjacentCellDescriptionIndex(TWO_POWER_D - cellIndex - 1, cellDescriptionIndex);
}

int peanoclaw::Vertex::getAdjacentCellDescriptionIndex(int cellIndex) const {
  return _vertexData.getIndicesOfAdjacentCellDescriptions(cellIndex);
}

int peanoclaw::Vertex::getAdjacentCellDescriptionIndexInPeanoOrder(int cellIndex) const {
  return getAdjacentCellDescriptionIndex(TWO_POWER_D - cellIndex - 1);
}

void peanoclaw::Vertex::fillAdjacentGhostLayers(
  int level,
  bool useDimensionalSplitting,
  peanoclaw::Numerics& numerics,
  const tarch::la::Vector<DIMENSIONS, double>& position,
  int destinationPatch
) const {

  //Fill ghost layers of adjacent cells
  //Get adjacent cell descriptions
  CellDescription* cellDescriptions[TWO_POWER_D];
  for(int cellIndex = 0; cellIndex < TWO_POWER_D; cellIndex++) {
    if(getAdjacentCellDescriptionIndex(cellIndex) != -1) {
      cellDescriptions[cellIndex] = &CellDescriptionHeap::getInstance().getData(getAdjacentCellDescriptionIndex(cellIndex)).at(0);
    }
  }

  // Prepare ghostlayer
  Patch patches[TWO_POWER_D];
  dfor2(cellIndex)
    if(getAdjacentCellDescriptionIndex(cellIndexScalar) != -1) {

//      assertion4(DataHeap::getInstance().isValidIndex(cellDescriptions[cellIndexScalar]->getUNewIndex()), cellIndexScalar, getX(), level, cellDescriptions[cellIndexScalar]->getUNewIndex());
//      assertion4(DataHeap::getInstance().isValidIndex(cellDescriptions[cellIndexScalar]->getUOldIndex()), cellIndexScalar, getX(), level, cellDescriptions[cellIndexScalar]->getUOldIndex());

      patches[cellIndexScalar] = Patch(
        *cellDescriptions[cellIndexScalar]
      );
    }
  enddforx

  //TODO unterweg Debug
  #ifdef Debug
  bool plotVertex = false;
//  plotVertex =
//      tarch::la::equals(position(0), 48.0/81.0)
//      && tarch::la::equals(position(1), 7.0/9.0)
//      && level == 4
//  ;

  if(plotVertex) {
    std::cerr << "Filling vertex ("
        #ifdef Parallel
        << "rank=" << tarch::parallel::Node::getInstance().getRank() << ","
        #endif
        <<"hanging=" << isHangingNode() << ") at " << position << " on level " << level << std::endl;

    for (int i = 0; i < TWO_POWER_D; i++) {
      std::cerr << " cellDescription(" << i << ")=" << getAdjacentCellDescriptionIndex(i);
    }
    std::cerr << std::endl;

    for (int i = 0; i < TWO_POWER_D; i++) {
      std::cerr << "Patch " << i << " " << patches[i].toString() << ": "
          << std::endl << patches[i].toStringUNew() << std::endl << patches[i].toStringUOldWithGhostLayer();
    }
  }
  #endif

  interSubgridCommunication::GhostLayerCompositor ghostLayerCompositor(patches, level, numerics, useDimensionalSplitting);

  ghostLayerCompositor.updateNeighborTimes();
  ghostLayerCompositor.fillGhostLayers(destinationPatch);
  ghostLayerCompositor.updateGhostlayerBounds();

  //TODO unterweg Debug
  #ifdef Debug
  if(plotVertex
  ) {
    for (int i = 0; i < TWO_POWER_D; i++) {
      if(patches[i].isValid()) {
        std::cerr << "Patch " << i << " at " << patches[i].getPosition() << " of size " << patches[i].getSize() << ": "
            << std::endl << patches[i].toStringUNew() << std::endl << patches[i].toStringUOldWithGhostLayer();
      }
    }
  }
  #endif
}

void peanoclaw::Vertex::applyFluxCorrection(
  peanoclaw::Numerics& numerics
) const {
  //Fill ghost layers of adjacent cells
  CellDescription* cellDescriptions[TWO_POWER_D];
  for(int cellIndex = 0; cellIndex < TWO_POWER_D; cellIndex++) {
    if(getAdjacentCellDescriptionIndex(cellIndex) != -1) {
      cellDescriptions[cellIndex] = &CellDescriptionHeap::getInstance().getData(getAdjacentCellDescriptionIndex(cellIndex)).at(0);
    }
  }

  // Prepare ghostlayer
  Patch patches[TWO_POWER_D];
  dfor2(cellIndex)
    if(getAdjacentCellDescriptionIndex(cellIndexScalar) != -1) {
      patches[cellIndexScalar] = Patch(
        *cellDescriptions[cellIndexScalar]
      );
    }
  enddforx

  //Apply coarse grid correction
  interSubgridCommunication::GhostLayerCompositor ghostLayerCompositor(patches, 0, numerics, false);
  ghostLayerCompositor.applyFluxCorrection();
}

void peanoclaw::Vertex::setShouldRefine(bool shouldRefine) {
  _vertexData.setShouldRefine(shouldRefine);
}

bool peanoclaw::Vertex::shouldRefine() const {
  return _vertexData.getShouldRefine();
}

void peanoclaw::Vertex::resetSubcellsEraseVeto() {
  for(int i = 0; i < TWO_POWER_D; i++) {
    _vertexData.setAdjacentSubcellsEraseVeto(i, false);
  }
}

void peanoclaw::Vertex::setSubcellEraseVeto(
  int cellIndex
) {
  _vertexData.setAdjacentSubcellsEraseVeto(cellIndex, true);
}

bool peanoclaw::Vertex::shouldErase() const {
  bool eraseAllSubcells = true;
  for(int i = 0; i < TWO_POWER_D; i++) {
    eraseAllSubcells &= !_vertexData.getAdjacentSubcellsEraseVeto(i);
  }

  return eraseAllSubcells;
}

void peanoclaw::Vertex::setWasCreatedInThisIteration(bool flag) {
  _vertexData.setWasCreatedInThisIteration(flag);
}

bool peanoclaw::Vertex::wasCreatedInThisIteration() const {
  return _vertexData.getWasCreatedInThisIteration();
}
