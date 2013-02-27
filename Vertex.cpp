#include "peanoclaw/Vertex.h"
#include "peano/utils/Loop.h"
#include "peano/grid/Checkpoint.h"
#include "tarch/la/WrappedVector.h"
#include "peano/heap/Heap.h"

#include "peanoclaw/Patch.h"
#include "peanoclaw/PyClaw.h"
#include "peanoclaw/GhostLayerCompositor.h"

peanoclaw::Vertex::Vertex():
  Base() { 
  // @todo Insert your code here
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
//  logInfo("", __LINE__ << "Accessing index " << _vertexData.getVertexDescriptionIndex());
//  assertionMsg(!isHangingNode(), "Storing adjacent indices on hanging nodes is not allowed.");
//  peano::heap::Heap<VertexDescription>::getInstance().getData(_vertexData.getVertexDescriptionIndex())[0].setIndicesOfAdjacentCellDescriptions(cellIndex, cellDescriptionIndex);
  _vertexData.setIndicesOfAdjacentCellDescriptions(cellIndex, cellDescriptionIndex);
}

int peanoclaw::Vertex::getAdjacentCellDescriptionIndex(int cellIndex) const {
//  logInfo("", __LINE__ << "Accessing index " << _vertexData.getVertexDescriptionIndex());
//  assertionMsg(!isHangingNode(), "Storing adjacent indices on hanging nodes is not allowed.");
//  return peano::heap::Heap<VertexDescription>::getInstance().getData(_vertexData.getVertexDescriptionIndex())[0].getIndicesOfAdjacentCellDescriptions(cellIndex);
  return _vertexData.getIndicesOfAdjacentCellDescriptions(cellIndex);
}

void peanoclaw::Vertex::fillAdjacentGhostLayers(
  int level,
  bool useDimensionalSplitting,
  peanoclaw::PyClaw& pyClaw,
  int destinationPatch
) const {

  //TODO unterweg Debug
  #ifdef Debug
  bool plotVertex = //tarch::la::equals(getX()(0), 19.0/27.0) && tarch::la::equals(getX()(1), 16.0/27.0)
      //|| tarch::la::equals(getX()(0), 1.0/3.0) && tarch::la::equals(getX()(1), 5.0/9.0)
      //|| tarch::la::equals(getX()(0), 1.0/3.0) && tarch::la::equals(getX()(1), 4.0/9.0)
      //|| tarch::la::equals(getX()(0), 4.0/9.0) && tarch::la::equals(getX()(1), 4.0/9.0)
      //||tarch::la::equals(getX()(0), 1.0/3.0) && tarch::la::equals(getX()(1), 14.0/27.0)
//      tarch::la::equals(getX()(0), 3.0/9.0) && tarch::la::equals(getX()(1), 7.0/9.0)
      false
  ;
  #endif

  //Fill ghost layers of adjacent cells
  //Get adjacent cell descriptions
  CellDescription* cellDescriptions[TWO_POWER_D];
  for(int cellIndex = 0; cellIndex < TWO_POWER_D; cellIndex++) {
    if(getAdjacentCellDescriptionIndex(cellIndex) != -1) {
      cellDescriptions[cellIndex] = &peano::heap::Heap<CellDescription>::getInstance().getData(getAdjacentCellDescriptionIndex(cellIndex)).at(0);
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

  #ifdef Debug
  if(plotVertex) {
    std::cout << "Filling vertex (hanging=" << isHangingNode() << ") at " << getX() << " on level " << getLevel() << std::endl;

    for (int i = 0; i < TWO_POWER_D; i++) {
      std::cout << " cellDescription(" << i << ")=" << getAdjacentCellDescriptionIndex(i);
    }
    std::cout << std::endl;

    for (int i = 0; i < TWO_POWER_D; i++) {
      std::cout << "Patch " << i << " " << patches[i].toString() << ": "
          << std::endl << patches[i].toStringUNew() << std::endl << patches[i].toStringUOldWithGhostLayer();
    }
  }
  #endif

  GhostLayerCompositor ghostLayerCompositor(patches, level, pyClaw, useDimensionalSplitting);

  ghostLayerCompositor.updateNeighborTimes();
  ghostLayerCompositor.fillGhostLayers(destinationPatch);
  ghostLayerCompositor.updateGhostlayerBounds();

  //TODO unterweg Debug
  #ifdef Debug
  if(plotVertex
  ) {
    for (int i = 0; i < TWO_POWER_D; i++) {
      std::cout << "Patch " << i << " at " << patches[i].getPosition() << " of size " << patches[i].getSize() << ": "
          << std::endl << patches[i].toStringUNew() << std::endl << patches[i].toStringUOldWithGhostLayer();
    }
  }
  #endif
}

void peanoclaw::Vertex::applyCoarseGridCorrection(
  peanoclaw::PyClaw& pyClaw
) const {
  //Fill ghost layers of adjacent cells
  CellDescription* cellDescriptions[TWO_POWER_D];
  for(int cellIndex = 0; cellIndex < TWO_POWER_D; cellIndex++) {
    if(getAdjacentCellDescriptionIndex(cellIndex) != -1) {
      cellDescriptions[cellIndex] = &peano::heap::Heap<CellDescription>::getInstance().getData(getAdjacentCellDescriptionIndex(cellIndex)).at(0);
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
  GhostLayerCompositor ghostLayerCompositor(patches, 0, pyClaw, false);
  ghostLayerCompositor.applyCoarseGridCorrection();
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
