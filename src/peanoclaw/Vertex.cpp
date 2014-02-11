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
  //_vertexData.setWasCreatedInThisIteration(true);
  _vertexData.setAgeInGridIterations(0);
  _vertexData.setShouldRefine(false);
  #ifdef Parallel
  _vertexData.setAdjacentRanksChanged(false);
  #endif
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
      patches[cellIndexScalar] = Patch(
        *cellDescriptions[cellIndexScalar]
      );
    }
  enddforx

  //TODO unterweg Debug
  #ifdef Debug
  bool plotVertex = false;
  plotVertex =
      tarch::la::equals(position(0), 2.0/9.0)
      && tarch::la::equals(position(1), 1.0/9.0)
      //&& tarch::la::equals(position(2), 1.0/9.0)
      && level == 3
  ;

  if(plotVertex) {
    std::cerr << "Filling vertex ("
        #ifdef Parallel
        << "rank=" << tarch::parallel::Node::getInstance().getRank() << ","
        #endif
        <<"hanging=" << isHangingNode() << ") at " << position << " on level " << level
        <<", destinationPatch=" << destinationPatch << std::endl;

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

//  ghostLayerCompositor.updateNeighborTimes();
  ghostLayerCompositor.fillGhostLayersAndUpdateNeighborTimes(destinationPatch);
  ghostLayerCompositor.updateGhostlayerBounds();

  //TODO unterweg Debug
  #ifdef Debug
  if(plotVertex
  ) {
    std::cerr << "After filling:" << std::endl;
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


void peanoclaw::Vertex::setAllSubcellEraseVetos() {
  for(int i = 0; i < TWO_POWER_D; i++) {
    setSubcellEraseVeto(i);
  }
}

bool peanoclaw::Vertex::shouldErase() const {
  bool eraseAllSubcells = true;
  for(int i = 0; i < TWO_POWER_D; i++) {
    eraseAllSubcells &= !_vertexData.getAdjacentSubcellsEraseVeto(i);
  }

  return eraseAllSubcells;
}

void peanoclaw::Vertex::increaseAgeInGridIterations() {
  _vertexData.setAgeInGridIterations(_vertexData.getAgeInGridIterations() + 1);
}

int peanoclaw::Vertex::getAgeInGridIterations() const {
  return _vertexData.getAgeInGridIterations();
}

void peanoclaw::Vertex::resetAgeInGridIterations() {
  _vertexData.setAgeInGridIterations(0);
}

void peanoclaw::Vertex::mergeWithNeighbor(const Vertex& neighbor) {
  //Merge vertices
  for(int i = 0; i < TWO_POWER_D; i++) {
    _vertexData.setAdjacentSubcellsEraseVeto(i, _vertexData.getAdjacentSubcellsEraseVeto(i) || neighbor._vertexData.getAdjacentSubcellsEraseVeto(i));
  }
  setShouldRefine(shouldRefine() || neighbor.shouldRefine());
}

void peanoclaw::Vertex::setAdjacentRanksInFormerGridIteration(const tarch::la::Vector<TWO_POWER_D, int>& adjacentRanksInFormerGridIteration) {
  _vertexData.setAdjacentRanksInFormerIteration(adjacentRanksInFormerGridIteration);
}

tarch::la::Vector<TWO_POWER_D, int> peanoclaw::Vertex::getAdjacentRanksInFormerGridIteration() const {
  return _vertexData.getAdjacentRanksInFormerIteration();
}

void peanoclaw::Vertex::setWhetherAdjacentRanksChanged(bool adjacentRanksChanged) {
  _vertexData.setAdjacentRanksChanged(adjacentRanksChanged);
}

bool peanoclaw::Vertex::wereAdjacentRanksChanged() const {
  return _vertexData.getAdjacentRanksChanged();
}


