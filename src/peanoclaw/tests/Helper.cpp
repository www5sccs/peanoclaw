/*
 * Helper.cpp
 *
 *  Created on: Jun 10, 2012
 *      Author: kristof
 */
#include "peano/applications/peanoclaw/tests/Helper.h"

#include "peano/applications/peanoclaw/Patch.h"
#include "peano/applications/peanoclaw/records/CellDescription.h"
#include "peano/applications/peanoclaw/records/Data.h"

#include "peano/kernel/heap/Heap.h"

peano::applications::peanoclaw::Patch peano::applications::peanoclaw::tests::createPatch(
  int unknownsPerSubcell,
  int auxFieldsPerSubcell,
  int subdivisionFactor,
  int ghostlayerWidth,
  tarch::la::Vector<DIMENSIONS, double> position,
  tarch::la::Vector<DIMENSIONS, double> size,
  int level,
  double time,
  double timestepSize,
  double minimalNeighborTime,
  bool virtualPatch
) {
  int cellDescriptionIndex = peano::kernel::heap::Heap<peano::applications::peanoclaw::records::CellDescription>::getInstance().createData();
  std::vector<peano::applications::peanoclaw::records::CellDescription>& cellDescriptions = peano::kernel::heap::Heap<peano::applications::peanoclaw::records::CellDescription>::getInstance().getData(cellDescriptionIndex);
  cellDescriptions.push_back(peano::applications::peanoclaw::records::CellDescription());
  peano::applications::peanoclaw::records::CellDescription& cellDescription = peano::kernel::heap::Heap<peano::applications::peanoclaw::records::CellDescription>::getInstance().getData(cellDescriptionIndex).at(0);
  cellDescription.setSubdivisionFactor(subdivisionFactor);
  cellDescription.setGhostLayerWidth(ghostlayerWidth);
  cellDescription.setUnknownsPerSubcell(unknownsPerSubcell);
  cellDescription.setSize(size);
  cellDescription.setPosition(position);
  cellDescription.setLevel(level);
  cellDescription.setTime(time);
  cellDescription.setTimestepSize(timestepSize);
  cellDescription.setMinimalNeighborTime(minimalNeighborTime);
  cellDescription.setCellDescriptionIndex(cellDescriptionIndex);
  cellDescription.setIsVirtual(false);

  peano::kernel::heap::Heap<peano::applications::peanoclaw::records::Data>& heap = peano::kernel::heap::Heap<peano::applications::peanoclaw::records::Data>::getInstance();

  //uNew array
  int uNewIndex = heap.createData();
  std::vector<peano::applications::peanoclaw::records::Data>& uNew = heap.getData(uNewIndex);
  for(int i = 0; i < tarch::la::aPowI(DIMENSIONS, subdivisionFactor)*unknownsPerSubcell; i++) {
    uNew.push_back(peano::applications::peanoclaw::records::Data());
  }
  cellDescription.setUNewIndex(uNewIndex);

  //uOld array
  if(!virtualPatch) {
    int uOldIndex = heap.createData();
    std::vector<peano::applications::peanoclaw::records::Data>& uOld = heap.getData(uOldIndex);
    for(int i = 0; i < tarch::la::aPowI(DIMENSIONS, (subdivisionFactor+2*ghostlayerWidth)) * unknownsPerSubcell; i++) {
      uOld.push_back(peano::applications::peanoclaw::records::Data());
    }
    cellDescription.setUOldIndex(uOldIndex);
  } else {
    cellDescription.setUOldIndex(-1);
  }

  //Initialise aux array
  if(auxFieldsPerSubcell > 0) {
    cellDescription.setAuxIndex(peano::kernel::heap::Heap<peano::applications::peanoclaw::records::Data>::getInstance().createData());
    std::vector<peano::applications::peanoclaw::records::Data>& auxArray =
        peano::kernel::heap::Heap<peano::applications::peanoclaw::records::Data>::getInstance().getData(cellDescription.getAuxIndex());
    for(int i = 0; i < tarch::la::aPowI(DIMENSIONS, subdivisionFactor + 2*ghostlayerWidth) * auxFieldsPerSubcell; i++) {
      auxArray.push_back(-1.0);
    }
  } else {
    cellDescription.setAuxIndex(-1);
  }

  return Patch(cellDescription);
}



