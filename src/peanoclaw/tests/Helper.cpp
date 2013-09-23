/*
 * Helper.cpp
 *
 *  Created on: Jun 10, 2012
 *      Author: kristof
 */
#include "peanoclaw/tests/Helper.h"

#include "peanoclaw/Patch.h"
#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/records/Data.h"

#include "peano/heap/Heap.h"

peanoclaw::Patch peanoclaw::tests::createPatch(
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
  int cellDescriptionIndex = peano::heap::PlainHeap<peanoclaw::records::CellDescription>::getInstance().createData();
  std::vector<peanoclaw::records::CellDescription>& cellDescriptions = peano::heap::PlainHeap<peanoclaw::records::CellDescription>::getInstance().getData(cellDescriptionIndex);
  cellDescriptions.push_back(peanoclaw::records::CellDescription());
  peanoclaw::records::CellDescription& cellDescription = peano::heap::PlainHeap<peanoclaw::records::CellDescription>::getInstance().getData(cellDescriptionIndex).at(0);
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

//  peano::heap::PlainHeap<peanoclaw::records::Data>& heap = peano::heap::PlainHeap<peanoclaw::records::Data>::getInstance();

  //uNew array
  int uNewIndex = peano::heap::PlainHeap<peanoclaw::records::Data>::getInstance().createData();
  std::vector<peanoclaw::records::Data>& uNew = peano::heap::PlainHeap<peanoclaw::records::Data>::getInstance().getData(uNewIndex);

  int uNewArraySize = tarch::la::aPowI(DIMENSIONS, subdivisionFactor)*unknownsPerSubcell;
  int uOldArraySize = tarch::la::aPowI(DIMENSIONS, (subdivisionFactor+2*ghostlayerWidth)) * unknownsPerSubcell;
  int auxArraySize = tarch::la::aPowI(DIMENSIONS, subdivisionFactor + 2*ghostlayerWidth) * auxFieldsPerSubcell;

  for(int i = 0; i < uNewArraySize + uOldArraySize + auxArraySize; i++) {
    peanoclaw::records::Data data;
    data.setU(0.0);
    uNew.push_back(data);
  }
  cellDescription.setUNewIndex(uNewIndex);

  //uOld array
//  if(!virtualPatch) {
//    int uOldIndex = heap.createData();
//    std::vector<peanoclaw::records::Data>& uOld = heap.getData(uOldIndex);
//    for(int i = 0; i < tarch::la::aPowI(DIMENSIONS, (subdivisionFactor+2*ghostlayerWidth)) * unknownsPerSubcell; i++) {
//      peanoclaw::records::Data data;
//      data.setU(0.0);
//      uOld.push_back(data);
//    }
//    cellDescription.setUOldIndex(uOldIndex);
//  } else {
//    cellDescription.setUOldIndex(-1);
//  }

  //Initialise aux array
//  if(auxFieldsPerSubcell > 0) {
//    cellDescription.setAuxIndex(peano::heap::PlainHeap<peanoclaw::records::Data>::getInstance().createData());
//    std::vector<peanoclaw::records::Data>& auxArray =
//        peano::heap::PlainHeap<peanoclaw::records::Data>::getInstance().getData(cellDescription.getAuxIndex());
//    for(int i = 0; i < tarch::la::aPowI(DIMENSIONS, subdivisionFactor + 2*ghostlayerWidth) * auxFieldsPerSubcell; i++) {
//      auxArray.push_back(-1.0);
//    }
//  } else {
//    cellDescription.setAuxIndex(-1);
//  }

  return Patch(cellDescription);
}



