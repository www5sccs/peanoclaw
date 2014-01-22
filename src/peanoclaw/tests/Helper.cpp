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

#include "peanoclaw/Heap.h"

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
  int cellDescriptionIndex = CellDescriptionHeap::getInstance().createData();
  std::vector<peanoclaw::records::CellDescription>& cellDescriptions = CellDescriptionHeap::getInstance().getData(cellDescriptionIndex);
  cellDescriptions.push_back(peanoclaw::records::CellDescription());
  peanoclaw::records::CellDescription& cellDescription = CellDescriptionHeap::getInstance().getData(cellDescriptionIndex).at(0);
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

  //uNew array
  int uNewIndex = DataHeap::getInstance().createData();
  std::vector<peanoclaw::records::Data>& uNew = DataHeap::getInstance().getData(uNewIndex);

  int uNewArraySize = tarch::la::aPowI(DIMENSIONS, subdivisionFactor)*unknownsPerSubcell;
  int uOldArraySize = tarch::la::aPowI(DIMENSIONS, (subdivisionFactor+2*ghostlayerWidth)) * unknownsPerSubcell;
  int auxArraySize = tarch::la::aPowI(DIMENSIONS, subdivisionFactor + 2*ghostlayerWidth) * auxFieldsPerSubcell;

  for(int i = 0; i < uNewArraySize + uOldArraySize + auxArraySize; i++) {
    peanoclaw::records::Data data;
    data.setU(0.0);
    uNew.push_back(data);
  }
  cellDescription.setUIndex(uNewIndex);

  return Patch(cellDescription);
}



