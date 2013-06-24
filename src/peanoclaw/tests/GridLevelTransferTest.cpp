/*
 * GridLevelTransferTest.cpp
 *
 *  Created on: Mar 19, 2012
 *      Author: Kristof Unterweger
 */
#include "peano/applications/peanoclaw/tests/GridLevelTransferTest.h"

#include "peano/applications/peanoclaw/tests/Helper.h"
#include "peano/applications/peanoclaw/tests/PyClawTestStump.h"
#include "peano/applications/peanoclaw/tests/TestVertexEnumerator.h"

#include "peano/applications/peanoclaw/GridLevelTransfer.h"
#include "peano/applications/peanoclaw/Patch.h"
#include "peano/applications/peanoclaw/PatchOperations.h"
#include "peano/applications/peanoclaw/SpacetreeGridVertex.h"

#include "peano/kernel/heap/Heap.h"
#include "peano/kernel/spacetreegrid/SingleLevelEnumerator.h"
#include "peano/kernel/spacetreegrid/aspects/VertexStateAnalysis.h"

#include "peano/utils/Globals.h"

#include <cstring>

#include "tarch/tests/TestCaseFactory.h"
registerTest(peano::applications::peanoclaw::tests::GridLevelTransferTest)

#ifdef UseTestSpecificCompilerSettings
#pragma optimize("",off)
#endif

void peano::applications::peanoclaw::tests::GridLevelTransferTest::testAdjacentPatchIndicesForSingleRefinedCell() {
#ifdef Dim2

  //  *  - a: [-1,  3,  1,  0]
  //  *  - b: [ 4, -1,  2,  1]
  //  *  - c: [ 6,  5, -1,  3]
  //  *  - d: [ 7,  6,  4, -1]

  peano::applications::peanoclaw::SpacetreeGridVertex coarseGridVertices[FOUR_POWER_D];
  peano::kernel::spacetreegrid::SingleLevelEnumerator vertexEnumerator(
      1.0, //Coarse cell size
      0.0, //Domain offset
      0    //Coarse level
  );
  vertexEnumerator.setOffset(0.0);


  int bottomLeftDescriptionIndex = peano::kernel::heap::Heap<CellDescription>::getInstance().createData();
  CellDescription bottomLeftDescription; bottomLeftDescription.setUNewIndex(-1);
  peano::kernel::heap::Heap<CellDescription>::getInstance().getData(bottomLeftDescriptionIndex).push_back(bottomLeftDescription);

  int bottomCenterDescriptionIndex = peano::kernel::heap::Heap<CellDescription>::getInstance().createData();
  CellDescription bottomCenterDescription; bottomCenterDescription.setUNewIndex(-1);
  peano::kernel::heap::Heap<CellDescription>::getInstance().getData(bottomCenterDescriptionIndex).push_back(bottomCenterDescription);

  int bottomRightDescriptionIndex = peano::kernel::heap::Heap<CellDescription>::getInstance().createData();
  CellDescription bottomRightDescription; bottomRightDescription.setUNewIndex(-1);
  peano::kernel::heap::Heap<CellDescription>::getInstance().getData(bottomRightDescriptionIndex).push_back(bottomRightDescription);

  int centerLeftDescriptionIndex = peano::kernel::heap::Heap<CellDescription>::getInstance().createData();
  CellDescription centerLeftDescription; centerLeftDescription.setUNewIndex(-1);
  peano::kernel::heap::Heap<CellDescription>::getInstance().getData(centerLeftDescriptionIndex).push_back(centerLeftDescription);

  int refinedDescriptionIndex = peano::kernel::heap::Heap<CellDescription>::getInstance().createData();
  CellDescription refinedDescription; refinedDescription.setUNewIndex(-1);
  peano::kernel::heap::Heap<CellDescription>::getInstance().getData(refinedDescriptionIndex).push_back(refinedDescription);

  int centerRightDescriptionIndex = peano::kernel::heap::Heap<CellDescription>::getInstance().createData();
  CellDescription centerRightDescription; centerRightDescription.setUNewIndex(-1);
  peano::kernel::heap::Heap<CellDescription>::getInstance().getData(centerRightDescriptionIndex).push_back(centerRightDescription);

  int topLeftDescriptionIndex = peano::kernel::heap::Heap<CellDescription>::getInstance().createData();
  CellDescription topLeftDescription; topLeftDescription.setUNewIndex(-1);
  peano::kernel::heap::Heap<CellDescription>::getInstance().getData(topLeftDescriptionIndex).push_back(topLeftDescription);

  int topCenterDescriptionIndex = peano::kernel::heap::Heap<CellDescription>::getInstance().createData();
  CellDescription topCenterDescription; topCenterDescription.setUNewIndex(-1);
  peano::kernel::heap::Heap<CellDescription>::getInstance().getData(topCenterDescriptionIndex).push_back(topCenterDescription);

  int topRightDescriptionIndex = peano::kernel::heap::Heap<CellDescription>::getInstance().createData();
  CellDescription topRightDescription; topRightDescription.setUNewIndex(-1);
  peano::kernel::heap::Heap<CellDescription>::getInstance().getData(topRightDescriptionIndex).push_back(topRightDescription);

  //Vertex a
  coarseGridVertices[vertexEnumerator(0)].setAdjacentCellDescriptionIndex(0, refinedDescriptionIndex);
  coarseGridVertices[vertexEnumerator(0)].setAdjacentCellDescriptionIndex(1, centerLeftDescriptionIndex);
  coarseGridVertices[vertexEnumerator(0)].setAdjacentCellDescriptionIndex(2, bottomCenterDescriptionIndex);
  coarseGridVertices[vertexEnumerator(0)].setAdjacentCellDescriptionIndex(3, bottomLeftDescriptionIndex);

  //Vertex b
  coarseGridVertices[vertexEnumerator(1)].setAdjacentCellDescriptionIndex(0, centerRightDescriptionIndex);
  coarseGridVertices[vertexEnumerator(1)].setAdjacentCellDescriptionIndex(1, refinedDescriptionIndex);
  coarseGridVertices[vertexEnumerator(1)].setAdjacentCellDescriptionIndex(2, bottomRightDescriptionIndex);
  coarseGridVertices[vertexEnumerator(1)].setAdjacentCellDescriptionIndex(3, bottomCenterDescriptionIndex);

  //Vertex c
  coarseGridVertices[vertexEnumerator(2)].setAdjacentCellDescriptionIndex(0, topCenterDescriptionIndex);
  coarseGridVertices[vertexEnumerator(2)].setAdjacentCellDescriptionIndex(1, topLeftDescriptionIndex);
  coarseGridVertices[vertexEnumerator(2)].setAdjacentCellDescriptionIndex(2, refinedDescriptionIndex);
  coarseGridVertices[vertexEnumerator(2)].setAdjacentCellDescriptionIndex(3, centerLeftDescriptionIndex);

  //Vertex d
  coarseGridVertices[vertexEnumerator(3)].setAdjacentCellDescriptionIndex(0, topRightDescriptionIndex);
  coarseGridVertices[vertexEnumerator(3)].setAdjacentCellDescriptionIndex(1, topCenterDescriptionIndex);
  coarseGridVertices[vertexEnumerator(3)].setAdjacentCellDescriptionIndex(2, centerRightDescriptionIndex);
  coarseGridVertices[vertexEnumerator(3)].setAdjacentCellDescriptionIndex(3, refinedDescriptionIndex);

  //Copy to other fields
//  for(int coarseGridVertexIndex = 0; coarseGridVertexIndex < 4; coarseGridVertexIndex++) {
//    for(int adjacentIndex = 0; adjacentIndex < 4; adjacentIndex++) {
//      int value = coarseGridVertices[vertexEnumerator(coarseGridVertexIndex)].getAdjacentUNewIndex(adjacentIndex);
//      coarseGridVertices[vertexEnumerator(coarseGridVertexIndex)].setAdjacentUOldIndex(adjacentIndex, value);
//      coarseGridVertices[vertexEnumerator(coarseGridVertexIndex)].setAdjacentCellDescriptionIndex(adjacentIndex, value);
//    }
//  }

  tarch::la::Vector<DIMENSIONS, int> hangingVertexPosition;
  tarch::la::Vector<TWO_POWER_D, int> expectedAdjacentIndices;
  peano::applications::peanoclaw::SpacetreeGridVertex fineGridVertex;

  PyClawTestStump pyClaw;

  GridLevelTransfer gridLevelTransfer(false, pyClaw);

  //Hanging Vertex 0, 0
  assignList(hangingVertexPosition) = 0, 0;
  gridLevelTransfer.fillAdjacentPatchIndicesFromCoarseVertices(
      coarseGridVertices,
      vertexEnumerator,
      fineGridVertex,
      hangingVertexPosition);
  assignList(expectedAdjacentIndices) = refinedDescriptionIndex, centerLeftDescriptionIndex, bottomCenterDescriptionIndex, bottomLeftDescriptionIndex;
  for(int i = 0; i < 4; i++) {
    validateEqualsWithParams3(fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices(i), i, fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices);
  }

  //Hanging Vertex 1, 0
  assignList(hangingVertexPosition) = 1, 0;
  gridLevelTransfer.fillAdjacentPatchIndicesFromCoarseVertices(
      coarseGridVertices,
      vertexEnumerator,
      fineGridVertex,
      hangingVertexPosition);
  assignList(expectedAdjacentIndices) = refinedDescriptionIndex, refinedDescriptionIndex, bottomCenterDescriptionIndex, bottomCenterDescriptionIndex;
  for(int i = 0; i < 4; i++) {
    validateEqualsWithParams3(fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices(i), i, fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices);
  }
  //Hanging Vertex 2, 0
  assignList(hangingVertexPosition) = 2, 0;
  gridLevelTransfer.fillAdjacentPatchIndicesFromCoarseVertices(
      coarseGridVertices,
      vertexEnumerator,
      fineGridVertex,
      hangingVertexPosition);
  assignList(expectedAdjacentIndices) = refinedDescriptionIndex, refinedDescriptionIndex, bottomCenterDescriptionIndex, bottomCenterDescriptionIndex;
  for(int i = 0; i < 4; i++) {
    validateEqualsWithParams3(fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices(i), i, fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices);
  }
  //Hanging Vertex 3, 0
  assignList(hangingVertexPosition) = 3, 0;
  gridLevelTransfer.fillAdjacentPatchIndicesFromCoarseVertices(
      coarseGridVertices,
      vertexEnumerator,
      fineGridVertex,
      hangingVertexPosition);
  assignList(expectedAdjacentIndices) = centerRightDescriptionIndex, refinedDescriptionIndex, bottomRightDescriptionIndex, bottomCenterDescriptionIndex;
  for(int i = 0; i < 4; i++) {
    validateEqualsWithParams3(fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices(i), i, fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices);
  }
  //Hanging Vertex 0, 1
  assignList(hangingVertexPosition) = 0, 1;
  gridLevelTransfer.fillAdjacentPatchIndicesFromCoarseVertices(
      coarseGridVertices,
      vertexEnumerator,
      fineGridVertex,
      hangingVertexPosition);
  assignList(expectedAdjacentIndices) = refinedDescriptionIndex, centerLeftDescriptionIndex, refinedDescriptionIndex, centerLeftDescriptionIndex;
  for(int i = 0; i < 4; i++) {
    validateEqualsWithParams3(fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices(i), i, fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices);
  }
  //Hanging Vertex 3, 1
  assignList(hangingVertexPosition) = 3, 1;
  gridLevelTransfer.fillAdjacentPatchIndicesFromCoarseVertices(
      coarseGridVertices,
      vertexEnumerator,
      fineGridVertex,
      hangingVertexPosition);
  assignList(expectedAdjacentIndices) = centerRightDescriptionIndex, refinedDescriptionIndex, centerRightDescriptionIndex, refinedDescriptionIndex;
  for(int i = 0; i < 4; i++) {
    validateEqualsWithParams3(fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices(i), i, fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices);
  }
  //Hanging Vertex 0, 2
  assignList(hangingVertexPosition) = 0, 2;
  gridLevelTransfer.fillAdjacentPatchIndicesFromCoarseVertices(
      coarseGridVertices,
      vertexEnumerator,
      fineGridVertex,
      hangingVertexPosition);
  assignList(expectedAdjacentIndices) = refinedDescriptionIndex, centerLeftDescriptionIndex, refinedDescriptionIndex, centerLeftDescriptionIndex;
  for(int i = 0; i < 4; i++) {
    validateEqualsWithParams3(fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices(i), i, fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices);
  }
  //Hanging Vertex 3, 2
  assignList(hangingVertexPosition) = 3, 2;
  gridLevelTransfer.fillAdjacentPatchIndicesFromCoarseVertices(
      coarseGridVertices,
      vertexEnumerator,
      fineGridVertex,
      hangingVertexPosition);
  assignList(expectedAdjacentIndices) = centerRightDescriptionIndex, refinedDescriptionIndex, centerRightDescriptionIndex, refinedDescriptionIndex;
  for(int i = 0; i < 4; i++) {
    validateEqualsWithParams3(fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices(i), i, fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices);
  }
  //Hanging Vertex 0, 3
  assignList(hangingVertexPosition) = 0, 3;
  gridLevelTransfer.fillAdjacentPatchIndicesFromCoarseVertices(
      coarseGridVertices,
      vertexEnumerator,
      fineGridVertex,
      hangingVertexPosition);
  assignList(expectedAdjacentIndices) = topCenterDescriptionIndex, topLeftDescriptionIndex, refinedDescriptionIndex, centerLeftDescriptionIndex;
  for(int i = 0; i < 4; i++) {
    validateEqualsWithParams3(fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices(i), i, fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices);
  }
  //Hanging Vertex 1, 3
  assignList(hangingVertexPosition) = 1, 3;
  gridLevelTransfer.fillAdjacentPatchIndicesFromCoarseVertices(
      coarseGridVertices,
      vertexEnumerator,
      fineGridVertex,
      hangingVertexPosition);
  assignList(expectedAdjacentIndices) = topCenterDescriptionIndex, topCenterDescriptionIndex, refinedDescriptionIndex, refinedDescriptionIndex;
  for(int i = 0; i < 4; i++) {
    validateEqualsWithParams3(fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices(i), i, fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices);
  }
  //Hanging Vertex 2, 3
  assignList(hangingVertexPosition) = 2, 3;
  gridLevelTransfer.fillAdjacentPatchIndicesFromCoarseVertices(
      coarseGridVertices,
      vertexEnumerator,
      fineGridVertex,
      hangingVertexPosition);
  assignList(expectedAdjacentIndices) = topCenterDescriptionIndex, topCenterDescriptionIndex, refinedDescriptionIndex, refinedDescriptionIndex;
  for(int i = 0; i < 4; i++) {
    validateEqualsWithParams3(fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices(i), i, fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices);
  }
  //Hanging Vertex 3, 3
  assignList(hangingVertexPosition) = 3, 3;
  gridLevelTransfer.fillAdjacentPatchIndicesFromCoarseVertices(
      coarseGridVertices,
      vertexEnumerator,
      fineGridVertex,
      hangingVertexPosition);
  assignList(expectedAdjacentIndices) = topRightDescriptionIndex, topCenterDescriptionIndex, centerRightDescriptionIndex, refinedDescriptionIndex;
  for(int i = 0; i < 4; i++) {
    validateEqualsWithParams3(fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices(i), i, fineGridVertex.getAdjacentCellDescriptionIndex(i), expectedAdjacentIndices);
  }

  peano::kernel::heap::Heap<CellDescription>::getInstance().deleteAllData();
#endif
}

void peano::applications::peanoclaw::tests::GridLevelTransferTest::testUpdateMinimalNeighborTime() {
  peano::applications::peanoclaw::Patch finePatch;

  finePatch.resetMinimalNeighborTimeConstraint();
  validate(finePatch.getMinimalNeighborTimeConstraint() > 1e20);

  finePatch.updateMinimalNeighborTimeConstraint(1.0);

  validateEquals(finePatch.getMinimalNeighborTimeConstraint(), 1.0);
}


void peano::applications::peanoclaw::tests::GridLevelTransferTest::testOverlappingAreaWithRealOverlap() {
#ifdef Dim2
  tarch::la::Vector<DIMENSIONS, double> position1(0.5);
  tarch::la::Vector<DIMENSIONS, double> size1(5.0);

  tarch::la::Vector<DIMENSIONS, double> position2(1.0);
  position2(1) = 4.0;
  tarch::la::Vector<DIMENSIONS, double> size2(1.5);
  size2(1) = 2.0;


  validateNumericalEquals( peano::applications::peanoclaw::calculateOverlappingArea(position1, size1, position2, size2), 2.25);
#endif
}

void peano::applications::peanoclaw::tests::GridLevelTransferTest::testOverlappingAreaWithTouchingPatches() {
#ifdef Dim2
  tarch::la::Vector<DIMENSIONS, double> position1(1.0);
  tarch::la::Vector<DIMENSIONS, double> size1(1.0);

  tarch::la::Vector<DIMENSIONS, double> position2(2.0);
  position2(1) = 1.5;
  tarch::la::Vector<DIMENSIONS, double> size2(1.0);

  validateNumericalEquals( peano::applications::peanoclaw::calculateOverlappingArea(position1, size1, position2, size2), 0.0);
#endif
}

void peano::applications::peanoclaw::tests::GridLevelTransferTest::testOverlappingAreaWithoutOverlap() {
#ifdef Dim2
  tarch::la::Vector<DIMENSIONS, double> position1(-1.0);
  tarch::la::Vector<DIMENSIONS, double> size1(3.0);

  tarch::la::Vector<DIMENSIONS, double> position2(0.0);
  position2(1) = 3;
  tarch::la::Vector<DIMENSIONS, double> size2(5.0);

  validateNumericalEquals( peano::applications::peanoclaw::calculateOverlappingArea(position1, size1, position2, size2), 0.0);
#endif
}

void peano::applications::peanoclaw::tests::GridLevelTransferTest::testRestrictionToVirtualPatch() {
#ifdef Dim2
  typedef peano::applications::peanoclaw::records::CellDescription CellDescription;

  //CellDescriptions
  int unknownsPerSubcell = 1;
  int ghostlayerWidth = 2;
  int coarseSubdivisionFactor = 2;
  int fineSubdivisionFactor = 3;
  peano::kernel::heap::Heap<CellDescription>::getInstance().deleteAllData();

  //Create patch for neighboring coarse cell
  tarch::la::Vector<DIMENSIONS, double> coarsePosition;
  assignList(coarsePosition) = 1.0, 0;
  Patch neighboringCoarsePatch = createPatch(
    unknownsPerSubcell,
    0,   //Aux fields per subcell
    coarseSubdivisionFactor,
    ghostlayerWidth,
    coarsePosition,
    1.0, //coarse size
    0,   //level
    1.0, //time
    4.0/3.0, //timestep size
    1.0  //Minimal neighbor time
    );

  //Create refined cell that gets the virtual patch
  tarch::la::Vector<DIMENSIONS, double> virtualPatchPosition;
  assignList(virtualPatchPosition) = 1.0, 0;
  int virtualCellDescriptionIndex = peano::kernel::heap::Heap<CellDescription>::getInstance().createData();
  std::vector<CellDescription>& virtualCellDescriptions = peano::kernel::heap::Heap<CellDescription>::getInstance().getData(virtualCellDescriptionIndex);
  virtualCellDescriptions.push_back(CellDescription());
  CellDescription& virtualCellDescription = peano::kernel::heap::Heap<CellDescription>::getInstance().getData(virtualCellDescriptionIndex).at(0);
  virtualCellDescription.setSubdivisionFactor(coarseSubdivisionFactor);
  virtualCellDescription.setGhostLayerWidth(ghostlayerWidth);
  virtualCellDescription.setUnknownsPerSubcell(unknownsPerSubcell);
  virtualCellDescription.setSize(1.0);
  virtualCellDescription.setPosition(0.0);
  virtualCellDescription.setLevel(0);
  virtualCellDescription.setTime(0.0);
  virtualCellDescription.setTimestepSize(1.0+4.0/3.0);
  virtualCellDescription.setMaximumFineGridTime(0.0);
  virtualCellDescription.setMinimumFineGridTimestep(1.0+4.0/3.0);
  virtualCellDescription.setMinimalNeighborTime(0.0);
  virtualCellDescription.setMaximalNeighborTimestep(1.0+4.0/3.0);
  virtualCellDescription.setUNewIndex(-1);
  virtualCellDescription.setUOldIndex(-1);
  virtualCellDescription.setAuxIndex(-1);
  virtualCellDescription.setCellDescriptionIndex(virtualCellDescriptionIndex);
  virtualCellDescription.setIsVirtual(false);

  //Create cell description for fine leaf cell
  tarch::la::Vector<DIMENSIONS, double> finePosition(1.0/3.0);
  Patch finePatch = createPatch(
    unknownsPerSubcell,
    0,   //Aux fields per subcell
    fineSubdivisionFactor,
    ghostlayerWidth,
    finePosition,
    1.0/3.0, //coarse size
    1,   //level
    1.0, //time
    1.0, //timestep size
    1.0  //Minimal neighbor time
  );

  tarch::la::Vector<DIMENSIONS, int> subcellIndex;
  //uNew
  assignList(subcellIndex) = 0, 0;
  finePatch.setValueUNew(subcellIndex, 0, 0.0);
  assignList(subcellIndex) = 0, 1;
  finePatch.setValueUNew(subcellIndex, 0, 1.0);
  assignList(subcellIndex) = 0, 2;
  finePatch.setValueUNew(subcellIndex, 0, 2.0);
  assignList(subcellIndex) = 1, 0;
  finePatch.setValueUNew(subcellIndex, 0, 3.0);
  assignList(subcellIndex) = 1, 1;
  finePatch.setValueUNew(subcellIndex, 0, 4.0);
  assignList(subcellIndex) = 1, 2;
  finePatch.setValueUNew(subcellIndex, 0, 5.0);
  assignList(subcellIndex) = 2, 0;
  finePatch.setValueUNew(subcellIndex, 0, 6.0);
  assignList(subcellIndex) = 2, 1;
  finePatch.setValueUNew(subcellIndex, 0, 7.0);
  assignList(subcellIndex) = 2, 2;
  finePatch.setValueUNew(subcellIndex, 0, 8.0);

  //uOld
  assignList(subcellIndex) = 0, 0;
  finePatch.setValueUOld(subcellIndex, 0, 9.0);
  assignList(subcellIndex) = 0, 1;
  finePatch.setValueUOld(subcellIndex, 0, 8.0);
  assignList(subcellIndex) = 0, 2;
  finePatch.setValueUOld(subcellIndex, 0, 7.0);
  assignList(subcellIndex) = 1, 0;
  finePatch.setValueUOld(subcellIndex, 0, 6.0);
  assignList(subcellIndex) = 1, 1;
  finePatch.setValueUOld(subcellIndex, 0, 5.0);
  assignList(subcellIndex) = 1, 2;
  finePatch.setValueUOld(subcellIndex, 0, 4.0);
  assignList(subcellIndex) = 2, 0;
  finePatch.setValueUOld(subcellIndex, 0, 3.0);
  assignList(subcellIndex) = 2, 1;
  finePatch.setValueUOld(subcellIndex, 0, 2.0);
  assignList(subcellIndex) = 2, 2;
  finePatch.setValueUOld(subcellIndex, 0, 1.0);

  //Coarse vertices
  SpacetreeGridVertex coarseVertices[TWO_POWER_D];
  memset(coarseVertices, 0, sizeof(SpacetreeGridVertex) * TWO_POWER_D);
  for(int vertexIndex = 0; vertexIndex < TWO_POWER_D; vertexIndex++) {
    for(int cellIndex = 0; cellIndex < TWO_POWER_D; cellIndex++) {
      coarseVertices[vertexIndex].setAdjacentCellDescriptionIndex(cellIndex, -1);
    }
    coarseVertices[vertexIndex].refine();
    assertion(coarseVertices[vertexIndex].getRefinementControl() == peano::applications::peanoclaw::records::SpacetreeGridVertex::RefinementTriggered);
    coarseVertices[vertexIndex].switchRefinementTriggeredToRefining();
    assertion(coarseVertices[vertexIndex].getRefinementControl() == peano::applications::peanoclaw::records::SpacetreeGridVertex::Refining);
  }
  coarseVertices[1].setAdjacentCellDescriptionIndex(0, neighboringCoarsePatch.getCellDescriptionIndex());
  coarseVertices[1].setAdjacentCellDescriptionIndex(1, virtualCellDescriptionIndex);

  coarseVertices[3].setAdjacentCellDescriptionIndex(2, neighboringCoarsePatch.getCellDescriptionIndex());
  coarseVertices[3].setAdjacentCellDescriptionIndex(3, virtualCellDescriptionIndex);

  //GridLevelTransfer
  PyClawTestStump pyClaw;
  peano::applications::peanoclaw::GridLevelTransfer gridLevelTransfer(false, pyClaw);
  TestVertexEnumerator enumerator(1.0);

  //Virtual patch
  Patch virtualPatch(
    virtualCellDescription
  );
  virtualPatch.switchToVirtual();
  virtualPatch.switchToNonVirtual();
  validate(!virtualPatch.isVirtual());
  validate(!virtualPatch.isLeaf());
  assertion(peano::kernel::spacetreegrid::aspects::VertexStateAnalysis::doesOneVertexCarryRefinementFlag
        (
          coarseVertices,
          enumerator,
          peano::applications::peanoclaw::records::SpacetreeGridVertex::Refining
        ));

  gridLevelTransfer.stepDown(-1, virtualPatch, coarseVertices, enumerator, false);

  validateEquals(gridLevelTransfer._virtualPatchDescriptionIndices.size(), 1);

  //Recreate virtual patch to cache uNew and uOld in the Patch object
  virtualPatch = Patch(virtualCellDescription);
  validate(virtualPatch.isVirtual());

  gridLevelTransfer.stepUp(-1, finePatch, true, 0, enumerator);
  gridLevelTransfer.stepUp(-1, virtualPatch, false, coarseVertices, enumerator);

  //Recreate virtual patch to assure that the created data arrays are involved
  virtualPatch = Patch (
    virtualCellDescription
  );

  //Check results
  double areaFraction = (1.0/9.0)*(1.0/9.0) / (1.0/2.0) / (1.0/2.0);
  assignList(subcellIndex) = -2, 0;
  validateNumericalEquals(neighboringCoarsePatch.getValueUOld(subcellIndex, 0), areaFraction * (-1.0/3.0*9.0 + 1.0/2.0*(4.0/3.0 - 1.0/3.0*8.0) + 1.0/2.0*(4.0/3.0*3.0 - 1.0/3.0*6.0) + 1.0/4.0*(4.0/3.0*4.0 - 1.0/3.0*5.0)));
  assignList(subcellIndex) = -2, 1;
  validateNumericalEquals(neighboringCoarsePatch.getValueUOld(subcellIndex, 0), areaFraction * (4.0/3.0*2.0 - 1.0/3.0*7.0 + 1.0/2.0*(4.0/3.0 - 1.0/3.0*8.0) + 1.0/2.0*(4.0/3.0*5.0 - 1.0/3.0*4.0) + 1.0/4.0*(4.0/3.0*4.0 - 1.0/3.0*5.0)));
  assignList(subcellIndex) = -1, 0;
  validateNumericalEquals(neighboringCoarsePatch.getValueUOld(subcellIndex, 0), areaFraction * (4.0/3.0*6.0 - 1.0/3.0*3.0 + 1.0/2.0*(4.0/3.0*3.0 - 1.0/3.0*6.0) + 1.0/2.0*(4.0/3.0*7.0 - 1.0/3.0*2.0) + 1.0/4.0*(4.0/3.0*4.0 - 1.0/3.0*5.0)));
  assignList(subcellIndex) = -1, 1;
  validateNumericalEquals(neighboringCoarsePatch.getValueUOld(subcellIndex, 0), areaFraction * (4.0/3.0*8.0 - 1.0/3.0 + 1.0/2.0*(4.0/3.0*5.0 - 1.0/3.0*4.0) + 1.0/2.0*(4.0/3.0*7.0 - 1.0/3.0*2.0) + 1.0/4.0*(4.0/3.0*4.0 - 1.0/3.0*5.0)));

  //Clean up
  peano::kernel::heap::Heap<CellDescription>::getInstance().deleteAllData();
  peano::kernel::heap::Heap<Data>::getInstance().deleteAllData();
#endif
}

peano::applications::peanoclaw::tests::GridLevelTransferTest::GridLevelTransferTest(){
}
peano::applications::peanoclaw::tests::GridLevelTransferTest::~GridLevelTransferTest(){
}

void peano::applications::peanoclaw::tests::GridLevelTransferTest::run() {
  //TODO unterweg debug
//  testMethod(testAdjacentPatchIndicesForSingleRefinedCell);
//  testMethod(testOverlappingAreaWithRealOverlap);
//  testMethod(testOverlappingAreaWithTouchingPatches);
//  testMethod(testOverlappingAreaWithoutOverlap);
//  testMethod(testRestrictionToVirtualPatch);
}

void peano::applications::peanoclaw::tests::GridLevelTransferTest::setUp() {

}


#ifdef UseTestSpecificCompilerSettings
#pragma optimize("",on)
#endif
