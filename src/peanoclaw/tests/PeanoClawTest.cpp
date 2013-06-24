#include "peano/applications/peanoclaw/tests/PeanoClawTest.h"

#include "peano/utils/Globals.h"

#include "peano/applications/peanoclaw/SpacetreeGridVertex.h"
#include "peano/applications/peanoclaw/SpacetreeGridCell.h"

#include "tarch/tests/TestCaseFactory.h"
registerTest(peano::applications::peanoclaw::tests::PeanoClawTest)


#ifdef UseTestSpecificCompilerSettings
#pragma optimize("",off)
#endif

 
peano::applications::peanoclaw::tests::PeanoClawTest::PeanoClawTest():
  tarch::tests::TestCase( "peano::applications::peanoclaw::tests::PeanoClawTest" ) {
}


peano::applications::peanoclaw::tests::PeanoClawTest::~PeanoClawTest() {
}


void peano::applications::peanoclaw::tests::PeanoClawTest::setUp() {
  // @todo If you have to configure your global test object, please do this 
  //       here. Typically this operation remains empty.
}


void peano::applications::peanoclaw::tests::PeanoClawTest::run() {
  testMethod( testVertexAdjacentIndices );
  testMethod( testCellIndices );
}


void peano::applications::peanoclaw::tests::PeanoClawTest::testVertexAdjacentIndices() {

  SpacetreeGridVertex vertex;
  for(int i = 0; i < TWO_POWER_D; i++) {
    vertex.setAdjacentCellDescriptionIndex(i, 100 + i);
//    vertex.setAdjacentUNewIndex(i, 200 + i);
//    vertex.setAdjacentUOldIndex(i, 300 + i);
  }

  for(int i = 0; i < TWO_POWER_D; i++) {
    validateEquals(vertex.getAdjacentCellDescriptionIndex(i), 100 + i);
//    validateEquals(vertex.getAdjacentUNewIndex(i), 200 + i);
//    validateEquals(vertex.getAdjacentUOldIndex(i), 300 + i);
  }
}

void peano::applications::peanoclaw::tests::PeanoClawTest::testCellIndices() {

  SpacetreeGridCell cell;
  cell.setCellDescriptionIndex(100);
//  cell.setUNewIndex(200);
//  cell.setUOldIndex(300);

  validateEquals(cell.getCellDescriptionIndex(), 100);
//  validateEquals(cell.getUNewIndex(), 200);
//  validateEquals(cell.getUOldIndex(), 300);
}

#ifdef UseTestSpecificCompilerSettings
#pragma optimize("",on)
#endif
