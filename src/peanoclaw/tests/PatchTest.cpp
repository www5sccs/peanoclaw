/*
 * PatchTest.cpp
 *
 *  Created on: Feb 15, 2012
 *      Author: Kristof Unterweger
 */
#include "peanoclaw/tests/PatchTest.h"

#include "peanoclaw/ParallelSubgrid.h"
#include "peanoclaw/geometry/Region.h"
#include "peanoclaw/Patch.h"
#include "peanoclaw/Vertex.h"
#include "peanoclaw/tests/Helper.h"
#include "peanoclaw/tests/TestVertexEnumerator.h"
#include "peanoclaw/interSubgridCommunication/aspects/AdjacentSubgrids.h"

#include "peano/utils/Globals.h"

#include "tarch/tests/TestCaseFactory.h"
registerTest(peanoclaw::tests::PatchTest)

using namespace peanoclaw::geometry;


#ifdef UseTestSpecificCompilerSettings
#pragma optimize("",off)
#endif


peanoclaw::tests::PatchTest::PatchTest():
  tarch::tests::TestCase( "peanoclaw::tests::PeanoClawTest" ) {
}


peanoclaw::tests::PatchTest::~PatchTest() {
}


void peanoclaw::tests::PatchTest::setUp() {
}


void peanoclaw::tests::PatchTest::run() {
  testMethod( testFillingOfUNewArray );
  testMethod( testFillingOfUOldArray );
  testMethod( testInvalidPatch );
  testMethod( testCoarsePatchTimeInterval );
  testMethod( testCountingOfAdjacentParallelSubgrids );
  testMethod( testCountingOfAdjacentParallelSubgridsFourNeighboringRanks );
  testMethod( testSettingOverlapOfRemoteGhostlayer );
  testMethod( testNumberOfAdjacentManifolds );
  testMethod( testAdjacentManifolds );
  testMethod( testManifolds );
  testMethod( testOverlapOfRemoteGhostlayers );
  testMethod( testOverlapOfRemoteGhostlayers2 );
  testMethod( testOverlapOfRemoteGhostlayers3D );
}

void peanoclaw::tests::PatchTest::testFillingOfUNewArray() {

  Patch patch = createPatch(
    2,   //Unknowns per subcell
    0,   //Aux fields per subcell
    0,   //Aux fields per subcell
    3,   //Subdivision factor
    1,   //Ghostlayer width
    0.0, //Position
    1.0, //Size
    0,   //Level
    0.0, //Current time
    0.0, //Timestep size
    0.0  //Minimal neighbor time
  );

  #ifdef Dim2
  peanoclaw::grid::SubgridAccessor accessor = patch.getAccessor();
  for(int unknown = 0; unknown < 2; unknown++) {
    for(int x = 0; x < 3; x++) {
      for(int y = 0; y < 3; y++) {
        tarch::la::Vector<DIMENSIONS, int> subcellIndex;
        assignList(subcellIndex) = x, y;
        accessor.setValueUNew(subcellIndex, unknown, unknown * 9 + x * 3 + y);
      }
    }
  }
  #endif

  //TODO unterweg fixen!
//  double uNewArray[3*3*2];
//  patch.fillUNewArray(uNewArray);
//
//  for(int i = 0; i < 18; i++) {
//    validateNumericalEquals(uNewArray[i], i);
//  }
}

void peanoclaw::tests::PatchTest::testFillingOfUOldArray() {
  Patch patch = createPatch(
    2,       //Unknowns per subcell
    0,       //Aux fields per subcell
    0,       //Aux fields per subcell
    3,       //Subdivision factor
    1,       //Ghostlayer width
    0.0,     //Position
    1.0,     //Size
    0,       //Level
    0.0,     //Current time
    0.0,     //Timestep size
    0.0      //Minimal neighbor time
  );

  #ifdef Dim2
  peanoclaw::grid::SubgridAccessor accessor = patch.getAccessor();
  int counter = 0;
  for(int unknown = 0; unknown < 2; unknown++) {
    for(int x = -1; x < 3 + 1; x++) {
      for(int y = -1; y < 3 + 1; y++) {
        tarch::la::Vector<DIMENSIONS, int> subcellIndex;
        assignList(subcellIndex) = x, y;
        accessor.setValueUOld(subcellIndex, unknown, counter++);
      }
    }
  }
  #endif

  //TODO unterweg fixen!
//  double uOldArray[5*5*2];
//  patch.fillUOldWithGhostLayerArray(uOldArray);
//
//  for(int i = 0; i < 50; i++) {
//    validateNumericalEquals(uOldArray[i], i);
//  }
}

void peanoclaw::tests::PatchTest::testInvalidPatch() {
  Patch patch;
  validate(!patch.isValid());
}

void peanoclaw::tests::PatchTest::testCoarsePatchTimeInterval() {
  peanoclaw::Patch coarsePatch = createPatch(
    2,       //Unknowns per subcell
    0,       //Aux fields per subcell
    0,       //Aux fields per subcell
    3,       //Subdivision factor
    1,       //Ghostlayer width
    0.0,     //Position
    1.0,     //Size
    0,       //Level
    0.0,     //Current time
    0.0,     //Timestep size
    0.0      //Minimal neighbor time
  );
  peanoclaw::Patch finePatches[THREE_POWER_D];

  for(int i = 0; i < THREE_POWER_D; i++) {
    finePatches[i] = createPatch(
      2,       //Unknowns per subcell
      0,       //Aux fields per subcell
      0,       //Aux fields per subcell
      3,       //Subdivision factor
      1,       //Ghostlayer width
      0.0,     //Position
      1.0/3.0, //Size
      1,       //Level
      0.0,     //Current time
      0.0,     //Timestep size
      0.0      //Minimal neighbor time
    );
  }

  //Prepare fine patches
  finePatches[0].getTimeIntervals().setCurrentTime(1.0);
  finePatches[0].getTimeIntervals().setTimestepSize(2.0);

  finePatches[1].getTimeIntervals().setCurrentTime(1.0);
  finePatches[1].getTimeIntervals().setTimestepSize(2.0);

  finePatches[2].getTimeIntervals().setCurrentTime(1.0);
  finePatches[2].getTimeIntervals().setTimestepSize(3.0);

  finePatches[3].getTimeIntervals().setCurrentTime(0.0);
  finePatches[3].getTimeIntervals().setTimestepSize(3.0);

  finePatches[4].getTimeIntervals().setCurrentTime(1.0);
  finePatches[4].getTimeIntervals().setTimestepSize(1.7);

  finePatches[5].getTimeIntervals().setCurrentTime(0.0);
  finePatches[5].getTimeIntervals().setTimestepSize(4.0);

  finePatches[6].getTimeIntervals().setCurrentTime(0.0);
  finePatches[6].getTimeIntervals().setTimestepSize(3.0);

  finePatches[7].getTimeIntervals().setCurrentTime(1.0);
  finePatches[7].getTimeIntervals().setTimestepSize(2.0);

  finePatches[8].getTimeIntervals().setCurrentTime(2.5);
  finePatches[8].getTimeIntervals().setTimestepSize(1.5);

  for(int i = 9; i < THREE_POWER_D; i++) {
    finePatches[i].getTimeIntervals().setCurrentTime(0.0);
    finePatches[i].getTimeIntervals().setTimestepSize(3.0);
  }

  //execute methods
  coarsePatch.getTimeIntervals().resetMinimalFineGridTimeInterval();
  for(int i = 0; i < THREE_POWER_D; i++) {
    coarsePatch.getTimeIntervals().updateMinimalFineGridTimeInterval(finePatches[i].getTimeIntervals().getCurrentTime(), finePatches[i].getTimeIntervals().getTimestepSize());
  }
  coarsePatch.switchValuesAndTimeIntervalToMinimalFineGridTimeInterval();

  //Check
  validateNumericalEquals(coarsePatch.getTimeIntervals().getCurrentTime(), 2.5);
  validateNumericalEquals(coarsePatch.getTimeIntervals().getTimestepSize(), 0.2);
}

void peanoclaw::tests::PatchTest::testCountingOfAdjacentParallelSubgrids() {
  #if defined(Parallel) && defined(Dim2)
  Vertex vertices[TWO_POWER_D];
  TestVertexEnumerator enumerator(1.0);

  for(int i = 0; i < TWO_POWER_D; i++) {
    vertices[i].switchToNonhangingNode();
  }

  vertices[0].setAdjacentRank(0, 0);
  vertices[0].setAdjacentRank(1, 0);
  vertices[0].setAdjacentRank(2, 0);
  vertices[0].setAdjacentRank(3, 0);

  vertices[1].setAdjacentRank(0, 0);
  vertices[1].setAdjacentRank(1, 0);
  vertices[1].setAdjacentRank(2, 0);
  vertices[1].setAdjacentRank(3, 0);

  vertices[2].setAdjacentRank(0, 0);
  vertices[2].setAdjacentRank(1, 0);
  vertices[2].setAdjacentRank(2, 1);
  vertices[2].setAdjacentRank(3, 1);

  vertices[3].setAdjacentRank(0, 0);
  vertices[3].setAdjacentRank(1, 0);
  vertices[3].setAdjacentRank(2, 1);
  vertices[3].setAdjacentRank(3, 1);

  peanoclaw::records::CellDescription cellDescription;

  //Test
  peanoclaw::ParallelSubgrid parallelSubgrid(cellDescription);
  parallelSubgrid.countNumberOfAdjacentParallelSubgrids(vertices, enumerator);

  validateEqualsWithParams1(parallelSubgrid.getAdjacentRanks()(0), 0, parallelSubgrid.getAdjacentRanks());
  validateEqualsWithParams1(parallelSubgrid.getAdjacentRanks()(1), 0, parallelSubgrid.getAdjacentRanks());
  validateEqualsWithParams1(parallelSubgrid.getAdjacentRanks()(2), 0, parallelSubgrid.getAdjacentRanks());
  validateEqualsWithParams1(parallelSubgrid.getAdjacentRanks()(3), 0, parallelSubgrid.getAdjacentRanks());
  validateEqualsWithParams1(parallelSubgrid.getAdjacentRanks()(4), 0, parallelSubgrid.getAdjacentRanks());
  validateEqualsWithParams1(parallelSubgrid.getAdjacentRanks()(5),  1, parallelSubgrid.getAdjacentRanks());
  validateEqualsWithParams1(parallelSubgrid.getAdjacentRanks()(6),  1, parallelSubgrid.getAdjacentRanks());
  validateEqualsWithParams1(parallelSubgrid.getAdjacentRanks()(7),  1, parallelSubgrid.getAdjacentRanks());

  validateEqualsWithParams1(parallelSubgrid.getNumberOfSharedAdjacentVertices()(0), 4, parallelSubgrid.getNumberOfSharedAdjacentVertices());
  validateEqualsWithParams1(parallelSubgrid.getNumberOfSharedAdjacentVertices()(1), 0, parallelSubgrid.getNumberOfSharedAdjacentVertices());
  validateEqualsWithParams1(parallelSubgrid.getNumberOfSharedAdjacentVertices()(2), 0, parallelSubgrid.getNumberOfSharedAdjacentVertices());
  validateEqualsWithParams1(parallelSubgrid.getNumberOfSharedAdjacentVertices()(3), 0, parallelSubgrid.getNumberOfSharedAdjacentVertices());
  validateEqualsWithParams1(parallelSubgrid.getNumberOfSharedAdjacentVertices()(4), 0, parallelSubgrid.getNumberOfSharedAdjacentVertices());
  validateEqualsWithParams1(parallelSubgrid.getNumberOfSharedAdjacentVertices()(5), 2, parallelSubgrid.getNumberOfSharedAdjacentVertices());
  validateEqualsWithParams1(parallelSubgrid.getNumberOfSharedAdjacentVertices()(6), 0, parallelSubgrid.getNumberOfSharedAdjacentVertices());
  validateEqualsWithParams1(parallelSubgrid.getNumberOfSharedAdjacentVertices()(7), 0, parallelSubgrid.getNumberOfSharedAdjacentVertices());
  #endif
}

void peanoclaw::tests::PatchTest::testCountingOfAdjacentParallelSubgridsFourNeighboringRanks() {
  #if defined(Parallel) && defined(Dim2)
  Vertex vertices[TWO_POWER_D];
  TestVertexEnumerator enumerator(1.0);

  vertices[0].setAdjacentRank(0, 3);
  vertices[0].setAdjacentRank(1, 3);
  vertices[0].setAdjacentRank(2, 4);
  vertices[0].setAdjacentRank(3, 0);

  vertices[1].setAdjacentRank(0, 3);
  vertices[1].setAdjacentRank(1, 3);
  vertices[1].setAdjacentRank(2, 0);
  vertices[1].setAdjacentRank(3, 2);

  vertices[2].setAdjacentRank(0, 4);
  vertices[2].setAdjacentRank(1, 0);
  vertices[2].setAdjacentRank(2, 1);
  vertices[2].setAdjacentRank(3, 1);

  vertices[3].setAdjacentRank(0, 0);
  vertices[3].setAdjacentRank(1, 2);
  vertices[3].setAdjacentRank(2, 1);
  vertices[3].setAdjacentRank(3, 1);

  peanoclaw::records::CellDescription cellDescription;

  //Test
  peanoclaw::ParallelSubgrid parallelSubgrid(cellDescription);
  parallelSubgrid.countNumberOfAdjacentParallelSubgrids(vertices, enumerator);

  validateEquals(parallelSubgrid.getAdjacentRanks()(0), 3);
  validateEquals(parallelSubgrid.getAdjacentRanks()(1), 3);
  validateEquals(parallelSubgrid.getAdjacentRanks()(2), 3);
  validateEquals(parallelSubgrid.getAdjacentRanks()(3), 4);
  validateEquals(parallelSubgrid.getAdjacentRanks()(4), 2);
  validateEquals(parallelSubgrid.getAdjacentRanks()(5), 1);
  validateEquals(parallelSubgrid.getAdjacentRanks()(6), 1);
  validateEquals(parallelSubgrid.getAdjacentRanks()(7), 1);
  validateEqualsWithParams1(parallelSubgrid.getNumberOfSharedAdjacentVertices()(0), 2, parallelSubgrid.getNumberOfSharedAdjacentVertices());
  validateEqualsWithParams1(parallelSubgrid.getNumberOfSharedAdjacentVertices()(1), 0, parallelSubgrid.getNumberOfSharedAdjacentVertices());
  validateEqualsWithParams2(parallelSubgrid.getNumberOfSharedAdjacentVertices()(2), 0, parallelSubgrid.getAdjacentRanks(), parallelSubgrid.getNumberOfSharedAdjacentVertices());
  validateEqualsWithParams1(parallelSubgrid.getNumberOfSharedAdjacentVertices()(3), 2, parallelSubgrid.getNumberOfSharedAdjacentVertices());
  validateEqualsWithParams1(parallelSubgrid.getNumberOfSharedAdjacentVertices()(4), 2, parallelSubgrid.getNumberOfSharedAdjacentVertices());
  validateEqualsWithParams1(parallelSubgrid.getNumberOfSharedAdjacentVertices()(5), 2, parallelSubgrid.getNumberOfSharedAdjacentVertices());
  validateEqualsWithParams1(parallelSubgrid.getNumberOfSharedAdjacentVertices()(6), 0, parallelSubgrid.getNumberOfSharedAdjacentVertices());
  validateEqualsWithParams1(parallelSubgrid.getNumberOfSharedAdjacentVertices()(7), 0, parallelSubgrid.getNumberOfSharedAdjacentVertices());
  #endif
}

void peanoclaw::tests::PatchTest::testSettingOverlapOfRemoteGhostlayer() {
  #if defined(Dim2) && defined(Parallel)
  peanoclaw::Vertex fineGridVertex;
  fineGridVertex.setAdjacentRank(0, 1);
  fineGridVertex.setAdjacentRank(1, 2);
  fineGridVertex.setAdjacentRank(2, 0);
  fineGridVertex.setAdjacentRank(3, 0);

  Patch subgridRank0_1 = createPatch(
    1, 1, 0,
    10, //Subdivision Factor
    1,  //Ghostlayer width
    tarch::la::Vector<DIMENSIONS, double>(0.0), //Position
    tarch::la::Vector<DIMENSIONS, double>(1.0), //Size
    0, 0.0, 0.0, 0.0, false
  );
  tarch::la::Vector<DIMENSIONS, double> rank0_2Position;
  assignList(rank0_2Position) = -1.0, 0.0;
  Patch subgridRank0_2 = createPatch(
    1, 1, 0,
    10, //Subdivision Factor
    1,  //Ghostlayer width
    rank0_2Position,
    tarch::la::Vector<DIMENSIONS, double>(1.0), //Size
    0, 0.0, 0.0, 0.0, false
  );
  Patch subgridRank1 = createPatch(
    1, 1, 0,
    10, //Subdivision Factor
    2,  //Ghostlayer width
    tarch::la::Vector<DIMENSIONS, double>(-1.0), //Position
    tarch::la::Vector<DIMENSIONS, double>(1.0), //Size
    0, 0.0, 0.0, 0.0, false
  );
  tarch::la::Vector<DIMENSIONS, double> rank2Position;
  assignList(rank2Position) = 0.0, -1.0;
  Patch subgridRank2 = createPatch(
    1, 1, 0,
    10, //Subdivision Factor
    1,  //Ghostlayer width
    rank2Position,
    tarch::la::Vector<DIMENSIONS, double>(1.0), //Size
    0, 0.0, 0.0, 0.0, false
  );

  fineGridVertex.setAdjacentCellDescriptionIndexInPeanoOrder(0, subgridRank1.getCellDescriptionIndex());
  fineGridVertex.setAdjacentCellDescriptionIndexInPeanoOrder(1, subgridRank2.getCellDescriptionIndex());
  fineGridVertex.setAdjacentCellDescriptionIndexInPeanoOrder(2, subgridRank0_2.getCellDescriptionIndex());
  fineGridVertex.setAdjacentCellDescriptionIndexInPeanoOrder(3, subgridRank0_1.getCellDescriptionIndex());


  interSubgridCommunication::aspects::AdjacentSubgrids::VertexMap vertexMap;
  interSubgridCommunication::aspects::AdjacentSubgrids adjacentSubgrids(
    fineGridVertex,
    vertexMap,
    tarch::la::Vector<DIMENSIONS, double>(0),
    0
  );

  adjacentSubgrids.setOverlapOfRemoteGhostlayers(0);

  ParallelSubgrid parallelSubgridRank0(subgridRank0_1);
  validateEquals(parallelSubgridRank0.getOverlapOfRemoteGhostlayer(0), 2);
  validateEquals(parallelSubgridRank0.getOverlapOfRemoteGhostlayer(1), 1);
  validateEquals(parallelSubgridRank0.getOverlapOfRemoteGhostlayer(2), 0);
  validateEquals(parallelSubgridRank0.getOverlapOfRemoteGhostlayer(3), 0);
  validateEquals(parallelSubgridRank0.getOverlapOfRemoteGhostlayer(4), 0);
  validateEquals(parallelSubgridRank0.getOverlapOfRemoteGhostlayer(5), 0);
  validateEquals(parallelSubgridRank0.getOverlapOfRemoteGhostlayer(6), 0);
  validateEquals(parallelSubgridRank0.getOverlapOfRemoteGhostlayer(7), 0);
  #endif
}

void peanoclaw::tests::PatchTest::testManifolds() {
  #ifdef Parallel
  tarch::la::Vector<DIMENSIONS, int> manifoldPosition;
  #if defined(Dim2)
  validateEquals(Region::getNumberOfManifolds(0), 4);
  validateEquals(Region::getNumberOfManifolds(1), 4);

  //Vertices
  assignList(manifoldPosition) = -1, -1;
  validateEqualsWithParams1(Region::getManifold(0, 0), manifoldPosition, Region::getManifold(0, 0));

  assignList(manifoldPosition) =  1, -1;
  validateEqualsWithParams1(Region::getManifold(0, 1), manifoldPosition, Region::getManifold(0, 1));

  assignList(manifoldPosition) = -1,  1;
  validateEqualsWithParams1(Region::getManifold(0, 2), manifoldPosition, Region::getManifold(0, 2));

  assignList(manifoldPosition) =  1,  1;
  validateEqualsWithParams1(Region::getManifold(0, 3), manifoldPosition, Region::getManifold(0, 3));

  //Edges
  assignList(manifoldPosition) = -1,  0;
  validateEqualsWithParams1(Region::getManifold(1, 0), manifoldPosition, Region::getManifold(1, 0));

  assignList(manifoldPosition) =  1,  0;
  validateEqualsWithParams1(Region::getManifold(1, 1), manifoldPosition, Region::getManifold(1, 1));

  assignList(manifoldPosition) =  0, -1;
  validateEqualsWithParams1(Region::getManifold(1, 2), manifoldPosition, Region::getManifold(1, 2));

  assignList(manifoldPosition) =  0,  1;
  validateEqualsWithParams1(Region::getManifold(1, 3), manifoldPosition, Region::getManifold(1, 3));
  #elif Dim3
  validateEqualsWithParams1(Region::getNumberOfManifolds(0),  8, Region::getNumberOfManifolds(0));
  validateEqualsWithParams1(Region::getNumberOfManifolds(1), 12, Region::getNumberOfManifolds(1));
  validateEqualsWithParams1(Region::getNumberOfManifolds(2),  6, Region::getNumberOfManifolds(2));

  //Vertices
  assignList(manifoldPosition) = -1, -1, -1;
  validateEqualsWithParams1(Region::getManifold(0, 0), manifoldPosition, Region::getManifold(0, 0));

  assignList(manifoldPosition) =  1, -1, -1;
  validateEqualsWithParams1(Region::getManifold(0, 1), manifoldPosition, Region::getManifold(0, 1));

  assignList(manifoldPosition) = -1,  1, -1;
  validateEqualsWithParams1(Region::getManifold(0, 2), manifoldPosition, Region::getManifold(0, 2));

  assignList(manifoldPosition) =  1,  1, -1;
  validateEqualsWithParams1(Region::getManifold(0, 3), manifoldPosition, Region::getManifold(0, 3));

  assignList(manifoldPosition) = -1, -1,  1;
  validateEqualsWithParams1(Region::getManifold(0, 4), manifoldPosition, Region::getManifold(0, 4));

  assignList(manifoldPosition) =  1, -1,  1;
  validateEqualsWithParams1(Region::getManifold(0, 5), manifoldPosition, Region::getManifold(0, 5));

  assignList(manifoldPosition) = -1,  1,  1;
  validateEqualsWithParams1(Region::getManifold(0, 6), manifoldPosition, Region::getManifold(0, 6));

  assignList(manifoldPosition) =  1,  1,  1;
  validateEqualsWithParams1(Region::getManifold(0, 7), manifoldPosition, Region::getManifold(0, 7));

  //Edges
  assignList(manifoldPosition) = -1, -1,  0;
  validateEqualsWithParams1(Region::getManifold(1, 0), manifoldPosition, Region::getManifold(1, 0));

  assignList(manifoldPosition) =  1, -1,  0;
  validateEqualsWithParams1(Region::getManifold(1, 1), manifoldPosition, Region::getManifold(1, 1));

  assignList(manifoldPosition) = -1,  1,  0;
  validateEqualsWithParams1(Region::getManifold(1, 2), manifoldPosition, Region::getManifold(1, 2));

  assignList(manifoldPosition) =  1,  1,  0;
  validateEqualsWithParams1(Region::getManifold(1, 3), manifoldPosition, Region::getManifold(1, 3));

  assignList(manifoldPosition) = -1,  0, -1;
  validateEqualsWithParams1(Region::getManifold(1, 4), manifoldPosition, Region::getManifold(1, 4));

  assignList(manifoldPosition) =  1,  0, -1;
  validateEqualsWithParams1(Region::getManifold(1, 5), manifoldPosition, Region::getManifold(1, 5));

  assignList(manifoldPosition) =  -1,  0,  1;
  validateEqualsWithParams1(Region::getManifold(1, 6), manifoldPosition, Region::getManifold(1, 6));

  assignList(manifoldPosition) =   1,  0,  1;
  validateEqualsWithParams1(Region::getManifold(1, 7), manifoldPosition, Region::getManifold(1, 7));

  assignList(manifoldPosition) =  0, -1, -1;
  validateEqualsWithParams1(Region::getManifold(1, 8), manifoldPosition, Region::getManifold(1, 8));

  assignList(manifoldPosition) =  0,  1, -1;
  validateEqualsWithParams1(Region::getManifold(1, 9), manifoldPosition, Region::getManifold(1, 9));

  assignList(manifoldPosition) =  0, -1,  1;
  validateEqualsWithParams1(Region::getManifold(1, 10), manifoldPosition, Region::getManifold(1, 10));

  assignList(manifoldPosition) =  0,  1,  1;
  validateEqualsWithParams1(Region::getManifold(1, 11), manifoldPosition, Region::getManifold(1, 11));

  //Faces
  assignList(manifoldPosition) = -1,  0,  0;
  validateEqualsWithParams1(Region::getManifold(2, 0), manifoldPosition, Region::getManifold(2, 0));

  assignList(manifoldPosition) =  1,  0,  0;
  validateEqualsWithParams1(Region::getManifold(2, 1), manifoldPosition, Region::getManifold(2, 1));

  assignList(manifoldPosition) =  0, -1,  0;
  validateEqualsWithParams1(Region::getManifold(2, 2), manifoldPosition, Region::getManifold(2, 2));

  assignList(manifoldPosition) =  0,  1,  0;
  validateEqualsWithParams1(Region::getManifold(2, 3), manifoldPosition, Region::getManifold(2, 3));

  assignList(manifoldPosition) =  0,  0, -1;
  validateEqualsWithParams1(Region::getManifold(2, 4), manifoldPosition, Region::getManifold(2, 4));

  assignList(manifoldPosition) =  0,  0,  1;
  validateEqualsWithParams1(Region::getManifold(2, 5), manifoldPosition, Region::getManifold(2, 5));
  #endif
  #endif
}

void peanoclaw::tests::PatchTest::testNumberOfAdjacentManifolds() {
  #ifdef Parallel
  tarch::la::Vector<DIMENSIONS, int> manifoldPosition;
  #ifdef Dim2
  assignList(manifoldPosition) = -1, -1;
  validateEquals(Region::getNumberOfAdjacentManifolds(manifoldPosition, 0, 1), 2);

  assignList(manifoldPosition) = -1, 0;
  validateEquals(Region::getNumberOfAdjacentManifolds(manifoldPosition, 1, 0), 2);
  #elif Dim3
  assignList(manifoldPosition) = -1, -1, -1;
  validateEquals(Region::getNumberOfAdjacentManifolds(manifoldPosition, 0, 1), 3);

  assignList(manifoldPosition) = -1, -1, -1;
  validateEquals(Region::getNumberOfAdjacentManifolds(manifoldPosition, 0, 2), 3);

  assignList(manifoldPosition) = -1, -1, 0;
  validateEquals(Region::getNumberOfAdjacentManifolds(manifoldPosition, 1, 2), 2);

  assignList(manifoldPosition) = -1, 0, 0;
  validateEquals(Region::getNumberOfAdjacentManifolds(manifoldPosition, 2, 1), 4);

  assignList(manifoldPosition) = -1, 0, 0;
  validateEquals(Region::getNumberOfAdjacentManifolds(manifoldPosition, 2, 0), 4);

  assignList(manifoldPosition) = -1, -1, 0;
  validateEquals(Region::getNumberOfAdjacentManifolds(manifoldPosition, 1, 0), 2);
  #endif
  #endif
}

void peanoclaw::tests::PatchTest::testAdjacentManifolds() {
  tarch::la::Vector<DIMENSIONS, int> manifoldPosition;
  tarch::la::Vector<DIMENSIONS, int> adjacentManifoldPosition;
  #ifdef Dim2
  //Vertex to edge1
  assignList(manifoldPosition)         = -1, -1;
  assignList(adjacentManifoldPosition) =  0, -1;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 0, 1, 0), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 0, 1, 0));

  //Vertex to edge2
  assignList(manifoldPosition)         = -1, -1;
  assignList(adjacentManifoldPosition) = -1,  0;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 0, 1, 1), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 0, 1, 1));

  //Edge to vertex1
  assignList(manifoldPosition)         = -1,  0;
  assignList(adjacentManifoldPosition) = -1, -1;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 1, 0, 0), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 1, 0, 0));

  //Edge to vertex2
  assignList(manifoldPosition)         = -1, 0;
  assignList(adjacentManifoldPosition) = -1, 1;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 1, 0, 1), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 1, 0, 1));

  #elif Dim3
  //Vertex to edge1
  assignList(manifoldPosition)         = -1, -1, -1;
  assignList(adjacentManifoldPosition) =  0, -1, -1;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 0, 1, 0), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 0, 1, 0));

  //Vertex to edge1
  assignList(manifoldPosition)         = -1, -1, -1;
  assignList(adjacentManifoldPosition) = -1,  0, -1;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 0, 1, 1), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 0, 1, 1));

  //Vertex to edge2
  assignList(manifoldPosition)         = -1, -1, -1;
  assignList(adjacentManifoldPosition) = -1, -1,  0;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 0, 1, 2), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 0, 1, 2));

  //Vertex to face1
  assignList(manifoldPosition)         = -1, -1, -1;
  assignList(adjacentManifoldPosition) =  0,  0, -1;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 0, 2, 0), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 0, 2, 0));

  //Vertex to face2
  assignList(manifoldPosition)         = -1, -1, -1;
  assignList(adjacentManifoldPosition) =  0, -1,  0;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 0, 2, 1), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 0, 2, 1));

  //Vertex to face3
  assignList(manifoldPosition)         = -1, -1, -1;
  assignList(adjacentManifoldPosition) = -1,  0,  0;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 0, 2, 2), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 0, 2, 2));

  //Edge to face1
  assignList(manifoldPosition)         = -1, -1,  0;
  assignList(adjacentManifoldPosition) =  0, -1,  0;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 1, 2, 0), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 0, 2, 1));

  //Edge to face2
  assignList(manifoldPosition)         = -1, -1,  0;
  assignList(adjacentManifoldPosition) = -1,  0,  0;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 1, 2, 1), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 0, 2, 2));



  //Face to edge0
  assignList(manifoldPosition)         = -1,  0,  0;
  assignList(adjacentManifoldPosition) = -1, -1,  0;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 2, 1, 0), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 2, 1, 0));

  //Face to edge1
  assignList(manifoldPosition)         = -1,  0,  0;
  assignList(adjacentManifoldPosition) = -1,  1,  0;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 2, 1, 1), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 2, 1, 1));

  //Face to edge2
  assignList(manifoldPosition)         = -1,  0,  0;
  assignList(adjacentManifoldPosition) = -1,  0, -1;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 2, 1, 2), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 2, 1, 2));

  //Face to edge3
  assignList(manifoldPosition)         = -1,  0,  0;
  assignList(adjacentManifoldPosition) = -1,  0,  1;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 2, 1, 3), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 2, 1, 3));

  //Face to vertex0
  assignList(manifoldPosition)         = -1,  0,  0;
  assignList(adjacentManifoldPosition) = -1, -1, -1;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 2, 0, 0), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 2, 0, 0));

  //Face to vertex1
  assignList(manifoldPosition)         = -1,  0,  0;
  assignList(adjacentManifoldPosition) = -1,  1, -1;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 2, 0, 1), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 2, 0, 1));

  //Face to vertex2
  assignList(manifoldPosition)         = -1,  0,  0;
  assignList(adjacentManifoldPosition) = -1, -1,  1;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 2, 0, 2), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 2, 0, 2));

  //Face to vertex3
  assignList(manifoldPosition)         = -1,  0,  0;
  assignList(adjacentManifoldPosition) = -1,  1,  1;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 2, 0, 3), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 2, 0, 3));

  //Edge to vertex0
  assignList(manifoldPosition)         = -1, -1,  0;
  assignList(adjacentManifoldPosition) = -1, -1, -1;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 1, 0, 0), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 1, 0, 0));

  //Edge to vertex1
  assignList(manifoldPosition)         = -1, -1,  0;
  assignList(adjacentManifoldPosition) = -1, -1,  1;
  validateEqualsWithParams1(Region::getIndexOfAdjacentManifold(manifoldPosition, 1, 0, 1), adjacentManifoldPosition, Region::getIndexOfAdjacentManifold(manifoldPosition, 1, 0, 1));
  #endif
}

void peanoclaw::tests::PatchTest::testOverlapOfRemoteGhostlayers() {
  #ifdef Dim2
  tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int> adjacentRanks;
  assignList(adjacentRanks)              = 1, 1, 0,   1, 0,   1, 2, 2;
  tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int> overlapOfRemoteGhostlayers;
  assignList(overlapOfRemoteGhostlayers) = 3, 2, 2,   2, 2,   2, 2, 3;
  tarch::la::Vector<DIMENSIONS, int> subdivisionFactor(6);

  //Check rank 1
  peanoclaw::geometry::Region regions[THREE_POWER_D_MINUS_ONE];
  int numberOfRegions = peanoclaw::geometry::Region::getRegionsOverlappedByRemoteGhostlayers(
    adjacentRanks,
    overlapOfRemoteGhostlayers,
    subdivisionFactor,
    1, //Rank
    regions
  );

  tarch::la::Vector<DIMENSIONS, int> expectedOffset;
  tarch::la::Vector<DIMENSIONS, int> expectedSize;

  validateEquals(numberOfRegions, 3);

  assignList(expectedOffset) = 0, 0;
  assignList(expectedSize) = 3, 3;
  validateWithParams1(tarch::la::equals(regions[0]._offset, expectedOffset), regions[0]._offset);
  validateWithParams1(tarch::la::equals(regions[0]._size, expectedSize), regions[0]._size);

  assignList(expectedOffset) = 0, 3;
  assignList(expectedSize) = 2, 3;
  validateWithParams1(tarch::la::equals(regions[1]._offset, expectedOffset), regions[1]._offset);
  validateWithParams1(tarch::la::equals(regions[1]._size, expectedSize), regions[1]._size);

  assignList(expectedOffset) = 3, 0;
  assignList(expectedSize) = 3, 2;
  validateWithParams1(tarch::la::equals(regions[2]._offset, expectedOffset), regions[2]._offset);
  validateWithParams1(tarch::la::equals(regions[2]._size, expectedSize), regions[2]._size);

  //Check rank 2
  numberOfRegions = peanoclaw::geometry::Region::getRegionsOverlappedByRemoteGhostlayers(
    adjacentRanks,
    overlapOfRemoteGhostlayers,
    subdivisionFactor,
    2, //Rank
    regions
  );

  validateEquals(numberOfRegions, 2);

  assignList(expectedOffset) = 3, 3;
  assignList(expectedSize) = 3, 3;
  validateWithParams1(tarch::la::equals(regions[0]._offset, expectedOffset), regions[0]._offset);
  validateWithParams1(tarch::la::equals(regions[0]._size, expectedSize), regions[0]._size);

  assignList(expectedOffset) = 0, 4;
  assignList(expectedSize) = 3, 2;
  validateWithParams1(tarch::la::equals(regions[1]._offset, expectedOffset), regions[1]._offset);
  validateWithParams1(tarch::la::equals(regions[1]._size, expectedSize), regions[1]._size);
  #elif Dim3
  tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int> adjacentRanks;
  assignList(adjacentRanks)              = 1, 1, 0,   1, 1, 0,   0, 0, 0, //Lower plane
                                           1, 1, 0,   1,    2,   1, 0, 2, //Mid plane
                                           1, 1, 0,   1, 2, 2,   1, 2, 2; //Upper plane
  tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int> overlapOfRemoteGhostlayers;
  assignList(overlapOfRemoteGhostlayers) = 3, 2, 3,   2, 2, 3,   2, 2, 2, //Lower plane
                                           2, 2, 2,   2,    2,   2, 2, 2, //Mid plane
                                           3, 2, 3,   2, 2, 2,   2, 2, 2;
  tarch::la::Vector<DIMENSIONS, int> subdivisionFactor(6);

  //Check rank 1
//  peanoclaw::geometry::Region regions[THREE_POWER_D_MINUS_ONE];
//  int numberOfRegions = peanoclaw::geometry::Region::getRegionsOverlappedByRemoteGhostlayers(
//    adjacentRanks,
//    overlapOfRemoteGhostlayers,
//    subdivisionFactor,
//    1, //Rank
//    regions
//  );
//  validateEquals(numberOfRegions, 5);
  #endif
}

void peanoclaw::tests::PatchTest::testOverlapOfRemoteGhostlayers2() {
  #if defined(Parallel) && defined(Dim2)
  tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int> adjacentRanks;
  assignList(adjacentRanks)              = 1,1,2,1,2,0,0,0;
  tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int> overlapOfRemoteGhostlayers;
  assignList(overlapOfRemoteGhostlayers) = 0,0,2,0,2,18,18,18;
  tarch::la::Vector<DIMENSIONS, int> subdivisionFactor(18);

  //Check rank 2
  peanoclaw::geometry::Region regions[THREE_POWER_D_MINUS_ONE];
  int numberOfRegions = peanoclaw::geometry::Region::getRegionsOverlappedByRemoteGhostlayers(
    adjacentRanks,
    overlapOfRemoteGhostlayers,
    subdivisionFactor,
    2, //Rank
    regions
  );

  tarch::la::Vector<DIMENSIONS, int> expectedOffset;
  tarch::la::Vector<DIMENSIONS, int> expectedSize;

  validateEquals(numberOfRegions, 1);

  assignList(expectedOffset) = 16, 0;
  assignList(expectedSize) = 2, 18;
  validateWithParams1(tarch::la::equals(regions[0]._offset, expectedOffset), regions[0]._offset);
  validateWithParams1(tarch::la::equals(regions[0]._size, expectedSize), regions[0]._size);
  #endif
}

void peanoclaw::tests::PatchTest::testOverlapOfRemoteGhostlayers3D() {
  #if defined(Dim3) && defined(Parallel)
  tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int> adjacentRanks;
  assignList(adjacentRanks) = 2,2,2,2,2,1,1,1,1,1,1,1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1;
  tarch::la::Vector<THREE_POWER_D_MINUS_ONE, int> overlapOfRemoteGhostlayers;
  assignList(overlapOfRemoteGhostlayers) = 2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
  tarch::la::Vector<DIMENSIONS, int> subdivisionFactor;
  assignList(subdivisionFactor) = 6, 3, 3;

  //Check rank 2
  peanoclaw::geometry::Region regions[THREE_POWER_D_MINUS_ONE];
  int numberOfRegions = peanoclaw::geometry::Region::getRegionsOverlappedByRemoteGhostlayers(
    adjacentRanks,
    overlapOfRemoteGhostlayers,
    subdivisionFactor,
    2, //Rank
    regions
  );

  tarch::la::Vector<DIMENSIONS, int> expectedOffset;
  tarch::la::Vector<DIMENSIONS, int> expectedSize;

  validateEquals(numberOfRegions, 1);

  assignList(expectedOffset) = 0, 0, 0;
  assignList(expectedSize) = 6, 3, 2;
  validateWithParams1(tarch::la::equals(regions[0]._offset, expectedOffset), regions[0]._offset);
  validateWithParams1(tarch::la::equals(regions[0]._size, expectedSize), regions[0]._size);
  #endif
}

#ifdef UseTestSpecificCompilerSettings
#pragma optimize("",on)
#endif

