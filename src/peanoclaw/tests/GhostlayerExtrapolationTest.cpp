/*
 * GhostlayerExtrapolationTest.cpp
 *
 *  Created on: Jul 8, 2013
 *      Author: kristof
 */
#include "peanoclaw/tests/GhostlayerExtrapolationTest.h"

#include "peanoclaw/interSubgridCommunication/Extrapolation.h"
#include "peanoclaw/tests/Helper.h"

#include "peano/utils/Loop.h"

#include "tarch/Assertions.h"
#include "tarch/tests/TestCaseFactory.h"
registerTest(peanoclaw::tests::GhostlayerExtrapolationTest)

#ifdef UseTestSpecificCompilerSettings
#pragma optimize("",off)
#endif

void peanoclaw::tests::GhostlayerExtrapolationTest::testEdgeExtrapolation2D() {
  #ifdef Dim2
  int ghostlayerWidth = 2;
  int subdivisionFactor = 3;

  Patch subgrid = createPatch(
    3,   //Unknowns per subcell
    0,   //Aux fields per subcell
    0,   //Aux fields per subcell
    subdivisionFactor,
    ghostlayerWidth,
    0.0, //Position
    1.0, //Size
    0,   //Level
    0.0, //Current time
    0.0, //Timestep size
    0.0  //Minimal neighbor time
  );
  peanoclaw::grid::SubgridAccessor& accessor = subgrid.getAccessor();

  //Set subgrid to zero
  accessor.clearRegion(Region(-ghostlayerWidth, 3 + 2*ghostlayerWidth), true);

  //Fill subgrid with test data
  tarch::la::Vector<DIMENSIONS, int> faceSize;
  assignList(faceSize) = 3, ghostlayerWidth;
  int counter = 1;
  dfor(subcellIndexInFace, faceSize) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndex;

    //Lower
    subcellIndex = subcellIndexInFace;
    subcellIndex(1) -= ghostlayerWidth;
    accessor.setValueUOld(subcellIndex, 0, counter);

    //Upper
    subcellIndex(0) = subgrid.getSubdivisionFactor()(0) - subcellIndexInFace(0) - 1;
    subcellIndex(1) = subgrid.getSubdivisionFactor()(1) + ghostlayerWidth - subcellIndexInFace(1) - 1;
    accessor.setValueUOld(subcellIndex, 0, counter);

    //Left
    subcellIndex(0) = -ghostlayerWidth + subcellIndexInFace(1);
    subcellIndex(1) = subgrid.getSubdivisionFactor()(1) - subcellIndexInFace(0) - 1;
    accessor.setValueUOld(subcellIndex, 0, counter);

    //Right
    subcellIndex(0) = subgrid.getSubdivisionFactor()(0) + ghostlayerWidth - subcellIndexInFace(1) - 1;
    subcellIndex(1) = subcellIndexInFace(0);
    accessor.setValueUOld(subcellIndex, 0, counter);

    counter++;
  }

  //Run extrapolation
  peanoclaw::interSubgridCommunication::Extrapolation extrapolation(subgrid);
  extrapolation.extrapolateEdges();

  //Validate results
  {
    tarch::la::Vector<DIMENSIONS, int> subcell;
    double c1 = 2.0;
    double c2 = 4.0;
    double c3 = 3.0;
    double c4 = 5.0;

    //TODO unterweg debug
    std::cout << subgrid.toStringUOldWithGhostLayer() << std::endl;

    //Lower-Left
    assignList(subcell) = -ghostlayerWidth, -ghostlayerWidth;
    validateNumericalEquals(accessor.getValueUOld(subcell, 0), c1);
    assignList(subcell) = -1, -ghostlayerWidth;
    validateNumericalEquals(accessor.getValueUOld(subcell, 0), c2);
    assignList(subcell) = -ghostlayerWidth, -1;
    validateNumericalEquals(accessor.getValueUOld(subcell, 0), c3);
    assignList(subcell) = -1, -1;
    validateNumericalEquals(accessor.getValueUOld(subcell, 0), c4);

    //Upper-Left
    assignList(subcell) = -ghostlayerWidth, subdivisionFactor + ghostlayerWidth - 1;
    validateNumericalEquals(accessor.getValueUOld(subcell, 0), c1);
    assignList(subcell) = -ghostlayerWidth, subdivisionFactor + ghostlayerWidth - 2;
    validateNumericalEquals(accessor.getValueUOld(subcell, 0), c2);
    assignList(subcell) = -1, subdivisionFactor + ghostlayerWidth - 1;
    validateNumericalEquals(accessor.getValueUOld(subcell, 0), c3);
    assignList(subcell) = -1, subdivisionFactor + ghostlayerWidth - 2;
    validateNumericalEquals(accessor.getValueUOld(subcell, 0), c4);

    //Lower-Right
    assignList(subcell) = subdivisionFactor + ghostlayerWidth - 1, -ghostlayerWidth;
    validateNumericalEquals(accessor.getValueUOld(subcell, 0), c1);
    assignList(subcell) = subdivisionFactor + ghostlayerWidth - 1, -1;
    validateNumericalEquals(accessor.getValueUOld(subcell, 0), c2);
    assignList(subcell) = subdivisionFactor, -ghostlayerWidth;
    validateNumericalEquals(accessor.getValueUOld(subcell, 0), c3);
    assignList(subcell) = subdivisionFactor, -1;
    validateNumericalEquals(accessor.getValueUOld(subcell, 0), c4);

    //Upper-Right
    assignList(subcell) = subdivisionFactor + ghostlayerWidth - 1, subdivisionFactor + ghostlayerWidth - 1;
    validateNumericalEquals(accessor.getValueUOld(subcell, 0), c1);
    assignList(subcell) = subdivisionFactor, subdivisionFactor + ghostlayerWidth - 1;
    validateNumericalEquals(accessor.getValueUOld(subcell, 0), c2);
    assignList(subcell) = subdivisionFactor + ghostlayerWidth - 1, subdivisionFactor;
    validateNumericalEquals(accessor.getValueUOld(subcell, 0), c3);
    assignList(subcell) = subdivisionFactor, subdivisionFactor;
    validateNumericalEquals(accessor.getValueUOld(subcell, 0), c4);
  }
  #endif
}

void peanoclaw::tests::GhostlayerExtrapolationTest::testEdgeAndCornerExtrapolation3D() {
  #ifdef Dim3
  int ghostlayerWidth = 2;
  int subdivisionFactor = 3;

  Patch subgrid = createPatch(
    3,   //Unknowns per subcell
    0,   //Aux fields per subcell
    0,   //Aux fields per subcell
    subdivisionFactor,
    ghostlayerWidth,
    0.0, //Position
    1.0, //Size
    0,   //Level
    0.0, //Current time
    0.0, //Timestep size
    0.0  //Minimal neighbor time
  );
  peanoclaw::grid::SubgridAccessor& accessor = subgrid.getAccessor();

  //Set subgrid to zero
  accessor.clearRegion(Region(-ghostlayerWidth, 3 + 2*ghostlayerWidth), true);

  //Fill subgrid with test data
  tarch::la::Vector<DIMENSIONS, int> faceSize;
  assignList(faceSize) = ghostlayerWidth, 3, 3;
  int counter = 1;
  dfor(subcellIndexInFace, faceSize) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndex;

    //Left
    subcellIndex(0) = -ghostlayerWidth + subcellIndexInFace(0);
    subcellIndex(1) = subcellIndexInFace(1);
    subcellIndex(2) = subcellIndexInFace(2);
    accessor.setValueUOld(subcellIndex, 0, counter);

    //Right
    subcellIndex(0) = subgrid.getSubdivisionFactor()(0) + ghostlayerWidth - subcellIndexInFace(0) - 1;
    subcellIndex(1) = subcellIndexInFace(1);
    subcellIndex(2) = subcellIndexInFace(2);
    accessor.setValueUOld(subcellIndex, 0, counter);

    //Front
    subcellIndex(0) = subcellIndexInFace(1);
    subcellIndex(1) = -ghostlayerWidth + subcellIndexInFace(0);
    subcellIndex(2) = subcellIndexInFace(2);
    accessor.setValueUOld(subcellIndex, 0, counter);

    //Back
    subcellIndex(0) = subcellIndexInFace(1);
    subcellIndex(1) = subgrid.getSubdivisionFactor()(1) + ghostlayerWidth - subcellIndexInFace(0) - 1;
    subcellIndex(2) = subcellIndexInFace(2);
    accessor.setValueUOld(subcellIndex, 0, counter);

    //Lower
    subcellIndex = subcellIndexInFace;
    subcellIndex(0) = subcellIndexInFace(1);
    subcellIndex(1) = subcellIndexInFace(2);
    subcellIndex(2) = -ghostlayerWidth + subcellIndexInFace(0);
    accessor.setValueUOld(subcellIndex, 0, counter);

    //Upper
    subcellIndex = subcellIndexInFace;
    subcellIndex(0) = subcellIndexInFace(1);
    subcellIndex(1) = subcellIndexInFace(2);
    subcellIndex(2) = subgrid.getSubdivisionFactor()(2) + ghostlayerWidth - subcellIndexInFace(0) - 1;
    accessor.setValueUOld(subcellIndex, 0, counter);

    counter++;
  }

  //TODO unterweg debug
//  std::cout << "Subgrid before extrapolation: " << std::endl << subgrid.toStringUOldWithGhostLayer() << std::endl;

  //Run extrapolation
  peanoclaw::interSubgridCommunication::Extrapolation extrapolation(subgrid);
  extrapolation.extrapolateEdges();
  extrapolation.extrapolateCorners();

  //TODO unterweg debug
//  std::cout << "Subgrid after extrapolation: " << std::endl << subgrid.toStringUOldWithGhostLayer() << std::endl;

  //Validation
  {
    tarch::la::Vector<DIMENSIONS,int> subcellIndex;
    double actualValue;

    //Edge 0
    assignList(subcellIndex) = 0, -1, -2;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, -7.5, actualValue);

    assignList(subcellIndex) = 1, -1, -2;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, -5.5, actualValue);

    assignList(subcellIndex) = 2, -1, -2;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, -3.5, actualValue);

    assignList(subcellIndex) = 0, -2, -2;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, -11, actualValue);

    assignList(subcellIndex) = 1, -2, -2;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, -9, actualValue);

    assignList(subcellIndex) = 2, -2, -2;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, -7, actualValue);

    assignList(subcellIndex) = 0, -1, -1;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, -4, actualValue);

    assignList(subcellIndex) = 1, -1, -1;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, -2, actualValue);

    assignList(subcellIndex) = 2, -1, -1;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 0, actualValue);

    assignList(subcellIndex) = 0, -2, -1;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, -7.5, actualValue);

    assignList(subcellIndex) = 1, -2, -1;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, -5.5, actualValue);

    assignList(subcellIndex) = 2, -2, -1;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, -3.5, actualValue);



    //Edge 1
    assignList(subcellIndex) = 0, 3, -2;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 4.5, actualValue);

    assignList(subcellIndex) = 1, 3, -2;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 6.5, actualValue);

    assignList(subcellIndex) = 2, 3, -2;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 8.5, actualValue);

    assignList(subcellIndex) = 0, 4, -2;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 7, actualValue);

    assignList(subcellIndex) = 1, 4, -2;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 9, actualValue);

    assignList(subcellIndex) = 2, 4, -2;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 11, actualValue);

    assignList(subcellIndex) = 0, 3, -1;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 8, actualValue);

    assignList(subcellIndex) = 1, 3, -1;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 10, actualValue);

    assignList(subcellIndex) = 2, 3, -1;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 12, actualValue);

    assignList(subcellIndex) = 0, 4, -1;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 10.5, actualValue);

    assignList(subcellIndex) = 1, 4, -1;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 12.5, actualValue);

    assignList(subcellIndex) = 2, 4, -1;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 14.5, actualValue);




    //Edge 2
    assignList(subcellIndex) = 3, 0, -2;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, -1.5, actualValue);

    assignList(subcellIndex) = 3, 1, -2;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 2.5, actualValue);

    assignList(subcellIndex) = 3, 2, -2;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 6.5, actualValue);

    assignList(subcellIndex) = 4, 0, -2;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, -1, actualValue);

    assignList(subcellIndex) = 4, 1, -2;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 3, actualValue);

    assignList(subcellIndex) = 4, 2, -2;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 7, actualValue);

    assignList(subcellIndex) = 3, 0, -1;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 2, actualValue);

    assignList(subcellIndex) = 3, 1, -1;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 6, actualValue);

    assignList(subcellIndex) = 3, 2, -1;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 10, actualValue);

    assignList(subcellIndex) = 4, 0, -1;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 2.5, actualValue);

    assignList(subcellIndex) = 4, 1, -1;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 6.5, actualValue);

    assignList(subcellIndex) = 4, 2, -1;
    actualValue = accessor.getValueUOld(subcellIndex, 0);
    validateNumericalEqualsWithParams1(actualValue, 10.5, actualValue);
  }
  #endif
}

void peanoclaw::tests::GhostlayerExtrapolationTest::run() {
  testMethod( testEdgeExtrapolation2D );
  testMethod( testEdgeAndCornerExtrapolation3D );
}

void peanoclaw::tests::GhostlayerExtrapolationTest::setUp() {
}

#ifdef UseTestSpecificCompilerSettings
#pragma optimize("",on)
#endif
