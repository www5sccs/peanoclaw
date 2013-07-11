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

void peanoclaw::tests::GhostlayerExtrapolationTest::testCornerExtrapolation2D() {
  #ifdef Dim2
  int ghostlayerWidth = 2;
  int subdivisionFactor = 3;

  Patch patch = createPatch(
    3,   //Unknowns per subcell
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

  //Set patch to zero
  patch.clearRegion(-ghostlayerWidth, 3 + 2*ghostlayerWidth, true);

  //Fill patch with test data
  tarch::la::Vector<DIMENSIONS, int> faceSize;
  assignList(faceSize) = 3, ghostlayerWidth;
  int counter = 1;
  dfor(subcellIndexInFace, faceSize) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndex;

    //Lower
    subcellIndex = subcellIndexInFace;
    subcellIndex(1) -= ghostlayerWidth;
    patch.setValueUOld(subcellIndex, 0, counter);

    //Upper
    subcellIndex(0) = patch.getSubdivisionFactor()(0) - subcellIndexInFace(0) - 1;
    subcellIndex(1) = patch.getSubdivisionFactor()(1) + ghostlayerWidth - subcellIndexInFace(1) - 1;
    patch.setValueUOld(subcellIndex, 0, counter);

    //Left
    subcellIndex(0) = -ghostlayerWidth + subcellIndexInFace(1);
    subcellIndex(1) = patch.getSubdivisionFactor()(1) - subcellIndexInFace(0) - 1;
    patch.setValueUOld(subcellIndex, 0, counter);

    //Right
    subcellIndex(0) = patch.getSubdivisionFactor()(0) + ghostlayerWidth - subcellIndexInFace(1) - 1;
    subcellIndex(1) = subcellIndexInFace(0);
    patch.setValueUOld(subcellIndex, 0, counter);

    counter++;
  }

  //Run extrapolation
  peanoclaw::interSubgridCommunication::Extrapolation extrapolation(patch);
  extrapolation.extrapolateGhostlayer();

  //Validate results
  {
    tarch::la::Vector<DIMENSIONS, int> subcell;
    double c1 = 2.0;
    double c2 = 4.0;
    double c3 = 3.0;
    double c4 = 5.0;

    //Lower-Left
    assignList(subcell) = -ghostlayerWidth, -ghostlayerWidth;
    validateNumericalEquals(patch.getValueUOld(subcell, 0), c1);
    assignList(subcell) = -1, -ghostlayerWidth;
    validateNumericalEquals(patch.getValueUOld(subcell, 0), c2);
    assignList(subcell) = -ghostlayerWidth, -1;
    validateNumericalEquals(patch.getValueUOld(subcell, 0), c3);
    assignList(subcell) = -1, -1;
    validateNumericalEquals(patch.getValueUOld(subcell, 0), c4);

    //Upper-Left
    assignList(subcell) = -ghostlayerWidth, subdivisionFactor + ghostlayerWidth - 1;
    validateNumericalEquals(patch.getValueUOld(subcell, 0), c1);
    assignList(subcell) = -ghostlayerWidth, subdivisionFactor + ghostlayerWidth - 2;
    validateNumericalEquals(patch.getValueUOld(subcell, 0), c2);
    assignList(subcell) = -1, subdivisionFactor + ghostlayerWidth - 1;
    validateNumericalEquals(patch.getValueUOld(subcell, 0), c3);
    assignList(subcell) = -1, subdivisionFactor + ghostlayerWidth - 2;
    validateNumericalEquals(patch.getValueUOld(subcell, 0), c4);

    //Lower-Right
    assignList(subcell) = subdivisionFactor + ghostlayerWidth - 1, -ghostlayerWidth;
    validateNumericalEquals(patch.getValueUOld(subcell, 0), c1);
    assignList(subcell) = subdivisionFactor + ghostlayerWidth - 1, -1;
    validateNumericalEquals(patch.getValueUOld(subcell, 0), c2);
    assignList(subcell) = subdivisionFactor, -ghostlayerWidth;
    validateNumericalEquals(patch.getValueUOld(subcell, 0), c3);
    assignList(subcell) = subdivisionFactor, -1;
    validateNumericalEquals(patch.getValueUOld(subcell, 0), c4);

    //Upper-Right
    assignList(subcell) = subdivisionFactor + ghostlayerWidth - 1, subdivisionFactor + ghostlayerWidth - 1;
    validateNumericalEquals(patch.getValueUOld(subcell, 0), c1);
    assignList(subcell) = subdivisionFactor, subdivisionFactor + ghostlayerWidth - 1;
    validateNumericalEquals(patch.getValueUOld(subcell, 0), c2);
    assignList(subcell) = subdivisionFactor + ghostlayerWidth - 1, subdivisionFactor;
    validateNumericalEquals(patch.getValueUOld(subcell, 0), c3);
    assignList(subcell) = subdivisionFactor, subdivisionFactor;
    validateNumericalEquals(patch.getValueUOld(subcell, 0), c4);
  }
  #endif
}

void peanoclaw::tests::GhostlayerExtrapolationTest::run() {
  testMethod(testCornerExtrapolation2D);
}

void peanoclaw::tests::GhostlayerExtrapolationTest::setUp() {
}

#ifdef UseTestSpecificCompilerSettings
#pragma optimize("",on)
#endif
