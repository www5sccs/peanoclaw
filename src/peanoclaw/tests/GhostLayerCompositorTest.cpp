/*
 * GhostLayerCompositorTest.cpp
 *
 *  Created on: Mar 6, 2012
 *      Author: Kristof Unterweger
 */
#include "peanoclaw/tests/GhostLayerCompositorTest.h"

#include "peanoclaw/tests/NumericsTestStump.h"
#include "peanoclaw/tests/Helper.h"

#include "peanoclaw/Area.h"
#include "peanoclaw/Patch.h"
#include "peanoclaw/interSubgridCommunication/GhostLayerCompositor.h"
#include "peanoclaw/interSubgridCommunication/DefaultFluxCorrection.h"
#include "peanoclaw/interSubgridCommunication/DefaultRestriction.h"
#include "peanoclaw/interSubgridCommunication/aspects/FaceAdjacentPatchTraversal.h"
#include "peanoclaw/interSubgridCommunication/aspects/EdgeAdjacentPatchTraversal.h"
#include "peanoclaw/records/CellDescription.h"

#include "peano/heap/Heap.h"
#include "peano/utils/Loop.h"

#include "tarch/tests/TestCaseFactory.h"
registerTest(peanoclaw::tests::GhostLayerCompositorTest)


#ifdef UseTestSpecificCompilerSettings
#pragma optimize("",off)
#endif

tarch::logging::Log peanoclaw::tests::GhostLayerCompositorTest::_log("peanoclaw::tests::GhostLayerCompositorTest");

peanoclaw::tests::GhostLayerCompositorTest::GhostLayerCompositorTest() {
}

peanoclaw::tests::GhostLayerCompositorTest::~GhostLayerCompositorTest() {
}

void peanoclaw::tests::GhostLayerCompositorTest::setUp() {
}

void peanoclaw::tests::GhostLayerCompositorTest::run() {
  testMethod( testTimesteppingVeto2D );
  testMethod( testInterpolationFromCoarseToFinePatchLeftGhostLayer2D );
  testMethod( testInterpolationFromCoarseToFinePatchRightGhostLayer2D );
  testMethod( testProjectionFromCoarseToFinePatchRightGhostLayer2D );
  //TODO unterweg debug
//  testMethod( testFluxCorrection );
  testMethod( testRestrictionWithOverlappingBounds );
  testMethod( testPartialRestrictionAreas );
  testMethod( testPartialRestrictionAreasWithInfiniteLowerBounds );
  testMethod( testFaceAdjacentPatchTraversal2D );
  testMethod( testEdgeAdjacentPatchTraversal2D );
}

void peanoclaw::tests::GhostLayerCompositorTest::testTimesteppingVeto2D() {
  #if defined(Dim2)

  for(int vetoIndex = 0; vetoIndex < TWO_POWER_D; vetoIndex++) {

    logDebug("testTimesteppingVeto2D", "testing vetoIndex=" << vetoIndex);

    peanoclaw::Patch patches[TWO_POWER_D];

    for(int i = 0; i < TWO_POWER_D; i++) {
      patches[i] = createPatch(
        3,       //Unknowns per subcell
        0,       //Aux fields per subcell
        16,      //Subdivision factor
        2,       //Ghostlayer width
        0.0,     //Position
        1.0/3.0, //Size
        0,       //Level
        0.0,     //Current time
        1.0,     //Timestep size
        1.0      //Minimal neighbor time
      );
    }
    patches[vetoIndex] =
        createPatch(
            3,       //Unknowns per subcell
            0,       //Aux fields per subcell
            16,      //Subdivision factor
            2,       //Ghostlayer width
            0.0,     //Position
            1.0/3.0, //Size
            0,       //Level
            0.5,     //Current time
            1.0,     //Timestep size
            1.0      //Minimal neighbor time
           );

    for(int i = 0; i < TWO_POWER_D; i++) {
      patches[i].resetMinimalNeighborTimeConstraint();
      validate(patches[i].isValid());
    }

    peanoclaw::tests::NumericsTestStump numerics;
    peanoclaw::interSubgridCommunication::GhostLayerCompositor ghostLayerCompositor
      = peanoclaw::interSubgridCommunication::GhostLayerCompositor(
          patches,
          0,
          numerics,
          false
          );
    ghostLayerCompositor.fillGhostLayers(-1);
    ghostLayerCompositor.updateNeighborTimes();

    for(int cellIndex = 0; cellIndex < 4; cellIndex++) {
      if(cellIndex == vetoIndex) {
        validateNumericalEqualsWithParams2(patches[cellIndex].getMinimalNeighborTimeConstraint(), 1.0, vetoIndex, cellIndex);
        validateNumericalEqualsWithParams2(patches[cellIndex].isAllowedToAdvanceInTime(), false, vetoIndex, cellIndex);
      } else {
        validateNumericalEqualsWithParams2(patches[cellIndex].getMinimalNeighborTimeConstraint(), 1.0, vetoIndex, cellIndex);
        validateNumericalEqualsWithParams2(patches[cellIndex].isAllowedToAdvanceInTime(), true, vetoIndex, cellIndex);
      }
    }
  }
  #endif
}

void peanoclaw::tests::GhostLayerCompositorTest::testInterpolationFromCoarseToFinePatchLeftGhostLayer2D() {
  #if defined(Dim2)
  //Patch-array for lower-left vertex in fine patch
  peanoclaw::Patch patches[TWO_POWER_D];

  //Settings
  int coarseSubdivisionFactor = 2;
  int coarseGhostlayerWidth = 1;
  int fineSubdivisionFactor = 2;
  int fineGhostlayerWidth = 2;
  int unknownsPerSubcell = 1;

  //Setup coarse patch
  tarch::la::Vector<DIMENSIONS, double> coarsePosition;
  assignList(coarsePosition) = 2.0, 1.0;
  patches[1] = createPatch(
    unknownsPerSubcell,
    0,   //Aux fields per subcell
    coarseSubdivisionFactor,
    coarseGhostlayerWidth,
    coarsePosition,
    3.0, //Coarse size
    0,   //Level
    0.0, //Time
    1.0, //Timestep size
    1.0, //Minimal neighbor time
    false   //Overlapped by coarse ghost layer
  );
  patches[3] = patches[1];

  //Setup fine patch
  tarch::la::Vector<DIMENSIONS, double> finePosition;
  assignList(finePosition) = 5.0, 2.0;
  patches[0] = createPatch(
    unknownsPerSubcell,
    0,   //Aux fields per subcell
    fineSubdivisionFactor,
    fineGhostlayerWidth,
    finePosition,
    1.0, //Size
    1,   //Level
    0.0, //Time
    1.0/3.0, //Timestep size
    1.0  //Minimal neighbor time
  );

  //Fill coarse patch
  dfor(index, coarseSubdivisionFactor) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndex;
    subcellIndex(0) = index(1);
    subcellIndex(1) = index(0);
    patches[1].setValueUNew(subcellIndex, 0, 1.0);
  }
  dfor(index, coarseSubdivisionFactor + 2*coarseGhostlayerWidth) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndex;
    subcellIndex(0) = index(1) - coarseGhostlayerWidth;
    subcellIndex(1) = index(0) - coarseGhostlayerWidth;
    patches[1].setValueUOld(subcellIndex, 0, -1.0);
  }
  tarch::la::Vector<DIMENSIONS, int> subcellIndex;
  assignList(subcellIndex) = 0, 0;
  patches[1].setValueUOld(subcellIndex, 0, 6.0);
  assignList(subcellIndex) = 0, 1;
  patches[1].setValueUOld(subcellIndex, 0, 7.0);
  assignList(subcellIndex) = 1, 0;
  patches[1].setValueUOld(subcellIndex, 0, 10.0);
  assignList(subcellIndex) = 1, 1;
  patches[1].setValueUOld(subcellIndex, 0, 11.0);

  //Fill left ghostlayer
  peanoclaw::tests::NumericsTestStump numerics;
  peanoclaw::interSubgridCommunication::GhostLayerCompositor ghostLayerCompositor(
    patches,
    1,
    numerics,
    false
  );
  ghostLayerCompositor.fillGhostLayers(-1);

  assignList(subcellIndex) = -2, 0;
  validateNumericalEquals(patches[0].getValueUOld(subcellIndex, 0), 65.0/9.0);
  assignList(subcellIndex) = -2, 1;
  validateNumericalEquals(patches[0].getValueUOld(subcellIndex, 0), 67.0/9.0);
  assignList(subcellIndex) = -1, 0;
  validateNumericalEquals(patches[0].getValueUOld(subcellIndex, 0), 73.0/9.0);
  assignList(subcellIndex) = -1, 1;
  validateNumericalEquals(patches[0].getValueUOld(subcellIndex, 0), 25.0/3.0);
  #endif
}

void peanoclaw::tests::GhostLayerCompositorTest::testInterpolationFromCoarseToFinePatchRightGhostLayer2D() {
  #if defined(Dim2)
  //Patch-array for lower-left vertex in fine patch
  peanoclaw::Patch patches[TWO_POWER_D];

  //Settings
  int coarseSubdivisionFactor = 2;
  int coarseGhostlayerWidth = 1;
  int fineSubdivisionFactor = 2;
  int fineGhostlayerWidth = 2;
  int unknownsPerSubcell = 1;

  //Setup coarse patch
  tarch::la::Vector<DIMENSIONS, double> coarsePosition;
  assignList(coarsePosition) = 2.0, 1.0;
  patches[2] = createPatch(
    unknownsPerSubcell,
    0,   //Aux fields per subcell
    coarseSubdivisionFactor,
    coarseGhostlayerWidth,
    coarsePosition,
    3.0, //Coarse size
    0,   //Level
    0.0, //Time
    1.0, //Timestep size
    1.0  //Minimal neighbor time
  );
  patches[0] = patches[2];

  //Setup fine patch
  tarch::la::Vector<DIMENSIONS, double> finePosition;
  assignList(finePosition) = 1.0, 1.0;
  patches[3] = createPatch(
    unknownsPerSubcell,
    0,   //Aux fields per subcell
    fineSubdivisionFactor,
    fineGhostlayerWidth,
    finePosition,
    1.0, //Size
    1,   //Level
    0.0, //Time
    1.0/3.0, //Timestep size
    1.0  //Minimal neighbor time
  );

  //Fill coarse patch
  dfor(index, coarseSubdivisionFactor) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndex;
    subcellIndex(0) = index(1);
    subcellIndex(1) = index(0);
    patches[2].setValueUNew(subcellIndex, 0, 1.0);
  }
  dfor(index, coarseSubdivisionFactor + 2*coarseGhostlayerWidth) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndex;
    subcellIndex(0) = index(1) - coarseGhostlayerWidth;
    subcellIndex(1) = index(0) - coarseGhostlayerWidth;
    patches[2].setValueUOld(subcellIndex, 0, -1.0);
  }
  tarch::la::Vector<DIMENSIONS, int> subcellIndex;
  assignList(subcellIndex) = 0, 0;
  patches[2].setValueUOld(subcellIndex, 0, 6.0);
  assignList(subcellIndex) = 0, 1;
  patches[2].setValueUOld(subcellIndex, 0, 7.0);
  assignList(subcellIndex) = 1, 0;
  patches[2].setValueUOld(subcellIndex, 0, 10.0);
  assignList(subcellIndex) = 1, 1;
  patches[2].setValueUOld(subcellIndex, 0, 11.0);

  //Fill left ghostlayer
  peanoclaw::tests::NumericsTestStump numerics;
  peanoclaw::interSubgridCommunication::GhostLayerCompositor ghostLayerCompositor(
    patches,
    1,
    numerics,
    false
  );
  ghostLayerCompositor.fillGhostLayers(-1);

  assignList(subcellIndex) = 2, 0;
  validateNumericalEquals(patches[3].getValueUOld(subcellIndex, 0), 29.0/9.0);
  assignList(subcellIndex) = 2, 1;
  validateNumericalEquals(patches[3].getValueUOld(subcellIndex, 0), 31.0/9.0);
  assignList(subcellIndex) = 3, 0;
  validateNumericalEquals(patches[3].getValueUOld(subcellIndex, 0), 37.0/9.0);
  assignList(subcellIndex) = 3, 1;
  validateNumericalEquals(patches[3].getValueUOld(subcellIndex, 0), 13.0/3.0);
  #endif
}

void peanoclaw::tests::GhostLayerCompositorTest::testProjectionFromCoarseToFinePatchRightGhostLayer2D() {
  #ifdef Dim2

//  //Patch-array for upper-right vertex in fine patch
//  peanoclaw::Patch patches[TWO_POWER_D];
//
//  //Setup coarse patch
//  tarch::la::Vector<DIMENSIONS, double> coarsePosition;
//  assignList(coarsePosition) = 1.0, 2.0;
//  std::vector<peanoclaw::records::Data> uNewCoarse;
//  for(int i = 0; i < 4*4*3; i++) {
//    uNewCoarse.push_back(peanoclaw::records::Data());
//    uNewCoarse[i].setU(i);
//  }
//  std::vector<peanoclaw::records::Data> uOldCoarse;
//  for(int i = 0; i < (4+4) * (4+4) * 3; i++) {
//    uOldCoarse.push_back(peanoclaw::records::Data());
//    uOldCoarse[i] = -1.0;
//  }
//  patches[0] = peanoclaw::Patch(
//    coarsePosition, //Position
//    0.3,            //Size
//    3,              //Unknows per subcell
//    4,              //Subdivision factor
//    2,              //Ghostlayer width
//    0.0,            //Current time
//    1.0,            //Timestep size
//    0.0,            //CFL
//    1.0,            //Maximum timestep size
//    0,              //Level
//    &uNewCoarse,    //uNew
//    &uOldCoarse     //uOld
//  );
//  patches[2] = patches[0];
//
//  //Setup fine patch
//  tarch::la::Vector<DIMENSIONS, double> finePosition;
//  assignList(finePosition) = 0.9, 2.0;
//  std::vector<peanoclaw::records::Data> uNewFine;
//  for(int i = 0; i < 4*4*3; i++) {
//    uNewFine.push_back(peanoclaw::records::Data());
//    uNewFine[i].setU(-2.0);
//  }
//  std::vector<peanoclaw::records::Data> uOldFine;
//  for(int i = 0; i < (4+4) * (4+4) * 3; i++) {
//    uOldFine.push_back(peanoclaw::records::Data());
//    uOldFine[i].setU(-1.0);
//  }
//
//  patches[3] = peanoclaw::Patch(
//    finePosition,   //Position
//    0.1,            //Size
//    3,              //Unknows per subcell
//    4,              //Subdivision factor
//    2,              //Ghostlayer width
//    0.0,            //Current time
//    1.0,            //Timestep size
//    0.0,            //CFL
//    1.0,            //Maximum timestep size
//    1,              //Level
//    &uNewFine,      //uNew
//    &uOldFine       //uOld
//  );
//
//  //Fill right ghostlayer
//  peanoclaw::tests::NumericsTestStump numerics;
//  peanoclaw::interSubgridCommunication::GhostLayerCompositor ghostLayerCompositor(
//    patches,
//    1,
//    numerics
//  );
//  ghostLayerCompositor.fillGhostLayers();
//
//  validateNumericalEquals(patches[3].getValueUOld(4, -2, 0), -1.0);
//  validateNumericalEquals(patches[3].getValueUOld(4, -1, 0), -1.0);
//  validateNumericalEquals(patches[3].getValueUOld(4, 0, 0), 0.0);
//  validateNumericalEquals(patches[3].getValueUOld(4, 1, 0), 0.0);
//  validateNumericalEquals(patches[3].getValueUOld(4, 2, 0), 0.0);
//  validateNumericalEquals(patches[3].getValueUOld(4, 3, 0), 1.0);
//  validateNumericalEquals(patches[3].getValueUOld(4, 4, 0), 1.0);
//  validateNumericalEquals(patches[3].getValueUOld(4, 5, 0), 1.0);
//
//  validateNumericalEquals(patches[3].getValueUOld(5, -2, 0), -1.0);
//  validateNumericalEquals(patches[3].getValueUOld(5, -1, 0), -1.0);
//  validateNumericalEquals(patches[3].getValueUOld(5, 0, 0), 0.0);
//  validateNumericalEquals(patches[3].getValueUOld(5, 1, 0), 0.0);
//  validateNumericalEquals(patches[3].getValueUOld(5, 2, 0), 0.0);
//  validateNumericalEquals(patches[3].getValueUOld(5, 3, 0), 1.0);
//  validateNumericalEquals(patches[3].getValueUOld(5, 4, 0), 1.0);
//  validateNumericalEquals(patches[3].getValueUOld(5, 5, 0), 1.0);
  #endif
}

void peanoclaw::tests::GhostLayerCompositorTest::testFluxCorrection() {
  int unknownsPerSubcell = 3;
  int coarseSubdivisionFactor = 2;
  int fineSubdivisionFactor = 3;
  int ghostlayerWidth = 2;

  tarch::la::Vector<DIMENSIONS, double> coarsePosition(0.0);
  coarsePosition(0) = 1.0;

  Patch coarsePatch = createPatch(
    unknownsPerSubcell,
    0,   //Aux per subcell
    coarseSubdivisionFactor,
    ghostlayerWidth,
    coarsePosition,
    1.0, //Size
    0, //Level
    0.0, //Current time
    1.0, //Timestep size
    1.0  //Minimal neighbor time
  );

  //Fill coarse patch
  dfor(subcellIndex, coarseSubdivisionFactor) {
    for(int unknown = 0; unknown < unknownsPerSubcell; unknown++) {
      coarsePatch.setValueUNew(subcellIndex, unknown, (unknown == 0) ? 1.0 : 0.5);
    }
  }

  dfor(subcellIndex, coarseSubdivisionFactor + 2*ghostlayerWidth) {
    for(int unknown = 0; unknown < unknownsPerSubcell; unknown++) {
      coarsePatch.setValueUOld(subcellIndex - ghostlayerWidth, unknown, (unknown == 0) ? 1.0 : 0.5);
    }
  }
//  std::vector<peanoclaw::records::Data>& uNewCoarse
//    = DataHeap::getInstance()
//    .getData(coarsePatch.getUNewIndex());
//  for(int i = 0; i < coarseSubdivisionFactor*coarseSubdivisionFactor*unknownsPerSubcell; i++) {
//    uNewCoarse.push_back(peanoclaw::records::Data());
//    if(i < coarseSubdivisionFactor*coarseSubdivisionFactor) {
//      uNewCoarse.at(i).setU(1.0);
//    } else {
//      uNewCoarse.at(i).setU(0.5);
//    }
//  }
//  std::vector<peanoclaw::records::Data>& uOldCoarse
//    = DataHeap::getInstance()
//    .getData(coarsePatch.getUOldIndex());
//  for(int i = 0; i < (coarseSubdivisionFactor+2*ghostlayerWidth) * (coarseSubdivisionFactor+2*ghostlayerWidth) * unknownsPerSubcell; i++) {
//    uOldCoarse.push_back(peanoclaw::records::Data());
//    if(i < (coarseSubdivisionFactor+2*ghostlayerWidth) * (coarseSubdivisionFactor+2*ghostlayerWidth)) {
//      uOldCoarse.at(i).setU(1.0);
//    } else {
//      uOldCoarse.at(i).setU(0.5);
//    }
//  }

  tarch::la::Vector<DIMENSIONS, double> finePosition(2.0/3.0);
  finePosition(1) = 1.0/3.0;

  Patch finePatch = createPatch(
    unknownsPerSubcell,
    0,         //Aux per subcell
    fineSubdivisionFactor,
    ghostlayerWidth,
    finePosition,
    1.0 / 3.0, //Size
    1,         //Level
    0.0,       //Current time
    1.0 / 3.0, //Timestep size
    1.0       //Minimal neighbor time
  );

  //Fine patch
  dfor(subcellIndex, fineSubdivisionFactor) {
    for(int unknown = 0; unknown < unknownsPerSubcell; unknown++) {
      finePatch.setValueUNew(subcellIndex, unknown, (unknown == 0) ? 1.0 : 0.5);
    }
  }
  tarch::la::Vector<DIMENSIONS, int> subcellIndex;
  assignList(subcellIndex) = 2, 0;
  finePatch.setValueUNew(subcellIndex, 0, 18.0);
  assignList(subcellIndex) = 2, 1;
  finePatch.setValueUNew(subcellIndex, 0, 36.0);
  assignList(subcellIndex) = 2, 2;
  finePatch.setValueUNew(subcellIndex, 0, 54.0);
//  uNewFine[6] = 18.0;
//  uNewFine[7] = 36.0;
//  uNewFine[8] = 54.0;

  dfor(subcellIndex, fineSubdivisionFactor + 2*ghostlayerWidth) {
    for(int unknown = 0; unknown < unknownsPerSubcell; unknown++) {
      finePatch.setValueUOld(subcellIndex - ghostlayerWidth, unknown, (unknown == 0) ? 1.0 : 0.5);
    }
  }
//  std::vector<peanoclaw::records::Data>& uNewFine
//    = DataHeap::getInstance()
//    .getData(finePatch.getUNewIndex());
//  for(int i = 0; i < fineSubdivisionFactor*fineSubdivisionFactor*unknownsPerSubcell; i++) {
//    uNewFine.push_back(peanoclaw::records::Data());
//    if(i < fineSubdivisionFactor*fineSubdivisionFactor) {
//      uNewFine.at(i).setU(1.0);
//    } else {
//      uNewFine.at(i).setU(0.5);
//    }
//  }
//  std::vector<peanoclaw::records::Data>& uOldFine
//    = DataHeap::getInstance()
//    .getData(finePatch.getUOldIndex());
//  for(int i = 0; i < (fineSubdivisionFactor+2*ghostlayerWidth) * (fineSubdivisionFactor+2*ghostlayerWidth) * unknownsPerSubcell; i++) {
//    uOldFine.push_back(peanoclaw::records::Data());
//    if(i < (fineSubdivisionFactor+2*ghostlayerWidth) * (fineSubdivisionFactor+2*ghostlayerWidth)) {
//      uOldFine.at(i).setU(1.0);
//    } else {
//      uOldFine.at(i).setU(0.5);
//    }
//  }

  //TODO unterweg debug
  logInfo("", "Fine Patch: \n" << finePatch.toStringUNew());
  logInfo("", "Coarse Patch: \n" << coarsePatch.toStringUNew());

  //GhostLayerCompositor
//  NumericsTestStump numerics;
//  peanoclaw::Patch patches[TWO_POWER_D];
//  interSubgridCommunication::GhostLayerCompositor ghostLayerCompositor(patches, 0, numerics, false);
  peanoclaw::interSubgridCommunication::DefaultFluxCorrection fluxCorrection;

  fluxCorrection.applyCorrection(finePatch, coarsePatch, 0, 1);

  //TODO unterweg debug
  logInfo("", "Coarse Patch after: \n" << coarsePatch.toStringUNew());

  assignList(subcellIndex) = 0, 0;
  validateNumericalEquals(coarsePatch.getValueUNew(subcellIndex, 0), 2558.0 / 729.0);
  assignList(subcellIndex) = 0, 1;
  validateNumericalEquals(coarsePatch.getValueUNew(subcellIndex, 0), 4466.0/729.0);
  assignList(subcellIndex) = 1, 0;
  validateNumericalEquals(coarsePatch.getValueUNew(subcellIndex, 0), 1.0);
  assignList(subcellIndex) = 1, 1;
  validateNumericalEquals(coarsePatch.getValueUNew(subcellIndex, 0), 1.0);
}

void peanoclaw::tests::GhostLayerCompositorTest::testInterpolationInTime() {
  Patch sourcePatch
    = createPatch(
      1,       //Unknowns per subcell
      0,       //Aux fields per subcell
      2,       //Subdivision factor
      1,       //Ghostlayer width
      0.0,     //Position
      1.0,     //Size
      0,       //Level
      0.0,     //Current time
      1.0,     //Timestep size
      1.0      //Minimal neighbor time
    );

  tarch::la::Vector<DIMENSIONS, double> sourcePosition;
  assignList(sourcePosition) = 1.0, 0.0;
  Patch destinationPatch
    = createPatch(
      1,              //Unknowns per subcell
      0,              //Aux fields per subcell
      2,              //Subdivision factor
      1,              //Ghostlayer width
      sourcePosition,
      1.0,            //Size
      0,              //Level
      0.0,            //Current time
      1.0/3.0,        //Timestep size
      1.0             //Minimal neighbor time
    );

  //Fill source
  tarch::la::Vector<DIMENSIONS, int> subcellIndex;
  assignList(subcellIndex) = 0, 0;
  sourcePatch.setValueUOld(subcellIndex, 0, 1.0);
  assignList(subcellIndex) = 1, 0;
  sourcePatch.setValueUOld(subcellIndex, 0, 1.0);
  assignList(subcellIndex) = 0, 1;
  sourcePatch.setValueUOld(subcellIndex, 0, 1.0);
  assignList(subcellIndex) = 1, 1;
  sourcePatch.setValueUOld(subcellIndex, 0, 1.0);

  assignList(subcellIndex) = 0, 0;
  sourcePatch.setValueUNew(subcellIndex, 0, 10.0);
  assignList(subcellIndex) = 1, 0;
  sourcePatch.setValueUNew(subcellIndex, 0, 20.0);
  assignList(subcellIndex) = 0, 1;
  sourcePatch.setValueUNew(subcellIndex, 0, 30.0);
  assignList(subcellIndex) = 1, 1;
  sourcePatch.setValueUNew(subcellIndex, 0, 40.0);

  peanoclaw::Patch patches[TWO_POWER_D];
  patches[3] = sourcePatch;
  patches[2] = destinationPatch;

  peanoclaw::tests::NumericsTestStump numerics;
  peanoclaw::interSubgridCommunication::GhostLayerCompositor ghostLayerCompositor
    = peanoclaw::interSubgridCommunication::GhostLayerCompositor(
        patches,
        0,
        numerics,
        true
      );
  //TODO Add a test for filling only one adjacent patch? (i.e. parameter != -1)
  ghostLayerCompositor.fillGhostLayers(-1);

  assignList(subcellIndex) = -1, 0;
  validateNumericalEquals(destinationPatch.getValueUOld(subcellIndex, 0), 320.0/3.0);
  assignList(subcellIndex) = -1, 1;
  validateNumericalEquals(destinationPatch.getValueUOld(subcellIndex, 0), 80.0/3.0);
}

void peanoclaw::tests::GhostLayerCompositorTest::testRestrictionWithOverlappingBounds() {
  tarch::la::Vector<DIMENSIONS, double> lowerNeighboringGhostlayerBounds(1.0);
  tarch::la::Vector<DIMENSIONS, double> upperNeighboringGhostlayerBounds(1.1);
  tarch::la::Vector<DIMENSIONS, double> sourceSubcellSize(0.11);
  tarch::la::Vector<DIMENSIONS, double> sourcePosition(0.0);
  tarch::la::Vector<DIMENSIONS, double> sourceSize(2.2);
  tarch::la::Vector<DIMENSIONS, int> sourceSubdivisionFactor(20);

  peanoclaw::interSubgridCommunication::DefaultRestriction defaultRestriction;
  peanoclaw::Area areas[TWO_POWER_D];
  defaultRestriction.getAreasForRestriction(
    lowerNeighboringGhostlayerBounds,
    upperNeighboringGhostlayerBounds,
    sourcePosition,
    sourceSize,
    sourceSubcellSize,
    sourceSubdivisionFactor,
    areas
  );

  tarch::la::Vector<DIMENSIONS,int> zero(0);

  for(int d = 0; d < DIMENSIONS; d++) {
    validateEqualsWithParams1(areas[2*d]._offset, zero, d);
    validateEqualsWithParams1(areas[2*d + 1]._offset, zero, d);

    if(d == 0) {
      validateEqualsWithParams1(areas[2*d]._size, sourceSubdivisionFactor, d);
      validateEqualsWithParams1(areas[2*d + 1]._size, zero, d);
    } else {
      validateEqualsWithParams1(areas[2*d]._size, zero, d);
      validateEqualsWithParams1(areas[2*d + 1]._size, zero, d);
    }
  }
}

void peanoclaw::tests::GhostLayerCompositorTest::testPartialRestrictionAreas() {
  #ifdef Dim2
  tarch::la::Vector<DIMENSIONS, double> lowerNeighboringGhostlayerBounds(1.8 + 1e-13);
  tarch::la::Vector<DIMENSIONS, double> upperNeighboringGhostlayerBounds(1.2);
  tarch::la::Vector<DIMENSIONS, double> sourcePosition(1.0);
  tarch::la::Vector<DIMENSIONS, double> sourceSize(1.0);
  tarch::la::Vector<DIMENSIONS, double> sourceSubcellSize(0.1);
  tarch::la::Vector<DIMENSIONS, int> sourceSubdivisionFactor(10);

  peanoclaw::interSubgridCommunication::DefaultRestriction defaultRestriction;
  peanoclaw::Area areas[TWO_POWER_D];
  defaultRestriction.getAreasForRestriction(
    lowerNeighboringGhostlayerBounds,
    upperNeighboringGhostlayerBounds,
    sourcePosition,
    sourceSize,
    sourceSubcellSize,
    sourceSubdivisionFactor,
    areas
  );

  tarch::la::Vector<DIMENSIONS,int> offset;
  tarch::la::Vector<DIMENSIONS,int> size;

  //Left and Right
  assignList(offset) = 0, 0;
  assignList(size) = 2, 10;
  validateEquals(areas[0]._offset, offset);
  validateEquals(areas[0]._size, size);
  assignList(offset) = 8, 0;
  assignList(size) = 2, 10;
  validateEquals(areas[1]._offset, offset);
  validateEquals(areas[1]._size, size);
  assignList(offset) = 2, 0;
  assignList(size) = 6, 2;
  validateEquals(areas[2]._offset, offset);
  validateEquals(areas[2]._size, size);
  assignList(offset) = 2, 8;
  assignList(size) = 6, 2;
  validateEquals(areas[3]._offset, offset);
  validateEquals(areas[3]._size, size);
  #endif
}

void peanoclaw::tests::GhostLayerCompositorTest::testPartialRestrictionAreasWithInfiniteLowerBounds() {
  #ifdef Dim2
  tarch::la::Vector<DIMENSIONS, double> lowerNeighboringGhostlayerBounds(std::numeric_limits<double>::max());
  tarch::la::Vector<DIMENSIONS, double> upperNeighboringGhostlayerBounds;
  assignList(upperNeighboringGhostlayerBounds) = 4.0/27.0, 6.0/27.0 + 1.0/27.0/6.0;
  tarch::la::Vector<DIMENSIONS, double> sourcePosition;
  assignList(sourcePosition) = 4.0/27.0, 2.0/9.0;
  tarch::la::Vector<DIMENSIONS, double> sourceSize(1.0/27.0);
  tarch::la::Vector<DIMENSIONS, double> sourceSubcellSize(1.0/27.0/6.0);
  tarch::la::Vector<DIMENSIONS, int> sourceSubdivisionFactor(6);

  peanoclaw::interSubgridCommunication::DefaultRestriction defaultRestriction;
  peanoclaw::Area areas[TWO_POWER_D];
  defaultRestriction.getAreasForRestriction(
      lowerNeighboringGhostlayerBounds,
      upperNeighboringGhostlayerBounds,
      sourcePosition,
      sourceSize,
      sourceSubcellSize,
      sourceSubdivisionFactor,
      areas
  );

  tarch::la::Vector<DIMENSIONS,int> offset;
  tarch::la::Vector<DIMENSIONS,int> size;

  //Left and Right
  assignList(offset) = 0, 0;
  assignList(size) = 0, 6;
  validateEquals(areas[0]._offset, offset);
  validateEquals(areas[0]._size, size);
  assignList(offset) = 6, 0;
  assignList(size) = 0, 6;
  validateEquals(areas[1]._offset, offset);
  validateEquals(areas[1]._size, size);
  assignList(offset) = 0, 0;
  assignList(size) = 6, 1;
  validateEquals(areas[2]._offset, offset);
  validateEquals(areas[2]._size, size);
  assignList(offset) = 0, 6;
  assignList(size) = 6, 0;
  validateEquals(areas[3]._offset, offset);
  validateEquals(areas[3]._size, size);
  #endif
}

void peanoclaw::tests::GhostLayerCompositorTest::testFaceAdjacentPatchTraversal2D() {
  #ifdef Dim2
  peanoclaw::Patch patches[TWO_POWER_D];
  TestFaceAdjacentPatchTraversalFunctor functor;
  peanoclaw::interSubgridCommunication::aspects::FaceAdjacentPatchTraversal<TestFaceAdjacentPatchTraversalFunctor>(
    patches,
    functor
  );

  validateEquals(functor._calls.size(), 8);

  //Patch3 -> Patch2
  validateEquals(functor._calls[0][0], 3);
  validateEquals(functor._calls[0][1], 2);
  validateEquals(functor._calls[0][2], 1);
  validateEquals(functor._calls[0][3], 0);

  //Patch3 -> Patch1
  validateEquals(functor._calls[1][0], 3);
  validateEquals(functor._calls[1][1], 1);
  validateEquals(functor._calls[1][2], 0);
  validateEquals(functor._calls[1][3], 1);

  //Patch2 -> Patch3
  validateEquals(functor._calls[2][0], 2);
  validateEquals(functor._calls[2][1], 3);
  validateEquals(functor._calls[2][2], -1);
  validateEquals(functor._calls[2][3], 0);

  //Patch2 -> Patch0
  validateEquals(functor._calls[3][0], 2);
  validateEquals(functor._calls[3][1], 0);
  validateEquals(functor._calls[3][2], 0);
  validateEquals(functor._calls[3][3], 1);

  //Patch1 -> Patch0
  validateEquals(functor._calls[4][0], 1);
  validateEquals(functor._calls[4][1], 0);
  validateEquals(functor._calls[4][2], 1);
  validateEquals(functor._calls[4][3], 0);

  //Patch1 -> Patch3
  validateEquals(functor._calls[5][0], 1);
  validateEquals(functor._calls[5][1], 3);
  validateEquals(functor._calls[5][2], 0);
  validateEquals(functor._calls[5][3], -1);

  //Patch0 -> Patch1
  validateEquals(functor._calls[6][0], 0);
  validateEquals(functor._calls[6][1], 1);
  validateEquals(functor._calls[6][2], -1);
  validateEquals(functor._calls[6][3], 0);

  //Patch0 -> Patch2
  validateEquals(functor._calls[7][0], 0);
  validateEquals(functor._calls[7][1], 2);
  validateEquals(functor._calls[7][2], 0);
  validateEquals(functor._calls[7][3], -1);
  #endif
}

void peanoclaw::tests::GhostLayerCompositorTest::testEdgeAdjacentPatchTraversal2D() {
  #ifdef Dim2
  peanoclaw::Patch patches[TWO_POWER_D];
  TestEdgeAdjacentPatchTraversalFunctor functor;
  peanoclaw::interSubgridCommunication::aspects::EdgeAdjacentPatchTraversal<TestEdgeAdjacentPatchTraversalFunctor>(
    patches,
    functor
  );

  validateEquals(functor._calls.size(), 4);

  //Lower-left <-> Upper-right
  validateEquals(functor._calls[0][0], 3);
  validateEquals(functor._calls[0][1], 0);
  validateEquals(functor._calls[0][2], 1);
  validateEquals(functor._calls[0][3], 1);

  //Lower-right <-> Upper-left
  validateEquals(functor._calls[1][0], 2);
  validateEquals(functor._calls[1][1], 1);
  validateEquals(functor._calls[1][2], -1);
  validateEquals(functor._calls[1][3], 1);

  //Upper-left <-> Lower-right
  validateEquals(functor._calls[2][0], 1);
  validateEquals(functor._calls[2][1], 2);
  validateEquals(functor._calls[2][2], 1);
  validateEquals(functor._calls[2][3], -1);

  //Upper-right <-> Lower-left
  validateEquals(functor._calls[3][0], 0);
  validateEquals(functor._calls[3][1], 3);
  validateEquals(functor._calls[3][2], -1);
  validateEquals(functor._calls[3][3], -1);
  #endif
}

void peanoclaw::tests::TestFaceAdjacentPatchTraversalFunctor::operator()(
  peanoclaw::Patch& patch1,
  int               index1,
  peanoclaw::Patch& patch2,
  int               index2,
  tarch::la::Vector<DIMENSIONS, int> direction
) {
  std::vector<int> parameters;
  parameters.push_back(index1);
  parameters.push_back(index2);
  for(int d = 0; d < DIMENSIONS; d++) {
    parameters.push_back(direction(d));
  }
  _calls.push_back(parameters);
}

void peanoclaw::tests::TestEdgeAdjacentPatchTraversalFunctor::operator()(
  peanoclaw::Patch&                  patch1,
  int                                index1,
  peanoclaw::Patch&                  patch2,
  int                                index2,
  tarch::la::Vector<DIMENSIONS, int> direction
) {
  std::vector<int> parameters;
  parameters.push_back(index1);
  parameters.push_back(index2);
  for(int d = 0; d < DIMENSIONS; d++) {
    parameters.push_back(direction(d));
  }
  _calls.push_back(parameters);
}

#ifdef UseTestSpecificCompilerSettings
#pragma optimize("",on)
#endif
