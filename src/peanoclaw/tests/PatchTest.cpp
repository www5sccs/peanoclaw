/*
 * PatchTest.cpp
 *
 *  Created on: Feb 15, 2012
 *      Author: Kristof Unterweger
 */

#include "peanoclaw/tests/PatchTest.h"

#include "peanoclaw/tests/Helper.h"

#include "peano/utils/Globals.h"

#include "peanoclaw/Patch.h"

#include "tarch/tests/TestCaseFactory.h"
registerTest(peanoclaw::tests::PatchTest)


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
  testMethod(testCoarsePatchTimeInterval);
}

void peanoclaw::tests::PatchTest::testFillingOfUNewArray() {

//  Patch patch(
//    0.0,    //Position
//    1.0,    //Size
//    2,      //Number of unknowns
//    3,      //Subdivision factor
//    1,      //Ghostlayer width
//    0.0,    //Current time
//    0.0,    //Timestep size
//    0.0,    //CFL
//    0.0,    //Maximum timestep size
//    0,      //Level
//    &uNew,
//    &uOld
//  );

  Patch patch = createPatch(
    2,   //Unknowns per subcell
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

  for(int unknown = 0; unknown < 2; unknown++) {
    for(int x = 0; x < 3; x++) {
      for(int y = 0; y < 3; y++) {
        tarch::la::Vector<DIMENSIONS, int> subcellIndex;
        assignList(subcellIndex) = x, y;
        patch.setValueUNew(subcellIndex, unknown, unknown * 9 + x * 3 + y);
      }
    }
  }

  //TODO unterweg fixen!
//  double uNewArray[3*3*2];
//  patch.fillUNewArray(uNewArray);
//
//  for(int i = 0; i < 18; i++) {
//    validateNumericalEquals(uNewArray[i], i);
//  }
}

void peanoclaw::tests::PatchTest::testFillingOfUOldArray() {
//  Patch patch(
//    0.0,    //Position
//    1.0,    //Size
//    2,      //Number of unknowns
//    3,      //Subdivision factor
//    1,      //Ghostlayer width
//    0.0,    //Current time
//    0.0,    //Timestep size
//    0.0,    //CFL
//    0.0,    //Maximum timestep size
//    0,      //Level
//    &uNew,
//    &uOld
//  );
  Patch patch = createPatch(
    2,       //Unknowns per subcell
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

//  std::vector<Data> uNew;
//  std::vector<Data> uOld;
//  for(int i = 0; i < 50; i++) {
//    Data data;
//    data.setU(i);
//    uOld.push_back(data);
//  }

  int counter = 0;
  for(int unknown = 0; unknown < 2; unknown++) {
    for(int x = -1; x < 3 + 1; x++) {
      for(int y = -1; y < 3 + 1; y++) {
        tarch::la::Vector<DIMENSIONS, int> subcellIndex;
        assignList(subcellIndex) = x, y;
        patch.setValueUOld(subcellIndex, unknown, counter++);
      }
    }
  }

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
  finePatches[0].setCurrentTime(1.0);
  finePatches[0].setTimestepSize(2.0);

  finePatches[1].setCurrentTime(1.0);
  finePatches[1].setTimestepSize(2.0);

  finePatches[2].setCurrentTime(1.0);
  finePatches[2].setTimestepSize(3.0);

  finePatches[3].setCurrentTime(0.0);
  finePatches[3].setTimestepSize(3.0);

  finePatches[4].setCurrentTime(1.0);
  finePatches[4].setTimestepSize(1.7);

  finePatches[5].setCurrentTime(0.0);
  finePatches[5].setTimestepSize(4.0);

  finePatches[6].setCurrentTime(0.0);
  finePatches[6].setTimestepSize(3.0);

  finePatches[7].setCurrentTime(1.0);
  finePatches[7].setTimestepSize(2.0);

  finePatches[8].setCurrentTime(2.5);
  finePatches[8].setTimestepSize(1.5);

  //execute methods
  coarsePatch.resetMinimalFineGridTimeInterval();
  for(int i = 0; i < THREE_POWER_D; i++) {
    coarsePatch.updateMinimalFineGridTimeInterval(finePatches[i].getCurrentTime(), finePatches[i].getTimestepSize());
  }
  coarsePatch.switchValuesAndTimeIntervalToMinimalFineGridTimeInterval();

  //Check
  validateNumericalEquals(coarsePatch.getCurrentTime(), 2.5);
  validateNumericalEquals(coarsePatch.getTimestepSize(), 0.2);
}

#ifdef UseTestSpecificCompilerSettings
#pragma optimize("",on)
#endif

