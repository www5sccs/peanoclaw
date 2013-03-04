#include "tests/TestCase.h"


#include "tarch/compiler/CompilerSpecificSettings.h"
#include "tarch/tests/TestCaseFactory.h"
registerTest(peanoclaw::tests::TestCase)


#ifdef UseTestSpecificCompilerSettings
#pragma optimize("",off)
#endif

 
peanoclaw::tests::TestCase::TestCase():
  tarch::tests::TestCase( "peanoclaw::tests::TestCase" ) {
}


peanoclaw::tests::TestCase::~TestCase() {
}


void peanoclaw::tests::TestCase::run() {
  // @todo If you have further tests, add them here
  testMethod( test1 );
  testMethod( test2 );
  testMethod( test3 );
}


void peanoclaw::tests::TestCase::test1() {
  // @todo Add your test here
  validateEquals(1,1);
}


void peanoclaw::tests::TestCase::test2() {
  // @todo Add your test here
  validateEquals(2,2);
}


void peanoclaw::tests::TestCase::test3() {
  // @todo Add your test here
  validateEquals(3,3);
}


#ifdef UseTestSpecificCompilerSettings
#pragma optimize("",on)
#endif
