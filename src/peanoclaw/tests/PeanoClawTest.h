// Copyright (C) 2009 Technische Universitaet Muenchen 
// This file is part of the Peano project. For conditions of distribution and 
// use, please see the copyright notice at www5.in.tum.de/peano
#ifndef _PEANO_APPLICATIONS_PEANOCLAW_TESTS_PeanoClawTest_H_
#define _PEANO_APPLICATIONS_PEANOCLAW_TESTS_PeanoClawTest_H_
 


#include "tarch/tests/TestCase.h"


namespace peano { 
  namespace applications { 
    namespace peanoclaw {
      namespace tests {
      class PeanoClawTest;
      } 
}
}
}
 

/**
 * This is just a default test case that demonstrated how to write unit tests 
 * in Peano. Feel free to rename, remove, or duplicate it. It is not required 
 * by the project but often useful if you wanna write unit tests.
 */ 
class peano::applications::peanoclaw::tests::PeanoClawTest: public tarch::tests::TestCase {
  private:
    /**
     * Test wether the getters and setters for the adjacent patch indices in the
     * vertex are correct.
     */
    void testVertexAdjacentIndices();

    /**
     * Test wether the getters and setters for the patch indices in the
     * cell are correct.
     */
    void testCellIndices();


  public: 
    PeanoClawTest(); 
    virtual ~PeanoClawTest();
     
    virtual void run();

    void virtual setUp();
};


#endif
