/*
 * fullswof2DMain.h
 *
 *  Created on: Jul 7, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_NATIVE_FULLSWOF2DMAIN_H_
#define PEANOCLAW_NATIVE_FULLSWOF2DMAIN_H_

#include "peanoclaw/native/scenarios/SWEScenario.h"

namespace peanoclaw {
  namespace native {
    void fullswof2DMain(
      peanoclaw::native::scenarios::SWEScenario& scenario,
      tarch::la::Vector<DIMENSIONS,int> numberOfCells
    );
  }
}

#endif /* PEANOCLAW_NATIVE_FULLSWOF2DMAIN_H_ */
