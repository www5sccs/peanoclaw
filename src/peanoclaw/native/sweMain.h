/*
 * sweMain.h
 *
 *  Created on: May 30, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_NATIVE_SWE_CPP_
#define PEANOCLAW_NATIVE_SWE_CPP_

#include "peanoclaw/native/BreakingDam.h"
#include "tools/Logger.hh"

void sweMain(
  peanoclaw::native::BreakingDam_SWEKernelScenario& scenario,
  tarch::la::Vector<DIMENSIONS,int> numberOfCells
);

#endif /* PEANOCLAW_NATIVE_SWE_CPP_ */
