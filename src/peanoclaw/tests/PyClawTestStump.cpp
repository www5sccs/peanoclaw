/*
 * PyClawTestStump.cpp
 *
 *  Created on: May 24, 2012
 *      Author: unterweg
 */

#include "peanoclaw/interSubgridCommunication/DefaultInterpolation.h"
#include "peanoclaw/interSubgridCommunication/DefaultRestriction.h"
#include "peanoclaw/interSubgridCommunication/DefaultFluxCorrection.h"
#include "peanoclaw/tests/PyClawTestStump.h"

peanoclaw::tests::PyClawTestStump::PyClawTestStump()
 : peanoclaw::pyclaw::PyClaw (
     0,
     0,
     0,
     0,
     new peanoclaw::interSubgridCommunication::DefaultInterpolation,
     new peanoclaw::interSubgridCommunication::DefaultRestriction,
     new peanoclaw::interSubgridCommunication::DefaultFluxCorrection
) {

}
