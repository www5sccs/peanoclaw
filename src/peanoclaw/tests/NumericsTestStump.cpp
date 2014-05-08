/*
 * NumericsTestStump.cpp
 *
 *  Created on: May 24, 2012
 *      Author: unterweg
 */
#include "peanoclaw/tests/NumericsTestStump.h"

#include "peanoclaw/interSubgridCommunication/DefaultInterpolation.h"
#include "peanoclaw/interSubgridCommunication/DefaultRestriction.h"
#include "peanoclaw/interSubgridCommunication/DefaultFluxCorrection.h"
#include "peanoclaw/interSubgridCommunication/DefaultTransfer.h"

peanoclaw::tests::NumericsTestStump::NumericsTestStump()
 : peanoclaw::Numerics (
     new peanoclaw::interSubgridCommunication::DefaultTransfer,
     new peanoclaw::interSubgridCommunication::DefaultInterpolation,
     new peanoclaw::interSubgridCommunication::DefaultRestriction,
     new peanoclaw::interSubgridCommunication::DefaultFluxCorrection
) {
}
