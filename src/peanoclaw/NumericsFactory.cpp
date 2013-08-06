/*
 * NumericsFactory.cpp
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */
#include "peanoclaw/NumericsFactory.h"

#include "peanoclaw/native/NativeKernel.h"
#include "peanoclaw/pyclaw/PyClaw.h"
#include "peanoclaw/pyclaw/InterpolationCallbackWrapper.h"
#include "peanoclaw/pyclaw/RestrictionCallbackWrapper.h"
#include "peanoclaw/pyclaw/FluxCorrectionCallbackWrapper.h"
#include "peanoclaw/Numerics.h"

#include "peanoclaw/interSubgridCommunication/Interpolation.h"
#include "peanoclaw/interSubgridCommunication/DefaultInterpolation.h"
#include "peanoclaw/interSubgridCommunication/Restriction.h"
#include "peanoclaw/interSubgridCommunication/DefaultRestriction.h"
#include "peanoclaw/interSubgridCommunication/FluxCorrection.h"
#include "peanoclaw/interSubgridCommunication/DefaultFluxCorrection.h"

tarch::logging::Log peanoclaw::NumericsFactory::_log("peanoclaw::NumericsFactory");

peanoclaw::Numerics* peanoclaw::NumericsFactory::createPyClawNumerics(
  InitializationCallback initializationCallback,
  BoundaryConditionCallback boundaryConditionCallback,
  SolverCallback solverCallback,
  AddPatchToSolutionCallback addPatchToSolutionCallback,
  InterPatchCommunicationCallback interpolationCallback,
  InterPatchCommunicationCallback restrictionCallback,
  InterPatchCommunicationCallback fluxCorrectionCallback
) {

  //Interpolation Callback
  peanoclaw::interSubgridCommunication::Interpolation* interpolation;
  if(interpolationCallback == 0) {
    logInfo("createPyClawNumerics", "Using default interpolation.");
    interpolation = new peanoclaw::interSubgridCommunication::DefaultInterpolation();
  } else {
    logInfo("createPyClawNumerics", "Using PyClaw interpolation.");
    interpolation = new peanoclaw::pyclaw::InterpolationCallbackWrapper(interpolationCallback);
  }

  //Restriction Callback
  peanoclaw::interSubgridCommunication::Restriction* restriction;
  if(restrictionCallback == 0) {
    logInfo("createPyClawNumerics", "Using default restriction.");
    restriction = new peanoclaw::interSubgridCommunication::DefaultRestriction();
  } else {
    logInfo("createPyClawNumerics", "Using PyClaw restriction.");
    restriction = new peanoclaw::pyclaw::RestrictionCallbackWrapper(restrictionCallback);
  }

  //Flux Correction Callback
  peanoclaw::interSubgridCommunication::FluxCorrection* fluxCorrection;
  if(fluxCorrectionCallback == 0) {
    logInfo("createPyClawNumerics", "Using default flux correction.");
    fluxCorrection = new peanoclaw::interSubgridCommunication::DefaultFluxCorrection();
  } else {
    logInfo("createPyClawNumerics", "Using PyClaw flux correction.");
    fluxCorrection = new peanoclaw::pyclaw::FluxCorrectionCallbackWrapper(fluxCorrectionCallback);
  }

  return new peanoclaw::pyclaw::PyClaw(
    initializationCallback,
    boundaryConditionCallback,
    solverCallback,
    addPatchToSolutionCallback,
    interpolation,
    restriction,
    fluxCorrection
  );
}

#if defined(SWE)
peanoclaw::Numerics* peanoclaw::NumericsFactory::createSWENumerics(
  peanoclaw::native::SWEKernelScenario& scenario
) {

  //Interpolation Callback
  peanoclaw::interSubgridCommunication::Interpolation* interpolation;
  interpolation = new peanoclaw::interSubgridCommunication::DefaultInterpolation();

  //Restriction Callback
  peanoclaw::interSubgridCommunication::Restriction* restriction;
  restriction = new peanoclaw::interSubgridCommunication::DefaultRestriction();

  //Flux Correction Callback
  peanoclaw::interSubgridCommunication::FluxCorrection* fluxCorrection;
  fluxCorrection = new peanoclaw::interSubgridCommunication::DefaultFluxCorrection();

  return new peanoclaw::native::SWEKernel(
    scenario,
    interpolation,
    restriction,
    fluxCorrection
  );
}
#endif

peanoclaw::Numerics* peanoclaw::NumericsFactory::createNativeNumerics() {
  return new peanoclaw::native::NativeKernel;
}



