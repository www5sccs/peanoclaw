/*
 * NumericsFactory.cpp
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */
#include "peanoclaw/NumericsFactory.h"

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
#include "peanoclaw/interSubgridCommunication/DefaultTransfer.h"

tarch::logging::Log peanoclaw::NumericsFactory::_log("peanoclaw::NumericsFactory");

#if defined(SWE)
peanoclaw::Numerics* peanoclaw::NumericsFactory::createSWENumerics(
  peanoclaw::native::scenarios::SWEScenario& scenario
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
    new peanoclaw::interSubgridCommunication::DefaultTransfer,
    interpolation,
    restriction,
    fluxCorrection
  );
}
#endif

#if defined(PEANOCLAW_PYCLAW)

peanoclaw::Numerics* peanoclaw::NumericsFactory::createPyClawNumerics(
  InitializationCallback initializationCallback,
  BoundaryConditionCallback boundaryConditionCallback,
  SolverCallback solverCallback,
  RefinementCriterionCallback refinementCriterionCallback,
  AddPatchToSolutionCallback addPatchToSolutionCallback,
  InterPatchCommunicationCallback interpolationCallback,
  InterPatchCommunicationCallback restrictionCallback,
  InterPatchCommunicationCallback fluxCorrectionCallback,
  int numberOfUnknowns,
  int numberOfAuxFields,
  int ghostlayerWidth
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
    refinementCriterionCallback,
    addPatchToSolutionCallback,
    new peanoclaw::interSubgridCommunication::DefaultTransfer,
    interpolation,
    restriction,
    fluxCorrection,
    numberOfUnknowns,
    numberOfAuxFields,
    ghostlayerWidth
  );
}
#endif

#if defined(PEANOCLAW_FULLSWOF2D)
peanoclaw::Numerics* peanoclaw::NumericsFactory::createFullSWOF2DNumerics(
  peanoclaw::native::scenarios::SWEScenario& scenario
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

  return new peanoclaw::native::FullSWOF2D(
    scenario,
    new peanoclaw::interSubgridCommunication::DefaultTransfer,
    interpolation,
    restriction,
    fluxCorrection
  );
}
#endif

#if defined(PEANOCLAW_EULER3D)
peanoclaw::Numerics* peanoclaw::NumericsFactory::createEuler3DNumerics(
  peanoclaw::native::scenarios::SWEScenario& scenario
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

  return new peanoclaw::solver::euler3d::Euler3DKernel(
    scenario,
    new peanoclaw::interSubgridCommunication::DefaultTransfer,
    interpolation,
    restriction,
    fluxCorrection
  );
}
#endif


