/*
 * fullswof2DMain.cpp
 *
 *  Created on: Jul 7, 2014
 *      Author: kristof
 */
#include "peanoclaw/native/fullswof2DMain.h"

#include "peanoclaw/native/MekkaFlood_solver.h"
#ifndef CHOICE_SCHEME_HPP
#include "choice_scheme.hpp"
#endif

#include "peanoclaw/native/FullSWOF2D.h"

void peanoclaw::native::fullswof2DMain(
  peanoclaw::native::scenarios::SWEScenario& scenario,
  tarch::la::Vector<DIMENSIONS,int> numberOfCells
) {
  FullSWOF2D_Parameters parameters(
    1, //Ghostlayer Width
    numberOfCells[0],
    numberOfCells[1],
    scenario.getInitialMinimalMeshWidth()[0],
    scenario.getInitialMinimalMeshWidth()[1],
    scenario.getEndTime()
  );

  Choice_scheme * scheme;
  scheme = new Choice_scheme(parameters);

  //Set bathymetry

  double t = 0.0;
  while(tarch::la::smaller(t, scenario.getEndTime())) {
    scheme->calcul();
    t += scheme->getInternalScheme()->getTimestep();
  }

  delete scheme;
}



