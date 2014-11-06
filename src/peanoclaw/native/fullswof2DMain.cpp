/*
 * fullswof2DMain.cpp
 *
 *  Created on: Jul 7, 2014
 *      Author: kristof
 */
#include "peanoclaw/native/fullswof2DMain.h"

#include "peanoclaw/statistics/MemoryInformation.h"

#include "tarch/logging/Log.h"
#include "tarch/timing/Watch.h"

#include "peanoclaw/native/MekkaFlood_solver.h"
#ifndef CHOICE_SCHEME_HPP
#include "choice_scheme.hpp"
#endif

#include "peanoclaw/native/FullSWOF2D.h"

void peanoclaw::native::fullswof2DMain(
  peanoclaw::native::scenarios::SWEScenario& scenario,
  tarch::la::Vector<DIMENSIONS,int> numberOfCells
) {
  tarch::logging::Log _log("peanoclaw::native::fullswof2DMain(...)");

  int ghostlayerWidth = 1;

  FullSWOF2D_Parameters parameters(
    ghostlayerWidth,
    numberOfCells[0],
    numberOfCells[1],
    scenario.getInitialMinimalMeshWidth()[0],
    scenario.getInitialMinimalMeshWidth()[1],
    scenario.getDomainSize(),
    scenario.getEndTime(),
    #ifdef PEANOCLAW_FULLSWOF2D
    scenario.enableRain(),
    #else
    true,
    #endif
    scenario.getBoundaryCondition(0, false),
    scenario.getBoundaryCondition(0, true),
    scenario.getBoundaryCondition(1, false),
    scenario.getBoundaryCondition(1, true)
  );

  Choice_scheme * schemeWrapper;
  schemeWrapper = new Choice_scheme(parameters);
  Scheme* scheme = schemeWrapper->getInternalScheme();

  tarch::la::Vector<DIMENSIONS,int> subcellIndex;

  /** Water height.*/
  TAB& h = scheme->getH();
  for (int x = -ghostlayerWidth; x < numberOfCells(0)+ghostlayerWidth; x++) {
    for (int y = -ghostlayerWidth; y < numberOfCells(1)+ghostlayerWidth; y++) {
      assignList(subcellIndex) = x, y;
      tarch::la::Vector<DIMENSIONS,double> position
        = subcellIndex.convertScalar<double>() * tarch::la::invertEntries(numberOfCells.convertScalar<double>()) * scenario.getDomainSize();
      position += scenario.getDomainOffset();
      h[x+ghostlayerWidth][y+ghostlayerWidth] = scenario.getWaterHeight(
        (float)x / numberOfCells[0] * scenario.getDomainSize()[0] + scenario.getDomainOffset()[0],
        (float)y / numberOfCells[1] * scenario.getDomainSize()[1] + scenario.getDomainOffset()[1]
      );
    }
  }

  /** X Velocity.*/
  TAB& u = scheme->getU();
  for (int x = -ghostlayerWidth; x < numberOfCells(0)+ghostlayerWidth; x++) {
    for (int y = -ghostlayerWidth; y < numberOfCells(1)+ghostlayerWidth; y++) {
        assignList(subcellIndex) = x, y;
        u[x+ghostlayerWidth][y+ghostlayerWidth] = scenario.getVeloc_u(
          (float)x / numberOfCells[0] * scenario.getDomainSize()[0] + scenario.getDomainOffset()[0],
          (float)y / numberOfCells[1] * scenario.getDomainSize()[1] + scenario.getDomainOffset()[1]
        );
    }
  }

  /** Y Velocity.*/
  TAB& v = scheme->getV();
  for (int x = -ghostlayerWidth; x < numberOfCells(0)+ghostlayerWidth; x++) {
    for (int y = -ghostlayerWidth; y < numberOfCells(1)+ghostlayerWidth; y++) {
      assignList(subcellIndex) = x, y;
      v[x+ghostlayerWidth][y+ghostlayerWidth] = scenario.getVeloc_v(
        (float)x / numberOfCells[0] * scenario.getDomainSize()[0] + scenario.getDomainOffset()[0],
        (float)y / numberOfCells[1] * scenario.getDomainSize()[1] + scenario.getDomainOffset()[1]
      );
    }
  }

  /** Topography.*/
  TAB& z = scheme->getZ();
  for (int x = -ghostlayerWidth; x < numberOfCells(0)+ghostlayerWidth; x++) {
    for (int y = -ghostlayerWidth; y < numberOfCells(1)+ghostlayerWidth; y++) {
      assignList(subcellIndex) = x, y;
      z[x+ghostlayerWidth][y+ghostlayerWidth] = scenario.getBathymetry(
        (float)x / numberOfCells[0] * scenario.getDomainSize()[0] + scenario.getDomainOffset()[0],
        (float)y / numberOfCells[1] * scenario.getDomainSize()[1] + scenario.getDomainOffset()[1]
      );
    }
  }

  tarch::timing::Watch runtimeWatch("peanoclaw::native", "fullswof2DMain", true);

  double t = 0.0;
  while(tarch::la::smaller(t, scenario.getEndTime())) {

    //TODO unterweg debug
    std::cout << "t=" << t << std::endl;

    scheme->resetN();
    schemeWrapper->calcul();
    t += scheme->getTimestep();
  }

  runtimeWatch.stopTimer();

  //Print maximum memory demand
  logInfo("fullswof2DMain", "Peak resident set size: " << peanoclaw::statistics::getPeakRSS() << "b");
  delete scheme;
}



