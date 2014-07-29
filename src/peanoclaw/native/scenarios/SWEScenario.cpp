/*
 * SWEScenario.cpp
 *
 *  Created on: Jun 10, 2014
 *      Author: kristof
 */
#include "peanoclaw/native/scenarios/SWEScenario.h"

#include "peanoclaw/native/scenarios/BowlOcean.h"
#include "peanoclaw/native/scenarios/BreakingDam.h"
#include "peanoclaw/native/scenarios/CalmOcean.h"
#include "peanoclaw/native/scenarios/Gaussian.h"
#include "peanoclaw/native/scenarios/MekkaFlood.h"


#include "peanoclaw/Patch.h"

tarch::logging::Log peanoclaw::native::scenarios::SWEScenario::_log("peanoclaw::native::scenarios::SWEScenario");

peanoclaw::native::scenarios::SWEScenario* peanoclaw::native::scenarios::SWEScenario::createScenario(int argc, char** argv) {

  if(argc == 1) {
  #ifdef PEANOCLAW_FULLSWOF2D
    DEM dem;

    dem.load("DEM_400cm.bin");

    // keep aspect ratio of map: 4000 3000: ratio 4:3
    tarch::la::Vector<DIMENSIONS, int> subdivisionFactor;
    subdivisionFactor(0) = static_cast<int>(16); // 96 //  6 * 4, optimum in non optimized version
    subdivisionFactor(1) = static_cast<int>(9);  // 54 //  6 * 3, optimum in non optimized version

    tarch::la::Vector<DIMENSIONS, double> maximalMeshWidth(1.0/(9.0 * subdivisionFactor(0)));
    tarch::la::Vector<DIMENSIONS, double> minimalMeshWidth(1.0/(81.0 * subdivisionFactor(0)));

    double globalTimestepSize = 2.0; //0.1;//1.0;
    double endTime = 3600.0; // 100.0;

    bool usePeanoClaw = true;

    peanoclaw::native::scenarios::MekkaFloodSWEScenario* scenario
      = new peanoclaw::native::scenarios::MekkaFloodSWEScenario(
            dem,
            subdivisionFactor,
            minimalMeshWidth,
            maximalMeshWidth,
            globalTimestepSize,
            endTime
          );
    std::cout << "domainSize " << scenario->getDomainSize() << std::endl;
    return scenario;
  #elif PEANOCLAW_SWE
    //Default Scenario
    tarch::la::Vector<DIMENSIONS, double> domainOffset(0);
    tarch::la::Vector<DIMENSIONS, double> domainSize(10.0);
    tarch::la::Vector<DIMENSIONS, int> finestSubgridTopology(9);
    tarch::la::Vector<DIMENSIONS, int> coarsestSubgridTopology(3);

    double globalTimestepSize = 0.1;
// double endTime = 2.0; //0.5; // 100.0;
    double endTime = 0.5; // 100.0;
    tarch::la::Vector<DIMENSIONS, int> subdivisionFactor(6);

    return new peanoclaw::native::scenarios::BreakingDamSWEScenario(
      domainOffset,
      domainSize,
      finestSubgridTopology,
      coarsestSubgridTopology,
      subdivisionFactor,
      globalTimestepSize,
      endTime
    );
  #endif
  } else {
    std::string scenarioName = std::string(argv[1]);

    std::vector<std::string> arguments;
    for(int i = 2; i < argc; i++) {
      arguments.push_back(std::string(argv[i]));
    }

    if(scenarioName == "breakingDam") {
      return new peanoclaw::native::scenarios::BreakingDamSWEScenario(arguments);
    } else if(scenarioName == "calmOcean") {
      return new peanoclaw::native::scenarios::CalmOcean(arguments);
    } else if(scenarioName == "bowlOcean") {
      return new peanoclaw::native::scenarios::BowlOcean(arguments);
    } else if(scenarioName == "gaussian") {
      return new peanoclaw::native::scenarios::GaussianSWEScenario(arguments);
    } else {
      std::cerr << "Unknown scenario '" << scenarioName << "'." << std::endl;
    }

    return 0;
  }
  return 0;
}

double peanoclaw::native::scenarios::SWEScenario::getInitialTimestepSize() const {
#if defined(PEANOCLAW_SWE) || defined(PEANOCLAW_EULER3D)
  return 0.1;
#elif defined(PEANOCLAW_FULLSWOF2D)
  return 1.0;
#endif
}

