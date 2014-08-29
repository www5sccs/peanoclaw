/*
 * ConstantScenario.cpp
 *
 *  Created on: Aug 29, 2014
 *      Author: kristof
 */
#include "peanoclaw/native/scenarios/ConstantScenario.h"

#include "peanoclaw/Patch.h"

peanoclaw::native::scenarios::ConstantScenario::ConstantScenario(
  std::vector<std::string> arguments
) : _domainSize(1.0) {
  if(arguments.size() != 4) {
    std::cerr << "Expected arguments for Scenario 'ConstantScenario': subgridTopology subdivisionFactor endTime globalTimestepSize" << std::endl
        << "\tGot " << arguments.size() << " arguments." << std::endl;
    throw "";
  }

  double subgridTopologyPerDimension = atof(arguments[0].c_str());
  _demandedMeshWidth = _domainSize/ subgridTopologyPerDimension;

  _subdivisionFactor = tarch::la::Vector<DIMENSIONS,int>(atoi(arguments[1].c_str()));

  _endTime = atof(arguments[2].c_str());

  _globalTimestepSize = atof(arguments[3].c_str());
}

void peanoclaw::native::scenarios::ConstantScenario::initializePatch(peanoclaw::Patch& subgrid) {
  peanoclaw::grid::SubgridAccessor& accessor = subgrid.getAccessor();

  dfor(subcellIndex, subgrid.getSubdivisionFactor()) {
    accessor.setValueUNew(subcellIndex, 0, 1.0);
    for(int unknown = 1; unknown < subgrid.getUnknownsPerSubcell(); unknown++) {
      accessor.setValueUNew(subcellIndex, unknown, 0.0);
    }
    for(int parameter = 0; parameter < subgrid.getNumberOfParametersWithGhostlayerPerSubcell(); parameter++) {
      accessor.setParameterWithGhostlayer(subcellIndex, parameter, 0.0);
    }
    for(int parameter = 0; parameter < subgrid.getNumberOfParametersWithoutGhostlayerPerSubcell(); parameter++) {
      accessor.setParameterWithoutGhostlayer(subcellIndex, parameter, 0.0);
    }
  }
}

tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::ConstantScenario::computeDemandedMeshWidth(peanoclaw::Patch& patch, bool isInitializing) {
  return _demandedMeshWidth;
}

//PeanoClaw-Scenario
tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::ConstantScenario::getDomainOffset() const {
  return 0;
}
tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::ConstantScenario::getDomainSize() const {
  return _domainSize;
}
tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::ConstantScenario::getInitialMinimalMeshWidth() const {
  return _demandedMeshWidth;
}
tarch::la::Vector<DIMENSIONS,int> peanoclaw::native::scenarios::ConstantScenario::getSubdivisionFactor() const {
  return _subdivisionFactor;
}
double peanoclaw::native::scenarios::ConstantScenario::getGlobalTimestepSize() const {
  return _globalTimestepSize;
}
double peanoclaw::native::scenarios::ConstantScenario::getEndTime() const {
  return _endTime;
}

//pure SWE-Scenario
float peanoclaw::native::scenarios::ConstantScenario::getWaterHeight(float x, float y) {
  return 1.0f;
}
float peanoclaw::native::scenarios::ConstantScenario::waterHeightAtRest() {
  return 0.0;
}
float peanoclaw::native::scenarios::ConstantScenario::endSimulation() {
  return (float)getEndTime();
}







