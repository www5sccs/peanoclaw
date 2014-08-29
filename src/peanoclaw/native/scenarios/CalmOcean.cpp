/*
 * CalmOcean.cpp
 *
 *  Created on: Jun 10, 2014
 *      Author: kristof
 */
#include "peanoclaw/native/scenarios/CalmOcean.h"

#include "peanoclaw/Patch.h"

peanoclaw::native::scenarios::CalmOcean::CalmOcean(
  std::vector<std::string> arguments
) : _domainSize(1.0) {
  if(arguments.size() != 4) {
    std::cerr << "Expected arguments for Scenario 'CalmOcean': subgridTopology subdivisionFactor endTime globalTimestepSize" << std::endl
        << "\tGot " << arguments.size() << " arguments." << std::endl;
    throw "";
  }

  double subgridTopologyPerDimension = atof(arguments[0].c_str());
  _demandedMeshWidth = _domainSize/ subgridTopologyPerDimension;

  _subdivisionFactor = tarch::la::Vector<DIMENSIONS,int>(atoi(arguments[1].c_str()));

  _endTime = atof(arguments[2].c_str());

  _globalTimestepSize = atof(arguments[3].c_str());
}

void peanoclaw::native::scenarios::CalmOcean::initializePatch(peanoclaw::Patch& patch) {
  const tarch::la::Vector<DIMENSIONS, double> patchPosition = patch.getPosition();
  const tarch::la::Vector<DIMENSIONS, double> meshWidth = patch.getSubcellSize();
  peanoclaw::grid::SubgridAccessor& accessor = patch.getAccessor();

  tarch::la::Vector<DIMENSIONS, int> subcellIndex;
  for (int yi = 0; yi < patch.getSubdivisionFactor()(1); yi++) {
    for (int xi = 0; xi < patch.getSubdivisionFactor()(0); xi++) {
      subcellIndex(0) = xi;
      subcellIndex(1) = yi;

      double X = patchPosition(0) + xi*meshWidth(0);
      double Y = patchPosition(1) + yi*meshWidth(1);

      double q0 = getWaterHeight(X, Y);
      double q1 = 0.0;
      double q2 = 0.0;

      accessor.setValueUNew(subcellIndex, 0, q0);
      accessor.setValueUNew(subcellIndex, 1, q1);
      accessor.setValueUNew(subcellIndex, 2, q2);

      accessor.setParameterWithGhostlayer(subcellIndex, 0, 0.0);

      assertion2(tarch::la::equals(accessor.getValueUNew(subcellIndex, 0), q0, 1e-5), accessor.getValueUNew(subcellIndex, 0), q0);
      assertion2(tarch::la::equals(accessor.getValueUNew(subcellIndex, 1), q1, 1e-5), accessor.getValueUNew(subcellIndex, 1), q1);
      assertion2(tarch::la::equals(accessor.getValueUNew(subcellIndex, 2), q2, 1e-5), accessor.getValueUNew(subcellIndex, 2), q2);
    }
  }
}

tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::CalmOcean::computeDemandedMeshWidth(peanoclaw::Patch& patch, bool isInitializing) {
  return _demandedMeshWidth;
}

//PeanoClaw-Scenario
tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::CalmOcean::getDomainOffset() const {
  return 0;
}
tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::CalmOcean::getDomainSize() const {
  return _domainSize;
}
tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::CalmOcean::getInitialMinimalMeshWidth() const {
  return _demandedMeshWidth;
}
tarch::la::Vector<DIMENSIONS,int> peanoclaw::native::scenarios::CalmOcean::getSubdivisionFactor() const {
  return _subdivisionFactor;
}
double peanoclaw::native::scenarios::CalmOcean::getGlobalTimestepSize() const {
  return _globalTimestepSize;
}
double peanoclaw::native::scenarios::CalmOcean::getEndTime() const {
  return _endTime;
}

//pure SWE-Scenario
float peanoclaw::native::scenarios::CalmOcean::getWaterHeight(float x, float y) {
  return 1.0f;
}
float peanoclaw::native::scenarios::CalmOcean::waterHeightAtRest() {
  return getWaterHeight(0, 0);
}
float peanoclaw::native::scenarios::CalmOcean::endSimulation() {
  return (float)getEndTime();
}



