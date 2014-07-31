/*
 * ShockBubble.cpp
 *
 *  Created on: Jul 30, 2014
 *      Author: kristof
 */

#include "peanoclaw/Patch.h"
#include "peanoclaw/native/scenarios/ShockBubble.h"

peanoclaw::native::scenarios::ShockBubble::ShockBubble(
  const tarch::la::Vector<DIMENSIONS, double>& domainOffset,
  const tarch::la::Vector<DIMENSIONS, double>& domainSize,
  const tarch::la::Vector<DIMENSIONS, int>&    finestSubgridTopology,
  const tarch::la::Vector<DIMENSIONS, int>&    coarsestSubgridTopology,
  const tarch::la::Vector<DIMENSIONS, int>&    subdivisionFactor,
  double                                       globalTimestepSize,
  double                                       endTime
) : _domainOffset(domainOffset),
    _domainSize(domainSize),
    _minimalMeshWidth(-1),
    _maximalMeshWidth(-1),
    _subdivisionFactor(subdivisionFactor),
    _globalTimestepSize(globalTimestepSize),
    _endTime(endTime)
{
  _minimalMeshWidth
    = tarch::la::multiplyComponents(domainSize, tarch::la::invertEntries(finestSubgridTopology.convertScalar<double>()));
  _maximalMeshWidth
    = tarch::la::multiplyComponents(domainSize, tarch::la::invertEntries(coarsestSubgridTopology.convertScalar<double>()));

  //  assertion2(tarch::la::allGreaterEquals(_maximalMeshWidth, _minimalMeshWidth), _minimalMeshWidth, _maximalMeshWidth);
  assertion2(tarch::la::allSmallerEquals(_minimalMeshWidth, _maximalMeshWidth), _minimalMeshWidth, _maximalMeshWidth);
}

peanoclaw::native::scenarios::ShockBubble::ShockBubble(
  std::vector<std::string> arguments
) : _domainOffset(0),
    _domainSize(1){
  if(arguments.size() != 5) {
    std::cerr << "Expected arguments for Scenario 'BreakingDam': finestSubgridTopology coarsestSubgridTopology subdivisionFactor endTime globalTimestepSize" << std::endl
        << "\tGot " << arguments.size() << " arguments." << std::endl;
    throw "";
  }

  double finestSubgridTopologyPerDimension = atof(arguments[0].c_str());
  _minimalMeshWidth = _domainSize / finestSubgridTopologyPerDimension;

  double coarsestSubgridTopologyPerDimension = atof(arguments[1].c_str());
  _maximalMeshWidth = _domainSize / coarsestSubgridTopologyPerDimension;

  _subdivisionFactor = tarch::la::Vector<DIMENSIONS,int>(atoi(arguments[2].c_str()));

  _endTime = atof(arguments[3].c_str());

  _globalTimestepSize = atof(arguments[4].c_str());
}

peanoclaw::native::scenarios::ShockBubble::~ShockBubble() {
}

void peanoclaw::native::scenarios::ShockBubble::initializePatch(peanoclaw::Patch& subgrid) {
  // compute from mesh data
  const tarch::la::Vector<DIMENSIONS, double> subgridPosition = subgrid.getPosition();
  const tarch::la::Vector<DIMENSIONS, double> meshWidth = subgrid.getSubcellSize();
  peanoclaw::grid::SubgridAccessor& accessor = subgrid.getAccessor();

  tarch::la::Vector<DIMENSIONS, int> subcellIndex;
  for (int zi = 0; zi < subgrid.getSubdivisionFactor()(2); zi++) {
    for (int yi = 0; yi < subgrid.getSubdivisionFactor()(1); yi++) {
      for (int xi = 0; xi < subgrid.getSubdivisionFactor()(0); xi++) {
        subcellIndex(0) = xi;
        subcellIndex(1) = yi;
        subcellIndex(2) = zi;

//        double X = subgridPosition(0) + xi*meshWidth(0);
//        double Y = subgridPosition(1) + yi*meshWidth(1);
//        double Z = subgridPosition(2) + zi*meshWidth(2);

        double rho = 1.0;
        double px = 0.0;
        double py = 0.0;
        double pz = 0.0;

        accessor.setValueUNew(subcellIndex, 0, rho);
        accessor.setValueUNew(subcellIndex, 1, px);
        accessor.setValueUNew(subcellIndex, 2, py);
        accessor.setValueUNew(subcellIndex, 3, pz);

        assertion2(tarch::la::equals(accessor.getValueUNew(subcellIndex, 0), rho, 1e-5), accessor.getValueUNew(subcellIndex, 0), rho);
        assertion2(tarch::la::equals(accessor.getValueUNew(subcellIndex, 1), px, 1e-5), accessor.getValueUNew(subcellIndex, 1), px);
        assertion2(tarch::la::equals(accessor.getValueUNew(subcellIndex, 2), py, 1e-5), accessor.getValueUNew(subcellIndex, 2), py);
        assertion2(tarch::la::equals(accessor.getValueUNew(subcellIndex, 3), pz, 1e-5), accessor.getValueUNew(subcellIndex, 2), pz);
      }
    }
  }
}

tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::ShockBubble::computeDemandedMeshWidth(
  peanoclaw::Patch& patch,
  bool isInitializing
) {
  return _maximalMeshWidth;
}

tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::ShockBubble::getDomainOffset() const {
  return _domainOffset;
}

tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::ShockBubble::getDomainSize() const {
  return _domainSize;
}

tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::ShockBubble::getInitialMinimalMeshWidth() const {
  return _maximalMeshWidth;
}

tarch::la::Vector<DIMENSIONS,int> peanoclaw::native::scenarios::ShockBubble::getSubdivisionFactor() const {
  return _subdivisionFactor;
}

double peanoclaw::native::scenarios::ShockBubble::getGlobalTimestepSize() const {
  return _globalTimestepSize;
}

double peanoclaw::native::scenarios::ShockBubble::getEndTime() const {
  return _endTime;
}
