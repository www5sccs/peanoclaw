/*
 * ShockBubble.cpp
 *
 *  Created on: Jul 30, 2014
 *      Author: kristof
 */

#include "peanoclaw/Patch.h"
#include "peanoclaw/native/scenarios/ShockBubble.h"

#ifdef PEANOCLAW_EULER3D
#include "Uni/EulerEquations/Cell"
#endif

double peanoclaw::native::scenarios::ShockBubble::_rhoOutside = 1.0;
double peanoclaw::native::scenarios::ShockBubble::_rhoInside = 0.1;
double peanoclaw::native::scenarios::ShockBubble::_gamma = 1.4;
double peanoclaw::native::scenarios::ShockBubble::_bubbleRadius = 0.1; //0.2;
double peanoclaw::native::scenarios::ShockBubble::_shockX = 0.2;
double peanoclaw::native::scenarios::ShockBubble::_pInflow = 5.0;

void peanoclaw::native::scenarios::ShockBubble::setCellValues(
  peanoclaw::Patch& subgrid,
  peanoclaw::grid::SubgridAccessor& accessor,
  const tarch::la::Vector<DIMENSIONS, int>& subcellIndex,
  bool setUNew
) {
  tarch::la::Vector<DIMENSIONS,double> meshWidth = subgrid.getSubcellSize();
  double x = subgrid.getPosition()[0] + (subcellIndex[0]+0.5)*meshWidth(0);
  double y = subgrid.getPosition()[1] + (subcellIndex[1]+0.5)*meshWidth(1);
  double z = subgrid.getPosition()[2] + (subcellIndex[2]+0.5)*meshWidth(2);

  double pInside = 1.0;
  double pOutside = 1.0;

  double gamma1 = _gamma - 1.0;
  double rhoInflow = (gamma1 + _pInflow * (_gamma + 1.0)) / ((_gamma + 1.0) + gamma1 * _pInflow);
  double vInflow = 1.0 / sqrt(_gamma) * (_pInflow - 1.0) / sqrt(0.5 * ((_gamma + 1.) / _gamma) * _pInflow + 0.5 * gamma1 / _gamma);
  double eInflow = 0.5 * rhoInflow * vInflow * vInflow + _pInflow / gamma1;

  double r = sqrt((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) + (z-0.5)*(z-0.5));

  double rho = (x < _shockX) ? rhoInflow : ((r < _bubbleRadius) ? _rhoInside : _rhoOutside);
  double px = (x < _shockX) ? rhoInflow * vInflow : 0.0;
  double e = (x < _shockX) ? eInflow : ((r < _bubbleRadius) ? pInside / gamma1:  pOutside / gamma1);
  double py = 0.0;
  double pz = 0.0;

  #ifdef PEANOCLAW_EULER3D
  Uni::EulerEquations::Cell<double,3>::Vector velocity;
  velocity(0) = px;
  velocity(1) = 0;
  velocity(2) = 0;
  if(x < _shockX) {
    e = Uni::EulerEquations::Cell<double,3>::computeEnergyFromDensityVelocityTemperature(
        rho,
        velocity,
        273,
        1.4,
        8.3144621757575);
  }
  #endif

  if(setUNew) {
    accessor.setValueUNew(subcellIndex, 0, rho);
    accessor.setValueUNew(subcellIndex, 1, px);
    accessor.setValueUNew(subcellIndex, 2, py);
    accessor.setValueUNew(subcellIndex, 3, pz);
    accessor.setValueUNew(subcellIndex, 4, e); //Energy
    accessor.setValueUNew(subcellIndex, 5, 1.0); //Marker
  } else {
    accessor.setValueUOld(subcellIndex, 0, rho);
    accessor.setValueUOld(subcellIndex, 1, px);
    accessor.setValueUOld(subcellIndex, 2, py);
    accessor.setValueUOld(subcellIndex, 3, pz);
    accessor.setValueUOld(subcellIndex, 4, e); //Energy
    accessor.setValueUOld(subcellIndex, 5, 1.0); //Marker
  }
}

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
    std::cerr << "Expected arguments for Scenario 'shockBubble': finestSubgridTopology coarsestSubgridTopology subdivisionFactor endTime globalTimestepSize" << std::endl
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
  peanoclaw::grid::SubgridAccessor& accessor = subgrid.getAccessor();

  tarch::la::Vector<DIMENSIONS, int> subcellIndex;
  for (int zi = 0; zi < subgrid.getSubdivisionFactor()(2); zi++) {
    for (int yi = 0; yi < subgrid.getSubdivisionFactor()(1); yi++) {
      for (int xi = 0; xi < subgrid.getSubdivisionFactor()(0); xi++) {
        subcellIndex(0) = xi;
        subcellIndex(1) = yi;
        subcellIndex(2) = zi;

        setCellValues(
          subgrid,
          accessor,
          subcellIndex,
          true
        );

        //assertion2(tarch::la::equals(accessor.getValueUNew(subcellIndex, 0), rho, 1e-5), accessor.getValueUNew(subcellIndex, 0), rho);
//        assertion2(tarch::la::equals(accessor.getValueUNew(subcellIndex, 1), px, 1e-5), accessor.getValueUNew(subcellIndex, 1), px);
//        assertion2(tarch::la::equals(accessor.getValueUNew(subcellIndex, 2), py, 1e-5), accessor.getValueUNew(subcellIndex, 2), py);
//        assertion2(tarch::la::equals(accessor.getValueUNew(subcellIndex, 3), pz, 1e-5), accessor.getValueUNew(subcellIndex, 2), pz);
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

void peanoclaw::native::scenarios::ShockBubble::setBoundaryCondition(
  peanoclaw::Patch& subgrid,
  peanoclaw::grid::SubgridAccessor& accessor,
  int dimension,
  bool setUpper,
  tarch::la::Vector<DIMENSIONS,int> sourceSubcellIndex,
  tarch::la::Vector<DIMENSIONS,int> destinationSubcellIndex
) {
  if(dimension == 0 && !setUpper) {
    setCellValues(
      subgrid,
      accessor,
      destinationSubcellIndex,
      false
    );
  }
}
