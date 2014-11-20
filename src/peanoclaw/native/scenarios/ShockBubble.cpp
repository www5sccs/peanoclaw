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
double peanoclaw::native::scenarios::ShockBubble::_gamma = 1.4;
double peanoclaw::native::scenarios::ShockBubble::_bubbleRadius = 0.1; //0.2;
double peanoclaw::native::scenarios::ShockBubble::_shockX = 0.2;
double peanoclaw::native::scenarios::ShockBubble::_pInflow = 10.0;
double peanoclaw::native::scenarios::ShockBubble::_shockVelocity = 500;

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

  double soundSpeed = sqrt(_gamma * _pInflow / pOutside);

  double gamma1 = _gamma - 1.0;
  double vInflow = 500;
      //_shockVelocity - soundSpeed * sqrt(1 + (_gamma+1) / (2*_gamma) * (pOutside / _pInflow - 1));
      //1.0 / sqrt(_gamma) * (_pInflow - 1.0) / sqrt(0.5 * ((_gamma + 1.) / _gamma) * _pInflow + 0.5 * gamma1 / _gamma);
  double rhoInflow = 2.0; //4.5;
      //-(_shockVelocity * (pOutside - _pInflow)) / vInflow;
      //(gamma1 + _pInflow * (_gamma + 1.0)) / ((_gamma + 1.0) + gamma1 * _pInflow);
  double eInflow = 0.5 * rhoInflow * vInflow * vInflow + _pInflow / gamma1;
  double tInflow = 273;
  double t = 273;

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
  e = Uni::EulerEquations::Cell<double,3>::computeEnergyFromDensityVelocityTemperature(
      rho,
      velocity,
      (x < _shockX) ? tInflow : t,
      1.4,
      8.3144621757575);
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
    _domainSize(),
    _minimalMeshWidth(-1),
    _maximalMeshWidth(-1),
    _subdivisionFactor(subdivisionFactor),
    _globalTimestepSize(globalTimestepSize),
    _endTime(endTime),
    _rhoInside(0.1)
{
  tarch::la::assignList(_domainSize) = 2, 1, 1;
  for(int d = 1; d < DIMENSIONS; d++) {
    _subdivisionFactor[d] /= 2;
  }

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
    _domainSize(){
  if(arguments.size() != 6) {
    std::cerr << "Expected arguments for Scenario 'shockBubble': finestSubgridTopology coarsestSubgridTopology subdivisionFactor endTime globalTimestepSize rhoInside" << std::endl
        << "\tGot " << arguments.size() << " arguments." << std::endl;
    throw "";
  }

  tarch::la::assignList(_domainSize) = 2, 1, 1;

  _subdivisionFactor = tarch::la::Vector<DIMENSIONS,int>(atoi(arguments[2].c_str()));
  for(int d = 1; d < DIMENSIONS; d++) {
    _subdivisionFactor[d] /= 2;
  }

  double finestSubgridTopologyPerDimension = atof(arguments[0].c_str());
  _minimalMeshWidth = _domainSize[0] / finestSubgridTopologyPerDimension;

  double coarsestSubgridTopologyPerDimension = atof(arguments[1].c_str());
  _maximalMeshWidth = _domainSize[0] / coarsestSubgridTopologyPerDimension;

  _endTime = atof(arguments[3].c_str());

  _globalTimestepSize = atof(arguments[4].c_str());

  _rhoInside = atof(arguments[5].c_str());
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
