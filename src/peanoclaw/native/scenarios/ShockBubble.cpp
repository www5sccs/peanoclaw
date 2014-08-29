/*
 * ShockBubble.cpp
 *
 *  Created on: Jul 30, 2014
 *      Author: kristof
 */

#include "peanoclaw/Patch.h"
#include "peanoclaw/native/scenarios/ShockBubble.h"

double peanoclaw::native::scenarios::ShockBubble::_rhoOutside = 1.0;
double peanoclaw::native::scenarios::ShockBubble::_rhoInside = 0.1;
double peanoclaw::native::scenarios::ShockBubble::_gamma = 1.4;
double peanoclaw::native::scenarios::ShockBubble::_bubbleRadius = 0.1; //0.2;
double peanoclaw::native::scenarios::ShockBubble::_shockX = 0.2;
double peanoclaw::native::scenarios::ShockBubble::_pInflow = 5.0;

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
  const tarch::la::Vector<DIMENSIONS, double> subgridPosition = subgrid.getPosition();
  const tarch::la::Vector<DIMENSIONS, double> meshWidth = subgrid.getSubcellSize();
  peanoclaw::grid::SubgridAccessor& accessor = subgrid.getAccessor();

  double pInside = 1.0;
  double pOutside = 1.0;

  double gamma1 = _gamma - 1.0;
  double rhoInflow = (gamma1 + _pInflow * (_gamma + 1.0)) / ((_gamma + 1.0) + gamma1 * _pInflow);
  double vInflow = 1.0 / sqrt(_gamma) * (_pInflow - 1.0) / sqrt(0.5 * ((_gamma + 1.) / _gamma) * _pInflow + 0.5 * gamma1 / _gamma);
  double eInflow = 0.5 * rhoInflow * vInflow * vInflow + _pInflow / gamma1;

  tarch::la::Vector<DIMENSIONS, int> subcellIndex;
  for (int zi = 0; zi < subgrid.getSubdivisionFactor()(2); zi++) {
    for (int yi = 0; yi < subgrid.getSubdivisionFactor()(1); yi++) {
      for (int xi = 0; xi < subgrid.getSubdivisionFactor()(0); xi++) {
        subcellIndex(0) = xi;
        subcellIndex(1) = yi;
        subcellIndex(2) = zi;

        double X = subgridPosition(0) + (xi+0.5)*meshWidth(0);
        double Y = subgridPosition(1) + (yi+0.5)*meshWidth(1);
        double Z = subgridPosition(2) + (zi+0.5)*meshWidth(2);
        double r = sqrt((X-0.5)*(X-0.5) + (Y-0.5)*(Y-0.5) + (Z-0.5)*(Z-0.5));

        double rho = (X < _shockX) ? rhoInflow : ((r < _bubbleRadius) ? _rhoInside : _rhoOutside);
        double px = (X < _shockX) ? rhoInflow * vInflow : 0.0;
        double e = (X < _shockX) ? eInflow : ((r < _bubbleRadius) ? pInside :  pOutside / gamma1);
        double py = 0.0;
        double pz = 0.0;

        accessor.setValueUNew(subcellIndex, 0, rho);
        accessor.setValueUNew(subcellIndex, 1, px);
        accessor.setValueUNew(subcellIndex, 2, py);
        accessor.setValueUNew(subcellIndex, 3, pz);
        accessor.setValueUNew(subcellIndex, 4, e); //Energy
        accessor.setValueUNew(subcellIndex, 5, 0.0); //Marker

        //assertion2(tarch::la::equals(accessor.getValueUNew(subcellIndex, 0), rho, 1e-5), accessor.getValueUNew(subcellIndex, 0), rho);
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

void peanoclaw::native::scenarios::ShockBubble::setBoundaryCondition(
  peanoclaw::Patch& subgrid,
  peanoclaw::grid::SubgridAccessor& accessor,
  int dimension,
  bool setUpper,
  tarch::la::Vector<DIMENSIONS,int> sourceSubcellIndex,
  tarch::la::Vector<DIMENSIONS,int> destinationSubcellIndex
) {
  if(dimension == 0 && !setUpper && false) {

    double gamma1 = _gamma - 1.0;
//    rinf = (gamma1 + pinf * (gamma + 1.)) / ((gamma + 1.) + gamma1 * pinf)
    double rhoInflow = (gamma1 + _pInflow * (_gamma + 1.0)) / ((_gamma + 1.0) + gamma1 * _pInflow);
//    vinf = 1. / np.sqrt(gamma) * (pinf - 1.) / np.sqrt(0.5 * ((gamma + 1.) / gamma) * pinf + 0.5 * gamma1 / gamma)
    double mInflow = 0.0; //1.0 / sqrt(_gamma) * (_pInflow - 1.0) / sqrt(0.5 * ((_gamma + 1.) / _gamma) * _pInflow + 0.5 * gamma1 / _gamma);
//    einf = 0.5 * rinf * vinf ** 2 + pinf / gamma1
    double eInflow = 0.5 * rhoInflow * mInflow * mInflow + _pInflow / gamma1;

    accessor.setValueUOld(destinationSubcellIndex, 0, rhoInflow);
    accessor.setValueUOld(destinationSubcellIndex, 1, mInflow);
    accessor.setValueUOld(destinationSubcellIndex, 2, 0.0);
    accessor.setValueUOld(destinationSubcellIndex, 3, 0.0);
    accessor.setValueUOld(destinationSubcellIndex, 4, eInflow);
    accessor.setValueUOld(destinationSubcellIndex, 5, 0.0);
  } else {
     //Copy
     for(int unknown = 0; unknown < subgrid.getUnknownsPerSubcell(); unknown++) {
       accessor.setValueUOld(destinationSubcellIndex, unknown, accessor.getValueUOld(sourceSubcellIndex, unknown));
     }
  }
}
