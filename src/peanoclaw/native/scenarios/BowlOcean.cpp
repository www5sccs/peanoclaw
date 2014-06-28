#include "peanoclaw/native/scenarios/BowlOcean.h"

#include "peanoclaw/Patch.h"

#if defined(SWE)
//peanoclaw::native::scenarios::BowlOcean::BowlOcean(
//  const tarch::la::Vector<DIMENSIONS, double>& domainOffset,
//  const tarch::la::Vector<DIMENSIONS, double>& domainSize,
//  const tarch::la::Vector<DIMENSIONS, int>&    finestSubgridTopology,
//  const tarch::la::Vector<DIMENSIONS, int>&    coarsestSubgridTopology,
//  const tarch::la::Vector<DIMENSIONS, int>& subdivisionFactor,
//  double                                    globalTimestepSize,
//  double                                    endTime
//) : _domainOffset(domainOffset),
//    _domainSize(domainSize),
//    _minimalMeshWidth(-1),
//    _maximalMeshWidth(-1),
//    _subdivisionFactor(subdivisionFactor),
//    _globalTimestepSize(globalTimestepSize),
//    _endTime(endTime)
//{
//  _minimalMeshWidth
//    = tarch::la::multiplyComponents(domainSize, tarch::la::invertEntries(finestSubgridTopology.convertScalar<double>()));
//  _maximalMeshWidth
//    = tarch::la::multiplyComponents(domainSize, tarch::la::invertEntries(coarsestSubgridTopology.convertScalar<double>()));
//
////  assertion2(tarch::la::allGreaterEquals(_maximalMeshWidth, _minimalMeshWidth), _minimalMeshWidth, _maximalMeshWidth);
//  assertion2(tarch::la::allSmallerEquals(_minimalMeshWidth, _maximalMeshWidth), _minimalMeshWidth, _maximalMeshWidth);
//}

peanoclaw::native::scenarios::BowlOcean::BowlOcean(
  std::vector<std::string> arguments
) : _domainSize(1000){
  if(arguments.size() != 9) {
    std::cerr << "Expected arguments for Scenario 'BreakingDam': finestSubgridTopology coarsestSubgridTopology subdivisionFactor endTime globalTimestepSize numberOfRamps relDamCenterX relDamCenterY refinementType" << std::endl
        << "\tGot " << arguments.size() << " arguments." << std::endl
        << "Parameters:" << std::endl
        << " - numberOfRamps: 0-4" << std::endl
       << " - refinementType: refineWave - Refinement of wave front; refineCoast - Refinement of coast line" << std::endl;
    throw "";
  }

  double finestSubgridTopologyPerDimension = atof(arguments[0].c_str());
  _minimalMeshWidth = _domainSize/ finestSubgridTopologyPerDimension;

  double coarsestSubgridTopologyPerDimension = atof(arguments[1].c_str());
  _maximalMeshWidth = _domainSize/ coarsestSubgridTopologyPerDimension;

  _subdivisionFactor = tarch::la::Vector<DIMENSIONS,int>(atoi(arguments[2].c_str()));

  _endTime = atof(arguments[3].c_str());

  _globalTimestepSize = atof(arguments[4].c_str());

  _numberOfRampSides = atoi(arguments[5].c_str());

  _damCenter[0] = atof(arguments[6].c_str()) * _domainSize[1];
  _damCenter[1] = atof(arguments[7].c_str()) * _domainSize[1];

  if(arguments[8] == "refineWave") {
    _refinementType = RefineWaveFront;
  } else if(arguments[8] == "refineCoast") {
    _refinementType = RefineCoastline;
  }

  _deepestDepth = 100;
  _shallowestDepth = 10;
}

peanoclaw::native::scenarios::BowlOcean::~BowlOcean() {}

void peanoclaw::native::scenarios::BowlOcean::initializePatch(peanoclaw::Patch& subgrid) {
    // compute from mesh data
    peanoclaw::grid::SubgridAccessor& accessor = subgrid.getAccessor();

    tarch::la::Vector<DIMENSIONS, int> subcellIndex;
    for (int subcellY = 0; subcellY < subgrid.getSubdivisionFactor()(1); subcellY++) {
        for (int subcellX = 0; subcellX < subgrid.getSubdivisionFactor()(0); subcellX++) {
            subcellIndex(0) = subcellX;
            subcellIndex(1) = subcellY;

            tarch::la::Vector<DIMENSIONS,double> subcellCenter = subgrid.getSubcellCenter(subcellIndex);

            double q0 = getWaterHeight(subcellCenter[0], subcellCenter[1]);
            double q1 = 0.0;
            double q2 = 0.0;

            accessor.setValueUNew(subcellIndex, 0, q0);
            accessor.setValueUNew(subcellIndex, 1, q1);
            accessor.setValueUNew(subcellIndex, 2, q2);

            assertion2(tarch::la::equals(accessor.getValueUNew(subcellIndex, 0), q0, 1e-5), accessor.getValueUNew(subcellIndex, 0), q0);
            assertion2(tarch::la::equals(accessor.getValueUNew(subcellIndex, 1), q1, 1e-5), accessor.getValueUNew(subcellIndex, 1), q1);
            assertion2(tarch::la::equals(accessor.getValueUNew(subcellIndex, 2), q2, 1e-5), accessor.getValueUNew(subcellIndex, 2), q2);
        }
    }

    update(subgrid);
}

void peanoclaw::native::scenarios::BowlOcean::update(peanoclaw::Patch& subgrid) {
  peanoclaw::grid::SubgridAccessor& accessor = subgrid.getAccessor();

  //Bathymetry
  tarch::la::Vector<DIMENSIONS, int> subcellIndex;
  for (int subcellY = -subgrid.getGhostlayerWidth(); subcellY < subgrid.getSubdivisionFactor()(1) + subgrid.getGhostlayerWidth(); subcellY++) {
      for (int subcellX = -subgrid.getGhostlayerWidth(); subcellX < subgrid.getSubdivisionFactor()(0) + subgrid.getGhostlayerWidth(); subcellX++) {
        subcellIndex(0) = subcellX;
        subcellIndex(1) = subcellY;

        tarch::la::Vector<DIMENSIONS,double> subcellCenter = subgrid.getSubcellCenter(subcellIndex);
        accessor.setParameterWithGhostlayer(subcellIndex, 0, getBathymetry(subcellCenter[0], subcellCenter[1]));
      }
  }
}

tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::BowlOcean::computeDemandedMeshWidth(
  peanoclaw::Patch& subgrid,
  bool isInitializing
) {
  if(tarch::la::equals(_minimalMeshWidth, _maximalMeshWidth)) {
    return _minimalMeshWidth;
  }


    double max_gradient = 0.0;
    peanoclaw::grid::SubgridAccessor& accessor = subgrid.getAccessor();

    tarch::la::Vector<DIMENSIONS, int> this_subcellIndex;
    tarch::la::Vector<DIMENSIONS, int> next_subcellIndex_x;
    tarch::la::Vector<DIMENSIONS, int> next_subcellIndex_y;
    for (int yi = 0; yi < subgrid.getSubdivisionFactor()(1)-1; yi++) {
        for (int xi = 0; xi < subgrid.getSubdivisionFactor()(0)-1; xi++) {
            this_subcellIndex(0) = xi;
            this_subcellIndex(1) = yi;

            next_subcellIndex_x(0) = xi+1;
            next_subcellIndex_x(1) = yi;

            next_subcellIndex_y(0) = xi;
            next_subcellIndex_y(1) = yi+1;

            double q0 =  accessor.getValueUNew(this_subcellIndex, 0) + accessor.getParameterWithGhostlayer(this_subcellIndex, 0);
            double q0_x =  (accessor.getValueUNew(next_subcellIndex_x, 0) + accessor.getParameterWithGhostlayer(next_subcellIndex_x, 0) - q0);// / meshWidth(0);
            double q0_y =  (accessor.getValueUNew(next_subcellIndex_y, 0) + accessor.getParameterWithGhostlayer(next_subcellIndex_y, 0) - q0);// / meshWidth(1);

            max_gradient = fmax(max_gradient, sqrt(fabs(q0_x)*fabs(q0_x) + fabs(q0_y)*fabs(q0_y)));
        }
    }

    tarch::la::Vector<DIMENSIONS, int> subcellIndex;
    double minimalDepth = std::numeric_limits<double>::max();
    for (int yi = 0; yi < subgrid.getSubdivisionFactor()(1); yi++) {
        for (int xi = 0; xi < subgrid.getSubdivisionFactor()(0); xi++) {
          subcellIndex(0) = xi;
          subcellIndex(1) = yi;
          tarch::la::Vector<DIMENSIONS,double> subcellCenter = subgrid.getSubcellCenter(subcellIndex);
          minimalDepth = std::min(minimalDepth, -static_cast<double>(getBathymetry(subcellCenter[0], subcellCenter[1])));
        }
    }

    tarch::la::Vector<DIMENSIONS,double> demandedMeshWidth;
    if(_refinementType == RefineWaveFront) {
      if (
         max_gradient > 0.05
          ) {
        demandedMeshWidth = _minimalMeshWidth;
      } else if (max_gradient < 0.05) {
        demandedMeshWidth = _maximalMeshWidth;
      } else {
        demandedMeshWidth = subgrid.getSubcellSize();
      }
    } else if(_refinementType == RefineCoastline) {
      if (
          minimalDepth < _shallowestDepth + 0.1
        ) {
        demandedMeshWidth = _minimalMeshWidth;
      } else {
        demandedMeshWidth = _maximalMeshWidth;
      }
    } else {
      assertionFail("Not implemented, yet!");
      throw "Not implemented, yet!";
    }

    return demandedMeshWidth;
}

tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::BowlOcean::getDomainOffset() const {
  return tarch::la::Vector<DIMENSIONS, double>(0.0);
}

tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::BowlOcean::getDomainSize() const {
  return _domainSize;
}

tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::BowlOcean::getInitialMinimalMeshWidth() const {
  return _maximalMeshWidth;
}

tarch::la::Vector<DIMENSIONS,int> peanoclaw::native::scenarios::BowlOcean::getSubdivisionFactor() const {
  return _subdivisionFactor;
}

double peanoclaw::native::scenarios::BowlOcean::getGlobalTimestepSize() const {
  return _globalTimestepSize;
}

double peanoclaw::native::scenarios::BowlOcean::getEndTime() const {
  return _endTime;
}

float peanoclaw::native::scenarios::BowlOcean::getWaterHeight(float x, float y) {
  // dam coordinates
  const double x0=_damCenter[0];
  const double y0=_damCenter[1];
  // Riemann states of the dam break problem
  const double radDam = _domainSize[0] / 5;
  const double hl = 2.;
  const double hr = 1.;

  double r = sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0));
  return hl*(r<=radDam) + hr*(r>radDam) - getBathymetry(x, y);
}

float peanoclaw::native::scenarios::BowlOcean::waterHeightAtRest() {
  return 0;
}

float peanoclaw::native::scenarios::BowlOcean::endSimulation() {
  return _endTime;
}

float peanoclaw::native::scenarios::BowlOcean::getBathymetry(float xf, float yf) {
  double x = (double)xf;
  double y = (double)yf;

  double relativeCoastWidth = 0.05;

  double minDistance =  std::numeric_limits<double>::max();
  if(_numberOfRampSides > 0) {
    minDistance = std::min(minDistance, x);
  }
  if(_numberOfRampSides > 1) {
    minDistance = std::min(minDistance, y);
  }
  if(_numberOfRampSides > 2) {
    minDistance = std::min(minDistance, _domainSize[0] - x);
  }
  if(_numberOfRampSides > 3) {
    minDistance = std::min(minDistance, _domainSize[1] - y);
  }

  minDistance = std::max(0.0, minDistance - _domainSize[0] * relativeCoastWidth);

  double a = (_deepestDepth - _shallowestDepth) / (_domainSize[0]*(1.0-relativeCoastWidth)/3.0);

  return -std::min(_deepestDepth, _shallowestDepth + a * minDistance);
}

float peanoclaw::native::scenarios::BowlOcean::getBoundaryPos(BoundaryEdge edge) {
   if (edge==BND_LEFT || edge==BND_BOTTOM) {
      return 0.0f;
   } else {
      return _domainSize[0];
   }
};
#endif


