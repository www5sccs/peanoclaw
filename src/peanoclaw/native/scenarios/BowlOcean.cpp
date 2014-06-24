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
  if(arguments.size() != 6) {
    std::cerr << "Expected arguments for Scenario 'BreakingDam': finestSubgridTopology coarsestSubgridTopology subdivisionFactor endTime globalTimestepSize" << std::endl
        << "\tGot " << arguments.size() << " arguments." << std::endl;
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

  _deepestDepth = 100;
  _shallowestDepth = 1;
}

peanoclaw::native::scenarios::BowlOcean::~BowlOcean() {}

void peanoclaw::native::scenarios::BowlOcean::initializePatch(peanoclaw::Patch& patch) {
    // compute from mesh data
    const tarch::la::Vector<DIMENSIONS, double> patchPosition = patch.getPosition();
    const tarch::la::Vector<DIMENSIONS, double> subcellSize = patch.getSubcellSize();
    peanoclaw::grid::SubgridAccessor& accessor = patch.getAccessor();

    tarch::la::Vector<DIMENSIONS, int> subcellIndex;
    for (int subcellY = 0; subcellY < patch.getSubdivisionFactor()(1); subcellY++) {
        for (int subcellX = 0; subcellX < patch.getSubdivisionFactor()(0); subcellX++) {
            subcellIndex(0) = subcellX;
            subcellIndex(1) = subcellY;

            double x = patchPosition(0) + subcellX*subcellSize(0);
            double y = patchPosition(1) + subcellY*subcellSize(1);

            double q0 = getWaterHeight(x, y);
            double q1 = 0.0; //hl*ul*(r<=radDam) + hr*ur*(r>radDam);
            double q2 = 0.0; //hl*vl*(r<=radDam) + hr*vr*(r>radDam);

            accessor.setValueUNew(subcellIndex, 0, q0);
            accessor.setValueUNew(subcellIndex, 1, q1);
            accessor.setValueUNew(subcellIndex, 2, q2);

            assertion2(tarch::la::equals(accessor.getValueUNew(subcellIndex, 0), q0, 1e-5), accessor.getValueUNew(subcellIndex, 0), q0);
            assertion2(tarch::la::equals(accessor.getValueUNew(subcellIndex, 1), q1, 1e-5), accessor.getValueUNew(subcellIndex, 1), q1);
            assertion2(tarch::la::equals(accessor.getValueUNew(subcellIndex, 2), q2, 1e-5), accessor.getValueUNew(subcellIndex, 2), q2);
        }
    }

    //Bathymetry
    for (int subcellY = -patch.getGhostlayerWidth(); subcellY < patch.getSubdivisionFactor()(1) + patch.getGhostlayerWidth(); subcellY++) {
        for (int subcellX = -patch.getGhostlayerWidth(); subcellX < patch.getSubdivisionFactor()(0) + patch.getGhostlayerWidth(); subcellX++) {
          subcellIndex(0) = subcellX;
          subcellIndex(1) = subcellY;

          double x = patchPosition(0) + subcellX*subcellSize(0);
          double y = patchPosition(1) + subcellY*subcellSize(1);

          accessor.setParameterWithGhostlayer(subcellIndex, 0, getBathymetry(x, y));
        }
    }
}

tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::BowlOcean::computeDemandedMeshWidth(
  peanoclaw::Patch& patch,
  bool isInitializing
) {
  if(tarch::la::equals(_minimalMeshWidth, _maximalMeshWidth)) {
    return _minimalMeshWidth;
  }


    double max_gradient = 0.0;
    peanoclaw::grid::SubgridAccessor& accessor = patch.getAccessor();

    tarch::la::Vector<DIMENSIONS, int> this_subcellIndex;
    tarch::la::Vector<DIMENSIONS, int> next_subcellIndex_x;
    tarch::la::Vector<DIMENSIONS, int> next_subcellIndex_y;
    for (int yi = 0; yi < patch.getSubdivisionFactor()(1)-1; yi++) {
        for (int xi = 0; xi < patch.getSubdivisionFactor()(0)-1; xi++) {
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
    double minWaterHeight = std::numeric_limits<double>::max();
//    for (int yi = 0; yi < patch.getSubdivisionFactor()(1); yi++) {
//        for (int xi = 0; xi < patch.getSubdivisionFactor()(0); xi++) {
//          subcellIndex(0) = xi;
//          subcellIndex(1) = yi;
//          minWaterHeight = std::min(accessor.getValueUNew(subcellIndex, 0), minWaterHeight);
//        }
//    }

    tarch::la::Vector<DIMENSIONS,double> demandedMeshWidth;
    if (max_gradient > 0.1 || minWaterHeight < 1.1) {
        //demandedMeshWidth = 1.0/243;
        //demandedMeshWidth = tarch::la::Vector<DIMENSIONS,double>(10.0/6/27);
      demandedMeshWidth = _minimalMeshWidth;
    } else if (max_gradient < 0.1) {
        //demandedMeshWidth = 10.0/130/27;
        //demandedMeshWidth = tarch::la::Vector<DIMENSIONS,double>(10.0/6/9);
      demandedMeshWidth = _maximalMeshWidth;
    } else {
      demandedMeshWidth = patch.getSubcellSize();
    }

//    if(isInitializing) {
//      return 10.0/6/9;
//    } else {
//      return demandedMeshWidth;
//    }
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
  const double x0=_domainSize[0]/2.0;
  const double y0=_domainSize[1]/2.0;
  // Riemann states of the dam break problem
  const double radDam = _domainSize[0] / 5;
  const double hl = 1.;
  const double hr = 0.;

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
//  double r2 = (x - _domainSize[0]/2.0) * (x - _domainSize[0]/2.0) + (y - _domainSize[1]/2.0) * (y - _domainSize[1]/2.0);
//  double rMax2 = (_domainSize[0]/2.0) * (_domainSize[0]/2.0) + (_domainSize[1]/2.0) * (_domainSize[1]/2.0);
//
//  double a = (_deepestDepth - _shallowestDepth) / rMax2;
//
//  return a * r2 - _deepestDepth;
  double x = (double)xf;
  double y = (double)yf;

  double relativeCoastWidth = 0.05;

//  double distanceX = std::numeric_limits<double>::max();// = std::min(x, _domainSize[0] - x);
//  double distanceY = std::numeric_limits<double>::max();// = std::min(y, _domainSize[1] - y);

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


