#if defined(SWE)
#include "BreakingDam.h"

peanoclaw::native::BreakingDam_SWEKernelScenario::BreakingDam_SWEKernelScenario(
  const tarch::la::Vector<DIMENSIONS, double>& domainOffset,
  const tarch::la::Vector<DIMENSIONS, double>& domainSize,
  const tarch::la::Vector<DIMENSIONS, double>& minimalMeshWidth,
  const tarch::la::Vector<DIMENSIONS, double>& maximalMeshWidth,
  const tarch::la::Vector<DIMENSIONS, int>& subdivisionFactor,
  double                                    globalTimestepSize,
  double                                    endTime
) : _domainOffset(domainOffset),
    _domainSize(domainSize),
    _minimalMeshWidth(minimalMeshWidth),
    _maximalMeshWidth(maximalMeshWidth),
    _subdivisionFactor(subdivisionFactor),
    _globalTimestepSize(globalTimestepSize),
    _endTime(endTime)
{
  assertion(tarch::la::allSmallerEquals(minimalMeshWidth, maximalMeshWidth));
}

peanoclaw::native::BreakingDam_SWEKernelScenario::~BreakingDam_SWEKernelScenario() {}

void peanoclaw::native::BreakingDam_SWEKernelScenario::initializePatch(peanoclaw::Patch& patch) {
    // dam coordinates
    double x0=10/3.0;
    double y0=10/3.0;
    
    // Riemann states of the dam break problem
    double radDam = 1;
    double hl = 2.;
    double ul = 0.;
    double vl = 0.;
    double hr = 1.;
    double ur = 0.;
    double vr = 0.;
    
    // compute from mesh data
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
 
            double r = sqrt((X-x0)*(X-x0) + (Y-y0)*(Y-y0));
            double q0 = hl*(r<=radDam) + hr*(r>radDam);
            double q1 = hl*ul*(r<=radDam) + hr*ur*(r>radDam);
            double q2 = hl*vl*(r<=radDam) + hr*vr*(r>radDam);

            accessor.setValueUNew(subcellIndex, 0, q0);
            accessor.setValueUNew(subcellIndex, 1, q1);
            accessor.setValueUNew(subcellIndex, 2, q2);

            assertionEquals(accessor.getValueUNew(subcellIndex, 0), q0);
            assertionEquals(accessor.getValueUNew(subcellIndex, 1), q1);
            assertionEquals(accessor.getValueUNew(subcellIndex, 2), q2);
        }
    }
}

tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::BreakingDam_SWEKernelScenario::computeDemandedMeshWidth(
  peanoclaw::Patch& patch,
  bool isInitializing
) {
    double max_gradient = 0.0;
    const tarch::la::Vector<DIMENSIONS, double> meshWidth = patch.getSubcellSize();
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
 
            double q0 =  accessor.getValueUNew(this_subcellIndex, 0);
            double q0_x =  (accessor.getValueUNew(next_subcellIndex_x, 0) - q0) / meshWidth(0);
            double q0_y =  (accessor.getValueUNew(next_subcellIndex_y, 0) - q0) / meshWidth(1);

            max_gradient = fmax(max_gradient, sqrt(fabs(q0_x)*fabs(q0_x) + fabs(q0_y)*fabs(q0_y)));
        }
    }
  
    tarch::la::Vector<DIMENSIONS,double> demandedMeshWidth;
    if (max_gradient > 0.1) {
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

tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::BreakingDam_SWEKernelScenario::getDomainOffset() const {
  return _domainOffset;
}

tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::BreakingDam_SWEKernelScenario::getDomainSize() const {
  return _domainSize;
}

tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::BreakingDam_SWEKernelScenario::getInitialMinimalMeshWidth() const {
  return _maximalMeshWidth;
}

tarch::la::Vector<DIMENSIONS,int> peanoclaw::native::BreakingDam_SWEKernelScenario::getSubdivisionFactor() const {
  return _subdivisionFactor;
}

double peanoclaw::native::BreakingDam_SWEKernelScenario::getGlobalTimestepSize() const {
  return _globalTimestepSize;
}

double peanoclaw::native::BreakingDam_SWEKernelScenario::getEndTime() const {
  return _endTime;
}

double peanoclaw::native::BreakingDam_SWEKernelScenario::getInitialTimestepSize() const {
  return 0.1;
}
#endif
