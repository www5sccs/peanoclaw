#if defined(SWE)
#include "BreakingDam.h"

BreakingDam_SWEKernelScenario::BreakingDam_SWEKernelScenario() {}
BreakingDam_SWEKernelScenario::~BreakingDam_SWEKernelScenario() {}

double BreakingDam_SWEKernelScenario::initializePatch(peanoclaw::Patch& patch) {
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
    const tarch::la::Vector<DIMENSIONS, double> patchSize = patch.getSize();
    const tarch::la::Vector<DIMENSIONS, double> patchPosition = patch.getPosition();
    const tarch::la::Vector<DIMENSIONS, double> meshWidth = patch.getSubcellSize();

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
  
            patch.setValueUNew(subcellIndex, 0, q0);
            patch.setValueUNew(subcellIndex, 1, q1);
            patch.setValueUNew(subcellIndex, 2, q2);
        }
    }
    return 10.0/9/9;
}

double BreakingDam_SWEKernelScenario::computeDemandedMeshWidth(peanoclaw::Patch& patch) {
    double max_gradient = 0.0;
    const tarch::la::Vector<DIMENSIONS, double> meshWidth = patch.getSubcellSize();
    
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
 
            double q0 =  patch.getValueUNew(this_subcellIndex, 0);
            double q0_x =  (patch.getValueUNew(next_subcellIndex_x, 0) - q0) / meshWidth(0);
            double q0_y =  (patch.getValueUNew(next_subcellIndex_y, 0) - q0) / meshWidth(1);

            max_gradient = fmax(max_gradient, sqrt(fabs(q0_x)*fabs(q0_x) + fabs(q0_y)*fabs(q0_y)));
        }
    }
  
    double demandedMeshWidth = patch.getSubcellSize()(0);
    if (max_gradient > 0.1) {
        //demandedMeshWidth = 1.0/243;
        demandedMeshWidth = 10.0/9/27;
    } else if (max_gradient < 0.5) {
        //demandedMeshWidth = 10.0/130/27;
        demandedMeshWidth = 10.0/9/27;
    } else {
      demandedMeshWidth = patch.getSubcellSize()(0);
    }

    return demandedMeshWidth;
}
#endif
