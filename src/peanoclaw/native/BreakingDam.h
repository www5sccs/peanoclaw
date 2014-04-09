#ifndef __BREAKINGDAM_H__
#define __BREAKINGDAM_H__

#if defined(SWE)
#include "peanoclaw/Patch.h"
#include "peanoclaw/native/SWEKernel.h"

class BreakingDam_SWEKernelScenario : public peanoclaw::native::SWEKernelScenario {
    public:
        BreakingDam_SWEKernelScenario();
        ~BreakingDam_SWEKernelScenario();

        virtual void initializePatch(peanoclaw::Patch& patch);
        virtual tarch::la::Vector<DIMENSIONS,double> computeDemandedMeshWidth(peanoclaw::Patch& patch, bool isInitializing);

        virtual void update(peanoclaw::Patch& patch) {}
};
#endif

#endif // __BREAKINGDAM_H__
