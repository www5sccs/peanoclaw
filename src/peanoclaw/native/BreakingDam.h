#ifndef __BREAKINGDAM_H__
#define __BREAKINGDAM_H__

#if defined(SWE)
#include "peanoclaw/Patch.h"
#include "peanoclaw/native/SWEKernel.h"

#include "scenarios/SWE_simple_scenarios.hh"

namespace peanoclaw {
  namespace native {
    class BreakingDam_SWEKernelScenario;
  }
}

class peanoclaw::native::BreakingDam_SWEKernelScenario
    : public peanoclaw::native::SWEKernelScenario, public SWE_Scenario {
  private:
    tarch::la::Vector<DIMENSIONS, double> _domainOffset;
    tarch::la::Vector<DIMENSIONS, double> _domainSize;
    tarch::la::Vector<DIMENSIONS, double> _minimalMeshWidth;
    tarch::la::Vector<DIMENSIONS, double> _maximalMeshWidth;
    tarch::la::Vector<DIMENSIONS, int>    _subdivisionFactor;
    double                                _globalTimestepSize;
    double                                _endTime;

  public:
      BreakingDam_SWEKernelScenario(
        const tarch::la::Vector<DIMENSIONS, double>& domainOffset,
        const tarch::la::Vector<DIMENSIONS, double>& domainSize,
        const tarch::la::Vector<DIMENSIONS, int>&    finestSubgridTopology,
        const tarch::la::Vector<DIMENSIONS, int>&    coarsestSubgridTopology,
        const tarch::la::Vector<DIMENSIONS, int>&    subdivisionFactor,
        double                                       globalTimestepSize,
        double                                       endTime
      );
      ~BreakingDam_SWEKernelScenario();

      virtual void initializePatch(peanoclaw::Patch& patch);
      virtual tarch::la::Vector<DIMENSIONS,double> computeDemandedMeshWidth(peanoclaw::Patch& patch, bool isInitializing);

      virtual void update(peanoclaw::Patch& patch) {}

      //PeanoClaw-Scenario
      tarch::la::Vector<DIMENSIONS,double> getDomainOffset() const;
      tarch::la::Vector<DIMENSIONS,double> getDomainSize() const;
      tarch::la::Vector<DIMENSIONS,double> getInitialMinimalMeshWidth() const;
      tarch::la::Vector<DIMENSIONS,int>    getSubdivisionFactor() const;
      double                               getGlobalTimestepSize() const;
      double                               getEndTime() const;
      double                               getInitialTimestepSize() const;

      //pure SWE-Scenario
      virtual float getWaterHeight(float x, float y);
      virtual float waterHeightAtRest() { return getWaterHeight(0, 0); };
      virtual float endSimulation() { return (float)getEndTime(); };
};
#endif

#endif // __BREAKINGDAM_H__
