/*
 * Scenario.h
 *
 *  Created on: Jun 10, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_NATIVE_SCENARIOS_SCENARIO_H_
#define PEANOCLAW_NATIVE_SCENARIOS_SCENARIO_H_

#if defined(PEANOCLAW_SWE)
#include "scenarios/SWE_Scenario.hh"
#endif

#include "peano/utils/Dimensions.h"
#include "tarch/la/Vector.h"
#include "tarch/logging/Log.h"

namespace peanoclaw {
  class Patch;

  namespace grid {
    class SubgridAccessor;
  }

  namespace native {
    namespace scenarios {
      class SWEScenario;
    }
  }
}

/**
 * Abstract class that describes a scenario for
 * native solvers.
 */
class peanoclaw::native::scenarios::SWEScenario
    #ifdef PEANOCLAW_SWE
    : public SWE_Scenario
    #endif
      {

private:
  /**
   * Logging device.
   */
  static tarch::logging::Log _log;

public:
    virtual ~SWEScenario() {}
    virtual void initializePatch(Patch& patch) = 0;
    virtual tarch::la::Vector<DIMENSIONS,double> computeDemandedMeshWidth(Patch& patch, bool isInitializing) = 0;
    virtual void update(Patch& patch) = 0;

    virtual void setBoundaryCondition(
      peanoclaw::Patch& subgrid,
      peanoclaw::grid::SubgridAccessor& accessor,
      int dimension,
      bool setUpper,
      tarch::la::Vector<DIMENSIONS,int> sourceSubcellIndex,
      tarch::la::Vector<DIMENSIONS,int> destinationSubcellIndex
    ) {}

    virtual tarch::la::Vector<DIMENSIONS,double> getDomainOffset() const = 0;
    virtual tarch::la::Vector<DIMENSIONS,double> getDomainSize() const = 0;
    virtual tarch::la::Vector<DIMENSIONS,double> getInitialMinimalMeshWidth() const = 0;
    virtual tarch::la::Vector<DIMENSIONS,int>    getSubdivisionFactor() const = 0;
    virtual double getGlobalTimestepSize() const = 0;
    virtual double getEndTime() const = 0;
    virtual double getInitialTimestepSize() const;

    #ifndef PEANOCLAW_SWE
    virtual float getWaterHeight(float x, float y) { return 10.0f; }
    virtual float getVeloc_u(float x, float y) { return 0.0f; }
    virtual float getVeloc_v(float x, float y) { return 0.0f; }
    virtual float getBathymetry(float x, float y) { return 0.0f; }
    #endif

    #ifdef PEANOCLAW_FULLSWOF2D
    virtual bool enableRain() const { return true; }
    #endif

    /**
     * Creates a scenario based on the passed command line parameters.
     */
    static SWEScenario* createScenario(int argc, char** argv);
protected:
    SWEScenario() {}
};

#endif /* PEANOCLAW_NATIVE_SCENARIOS_SCENARIO_H_ */
