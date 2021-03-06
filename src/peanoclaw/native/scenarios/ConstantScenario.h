/*
 * ConstantScenario.h
 *
 *  Created on: Aug 29, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_NATIVE_SCENARIOS_CONSTANTSCENARIO_H_
#define PEANOCLAW_NATIVE_SCENARIOS_CONSTANTSCENARIO_H_

#include "peanoclaw/native/scenarios/SWEScenario.h"

#include <vector>

namespace peanoclaw {
  namespace native {
    namespace scenarios {
      class ConstantScenario;
    }
  }
}

class peanoclaw::native::scenarios::ConstantScenario : public peanoclaw::native::scenarios::SWEScenario {
  private:
    tarch::la::Vector<DIMENSIONS,double> _domainSize;
    tarch::la::Vector<DIMENSIONS,double> _demandedMeshWidth;
    tarch::la::Vector<DIMENSIONS,int>    _subdivisionFactor;
    double                               _endTime;
    double                               _globalTimestepSize;

  public:
    ConstantScenario(
      std::vector<std::string> arguments
    );

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

    //pure SWE-Scenario
    virtual float getWaterHeight(float x, float y);
    virtual float waterHeightAtRest();
    virtual float endSimulation();
};


#endif /* PEANOCLAW_NATIVE_SCENARIOS_CONSTANTSCENARIO_H_ */
