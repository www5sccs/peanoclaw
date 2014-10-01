/*
 * ShockBubble.h
 *
 *  Created on: Jul 30, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_NATIVE_SCENARIOS_SHOCKBUBBLE_H_
#define PEANOCLAW_NATIVE_SCENARIOS_SHOCKBUBBLE_H_

namespace peanoclaw {
  namespace native {
    namespace scenarios {
      class ShockBubble;
    }
  }
}

#include "peanoclaw/native/scenarios/SWEScenario.h"

#include <vector>

class peanoclaw::native::scenarios::ShockBubble : public peanoclaw::native::scenarios::SWEScenario {
private:
  tarch::la::Vector<DIMENSIONS, double> _domainOffset;
  tarch::la::Vector<DIMENSIONS, double> _domainSize;
  tarch::la::Vector<DIMENSIONS, double> _minimalMeshWidth;
  tarch::la::Vector<DIMENSIONS, double> _maximalMeshWidth;
  tarch::la::Vector<DIMENSIONS, int>    _subdivisionFactor;
  double                                _globalTimestepSize;
  double                                _endTime;

  static double _rhoOutside;
  double _rhoInside;
  static double _gamma;
  static double _bubbleRadius;
  static double _shockX;
  static double _pInflow;
  static double _shockVelocity;

  void setCellValues(
    peanoclaw::Patch& subgrid,
    peanoclaw::grid::SubgridAccessor& accessor,
    const tarch::la::Vector<DIMENSIONS, int>& subcellIndex,
    bool setUNew
  );

public:
    ShockBubble(
      const tarch::la::Vector<DIMENSIONS, double>& domainOffset,
      const tarch::la::Vector<DIMENSIONS, double>& domainSize,
      const tarch::la::Vector<DIMENSIONS, int>&    finestSubgridTopology,
      const tarch::la::Vector<DIMENSIONS, int>&    coarsestSubgridTopology,
      const tarch::la::Vector<DIMENSIONS, int>&    subdivisionFactor,
      double                                       globalTimestepSize,
      double                                       endTime
    );
    ShockBubble(std::vector<std::string> arguments);
    ~ShockBubble();

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

    void setBoundaryCondition(
      peanoclaw::Patch& subgrid,
      peanoclaw::grid::SubgridAccessor& accessor,
      int dimension,
      bool setUpper,
      tarch::la::Vector<DIMENSIONS,int> sourceSubcellIndex,
      tarch::la::Vector<DIMENSIONS,int> destinationSubcellIndex
    );

    //pure SWE-Scenario
//    virtual float getWaterHeight(float x, float y);
//    virtual float waterHeightAtRest() { return getWaterHeight(0, 0); };
//    virtual float endSimulation() { return (float)getEndTime(); };
};


#endif /* PEANOCLAW_NATIVE_SCENARIOS_SHOCKBUBBLE_H_ */
