/*
 * BowlOcean.h
 *
 *  Created on: Jun 23, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_NATIVE_SCENARIOS_BOWLOCEAN_H_
#define PEANOCLAW_NATIVE_SCENARIOS_BOWLOCEAN_H_

#include "peanoclaw/native/scenarios/SWEScenario.h"

#include <vector>

namespace peanoclaw {
  namespace native {
    namespace scenarios {
      class BowlOcean;
    }
  }
}

/**
 * This class ressembles an ocean that is squared and has the topology of a bowl. ;-)
 * I.e. the center of the ocean has depth $h_1$ while the corners have depth $h_2$.
 * The depth is determined by a parabolic function based on the distance to the center
 * of the ocean.
 *
 * The initial condition for the water surface is a radial breaking dam in the center
 * of the domain.
 */
class peanoclaw::native::scenarios::BowlOcean : public peanoclaw::native::scenarios::SWEScenario {
private:
  tarch::la::Vector<DIMENSIONS,double>  _domainSize;
  tarch::la::Vector<DIMENSIONS,double>  _demandedMeshWidth;
  tarch::la::Vector<DIMENSIONS,int>     _subdivisionFactor;
  double                                _endTime;
  double                                _globalTimestepSize;
  tarch::la::Vector<DIMENSIONS, double> _minimalMeshWidth;
  tarch::la::Vector<DIMENSIONS, double> _maximalMeshWidth;
  int                                   _numberOfRampSides;

  double                                _deepestDepth;
  double                                _shallowestDepth;

  tarch::la::Vector<DIMENSIONS, double> _damCenter;

  enum RefinementType {
    RefineWaveFront,
    RefineCoastline
  };

  RefinementType                        _refinementType;

public:
  BowlOcean(
    std::vector<std::string> arguments
  );

  virtual ~BowlOcean();

  virtual void initializePatch(peanoclaw::Patch& patch);
  virtual tarch::la::Vector<DIMENSIONS,double> computeDemandedMeshWidth(peanoclaw::Patch& patch, bool isInitializing);

  virtual void update(peanoclaw::Patch& subgrid);

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

  float getBathymetry(float x, float y);

  float getBoundaryPos(BoundaryEdge edge);
};


#endif /* PEANOCLAW_NATIVE_SCENARIOS_BOWLOCEAN_H_ */
