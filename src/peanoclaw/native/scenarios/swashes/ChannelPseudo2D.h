/*
 * Pseudo2D.h
 *
 *  Created on: Oct 15, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_NATIVE_SCENARIOS_SWASHES_CHANNELPSEUDO2D_H_
#define PEANOCLAW_NATIVE_SCENARIOS_SWASHES_CHANNELPSEUDO2D_H_

#include "peanoclaw/native/scenarios/SWEScenario.h"

#include <vector>

namespace peanoclaw {
  namespace native {
    namespace scenarios {
      namespace swashes {
        class ChannelPseudo2D;
      }
    }
  }
}

/**
 * This class ressembles the pseudo 2d testcases of the swashes compilation.
 *
 * https://hal.archives-ouvertes.fr/hal-00628246v5/document (Page 18 ff.)
 *
 * z: topography
 * R: rain intensity
 * h: water height
 * I: infiltration rate
 * Sf: friction force
 * (u,v): velocity vector
 * Cf = n^2: Manning's coefficient
 * µSd: viscous term
 * q = hu
 * Z: slope width
 *
 */
class peanoclaw::native::scenarios::swashes::ChannelPseudo2D : public peanoclaw::native::scenarios::SWEScenario {
private:

  const double g;

  tarch::la::Vector<DIMENSIONS,double>  _domainSize;
  tarch::la::Vector<DIMENSIONS,double>  _domainOffset;
  tarch::la::Vector<DIMENSIONS,double>  _demandedMeshWidth;
  tarch::la::Vector<DIMENSIONS,int>     _subdivisionFactor;
  double                                _endTime;
  double                                _globalTimestepSize;
  tarch::la::Vector<DIMENSIONS, double> _minimalMeshWidth;
  tarch::la::Vector<DIMENSIONS, double> _maximalMeshWidth;

  double                                _discharge;

  enum ChannelType {
    Short,
    Long
  };
  ChannelType _channelType;

  /**
   * Function for defining the bed width for the short channel cases.
   */
  double shortBedWidth(double x) const;

  /**
   * Function for defining the bed width for the long channel cases.
   */
  double longBedWidth(double x) const;

  double bedWidth(double x) const;

  double slope(double x, double bedWidth, double g, double q = 20) const;

  double topography(double x, ) const;

public:
  ChannelPseudo2D(
    std::vector<std::string> arguments
  );

  virtual ~ChannelPseudo2D();

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
};


#endif /* PEANOCLAW_NATIVE_SCENARIOS_SWASHES_CHANNELPSEUDO2D_H_ */
