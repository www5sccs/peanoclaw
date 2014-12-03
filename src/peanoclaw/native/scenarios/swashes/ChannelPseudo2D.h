/*
 * Pseudo2D.h
 *
 *  Created on: Oct 15, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_NATIVE_SCENARIOS_SWASHES_CHANNELPSEUDO2D_H_
#define PEANOCLAW_NATIVE_SCENARIOS_SWASHES_CHANNELPSEUDO2D_H_

#include "peanoclaw/native/scenarios/SWEScenario.h"
#include "peanoclaw/native/scenarios/swashes/SWASHESChannel.h"
#include "peanoclaw/native/scenarios/swashes/SWASHESParameters.h"

#include "peano/utils/Dimensions.h"
#include "tarch/la/Vector.h"

#include <vector>
#include <string>

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
  const double BED_HEIGHT = 9.0;

  tarch::la::Vector<DIMENSIONS,double>  _domainSize;
  tarch::la::Vector<DIMENSIONS,double>  _domainOffset;
  tarch::la::Vector<DIMENSIONS,double>  _demandedMeshWidth;
  tarch::la::Vector<DIMENSIONS,int>     _subdivisionFactor;
  double                                _endTime;
  double                                _globalTimestepSize;
  tarch::la::Vector<DIMENSIONS, double> _minimalMeshWidth;
  tarch::la::Vector<DIMENSIONS, double> _maximalMeshWidth;

  double                                _discharge;

  SWASHESChannel*                       _swashesChannel;

  enum ChannelType {
    Short,
    Long,
    CornerTest
  };
  ChannelType _channelType;

  enum Criticality {
    Sub,
    Super
  };
  Criticality _criticality;

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

  void setBoundaryCondition(
    peanoclaw::Patch& subgrid,
    peanoclaw::grid::SubgridAccessor& accessor,
    int dimension,
    bool setUpper,
    tarch::la::Vector<DIMENSIONS,int> sourceSubcellIndex,
    tarch::la::Vector<DIMENSIONS,int> destinationSubcellIndex
  ) {};

  bool enableRain() const { return false; }

  double getFrictionCoefficient() const { return 0.03; }
//  double getFrictionCoefficient() const { return 0.0; }

  FullSWOF2DBoundaryCondition getBoundaryCondition(int dimension, bool upper) const;
};


#endif /* PEANOCLAW_NATIVE_SCENARIOS_SWASHES_CHANNELPSEUDO2D_H_ */
