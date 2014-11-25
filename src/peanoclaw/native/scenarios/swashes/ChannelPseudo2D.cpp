/*
 * ChannelPseudo2D.cpp
 *
 *  Created on: Oct 15, 2014
 *      Author: kristof
 */
#include "peanoclaw/native/scenarios/swashes/ChannelPseudo2D.h"

#include "peanoclaw/Patch.h"

#include "peanoclaw/native/FullSWOF2D.h"

#include "peano/utils/Dimensions.h"
#include "tarch/la/Vector.h"
#include "tarch/la/VectorAssignList.h"

peanoclaw::native::scenarios::swashes::ChannelPseudo2D::ChannelPseudo2D(
  std::vector<std::string> arguments
) : _discharge(20),
    _channelType(Short),
    _swashesChannel(0)
{
  if(arguments.size() != 6) {
    std::cerr << "Expected arguments for Scenario 'ChannelPseudo2D': finestSubgridTopology coarsestSubgridTopology subdivisionFactor endTime globalTimestepSize channelLength" << std::endl
        << "\tGot " << arguments.size() << " arguments." << std::endl
        << "Parameters:" << std::endl
        << " - channelLength: 'short': 200m channel length, 'long': 400m channel length" << std::endl;
    throw "";
  }

  double finestSubgridTopologyPerDimension = atof(arguments[0].c_str());

  double coarsestSubgridTopologyPerDimension = atof(arguments[1].c_str());

  _domainSize[1] = 9.589575006880507;
  if(arguments[5] == "short") {
    _channelType = Short;
    _domainSize[0] = 200;
    #ifdef PEANOCLAW_SWASHES
    SWASHESParameters parameters(finestSubgridTopologyPerDimension * _domainSize[0] / _domainSize[1]);
    _swashesChannel = new SWASHESShortChannel(parameters);
    #endif
  } else if (arguments[6] == "long") {
    _channelType = Long;
    _domainSize[0] = 400;
//    _swashesChannel = new SWASHESLongChannel();
  } else {
    std::cerr << " Possible values for parameter channelLength are: 'short': 200m channel length, 'long': 400m channel length" << std::endl;
  }
  _swashesChannel->initialize();

  _subdivisionFactor[1] = atoi(arguments[2].c_str());
  _subdivisionFactor[0] = _subdivisionFactor[1] * _domainSize[0] / _domainSize[1];

  _minimalMeshWidth = _domainSize / finestSubgridTopologyPerDimension;

  _maximalMeshWidth = _domainSize / coarsestSubgridTopologyPerDimension;

  _endTime = atof(arguments[3].c_str());

  _globalTimestepSize = atof(arguments[4].c_str());

  #ifdef Dim2
  assignList(_domainOffset) = 0, -_domainSize[1]/2;
  #endif
}

peanoclaw::native::scenarios::swashes::ChannelPseudo2D::~ChannelPseudo2D() {
  delete _swashesChannel;
}

void peanoclaw::native::scenarios::swashes::ChannelPseudo2D::initializePatch(peanoclaw::Patch& subgrid) {
  #ifdef Dim2
  peanoclaw::grid::SubgridAccessor accessor = subgrid.getAccessor();

  for(int y = -subgrid.getGhostlayerWidth(); y < subgrid.getSubdivisionFactor()[1] + subgrid.getGhostlayerWidth(); y++) {
    for(int x = -subgrid.getGhostlayerWidth(); x < subgrid.getSubdivisionFactor()[0] + subgrid.getGhostlayerWidth(); x++) {
      tarch::la::Vector<DIMENSIONS,int> subcellIndex;
      assignList(subcellIndex) = x, y;
      tarch::la::Vector<DIMENSIONS,double> position = subgrid.getSubcellCenter(subcellIndex);

      double distanceFromCenterLine = abs(position[1]);
      double bedWidth = _swashesChannel->getBedWidth(position[0]);
      double topography = _swashesChannel->getTopography(position[0]);

      if(distanceFromCenterLine > bedWidth / 2) {
        topography += BED_HEIGHT;
      }
//      topography = 0.0;
      accessor.setParameterWithGhostlayer(subcellIndex, 0, topography);

      if(x >= 0 && y >= 0 && x < subgrid.getSubdivisionFactor()[0] &&  y < subgrid.getSubdivisionFactor()[1]) {
        double waterheight = _swashesChannel->getInitialWaterHeight(position);
        if(topography >= BED_HEIGHT) {
          waterheight = 0;
        }
        topography = 0.0;
        accessor.setValueUNew(subcellIndex, 0, waterheight);
        accessor.setValueUNew(subcellIndex, 1, 0);
        accessor.setValueUNew(subcellIndex, 2, 0);
      }
    }
  }

  #endif
}


tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::swashes::ChannelPseudo2D::computeDemandedMeshWidth(
  peanoclaw::Patch& patch,
  bool isInitializing
) {
  return _maximalMeshWidth;
}

void peanoclaw::native::scenarios::swashes::ChannelPseudo2D::update(peanoclaw::Patch& subgrid) {
}

tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::swashes::ChannelPseudo2D::getDomainOffset() const {
  return _domainOffset;
}

tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::swashes::ChannelPseudo2D::getDomainSize() const {
  return _domainSize;
}

tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::swashes::ChannelPseudo2D::getInitialMinimalMeshWidth() const {
  return _maximalMeshWidth;
}

tarch::la::Vector<DIMENSIONS,int> peanoclaw::native::scenarios::swashes::ChannelPseudo2D::getSubdivisionFactor() const {
  return _subdivisionFactor;
}


double peanoclaw::native::scenarios::swashes::ChannelPseudo2D::getGlobalTimestepSize() const {
  return _globalTimestepSize;
}

double peanoclaw::native::scenarios::swashes::ChannelPseudo2D::getEndTime() const {
  return _endTime;
}

void peanoclaw::native::scenarios::swashes::ChannelPseudo2D::setBoundaryCondition(
  peanoclaw::Patch& subgrid,
  peanoclaw::grid::SubgridAccessor& accessor,
  int dimension,
  bool setUpper,
  tarch::la::Vector<DIMENSIONS,int> sourceSubcellIndex,
  tarch::la::Vector<DIMENSIONS,int> destinationSubcellIndex
) {
//  FullSWOF2D_Parameters parameters(
//    subgrid.getGhostlayerWidth(),
//    subgrid.getSubdivisionFactor()[0],
//    subgrid.getSubdivisionFactor()[1],
//    subgrid.getSubcellSize()[0],
//    subgrid.getSubcellSize()[1],
//    subgrid.getTimeIntervals().getCurrentTime() + 10000, //end Time just large enough
//    false
//  );
//
//  TAB z;
//
//  int n1 = 0;
//  int n2 = 0;
//  int conditionType;
//  int normalIndex;
//  int tangentialIndex;
//  double imposedWaterHeight = 0.0;
//
//  //Map dimension,setUpper to n1,n2
//  if(dimension == 0) {
//    normalIndex = 1;
//    tangentialIndex = 2;
//    if(setUpper) {
//      imposedWaterHeight = 0.902921;
//      conditionType = 1;
//      n2 = 1;
//    } else {
//      conditionType = 5;
//      n2 = -1;
//    }
//  } else if (dimension == 1) {
//    conditionType = 2;
//    normalIndex = 2;
//    tangentialIndex = 1;
//    n1 = setUpper ? 1 : -1;
//  }
//
//  Choice_condition boundaryCondition(conditionType, parameters, z, n1, n2);
//
//  double waterHeight = accessor.getValueUOld(sourceSubcellIndex, 0);
//  double normalVelocity = 0.0;
//  double tangentialVelocity = 0.0;
//
//  if(tarch::la::greater(waterHeight, 0.0)) {
//    normalVelocity = accessor.getValueUOld(sourceSubcellIndex, normalIndex) / waterHeight;
//    tangentialVelocity = accessor.getValueUOld(sourceSubcellIndex, tangentialIndex) / waterHeight;
//  }
//
//  boundaryCondition.calcul(
//    waterHeight,
//    normalVelocity,
//    tangentialVelocity,
//    imposedWaterHeight, // imposed water height
//    20, // imposed discharge
//    0, // unused
//    0, // unused
//    0, // unused
//    0, // unused
//    n1,
//    n2
//  );
//
//  accessor.setValueUOld(destinationSubcellIndex, 0, boundaryCondition.get_hbound());
//  accessor.setValueUOld(destinationSubcellIndex, normalIndex, boundaryCondition.get_unormbound());
//  accessor.setValueUOld(destinationSubcellIndex, tangentialIndex, boundaryCondition.get_utanbound());
}

peanoclaw::native::scenarios::FullSWOF2DBoundaryCondition
peanoclaw::native::scenarios::swashes::ChannelPseudo2D::getBoundaryCondition(int dimension, bool upper) const {

  if(dimension == 0 && !upper) {
    return FullSWOF2DBoundaryCondition(5, 20, 1);
  }

  if(dimension == 0 && upper) {
    return FullSWOF2DBoundaryCondition(1, 0, _swashesChannel->getOutflowHeight());
  }

  return FullSWOF2DBoundaryCondition(2, 0, 0);
}
