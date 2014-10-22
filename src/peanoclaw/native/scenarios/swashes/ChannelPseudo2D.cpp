/*
 * ChannelPseudo2D.cpp
 *
 *  Created on: Oct 15, 2014
 *      Author: kristof
 */
#include "peanoclaw/native/scenarios/swashes/ChannelPseudo2D.h"

#include "peanoclaw/Patch.h"

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

  _domainSize[1] = 9;
  if(arguments[5] == "short") {
    _channelType = Short;
    _domainSize[0] = 200;
    SWASHESParameters parameters(finestSubgridTopologyPerDimension * _domainSize[0] / _domainSize[1]);
    _swashesChannel = new SWASHESShortChannel(parameters);
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

void peanoclaw::native::scenarios::swashes::ChannelPseudo2D::initializePatch(peanoclaw::Patch& patch) {
  #ifdef Dim2
  peanoclaw::grid::SubgridAccessor accessor = patch.getAccessor();

  for(int y = 0; y < patch.getSubdivisionFactor()[1]; y++) {
    for(int x = 0; x < patch.getSubdivisionFactor()[0]; x++) {
      tarch::la::Vector<DIMENSIONS,int> subcellIndex;
      assignList(subcellIndex) = x, y;
      tarch::la::Vector<DIMENSIONS,double> position = patch.getSubcellCenter(subcellIndex);

      double distanceFromCenterLine = abs(position[1]);
      double bedWidth = _swashesChannel->getBedWidth(position[0]);
      double topography = _swashesChannel->getTopography(position[0]);

      if(distanceFromCenterLine > bedWidth / 2) {
        topography += BED_HEIGHT;
      }
      accessor.setParameterWithGhostlayer(subcellIndex, 0, topography);

      double waterheight = position[0] < 100 ? 1 : 0;
      if(topography >= BED_HEIGHT) {
        waterheight = 0;
      }
      accessor.setValueUNew(subcellIndex, 0, waterheight);
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
