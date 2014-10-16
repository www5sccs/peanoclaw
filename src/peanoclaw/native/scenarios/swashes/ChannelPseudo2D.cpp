/*
 * ChannelPseudo2D.cpp
 *
 *  Created on: Oct 15, 2014
 *      Author: kristof
 */
#include "peanoclaw/native/scenarios/swashes/ChannelPseudo2D.h"

#include "tarch/la/VectorAssignList.h"

peanoclaw::native::scenarios::swashes::ChannelPseudo2D::ChannelPseudo2D(
  std::vector<std::string> arguments
) : _domainSize(9.98067039818582),
    _discharge(20)
{
  #ifdef Dim2
  assignList(_domainOffset) = 0, -5;
  #endif

  if(arguments.size() != 6) {
    std::cerr << "Expected arguments for Scenario 'ChannelPseudo2D': finestSubgridTopology coarsestSubgridTopology subdivisionFactor endTime globalTimestepSize channelLength" << std::endl
        << "\tGot " << arguments.size() << " arguments." << std::endl
        << "Parameters:" << std::endl
        << " - channelLength: 'short': 200m channel length, 'long': 400m channel length" << std::endl;
    throw "";
  }

  double finestSubgridTopologyPerDimension = atof(arguments[0].c_str());
  _minimalMeshWidth = _domainSize/ finestSubgridTopologyPerDimension;

  double coarsestSubgridTopologyPerDimension = atof(arguments[1].c_str());
  _maximalMeshWidth = _domainSize/ coarsestSubgridTopologyPerDimension;

  _subdivisionFactor = tarch::la::Vector<DIMENSIONS,int>(atoi(arguments[2].c_str()));

  _endTime = atof(arguments[3].c_str());

  _globalTimestepSize = atof(arguments[4].c_str());

  if(arguments[5] == "short") {
    _channelType = Short;
    _domainSize[0] = 200;
  } else if (arguments[6] == "long") {
    _channelType = Long;
    _domainSize[0] = 400;
  } else {
    std::cerr << " Possible values for parameter channelLength are: 'short': 200m channel length, 'long': 400m channel length" << std::endl;
  }
}

peanoclaw::native::scenarios::swashes::ChannelPseudo2D::~ChannelPseudo2D() {
}

void peanoclaw::native::scenarios::swashes::ChannelPseudo2D::initializePatch(peanoclaw::Patch& patch) {
  #ifdef Dim2
  for(int y = 0; y < patch.getSubdivisionFactor()[1]; y++) {
    for(int x = 0; x < patch.getSubdivisionFactor()[0]; x++) {
      tarch::la::Vector<DIMENSIONS,int> subcellIndex;
      assignList(subcellIndex) = x, y;
      tarch::la::Vector<DIMENSIONS,double> position = patch.getSubcellPosition(subcellIndex);

      double distanceFromCenterLine = abs(x - _domainSize[0] / 2.0);
      double bedWidth =

    }
  }
  #endif
}


tarch::la::Vector<DIMENSIONS,double> peanoclaw::native::scenarios::swashes::ChannelPseudo2D::computeDemandedMeshWidth(
  peanoclaw::Patch& patch,
  bool isInitializing
) {

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
