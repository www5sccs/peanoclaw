/*
 * ChannelPseudo2D.cpp
 *
 *  Created on: Oct 15, 2014
 *      Author: kristof
 */
#include "peanoclaw/native/scenarios/swashes/ChannelPseudo2D.h"

double peanoclaw::native::scenarios::swashes::ChannelPseudo2D::shortBedWidth(double x) const {
  return 10.0 - 5.0 * exp(-10 * pow(x/200 - 0.5, 2.0));
}

double peanoclaw::native::scenarios::swashes::ChannelPseudo2D::longBedWidth(double x) const {
  return 10.0 - 5.0 * exp(-50 * pow(x/400 - 1.0/3.0, 2.0)) - 5.0 * exp(-50 * pow(x/400 - 2.0/3.0, 2.0));
}

double peanoclaw::native::scenarios::swashes::ChannelPseudo2D::bedWidth(double x) const {
  if(_channelType == Short) {
    return shortBedWidth(x);
  } else {
    return longBedWidth(x);
  }
}

peanoclaw::native::scenarios::swashes::ChannelPseudo2D::ChannelPseudo2D(
  std::vector<std::string> arguments
) : _domainSize(10),
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

