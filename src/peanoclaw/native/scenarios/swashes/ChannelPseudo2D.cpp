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

#include <fstream>

peanoclaw::native::scenarios::swashes::ChannelPseudo2D::ChannelPseudo2D(
  std::vector<std::string> arguments
) : _discharge(20),
    _channelType(Short),
    _swashesChannel(0)
{
  if(arguments.size() != 7) {
    std::cerr << "Expected arguments for Scenario 'ChannelPseudo2D': finestSubgridTopology coarsestSubgridTopology subdivisionFactor endTime globalTimestepSize channelLength criticality" << std::endl
        << "\tGot " << arguments.size() << " arguments." << std::endl
        << "Parameters:" << std::endl
        << " - channelLength: 'short': 200m channel length, 'long': 400m channel length" << std::endl
        << " -criticality: 'sub' or 'super'" << std::endl;
    throw "";
  }

  double finestSubgridTopologyPerDimension = atof(arguments[0].c_str());

  double coarsestSubgridTopologyPerDimension = atof(arguments[1].c_str());

  _domainSize[1] = 9.589575006880507;
  if(arguments[5] == "short") {
    _channelType = Short;
    _domainSize[0] = 200;
  } else if (arguments[5] == "long") {
    _channelType = Long;
    _domainSize[0] = 400;
  } else if (arguments[5] == "cornerTest") {
    _channelType = CornerTest;
    _domainSize = tarch::la::Vector<DIMENSIONS,double>(1.0);
  } else {
    std::cerr << " Possible values for parameter channelLength are: 'short': 200m channel length, 'long': 400m channel length" << std::endl;
  }
  if(arguments[6] == "sub") {
    _criticality = Sub;
  } else if (arguments[6] == "super") {
    _criticality = Super;
  } else {
    std::cerr << " Possible values for parameter criticality are: 'sub' and 'super'" << std::endl;
  }

  _subdivisionFactor[1] = atoi(arguments[2].c_str());
  if(_channelType == CornerTest) {
    _subdivisionFactor[0] = _subdivisionFactor[1];
  } else {
    _subdivisionFactor[0] = _subdivisionFactor[1] * 2; //_domainSize[0] / _domainSize[1];
  }

  #ifdef PEANOCLAW_SWASHES
  SWASHESParameters parameters(finestSubgridTopologyPerDimension * (float)_subdivisionFactor[0] / _subdivisionFactor[1], _criticality == Sub ? 1 : 2);
  _swashesChannel = new SWASHESShortChannel(parameters);
  #endif
  _swashesChannel->initialize();

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
        topography = BED_HEIGHT;
      }
      if(_channelType == CornerTest) {
        topography = (x == 3 && y == 5) ? 9.0 : 0.0;
      }
      accessor.setParameterWithGhostlayer(subcellIndex, 0, topography);

      if(x >= 0 && y >= 0 && x < subgrid.getSubdivisionFactor()[0] &&  y < subgrid.getSubdivisionFactor()[1]) {
        double waterheight = _criticality == Sub ? _swashesChannel->getInitialWaterHeight(position) : 0.0;

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

  //Write topography
  std::ofstream topographyFile("peanoclawTopography.dat");
  for(int x = 0; x < subgrid.getSubdivisionFactor()[0]; x++) {
    for(int y = 0; y < subgrid.getSubdivisionFactor()[1]; y++) {
      tarch::la::Vector<DIMENSIONS,int> subcellIndex;
      assignList(subcellIndex) = x, y;
      tarch::la::Vector<DIMENSIONS,double> position = subgrid.getSubcellCenter(subcellIndex);
      topographyFile << position[0] << " " << (position[1] - _domainOffset[1]) << " " << accessor.getParameterWithGhostlayer(subcellIndex, 0) << "\n";
    }
  }
  topographyFile.close();
  std::ofstream huvFile("peanoclawHUVInit.dat");
  for(int y = 0; y < subgrid.getSubdivisionFactor()[1]; y++) {
    for(int x = 0; x < subgrid.getSubdivisionFactor()[0]; x++) {
      tarch::la::Vector<DIMENSIONS,int> subcellIndex;
      assignList(subcellIndex) = x, y;
      tarch::la::Vector<DIMENSIONS,double> position = subgrid.getSubcellCenter(subcellIndex);
      huvFile << position[0] << " " << (position[1] - _domainOffset[1]) << " " << "0 0 0" << std::endl;
    }
  }
  huvFile.close();

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

peanoclaw::native::scenarios::FullSWOF2DBoundaryCondition
peanoclaw::native::scenarios::swashes::ChannelPseudo2D::getBoundaryCondition(int dimension, bool upper) const {

  if(_channelType == CornerTest) {
    if(dimension == 0 && !upper) {
      return FullSWOF2DBoundaryCondition(1, 2, 0.1); //Implied height
    }
    if(dimension == 0 && upper) {
      return FullSWOF2DBoundaryCondition(3, 2, 0.1); //Neumann
    }
  } else {
    if(_criticality == Sub) {
      if(dimension == 0 && !upper) {
        return FullSWOF2DBoundaryCondition(5, 20, 1); //Implied discharge
      }

      if(dimension == 0 && upper) {
        return FullSWOF2DBoundaryCondition(1, 0, _swashesChannel->getOutflowHeight()); //Implied height
      }
    } else if (_criticality == Super) {
      if(dimension == 0 && !upper) {
        return FullSWOF2DBoundaryCondition(1, 20, 0.503369); //Implied height
      }

      if(dimension == 0 && upper) {
        return FullSWOF2DBoundaryCondition(3, 0, 0); //Neumann
      }
    } else {
      std::cerr << "Unknown Criticality!" << std::endl;
      throw "";
    }
  }

  //All non-defined boundaries are walls
  return FullSWOF2DBoundaryCondition(2, 0, 0);
}
