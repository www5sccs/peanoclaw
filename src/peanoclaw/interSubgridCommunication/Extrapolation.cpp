/*
 * Extrapolation.cpp
 *
 *  Created on: Jul 8, 2013
 *      Author: kristof
 */
#include "peanoclaw/interSubgridCommunication/Extrapolation.h"
#include "peanoclaw/interSubgridCommunication/aspects/CornerTraversal.h"
#include "peanoclaw/interSubgridCommunication/aspects/EdgeTraversal.h"

#include "peano/utils/Loop.h"

peanoclaw::interSubgridCommunication::ExtrapolationAxis::ExtrapolationAxis(
  const tarch::la::Vector<DIMENSIONS,int>& subcellIndex,
  const peanoclaw::Patch&                  subgrid,
  int                                      axis,
  int                                      direction
) : _subgrid(subgrid), _axis(axis), _maximumGradient(0.0) {
  _linearSubcellIndex = subgrid.getLinearIndexUOld(subcellIndex);

  tarch::la::Vector<DIMENSIONS, int> support0 = subcellIndex;
  support0(axis) = std::max(0, std::min(subgrid.getSubdivisionFactor()(axis) - 1, subcellIndex(axis)));
  tarch::la::Vector<DIMENSIONS, int> support1 = subcellIndex;
  support1(axis) = support0(axis) - direction;
  _linearIndexSupport0 = subgrid.getLinearIndexUOld(support0);
  _linearIndexSupport1 = subgrid.getLinearIndexUOld(support1);

  _distanceSupport0 = std::abs(support0(axis) - subcellIndex(axis));
  _distanceSupport1 = std::abs(support1(axis) - subcellIndex(axis));
}

double peanoclaw::interSubgridCommunication::ExtrapolationAxis::getExtrapolatedValue(
  int unknown
) {
  double valueSupport0 = _subgrid.getValueUOld(_linearIndexSupport0, unknown);
  double valueSupport1 = _subgrid.getValueUOld(_linearIndexSupport1, unknown);

  _maximumGradient = std::max(_maximumGradient, std::abs((valueSupport0 - valueSupport1) / _subgrid.getSubcellSize()(_axis)));

  //TODO unterweg debug
  std::cout << "vs0=" << valueSupport0 << ", vs1=" << valueSupport1 << ", ds0=" << _distanceSupport0 << ", ds1=" << _distanceSupport1 << std::endl;

  return (valueSupport0 * (_distanceSupport0+1) - valueSupport1 * (_distanceSupport1-1));
}

double peanoclaw::interSubgridCommunication::ExtrapolationAxis::getMaximumGradient() const {
  return _maximumGradient;
}

void peanoclaw::interSubgridCommunication::CornerExtrapolation::operator()(
  peanoclaw::Patch& subgrid,
  const peanoclaw::Area& area,
  const tarch::la::Vector<DIMENSIONS,int> cornerIndex
) {
  dfor(subcellIndexInArea, area._size) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndex = subcellIndexInArea + area._offset;
    int linearIndexSubcell = subgrid.getLinearIndexUOld(subcellIndex);
    ExtrapolationAxis axis0(subcellIndex, subgrid, 0, cornerIndex(0));
    ExtrapolationAxis axis1(subcellIndex, subgrid, 1, cornerIndex(1));
    ExtrapolationAxis axis2(subcellIndex, subgrid, 2, cornerIndex(2));

    for(int unknown = 0; unknown < subgrid.getUnknownsPerSubcell(); unknown++) {
      double valueAxis0 = axis0.getExtrapolatedValue( unknown );
      double valueAxis1 = axis1.getExtrapolatedValue( unknown );
      double valueAxis2 = axis2.getExtrapolatedValue( unknown );

      //TODO unterweg debug
      if(unknown == 0) {
        std::cout << "subcellIndex=" << subcellIndex << ", cornerIndex=" << cornerIndex << std::endl;
        std::cout << "Setting value to (" << valueAxis0 << "+" << valueAxis1 << "+" << valueAxis2 << ")/3=" << ((valueAxis0 + valueAxis1 + valueAxis2) / 3.0) << std::endl;
        std::cout << std::endl;
      }

      subgrid.setValueUOld(linearIndexSubcell, unknown, (valueAxis0 + valueAxis1 + valueAxis2) / 3.0);

      _maximumGradient = std::max(_maximumGradient, axis0.getMaximumGradient());
      _maximumGradient = std::max(_maximumGradient, axis1.getMaximumGradient());
      _maximumGradient = std::max(_maximumGradient, axis2.getMaximumGradient());
    }
  }
}

double peanoclaw::interSubgridCommunication::CornerExtrapolation::getMaximumGradient() const {
  return _maximumGradient;
}

peanoclaw::interSubgridCommunication::EdgeExtrapolation::EdgeExtrapolation()
  : _maximumGradient(0) {
}

void peanoclaw::interSubgridCommunication::EdgeExtrapolation::operator ()(
  peanoclaw::Patch& subgrid,
  const peanoclaw::Area& area,
  const tarch::la::Vector<DIMENSIONS,int>& direction
) {
  _maximumGradient = 0.0;

  int dimensionAxis0 = -1;
  int dimensionAxis1 = -1;

  for(int d = 0; d < DIMENSIONS; d++) {
    if(direction(d) != 0) {
      if(dimensionAxis0 == -1) {
        dimensionAxis0 = d;
      } else {
        dimensionAxis1 = d;
        break;
      }
    }
  }

  //TODO unterweg debug
  std::cout << "direction=" << direction << ", axis0=" << dimensionAxis0 << ", axis1=" << dimensionAxis1 << std::endl;

  subgrid.clearRegion(area._offset, area._size, true);

  dfor(subcellIndexInArea, area._size) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndex = subcellIndexInArea + area._offset;
    int linearIndexSubcell = subgrid.getLinearIndexUOld(subcellIndex);

    ExtrapolationAxis axis0(subcellIndex, subgrid, dimensionAxis0, direction(dimensionAxis0));
    ExtrapolationAxis axis1(subcellIndex, subgrid, dimensionAxis1, direction(dimensionAxis1));

    for(int unknown = 0; unknown < subgrid.getUnknownsPerSubcell(); unknown++) {
      double valueAxis0 = axis0.getExtrapolatedValue( unknown );
      double valueAxis1 = axis1.getExtrapolatedValue( unknown );

      //TODO unterweg debug
//      if(unknown == 0) {
//        std::cout << "subcellIndex=" << subcellIndex << std::endl;
//        std::cout << "Setting value to (" << valueAxis0 << "+" << valueAxis1 << ")/2=" << ((valueAxis0 + valueAxis1) / 2.0) << std::endl;
//        std::cout << std::endl;
//      }

      subgrid.setValueUOld(linearIndexSubcell, unknown, (valueAxis0 + valueAxis1) / 2.0);

      _maximumGradient = std::max(_maximumGradient, axis0.getMaximumGradient());
      _maximumGradient = std::max(_maximumGradient, axis1.getMaximumGradient());
    }
  }
}

double peanoclaw::interSubgridCommunication::EdgeExtrapolation::getMaximumGradient() const {
  return _maximumGradient;
}

double peanoclaw::interSubgridCommunication::Extrapolation::extrapolateEdges() {
  peanoclaw::interSubgridCommunication::EdgeExtrapolation edgeExtrapolation;
  peanoclaw::interSubgridCommunication::aspects::EdgeTraversal<peanoclaw::interSubgridCommunication::EdgeExtrapolation>(
    _subgrid,
    edgeExtrapolation
  );

  return edgeExtrapolation.getMaximumGradient();
}

double peanoclaw::interSubgridCommunication::Extrapolation::extrapolateCorners() {
  #ifdef Dim2
  return 0;
  #elif Dim3
  CornerExtrapolation cornerExtrapolation;
  peanoclaw::interSubgridCommunication::aspects::CornerTraversal<CornerExtrapolation> cornerTraversal(
    _subgrid,
    cornerExtrapolation
  );
  return cornerExtrapolation.getMaximumGradient();
  #endif
}

peanoclaw::interSubgridCommunication::Extrapolation::Extrapolation(
  peanoclaw::Patch& subgrid
) : _subgrid(subgrid) {
}


