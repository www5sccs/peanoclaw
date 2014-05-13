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
) : _subgrid(subgrid), _axis(axis), _maximumLinearError(0.0) {
  peanoclaw::grid::SubgridAccessor subgridAccessor = _subgrid.getAccessor();
  _linearSubcellIndex = subgridAccessor.getLinearIndexUOld(subcellIndex);

  tarch::la::Vector<DIMENSIONS, int> support0 = subcellIndex;
  support0(axis) = std::max(0, std::min(subgrid.getSubdivisionFactor()(axis) - 1, subcellIndex(axis)));
  tarch::la::Vector<DIMENSIONS, int> support1 = subcellIndex;
  support1(axis) = support0(axis) - direction;
  _linearIndexSupport0 = subgridAccessor.getLinearIndexUOld(support0);
  _linearIndexSupport1 = subgridAccessor.getLinearIndexUOld(support1);

  _distanceSupport0 = std::abs(support0(axis) - subcellIndex(axis));
  _distanceSupport1 = std::abs(support1(axis) - subcellIndex(axis));
}

double peanoclaw::interSubgridCommunication::ExtrapolationAxis::getExtrapolatedValue(
  int unknown
) {
  const peanoclaw::grid::SubgridAccessor& subgridAccessor = _subgrid.getAccessor();
  double valueSupport0 = subgridAccessor.getValueUOld(_linearIndexSupport0, unknown);
  double valueSupport1 = subgridAccessor.getValueUOld(_linearIndexSupport1, unknown);

  if(unknown == 0) {
    int extrapolationDistance = std::max(_distanceSupport0, _distanceSupport1) - 1;
    _maximumLinearError = std::max(_maximumLinearError,
      std::abs((valueSupport0 - valueSupport1) * extrapolationDistance) / std::min(valueSupport0, valueSupport1)
    );
  }

  logDebug("getExtrapolatedValue(int)", "valueSupport0=" << valueSupport0 << ", valueSupport1=" << valueSupport1 << ", distanceSupport0=" << _distanceSupport0 << ", distanceSupport1=" << _distanceSupport1);

  return (valueSupport0 * (_distanceSupport0+1) - valueSupport1 * (_distanceSupport1-1));
}

double peanoclaw::interSubgridCommunication::ExtrapolationAxis::getMaximumLinearError() const {
  return _maximumLinearError;
}

void peanoclaw::interSubgridCommunication::CornerExtrapolation::operator()(
  peanoclaw::Patch& subgrid,
  const peanoclaw::Area& area,
  const tarch::la::Vector<DIMENSIONS,int> cornerIndex
) {
  peanoclaw::grid::SubgridAccessor& subgridAccessor = subgrid.getAccessor();
  dfor(subcellIndexInArea, area._size) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndex = subcellIndexInArea + area._offset;
    int linearIndexSubcell = subgridAccessor.getLinearIndexUOld(subcellIndex);
    ExtrapolationAxis axis0(subcellIndex, subgrid, 0, cornerIndex(0));
    ExtrapolationAxis axis1(subcellIndex, subgrid, 1, cornerIndex(1));
    ExtrapolationAxis axis2(subcellIndex, subgrid, 2, cornerIndex(2));

    for(int unknown = 0; unknown < subgrid.getUnknownsPerSubcell(); unknown++) {
      double valueAxis0 = axis0.getExtrapolatedValue( unknown );
      double valueAxis1 = axis1.getExtrapolatedValue( unknown );
      double valueAxis2 = axis2.getExtrapolatedValue( unknown );

      //TODO unterweg debug
//      if(unknown == 0) {
//        std::cout << "subcellIndex=" << subcellIndex << ", cornerIndex=" << cornerIndex << std::endl;
//        std::cout << "Setting value to (" << valueAxis0 << "+" << valueAxis1 << "+" << valueAxis2 << ")/3=" << ((valueAxis0 + valueAxis1 + valueAxis2) / 3.0) << std::endl;
//        std::cout << std::endl;
//      }

      subgridAccessor.setValueUOld(linearIndexSubcell, unknown, (valueAxis0 + valueAxis1 + valueAxis2) / 3.0);

      _maximumLinearError = std::max(_maximumLinearError, axis0.getMaximumLinearError());
      _maximumLinearError = std::max(_maximumLinearError, axis1.getMaximumLinearError());
      _maximumLinearError = std::max(_maximumLinearError, axis2.getMaximumLinearError());
    }
  }
}

double peanoclaw::interSubgridCommunication::CornerExtrapolation::getMaximumLinearError() const {
  return _maximumLinearError;
}

peanoclaw::interSubgridCommunication::EdgeExtrapolation::EdgeExtrapolation()
  : _maximumLinearError(0) {
}

void peanoclaw::interSubgridCommunication::EdgeExtrapolation::operator ()(
  peanoclaw::Patch& subgrid,
  const peanoclaw::Area& area,
  const tarch::la::Vector<DIMENSIONS,int>& direction
) {
  _maximumLinearError = 0.0;
  peanoclaw::grid::SubgridAccessor& subgridAccessor = subgrid.getAccessor();

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

  subgridAccessor.clearRegion(area._offset, area._size, true);

  dfor(subcellIndexInArea, area._size) {
    tarch::la::Vector<DIMENSIONS, int> subcellIndex = subcellIndexInArea + area._offset;
    int linearIndexSubcell = subgridAccessor.getLinearIndexUOld(subcellIndex);

    ExtrapolationAxis axis0(subcellIndex, subgrid, dimensionAxis0, direction(dimensionAxis0));
    ExtrapolationAxis axis1(subcellIndex, subgrid, dimensionAxis1, direction(dimensionAxis1));

    for(int unknown = 0; unknown < subgrid.getUnknownsPerSubcell(); unknown++) {
      double valueAxis0 = axis0.getExtrapolatedValue( unknown );
      double valueAxis1 = axis1.getExtrapolatedValue( unknown );

      subgridAccessor.setValueUOld(linearIndexSubcell, unknown, (valueAxis0 + valueAxis1) / 2.0);

      _maximumLinearError = std::max(_maximumLinearError, axis0.getMaximumLinearError());
      _maximumLinearError = std::max(_maximumLinearError, axis1.getMaximumLinearError());
    }
  }
}

double peanoclaw::interSubgridCommunication::EdgeExtrapolation::getMaximumLinearError() const {
  return _maximumLinearError;
}

double peanoclaw::interSubgridCommunication::Extrapolation::extrapolateEdges() {
  peanoclaw::interSubgridCommunication::EdgeExtrapolation edgeExtrapolation;
  peanoclaw::interSubgridCommunication::aspects::EdgeTraversal<peanoclaw::interSubgridCommunication::EdgeExtrapolation>(
    _subgrid,
    edgeExtrapolation
  );

  return edgeExtrapolation.getMaximumLinearError();
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
  return cornerExtrapolation.getMaximumLinearError();
  #endif
}

peanoclaw::interSubgridCommunication::Extrapolation::Extrapolation(
  peanoclaw::Patch& subgrid
) : _subgrid(subgrid) {
}
