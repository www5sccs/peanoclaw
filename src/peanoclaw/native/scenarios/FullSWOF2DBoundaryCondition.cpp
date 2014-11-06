/*
 * FullSWOF2DBoundaryCondition.cpp
 *
 *  Created on: Nov 4, 2014
 *      Author: kristof
 */
#include "peanoclaw/native/scenarios/FullSWOF2DBoundaryCondition.h"

peanoclaw::native::scenarios::FullSWOF2DBoundaryCondition::FullSWOF2DBoundaryCondition(
  int type,
  double impliedDischarge,
  double impliedHeight
) : _type(type),
    _impliedDischarge(impliedDischarge),
    _impliedHeight(impliedHeight) {
}

int peanoclaw::native::scenarios::FullSWOF2DBoundaryCondition::getType() const {
  return _type;
}

double peanoclaw::native::scenarios::FullSWOF2DBoundaryCondition::getImpliedDischarge() const {
  return _impliedDischarge;
}

double peanoclaw::native::scenarios::FullSWOF2DBoundaryCondition::getImpliedHeight() const {
  return _impliedHeight;
}


