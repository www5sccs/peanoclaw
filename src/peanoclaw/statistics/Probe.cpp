/*
 * Probe.cpp
 *
 *  Created on: Oct 19, 2012
 *      Author: unterweg
 */
#include "peanoclaw/statistics/Probe.h"

#include "peanoclaw/Patch.h"

#include <sstream>

tarch::logging::Log peanoclaw::statistics::Probe::_log("peanoclaw::statistics::Probe");

peanoclaw::statistics::Probe::Probe (
   std::string name,
  tarch::la::Vector<DIMENSIONS, double> position,
  int unknown
) : _name(name), _position(position), _unknown(unknown) {
}

void peanoclaw::statistics::Probe::plotDataIfContainedInPatch(
  peanoclaw::Patch& patch
) {
  assertionFail("Not fixed, yet.");
//  if(!tarch::la::oneGreater(patch.getPosition(), _position)
//    && !tarch::la::oneGreater(_position, patch.getPosition() + patch.getSize())) {
//    std::stringstream stringstream;
//    stringstream << _name << " " << _position << " " << (patch.getTimeIntervals().getCurrentTime() + patch.getTimeIntervals().getTimestepSize()) << " ";
//
//    if(_unknown == -1) {
//      for(int unknown = 0; unknown < patch.getUnknownsPerSubcell(); unknown++) {
//        stringstream << patch.getValueUNew(_position, unknown) << " ";
//      }
//    } else {
//      stringstream << patch.getValueUNew(_position, _unknown) << " ";
//    }
//
//    logInfo("plotDataIfContainedInPatch(Patch)", stringstream.str() << "   ");
//  }
}
