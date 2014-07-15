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
  peanoclaw::Patch& subgrid
) {

  if(!tarch::la::oneGreater(subgrid.getPosition(), _position)
    && !tarch::la::oneGreater(_position, subgrid.getPosition() + subgrid.getSize())) {
    std::stringstream stringstream;
    stringstream << _name << " " << _position << " " << (subgrid.getTimeIntervals().getCurrentTime() + subgrid.getTimeIntervals().getTimestepSize()) << " ";

    peanoclaw::grid::SubgridAccessor accessor = subgrid.getAccessor();

    tarch::la::Vector<DIMENSIONS,double> relativeSubcellPosition = _position - subgrid.getPosition();
    tarch::la::Vector<DIMENSIONS,int> subcellIndex
      = tarch::la::multiplyComponents(relativeSubcellPosition, tarch::la::invertEntries(subgrid.getSubcellSize())).convertScalar<int>();

    if(_unknown == -1) {
      for(int unknown = 0; unknown < subgrid.getUnknownsPerSubcell(); unknown++) {
        stringstream << accessor.getValueUNew(subcellIndex, unknown) << " ";
      }
    } else {
      stringstream << accessor.getValueUNew(subcellIndex, _unknown) << " ";
    }

    logInfo("plotDataIfContainedInPatch(Patch)", stringstream.str() << "   ");
  }
}
