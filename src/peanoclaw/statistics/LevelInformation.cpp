/*
 * LevelInformation.cpp
 *
 *  Created on: Oct 19, 2012
 *      Author: unterweg
 */
#include "peanoclaw/statistics/LevelInformation.h"

peanoclaw::statistics::LevelInformation::LevelInformation()
: _region(0.0),
  _level(0),
  _numberOfPatches(0.0),
  _numberOfCells(0.0),
  _numberOfCellUpdates(0.0),
  _createdPatches(0.0),
  _destroyedPatches(0.0),
  _patchesBlockedDueToNeighbors(0.0),
  _patchesBlockedDueToGlobalTimestep(0.0),
  _patchesSkippingIteration(0.0),
  _patchesCoarsening(0.0)
  {

}

