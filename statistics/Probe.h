/*
 * Probe.h
 *
 *  Created on: Oct 19, 2012
 *      Author: unterweg
 */

#ifndef PEANO_APPLICATIONS_PEANOCLAW_STATISTICS_PROBE_H_
#define PEANO_APPLICATIONS_PEANOCLAW_STATISTICS_PROBE_H_

#include "tarch/logging/Log.h"
#include "tarch/la/Vector.h"
#include "peano/utils/Dimensions.h"

#include <string>

namespace peanoclaw {
  class Patch;

  namespace statistics {
    class Probe;
  }
}

/**
 * A probe defines a point on which the solution is plotted every time
 * the containing patch gets updated.
 *
 * Hence, a probe consists of a position and a value determining the
 * unknown from the solution that should be plotted. Furthermore a
 * probe has a name to identify it in the output.
 * The output that is plotted contains the current solution value
 * and the time on which the patch currently resides.
 *
 * If the index for the unknown is set to -1 all unknowns in the
 * solution are plotted in a row.
 */
class peanoclaw::statistics::Probe {
  private:
    /**
     * Logging device.
     */
    static tarch::logging::Log _log;

    std::string                           _name;
    tarch::la::Vector<DIMENSIONS, double> _position;
    int                                   _unknown;

  public:
    Probe(
       std::string name,
      tarch::la::Vector<DIMENSIONS, double> position,
      int unknown
    );

    /**
     * Plots the data for this probe if the position is contained
     * in the given patch.
     *
     * If it is, the format for the output is
     *
     * Probe: <name> [<position>] <time> <value>
     *
     * If the index for the unknown is set to -1, all values are
     * plotted one after the other.
     */
    void plotDataIfContainedInPatch(
      peanoclaw::Patch& patch
    );
};

#endif /* PEANO_APPLICATIONS_PEANOCLAW_STATISTICS_PROBE_H_ */
