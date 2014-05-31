/*
 * SWECommandLineParser.h
 *
 *  Created on: May 28, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_NATIVE_SWECOMMANDLINEPARSER_H_
#define PEANOCLAW_NATIVE_SWECOMMANDLINEPARSER_H_

#include "peano/utils/Dimensions.h"
#include "tarch/la/Vector.h"
#include "tarch/logging/Log.h"

namespace peanoclaw {
  namespace native {
    class SWECommandLineParser;
  }
}

class peanoclaw::native::SWECommandLineParser {

  private:
    /**
     * Logging device.
     */
    static tarch::logging::Log _log;

    tarch::la::Vector<DIMENSIONS,int> _finestSubgridTopology;
    tarch::la::Vector<DIMENSIONS,int> _coarsestSubgridTopology;
    tarch::la::Vector<DIMENSIONS,int> _subdivisionFactor;
    double                            _endTime;
    double                            _globalTimestepSize;
    bool                              _usePeanoClaw;

  public:
    SWECommandLineParser(int argc, char** argv);

    /**
     * Returns the subgrid topology, i.e. the way in which the grid
     * is subdivided into subgrids. This method refers to the finest
     * subgrid topology, so it defines the largest height of the
     * spacetree.
     */
    tarch::la::Vector<DIMENSIONS,int> getFinestSubgridTopology() const;

    /**
     * Returns the subgrid topology, i.e. the way in which the grid
     * is subdivided into subgrids. This method refers to the finest
     * subgrid topology, so it defines the smallest height of the
     * spacetree during the simulation time.
     */
    tarch::la::Vector<DIMENSIONS,int> getCoarsestSubgridTopology() const;

    /**
     * Returns the subdivision factor for every subgrid in the
     * grid.
     */
    tarch::la::Vector<DIMENSIONS,int> getSubdivisionFactor() const;

    /**
     * Returns the end time for the simulation. As the simulated time
     * starts at 0, the simulated time interval is $[0, getEndTime()]$
     */
    double getEndTime() const;

    /**
     * Returns the size for global timesteps, i.e. the intervals in
     * which all subgrids get synchronized for plotting etc.
     */
    double getGlobalTimestepSize() const;

    /**
     * Indicates whether the simulation should be run with PeanoClaw or
     * with the solver applied in a pure manner.
     */
    bool runSimulationWithPeanoClaw() const;
};

#endif /* PEANOCLAW_NATIVE_SWECOMMANDLINEPARSER_H_ */
