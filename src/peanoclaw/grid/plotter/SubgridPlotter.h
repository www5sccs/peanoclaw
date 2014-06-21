/*
 * SubgridPlotter.h
 *
 *  Created on: Jun 20, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_GRID_PLOTTER_SUBGRIDPLOTTER_H_
#define PEANOCLAW_GRID_PLOTTER_SUBGRIDPLOTTER_H_

namespace peanoclaw {
  class Patch;

  namespace grid {
    namespace plotter {
      class SubgridPlotter;
    }
  }
}

/**
 * Interface for plotter that deal with a single subgrid at a time.
 */
class peanoclaw::grid::plotter::SubgridPlotter {

  public:
    virtual ~SubgridPlotter(){}

    virtual void plotSubgrid(
      const Patch& subgrid
    ) = 0;
};

#endif /* PEANOCLAW_GRID_PLOTTER_SUBGRIDPLOTTER_H_ */
