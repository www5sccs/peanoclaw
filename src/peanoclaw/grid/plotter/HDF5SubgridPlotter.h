/*
 * NetCDFSubgridPlotter.h
 *
 *  Created on: Jun 20, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_GRID_PLOTTER_HDF5SUBGRIDPLOTTER_H_
#define PEANOCLAW_GRID_PLOTTER_HDF5SUBGRIDPLOTTER_H_

#include "peanoclaw/grid/plotter/SubgridPlotter.h"

#include <hdf5.h>

#include <string>
#include <set>
#include <map>
#include <vector>

namespace peanoclaw {
  class Patch;
  namespace grid {
    namespace plotter {
      class HDF5SubgridPlotter;
    }
  }
}

class peanoclaw::grid::plotter::HDF5SubgridPlotter : public peanoclaw::grid::plotter::SubgridPlotter {

  private:
    hid_t _fileID;
    int   _datasetCounter;

  public:
    HDF5SubgridPlotter(
      std::string fileName,
      int unknownsPerSubcell,
      int parametersWithoutGhostlayerPerSubcell,
      int parametersWithGhostlayerPerSubcell,
      std::set<int> plotQ,
      std::set<int> plotParameterWithoutGhostlayer,
      std::set<int> plotParameterWithGhostlayer,
      bool plotMetainformation
    );

    virtual ~HDF5SubgridPlotter();

    void plotSubgrid(
      const Patch& subgrid
    );

};

#endif /* PEANOCLAW_GRID_PLOTTER_HDF5SUBGRIDPLOTTER_H_ */
