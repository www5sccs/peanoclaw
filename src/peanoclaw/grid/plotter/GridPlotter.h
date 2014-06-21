/*
 * PatchPlotter.h
 *
 *      Author: Kristof Unterweger
 */

#ifndef PEANO_APPLICATIONS_PEANOCLAW_GRIDPLOTTER_H_
#define PEANO_APPLICATIONS_PEANOCLAW_GRIDPLOTTER_H_

#include "peanoclaw/records/CellDescription.h"
#include "peanoclaw/Vertex.h"

#include "tarch/plotter/griddata/unstructured/vtk/VTKTextFileWriter.h"
#include "tarch/la/VectorCompare.h"
#include "tarch/logging/Log.h"
#include "tarch/la/Vector.h"
#include "tarch/multicore/BooleanSemaphore.h"

#include <memory>
#include <map>
#include <set>
#include <vector>

#ifdef SharedTBB
#include <tbb/spin_mutex.h>
#endif

namespace peanoclaw {
  class Patch;

  namespace grid {
    namespace plotter {
      class GridPlotter;
      class SubgridPlotter;
      class VTKSubgridPlotter;
    }
    class SubgridAccessor;
  }
}

class peanoclaw::grid::plotter::GridPlotter {

private:
  /**
   * Logging device
   */
  static tarch::logging::Log _log;

  std::vector<SubgridPlotter*> _subgridPlotter;

public:
  /**
   * @param plotQ A list containing the indices of the q fields
   * that should be plotted. An empty vector is currently considered as
   * plotting all values.
   */
  GridPlotter(
    const std::string& plotName,
    int plotNumber,
    int unknownsPerSubcell,
    int parametersWithoutGhostlayerPerSubcell,
    int parametersWithGhostlayerPerSubcell,
    std::set<int> plotQ = std::set<int>(),
    std::set<int> plotParameterWithoutGhostlayer = std::set<int>(),
    std::set<int> plotParameterWithGhostlayer = std::set<int>(),
    bool plotMetainformation = true,
    bool plotVTK = true,
    bool plotNetCDF = true
  );

  ~GridPlotter();

  void plotSubgrid(
    const Patch&                         subgrid//,
//    peanoclaw::Vertex * const            vertices,
//    const peano::grid::VertexEnumerator& enumerator
  );

};

#endif /* PEANO_APPLICATIONS_PEANOCLAW_SUBGRIDPLOTTER_H_ */
