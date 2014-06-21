/*
 * PatchPlotter.cpp
 *
 *      Author: Kristof Unterweger
 */
#include "peanoclaw/grid/plotter/GridPlotter.h"

#include "peanoclaw/Patch.h"
#include "peanoclaw/grid/SubgridAccessor.h"
#include "peanoclaw/grid/plotter/SubgridPlotter.h"
#include "peanoclaw/grid/plotter/VTKSubgridPlotter.h"
#include "peanoclaw/grid/plotter/HDF5SubgridPlotter.h"

#include "peano/utils/Loop.h"

tarch::logging::Log peanoclaw::grid::plotter::GridPlotter::_log( "peanoclaw::grid::plotter::GridPlotter" );

peanoclaw::grid::plotter::GridPlotter::GridPlotter(
  const std::string& plotName,
  int plotNumber,
  int unknownsPerSubcell,
  int parameterWithoutGhostlayerPerSubcell,
  int parameterWithGhostlayerPerSubcell,
  std::set<int> plotQ,
  std::set<int> plotParameterWithoutGhostlayer,
  std::set<int> plotParameterWithGhostlayer,
  bool plotMetainformation,
  bool plotVTK,
  bool plotHDF5
) {
  std::ostringstream snapshotFileName;
  snapshotFileName << "vtkOutput/" << plotName << "-"
                   #ifdef Parallel
                   << "rank-" << tarch::parallel::Node::getInstance().getRank() << "-"
                   #endif
                   << plotNumber;

  if(plotVTK) {
    _subgridPlotter.push_back(new VTKSubgridPlotter(
        snapshotFileName.str() + ".vtk",
        unknownsPerSubcell,
        parameterWithoutGhostlayerPerSubcell,
        parameterWithGhostlayerPerSubcell,
        plotQ,
        plotParameterWithoutGhostlayer,
        plotParameterWithGhostlayer,
        plotMetainformation
        )
    );
  }

  if(plotHDF5) {
    _subgridPlotter.push_back(new HDF5SubgridPlotter(
        snapshotFileName.str() + ".hdf5",
        unknownsPerSubcell,
        parameterWithoutGhostlayerPerSubcell,
        parameterWithGhostlayerPerSubcell,
        plotQ,
        plotParameterWithoutGhostlayer,
        plotParameterWithGhostlayer,
        plotMetainformation
      )
    );
  }
}

peanoclaw::grid::plotter::GridPlotter::~GridPlotter() {
  for(std::vector<SubgridPlotter*>::iterator i = _subgridPlotter.begin(); i != _subgridPlotter.end(); i++) {
    delete *i;
  }
}

void peanoclaw::grid::plotter::GridPlotter::plotSubgrid(
  const Patch& subgrid//,
//  peanoclaw::Vertex * const        vertices,
//  const peano::grid::VertexEnumerator&              enumerator
) {
  assertion3(!subgrid.containsNaN(), subgrid, subgrid.toStringUNew(), subgrid.toStringUOldWithGhostLayer());
  for(std::vector<SubgridPlotter*>::iterator i = _subgridPlotter.begin(); i != _subgridPlotter.end(); i++) {
    (*i)->plotSubgrid(subgrid);
  }
}


