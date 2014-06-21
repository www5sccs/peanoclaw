/*
 * VTKPlotter.h
 *
 *  Created on: Jun 20, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_GRID_PLOTTER_VTKSUBGRIDPLOTTER_H_
#define PEANOCLAW_GRID_PLOTTER_VTKSUBGRIDPLOTTER_H_

#include "peanoclaw/grid/plotter/SubgridPlotter.h"

#include "peano/utils/Dimensions.h"
#include "tarch/la/Vector.h"
#include "tarch/la/VectorCompare.h"
#include "tarch/logging/Log.h"
#include "tarch/multicore/BooleanSemaphore.h"
#include "tarch/plotter/griddata/unstructured/vtk/VTKTextFileWriter.h"

#include <vector>
#include <set>
#include <map>

namespace peanoclaw {
  class Patch;
  namespace grid {
    class SubgridAccessor;
    namespace plotter {
      class VTKSubgridPlotter;
    }
  }
}

class peanoclaw::grid::plotter::VTKSubgridPlotter : public peanoclaw::grid::plotter::SubgridPlotter {

  private:
    /**
     * Logging device.
     */
    static tarch::logging::Log _log;

    std::string _fileName;

    /**
     * Vtk writer. This is a real instance and not a pointer, different to the
     * subwriters.
     */
    tarch::plotter::griddata::unstructured::vtk::VTKTextFileWriter _vtkWriter;

    tarch::multicore::BooleanSemaphore _vertex2IndexMapSemaphore;
    /**
     * Map from vertex positions to vertex indices
     */
    std::map<tarch::la::Vector<DIMENSIONS,double> , int, tarch::la::VectorCompare<DIMENSIONS> >   _vertex2IndexMap;
    /**
     * Plotter for vertices
     */
    tarch::plotter::griddata::unstructured::UnstructuredGridWriter::VertexWriter*                 _vertexWriter;
    /**
     * Plotter for cells
     */
    tarch::plotter::griddata::unstructured::UnstructuredGridWriter::CellWriter*                   _cellWriter;
    /**
     * Plotter for the subdivision factor
     */
    tarch::plotter::griddata::Writer::CellDataWriter*                                             _cellSubdivisionFactorWriter;
    /**
     * Plotter for the width of the ghost layer of this cell
     */
    tarch::plotter::griddata::Writer::CellDataWriter*                                             _cellGhostLayerWidthWriter;
    /**
     * Vector of writer for the actual data stored in the q array
     */
    std::vector<tarch::plotter::griddata::Writer::CellDataWriter*>                                _cellQWriter;
    /**
     * Vector of writer for the data stored in the parameter array without ghostlayer
     */
    std::vector<tarch::plotter::griddata::Writer::CellDataWriter*>                                _cellParameterWithoutGhostlayerWriter;
    /**
     * Vector of writer for the data stored in the parameter array with ghostlayer
     */
    std::vector<tarch::plotter::griddata::Writer::CellDataWriter*>                                _cellParameterWithGhostlayerWriter;
    /**
     * Plotter for writing the old time of the current Patch.
     */
    tarch::plotter::griddata::Writer::CellDataWriter*                                             _cellTimeOldWriter;
    /**
     * Plotter for writing the new time of the current Patch.
     */
    tarch::plotter::griddata::Writer::CellDataWriter*                                             _cellTimeNewWriter;
    /**
     * Plotter for writing the demanded mesh width.
     */
    tarch::plotter::griddata::Writer::CellDataWriter*                                             _cellDemandedMeshWidthWriter;
    /**
     * Plotter for writing the age of the patch.
     */
    tarch::plotter::griddata::Writer::CellDataWriter*                                             _cellAgeWriter;

  #ifdef Parallel
    /**
     * Writes the current MPI rank of the cell.
     */
    tarch::plotter::griddata::Writer::CellDataWriter*                                             _cellRankWriter;
  #endif

    double _gap;

    std::set<int> _plotQ;

    std::set<int> _plotParameterWithoutGhostlayer;
    std::set<int> _plotParameterWithGhostlayer;

    bool _plotMetainformation;

    /**
     * Computes the gradient for a subcell within the given
     * subgrid.
     */
    tarch::la::Vector<DIMENSIONS, double> computeGradient(
      const Patch&                            patch,
      const peanoclaw::grid::SubgridAccessor& accessor,
      tarch::la::Vector<DIMENSIONS, int>      subcellIndex,
      int                                     unknown
    );

    /**
     * Closes all writers.
     */
    void close();

  public:
    VTKSubgridPlotter(
      std::string fileName,
      int unknownsPerSubcell,
      int parametersWithoutGhostlayerPerSubcell,
      int parametersWithGhostlayerPerSubcell,
      std::set<int> plotQ,
      std::set<int> plotParameterWithoutGhostlayer,
      std::set<int> plotParameterWithGhostlayer,
      bool plotMetainformation
    );

    virtual ~VTKSubgridPlotter();

    /**
     * Plots a single subcell.
     */
    void plotSubcell(
      const Patch&                            patch,
      const peanoclaw::grid::SubgridAccessor& accessor,
      tarch::la::Vector<DIMENSIONS, int>      subcellIndex
    );

    void plotSubgrid(
      const Patch& subgrid
    );
};


#endif /* PEANOCLAW_GRID_PLOTTER_VTKSUBGRIDPLOTTER_H_ */
