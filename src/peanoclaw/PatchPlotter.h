/*
 * PatchPlotter.h
 *
 *      Author: Kristof Unterweger
 */

#ifndef PEANO_APPLICATIONS_PEANOCLAW_PATCHPLOTTER_H_
#define PEANO_APPLICATIONS_PEANOCLAW_PATCHPLOTTER_H_

#include "tarch/logging/Log.h"
#include "tarch/la/Vector.h"
#include "tarch/multicore/BooleanSemaphore.h"

#include "records/CellDescription.h"

#include "Vertex.h"

#include "tarch/plotter/griddata/unstructured/vtk/VTKTextFileWriter.h"
#include "tarch/la/VectorCompare.h"

#include <memory>
#include <map>
#include <vector>

#ifdef SharedTBB
#include <tbb/spin_mutex.h>
#endif

namespace peanoclaw {
  class Patch;
  class PatchPlotter;
}

class peanoclaw::PatchPlotter {

private:
  /**
   * Logging device
   */
  static tarch::logging::Log _log;

  tarch::multicore::BooleanSemaphore _vertex2IndexMapSemaphore;
  /**
   * Map from vertex positions to vertex indices
   */
  std::map<tarch::la::Vector<DIMENSIONS,double> , int, tarch::la::VectorCompare<DIMENSIONS> >   _vertex2IndexMap;
  /**
   * General vtk writer
   */
  tarch::plotter::griddata::unstructured::vtk::VTKTextFileWriter&                               _vtkWriter;
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
   * Vector of writer for the data stored in the aux array
   */
  std::vector<tarch::plotter::griddata::Writer::CellDataWriter*>                                _cellAuxWriter;
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

public:
  PatchPlotter(
    tarch::plotter::griddata::unstructured::vtk::VTKTextFileWriter& vtkWriter,
    int unknownsPerSubcell,
    int auxFieldsPerSubcell
  );

  ~PatchPlotter();

  void plotPatch(
    Patch& patch,
    peanoclaw::Vertex * const        vertices,
    const peano::grid::VertexEnumerator&              enumerator
  );

  /**
   * Close all writers used by the PatchPlotter.
   */
  void close();

};

#endif /* PEANO_APPLICATIONS_PEANOCLAW_PATCHPLOTTER_H_ */
