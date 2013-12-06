/*
 * PatchPlotter.cpp
 *
 *      Author: Kristof Unterweger
 */
#include "PatchPlotter.h"

#include "Patch.h"

#include "peano/utils/Loop.h"

tarch::logging::Log peanoclaw::PatchPlotter::_log( "peanoclaw::PatchPlotter" );

void peanoclaw::PatchPlotter::plotSubcell(
  const Patch&                         patch,
  tarch::la::Vector<DIMENSIONS, int>   subcellIndex,
  peanoclaw::Vertex * const            vertices,
  const peano::grid::VertexEnumerator& enumerator
) {
  double localGap = _gap * patch.getLevel();
  tarch::la::Vector<DIMENSIONS, double> subcellSize = patch.getSubcellSize() / (1.0 + localGap);
  tarch::la::Vector<DIMENSIONS, double> x = patch.getPosition() + patch.getSize() * localGap/2.0 + tarch::la::multiplyComponents(subcellIndex.convertScalar<double>(), subcellSize);

  //Retreive adjacent vertices.
  int vertexIndices[TWO_POWER_D];
  dfor2(vertexIndex)
    tarch::la::Vector<DIMENSIONS,double> currentVertexPosition = x + tarch::la::multiplyComponents(vertexIndex.convertScalar<double>(), subcellSize);
    assertion1 ( _vertex2IndexMap.find(currentVertexPosition) != _vertex2IndexMap.end(), currentVertexPosition );
    vertexIndices[vertexIndexScalar] = _vertex2IndexMap[currentVertexPosition];
  enddforx

  int number = -1;
#ifdef Dim2
    number = _cellWriter->plotQuadrangle(vertexIndices);
#elif Dim3
    number = _cellWriter->plotHexahedron(vertexIndices);
#endif
    _cellSubdivisionFactorWriter->plotCell(number, patch.getSubdivisionFactor()(0));
    _cellGhostLayerWidthWriter->plotCell(number, patch.getGhostLayerWidth());
    for(int i = 0; i < (int)_cellQWriter.size(); i++) {
      _cellQWriter[i]->plotCell(number, patch.getValueUNew(subcellIndex, i));
    }
    for(int i = 0; i < (int)_cellAuxWriter.size(); i++) {
      _cellAuxWriter[i]->plotCell(number, patch.getValueAux(subcellIndex, i));
    }
    _cellTimeOldWriter->plotCell(number, patch.getCurrentTime());
    _cellTimeNewWriter->plotCell(number, patch.getCurrentTime() + patch.getTimestepSize());
    _cellDemandedMeshWidthWriter->plotCell(number, patch.getDemandedMeshWidth());

    _cellAgeWriter->plotCell(number, patch.getAge());
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::PatchPlotter::computeGradient(
  const Patch&                       patch,
  tarch::la::Vector<DIMENSIONS, int> subcellIndex,
  int                                unknown
) {
  tarch::la::Vector<DIMENSIONS, double> gradient;

  for(int d = 0; d < DIMENSIONS; d++) {
    tarch::la::Vector<DIMENSIONS, int> neighborIndex = subcellIndex;
    neighborIndex(d)++;
    gradient(d) = (patch.getValueUOld(neighborIndex, unknown) - patch.getValueUOld(subcellIndex, unknown)) / patch.getSubcellSize()(d);
  }

  return gradient;
}

peanoclaw::PatchPlotter::PatchPlotter(
  tarch::plotter::griddata::unstructured::vtk::VTKTextFileWriter& vtkWriter,
  int unknownsPerSubcell,
  int auxFieldsPerSubcell
) : _vtkWriter(vtkWriter), _gap(0.01) {

  _vertex2IndexMap.clear();
  _vertexWriter     = _vtkWriter.createVertexWriter();
  _cellWriter       = _vtkWriter.createCellWriter();

  _cellSubdivisionFactorWriter = _vtkWriter.createCellDataWriter("SubdivisionFactor",1);
  _cellGhostLayerWidthWriter   = _vtkWriter.createCellDataWriter("GhostlayerWidth",1);
  for(int i = 0; i < unknownsPerSubcell; i++) {
    std::stringstream stringstream;
    stringstream << "q" << i;
    _cellQWriter.push_back( _vtkWriter.createCellDataWriter(stringstream.str(), 1) );
  }
  for(int i = 0; i < auxFieldsPerSubcell; i++) {
    std::stringstream stringstream;
    stringstream << "aux" << i;
    _cellAuxWriter.push_back( _vtkWriter.createCellDataWriter(stringstream.str(), 1) );
  }
  _cellTimeOldWriter           = _vtkWriter.createCellDataWriter("timeOld", 1);
  _cellTimeNewWriter           = _vtkWriter.createCellDataWriter("timeNew", 1);
  _cellDemandedMeshWidthWriter = _vtkWriter.createCellDataWriter("demandedMeshWidth", 1);
  _cellAgeWriter               = _vtkWriter.createCellDataWriter("age", 1);
  #ifdef Parallel
  _cellRankWriter               = _vtkWriter.createCellDataWriter("rank", 1);
  #endif
}

peanoclaw::PatchPlotter::~PatchPlotter() {
  delete _vertexWriter;
  delete _cellWriter;
  delete _cellSubdivisionFactorWriter;
  delete _cellGhostLayerWidthWriter;
  for(unsigned int i = 0; i < _cellQWriter.size(); i++) {
    delete _cellQWriter.at(i);
  }
  for(unsigned int i = 0; i < _cellAuxWriter.size(); i++) {
    delete _cellAuxWriter.at(i);
  }
  delete _cellTimeOldWriter;
  delete _cellTimeNewWriter;
  delete _cellDemandedMeshWidthWriter;
  delete _cellAgeWriter;
  #ifdef Parallel
  delete _cellRankWriter;
  #endif
}

void peanoclaw::PatchPlotter::plotPatch(
  const Patch& patch,
  peanoclaw::Vertex * const        vertices,
  const peano::grid::VertexEnumerator&              enumerator
) {
  double localGap = _gap * patch.getLevel();
  tarch::la::Vector<DIMENSIONS, double> subcellSize = patch.getSubcellSize() / (1.0 + localGap);

  // Plot vertices
  dfor( vertexIndex, patch.getSubdivisionFactor()+1) {
    tarch::la::Vector<DIMENSIONS, double> x = patch.getPosition() + patch.getSize() * localGap/2.0 + tarch::la::multiplyComponents(vertexIndex.convertScalar<double>(), subcellSize);
    if ( _vertex2IndexMap.find(x) == _vertex2IndexMap.end() ) {
      assertion(_vertexWriter!=NULL);
      #if defined(Dim2) || defined(Dim3)
      int index = _vertexWriter->plotVertex(x);
      _vertex2IndexMap[x] = index;
      #else
      _vertex2IndexMap[x] = _vertexWriter->plotVertex(tarch::la::Vector<3,double>(x.data()));
      #endif
    }
  }

  // Plot cells
  dfor(subcellIndex, patch.getSubdivisionFactor()) {
//    if(tarch::la::norm2(computeGradient(patch, subcellIndex, 0)) > 1.0) {
      plotSubcell(
        patch,
        subcellIndex,
        vertices,
        enumerator
      );
//    }
  }
}

void peanoclaw::PatchPlotter::close() {
  _vertexWriter->close();
  _cellWriter->close();
  _cellSubdivisionFactorWriter->close();
  _cellGhostLayerWidthWriter->close();
  for(std::vector<tarch::plotter::griddata::Writer::CellDataWriter*>::iterator i = _cellQWriter.begin(); i != _cellQWriter.end(); i++) {
    (*i)->close();
  }
  for(std::vector<tarch::plotter::griddata::Writer::CellDataWriter*>::iterator i = _cellAuxWriter.begin(); i != _cellAuxWriter.end(); i++) {
    (*i)->close();
  }
  _cellTimeOldWriter->close();
  _cellTimeNewWriter->close();
  _cellDemandedMeshWidthWriter->close();
  _cellAgeWriter->close();
  _cellRankWriter->close();
}


