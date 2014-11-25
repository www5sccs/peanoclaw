/*
 * NetCDFSubgridPlotter.cpp
 *
 *  Created on: Jun 20, 2014
 *      Author: kristof
 */
#include "peanoclaw/grid/plotter/HDF5SubgridPlotter.h"

#include "peanoclaw/Patch.h"

#include "peano/utils/Dimensions.h"

#include <sstream>
#ifdef PEANOCLAW_USE_HDF5
#include <hdf5.h>
#include <hdf5_hl.h>
#endif

peanoclaw::grid::plotter::HDF5SubgridPlotter::HDF5SubgridPlotter(
  std::string fileName,
  int unknownsPerSubcell,
  int parametersWithoutGhostlayerPerSubcell,
  int parametersWithGhostlayerPerSubcell,
  std::set<int> plotQ,
  std::set<int> plotParameterWithoutGhostlayer,
  std::set<int> plotParameterWithGhostlayer,
  bool plotMetainformation
) : _fileID(-1),
    _datasetCounter(0)
{
  #ifdef PEANOCLAW_USE_HDF5
  _fileID = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  #endif
}

peanoclaw::grid::plotter::HDF5SubgridPlotter::~HDF5SubgridPlotter() {
  #ifdef PEANOCLAW_USE_HDF5
  herr_t status = H5Fclose(_fileID);

  if(status < 0) {
    H5Eprint(stderr);
  }
  #endif
}


void peanoclaw::grid::plotter::HDF5SubgridPlotter::plotSubgrid(
  const Patch& subgrid
) {
  #ifdef PEANOCLAW_USE_HDF5
  //Dataset name
  std::stringstream s;
  s << "subgrid" << _datasetCounter;
  _datasetCounter++;

  //Dimensions
  hsize_t dimensions[DIMENSIONS];
  for(int d = 0; d < DIMENSIONS; d++) {
    dimensions[d] = subgrid.getSubdivisionFactor()[d];
  }

  //Dataset
  peanoclaw::grid::SubgridAccessor accessor = subgrid.getAccessor();
  std::vector<double> data;

  //dfor(subcellIndex, subgrid.getSubdivisionFactor()) {
  #ifdef Dim3
  for(int z = 0; z < subgrid.getSubdivisionFactor()[2]; z++) {
  #endif
    for(int y = 0; y < subgrid.getSubdivisionFactor()[1]; y++) {
      for(int x = 0; x < subgrid.getSubdivisionFactor()[0]; x++) {
        tarch::la::Vector<DIMENSIONS, int> subcellIndex;
        assignList(subcellIndex) = x, y
            #ifdef Dim3
            , z
            #endif
            ;
        data.push_back(accessor.getValueUNew(subcellIndex, 0));
      }
    }
  #ifdef Dim3
  }
  #endif
  //}

  H5LTmake_dataset(_fileID, s.str().c_str(), DIMENSIONS, dimensions, H5T_NATIVE_DOUBLE, data.data());

  //Attributes
  double position[DIMENSIONS];
  double size[DIMENSIONS];
  for(int d = 0; d < DIMENSIONS; d++) {
    position[d] = subgrid.getPosition()[d];
    size[d] = subgrid.getSize()[d];
  }
  H5LTset_attribute_double(_fileID, s.str().c_str(), "position", position, DIMENSIONS);
  H5LTset_attribute_double(_fileID, s.str().c_str(), "size", size, DIMENSIONS);

  //TODO unterweg debug
  //std::cout << "Plotted subgrid: " << std::endl << subgrid.toStringUNew() << std::endl;

  #endif
}

