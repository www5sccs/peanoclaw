/*
 * Patch.cpp
 *
 *      Author: Kristof Unterweger
 */
#include "Patch.h"

#include <sstream>
#include <iomanip>
#include <limits>

#include "peanoclaw/geometry/Region.h"
#include "Cell.h"
#include "Heap.h"

#include "tarch/la/AdditionalVectorOperations.h"

#include "peano/heap/Heap.h"
#include "peano/utils/Loop.h"

tarch::logging::Log peanoclaw::Patch::_log("peanoclaw::Patch");

//#define PATCH_VALUE_FORMAT std::setprecision(2) << std::scientific
#define PATCH_VALUE_FORMAT std::setprecision(7)

void peanoclaw::Patch::fillCaches() {
  #ifdef PEANOCLAW_SWE
  assertionEquals(sizeof(Data), sizeof(float));
  #else
  assertionEquals(sizeof(Data), sizeof(double));
  #endif
  tarch::la::Vector<DIMENSIONS, double> subdivisionFactor =
      _cellDescription->getSubdivisionFactor().convertScalar<double>();

  //Precompute subcell size
  tarch::la::Vector<DIMENSIONS,double> size = _cellDescription->getSize();
  for (int d = 0; d < DIMENSIONS; d++) {
    _subcellSize[d] = size[d] / subdivisionFactor[d];
  }
}

void peanoclaw::Patch::refreshAccessor() {
  _accessor = peanoclaw::grid::SubgridAccessor(
    isLeaf(),
    isVirtual(),
    *_cellDescription,
    _uNew
  );
}

void peanoclaw::Patch::resetAccessor() {
  _accessor = peanoclaw::grid::SubgridAccessor();
}

void peanoclaw::Patch::switchRegionToMinimalFineGridTimeInterval(
  const peanoclaw::geometry::Region& region,
  double factorForUOld,
  double factorForUNew
) {
  dfor(subcellIndex, region._size){
    int linearIndexUOld = _accessor.getLinearIndexUOld(subcellIndex + region._offset);
    int linearIndexUNew = _accessor.getLinearIndexUNew(subcellIndex + region._offset);

    for(int unknown = 0; unknown < _cellDescription->getUnknownsPerSubcell(); unknown++) {
      double valueUOld = _accessor.getValueUOld(linearIndexUOld, unknown);
      double valueUNew = _accessor.getValueUNew(linearIndexUNew, unknown);

      //UOld
      _accessor.setValueUOld(linearIndexUOld, unknown, valueUOld * (1.0 - factorForUOld) + valueUNew * factorForUOld);

      //UNew
      _accessor.setValueUNew(linearIndexUNew, unknown, valueUOld * (1.0 - factorForUNew) + valueUNew * factorForUNew);
    }
  }
}

bool peanoclaw::Patch::isLeaf(const CellDescription* cellDescription) {
  assertion(cellDescription != 0);
  return
      !cellDescription->getIsVirtual()
      && (cellDescription->getUIndex() != -1);
}

bool peanoclaw::Patch::isVirtual(const CellDescription* cellDescription) {
  if (cellDescription == 0) {
    return false;
  }
  assertion5(
      (!cellDescription->getIsVirtual() || ( cellDescription->getUIndex() != -1 )),
        cellDescription->getPosition(),
        cellDescription->getSize(),
        cellDescription->getIsVirtual(),
        cellDescription->getUIndex(),
        cellDescription->getCellDescriptionIndex());
  return cellDescription->getIsVirtual();
}

peanoclaw::Patch::Patch() :
    _cellDescription(0),
    _uNew(0)
    {
}

peanoclaw::Patch::Patch(const Cell& cell) :
  _cellDescription(0),
  _uNew(0)
{
  logTraceInWith1Argument("Patch(...)", cell);

  loadCellDescription(cell.getCellDescriptionIndex());

  logTraceOutWith1Argument("Patch(...)", _cellDescription);
}

peanoclaw::Patch::Patch(CellDescription& cellDescription) :
  _cellDescription(&cellDescription),
  _uNew(0)
{
  logTraceInWith2Arguments("Patch(CellDescription,int)", cellDescription.getPosition(), cellDescription.getLevel());

  assertion1(
      tarch::la::allGreater(getSize(), tarch::la::Vector<DIMENSIONS,double>(0.0)),
      toString());

  loadCellDescription(cellDescription.getCellDescriptionIndex());

  logTraceOut("Patch(CellDescription,int)");
}

peanoclaw::Patch::Patch(int cellDescriptionIndex)
  : _cellDescription(0),
    _uNew(0)
{
  loadCellDescription(cellDescriptionIndex);
}

peanoclaw::Patch::Patch(const tarch::la::Vector<DIMENSIONS, double>& position,
    const tarch::la::Vector<DIMENSIONS, double>& size, int unknownsPerSubcell,
    int parameterWithoutGhostlayer,
    int parameterWithGhostlayer,
    const tarch::la::Vector<DIMENSIONS, int>& subdivisionFactor,
    int ghostLayerWidth, double initialTimestepSize, int level)
: _cellDescription(0),
  _uNew(0)
{
  //Initialise Cell Description
  int cellDescriptionIndex =
      CellDescriptionHeap::getInstance().createData();

  std::vector<CellDescription>& cellDescriptions = CellDescriptionHeap::getInstance().getData(cellDescriptionIndex);

  CellDescription cellDescription;

  //Data
  cellDescription.setCellDescriptionIndex(cellDescriptionIndex);
  cellDescription.setUIndex(-1);

  //Geometry
  cellDescription.setSize(size);
  cellDescription.setPosition(position);
  cellDescription.setLevel(level);
  cellDescription.setSubdivisionFactor(subdivisionFactor);
  cellDescription.setGhostlayerWidth(ghostLayerWidth);

  //Timestepping
  cellDescription.setTime(0.0);
  cellDescription.setTimestepSize(0.0);
  cellDescription.setEstimatedNextTimestepSize(initialTimestepSize);
  cellDescription.setMinimalNeighborTimeConstraint(std::numeric_limits<double>::max());
  cellDescription.setConstrainingNeighborIndex(-1);
  cellDescription.setSkipGridIterations(0);
  cellDescription.setMinimalNeighborTime(std::numeric_limits<double>::max());
  cellDescription.setMaximalNeighborTimestep(-std::numeric_limits<double>::max());
  cellDescription.setMaximumFineGridTime(-1.0);
  cellDescription.setMinimumFineGridTimestep(std::numeric_limits<double>::max());
  cellDescription.setMinimalLeafNeighborTimeConstraint(std::numeric_limits<double>::max());

  //Numerics
  cellDescription.setUnknownsPerSubcell(unknownsPerSubcell);
  cellDescription.setNumberOfParametersWithoutGhostlayerPerSubcell(parameterWithoutGhostlayer);
  cellDescription.setNumberOfParametersWithGhostlayerPerSubcell(parameterWithGhostlayer);

  //Spacetree state
  cellDescription.setIsVirtual(false);
  cellDescription.setAgeInGridIterations(0);

  //Refinement
  cellDescription.setDemandedMeshWidth(
      tarch::la::multiplyComponents(size, tarch::la::invertEntries(subdivisionFactor.convertScalar<double>()))
  );
  cellDescription.setRestrictionLowerBounds(std::numeric_limits<double>::max());
  cellDescription.setRestrictionUpperBounds(-std::numeric_limits<double>::max());
  cellDescription.setSynchronizeFineGrids(false);
  cellDescription.setWillCoarsen(false);

  //Parallel
#ifdef Parallel
  cellDescription.setIsRemote(false);
  cellDescription.setIsPaddingSubgrid(false);
  cellDescription.setNumberOfSharedAdjacentVertices(0);
  cellDescription.setNumberOfTransfersToBeSkipped(0);
  cellDescription.setCurrentStateWasSent(false);
  cellDescription.setMarkStateAsSentInNextIteration(false);
#endif

  cellDescriptions.push_back(cellDescription);

  loadCellDescription(cellDescriptionIndex);

  assertionEquals(
      CellDescriptionHeap::getInstance().getData(
          cellDescriptionIndex).size(), 1);
}

peanoclaw::Patch::~Patch() {
}

const peanoclaw::grid::TimeIntervals& peanoclaw::Patch::getTimeIntervals() const {
  return _timeIntervals;
}

peanoclaw::grid::TimeIntervals& peanoclaw::Patch::getTimeIntervals() {
  return _timeIntervals;
}

const peanoclaw::grid::SubgridAccessor& peanoclaw::Patch::getAccessor() const {
  assertion(_accessor.isInitialized());
  return _accessor;
}
peanoclaw::grid::SubgridAccessor& peanoclaw::Patch::getAccessor() {
//  if(!_accessor.isInitialized()) {
//    refreshAccessor();
//    assertion(_accessor.isInitialized());
//  }
  return _accessor;
}

void peanoclaw::Patch::loadCellDescription(int cellDescriptionIndex) {
  _cellDescription = &(CellDescriptionHeap::getInstance().getData(cellDescriptionIndex)[0]);

  int uIndex = _cellDescription->getUIndex();
  if ( uIndex != -1) {
    _uNew = &DataHeap::getInstance().getData(uIndex);
  } else {
    _uNew = 0;
  }

  _timeIntervals = peanoclaw::grid::TimeIntervals(_cellDescription);
  refreshAccessor();

  fillCaches();
}

void peanoclaw::Patch::reloadCellDescription() {
  if (isValid()) {
    loadCellDescription(getCellDescriptionIndex());
  }
}

void peanoclaw::Patch::initializeNonParallelFields() {
  #ifdef Parallel
  assertion(isValid());
  _cellDescription->setConstrainingNeighborIndex(-1);
  _cellDescription->setCurrentStateWasSent(false);
  _cellDescription->setMarkStateAsSentInNextIteration(false);
  #endif
}

void peanoclaw::Patch::deleteData() {
  if(_cellDescription->getUIndex() != -1) {
    DataHeap::getInstance().deleteData(_cellDescription->getUIndex());
    _cellDescription->setUIndex(-1);
    _uNew = 0;
  }
  CellDescriptionHeap::getInstance().deleteData(_cellDescription->getCellDescriptionIndex());
  _cellDescription = 0;

  assertion1(!isValid(), toString());
}

const tarch::la::Vector<DIMENSIONS, double> peanoclaw::Patch::getSize() const {
  return _cellDescription->getSize();
}

const tarch::la::Vector<DIMENSIONS, double> peanoclaw::Patch::getPosition() const {
  return _cellDescription->getPosition();
}

int peanoclaw::Patch::getUnknownsPerSubcell() const {
  return _cellDescription->getUnknownsPerSubcell();
}

int peanoclaw::Patch::getNumberOfParametersWithoutGhostlayerPerSubcell() const {
  return _cellDescription->getNumberOfParametersWithoutGhostlayerPerSubcell();
}

int peanoclaw::Patch::getNumberOfParametersWithGhostlayerPerSubcell() const {
  return _cellDescription->getNumberOfParametersWithGhostlayerPerSubcell();
}

tarch::la::Vector<DIMENSIONS, int> peanoclaw::Patch::getSubdivisionFactor() const {
  return _cellDescription->getSubdivisionFactor();
}

int peanoclaw::Patch::getGhostlayerWidth() const {

  //TODO unterweg debug
//  std::cout << "_cellDescription=" << _cellDescription << std::endl;

  return _cellDescription->getGhostlayerWidth();
}

int peanoclaw::Patch::getLevel() const {
  return _cellDescription->getLevel();
}

void peanoclaw::Patch::setWillCoarsen(bool willCoarsen) {
  _cellDescription->setWillCoarsen(willCoarsen);
}

bool peanoclaw::Patch::willCoarsen() {
  return _cellDescription->getWillCoarsen();
}

void peanoclaw::Patch::switchValuesAndTimeIntervalToMinimalFineGridTimeInterval() {
  //TODO unterweg restricting to interval [0, 1]
//  double time = 0.0; //_cellDescription->getTime();
//  double timestepSize = 1.0; //_cellDescription->getTimestepSize();

  _cellDescription->setTime(_cellDescription->getMaximumFineGridTime());
  _cellDescription->setTimestepSize(
    _cellDescription->getMinimumFineGridTimestep()
  );

  //Interpolate patch values from former [time, time+timestepSize] to new [time, time+timestepSize]
//  if(isVirtual() || willCoarsen()) {
//    double factorForUOld = 1.0;
//    double factorForUNew = 1.0;
//    if(!tarch::la::equals(timestepSize, 0.0)) {
//      factorForUOld = (_cellDescription->getTime() - time) / timestepSize;
//      factorForUNew = (_cellDescription->getTime() + _cellDescription->getTimestepSize() - time) / timestepSize;
//    }
//
//    if (willCoarsen()) {
//      peanoclaw::geometry::Region region;
//      region._offset = tarch::la::Vector<DIMENSIONS, int>(0);
//      region._size = getSubdivisionFactor();
//      switchRegionToMinimalFineGridTimeInterval(region, factorForUOld, factorForUNew);
//    }
//    else if(isVirtual()) {
//      peanoclaw::geometry::Region regions[DIMENSIONS_TIMES_TWO];
//      int numberOfRegions = peanoclaw::getRegionsForRestriction(
//        getLowerNeighboringGhostlayerBounds(),
//        getUpperNeighboringGhostlayerBounds(),
//        getPosition(),
//        getSubcellSize(),
//        getSubcellSize(),
//        getSubdivisionFactor(),
//        regions
//      );
//
//      assertion1(numberOfRegions <= DIMENSIONS_TIMES_TWO, numberOfRegions);
//
//      for (int regionIndex = 0; regionIndex < numberOfRegions; regionIndex++) {
//        if(tarch::la::allGreater(regions[regionIndex]._size, tarch::la::Vector<DIMENSIONS, int>(0))) {
//          switchRegionToMinimalFineGridTimeInterval(regions[regionIndex], factorForUOld, factorForUNew);
//        }
//      }
//    }
//  }
}

void peanoclaw::Patch::setSkipNextGridIteration(int numberOfIterationsToSkip) {
  _cellDescription->setSkipGridIterations(numberOfIterationsToSkip);
}

bool peanoclaw::Patch::shouldSkipNextGridIteration() const {
  return _cellDescription->getSkipGridIterations() > 0;
}

void peanoclaw::Patch::reduceGridIterationsToBeSkipped() {
  _cellDescription->setSkipGridIterations(
      std::max(_cellDescription->getSkipGridIterations() - 1, 0));
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::Patch::getSubcellCenter(
    tarch::la::Vector<DIMENSIONS, int> subcellIndex) const {
  tarch::la::Vector<DIMENSIONS, double> subcellSize = getSubcellSize();
  tarch::la::Vector<DIMENSIONS, double> result = tarch::la::multiplyComponents(
      subcellSize, subcellIndex.convertScalar<double>());
  return _cellDescription->getPosition() + result + subcellSize * 0.5;
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::Patch::getSubcellPosition(
    tarch::la::Vector<DIMENSIONS, int> subcellIndex) const {
  tarch::la::Vector<DIMENSIONS, double> result = tarch::la::multiplyComponents(
      getSubcellSize(), subcellIndex.convertScalar<double>());
  return _cellDescription->getPosition() + result;
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::Patch::getSubcellSize() const {
  return _subcellSize;
}

int peanoclaw::Patch::getUIndex() const {
  return _cellDescription->getUIndex();
}

size_t peanoclaw::Patch::getUSize() const {
  return _uNew->size();
}

int peanoclaw::Patch::getCellDescriptionIndex() const {
  assertion1(_cellDescription != 0, *this);
  return _cellDescription->getCellDescriptionIndex();
}

void peanoclaw::Patch::setDemandedMeshWidth(const tarch::la::Vector<DIMENSIONS,double>& demandedMeshWidth) {
  _cellDescription->setDemandedMeshWidth(demandedMeshWidth);
}

tarch::la::Vector<DIMENSIONS,double> peanoclaw::Patch::getDemandedMeshWidth() const {
  return _cellDescription->getDemandedMeshWidth();
}

std::string peanoclaw::Patch::toStringUNew() const {
  if (isValid() && (isLeaf() || isVirtual())) {
    std::stringstream str;
    #ifdef Dim2
    //Plot patch
    for (int y = getSubdivisionFactor()(1) - 1; y >= 0; y--) {
      for (int x = 0; x < getSubdivisionFactor()(0); x++) {
        tarch::la::Vector<DIMENSIONS, int> subcellIndex;
        assignList(subcellIndex) = x, y;
        str << PATCH_VALUE_FORMAT << _accessor.getValueUNew(subcellIndex, 0) << " ";
      }
//      if (_cellDescription->getUnknownsPerSubcell() > 1) {
//        str << "\t";
//        for (int x = 0; x < getSubdivisionFactor()(0); x++) {
//          tarch::la::Vector<DIMENSIONS, int> subcellIndex;
//          assignList(subcellIndex) = x, y;
//          str << PATCH_VALUE_FORMAT << _accessor.getValueUNew(subcellIndex, 1) << ","
//              << PATCH_VALUE_FORMAT << _accessor.getValueUNew(subcellIndex, 2) << " ";
//        }
//      }
      str << std::endl;
    }
    str << std::endl;

    //Plot Bathymetry
//    for (int y = getSubdivisionFactor()(1) - 1; y >= 0; y--) {
//      for (int x = 0; x < getSubdivisionFactor()(0); x++) {
//        tarch::la::Vector<DIMENSIONS, int> subcellIndex;
//        assignList(subcellIndex) = x, y;
//        str << std::setprecision(6) << getValueUNew(subcellIndex, 3) << " ";
//      }
//      str << std::endl;
//    }
//    str << std::endl;

    #elif Dim3
    //Plot patch
    for(int z = 0; z < getSubdivisionFactor()(2); z++) {
      for (int y = getSubdivisionFactor()(1) - 1; y >= 0; y--) {
        for (int x = 0; x < getSubdivisionFactor()(0); x++) {
          tarch::la::Vector<DIMENSIONS, int> subcellIndex;
          assignList(subcellIndex) = x, y, z;
          str << PATCH_VALUE_FORMAT << _accessor.getValueUNew(subcellIndex, 0) << " ";
        }
//        if (_cellDescription->getUnknownsPerSubcell() > 1) {
//          str << "\t";
//          for (int x = 0; x < getSubdivisionFactor()(0); x++) {
//            tarch::la::Vector<DIMENSIONS, int> subcellIndex;
//            assignList(subcellIndex) = x, y, z;
//            str << PATCH_VALUE_FORMAT << getValueUNew(subcellIndex, 1) << ","
//                << PATCH_VALUE_FORMAT << getValueUNew(subcellIndex, 2) << ","
//                << PATCH_VALUE_FORMAT << getValueUNew(subcellIndex, 3) << " ";
//          }
//        }
        str << "\n";
      }
      str << "\n" << "\n";
    }
    str << "\n";
    #endif
    return str.str();
  } else {
    return "Is refined patch.\n";
  }
}

std::string peanoclaw::Patch::toStringUOldWithGhostLayer() const {
  if (isValid() && (isLeaf() || isVirtual())) {
    std::stringstream str;
    #ifdef Dim2
    for (int y = getSubdivisionFactor()(1) + getGhostlayerWidth() - 1;
        y >= -getGhostlayerWidth(); y--) {
      for (int x = -getGhostlayerWidth();
          x < getSubdivisionFactor()(0) + getGhostlayerWidth(); x++) {
        tarch::la::Vector<DIMENSIONS, int> subcellIndex;
        subcellIndex(0) = x;
        subcellIndex(1) = y;
        str << PATCH_VALUE_FORMAT << _accessor.getValueUOld(subcellIndex, 0) << " ";
      }
//      if (_cellDescription->getUnknownsPerSubcell() > 1) {
//        str << "\t";
//        for (int x = -getGhostlayerWidth();
//            x < getSubdivisionFactor()(0) + getGhostlayerWidth(); x++) {
//          tarch::la::Vector<DIMENSIONS, int> subcellIndex;
//          subcellIndex(0) = x;
//          subcellIndex(1) = y;
//          str << PATCH_VALUE_FORMAT << _accessor.getValueUOld(subcellIndex, 1) << ","
//              << PATCH_VALUE_FORMAT << _accessor.getValueUOld(subcellIndex, 2) << " ";
//        }
//      }
      str << "\n";
    }
    str << "\n";


    //Plot Bathymetry
//    for (int y = getSubdivisionFactor()(1)-1; y >= 0; y--) {
//      for (int x = 0; x < getSubdivisionFactor()(0); x++) {
//        tarch::la::Vector<DIMENSIONS, int> subcellIndex;
//        assignList(subcellIndex) = x, y;
//        str << std::setprecision(6) << getAccessor().getParameterWithoutGhostlayer(subcellIndex, 0) << " ";
//      }
//      str << std::endl;
//    }
//    str << std::endl;

    #elif Dim3
    //Plot patch
    for(int z = -getGhostlayerWidth(); z < getSubdivisionFactor()(2) + getGhostlayerWidth(); z++) {
      str << "z==" << z << std::endl;
      for (int y = getSubdivisionFactor()(1) + getGhostlayerWidth() - 1;
          y >= -getGhostlayerWidth(); y--) {
        for (int x = -getGhostlayerWidth();
            x < getSubdivisionFactor()(0) + getGhostlayerWidth(); x++) {
          tarch::la::Vector<DIMENSIONS, int> subcellIndex;
          assignList(subcellIndex) = x, y, z;
          str << PATCH_VALUE_FORMAT << _accessor.getValueUOld(subcellIndex, 0) << " ";
        }
        if (_cellDescription->getUnknownsPerSubcell() > 1) {
          str << "\t";
          for (int x = 0; x < getSubdivisionFactor()(0); x++) {
            tarch::la::Vector<DIMENSIONS, int> subcellIndex;
            assignList(subcellIndex) = x, y, z;
            str << PATCH_VALUE_FORMAT << _accessor.getValueUOld(subcellIndex, 1) << ","
                << PATCH_VALUE_FORMAT << _accessor.getValueUOld(subcellIndex, 2) << ","
                << PATCH_VALUE_FORMAT << _accessor.getValueUOld(subcellIndex, 3) << " ";
          }
        }
        str << std::endl;
      }
      str << "\n" << "\n";
    }
    str << "\n";
    #endif
    return str.str();
  } else {
    return "Is refined patch.\n";
  }
}

std::string peanoclaw::Patch::toStringParameters() const {
  if (isLeaf() || isVirtual()) {
    std::stringstream str;
    #ifdef Dim2

    //Plot parameters without ghostlayer
    for(int parameter = 0; parameter < getNumberOfParametersWithoutGhostlayerPerSubcell(); parameter++) {
      str << "Parameter without ghostlayer " << parameter << ":\n";
      for (int y = getSubdivisionFactor()(1) - 1; y >= 0; y--) {
        for (int x = 0; x < getSubdivisionFactor()(0); x++) {
          tarch::la::Vector<DIMENSIONS, int> subcellIndex;
          assignList(subcellIndex) = x, y;
          str << PATCH_VALUE_FORMAT << _accessor.getValueUNew(subcellIndex, 0) << " ";
        }
        str << "\n";
      }
      str << "\n";
    }

    //Plot parameters with ghostlayer
    for(int parameter = 0; parameter < getNumberOfParametersWithGhostlayerPerSubcell(); parameter++) {
      str << "Parameter with ghostlayer " << parameter << ": " << std::endl;
      for (int y = getSubdivisionFactor()(1) + getGhostlayerWidth() - 1;
          y >= -getGhostlayerWidth(); y--) {
        for (int x = -getGhostlayerWidth(); x < getSubdivisionFactor()(0) + getGhostlayerWidth(); x++) {
          tarch::la::Vector<DIMENSIONS, int> subcellIndex;
          subcellIndex(0) = x;
          subcellIndex(1) = y;
          str << PATCH_VALUE_FORMAT << _accessor.getParameterWithGhostlayer(subcellIndex, parameter) << " ";
        }
        str << "\n";
      }
      str << "\n";
    }

    #elif Dim3
    //Plot patch
    for(int z = -getGhostlayerWidth(); z < getSubdivisionFactor()(2) + getGhostlayerWidth(); z++) {
      str << "z==" << z << std::endl;
      for (int y = getSubdivisionFactor()(1) + getGhostlayerWidth() - 1;
          y >= -getGhostlayerWidth(); y--) {
        for (int x = -getGhostlayerWidth();
            x < getSubdivisionFactor()(0) + getGhostlayerWidth(); x++) {
          tarch::la::Vector<DIMENSIONS, int> subcellIndex;
          assignList(subcellIndex) = x, y, z;
          str << PATCH_VALUE_FORMAT << _accessor.getValueUOld(subcellIndex, 0) << " ";
        }
//        if (_cellDescription->getUnknownsPerSubcell() > 1) {
//          str << "\t";
//          for (int x = 0; x < getSubdivisionFactor()(0); x++) {
//            tarch::la::Vector<DIMENSIONS, int> subcellIndex;
//            assignList(subcellIndex) = x, y, z;
//            str << PATCH_VALUE_FORMAT << getValueUOld(subcellIndex, 1) << ","
//                << PATCH_VALUE_FORMAT << getValueUOld(subcellIndex, 2) << ","
//                << PATCH_VALUE_FORMAT << getValueUOld(subcellIndex, 3) << " ";
//          }
//        }
        str << std::endl;
      }
      str << std::endl << std::endl;
    }
    str << std::endl;
    #endif
    return str.str();
  } else {
    return "Is refined patch.\n";
  }
}

std::string peanoclaw::Patch::toString() const {
  std::stringstream str;
  if (_cellDescription != 0) {
    str << "size=" << _cellDescription->getSize() << ",position="
        << _cellDescription->getPosition() << ",level=" << getLevel()
        << ",isLeaf=" << isLeaf() << ",isVirtual=" << isVirtual()
        << ",subdivisionFactor=" << getSubdivisionFactor()
        << ",ghostlayerWidth=" << getGhostlayerWidth()
        << ",unknownsPerSubcell=" << _cellDescription->getUnknownsPerSubcell()
        << ",cellDescriptionIndex=" << getCellDescriptionIndex()
        << ",uIndex=" << getUIndex()
//        << ",elements in uNew=" << ((_uNew == 0) ? 0 : _uOldWithGhostlayerArrayIndex)
        << ",elements in uOld="
//        << ((_uNew == 0) ? 0 : (_parameterWithoutGhostlayerArrayIndex - _uOldWithGhostlayerArrayIndex))
        << ",age=" << _cellDescription->getAgeInGridIterations()

        << "," << _timeIntervals.toString()

        << ",constrainingIndex=" << _cellDescription->getConstrainingNeighborIndex()
        << ",skipNextGridIteration="
        << _cellDescription->getSkipGridIterations() << ",minimalNeighborTime="
        << _cellDescription->getMinimalNeighborTime()
        << ",maximalNeighborTimestepSize="
        << _cellDescription->getMaximalNeighborTimestep()
        << ",maximalFineGridTime=" << _cellDescription->getMaximumFineGridTime()
        << ",minimalFineGridTimestepSize="
        << _cellDescription->getMinimumFineGridTimestep()
        << ",estimatedNextTimestepSize="
        << _cellDescription->getEstimatedNextTimestepSize()
        << ",synchronizeFineGrids=" << _cellDescription->getSynchronizeFineGrids()
        << ",demandedMeshWidth=" << _cellDescription->getDemandedMeshWidth()
        << ",lowerGhostlayerBounds="
        << _cellDescription->getRestrictionLowerBounds()
        << ",upperGhostlayerBounds="
        << _cellDescription->getRestrictionUpperBounds()
#ifdef Parallel
        << ",isRemote=" << _cellDescription->getIsRemote()
        << ",adjacentRanks=" << _cellDescription->getAdjacentRanks()
        << ",overlapOfRemoteGhostlayers=" << _cellDescription->getOverlapByRemoteGhostlayer()
        << ",numberOfTransfersToBeSkipped=" << _cellDescription->getNumberOfTransfersToBeSkipped()
        << ",numberOfAdjacentSharedVertices=" << _cellDescription->getNumberOfSharedAdjacentVertices()
        << ",wasCurrentStateSent=" << _cellDescription->getCurrentStateWasSent()
        << ",markAsSentInNextIteration=" << _cellDescription->getMarkStateAsSentInNextIteration()
#endif
        ;
  } else {
    str << "null";
  }
  return str.str();
}

bool peanoclaw::Patch::isValid() const {
  return isValid(_cellDescription);
}

bool peanoclaw::Patch::isLeaf() const {
  if(isValid()) {
    assertionEquals2((_uNew == 0), (_cellDescription->getUIndex() == -1),
        _uNew, _cellDescription->getUIndex());
  }
  return isLeaf(_cellDescription);
}

bool peanoclaw::Patch::isVirtual() const {
  return isVirtual(_cellDescription);
}

void peanoclaw::Patch::switchToVirtual() {

  _cellDescription->setIsVirtual(true);
  //Create uNew if necessary
  if (_cellDescription->getUIndex() == -1) {
    int uNewIndex = DataHeap::getInstance().createData();
    _cellDescription->setUIndex(uNewIndex);
    std::vector<Data>& virtualUNew =
        DataHeap::getInstance().getData(uNewIndex);

    size_t uNewArraySize = tarch::la::volume(
        _cellDescription->getSubdivisionFactor())
        * _cellDescription->getUnknownsPerSubcell();

    size_t uOldWithGhostlayerArraySize = tarch::la::volume(
        _cellDescription->getSubdivisionFactor()
            + 2 * _cellDescription->getGhostlayerWidth())
        * _cellDescription->getUnknownsPerSubcell();

    size_t parameterWithoutGhostlayerArraySize = tarch::la::volume(
        _cellDescription->getSubdivisionFactor())
        * _cellDescription->getNumberOfParametersWithoutGhostlayerPerSubcell();

    size_t parameterWithGhostlayerArraySize = tarch::la::volume(
        _cellDescription->getSubdivisionFactor()
            + 2 * _cellDescription->getGhostlayerWidth())
        * _cellDescription->getNumberOfParametersWithGhostlayerPerSubcell();

    size_t fluxArraySize = 0;
    for(int d = 0; d < DIMENSIONS; d++) {
      fluxArraySize += 2 * _cellDescription->getUnknownsPerSubcell()
                         * tarch::la::projectedArea(_cellDescription->getSubdivisionFactor(), d);
    }

    virtualUNew.resize(
      uNewArraySize + uOldWithGhostlayerArraySize + parameterWithoutGhostlayerArraySize + parameterWithGhostlayerArraySize + fluxArraySize,
      0.0
    );
    _uNew = &virtualUNew;
  }

  refreshAccessor();

  assertion1(_uNew != 0 && _cellDescription->getUIndex() != -1, toString());
}

void peanoclaw::Patch::switchToNonVirtual() {
  assertion(!isLeaf())
  _cellDescription->setIsVirtual(false);
  if (_cellDescription->getUIndex() != -1) {
    DataHeap::getInstance().deleteData(getUIndex());
    _cellDescription->setUIndex(-1);
    _uNew = 0;
  }

  refreshAccessor();

  assertionEquals1(_uNew, 0, toString());
  assertion(!isLeaf() && !isVirtual());
}

void peanoclaw::Patch::switchToLeaf() {
  assertion(isVirtual() || isLeaf());
  assertion1(!tarch::la::smaller(_cellDescription->getTimestepSize(), 0.0),
      toString());
  assertion1(!tarch::la::smaller(_cellDescription->getMinimumFineGridTimestep(), 0.0),
      toString());
  _cellDescription->setIsVirtual(false);

  //TODO unterweg restricting to interval [0, 1]
  double time = 0.0; //_cellDescription->getTime();
  double timestepSize = 1.0; //_cellDescription->getTimestepSize();

  double factorForUOld = 1.0;
  double factorForUNew = 1.0;
  if (!tarch::la::equals(timestepSize, 0.0)) {
    factorForUOld = (_cellDescription->getTime() - time) / timestepSize;
    factorForUNew = (_cellDescription->getTime()
        + _cellDescription->getTimestepSize() - time) / timestepSize;
  }

  peanoclaw::geometry::Region region;
  region._offset = tarch::la::Vector<DIMENSIONS, int>(0);
  region._size = getSubdivisionFactor();
  switchRegionToMinimalFineGridTimeInterval(region, factorForUOld, factorForUNew);

  refreshAccessor();

  assertion1(!tarch::la::smaller(_cellDescription->getTimestepSize(), 0.0),
      toString());

#ifdef Asserts
  //Check uNew
  dfor(subcellIndex, getSubdivisionFactor()){
    assertion2(_accessor.getValueUNew(subcellIndex, 0) >= 0.0,
      toString(), subcellIndex);
  }
  //Check uOld
  dfor(subcellIndex, getSubdivisionFactor() + 2 * getGhostlayerWidth()){
    assertion2(_accessor.getValueUOld(subcellIndex - getGhostlayerWidth(), 0) >= 0.0,
      toString(), subcellIndex);
  }
#endif

  assertion(isLeaf());
}

bool peanoclaw::Patch::shouldRestrict() {
  return tarch::la::oneGreater(_cellDescription->getRestrictionLowerBounds(),
      _cellDescription->getPosition())
      || tarch::la::oneGreater(
          _cellDescription->getPosition() + _cellDescription->getSize(),
          _cellDescription->getRestrictionUpperBounds());
}

bool peanoclaw::Patch::containsNaN() const {
  //Check uNew
  dfor(subcellIndex, getSubdivisionFactor()){
  for(int unknown = 0; unknown < getUnknownsPerSubcell(); unknown++) {
    if(_accessor.getValueUNew(subcellIndex, unknown) != _accessor.getValueUNew(subcellIndex, unknown)) {
      return true;
    }
  }
}
//Check uOld
dfor(subcellIndex, getSubdivisionFactor() + 2 * getGhostlayerWidth()) {
  for(int unknown = 0; unknown < getUnknownsPerSubcell(); unknown++) {
    if(_accessor.getValueUOld(subcellIndex - getGhostlayerWidth(), unknown) != _accessor.getValueUOld(subcellIndex - getGhostlayerWidth(), unknown)) {
      return true;
    }
  }
}

return false;
}

bool peanoclaw::Patch::containsNonPositiveNumberInUnknownInUNew(int unknown) const {
  //Check uNew
  dfor(subcellIndex, getSubdivisionFactor()){
  if( !tarch::la::greater(_accessor.getValueUNew(subcellIndex, unknown), 0.0) ) {
    return true;
  }
}

return false;
}

bool peanoclaw::Patch::containsNonPositiveNumberInUnknownInGhostlayer(int unknown) const {
  dfor(subcellIndex, getSubdivisionFactor() + 2*getGhostlayerWidth()) {
    if(tarch::la::oneGreater(tarch::la::Vector<DIMENSIONS, int>(0), subcellIndex)
      || tarch::la::oneGreater(subcellIndex, getSubdivisionFactor())) {
      if(!tarch::la::greater(_accessor.getValueUOld(subcellIndex-getGhostlayerWidth(), unknown), 0.0)) {
        return true;
      }
    }
  }
  return false;
}

bool peanoclaw::Patch::containsNegativeNumberInUnknownInGhostlayer(int unknown) const {
  dfor(subcellIndex, getSubdivisionFactor() + 2*getGhostlayerWidth()) {
    if(tarch::la::oneGreater(tarch::la::Vector<DIMENSIONS, int>(0), subcellIndex)
      || tarch::la::oneGreater(subcellIndex, getSubdivisionFactor())) {
      if(tarch::la::smaller(_accessor.getValueUOld(subcellIndex-getGhostlayerWidth(), unknown), 0.0)) {
        return true;
      }
    }
  }
  return false;
}

void peanoclaw::Patch::increaseAgeByOneGridIteration() {
  _cellDescription->setAgeInGridIterations(
      _cellDescription->getAgeInGridIterations() + 1);
}

int peanoclaw::Patch::getAge() const {
  return _cellDescription->getAgeInGridIterations();
}

void peanoclaw::Patch::resetAge() const {
  return _cellDescription->setAgeInGridIterations(0);
}

void peanoclaw::Patch::resetNeighboringGhostlayerBounds() {
  _cellDescription->setRestrictionLowerBounds(
      std::numeric_limits<double>::max());
  _cellDescription->setRestrictionUpperBounds(
      -std::numeric_limits<double>::max());
}

void peanoclaw::Patch::updateLowerNeighboringGhostlayerBound(int dimension,
    double bound) {
  _cellDescription->setRestrictionLowerBounds(dimension,
      std::min(_cellDescription->getRestrictionLowerBounds(dimension), bound));
}

void peanoclaw::Patch::updateUpperNeighboringGhostlayerBound(int dimension,
    double bound) {
  _cellDescription->setRestrictionUpperBounds(dimension,
      std::max(_cellDescription->getRestrictionUpperBounds(dimension), bound));
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::Patch::getLowerNeighboringGhostlayerBounds() const {
  return _cellDescription->getRestrictionLowerBounds();
}

tarch::la::Vector<DIMENSIONS, double> peanoclaw::Patch::getUpperNeighboringGhostlayerBounds() const {
  return _cellDescription->getRestrictionUpperBounds();
}

#ifdef Parallel
void peanoclaw::Patch::setIsRemote(bool isRemote) {
  _cellDescription->setIsRemote(isRemote);
}

bool peanoclaw::Patch::isRemote() const {
  return _cellDescription->getIsRemote();
}
#endif

std::ostream& operator<<(std::ostream& out, const peanoclaw::Patch& patch) {
  return out << patch.toString();
}
