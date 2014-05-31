/*
 * SWECommandLineParser.cpp
 *
 *  Created on: May 28, 2014
 *      Author: kristof
 */
#include "peanoclaw/native/SWECommandLineParser.h"

#include "tarch/Assertions.h"

#include <sstream>
#include <string>
#include <stdexcept>

tarch::logging::Log peanoclaw::native::SWECommandLineParser::_log("peanoclaw::native::SWECommandLineParser");

peanoclaw::native::SWECommandLineParser::SWECommandLineParser(
  int argc,
  char** argv
) : _finestSubgridTopology(0),
    _coarsestSubgridTopology(0),
    _subdivisionFactor(0),
    _endTime(0),
    _globalTimestepSize(0),
    _usePeanoClaw(false) {
  const int requiredNumberOfArguments = 6;
  const int numberOfOptionalArguments = 1;
  if(argc != requiredNumberOfArguments && argc != requiredNumberOfArguments + numberOfOptionalArguments) {
    std::stringstream s;
    s << "There have to be " << requiredNumberOfArguments << " or " << (requiredNumberOfArguments + numberOfOptionalArguments)
      << " arguments instead of " << argc;
    s << "\nParameters: finestSubgridTopology coarsesSubgridTopology subdivisionFactor endTime globalTimestepSize [--usePeano]";
    throw std::invalid_argument(s.str());
  }

  //Finest subgrid topology
  {
    std::istringstream s(argv[1]);
    int subgridsPerDimension;
    s >> subgridsPerDimension;
    _finestSubgridTopology = tarch::la::Vector<DIMENSIONS,int>(subgridsPerDimension);
  }

  //Coarsest subgrid topology
  {
    std::istringstream s(argv[2]);
    int subgridsPerDimension;
    s >> subgridsPerDimension;
    _coarsestSubgridTopology = tarch::la::Vector<DIMENSIONS,int>(subgridsPerDimension);
  }

  //Subdivision factor
  {
    std::istringstream s(argv[3]);
    int subdivisionPerDimension;
    s >> subdivisionPerDimension;
    _subdivisionFactor = tarch::la::Vector<DIMENSIONS,int>(subdivisionPerDimension);
  }

  if(!tarch::la::allGreaterEquals(_coarsestSubgridTopology, _finestSubgridTopology)) {
    logError("SWECommandLineParser", "Finest subgrid topology has to be finer than the coarsest one.");
  }

  //End time
  {
    std::istringstream s(argv[4]);
    s >> _endTime;
  }

  //Global timestep size
  {
    std::istringstream s(argv[5]);
    s >> _globalTimestepSize;
  }

  if(argc > requiredNumberOfArguments) {
    assertionEquals(std::string(argv[6]), "--usePeano");
    _usePeanoClaw = true;
  }

  logInfo("SWECommandLineParser", "Parameter: finest subgrid topology=" << _finestSubgridTopology
          << ", coarsest subgrid topology=" << _coarsestSubgridTopology
          << ", subdivision factor=" << _subdivisionFactor
          << ", end time=" << _endTime
          << ", global timestep size=" << _globalTimestepSize);
}

tarch::la::Vector<DIMENSIONS,int> peanoclaw::native::SWECommandLineParser::getFinestSubgridTopology() const {
  return _finestSubgridTopology;
}

tarch::la::Vector<DIMENSIONS,int> peanoclaw::native::SWECommandLineParser::getCoarsestSubgridTopology() const {
  return _coarsestSubgridTopology;
}

tarch::la::Vector<DIMENSIONS,int> peanoclaw::native::SWECommandLineParser::getSubdivisionFactor() const {
  return _subdivisionFactor;
}

double peanoclaw::native::SWECommandLineParser::getEndTime() const {
  return _endTime;
}

double peanoclaw::native::SWECommandLineParser::getGlobalTimestepSize() const {
  return _globalTimestepSize;
}

bool peanoclaw::native::SWECommandLineParser::runSimulationWithPeanoClaw() const {
  return _usePeanoClaw;
}
