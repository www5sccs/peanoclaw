#include "peanoclaw/configurations/PeanoClawConfigurationForSpacetreeGrid.h"
#include "peanoclaw/runners/PeanoClawLibraryRunner.h"

#include "tarch/logging/CommandLineLogger.h"

#include <fstream>

tarch::logging::Log peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid::_log("peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid");

void peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid::processEntry(
  const std::string& name,
  const std::string& value
) {
  if(name == "plot" || name == "plotAtOutputTimes") {
    _plotAtOutputTimes = (value == "yes");

    assert(value == "yes" || value == "no");
  } else {
    _isValid = false;
    logError("processEntry(string,string)", "Invalid entry: '" << name << "' '" << value << "'");
  }
}

void peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid::parseLine(const std::string& line) {
  int startName = -1;
  int endName = -1;
  int startValue = -1;
  int endValue = -1;

  bool inFirstToken = true;

  for(size_t i = 0; i < line.size(); i++) {
    char c = line[i];
    const bool isWhiteSpace =
      c == ' '  ||
      c == '\t' ||
      c == '\n';

    if(!isWhiteSpace && c != '=') {
      if(startName == -1) {
        startName = i;
      } else if(!inFirstToken && startValue == -1) {
        startValue = i;
      }
    } else {
      if(startName != -1 && endName == -1) {
        endName = i;
        inFirstToken = false;
      } else if(!inFirstToken && startValue != -1 && endValue == -1) {
        endValue = i;
      }
    }
  }

  std::string name = line.substr(startName, endName-startName);
  std::string value = line.substr(startValue, endValue-startValue);

  //TODO unterweg debug
//  std::cout << "line=" << line << std::endl
//      << "name=[" << startName << "," << endName << "]: " << name << std::endl
//      << "value=[" << startValue << "," << endValue << "]: " << value << std::endl;

  processEntry(name, value);
}

peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid::PeanoClawConfigurationForSpacetreeGrid():
  _isValid(true),
  _plotAtOutputTimes(true),
  _plotSubsteps(false),
  _plotSubstepsAfterOutputTime(-1),
  _additionalLevelsForPredefinedRefinement(1),
  _disableDimensionalSplittingOptimization(false)
  {
  std::string configFileName = "peanoclaw.config";
  std::ifstream configFile(configFileName.c_str());
  if(configFile) {
    if (!configFile.good()) {
      logError( "PeanoClawConfigurationForSpacetreeGrid", "was not able to open input file " << configFileName );
    }

    int linenumber = 1;
    std::string line;
    while (!configFile.eof()) {

      std::getline(configFile, line);
      if(line.length() > 0 && line[0]!='#') {
        parseLine(line);
      }

      linenumber++;
    }
  }
}


peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid::~PeanoClawConfigurationForSpacetreeGrid() {
}


bool peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid::isValid() const { 
  return true;
}

bool peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid::plotAtOutputTimes() const {
  return _plotAtOutputTimes;
}
 
bool peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid::plotSubsteps() const {
  return _plotSubsteps;
}

int peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid::plotSubstepsAfterOutputTime() const {
  return _plotSubstepsAfterOutputTime;
}


bool peanoclaw::configurations::PeanoClawConfigurationForSpacetreeGrid::disableDimensionalSplittingOptimization() const {
  return _disableDimensionalSplittingOptimization;
}
