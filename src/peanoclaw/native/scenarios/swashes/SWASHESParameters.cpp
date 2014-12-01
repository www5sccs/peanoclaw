/*
 * SWASHESParameters.cpp
 *
 *  Created on: Oct 17, 2014
 *      Author: kristof
 */
#include "peanoclaw/native/scenarios/swashes/SWASHESParameters.h"

char** peanoclaw::native::scenarios::swashes::SWASHESParameters::getFilledParameterStrings(int numberOfCellsX, int choice) {
  _parameterStrings = new std::string[6];
  _parameterStrings[0] = std::string("not used");
  _parameterStrings[1] = std::string("1"); //dim
  _parameterStrings[2] = std::string("2"); //type: MacDonald
  _parameterStrings[3] = std::string("2"); //domain: short channel

  std::stringstream choiceStream;
  choiceStream << choice;
  _parameterStrings[4] = choiceStream.str(); //choice: 1: subcritical flow, 2: supercritical

  std::stringstream cellStream;
  cellStream << numberOfCellsX;
  _parameterStrings[5] = cellStream.str();

  _parameterCStrings = new char*[6];
  for(int i = 0; i < 6; i++) {
    _parameterCStrings[i] = (char*)_parameterStrings[i].c_str(); //Hack to fulfill the interface of the Parameters class
  }

  return _parameterCStrings;
}

peanoclaw::native::scenarios::swashes::SWASHESParameters::SWASHESParameters(int numberOfCellsX, int choice)
: SWASHES::Parameters(6,  getFilledParameterStrings(numberOfCellsX, choice))
{
}

peanoclaw::native::scenarios::swashes::SWASHESParameters::~SWASHESParameters() {
  delete[] _parameterCStrings;
  delete[] _parameterStrings;
}

