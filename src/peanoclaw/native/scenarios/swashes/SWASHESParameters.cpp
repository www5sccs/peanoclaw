/*
 * SWASHESParameters.cpp
 *
 *  Created on: Oct 17, 2014
 *      Author: kristof
 */
#include "peanoclaw/native/scenarios/swashes/SWASHESParameters.h"

char** peanoclaw::native::scenarios::swashes::SWASHESParameters::getFilledParameterStrings(int numberOfCellsX) {
  _parameterStrings = new std::string[6];
  _parameterStrings[0] = std::string("not used");
  _parameterStrings[1] = std::string("1"); //dim
  _parameterStrings[2] = std::string("2"); //type: MacDonald
  _parameterStrings[3] = std::string("2"); //domain: short channel
  _parameterStrings[4] = std::string("1"); //choice: subcritical flow
//  _parameterStrings[5] = std::string(); //number of cells x

  std::stringstream s;
  s << numberOfCellsX;
  _parameterStrings[5] = s.str();

  _parameterCStrings = new char*[6];
  for(int i = 0; i < 6; i++) {
    _parameterCStrings[i] = (char*)_parameterStrings[i].c_str(); //Hack to fulfill the interface of the Parameters class
  }

  return _parameterCStrings;
}

peanoclaw::native::scenarios::swashes::SWASHESParameters::SWASHESParameters(int numberOfCellsX)
: SWASHES::Parameters(6,  getFilledParameterStrings(numberOfCellsX))
{
}

peanoclaw::native::scenarios::swashes::SWASHESParameters::~SWASHESParameters() {
  delete[] _parameterCStrings;
  delete[] _parameterStrings;
}

