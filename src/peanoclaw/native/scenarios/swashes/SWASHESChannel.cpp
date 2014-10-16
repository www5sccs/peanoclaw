/*
 * SWASHESShortChannel.cpp
 *
 *  Created on: Oct 16, 2014
 *      Author: unterweg
 */
#include "peanoclaw/native/scenarios/swashes/SWASHESChannel.h"

#include <sstream>

//char** peanoclaw::native::scenarios::swashes::SWASHESChannel::PARAMETER_STRINGS
//  = (char *[]){
//    "not used",
//    "1", //dim
//    "2", //type: MacDonald
//    "2", //domain: short channel
//    "1", //choice: subcritical flow
//    "1000" //number of cells x
//  };

int peanoclaw::native::scenarios::swashes::SWASHESChannel::NUMBER_OF_CELLS_X = 1000;


char** peanoclaw::native::scenarios::swashes::SWASHESChannel::getFilledParameterStrings() {
  _parameterStrings = new std::string[6];
  _parameterStrings[0] = new std::string("not used");
  _parameterStrings[1] = new std::string("1"); //dim
  _parameterStrings[2] = new std::string("2"); //type: MacDonald
  _parameterStrings[3] = new std::string("2"); //domain: short channel
  _parameterStrings[4] = new std::string("1"); //choice: subcritical flow
  _parameterStrings[5] = new std::string(); //number of cells x

  std::stringstream s;
  _parameterStrings->append(s.str());

  _parameterCStrings = new char*[6];
  for(int i = 0; i < 6; i++) {
    _parameterCStrings[i] = _parameterStrings[i].c_str();
  }

  return _parameterCStrings;
}

int peanoclaw::native::scenarios::swashes::SWASHESShortChannel::getIndex(double x) const {
  #ifdef PEANOCLAW_SWASHES
  return (x / L) * NX_EX;
  #else
  return 0;
  #endif
}

double peanoclaw::native::scenarios::swashes::SWASHESShortChannel::shortBedWidth(double x) const {
  return 10.0 - 5.0 * exp(-10 * pow(x/200 - 0.5, 2.0));
}

double peanoclaw::native::scenarios::swashes::SWASHESShortChannel::longBedWidth(double x) const {
  return 10.0 - 5.0 * exp(-50 * pow(x/400 - 1.0/3.0, 2.0)) - 5.0 * exp(-50 * pow(x/400 - 2.0/3.0, 2.0));
}

peanoclaw::native::scenarios::swashes::SWASHESShortChannel::SWASHESShortChannel()
#ifdef PEANOCLAW_SWASHES
  : MacDonaldB1(Parameters(6, getFilledParameterStrings()))
#endif
{
}

peanoclaw::native::scenarios::swashes::SWASHESShortChannel::~SWASHESShortChannel() {
  for(int i = 0; i < 6; i++) {
    delete _parameterStrings[i];
  }
  delete[] _parameterStrings;
  delete[] _parameterCStrings;
}

double peanoclaw::native::scenarios::swashes::SWASHESShortChannel::getTopography(double x) const {
  #ifdef PEANOCLAW_SWASHES
  return zex[getIndex(x)];
  #else
  return 0.0;
  #endif
}

double peanoclaw::native::scenarios::swashes::SWASHESShortChannel::getInitialWaterHeight(double x) const {
  #ifdef PEANOCLAW_SWASHES
  return ;
  #else
  return 0.0;
  #endif
}

double peanoclaw::native::scenarios::swashes::SWASHESShortChannel::getExpectedWaterHeight(double x) const {
  #ifdef PEANOCLAW_SWASHES
  return hex[getIndex(x)];
  #else
  return 0.0;
  #endif
}

double peanoclaw::native::scenarios::swashes::SWASHESShortChannel::getBedWidth(double x) const {
  return getBedWidth(x);
}
