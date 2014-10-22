/*
 * SWASHESShortChannel.cpp
 *
 *  Created on: Oct 16, 2014
 *      Author: unterweg
 */
#include "peanoclaw/native/scenarios/swashes/SWASHESChannel.h"

#include <sstream>

void peanoclaw::native::scenarios::swashes::SWASHESChannel::setDomainWidth(double domainWidth) {
  _domainWidth = domainWidth;
}

peanoclaw::native::scenarios::swashes::SWASHESChannel::SWASHESChannel()
  : _domainWidth(-100)
{
}

peanoclaw::native::scenarios::swashes::SWASHESChannel::~SWASHESChannel() {
}

double peanoclaw::native::scenarios::swashes::SWASHESChannel::getTopography(tarch::la::Vector<DIMENSIONS,double> position) const {
  #ifdef PEANOCLAW_SWASHES
  double distanceToCenter = position[1] - _domainWidth / 2.0;
  double bedWidth = getBedWidth(position[0]);
  double topography = getTopography(position[0]);
  if(distanceToCenter < bedWidth / 2.0) {
    topography += HEIGHT_OF_BED_WALLS;
  }
  return topography;
  #else
  return 0.0;
  #endif
}

double peanoclaw::native::scenarios::swashes::SWASHESChannel::getInitialWaterHeight(tarch::la::Vector<DIMENSIONS,double> position) const {
  #ifdef PEANOCLAW_SWASHES
  return -1.0;
  #else
  return 0.0;
  #endif
}

double peanoclaw::native::scenarios::swashes::SWASHESChannel::getExpectedWaterHeight(tarch::la::Vector<DIMENSIONS,double> position) const {
  #ifdef PEANOCLAW_SWASHES
  return -1.0;
  #else
  return 0.0;
  #endif
}


//SHORT CHANNEL
int peanoclaw::native::scenarios::swashes::SWASHESShortChannel::getIndex(double x) const {
  #ifdef PEANOCLAW_SWASHES
  return (x / L) * NX_EX;
  #else
  return 0;
  #endif
}

double peanoclaw::native::scenarios::swashes::SWASHESShortChannel::getBedWidth(double x) const {
  return 10.0 - 5.0 * exp(-10 * pow(x/200 - 0.5, 2.0));
}

peanoclaw::native::scenarios::swashes::SWASHESShortChannel::SWASHESShortChannel(SWASHES::Parameters& parameters)
#ifdef PEANOCLAW_SWASHES
  : SWASHES::MacDonaldB1(parameters)
#endif
{
}

peanoclaw::native::scenarios::swashes::SWASHESShortChannel::~SWASHESShortChannel() {
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
  return -1.0;
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

void peanoclaw::native::scenarios::swashes::SWASHESShortChannel::initialize() {
  compute();
}

//LONG CHANNEL
double peanoclaw::native::scenarios::swashes::SWASHESLongChannel::getBedWidth(double x) const {
  return 10.0 - 5.0 * exp(-50 * pow(x/400 - 1.0/3.0, 2.0)) - 5.0 * exp(-50 * pow(x/400 - 2.0/3.0, 2.0));
}

void peanoclaw::native::scenarios::swashes::SWASHESLongChannel::initialize() {
  compute();
}
