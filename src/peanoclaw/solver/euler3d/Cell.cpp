/*
 * Cell.cpp
 *
 *  Created on: Jul 24, 2014
 *      Author: kristof
 */
#include "peanoclaw/solver/euler3d/Cell.h"

const double peanoclaw::solver::euler3d::Cell::ADIABATIC_CONSTANT = 1.4;

peanoclaw::solver::euler3d::Cell::Cell(const tarch::la::Vector<NUMBER_OF_EULER_UNKNOWNS,double>& data) : _data(data) {

}

double peanoclaw::solver::euler3d::Cell::density() const {
  return _data[0];
}

void peanoclaw::solver::euler3d::Cell::density(double const& value) {
  _data[0] = value;
}

EulerEquations::Cell<double,3>::Vector peanoclaw::solver::euler3d::Cell::velocity() const {
  return Vector(&_data[1]);
}

void peanoclaw::solver::euler3d::Cell::velocity(EulerEquations::Cell<double,3>::Vector const& value) {
  _data[1] = value(0);
  _data[2] = value(1);
  _data[3] = value(2);
}

void peanoclaw::solver::euler3d::Cell::velocity(int const& index, double const& value) {
  _data[1+index] = value;
}

double peanoclaw::solver::euler3d::Cell::energy() const {
  return _data[4];
}

void peanoclaw::solver::euler3d::Cell::energy(double const& value) {
  _data[4] = value;
}

double peanoclaw::solver::euler3d::Cell::adiabaticConstant() const {
  return ADIABATIC_CONSTANT;
}

double peanoclaw::solver::euler3d::Cell::marker() const {
  return _data[5];
}

void peanoclaw::solver::euler3d::Cell::marker(double const& value) {
  _data[5] = value;
}

tarch::la::Vector<NUMBER_OF_EULER_UNKNOWNS, double> peanoclaw::solver::euler3d::Cell::getUnknowns() const {
  tarch::la::Vector<NUMBER_OF_EULER_UNKNOWNS, double> unknowns;
  unknowns[0] = density();
  unknowns[1] = velocity()(0);
  unknowns[2] = velocity()(1);
  unknowns[3] = velocity()(2);
  unknowns[4] = energy();
  unknowns[5] = marker();

  return unknowns;
}

