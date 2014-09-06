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

Uni::EulerEquations::Cell<double,3>::Vector peanoclaw::solver::euler3d::Cell::velocity() const {
  //return Vector(&_data[1]);
  Vector v(&_data[1]);
  return v  * (1.0 / density());
}

void peanoclaw::solver::euler3d::Cell::velocity(Uni::EulerEquations::Cell<double,3>::Vector const& value) {
  _data[1] = value(0) * density();
  _data[2] = value(1) * density();
  _data[3] = value(2) * density();
}

void peanoclaw::solver::euler3d::Cell::velocity(int const& index, double const& value) {
  _data[1+index] = value * density();
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

double peanoclaw::solver::euler3d::Cell::temperature() const {
  return Base::computeTemperatureFromDensityVelocityEnergy(
    density(),
    velocity(),
    energy(),
    adiabaticConstant(),
    Base::gasConstant());
}

void peanoclaw::solver::euler3d::Cell::temperature(double const& temperature) {
  energy(Base::computeEnergyFromDensityVelocityTemperature(
         density(),
         velocity(),
         temperature,
         adiabaticConstant(),
         Base::gasConstant())
  );
}

double peanoclaw::solver::euler3d::Cell::pressure() const {
  return Base::computePressureFromDensityVelocityEnergy(
    density(),
    velocity(),
    energy(),
    adiabaticConstant());
}

void peanoclaw::solver::euler3d::Cell::pressure(double const& pressure) {
  energy(Base::computeEnergyFromDensityVelocityPressure(
     density(),
     velocity(),
     pressure,
     adiabaticConstant())
  );
}

double peanoclaw::solver::euler3d::Cell::enthalpy() const {
  return Base::computeEnthalpyFromDensityVelocityEnergy(
    density(),
    velocity(),
    energy(),
    adiabaticConstant()
  );
}

void peanoclaw::solver::euler3d::Cell::enthalpy(double const& enthalpy) {
  energy(Base::computeEnergyFromDensityVelocityEnthalpy(
     density(),
     velocity(),
     enthalpy,
     adiabaticConstant())
  );
}

double peanoclaw::solver::euler3d::Cell::soundSpeed() const {
  return Base::computeSoundSpeedFromDensityVelocityEnergy(
    density(),
    velocity(),
    energy(),
    adiabaticConstant());
}

//void peanoclaw::solver::euler3d::Cell::soundSpeed(double const& soundSpeed) {
//  //TODO unterweg debug
//  throw "";
//}


tarch::la::Vector<NUMBER_OF_EULER_UNKNOWNS, double> peanoclaw::solver::euler3d::Cell::getUnknowns() const {
  tarch::la::Vector<NUMBER_OF_EULER_UNKNOWNS, double> unknowns;
  unknowns[0] = density();
  unknowns[1] = _data[1];
  unknowns[2] = _data[2];
  unknowns[3] = _data[3];
  unknowns[4] = energy();
  unknowns[5] = marker();

  return unknowns;
}

