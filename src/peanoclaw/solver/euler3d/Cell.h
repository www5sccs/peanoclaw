/*
 * Cell.h
 *
 *  Created on: Jul 24, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_SOLVER_EULER3D_CELL_H_
#define PEANOCLAW_SOLVER_EULER3D_CELL_H_

namespace peanoclaw {
  namespace solver {
    namespace euler3d {
      class Cell;
    }
  }
}

#define NUMBER_OF_EULER_UNKNOWNS 6

#include "Uni/EulerEquations/Cell"


#include "peanoclaw/records/Data.h"

#include "peano/utils/Dimensions.h"

#include <Eigen/Core>

class peanoclaw::solver::euler3d::Cell : public Uni::EulerEquations::Cell<double,3> {

  public:
    typedef peanoclaw::records::Data Data;
    #ifdef Asserts
    tarch::la::Vector<DIMENSIONS,int> _index;
    #endif

  private:
    //tarch::la::Vector<NUMBER_OF_EULER_UNKNOWNS,double> _data;
    double* _data;
    static const double ADIABATIC_CONSTANT;
    typedef Uni::EulerEquations::Cell<double, 3> Base;

  public:
    /**
     * Constructs an Euler 3D cell from a given PeanoClaw cell data entry.
     */
//    Cell(
//      const tarch::la::Vector<NUMBER_OF_EULER_UNKNOWNS,double>& data,
//      const tarch::la::Vector<DIMENSIONS,int>& index
//    );

    Cell(
      double* data,
      const tarch::la::Vector<DIMENSIONS,int>& index
    );

    double density() const;

    void density(double const& value);

    Uni::EulerEquations::Cell<double,3>::Vector velocity() const;

    void velocity(Uni::EulerEquations::Cell<double,3>::Vector const& value);

    void velocity(int const& index, double const& value);

    double energy() const;

    void energy(double const& value);

    double adiabaticConstant() const;

    double marker() const;

    void marker(double const& value);

    double temperature() const;

    void temperature(double const& temperature);

    double pressure() const;

    void pressure(double const& pressure);

    double enthalpy() const;

    void enthalpy(double const& enthalpy);

    virtual double soundSpeed() const;

//    void soundSpeed(double const& soundSpeed);

    tarch::la::Vector<NUMBER_OF_EULER_UNKNOWNS, double> getUnknowns() const;
};


#endif /* PEANOCLAW_SOLVER_EULER3D_CELL_H_ */
