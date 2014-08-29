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

#include "EulerEquations/Cell.hpp"


#include "peanoclaw/records/Data.h"

#include "peano/utils/Dimensions.h"

#include <Eigen/Core>

class peanoclaw::solver::euler3d::Cell : public EulerEquations::Cell<double,3> {

  public:
    typedef peanoclaw::records::Data Data;

  private:
    tarch::la::Vector<NUMBER_OF_EULER_UNKNOWNS,double> _data;
    static const double ADIABATIC_CONSTANT;

  public:
    /**
     * Constructs a Euler 3D cell from a given PeanoClaw cell data entry.
     */
    Cell(const tarch::la::Vector<NUMBER_OF_EULER_UNKNOWNS,double>& data);

    double density() const;

    void density(double const& value);

    EulerEquations::Cell<double,3>::Vector velocity() const;

    void velocity(EulerEquations::Cell<double,3>::Vector const& value);

    void velocity(int const& index, double const& value);

    double energy() const;

    void energy(double const& value);

    double adiabaticConstant() const;

    double marker() const;

    void marker(double const& value);

    tarch::la::Vector<NUMBER_OF_EULER_UNKNOWNS, double> getUnknowns() const;
};


#endif /* PEANOCLAW_SOLVER_EULER3D_CELL_H_ */
