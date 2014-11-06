/*
 * FullSWOF2DBoundaryCondition.h
 *
 *  Created on: Nov 4, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_NATIVE_FULLSWOF2DBOUNDARYCONDITION_H_
#define PEANOCLAW_NATIVE_FULLSWOF2DBOUNDARYCONDITION_H_

namespace peanoclaw {
  namespace native {
    namespace scenarios {
      class FullSWOF2DBoundaryCondition;
    }
  }
}

class peanoclaw::native::scenarios::FullSWOF2DBoundaryCondition {
  private:
    int _type;
    double _impliedDischarge;
    double _impliedHeight;

  public:
    FullSWOF2DBoundaryCondition(int type, double impliedDischarge, double impliedHeight);

    int getType() const;

    double getImpliedDischarge() const;

    double getImpliedHeight() const;
};



#endif /* PEANOCLAW_NATIVE_FULLSWOF2DBOUNDARYCONDITION_H_ */
