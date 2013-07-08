/*
 * PyClawTestStump.h
 *
 *  Created on: May 24, 2012
 *      Author: unterweg
 */

#ifndef PEANO_APPLICATIONS_PEANOCLAW_TESTS_PYCLAWTESTSTUMP_H_
#define PEANO_APPLICATIONS_PEANOCLAW_TESTS_PYCLAWTESTSTUMP_H_

#include "peanoclaw/pyclaw/PyClaw.h"

namespace peanoclaw {
  namespace tests {
    class PyClawTestStump;
  }
}

class peanoclaw::tests::PyClawTestStump : public peanoclaw::pyclaw::PyClaw {

  public:
    PyClawTestStump();
};

#endif /* PEANO_APPLICATIONS_PEANOCLAW_TESTS_PYCLAWTESTSTUMP_H_ */
