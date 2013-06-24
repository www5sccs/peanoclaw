/*
 * PyClawTestStump.h
 *
 *  Created on: May 24, 2012
 *      Author: unterweg
 */

#ifndef PEANO_APPLICATIONS_PEANOCLAW_TESTS_PYCLAWTESTSTUMP_H_
#define PEANO_APPLICATIONS_PEANOCLAW_TESTS_PYCLAWTESTSTUMP_H_

#include "peano/applications/peanoclaw/PyClaw.h"

namespace peano {
  namespace applications {
    namespace peanoclaw {
      namespace tests {
        class PyClawTestStump;
      }
    }
  }
}

class peano::applications::peanoclaw::tests::PyClawTestStump : public peano::applications::peanoclaw::PyClaw {

  public:
    PyClawTestStump();
};

#endif /* PEANO_APPLICATIONS_PEANOCLAW_TESTS_PYCLAWTESTSTUMP_H_ */
