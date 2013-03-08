/*
 * PyClawState.h
 *
 *  Created on: Mar 7, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_PYCLAW_PYCLAWSTATE_H_
#define PEANOCLAW_PYCLAW_PYCLAWSTATE_H_

namespace peanoclaw {

  class Patch;

  namespace pyclaw {
    class PyClawState;
  }
}

#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif

/**
 * A state object encapsulates all NumPy arrays for the PyClaw state.
 *
 * This class is used for a RAII like allocation of memory. I.e. during
 * the constructor the memory for the arrays is allocated dynamically
 * and it is deleted on destruction of a PyClawState object.
 */
class peanoclaw::pyclaw::PyClawState {
  public:
    PyObject* _q;
    PyObject* _qbc;
    PyObject* _aux;

    /**
     * Creates the Numpy arrays and allocates memory when needed.
     */
    PyClawState(const Patch& patch);

    /**
     * Deletes allocated memory.
     */
    ~PyClawState();
};

#endif /* PEANOCLAW_PYCLAW_PYCLAWSTATE_H_ */
