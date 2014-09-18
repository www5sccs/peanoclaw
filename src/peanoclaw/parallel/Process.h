/*
 * Process.h
 *
 *  Created on: Sep 18, 2014
 *      Author: kristof
 */
#ifndef PEANOCLAW_PARALLEL_PROCESS_H_
#define PEANOCLAW_PARALLEL_PROCESS_H_

//TODO unterweg: Change to platform independent code.
#include <sys/types.h>

namespace peanoclaw {
  namespace parallel {
    class Process;
  }
}

/**
 * Class encapsulating the actual process running the current instance
 * of PeanoClaw. This is used currently for priority control.
 */
class peanoclaw::parallel::Process {

  private:
    id_t _pid;
    int  _priorityPenalty;

  public:
    /**
     * Default constructor.
     */
    Process();

    /**
     * Switches the process to run at a lower priority than normal.
     */
    void setToLowPriority();

    /**
     * Switches the process to run at normal priority.
     */
    void setToNormalPriority();
};


#endif /* PEANOCLAW_PARALLEL_PROCESS_H_ */
