/*
 * Process.cpp
 *
 *  Created on: Sep 18, 2014
 *      Author: kristof
 */
#include "peanoclaw/parallel/Process.h"

//TODO unterweg: Change to platform independent code.
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>

peanoclaw::parallel::Process::Process()
  : _pid(getpid()),
    _priorityPenalty(5)
{
}

void peanoclaw::parallel::Process::setToLowPriority() {
  setpriority(PRIO_PROCESS, _pid, _priorityPenalty);
}

void peanoclaw::parallel::Process::setToNormalPriority() {
  setpriority(PRIO_PROCESS, _pid, 0);
}




