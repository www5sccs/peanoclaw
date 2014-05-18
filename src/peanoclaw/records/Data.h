/*
 * Data.h
 *
 *  Created on: May 18, 2014
 *      Author: kristof
 */

#ifndef PEANOCLAW_RECORDS_DATA_H_
#define PEANOCLAW_RECORDS_DATA_H_

#include "peanoclaw/records/DoubleData.h"
#include "peanoclaw/records/FloatData.h"

namespace peanoclaw {
  namespace records {
    #ifdef PEANOCLAW_SWE
    typedef peanoclaw::records::FloatData Data;
    #else
    typedef peanoclaw::records::DoubleData Data;
    #endif
  }
};

#endif /* PEANOCLAW_RECORDS_DATA_H_ */
