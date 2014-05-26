/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 *
 * Adapted to PeanoClaw by Kristof Unterweger
 */

#include <stdio.h>

#ifndef PEANOCLAW_STATISTICS_MEMORYINFORMATION_H_
#define PEANOCLAW_STATISTICS_MEMORYINFORMATION_H_

namespace peanoclaw {
  namespace statistics {
    /**
     * Returns the current Resident Set Size for the current process in bytes.
     */
    size_t getCurrentRSS();

    /**
     * Returns the peak Resident Set Size for the current process in bytes.
     */
    size_t getPeakRSS();
  }
}


#endif /* PEANOCLAW_STATISTICS_MEMORYINFORMATION_H_ */
