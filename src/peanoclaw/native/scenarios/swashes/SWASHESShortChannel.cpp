/*
 * SWASHESShortChannel.cpp
 *
 *  Created on: Oct 16, 2014
 *      Author: unterweg
 */
#include "peanoclaw/native/scenarios/swashes/SWASHESShortChannel.h"

const char** peanoclaw::native::scenarios::swashes::SWASHESShortchannel::PARAMETER_STRINGS
  = (char *[]){
    "not used",
    "1", //dim
    "2", //type: MacDonald
    "2", //domain: short channel
    "1", //choice: subcritical flow
    "1000" //number of cells x
  };



peanoclaw::native::scenarios::swashes::SWASHESShortchannel::SWASHESShortchannel()
#ifdef PEANOCLAW_SWASHES
  : MacDonaldB1(Parameters(1, PARAMETER_STRINGS))
#endif
{
}
