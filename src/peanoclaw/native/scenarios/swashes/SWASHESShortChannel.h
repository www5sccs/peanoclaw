/*
 * SWASHESShortChannel.h
 *
 *  Created on: Oct 16, 2014
 *      Author: unterweg
 */

#ifndef PEANOCLAW_NATIVE_SCENARIOS_SWASHES_SWASHESSHORTCHANNEL_H_
#define PEANOCLAW_NATIVE_SCENARIOS_SWASHES_SWASHESSHORTCHANNEL_H_

namespace peanoclaw {
  namespace native {
    namespace scenarios {
      namespace swashes {
        class SWASHESShortchannel;
      }
    }
  }
}

class peanoclaw::native::scenarios::swashes::SWASHESShortchannel
  #ifdef PEANOCLAW_SWASHES
  : public MacDonaldB1
  #endif
{
  private:
    const char** PARAMETER_STRINGS;

  public:
    SWASHESShortchannel();
};


#endif /* PEANOCLAW_NATIVE_SCENARIOS_SWASHES_SWASHESSHORTCHANNEL_H_ */
