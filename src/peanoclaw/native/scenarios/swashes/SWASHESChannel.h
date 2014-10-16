/*
 * SWASHESShortChannel.h
 *
 *  Created on: Oct 16, 2014
 *      Author: unterweg
 */

#ifndef PEANOCLAW_NATIVE_SCENARIOS_SWASHES_SWASHESCHANNEL_H_
#define PEANOCLAW_NATIVE_SCENARIOS_SWASHES_SWASHESCHANNEL_H_

namespace peanoclaw {
  namespace native {
    namespace scenarios {
      namespace swashes {
        class SWASHESChannel;
        class SWASHESShortChannel;
        class SWASHESLongChannel;
      }
    }
  }
}

class peanoclaw::native::scenarios::swashes::SWASHESChannel {
  protected:
    const int NUMBER_OF_CELLS_X;
  private:
    char**       _parameterCStrings;
    std::string* _parameterStrings;
    char** getFilledParameterStrings();
  public:
    SWASHESChannel(){}
    virtual ~SWASHESChannel(){}
    virtual double getTopography(double x) const = 0;
    virtual double getInitialWaterHeight(double x) const = 0;
    virtual double getExpectedWaterHeight(double x) const = 0;
    virtual double getBedWidth(double x) const = 0;
};

class peanoclaw::native::scenarios::swashes::SWASHESShortChannel
  :
  #ifdef PEANOCLAW_SWASHES
  public MacDonaldB1,
  #endif
  public SWASHESChannel
{
  private:
    /**
     * Computes the index in the arrays of class Solution according to the real world position x.
     */
    int getIndex(double x) const;

  public:
    SWASHESShortChannel();

    virtual ~SWASHESShortChannel();

    double getTopography(double x) const;

    double getInitialWaterHeight(double x) const;

    double getExpectedWaterHeight(double x) const;

    double getBedWidth(double x) const;
};

class peanoclaw::native::scenarios::swashes::SWASHESLongChannel
  :
  #ifdef PEANOCLAW_SWASHES
  public MacDonaldB2,
  #endif
  public SWASHESChannel
{
  double getBedWidth(double x) const;
};

#endif /* PEANOCLAW_NATIVE_SCENARIOS_SWASHES_SWASHESCHANNEL_H_ */
