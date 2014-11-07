/*
 * SWASHESShortChannel.h
 *
 *  Created on: Oct 16, 2014
 *      Author: unterweg
 */

#ifndef PEANOCLAW_NATIVE_SCENARIOS_SWASHES_SWASHESCHANNEL_H_
#define PEANOCLAW_NATIVE_SCENARIOS_SWASHES_SWASHESCHANNEL_H_

#ifdef PEANOCLAW_SWASHES
#undef PI
#include "SWASHES/macdonaldb1.hpp"
#include "SWASHES/macdonaldb2.hpp"
#undef PI
#endif

#include "peano/utils/Dimensions.h"
#include "tarch/la/Vector.h"

#include <string>
#include <vector>

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
    const int NUMBER_OF_CELLS_X = 1000;
    const int HEIGHT_OF_BED_WALLS = 10;

    /**
     * Setter for the subclasses to set the domain width.
     */
    void setDomainWidth(double domainWidth);

    /**
     * Returns parameter strings accordingly to the settings.
     */
    char** getFilledParameterStrings();

  private:
    double       _domainWidth;

  public:
    SWASHESChannel();
    virtual ~SWASHESChannel();
    virtual double getTopography(double x) const = 0;
    double getTopography(tarch::la::Vector<DIMENSIONS,double> position) const;
    virtual double getInitialWaterHeight(double x) const = 0;
    double getInitialWaterHeight(tarch::la::Vector<DIMENSIONS,double> position) const;
    virtual double getExpectedWaterHeight(double x) const = 0;
    double getExpectedWaterHeight(tarch::la::Vector<DIMENSIONS,double> position) const;
    virtual double getBedWidth(double x) const = 0;
    virtual void initialize() = 0;
    virtual double getOutflowHeight() const = 0;
};

class peanoclaw::native::scenarios::swashes::SWASHESShortChannel
  :
  #ifdef PEANOCLAW_SWASHES
  public SWASHES::MacDonaldB1,
  #endif
  public SWASHESChannel
{
  private:
    /**
     * Computes the index in the arrays of class Solution according to the real world position x.
     */
    int getIndex(double x) const;

  public:
    SWASHESShortChannel(
      #ifdef PEANOCLAW_SWASHES
      SWASHES::Parameters& parameters
      #endif
    );

    virtual ~SWASHESShortChannel();

    double getTopography(double x) const;

    double getInitialWaterHeight(double x) const;

    double getExpectedWaterHeight(double x) const;

    double getBedWidth(double x) const;

    void initialize();

    double getOutflowHeight() const;
};

class peanoclaw::native::scenarios::swashes::SWASHESLongChannel
  :
  #ifdef PEANOCLAW_SWASHES
  public SWASHES::MacDonaldB2,
  #endif
  public SWASHESChannel
{
  double getBedWidth(double x) const;

  void initialize();

  double getOutflowHeight() const;
};

#endif /* PEANOCLAW_NATIVE_SCENARIOS_SWASHES_SWASHESCHANNEL_H_ */
