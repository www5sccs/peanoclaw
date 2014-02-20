/*
 * Extrapolation.h
 *
 *  Created on: Jul 8, 2013
 *      Author: kristof
 */

#ifndef PEANOCLAW_INTERSUBGRIDCOMMUNICATION_EXTRAPOLATION_H_
#define PEANOCLAW_INTERSUBGRIDCOMMUNICATION_EXTRAPOLATION_H_

#include "peanoclaw/Patch.h"

namespace peanoclaw {
  namespace interSubgridCommunication {
    class CornerExtrapolation;
    class EdgeExtrapolation;
    class Extrapolation;
    class ExtrapolationAxis;
  }
}

class peanoclaw::interSubgridCommunication::ExtrapolationAxis {

  private:
    const Patch& _subgrid;
    int          _axis;
    int          _linearSubcellIndex;
    int          _linearIndexSupport0;
    int          _linearIndexSupport1;
    int          _distanceSupport0;
    int          _distanceSupport1;
    double       _maximumLinearError;

  public:
    /**
     * @param subcellIndex The index of the subcell onto which the values
     *                     should be extrapolated.
     * @param subgrid The subgrid on which the extrapolation is performed.
     * @param axis The dimension along which this extrapolation axis is
     *             oriented.
     * @param direction The direction along the axis in that the extrapolation
     *                  is performed. I.e. this points from the source subcells
     *                  to the extrapolated subcell.
     */
    ExtrapolationAxis(
      const tarch::la::Vector<DIMENSIONS,int>& subcellIndex,
      const peanoclaw::Patch&                  subgrid,
      int                                      axis,
      int                                      direction
    );

    /**
     * Returns the extrapolated value for the subcell specified
     * in the constructor.
     */
    double getExtrapolatedValue(int unknown);

    /**
     * Returns the maximum linear error that may occur for all extrapolations done
     * so far.
     * TODO unterweg dissertation
     * Der maximale lineare Fehler schätzt ab, wie weit sich ein Wert durch die
     * Extrapolation verändern kann. Dazu wird der Gradient und die Extrapolationsdistanz
     * berücksichtigt. Der Fehler ist relativ bezogen auf das Minimum der beiden
     * Stützpunkte.
     */
    double getMaximumLinearError() const;
};

/**
 * Functor for traversing the corners in 3D of a subgrid and interpolating the ghostlayer values.
 */
class peanoclaw::interSubgridCommunication::CornerExtrapolation {
  private:
    double _maximumLinearError;

  public:
    void operator()(
      peanoclaw::Patch& subgrid,
      const peanoclaw::Area& area,
      const tarch::la::Vector<DIMENSIONS,int> cornerIndex
    );

    double getMaximumLinearError() const;
};

/**
 * Functor for traversing the edges (in 3D) or corners (in 2D) of a subgrid and
 * interpolating the ghostlayer values.
 */
class peanoclaw::interSubgridCommunication::EdgeExtrapolation {
  private:
    double _maximumLinearError;

  public:

    EdgeExtrapolation();

    void operator()(
      peanoclaw::Patch& subgrid,
      const peanoclaw::Area& area,
      const tarch::la::Vector<DIMENSIONS,int>& direction
    );

    double getMaximumLinearError() const;
};

/**
 * This class contains functionality to extrapolate parts of a
 * ghostlayer to other parts. This is used to fill edges or
 * corners of a ghostlayer to avoid dependency of some adjacent
 * subgrids.
 */
class peanoclaw::interSubgridCommunication::Extrapolation {

  private:
    peanoclaw::Patch& _subgrid;

  public:
    Extrapolation(Patch& subgrid);

    /**
     * Extrapolates the values from the ghostlayer faces to the edges.
     */
    double extrapolateEdges();

    /**
     * Extrapolates the values from the ghostlayer faces/edges to the corners.
     * In 2D this operation does nothing.
     */
    double extrapolateCorners();
};


#endif /* PEANOCLAW_INTERSUBGRIDCOMMUNICATION_EXTRAPOLATION_H_ */
