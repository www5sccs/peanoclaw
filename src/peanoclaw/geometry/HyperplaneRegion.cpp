/*
 * HyperplaneRegion.cpp
 *
 *  Created on: Dec 5, 2014
 *      Author: kristof
 */

#include "peanoclaw/geometry/HyperplaneRegion.h"

peanoclaw::geometry::HyperplaneRegion::HyperplaneRegion(
  const tarch::la::Vector<DIMENSIONS_MINUS_ONE,int>& offset,
  const tarch::la::Vector<DIMENSIONS_MINUS_ONE,int>& size
) : _offset(offset),
    _size(size)
{
}



